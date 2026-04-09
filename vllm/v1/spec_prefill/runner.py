# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Speculative prefill runner for vLLM v1 — native paged-KV approach.

This module loads a lightweight draft model through vLLM's own
``get_model`` (same loader used by ``DraftModelProposer``), so it
benefits from tensor-parallelism and **paged KV-cache** managed by the
vLLM memory allocator.  No HuggingFace dynamic cache is involved.

Architecture (mirrors the original ``speculative_prefill`` project):
1.  Draft model processes the full prompt via vLLM paged attention.
2.  Query vectors captured during look-ahead decode steps via
    ``register_forward_pre_hook`` on each ``Attention`` module.
3.  Keys read back from the paged KV-cache tensors.
4.  Importance scores computed, tokens selected.
5.  Compressed tokens + original position ids returned to the main
    model runner.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from vllm.config import SpecPrefillConfig, VllmConfig, get_layers_from_vllm_config
from vllm.forward_context import set_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.attention import Attention
from vllm.model_executor.model_loader import get_model
from vllm.v1.attention.backend import (
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
)
from vllm.v1.spec_prefill.compression import (
    compute_token_importance,
    select_kept_indices,
)
from vllm.v1.spec_prefill.metadata import (
    SpecPrefillBatchMetadata,
    SpecPrefillMetadata,
)

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

logger = init_logger(__name__)


class SpecPrefillRunner:
    """Orchestrates the speculative-prefill pipeline using vLLM-native
    model loading and paged KV-cache.

    Lifecycle
    ---------
    1. ``load_draft_model(vllm_config, runner)`` — called inside
       ``GPUModelRunner.load_model`` (within ``DeviceMemoryProfiler``).
    2. After ``initialize_attn_backend`` the runner must call
       ``init_attn_metadata_builder()`` so we can build proper metadata.
    3. ``run(req_ids, prompts)`` — called for each scheduling step that
       contains new prefill requests.
    """

    def __init__(
        self,
        config: SpecPrefillConfig,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.config = config
        self.device = device
        self.dtype = dtype

        self.model: nn.Module | None = None
        self.vllm_config: VllmConfig | None = None
        self.runner: GPUModelRunner | None = None

        # Attention layer book-keeping (populated after load)
        self.draft_attn_layer_names: list[str] = []
        self._num_layers: int = 0
        self._num_heads: int = 0
        self._num_kv_heads: int = 0
        self._head_dim: int = 0
        self._block_size: int = 0

        self._attn_metadata_builder: AttentionMetadataBuilder | None = None

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------
    def load_draft_model(self, target_vllm_config: VllmConfig) -> None:
        """Load draft model via vLLM ``get_model`` with tensor-parallelism."""
        from vllm.compilation.backends import set_model_tag
        from vllm.config import ModelConfig

        target_mc = target_vllm_config.model_config

        draft_model_config = ModelConfig(
            model=self.config.spec_model,
            tokenizer=target_mc.tokenizer,
            tokenizer_mode=target_mc.tokenizer_mode,
            trust_remote_code=target_mc.trust_remote_code,
            dtype=target_mc.dtype,
            seed=target_mc.seed,
            max_model_len=target_mc.max_model_len,
        )

        temp_config: VllmConfig = replace(
            target_vllm_config,
            model_config=draft_model_config,
            quant_config=None,
        )
        self.vllm_config = target_vllm_config

        attn_names_before = set(
            get_layers_from_vllm_config(
                target_vllm_config, Attention,
            ).keys()
        )

        logger.info(
            "Loading spec-prefill draft model: %s", self.config.spec_model
        )
        with set_model_tag("spec_prefill"):
            self.model = get_model(
                vllm_config=temp_config,
                prefix="spec_prefill",
            )

        all_attn_names = set(
            get_layers_from_vllm_config(
                target_vllm_config, Attention,
            ).keys()
        )
        self.draft_attn_layer_names = sorted(
            all_attn_names - attn_names_before
        )

        self._num_layers = len(self.draft_attn_layer_names)
        self._block_size = target_vllm_config.cache_config.block_size

        # Read per-rank (TP-sharded) head counts from the actual Attention
        # modules, NOT from hf_config which has the full-model counts.
        forward_ctx = (
            target_vllm_config.compilation_config.static_forward_context
        )
        first_attn: Attention = forward_ctx[self.draft_attn_layer_names[0]]
        self._num_heads = first_attn.num_heads
        self._num_kv_heads = first_attn.num_kv_heads
        self._head_dim = first_attn.head_size

        logger.info(
            "Draft model loaded — layers=%d, heads_per_rank=%d, "
            "kv_heads_per_rank=%d, head_dim=%d, draft_attn_layers=%d",
            self._num_layers,
            self._num_heads,
            self._num_kv_heads,
            self._head_dim,
            len(self.draft_attn_layer_names),
        )

    def init_attn_metadata_builder(self, runner: GPUModelRunner) -> None:
        """Locate the AttentionMetadataBuilder for the draft model layers.

        Must be called after ``runner.initialize_attn_backend()``.
        """
        self.runner = runner
        chosen_layer = self.draft_attn_layer_names[0]
        for kv_cache_group in runner.attn_groups:
            for attn_group in kv_cache_group:
                if chosen_layer in attn_group.layer_names:
                    self._attn_metadata_builder = (
                        attn_group.get_metadata_builder()
                    )
                    return
        raise RuntimeError(
            "Could not find AttentionMetadataBuilder for draft model layers"
        )

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def run(
        self,
        req_ids: list[str],
        prompt_token_ids_list: list[list[int]],
    ) -> SpecPrefillBatchMetadata:
        """Run speculative prefill for a batch of new prefill requests."""
        assert self.model is not None, "Draft model not loaded"
        batch_meta = SpecPrefillBatchMetadata()

        for req_id, prompt_token_ids in zip(req_ids, prompt_token_ids_list):
            meta = self._speculate_single(req_id, prompt_token_ids)
            batch_meta.add(meta)

        return batch_meta

    # ------------------------------------------------------------------
    # Per-request speculation
    # ------------------------------------------------------------------
    def _speculate_single(
        self,
        req_id: str,
        prompt_token_ids: list[int],
    ) -> SpecPrefillMetadata:
        cfg = self.config
        prompt_len = len(prompt_token_ids)
        total_len = prompt_len + cfg.look_ahead_cnt

        query_buffer: list[list[torch.Tensor]] = [
            [] for _ in range(self._num_layers)
        ]
        hooks = self._register_query_hooks(query_buffer)

        try:
            # --- Step 1: prefill prompt on draft model (chunked) ---
            all_input_ids = torch.tensor(
                prompt_token_ids, dtype=torch.long, device=self.device
            )
            all_positions = torch.arange(
                prompt_len, dtype=torch.long, device=self.device
            )
            block_table = self._make_block_table(total_len)

            chunk_size = self.config.draft_prefill_chunk_size
            for chunk_start in range(0, prompt_len, chunk_size):
                chunk_end = min(chunk_start + chunk_size, prompt_len)
                chunk_ids = all_input_ids[chunk_start:chunk_end]
                chunk_pos = all_positions[chunk_start:chunk_end]
                chunk_slots = chunk_pos.clone()
                chunk_query_len = chunk_end - chunk_start
                seq_len_so_far = chunk_end

                cad = self._build_common_attn_metadata(
                    num_tokens=chunk_query_len,
                    seq_len=seq_len_so_far,
                    query_len=chunk_query_len,
                    slot_mapping=chunk_slots,
                    block_table=block_table,
                )
                attn_metadata = self._build_attn_metadata(cad)
                self._run_forward(
                    chunk_ids, chunk_pos, attn_metadata, chunk_slots
                )

            # Prefill queries are not useful for importance scoring.
            for buf in query_buffer:
                buf.clear()

            cur_token_id = self._predict_next_token(
                all_input_ids[-1:], all_positions[-1:], cad, block_table
            )

            # --- Step 2: look-ahead autoregressive decode ---
            actual_look_ahead = cfg.look_ahead_cnt
            stop_set = set(cfg.stop_token_ids)
            seq_len = prompt_len

            for step in range(cfg.look_ahead_cnt):
                decode_pos = torch.tensor(
                    [seq_len], dtype=torch.long, device=self.device
                )
                decode_slot = decode_pos.clone()

                decode_cad = self._build_common_attn_metadata(
                    num_tokens=1,
                    seq_len=seq_len + 1,
                    query_len=1,
                    slot_mapping=decode_slot,
                    block_table=block_table,
                )
                decode_attn = self._build_attn_metadata(decode_cad)

                decode_ids = cur_token_id.unsqueeze(0)
                self._run_forward(
                    decode_ids, decode_pos, decode_attn, decode_slot
                )
                seq_len += 1

                next_token_id = self._predict_next_token(
                    decode_ids, decode_pos, decode_cad, block_table
                )

                if not cfg.ignore_eos and next_token_id.item() in stop_set:
                    actual_look_ahead = step + 1
                    break
                cur_token_id = next_token_id

        finally:
            for h in hooks:
                h.remove()

        # --- Step 3: read keys from paged KV-cache ---
        all_keys = self._collect_keys_from_paged_cache(prompt_len)

        # --- Step 4: compute importance and select ---
        all_queries = self._collect_queries(query_buffer, actual_look_ahead)

        all_queries = all_queries.cpu()
        if all_keys.device.type != "cpu":
            all_keys = all_keys.cpu()

        attn_scores = self._compute_attention_scores(
            all_queries, all_keys, actual_look_ahead
        )
        importance = compute_token_importance(
            attn_scores, pool_kernel_size=cfg.pool_kernel_size
        )
        kept_indices = select_kept_indices(
            importance,
            keep_percentage=cfg.keep_percentage,
            chunk_selection=cfg.chunk_selection,
            chunk_size=cfg.chunk_size,
        )

        kept_indices_cpu = kept_indices.tolist()
        compressed_tokens = [prompt_token_ids[i] for i in kept_indices_cpu]

        return SpecPrefillMetadata(
            req_id=req_id,
            original_prompt_len=prompt_len,
            compressed_token_ids=compressed_tokens,
            position_ids=kept_indices_cpu,
        )

    # ------------------------------------------------------------------
    # Attention metadata helpers
    # ------------------------------------------------------------------
    def _make_block_table(self, max_seq_len: int) -> torch.Tensor:
        """Sequential block table for our private draft-model KV pool."""
        num_blocks = (max_seq_len + self._block_size - 1) // self._block_size
        return torch.arange(
            num_blocks, dtype=torch.int32, device=self.device
        ).unsqueeze(0)

    def _build_common_attn_metadata(
        self,
        num_tokens: int,
        seq_len: int,
        query_len: int,
        slot_mapping: torch.Tensor,
        block_table: torch.Tensor,
    ) -> CommonAttentionMetadata:
        return CommonAttentionMetadata(
            query_start_loc=torch.tensor(
                [0, query_len], dtype=torch.int32, device=self.device
            ),
            query_start_loc_cpu=torch.tensor(
                [0, query_len], dtype=torch.int32
            ),
            seq_lens=torch.tensor(
                [seq_len], dtype=torch.int32, device=self.device
            ),
            num_reqs=1,
            num_actual_tokens=num_tokens,
            max_query_len=query_len,
            max_seq_len=seq_len,
            block_table_tensor=block_table,
            slot_mapping=slot_mapping,
        )

    def _build_attn_metadata(self, cad: CommonAttentionMetadata):
        assert self._attn_metadata_builder is not None
        return self._attn_metadata_builder.build(
            common_prefix_len=0,
            common_attn_metadata=cad,
            fast_build=True,
        )

    # ------------------------------------------------------------------
    # Draft-model forward helpers
    # ------------------------------------------------------------------
    def _run_forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        attn_metadata,
        slot_mapping: torch.Tensor,
    ) -> None:
        """Run the draft model forward under ``set_forward_context``."""
        assert self.vllm_config is not None
        num_tokens = input_ids.shape[0]

        per_layer_attn_metadata: dict = {}
        for layer_name in self.draft_attn_layer_names:
            per_layer_attn_metadata[layer_name] = attn_metadata

        slot_mapping_dict: dict[str, torch.Tensor] = {}
        for layer_name in self.draft_attn_layer_names:
            slot_mapping_dict[layer_name] = slot_mapping

        with set_forward_context(
            per_layer_attn_metadata,
            self.vllm_config,
            num_tokens=num_tokens,
            slot_mapping=slot_mapping_dict,
        ):
            self.model(input_ids=input_ids, positions=positions)

    def _predict_next_token(
        self,
        last_ids: torch.Tensor,
        last_pos: torch.Tensor,
        cad: CommonAttentionMetadata,
        block_table: torch.Tensor,
    ) -> torch.Tensor:
        """Run a read-only forward on the last token to get logits.

        Uses PADDING_SLOT_ID (-1) so no KV-cache is modified.
        """
        assert self.vllm_config is not None
        PAD = -1
        pad_slot = torch.full_like(last_pos, PAD)

        pad_cad = self._build_common_attn_metadata(
            num_tokens=1,
            seq_len=cad.seq_lens[0].item(),
            query_len=1,
            slot_mapping=pad_slot,
            block_table=block_table,
        )
        pad_attn = self._build_attn_metadata(pad_cad)

        per_layer_attn_metadata: dict = {}
        slot_mapping_dict: dict[str, torch.Tensor] = {}
        for layer_name in self.draft_attn_layer_names:
            per_layer_attn_metadata[layer_name] = pad_attn
            slot_mapping_dict[layer_name] = pad_slot

        with set_forward_context(
            per_layer_attn_metadata,
            self.vllm_config,
            num_tokens=1,
            slot_mapping=slot_mapping_dict,
        ):
            hidden = self.model(input_ids=last_ids, positions=last_pos)

        logits = self.model.compute_logits(hidden)
        return logits.argmax(dim=-1).squeeze(0)

    # ------------------------------------------------------------------
    # Query hook registration
    # ------------------------------------------------------------------
    def _register_query_hooks(
        self,
        query_buffer: list[list[torch.Tensor]],
    ) -> list[torch.utils.hooks.RemovableHook]:
        """Register pre-hooks on draft-model ``Attention`` to capture Q."""
        assert self.vllm_config is not None
        hooks: list[torch.utils.hooks.RemovableHook] = []

        forward_ctx = (
            self.vllm_config.compilation_config.static_forward_context
        )

        for layer_idx, layer_name in enumerate(self.draft_attn_layer_names):
            attn_module = forward_ctx[layer_name]

            def _pre_hook(
                module: nn.Module,
                args: tuple,
                buf: list[list[torch.Tensor]] = query_buffer,
                idx: int = layer_idx,
            ) -> None:
                query = args[0]  # post-RoPE: [num_tokens, num_heads*head_dim]
                buf[idx].append(query[-1:].detach().clone())

            h = attn_module.register_forward_pre_hook(_pre_hook)
            hooks.append(h)
        return hooks

    # ------------------------------------------------------------------
    # Collect Q / K tensors
    # ------------------------------------------------------------------
    def _collect_queries(
        self,
        query_buffer: list[list[torch.Tensor]],
        actual_look_ahead: int,
    ) -> torch.Tensor:
        """Stack queries: ``[num_layers, look_ahead, num_heads*head_dim]``."""
        per_layer = []
        for layer_bufs in query_buffer:
            qs = layer_bufs[:actual_look_ahead]
            if qs:
                per_layer.append(torch.cat(qs, dim=0))
            else:
                per_layer.append(
                    torch.zeros(
                        1, self._num_heads * self._head_dim,
                        device=self.device, dtype=self.dtype,
                    )
                )
        return torch.stack(per_layer, dim=0)

    def _collect_keys_from_paged_cache(
        self,
        prompt_len: int,
    ) -> torch.Tensor:
        """Read prompt-region keys from draft-model paged KV-cache.

        Returns: ``[num_layers, num_kv_heads, prompt_len, head_dim]``
        """
        assert self.vllm_config is not None
        forward_ctx = (
            self.vllm_config.compilation_config.static_forward_context
        )
        bs = self._block_size

        slot_indices = torch.arange(prompt_len, device=self.device)
        block_indices = slot_indices // bs
        block_offsets = slot_indices % bs

        per_layer = []
        for layer_name in self.draft_attn_layer_names:
            attn_layer: Attention = forward_ctx[layer_name]
            kv_cache = attn_layer.kv_cache[0]
            # Standard layout: [2, num_blocks, block_size, num_kv_heads, head_dim]
            if kv_cache.dim() == 5 and kv_cache.shape[0] == 2:
                key_cache = kv_cache[0]
            else:
                key_cache = kv_cache
            keys = key_cache[block_indices, block_offsets]
            per_layer.append(keys)

        stacked = torch.stack(per_layer, dim=0)
        # Move to CPU before .contiguous() to avoid a large GPU allocation
        # (the caller moves to CPU anyway for importance scoring).
        return stacked.transpose(1, 2).cpu().contiguous()

    # ------------------------------------------------------------------
    # Attention score computation
    # ------------------------------------------------------------------
    def _compute_attention_scores(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        actual_look_ahead: int,
    ) -> torch.Tensor:
        """Compute Q*K^T attention scores.

        Args:
            queries: ``[num_layers, look_ahead, num_heads * head_dim]``
            keys: ``[num_layers, num_kv_heads, prompt_len, head_dim]``

        Returns:
            ``[num_layers, num_heads, look_ahead, prompt_len]``
        """
        num_layers = queries.shape[0]
        look_ahead = queries.shape[1]

        q = queries.view(
            num_layers, look_ahead, self._num_heads, self._head_dim
        )
        q = q.transpose(1, 2)

        repeat_factor = self._num_heads // self._num_kv_heads
        k = keys
        if repeat_factor > 1:
            k = k.repeat_interleave(repeat_factor, dim=1)

        scale = 1.0 / math.sqrt(self._head_dim)
        attn = torch.matmul(q, k.transpose(-1, -2)) * scale
        return attn
