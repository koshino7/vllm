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
from vllm.distributed.parallel_state import get_tp_group
from vllm.forward_context import set_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.attention import Attention
from vllm.model_executor.model_loader import get_model
from vllm.v1.attention.backend import (
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
)
from vllm.v1.spec_prefill.compression import select_kept_indices
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

        # Pre-allocated reusable tensors for single-request metadata
        # (avoids per-call torch.tensor → H2D copies).
        self._qsl_gpu = torch.zeros(2, dtype=torch.int32, device=device)
        self._qsl_cpu = torch.zeros(2, dtype=torch.int32)
        self._seq_lens_1 = torch.zeros(1, dtype=torch.int32, device=device)
        # Constant decode-phase tensors (query_len == 1).
        self._decode_qsl_gpu = torch.tensor(
            [0, 1], dtype=torch.int32, device=device
        )
        self._decode_qsl_cpu = torch.tensor([0, 1], dtype=torch.int32)
        self._pad_slot_1 = torch.tensor(
            [-1], dtype=torch.long, device=device
        )

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
        """Run speculative prefill for a batch of new prefill requests.

        All TP ranks execute the draft-model forward (required for TP
        correctness), but only rank 0 performs scoring & compression.
        Results are broadcast to the other ranks.
        """
        assert self.model is not None, "Draft model not loaded"
        cfg = self.config
        batch_meta = SpecPrefillBatchMetadata()
        tp = get_tp_group()
        is_rank0 = tp.rank_in_group == 0

        eligible: list[tuple[str, list[int]]] = [
            (rid, ptids)
            for rid, ptids in zip(req_ids, prompt_token_ids_list)
            if len(ptids) >= cfg.min_prompt_len
        ]
        if not eligible:
            return batch_meta

        e_prompts = [p for _, p in eligible]

        # Phase 1+2: all ranks run the draft model (prefill + decode).
        query_buffers, actual_look_aheads, prompt_lens, blk_offsets = (
            self._run_draft_forward_batch(e_prompts)
        )

        # Phase 2b: rank 0 scores and compresses.
        per_req_results: list[tuple[list[int], list[int], int]] = []
        if is_rank0:
            for i, (_, ptids) in enumerate(eligible):
                all_queries = self._collect_queries(
                    query_buffers[i], actual_look_aheads[i]
                )
                importance = self._compute_importance_gpu(
                    all_queries, prompt_lens[i], cfg.pool_kernel_size,
                    slot_offset=blk_offsets[i] * self._block_size,
                )
                kept_indices = select_kept_indices(
                    importance,
                    keep_percentage=cfg.keep_percentage,
                    chunk_selection=cfg.chunk_selection,
                    chunk_size=cfg.chunk_size,
                )
                kept_cpu = kept_indices.tolist()
                compressed = [ptids[j] for j in kept_cpu]
                per_req_results.append(
                    (compressed, kept_cpu, prompt_lens[i])
                )

        # Phase 3: rank 0 broadcasts compression results.
        if tp.world_size > 1:
            bcast: list = [per_req_results if is_rank0 else None]
            tp.broadcast_object_list(bcast, src=0)
            if not is_rank0:
                per_req_results = bcast[0]

        # Build metadata (all ranks).
        for (req_id, _), (compressed, pos_ids, orig_len) in zip(
            eligible, per_req_results
        ):
            batch_meta.add(
                SpecPrefillMetadata(
                    req_id=req_id,
                    original_prompt_len=orig_len,
                    compressed_token_ids=compressed,
                    position_ids=pos_ids,
                )
            )
        return batch_meta

    # ------------------------------------------------------------------
    # Draft-model forward — batched across requests (all TP ranks)
    # ------------------------------------------------------------------
    def _run_draft_forward_batch(
        self,
        prompts: list[list[int]],
    ) -> tuple[
        list[list[list[torch.Tensor]]],  # per-request query buffers
        list[int],                        # per-request actual_look_ahead
        list[int],                        # per-request prompt_len
        list[int],                        # per-request block_offset
    ]:
        """Run draft-model prefill + batched look-ahead decode.

        Prefill is still per-request (large chunks are compute-bound).
        Look-ahead decode is batched: 8 forward passes with N tokens
        instead of 8*N separate forward passes.
        """
        cfg = self.config
        N = len(prompts)
        prompt_lens = [len(p) for p in prompts]

        # Allocate non-overlapping block regions per request.
        block_offsets: list[int] = []
        block_tables: list[torch.Tensor] = []
        cur_block = 0
        for plen in prompt_lens:
            total_len = plen + cfg.look_ahead_cnt
            num_blocks = (total_len + self._block_size - 1) // self._block_size
            bt = torch.arange(
                cur_block, cur_block + num_blocks,
                dtype=torch.int32, device=self.device,
            ).unsqueeze(0)
            block_tables.append(bt)
            block_offsets.append(cur_block)
            cur_block += num_blocks

        # Per-request query buffers (only used during decode).
        query_buffers: list[list[list[torch.Tensor]]] = [
            [[] for _ in range(self._num_layers)] for _ in range(N)
        ]

        # ----------------------------------------------------------
        # Phase 1: per-request chunked prefill (no hooks needed)
        # ----------------------------------------------------------
        chunk_size = cfg.draft_prefill_chunk_size
        for i in range(N):
            input_ids = torch.tensor(
                prompts[i], dtype=torch.long, device=self.device
            )
            positions = torch.arange(
                prompt_lens[i], dtype=torch.long, device=self.device
            )
            slot_base = block_offsets[i] * self._block_size

            for chunk_start in range(0, prompt_lens[i], chunk_size):
                chunk_end = min(chunk_start + chunk_size, prompt_lens[i])
                chunk_ids = input_ids[chunk_start:chunk_end]
                chunk_pos = positions[chunk_start:chunk_end]
                chunk_slots = chunk_pos + slot_base
                qlen = chunk_end - chunk_start

                cad = self._cad_single(
                    num_tokens=qlen,
                    seq_len=chunk_end,
                    query_len=qlen,
                    slot_mapping=chunk_slots,
                    block_table=block_tables[i],
                )
                attn_metadata = self._build_attn_metadata(cad)
                self._run_forward(chunk_ids, chunk_pos, attn_metadata,
                                  chunk_slots)

        # Get initial next-token for each request.
        cur_tokens: list[torch.Tensor] = []
        for i in range(N):
            last_id = torch.tensor(
                [prompts[i][-1]], dtype=torch.long, device=self.device
            )
            last_pos = torch.tensor(
                [prompt_lens[i] - 1], dtype=torch.long, device=self.device
            )
            cad_last = self._cad_single_decode(
                seq_len=prompt_lens[i],
                slot_mapping=self._pad_slot_1,
                block_table=block_tables[i],
            )
            cur_tokens.append(
                self._predict_next_token(
                    last_id, last_pos, cad_last, block_tables[i]
                )
            )

        # ----------------------------------------------------------
        # Phase 2: batched look-ahead decode with per-request hooks
        # ----------------------------------------------------------
        self._batched_hook_buffers = query_buffers
        self._batched_hook_N = N
        hooks = self._register_batched_query_hooks()

        actual_look_aheads = [cfg.look_ahead_cnt] * N
        stop_set = set(cfg.stop_token_ids)
        seq_lens = list(prompt_lens)  # mutable copy
        active = list(range(N))       # indices of still-active requests

        # Pre-allocate decode buffers sized for max batch (N).
        max_bt_cols = max(bt.shape[1] for bt in block_tables)
        buf_ids = torch.empty(N, dtype=torch.long, device=self.device)
        buf_pos = torch.empty(N, dtype=torch.long, device=self.device)
        buf_slots = torch.empty(N, dtype=torch.long, device=self.device)
        buf_seq_lens = torch.empty(N, dtype=torch.int32, device=self.device)
        buf_qsl_gpu = torch.arange(
            N + 1, dtype=torch.int32, device=self.device
        )
        buf_qsl_cpu = torch.arange(N + 1, dtype=torch.int32)
        buf_bt = torch.zeros(
            N, max_bt_cols, dtype=torch.int32, device=self.device
        )
        buf_pad_slots = torch.full(
            (N,), -1, dtype=torch.long, device=self.device
        )
        # Fill block table rows once (only changes if requests drop out).
        for j in range(N):
            cols = block_tables[j].shape[1]
            buf_bt[j, :cols] = block_tables[j][0]

        try:
            for step in range(cfg.look_ahead_cnt):
                if not active:
                    break

                na = len(active)
                for j, a in enumerate(active):
                    buf_ids[j] = cur_tokens[a]
                    buf_pos[j] = seq_lens[a]
                    buf_slots[j] = (
                        block_offsets[a] * self._block_size + seq_lens[a]
                    )
                    buf_seq_lens[j] = seq_lens[a] + 1

                # Rebuild padded block-table only for active subset.
                act_bt = buf_bt[:na]
                if na < N:
                    act_bt = act_bt.clone()
                    for j, a in enumerate(active):
                        cols = block_tables[a].shape[1]
                        act_bt[j].zero_()
                        act_bt[j, :cols] = block_tables[a][0]

                ids_v = buf_ids[:na]
                pos_v = buf_pos[:na]
                slots_v = buf_slots[:na]
                sls_v = buf_seq_lens[:na]
                qsl_v = buf_qsl_gpu[: na + 1]
                qsl_cpu_v = buf_qsl_cpu[: na + 1]
                max_sl = int(sls_v.max().item())

                cad = CommonAttentionMetadata(
                    query_start_loc=qsl_v,
                    query_start_loc_cpu=qsl_cpu_v,
                    seq_lens=sls_v,
                    num_reqs=na,
                    num_actual_tokens=na,
                    max_query_len=1,
                    max_seq_len=max_sl,
                    block_table_tensor=act_bt,
                    slot_mapping=slots_v,
                )
                attn_metadata = self._build_attn_metadata(cad)

                self._batched_hook_active = active
                self._hooks_enabled = True
                self._run_forward(ids_v, pos_v, attn_metadata, slots_v)
                self._hooks_enabled = False

                for j, a in enumerate(active):
                    seq_lens[a] += 1

                # Predict next tokens (batched, padding slot, no KV write).
                pad_v = buf_pad_slots[:na]
                pad_cad = CommonAttentionMetadata(
                    query_start_loc=qsl_v,
                    query_start_loc_cpu=qsl_cpu_v,
                    seq_lens=sls_v,
                    num_reqs=na,
                    num_actual_tokens=na,
                    max_query_len=1,
                    max_seq_len=max_sl,
                    block_table_tensor=act_bt,
                    slot_mapping=pad_v,
                )
                pad_attn = self._build_attn_metadata(pad_cad)
                per_layer_meta: dict = {}
                slot_dict: dict[str, torch.Tensor] = {}
                for ln in self.draft_attn_layer_names:
                    per_layer_meta[ln] = pad_attn
                    slot_dict[ln] = pad_v
                with set_forward_context(
                    per_layer_meta, self.vllm_config,
                    num_tokens=na, slot_mapping=slot_dict,
                ):
                    hidden = self.model(
                        input_ids=ids_v, positions=pos_v,
                    )
                logits = self.model.compute_logits(hidden)
                next_ids = logits.argmax(dim=-1)

                still_active = []
                for j, a in enumerate(active):
                    nid = next_ids[j]
                    if not cfg.ignore_eos and nid.item() in stop_set:
                        actual_look_aheads[a] = step + 1
                    else:
                        cur_tokens[a] = nid
                        still_active.append(a)
                active = still_active

        finally:
            for h in hooks:
                h.remove()

        return query_buffers, actual_look_aheads, prompt_lens, block_offsets

    # ------------------------------------------------------------------
    # Attention metadata helpers (zero-allocation fast paths)
    # ------------------------------------------------------------------
    def _cad_single(
        self,
        num_tokens: int,
        seq_len: int,
        query_len: int,
        slot_mapping: torch.Tensor,
        block_table: torch.Tensor,
    ) -> CommonAttentionMetadata:
        """Build single-request CAD by filling pre-allocated buffers."""
        self._qsl_gpu[1] = query_len
        self._qsl_cpu[1] = query_len
        self._seq_lens_1[0] = seq_len
        return CommonAttentionMetadata(
            query_start_loc=self._qsl_gpu,
            query_start_loc_cpu=self._qsl_cpu,
            seq_lens=self._seq_lens_1,
            num_reqs=1,
            num_actual_tokens=num_tokens,
            max_query_len=query_len,
            max_seq_len=seq_len,
            block_table_tensor=block_table,
            slot_mapping=slot_mapping,
        )

    def _cad_single_decode(
        self,
        seq_len: int,
        slot_mapping: torch.Tensor,
        block_table: torch.Tensor,
    ) -> CommonAttentionMetadata:
        """Decode fast-path: query_len=1, no tensor creation at all."""
        self._seq_lens_1[0] = seq_len
        return CommonAttentionMetadata(
            query_start_loc=self._decode_qsl_gpu,
            query_start_loc_cpu=self._decode_qsl_cpu,
            seq_lens=self._seq_lens_1,
            num_reqs=1,
            num_actual_tokens=1,
            max_query_len=1,
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
        pad_cad = self._cad_single_decode(
            seq_len=cad.seq_lens[0].item(),
            slot_mapping=self._pad_slot_1,
            block_table=block_table,
        )
        pad_attn = self._build_attn_metadata(pad_cad)

        per_layer_attn_metadata: dict = {}
        slot_mapping_dict: dict[str, torch.Tensor] = {}
        for layer_name in self.draft_attn_layer_names:
            per_layer_attn_metadata[layer_name] = pad_attn
            slot_mapping_dict[layer_name] = self._pad_slot_1

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
    def _register_batched_query_hooks(
        self,
    ) -> list[torch.utils.hooks.RemovableHook]:
        """Register pre-hooks that split per-request queries during
        batched decode.  Uses ``self._batched_hook_active`` to know
        which request indices are active in the current forward.
        """
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
                idx: int = layer_idx,
            ) -> None:
                if not getattr(self, "_hooks_enabled", False):
                    return
                query = args[0]  # [n_active, num_heads*head_dim]
                active = self._batched_hook_active
                for j, a in enumerate(active):
                    self._batched_hook_buffers[a][idx].append(
                        query[j : j + 1].detach().clone()
                    )

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

    # ------------------------------------------------------------------
    # GPU importance scoring (per-layer to bound memory)
    # ------------------------------------------------------------------
    def _compute_importance_gpu(
        self,
        queries: torch.Tensor,
        prompt_len: int,
        pool_kernel_size: int | None,
        slot_offset: int = 0,
    ) -> torch.Tensor:
        """Compute token importance directly on GPU, one layer at a time.

        This avoids materialising the full ``[L, H, look_ahead, prompt_len]``
        attention-score tensor, keeping peak memory at
        ``O(H * look_ahead * prompt_len)`` per layer.

        Args:
            queries: ``[num_layers, look_ahead, num_heads * head_dim]`` (GPU)
            prompt_len: number of prompt tokens.
            pool_kernel_size: smoothing kernel (None to skip).
            slot_offset: starting slot index for this request's KV region.

        Returns:
            1-D importance tensor ``[prompt_len]`` (GPU).
        """
        assert self.vllm_config is not None
        forward_ctx = (
            self.vllm_config.compilation_config.static_forward_context
        )
        bs = self._block_size
        H = self._num_heads
        D = self._head_dim
        KVH = self._num_kv_heads
        repeat_factor = H // KVH
        scale = 1.0 / math.sqrt(D)

        abs_slots = torch.arange(prompt_len, device=self.device) + slot_offset
        block_indices = abs_slots // bs
        block_offsets = abs_slots % bs

        running_max: torch.Tensor | None = None

        for layer_idx, layer_name in enumerate(self.draft_attn_layer_names):
            # -- keys for this layer from paged cache --
            attn_layer: Attention = forward_ctx[layer_name]
            kv_cache = attn_layer.kv_cache[0]
            if kv_cache.dim() == 5 and kv_cache.shape[0] == 2:
                key_cache = kv_cache[0]
            else:
                key_cache = kv_cache
            # [prompt_len, KVH, D]
            keys = key_cache[block_indices, block_offsets]
            # -> [KVH, prompt_len, D]
            keys = keys.transpose(0, 1)
            if repeat_factor > 1:
                keys = keys.repeat_interleave(repeat_factor, dim=0)
            # keys: [H, prompt_len, D]

            # -- queries for this layer --
            q = queries[layer_idx]  # [look_ahead, H*D]
            q = q.view(-1, H, D).transpose(0, 1)  # [H, look_ahead, D]

            # -- attention scores --
            # [H, look_ahead, prompt_len]
            attn = torch.matmul(q, keys.transpose(-1, -2)) * scale
            attn = torch.nn.functional.softmax(attn, dim=-1)

            if pool_kernel_size:
                look_ahead = attn.shape[1]
                attn = torch.nn.functional.avg_pool1d(
                    attn.reshape(H * look_ahead, 1, prompt_len),
                    kernel_size=pool_kernel_size,
                    padding=pool_kernel_size // 2,
                    stride=1,
                ).reshape(H, look_ahead, -1)

            # max over heads -> [look_ahead, prompt_len]
            layer_max = attn.max(0)[0]

            if running_max is None:
                running_max = layer_max
            else:
                running_max = torch.max(running_max, layer_max)

            del keys, attn

        assert running_max is not None
        importance = running_max.mean(0)  # [prompt_len]
        return importance
