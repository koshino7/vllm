# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Speculative prefill runner for vLLM v1.

This module loads a lightweight draft model on the same GPU as the main
model, runs look-ahead decoding to capture query/key vectors, estimates
token importance via attention scores, and returns compressed prompt
metadata to the main model runner.

The draft model's weight and activation memory are properly accounted for
by vLLM's memory profiler (DeviceMemoryProfiler + profile_run) so that
KV-cache allocation leaves enough headroom.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from vllm.config import SpecPrefillConfig
from vllm.logger import init_logger
from vllm.v1.spec_prefill.compression import (
    compute_token_importance,
    select_kept_indices,
)
from vllm.v1.spec_prefill.metadata import (
    SpecPrefillBatchMetadata,
    SpecPrefillMetadata,
)

if TYPE_CHECKING:
    from transformers import PreTrainedModel

logger = init_logger(__name__)


class SpecPrefillRunner:
    """Orchestrates the speculative prefill pipeline.

    The draft model lives on the same GPU as the main model.  Its weight
    and peak-activation memory are profiled during startup so that the
    KV-cache allocator reserves enough headroom.

    Lifecycle
    ---------
    1. ``load_draft_model``: Load the draft model (GPU, inside
       DeviceMemoryProfiler).
    2. ``dummy_run``: Called during ``profile_run`` so that draft-model
       activation memory is captured by ``torch_peak_increase``.
    3. ``run``: Called by ``GPUModelRunner`` for each scheduling step that
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
        self.draft_model: PreTrainedModel | None = None
        self._num_layers: int = 0
        self._num_heads: int = 0
        self._num_kv_heads: int = 0
        self._head_dim: int = 0

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------
    def load_draft_model(self) -> None:
        """Load the draft model onto the same GPU as the main model."""
        from transformers import AutoModelForCausalLM

        logger.info(
            "Loading spec-prefill draft model on %s: %s",
            self.device,
            self.config.spec_model,
        )
        self.draft_model = AutoModelForCausalLM.from_pretrained(
            self.config.spec_model,
            torch_dtype=self.dtype,
            attn_implementation="sdpa",
        ).to(self.device)
        self.draft_model.eval()

        hf_config = self.draft_model.config
        self._num_layers = hf_config.num_hidden_layers
        self._num_heads = hf_config.num_attention_heads
        self._num_kv_heads = getattr(
            hf_config, "num_key_value_heads", self._num_heads
        )
        self._head_dim = hf_config.hidden_size // self._num_heads
        logger.info(
            "Draft model loaded on %s — layers=%d, heads=%d, kv_heads=%d, "
            "head_dim=%d",
            self.device,
            self._num_layers,
            self._num_heads,
            self._num_kv_heads,
            self._head_dim,
        )

    # ------------------------------------------------------------------
    # Dummy run for memory profiling
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def dummy_run(self, max_num_tokens: int) -> None:
        """Run a dummy forward pass so that peak activation memory is
        captured by ``torch.cuda.max_memory_allocated`` during
        ``profile_run``.
        """
        assert self.draft_model is not None, "Draft model not loaded"
        cfg = self.config
        dummy_ids = torch.zeros(
            (1, max_num_tokens), dtype=torch.long, device=self.device
        )
        past_key_values = None
        for i in range(cfg.look_ahead_cnt + 1):
            if i == 0:
                base_out = self.draft_model.model(
                    input_ids=dummy_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                past_key_values = base_out.past_key_values
                last_hidden = base_out.last_hidden_state[:, -1:, :]
                dummy_ids = self.draft_model.lm_head(last_hidden).argmax(
                    dim=-1
                )
                del base_out, last_hidden
            else:
                outputs = self.draft_model(
                    input_ids=dummy_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                past_key_values = outputs.past_key_values
                dummy_ids = outputs.logits[:, -1:, :].argmax(dim=-1)
                del outputs
        del past_key_values
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def run(
        self,
        req_ids: list[str],
        prompt_token_ids_list: list[list[int]],
    ) -> SpecPrefillBatchMetadata:
        """Run speculative prefill for a batch of new prefill requests."""
        assert self.draft_model is not None, "Draft model not loaded"
        batch_meta = SpecPrefillBatchMetadata()

        for req_id, prompt_token_ids in zip(req_ids, prompt_token_ids_list):
            meta = self._speculate_single(req_id, prompt_token_ids)
            batch_meta.add(meta)

        torch.cuda.empty_cache()
        return batch_meta

    # ------------------------------------------------------------------
    # Per-request speculation
    # ------------------------------------------------------------------
    def _speculate_single(
        self,
        req_id: str,
        prompt_token_ids: list[int],
    ) -> SpecPrefillMetadata:
        """Run look-ahead + importance estimation for one request."""
        cfg = self.config
        input_ids = torch.tensor(
            [prompt_token_ids], dtype=torch.long, device=self.device
        )

        query_buffer: list[list[torch.Tensor]] = [
            [] for _ in range(self._num_layers)
        ]
        hooks = self._register_query_hooks(query_buffer)

        past_key_values = None
        cur_input_ids = input_ids
        actual_look_ahead = cfg.look_ahead_cnt
        stop_set = set(cfg.stop_token_ids)

        try:
            for step in range(cfg.look_ahead_cnt + 1):
                if step == 0:
                    base_out = self.draft_model.model(
                        input_ids=cur_input_ids,
                        past_key_values=past_key_values,
                        use_cache=True,
                    )
                    past_key_values = base_out.past_key_values
                    last_hidden = base_out.last_hidden_state[:, -1:, :]
                    logits_last = self.draft_model.lm_head(last_hidden)
                    cur_input_ids = logits_last.argmax(dim=-1)
                    del base_out, last_hidden, logits_last
                    for layer_buf in query_buffer:
                        layer_buf.clear()
                    continue

                outputs = self.draft_model(
                    input_ids=cur_input_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                past_key_values = outputs.past_key_values

                next_token = outputs.logits[:, -1, :].argmax(dim=-1)
                if not cfg.ignore_eos and next_token.item() in stop_set:
                    actual_look_ahead = step
                    break
                cur_input_ids = next_token.unsqueeze(0)
        finally:
            for h in hooks:
                h.remove()

        all_keys = self._collect_keys_from_cache(
            past_key_values, len(prompt_token_ids)
        )
        del past_key_values, outputs

        all_queries = self._collect_queries(query_buffer, actual_look_ahead)

        all_queries = all_queries.cpu()
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
            original_prompt_len=len(prompt_token_ids),
            compressed_token_ids=compressed_tokens,
            position_ids=kept_indices_cpu,
        )

    # ------------------------------------------------------------------
    # Query hook registration
    # ------------------------------------------------------------------
    def _register_query_hooks(
        self,
        query_buffer: list[list[torch.Tensor]],
    ) -> list[torch.utils.hooks.RemovableHook]:
        """Register forward hooks to capture the last token's Q per layer."""
        hooks: list[torch.utils.hooks.RemovableHook] = []
        for layer_idx, layer in enumerate(
            self.draft_model.model.layers  # type: ignore[union-attr]
        ):
            attn_module = layer.self_attn

            def _hook(
                module: torch.nn.Module,
                args: tuple,
                kwargs: dict,
                output: object,
                buf: list[list[torch.Tensor]] = query_buffer,
                idx: int = layer_idx,
            ) -> None:
                if args:
                    hidden = args[0]
                else:
                    hidden = kwargs.get("hidden_states")
                if hidden is None:
                    return
                last_hidden = hidden[:, -1:, :]
                q_last = module.q_proj(last_hidden)  # type: ignore[attr-defined]
                buf[idx].append(q_last.detach())

            h = attn_module.register_forward_hook(_hook, with_kwargs=True)
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
        """Stack captured queries into ``[num_layers, look_ahead, H*D]``."""
        per_layer = []
        for layer_bufs in query_buffer:
            qs = layer_bufs[:actual_look_ahead]
            if qs:
                per_layer.append(torch.cat(qs, dim=1))
            else:
                per_layer.append(
                    layer_bufs[0]
                    if layer_bufs
                    else torch.zeros(1, 1, self._num_heads * self._head_dim)
                )
        return torch.stack(per_layer, dim=0).squeeze(1)

    def _collect_keys_from_cache(
        self,
        past_key_values: tuple,
        prompt_len: int,
    ) -> torch.Tensor:
        """Extract prompt-region keys from HF KV cache.

        Returns: ``[num_layers, prompt_len, num_kv_heads, head_dim]``
        """
        per_layer = []
        for layer_kv in past_key_values:
            k = layer_kv[0]  # [batch, num_kv_heads, total_len, head_dim]
            k_prompt = k[:, :, :prompt_len, :].squeeze(0).contiguous()
            per_layer.append(k_prompt)
        return torch.stack(per_layer, dim=0)

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
            queries: ``[num_layers, look_ahead, hidden_dim]``
            keys: ``[num_layers, prompt_len, num_kv_heads, head_dim]``

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
        k = keys.transpose(1, 2)
        if repeat_factor > 1:
            k = k.repeat_interleave(repeat_factor, dim=1)

        scale = 1.0 / math.sqrt(self._head_dim)
        attn = torch.matmul(q, k.transpose(-1, -2)) * scale
        return attn
