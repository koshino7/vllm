# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Speculative prefill runner for vLLM v1.

This module loads a lightweight draft model, runs look-ahead decoding to
capture query/key vectors, estimates token importance via attention scores,
and returns compressed prompt metadata to the main model runner.
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
    """Orchestrates the speculative prefill pipeline on GPU.

    Lifecycle
    ---------
    1. ``__init__`` / ``load_draft_model``: Load the draft model once.
    2. ``run``: Called by ``GPUModelRunner`` for each scheduling step that
       contains new prefill requests.  Returns
       :class:`SpecPrefillBatchMetadata` which the main model runner uses
       to build compressed prompt inputs.
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
        self._current_step: int = 0

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------
    def load_draft_model(self) -> None:
        """Load the draft model onto ``self.device``."""
        from transformers import AutoModelForCausalLM

        logger.info(
            "Loading spec-prefill draft model: %s", self.config.spec_model
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
            "Draft model loaded — layers=%d, heads=%d, kv_heads=%d, "
            "head_dim=%d",
            self._num_layers,
            self._num_heads,
            self._num_kv_heads,
            self._head_dim,
        )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def run(
        self,
        req_ids: list[str],
        prompt_token_ids_list: list[list[int]],
    ) -> SpecPrefillBatchMetadata:
        """Run speculative prefill for a batch of new prefill requests.

        Args:
            req_ids: Request identifiers.
            prompt_token_ids_list: One list of token ids per request.

        Returns:
            :class:`SpecPrefillBatchMetadata` containing compressed prompts.
        """
        assert self.draft_model is not None, "Draft model not loaded"
        batch_meta = SpecPrefillBatchMetadata()

        # Reclaim fragmented GPU memory before running the draft model.
        torch.cuda.empty_cache()

        for req_id, prompt_token_ids in zip(req_ids, prompt_token_ids_list):
            meta = self._speculate_single(req_id, prompt_token_ids)
            batch_meta.add(meta)

        # Free any leftover activations.
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

        # query_buffer[layer_idx] stores one Q tensor per *look-ahead* step.
        # Step 0 (prefill) is skipped to save memory — only the last token's
        # Q from decode steps 1..N matters.
        query_buffer: list[list[torch.Tensor]] = [
            [] for _ in range(self._num_layers)
        ]
        self._current_step = 0
        hooks = self._register_query_hooks(query_buffer)

        past_key_values = None
        cur_input_ids = input_ids
        actual_look_ahead = cfg.look_ahead_cnt
        stop_set = set(cfg.stop_token_ids)

        try:
            for step in range(cfg.look_ahead_cnt + 1):
                self._current_step = step
                outputs = self.draft_model(
                    input_ids=cur_input_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                past_key_values = outputs.past_key_values

                if step == 0:
                    # Clear prefill-step Q tensors that were captured
                    # (the hook may still fire for step 0).
                    for layer_buf in query_buffer:
                        layer_buf.clear()
                    continue

                next_token = outputs.logits[:, -1, :].argmax(dim=-1)
                if (
                    not cfg.ignore_eos
                    and next_token.item() in stop_set
                ):
                    actual_look_ahead = step
                    break
                cur_input_ids = next_token.unsqueeze(0)
        finally:
            for h in hooks:
                h.remove()

        # Extract keys from KV cache (prompt region only), then free cache.
        all_keys = self._collect_keys_from_cache(
            past_key_values, len(prompt_token_ids)
        )
        del past_key_values, outputs
        torch.cuda.empty_cache()

        all_queries = self._collect_queries(query_buffer, actual_look_ahead)

        # Move Q/K to CPU for the attention score computation to save GPU mem.
        all_queries_cpu = all_queries.float().cpu()
        all_keys_cpu = all_keys.float().cpu()
        del all_queries, all_keys
        torch.cuda.empty_cache()

        attn_scores = self._compute_attention_scores(
            all_queries_cpu, all_keys_cpu, actual_look_ahead
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
    # Query hook registration (captures Q from each layer)
    # ------------------------------------------------------------------
    def _register_query_hooks(
        self,
        query_buffer: list[list[torch.Tensor]],
    ) -> list[torch.utils.hooks.RemovableHook]:
        """Register forward hooks on each attention layer to capture Q.

        Only the **last token's** Q projection is saved (moved to CPU)
        to minimise GPU memory pressure.  Uses ``with_kwargs=True`` so
        that models calling ``self_attn`` with keyword arguments (e.g.
        Qwen3) are handled correctly.
        """
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
                # Only project the last token to save memory.
                last_hidden = hidden[:, -1:, :]
                q_last = module.q_proj(last_hidden)  # type: ignore[attr-defined]
                buf[idx].append(q_last.detach().cpu())

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
        """Stack captured queries into ``[num_layers, look_ahead, H*D]``.

        Tensors live on CPU at this point.
        """
        per_layer = []
        for layer_bufs in query_buffer:
            qs = layer_bufs[:actual_look_ahead]
            if qs:
                per_layer.append(torch.cat(qs, dim=1))  # [1, look_ahead, H*D]
            else:
                per_layer.append(layer_bufs[0] if layer_bufs else
                                 torch.zeros(1, 1, self._num_heads * self._head_dim))
        result = torch.stack(per_layer, dim=0).squeeze(1)  # [L, look_ahead, H*D]
        return result

    def _collect_keys_from_cache(
        self,
        past_key_values: tuple,
        prompt_len: int,
    ) -> torch.Tensor:
        """Extract prompt-region keys from HF KV cache, move to CPU.

        Returns: ``[num_layers, prompt_len, num_kv_heads, head_dim]``
        """
        per_layer = []
        for layer_kv in past_key_values:
            k = layer_kv[0]  # [batch, num_kv_heads, total_len, head_dim]
            # .contiguous().cpu() to avoid keeping the full cache alive.
            k_prompt = k[:, :, :prompt_len, :].squeeze(0).contiguous().cpu()
            per_layer.append(k_prompt)
        return torch.stack(per_layer, dim=0)

    # ------------------------------------------------------------------
    # Attention score computation (runs on CPU)
    # ------------------------------------------------------------------
    def _compute_attention_scores(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        actual_look_ahead: int,
    ) -> torch.Tensor:
        """Compute Q*K^T attention scores on CPU.

        Args:
            queries: ``[num_layers, look_ahead, hidden_dim]``
            keys: ``[num_layers, prompt_len, num_kv_heads, head_dim]``

        Returns:
            ``[num_layers, num_heads, look_ahead, prompt_len]``
        """
        num_layers = queries.shape[0]
        look_ahead = queries.shape[1]

        # Reshape Q: [L, look_ahead, num_heads, head_dim]
        q = queries.view(num_layers, look_ahead, self._num_heads, self._head_dim)
        # -> [L, num_heads, look_ahead, head_dim]
        q = q.transpose(1, 2)

        # Expand KV heads to match Q heads
        repeat_factor = self._num_heads // self._num_kv_heads
        # keys: [L, prompt_len, kv_heads, head_dim] -> [L, kv_heads, prompt_len, head_dim]
        k = keys.transpose(1, 2)
        if repeat_factor > 1:
            k = k.repeat_interleave(repeat_factor, dim=1)

        scale = 1.0 / math.sqrt(self._head_dim)
        attn = torch.matmul(q, k.transpose(-1, -2)) * scale
        return attn
