# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Prompt compression utilities for speculative prefill.

These functions compute token importance from attention scores
produced by the draft model and select which tokens to keep.
"""

from __future__ import annotations

import math

import torch


def compute_token_importance(
    attn_scores: torch.Tensor,
    pool_kernel_size: int | None = 13,
) -> torch.Tensor:
    """Compute per-token importance from multi-layer, multi-head attn scores.

    Args:
        attn_scores: ``[num_layers, num_heads, look_ahead_cnt, context_len]``
        pool_kernel_size: Kernel size for avg-pool smoothing.  ``None``
            disables smoothing.

    Returns:
        1-D importance tensor of shape ``[context_len]``.
    """
    original_dtype = attn_scores.dtype
    attn = torch.nn.functional.softmax(
        attn_scores, dim=-1, dtype=torch.float32
    ).to(original_dtype)

    # Flatten layers and heads -> [L*H, look_ahead_cnt, context_len]
    attn = attn.flatten(0, 1)

    if pool_kernel_size:
        attn = torch.nn.functional.avg_pool1d(
            attn,
            kernel_size=pool_kernel_size,
            padding=pool_kernel_size // 2,
            stride=1,
        )

    # Max over (layers*heads), mean over look-ahead steps
    attn = attn.max(0)[0]
    importance = attn.mean(0)
    return importance


def select_kept_indices(
    importance: torch.Tensor,
    keep_percentage: float = 0.1,
    chunk_selection: bool = True,
    chunk_size: int = 32,
) -> torch.LongTensor:
    """Select token indices to keep based on importance scores.

    Args:
        importance: 1-D tensor of shape ``[seq_len]``.
        keep_percentage: Fraction of tokens (or chunks) to keep.
        chunk_selection: If True, select whole chunks by avg importance.
        chunk_size: Chunk size when ``chunk_selection`` is True.

    Returns:
        Sorted 1-D LongTensor of kept token indices.
    """
    seq_len = len(importance)

    if chunk_selection:
        chunk_ti = torch.split(importance, chunk_size, dim=-1)
        chunk_means = torch.stack([ct.mean() for ct in chunk_ti])
        keep_chunk_cnt = max(1, math.ceil(len(chunk_means) * keep_percentage))
        _, chunk_indices = torch.topk(chunk_means, k=keep_chunk_cnt, dim=-1)
        all_indices = torch.split(
            torch.arange(seq_len, device=importance.device), chunk_size, dim=-1
        )
        kept = torch.cat([all_indices[ci.item()] for ci in chunk_indices])
    else:
        topk = max(1, math.ceil(seq_len * keep_percentage))
        _, kept = torch.topk(importance, k=topk, dim=-1)

    return torch.sort(kept)[0]
