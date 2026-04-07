# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any, Literal

from pydantic import Field

from vllm.config.utils import config
from vllm.logger import init_logger

logger = init_logger(__name__)


@config
class SpecPrefillConfig:
    """Configuration for speculative prefill.

    Speculative prefill uses a smaller draft model to estimate token importance
    via look-ahead attention, then compresses the prompt by keeping only the
    most important tokens (with their original position ids) before sending
    them to the main model for prefill.
    """

    spec_model: str = ""
    """HuggingFace model name or path for the draft model used to estimate
    token importance. Must share the same tokenizer and vocabulary as the
    main model."""

    look_ahead_cnt: int = Field(default=8, ge=1)
    """Number of look-ahead decoding steps the draft model performs after
    prefill. The queries from these steps are used to compute attention
    scores against the prompt keys."""

    keep_strategy: Literal["percentage"] = "percentage"
    """Strategy for deciding which tokens to keep. Currently only
    ``percentage`` is supported."""

    keep_percentage: float = Field(default=0.1, gt=0.0, le=1.0)
    """Fraction of prompt tokens to retain when ``keep_strategy`` is
    ``percentage``."""

    chunk_selection: bool = True
    """If True, divide the prompt into fixed-size chunks and select whole
    chunks by their average importance score rather than individual tokens."""

    chunk_size: int = Field(default=32, ge=1)
    """Chunk size used when ``chunk_selection`` is True."""

    pool_kernel_size: int | None = 13
    """Kernel size for avg-pool smoothing of token importance scores.
    Set to None to disable smoothing."""

    ignore_eos: bool = False
    """Ignore EOS tokens during look-ahead. Useful for benchmarking only."""

    stop_token_ids: list[int] = Field(
        default_factory=lambda: [128001, 128008, 128009]
    )
    """Token ids treated as stop tokens during the draft model's look-ahead.
    Defaults to Llama-3 family EOS/EOT ids."""

    draft_prefill_chunk_size: int = Field(default=4096, ge=128)
    """Maximum number of tokens per chunk when feeding the prompt to the
    draft model. Keeps MLP intermediate activation memory bounded."""

    def compute_hash(self) -> str:
        from vllm.utils.hashing import safe_hash

        factors: list[Any] = [
            self.spec_model,
            self.look_ahead_cnt,
            self.keep_strategy,
            self.keep_percentage,
            self.chunk_selection,
            self.chunk_size,
            self.pool_kernel_size,
        ]
        return safe_hash(
            str(factors).encode(), usedforsecurity=False
        ).hexdigest()[:10]
