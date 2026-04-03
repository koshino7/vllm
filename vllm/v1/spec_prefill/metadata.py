# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Data structures for speculative prefill."""

from dataclasses import dataclass, field

import torch


@dataclass
class SpecPrefillMetadata:
    """Per-request metadata produced by the speculative prefill runner.

    After the draft model estimates token importance and the prompt is
    compressed, the metadata carries the compressed tokens and their
    original position ids back to the main model runner so it can build
    the correct attention input.
    """

    req_id: str
    original_prompt_len: int
    compressed_token_ids: list[int]
    position_ids: list[int]

    @property
    def compressed_len(self) -> int:
        return len(self.compressed_token_ids)


@dataclass
class SpecPrefillBatchMetadata:
    """Aggregated metadata for all spec-prefill requests in one step."""

    per_request: dict[str, SpecPrefillMetadata] = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        return len(self.per_request) == 0

    def add(self, meta: SpecPrefillMetadata) -> None:
        self.per_request[meta.req_id] = meta


@dataclass
class QueryBufferEntry:
    """Stores captured query tensors across layers for one forward pass."""

    queries: list[list[torch.Tensor]] = field(default_factory=list)

    def prepare(self, num_layers: int) -> None:
        self.queries = [[] for _ in range(num_layers)]

    def clear(self) -> None:
        for layer_buf in self.queries:
            layer_buf.clear()
