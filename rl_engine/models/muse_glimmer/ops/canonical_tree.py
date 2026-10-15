# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Canonical fixed fp32 combine tree of the Muse-Glimmer-30B strict contract (O3).

Row-parallel GEMMs split K into ``CANONICAL_K_CHUNKS = 8`` consecutive chunks; each chunk is
reduced to an fp32 partial and the partials are summed in ONE fixed binary tree::

    ((0 + 1) + (2 + 3)) + ((4 + 5) + (6 + 7))

At TP ``t`` a rank owns ``8 // t`` consecutive chunks (``rank_chunk_ids``) and emits the fp32
value of its sub-tree node; the ranks' partials are then combined with ``combine_tree`` (the same
tree, one level up).  Because every node of the tree is an exact IEEE fp32 add, the TP-``t``
result is bitwise equal to the TP-1 result as long as the per-chunk partials are bitwise equal.

Cast points: none.  Inputs must already be fp32; the caller rounds the root to bf16 (or keeps it
in fp32 for the lm_head logits) exactly once.  This module is intentionally tiny and pure torch so
that collectives, attention and lm_head agents can import it without pulling in Triton.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

CANONICAL_K_CHUNKS = 8
TREE_ORDER = "((0+1)+(2+3))+((4+5)+(6+7))"
_SUPPORTED_COUNTS = (1, 2, 4, 8)


def combine_tree(partials: Sequence[torch.Tensor]) -> torch.Tensor:
    """Sum 1, 2, 4 or 8 fp32 partials in the canonical fixed tree.

    * 8 partials: ``((p0+p1)+(p2+p3))+((p4+p5)+(p6+p7))``  (the full K tree, TP1 / TP8 ranks)
    * 4 partials: ``(p0+p1)+(p2+p3)``                       (TP4 ranks, or one half of the tree)
    * 2 partials: ``p0+p1``                                  (TP2 ranks, or one quarter)
    * 1 partial : the partial itself

    Each ``+`` is one torch fp32 elementwise add (exact IEEE round-to-nearest-even), so the
    result is bitwise identical to the same adds performed in registers inside a kernel.
    The input order is the chunk/rank order; the function never reorders or re-associates.
    """

    parts = list(partials)
    count = len(parts)
    if count not in _SUPPORTED_COUNTS:
        raise ValueError(f"combine_tree expects 1, 2, 4 or 8 partials, got {count}")
    first = parts[0]
    for index, part in enumerate(parts):
        if part.dtype != torch.float32:
            raise TypeError(f"combine_tree partial {index} must be float32, got {part.dtype}")
        if part.shape != first.shape:
            raise ValueError(
                f"combine_tree partial {index} has shape {tuple(part.shape)}, "
                f"expected {tuple(first.shape)}"
            )
    if count == 1:
        return first
    if count == 2:
        return parts[0] + parts[1]
    if count == 4:
        return (parts[0] + parts[1]) + (parts[2] + parts[3])
    lower = (parts[0] + parts[1]) + (parts[2] + parts[3])
    upper = (parts[4] + parts[5]) + (parts[6] + parts[7])
    return lower + upper


def rank_chunk_ids(tp_size: int, rank: int) -> tuple[int, ...]:
    """Canonical chunk ids owned by ``rank`` at tensor-parallel degree ``tp_size`` (1, 2, 4, 8)."""

    if tp_size not in _SUPPORTED_COUNTS:
        raise ValueError(f"tp_size must be one of {_SUPPORTED_COUNTS}, got {tp_size}")
    if not 0 <= rank < tp_size:
        raise ValueError(f"rank {rank} out of range for tp_size {tp_size}")
    per_rank = CANONICAL_K_CHUNKS // tp_size
    return tuple(range(rank * per_rank, (rank + 1) * per_rank))


__all__ = ["CANONICAL_K_CHUNKS", "TREE_ORDER", "combine_tree", "rank_chunk_ids"]
