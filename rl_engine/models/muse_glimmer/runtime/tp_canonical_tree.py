# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""``tp_canonical_tree`` (RFC #501, WS2): row-parallel GEMM on the canonical 8-leaf tree.

Contract (O3): a row-parallel projection ``y = x @ W^T`` with ``W (N, K)`` splits ``K`` into
8 canonical chunks (``o_proj``: 512, ``down_proj``: 2496). Chunk ``c`` is a det_gemm with fp32
accumulation and an fp32 result ``y_c`` (never rounded to bf16). The 8 fp32 partials are summed
in the FIXED tree ``((0+1)+(2+3))+((4+5)+(6+7))`` in fp32 and rounded to bf16 exactly once.

* TP1 computes the 8 chunk partials locally and folds the whole tree.
* TP t: rank r owns chunks ``[8r/t, 8(r+1)/t)``, folds them locally in tree order (its node is
  a sub-tree root of the same 8-leaf tree), then every rank's fp32 sub-tree root travels by a
  deterministic, rank-ordered **all-gather** (transport only, no arithmetic) and every rank folds
  the ``t`` roots with the same tree. The expression evaluated is identical at every t, so the
  bf16 result is byte-equal to TP1.

Casts: inputs bf16; GEMM accumulation fp32; chunk partials fp32; tree adds fp32; one bf16
rounding at the end. Reduction order: within a chunk the det_gemm kernel's fixed K walk; across
chunks and ranks the tree above.

Sibling-op dependency (fixed interface, may not be present yet):
``rl_engine.models.muse_glimmer.ops.det_gemm.fast_det_gemm_partial(a, w, chunk_ids)`` emits the
fp32 sub-tree partial over consecutive canonical chunks, and
``rl_engine.models.muse_glimmer.ops.canonical_tree.combine_tree(partials)`` folds partials. When
they are absent this module uses labelled fallbacks: the repo's fixed-order fp32 Triton det_gemm
kernel (``rl_engine/backends/shared/triton/gemm/det_gemm.py::_det_gemm_fp32_kernel``, one
program per output tile, BLOCK_K=32 ascending, no split-K, ``allow_tf32=False``) per chunk, and a
private pairwise fold. Fallback partials are genuine fp32 (not bf16-rounded).
"""

from __future__ import annotations

import time
from typing import Any, Callable, Sequence

import torch
import torch.distributed as dist

from rl_engine.models.muse_glimmer.runtime.tp_layout import (
    CANONICAL_TREE_ORDER,
    TPLayout,
    TPLayoutError,
)

Tensor = torch.Tensor

# --------------------------------------------------------------------------- #
# Sibling-op resolution
# --------------------------------------------------------------------------- #

try:  # fixed interface owned by the det_gemm agent
    from rl_engine.models.muse_glimmer.ops.det_gemm import (  # type: ignore[import-not-found]
        fast_det_gemm_partial as _sibling_partial,
    )

    GEMM_BACKEND = "muse_glimmer.ops.det_gemm.fast_det_gemm_partial"
except ImportError:  # pragma: no cover - depends on sibling branch
    _sibling_partial = None
    GEMM_BACKEND = "fallback:rl_engine.backends.shared.triton.gemm.det_gemm._det_gemm_fp32_kernel"

try:  # fixed interface owned by the canonical_tree agent
    from rl_engine.models.muse_glimmer.ops.canonical_tree import (  # type: ignore[import-not-found]
        combine_tree as _sibling_combine_tree,
    )

    TREE_BACKEND = "muse_glimmer.ops.canonical_tree.combine_tree"
except ImportError:  # pragma: no cover - depends on sibling branch
    _sibling_combine_tree = None
    TREE_BACKEND = "fallback:private_pairwise_fold"


def _fallback_combine_tree(partials: Sequence[Tensor] | Tensor) -> Tensor:
    """Balanced pairwise fold ``((0+1)+(2+3))+((4+5)+(6+7))`` over a power-of-two leaf count.

    Accepts a list of equally shaped tensors or a stacked tensor whose dim 0 indexes leaves.
    Each level adds leaf ``2i`` to leaf ``2i+1`` (left operand first) in the leaves' dtype.
    """

    if isinstance(partials, Tensor):
        stacked = partials
    else:
        if len(partials) == 0:
            raise ValueError("combine_tree requires at least one partial")
        stacked = torch.stack(tuple(partials), dim=0)
    n = stacked.size(0)
    if n & (n - 1):
        raise ValueError(f"combine_tree requires a power-of-two leaf count, got {n}")
    while stacked.size(0) > 1:
        stacked = stacked[0::2] + stacked[1::2]
    return stacked[0]


def combine_tree(partials: Sequence[Tensor] | Tensor) -> Tensor:
    """Fold partials in the canonical tree (sibling op when present, else the private fold)."""

    if _sibling_combine_tree is not None:
        leaves = list(partials.unbind(0)) if isinstance(partials, Tensor) else list(partials)
        return _sibling_combine_tree(leaves)
    return _fallback_combine_tree(partials)


def _fallback_chunk_partial_fp32(a_chunk: Tensor, w_chunk: Tensor) -> Tensor:
    """fp32 ``a_chunk @ w_chunk^T`` with the repo's fixed-order fp32 det_gemm kernel.

    The kernel is launched directly on ``w_chunk``'s transposed *view* (strides only change
    addressing, not the per-tile arithmetic), so no transposed copy of the weight is made.
    """

    from rl_engine.backends.shared.triton.gemm import det_gemm as _dg

    if a_chunk.dtype != torch.bfloat16 or w_chunk.dtype != torch.bfloat16:
        raise TypeError("row-parallel chunk partial requires bf16 operands")
    if a_chunk.dim() != 2 or w_chunk.dim() != 2 or a_chunk.size(1) != w_chunk.size(1):
        raise ValueError(
            f"chunk operands must be (M, Kc) and (N, Kc); got {tuple(a_chunk.shape)} and "
            f"{tuple(w_chunk.shape)}"
        )
    a = a_chunk.contiguous()
    w = w_chunk.contiguous()
    m_size, k_size = a.shape
    n_size = w.size(0)
    out = torch.empty((m_size, n_size), dtype=torch.float32, device=a.device)
    import triton

    grid = (triton.cdiv(m_size, _dg._BLOCK_M), triton.cdiv(n_size, _dg._BLOCK_N))
    _dg._det_gemm_fp32_kernel[grid](
        a,
        w,
        out,
        m_size,
        n_size,
        k_size,
        a.stride(0),
        a.stride(1),
        w.stride(1),  # b is w^T: moving along K walks w's second dim
        w.stride(0),  # moving along N walks w's rows
        out.stride(0),
        out.stride(1),
        BLOCK_M=_dg._BLOCK_M,
        BLOCK_N=_dg._BLOCK_N,
        BLOCK_K=_dg._BLOCK_K,
        PROMOTE_INPUTS=True,
    )
    return out


def subtree_partial_fp32(a_local: Tensor, w_local: Tensor, chunk_ids: Sequence[int]) -> Tensor:
    """fp32 sub-tree root over the consecutive canonical chunks ``chunk_ids`` of ``K_local``.

    ``a_local (M, K_local)`` and ``w_local (N, K_local)`` are the rank-local slices; the chunk
    width is ``K_local / len(chunk_ids)``. The chunk count must be a power of two and the ids
    consecutive and aligned (``first % count == 0``) so the fold is a node of the 8-leaf tree.
    """

    ids = [int(c) for c in chunk_ids]
    count = len(ids)
    if count == 0 or count & (count - 1):
        raise ValueError(f"chunk_ids must hold a power-of-two count, got {count}")
    if ids != list(range(ids[0], ids[0] + count)) or ids[0] % count:
        raise ValueError(f"chunk_ids {ids} are not a consecutive aligned sub-tree of the tree")
    if a_local.size(1) != w_local.size(1) or a_local.size(1) % count:
        raise ValueError(
            f"K_local {a_local.size(1)} / {w_local.size(1)} does not split into {count} chunks"
        )
    if _sibling_partial is not None:
        return _sibling_partial(a_local, w_local, ids)
    width = a_local.size(1) // count
    partials = [
        _fallback_chunk_partial_fp32(
            a_local.narrow(1, i * width, width), w_local.narrow(1, i * width, width)
        )
        for i in range(count)
    ]
    return partials[0] if count == 1 else combine_tree(partials)


# --------------------------------------------------------------------------- #
# Collectives
# --------------------------------------------------------------------------- #


def make_collective(
    group: dist.ProcessGroup | None = None,
    device: torch.device | int | None = None,
    *,
    max_size_bytes: int = 64 * 1024 * 1024,
) -> Any:
    """Deterministic collective for this process: CUDA-IPC when ``rl_engine._C`` is built,
    else the torch.distributed transport (NCCL all-gather moves bits; no arithmetic)."""

    from rl_engine.distributed import (
        TorchDistributedDeterministicCollective,
        create_deterministic_collective,
    )

    try:
        return create_deterministic_collective(group, device, max_size_bytes=max_size_bytes)
    except RuntimeError as exc:
        if "extension" not in str(exc):
            raise
        return TorchDistributedDeterministicCollective(
            group, device, max_size_bytes=max_size_bytes
        )


def _check_collective(collective: Any, layout: TPLayout, rank: int | None) -> int:
    if layout.tp == 1:
        return 0
    if collective is None:
        raise TPLayoutError(f"tp={layout.tp} requires a deterministic collective")
    world = int(getattr(collective, "world_size"))
    if world != layout.tp:
        raise TPLayoutError(f"collective world_size {world} != layout tp {layout.tp}")
    coll_rank = int(getattr(collective, "rank"))
    if rank is not None and int(rank) != coll_rank:
        raise TPLayoutError(f"rank {rank} != collective rank {coll_rank}")
    return coll_rank


# --------------------------------------------------------------------------- #
# Public op
# --------------------------------------------------------------------------- #


def row_parallel_tree_gemm(
    a_local: Tensor,
    w_local: Tensor,
    layout: TPLayout,
    collective: Any = None,
    *,
    rank: int | None = None,
    chunk_width: int | None = None,
) -> Tensor:
    """Row-parallel ``bf16 (M, N)`` output of ``x @ W^T`` on the canonical tree.

    ``a_local (M, K_local)`` and ``w_local (N, K_local)`` are this rank's K-slices (the full
    tensors at TP1). ``chunk_width`` defaults to ``K_local / chunks_per_rank`` and must be one
    of the declared row-parallel shards (512 for ``o_proj``, 2496 for ``down_proj``) unless the
    caller passes it explicitly (small-shape tests; out of contract).

    Reduction order: det_gemm per chunk (fp32) -> local tree over the owned chunks (fp32) ->
    deterministic rank-ordered all-gather of the fp32 roots -> tree over ranks (fp32) -> bf16.
    """

    if a_local.dtype != torch.bfloat16 or w_local.dtype != torch.bfloat16:
        raise TypeError("row_parallel_tree_gemm requires bf16 activations and weights")
    if a_local.dim() != 2 or w_local.dim() != 2 or a_local.size(1) != w_local.size(1):
        raise ValueError(
            f"expected a_local (M, K_local) and w_local (N, K_local); got "
            f"{tuple(a_local.shape)} and {tuple(w_local.shape)}"
        )
    rank = _check_collective(collective, layout, rank)
    k_local = a_local.size(1)
    count = layout.chunks_per_rank
    if k_local % count:
        raise ValueError(f"K_local {k_local} does not split into {count} canonical chunks")
    width = k_local // count
    if chunk_width is None:
        if width not in layout.row_parallel_chunk_widths():
            raise TPLayoutError(
                f"chunk width {width} (K_local {k_local} / {count}) is not a declared "
                f"row-parallel shard {layout.row_parallel_chunk_widths()}"
            )
    elif int(chunk_width) != width:
        raise ValueError(f"chunk_width {chunk_width} != K_local / chunks_per_rank = {width}")

    local_root = subtree_partial_fp32(a_local, w_local, layout.canonical_chunks(rank))
    if layout.tp == 1:
        return local_root.to(torch.bfloat16)
    gathered = collective.all_gather(local_root.unsqueeze(0).contiguous())  # (tp, M, N) fp32
    return combine_tree(gathered).to(torch.bfloat16)


def row_parallel_tree_gemm_tp1(a: Tensor, w: Tensor, layout: TPLayout | None = None) -> Tensor:
    """The TP1 reference: all 8 chunks local, same tree, bf16 once (no process group needed)."""

    base = layout if layout is not None else TPLayout(tp=1)
    ref_layout = TPLayout(
        tp=1,
        canonical_tp=base.canonical_tp,
        num_q_heads=base.num_q_heads,
        num_kv_heads=base.num_kv_heads,
        head_dim=base.head_dim,
        o_proj_chunk=base.o_proj_chunk,
        down_chunk=base.down_chunk,
        vocab_size=base.vocab_size,
        padded_vocab_size=base.padded_vocab_size,
    )
    width = a.size(1) // ref_layout.canonical_tp
    explicit = None if width in ref_layout.row_parallel_chunk_widths() else width
    return row_parallel_tree_gemm(a, w, ref_layout, None, chunk_width=explicit)


def nccl_all_reduce_control(local_root: Tensor, group: dist.ProcessGroup | None = None) -> Tensor:
    """Negative control: NCCL's own SUM all-reduce of the fp32 sub-tree roots, then bf16.

    NCCL picks a ring/tree schedule whose addition order is not the canonical tree, so for
    tp >= 4 the result is in general NOT byte-equal to TP1 (for tp == 2 ``a + b == b + a`` in
    IEEE arithmetic, so it accidentally is). Never use this in the strict path.
    """

    out = local_root.clone()
    dist.all_reduce(out, op=dist.ReduceOp.SUM, group=group)
    return out.to(torch.bfloat16)


def time_rank_reduction(
    local_root: Tensor,
    collective: Any,
    group: dist.ProcessGroup | None = None,
    *,
    reps: int = 20,
) -> dict[str, float]:
    """Milliseconds per call for: deterministic all-gather + tree, NCCL all-reduce (control),
    and the collective's own balanced-tree all-reduce. Includes a device sync per timing."""

    def timed(fn: Callable[[], Tensor]) -> float:
        fn()
        torch.cuda.synchronize()
        if group is not None or dist.is_initialized():
            dist.barrier(group=group)
        start = time.perf_counter()
        for _ in range(reps):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - start) / reps * 1e3

    staged = local_root.unsqueeze(0).contiguous()
    results = {
        "all_gather_plus_tree_ms": timed(lambda: combine_tree(collective.all_gather(staged))),
        "nccl_all_reduce_ms": timed(lambda: nccl_all_reduce_control(local_root, group)),
    }
    if hasattr(collective, "all_reduce"):
        results["collective_all_reduce_ms"] = timed(lambda: collective.all_reduce(local_root))
    return results


def trace(layout: TPLayout, collective: Any = None) -> dict[str, Any]:
    """Provenance per contract R4."""

    return {
        "op": "tp_canonical_tree.row_parallel_tree_gemm",
        "dtype_in": "bfloat16",
        "dtype_acc": "float32",
        "dtype_partial": "float32",
        "dtype_out": "bfloat16",
        "reduction_order": (
            "det_gemm fixed K walk per chunk -> local tree over owned chunks -> "
            f"rank-ordered all-gather -> tree over ranks; tree {CANONICAL_TREE_ORDER}"
        ),
        "tile_config": {
            "row_parallel_shards": {"o_proj": layout.o_proj_chunk, "down_proj": layout.down_chunk},
            "canonical_tp": layout.canonical_tp,
            "tp": layout.tp,
            "chunks_per_rank": layout.chunks_per_rank,
        },
        "gemm_backend": GEMM_BACKEND,
        "tree_backend": TREE_BACKEND,
        "collective_backend": (
            None if collective is None else getattr(collective, "backend_id", None)
        ),
        "chunk_to_rank": {c: layout.chunk_owner(c) for c in range(layout.canonical_tp)},
    }


__all__ = [
    "GEMM_BACKEND",
    "TREE_BACKEND",
    "combine_tree",
    "make_collective",
    "nccl_all_reduce_control",
    "row_parallel_tree_gemm",
    "row_parallel_tree_gemm_tp1",
    "subtree_partial_fp32",
    "time_rank_reduction",
    "trace",
]
