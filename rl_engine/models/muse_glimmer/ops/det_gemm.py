# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""``fast_det_gemm``: the strict deterministic GEMM of the Muse-Glimmer-30B POC (O3, RFC #501 row
``dense_gemm_qualification``).

Semantics
---------
``fast_det_gemm(a, w)`` computes ``y = a @ w.T`` for an ``nn.Linear`` weight ``w (N, K)``
(``w_kn=True`` takes a ``(K, N)`` operand and computes ``a @ w``).  Inputs are bf16; the
reduction over K is performed by ``det_gemm_tree_kernel`` (fp32 MMA accumulation, K split into
the 8 canonical chunks, chunk partials combined in registers in the fixed fp32 tree
``((0+1)+(2+3))+((4+5)+(6+7))``, one rounding to ``out_dtype`` at the end).  ``out_dtype`` is
bf16 for every layer GEMM and fp32 for the lm_head logits (O7).

Reduction-order / cast summary (details in the kernel module docstring):
  forward   y  = round_out( tree_c( sum_{k in chunk c} a[m,k] * w[n,k] ) ),  chunks over K;
            K % 256 != 0 raises unless the caller opts into the ragged rule (``ragged_k=True``,
            only for reductions over the token axis, e.g. a weight gradient written as a GEMM)
  backward  dY -> bf16 once (declared cast, also for the fp32-output lm_head path),
            dA = tree over N   of dY_bf16[m,n] * w[n,k]            -> bf16  (K-rule on N;
                 N % 256 != 0 uses the masked rule below with N_pad = roundup(N, 256), which
                 keeps dA batch-invariant; unreachable for the model shapes)
            dW = tree over M   of dY_bf16[m,n] * a[m,k]            -> bf16  (ragged token rule:
                 M_pad = roundup(M, 256), 8 chunks of M_pad/8 token rows, rows >= M masked to
                 zero -- repeatable; batch invariance of dW is not a contract requirement)
Every reduction length of the model forward / dA (K in {6656, 4096, 19968}, N in {4096, 256,
19968, 6656, 202752, 25344}) is a multiple of 256.  BLOCK_K is the largest of 64/32/16 that
divides the chunk length with at least 2 loop iterations per chunk (64 for every model forward
shape; 16 for 32-wide chunks, 32 for 64-wide chunks): a 1-iteration chunk lets Triton fold the
tree adds into the MMA chain (finding det_gemm:1), with >= 2 iterations BLOCK_K does not change
the bits.  The compiled kernel is checked to contain the 7 tree adds (fail closed).

Tensor-parallel entry points (autograd-capable, bitwise equal to TP1)
----------------------------------------------------------------------
* ``fast_det_gemm_partial(a_local, w_local, chunk_ids)``: fp32 sub-tree over the consecutive
  canonical K chunks a rank owns (row-parallel forward partial, no autograd).
* ``row_parallel_fast_det_gemm(a_local, w_local, tp_size=t, y_reduce=...)``: row-parallel
  (o_proj, down_proj) shard with autograd.  Forward: fp32 sub-tree partial over the rank's
  ``8/t`` K chunks -> ``y_reduce`` (rank-ordered all-gather + canonical tree) -> one rounding.
  Backward: dA_local and dW_local need no cross-rank reduction (dY is replicated).
* ``column_parallel_fast_det_gemm(a, w_local, tp_size=t, da_reduce=...)``: column-parallel
  (q, k, v, attn_gate, mlp_gate, up) shard with autograd.  Forward is local.  Backward: the
  input gradient reduces over N, which is split across ranks, so every rank emits the fp32
  sub-tree partial of dA over its ``8/t`` canonical N chunks and ``da_reduce`` combines the ranks'
  partials in the canonical tree BEFORE the single bf16 rounding (finding det_gemm:2).
``tree_reduce_over_ranks(collective)`` builds such a reduce from a rank-ordered all-gather and
``runtime.tp_canonical_tree.combine_tree``.  ``tests/models/muse_glimmer/test_det_gemm.py``
simulates TP 2/4/8 on one GPU and checks forward, dA and dW against TP1 with ``torch.equal``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict
from typing import Any

import torch

from rl_engine.models.muse_glimmer.ops.canonical_tree import (
    CANONICAL_K_CHUNKS,
    TREE_ORDER,
    combine_tree,
    rank_chunk_ids,
)
from rl_engine.models.muse_glimmer.triton import det_gemm_tree as _kernels
from rl_engine.ops.autograd.backward_runtime import record_backward

OP_NAME = "muse_glimmer.fast_det_gemm"
KERNEL_ID = "rl_engine.models.muse_glimmer.triton.det_gemm_tree.det_gemm_tree_kernel"
_SUPPORTED_CHUNK_COUNTS = (1, 2, 4, 8)
_SUPPORTED_TP = (1, 2, 4, 8)

ReduceFn = Callable[[torch.Tensor], torch.Tensor]

BLOCK_K_RULE = (
    "BLOCK_K = largest of (64, 32, 16) with chunk_len % BLOCK_K == 0 and chunk_len / BLOCK_K "
    ">= 2 (64 for every model forward shape; 16 for 32-wide chunks, 32 for 64-wide chunks); "
    "a 1-iteration chunk would let Triton fold the tree adds into the MMA chain, with >= 2 "
    "iterations BLOCK_K does not change the bits"
)
K_RULE = (
    "K % 256 == 0 required (8 chunks x 32); otherwise ValueError, unless the caller passes "
    "ragged_k=True (token-axis reductions only): K_pad = roundup(K, 256), 8 chunks of K_pad/8, "
    "columns >= K masked to zero"
)
DA_CHUNK_RULE = (
    "reduction over N with the canonical rule (8 chunks of N/8); N % 256 != 0 (never for the "
    "model) uses N_pad = roundup(N, 256) with masked columns (batch-invariant); column-parallel "
    "TP: each rank reduces its 8/t chunks to an fp32 sub-tree partial, partials combined in the "
    "canonical tree, one rounding"
)
DW_CHUNK_RULE = (
    "reduction over the M tokens: M_pad = roundup(M, 256), 8 chunks of M_pad/8 rows, rows >= M "
    "masked to zero (repeatable; not batch-invariant by design)"
)
GRAD_CAST = (
    "dY -> bf16 once (round-to-nearest-even) before the backward GEMMs, for bf16 and fp32 "
    "outputs alike; dA and dW are rounded once to the operand dtype"
)


def _as_kn(w: torch.Tensor, w_kn: bool) -> torch.Tensor:
    """View ``w`` as the ``(K, N)`` operand the kernel addresses (no copy)."""

    return w if w_kn else w.t()


def _check(a: torch.Tensor, w: torch.Tensor, *, w_kn: bool, out_dtype: torch.dtype) -> int:
    if a.dim() < 2:
        raise ValueError(f"fast_det_gemm expects a with at least 2 dims, got {tuple(a.shape)}")
    if w.dim() != 2:
        raise ValueError(f"fast_det_gemm expects a 2-D weight, got {tuple(w.shape)}")
    if a.dtype != torch.bfloat16 or w.dtype != torch.bfloat16:
        raise TypeError(f"fast_det_gemm requires bf16 operands, got {a.dtype} and {w.dtype}")
    if not a.is_cuda or not w.is_cuda or a.device != w.device:
        raise RuntimeError("fast_det_gemm operands must share one CUDA device")
    if out_dtype not in (torch.bfloat16, torch.float32):
        raise TypeError(f"fast_det_gemm out_dtype must be bf16 or fp32, got {out_dtype}")
    k_size = a.size(-1)
    w_k = w.size(0) if w_kn else w.size(1)
    if w_k != k_size:
        layout = "(K, N)" if w_kn else "(N, K)"
        raise ValueError(
            f"fast_det_gemm K mismatch: a[..., {k_size}] vs weight {tuple(w.shape)} in {layout}"
        )
    return k_size


def _flatten(a: torch.Tensor, k_size: int) -> torch.Tensor:
    # A 2-D (possibly transposed) view is kept as is: the kernel reads it through strides.
    return a if a.dim() == 2 else a.reshape(-1, k_size)


def _record() -> None:
    record_backward(
        "det_gemm",
        kernel_id=KERNEL_ID,
        impl="muse_glimmer_fast_det_gemm",
        family="triton",
    )


def _grad_bf16(grad_y: torch.Tensor) -> torch.Tensor:
    # Declared cast (GRAD_CAST): the upstream gradient enters the GEMMs as bf16.
    return grad_y.contiguous().to(torch.bfloat16)


def _dw(dy: torch.Tensor, a2d: torch.Tensor, w: torch.Tensor, w_kn: bool) -> torch.Tensor:
    """Weight gradient with the ragged token rule (reduction over the M rows)."""

    if w_kn:
        # dW[k, n] = sum_m A[m, k] * dY[m, n]
        return _kernels.tree_gemm(a2d.t(), dy, out_dtype=w.dtype, allow_pad=True)
    # dW[n, k] = sum_m dY[m, n] * A[m, k]
    return _kernels.tree_gemm(dy.t(), a2d, out_dtype=w.dtype, allow_pad=True)


class _FastDetGemmFn(torch.autograd.Function):
    """Deterministic forward and backward; see the module docstring for the reduction rules."""

    @staticmethod
    def forward(ctx, a2d, w, out_dtype, w_kn, k_chunks, ragged_k):
        ctx.save_for_backward(a2d, w)
        ctx.w_kn = bool(w_kn)
        return _kernels.tree_gemm(
            a2d, _as_kn(w, w_kn), out_dtype=out_dtype, num_chunks=k_chunks, allow_pad=ragged_k
        )

    @staticmethod
    def backward(ctx, grad_y):
        a2d, w = ctx.saved_tensors
        dy = _grad_bf16(grad_y)
        b_kn = _as_kn(w, ctx.w_kn)  # (K, N)
        da = dw = None
        if ctx.needs_input_grad[0]:
            # dA[m, k] = sum_n dY[m, n] * B[k, n]; reduction over N with the canonical chunk rule.
            da = _kernels.tree_gemm(dy, b_kn.t(), out_dtype=a2d.dtype, allow_pad=True)
        if ctx.needs_input_grad[1]:
            dw = _dw(dy, a2d, w, ctx.w_kn)
        _record()
        return da, dw, None, None, None, None


def fast_det_gemm(
    a: torch.Tensor,
    w: torch.Tensor,
    *,
    out_dtype: torch.dtype = torch.bfloat16,
    k_chunks: int = CANONICAL_K_CHUNKS,
    w_kn: bool = False,
    ragged_k: bool = False,
) -> torch.Tensor:
    """Strict GEMM ``a @ w.T`` (or ``a @ w`` with ``w_kn=True``) with deterministic autograd.

    ``a`` is ``(..., K)`` bf16 (leading dims are flattened to M; a 2-D strided view is read
    through its strides, no copy), ``w`` is ``(N, K)`` bf16 (an ``nn.Linear`` weight, any
    strides) or ``(K, N)`` when ``w_kn=True``.  ``out_dtype`` is bf16 (default) or fp32 (lm_head
    logits).  ``k_chunks`` is 8 under the contract; 1, 2 and 4 are accepted for sub-tree
    experiments only (the backward reductions over N and M always use 8 chunks).
    ``ragged_k=True`` opts into the declared ragged rule for K % 256 != 0 (``K_RULE``); use it
    only for reductions over the token axis.  The result is ``(..., N)``.
    """

    if k_chunks not in _SUPPORTED_CHUNK_COUNTS:
        raise ValueError(f"k_chunks must be one of {_SUPPORTED_CHUNK_COUNTS}, got {k_chunks}")
    k_size = _check(a, w, w_kn=w_kn, out_dtype=out_dtype)
    n_size = w.size(1) if w_kn else w.size(0)
    lead = a.shape[:-1]
    y = _FastDetGemmFn.apply(_flatten(a, k_size), w, out_dtype, w_kn, k_chunks, bool(ragged_k))
    return y.reshape(*lead, n_size)


def forward(a: torch.Tensor, w: torch.Tensor, *, w_kn: bool = False) -> torch.Tensor:
    """Contract R4 ``forward``: bf16 in, bf16 out."""

    return fast_det_gemm(a, w, out_dtype=torch.bfloat16, w_kn=w_kn)


def forward_fp32(a: torch.Tensor, w: torch.Tensor, *, w_kn: bool = False) -> torch.Tensor:
    """Contract R4 golden: the GEMM evaluated in fp64 (TF32 cannot apply), returned as fp32.

    The contract (O3, header) accepts an FP64 reference for GEMM correctness; a hand-written
    fixed-order torch loop is not feasible at 4096 x 19968 x 6656.  This function is the
    tolerance reference only, never a bitwise target, and is not part of the strict path.
    """

    k_size = _check(a, w, w_kn=w_kn, out_dtype=torch.float32)
    lead = a.shape[:-1]
    a64 = a.reshape(-1, k_size).double()
    w64 = _as_kn(w, w_kn).double()
    return (a64 @ w64).float().reshape(*lead, -1)


def _check_chunk_ids(chunk_ids: Sequence[int]) -> tuple[int, ...]:
    ids = tuple(int(c) for c in chunk_ids)
    count = len(ids)
    if count not in _SUPPORTED_CHUNK_COUNTS:
        raise ValueError(f"chunk_ids must hold 1, 2, 4 or 8 ids, got {count}")
    if ids != tuple(range(ids[0], ids[0] + count)) or ids[0] % count or ids[-1] >= 8 or ids[0] < 0:
        raise ValueError(f"chunk_ids {ids} is not an aligned node of the 8-chunk tree")
    return ids


@torch.no_grad()
def fast_det_gemm_partial(
    a_local: torch.Tensor,
    w_local: torch.Tensor,
    chunk_ids: Sequence[int],
    *,
    w_kn: bool = False,
) -> torch.Tensor:
    """fp32 sub-tree partial over the consecutive canonical chunks ``chunk_ids`` a rank owns.

    ``a_local`` is ``(..., K_local)`` and ``w_local`` is ``(N, K_local)`` (or ``(K_local, N)``
    with ``w_kn``), where ``K_local = len(chunk_ids) * K_full / 8`` is the rank's K slice, i.e.
    ``a_local = a_full[..., ids[0] * chunk_len : (ids[-1] + 1) * chunk_len]``.  ``chunk_ids``
    must be 1, 2, 4 or 8 consecutive ids aligned to their count (a node of the tree), as given by
    ``rank_chunk_ids(tp_size, rank)``.  ``combine_tree`` over the ranks' partials, rounded once to
    bf16, equals the TP1 ``fast_det_gemm`` output bitwise.  No autograd (see
    :func:`row_parallel_fast_det_gemm` for the differentiable row-parallel shard).
    """

    count = len(_check_chunk_ids(chunk_ids))
    k_local = _check(a_local, w_local, w_kn=w_kn, out_dtype=torch.float32)
    if k_local % (count * _kernels.MIN_BLOCK_K):
        raise ValueError(
            f"local K {k_local} must be a multiple of {count * _kernels.MIN_BLOCK_K} for "
            f"{count} chunks"
        )
    lead = a_local.shape[:-1]
    out = _kernels.tree_gemm(
        _flatten(a_local, k_local),
        _as_kn(w_local, w_kn),
        out_dtype=torch.float32,
        num_chunks=count,
    )
    return out.reshape(*lead, -1)


# ----------------------------------------------------------------------------------------------
# Tensor-parallel shards with autograd (finding det_gemm:2)
# ----------------------------------------------------------------------------------------------


def _tp_chunks(tp_size: int) -> int:
    if tp_size not in _SUPPORTED_TP:
        raise ValueError(f"tp_size must be one of {_SUPPORTED_TP}, got {tp_size}")
    return CANONICAL_K_CHUNKS // tp_size


def _reduce_ranks(partial: torch.Tensor, tp_size: int, reduce: ReduceFn | None, what: str):
    """Apply the cross-rank canonical-tree reduce to an fp32 sub-tree partial (TP1: identity)."""

    if tp_size == 1:
        return partial
    if reduce is None:
        raise ValueError(
            f"{what}: tp_size={tp_size} needs a reduce callable (rank-ordered all-gather + "
            "canonical tree, e.g. tree_reduce_over_ranks(collective)); a plain all-reduce would "
            "not reproduce the TP1 bits"
        )
    root = reduce(partial)
    if not isinstance(root, torch.Tensor) or root.dtype != torch.float32:
        raise TypeError(f"{what}: reduce must return the fp32 tree root")
    if root.shape != partial.shape:
        raise ValueError(f"{what}: reduce returned {tuple(root.shape)}, expected "
                         f"{tuple(partial.shape)}")
    return root


def column_parallel_da_partial(
    dy_local: torch.Tensor, w_local: torch.Tensor, tp_size: int
) -> torch.Tensor:
    """fp32 sub-tree partial of ``dA = dY @ W`` over this rank's ``8/tp`` canonical N chunks.

    ``dy_local (M, N_local)`` bf16 and ``w_local (N_local, K)`` bf16 are the rank's column
    shard (``N_local = N / tp``, consecutive rows of W in rank order).  The chunk length is
    ``N / 8`` at every TP degree, so the partial is a node of the TP1 dA tree.
    """

    chunks = _tp_chunks(tp_size)
    if dy_local.dim() != 2 or w_local.dim() != 2 or dy_local.size(1) != w_local.size(0):
        raise ValueError(
            f"column_parallel_da_partial: dy_local {tuple(dy_local.shape)} vs w_local "
            f"{tuple(w_local.shape)}"
        )
    if w_local.size(0) % (chunks * _kernels.MIN_BLOCK_K):
        raise ValueError(
            f"column-parallel shard N_local={w_local.size(0)} must be a multiple of "
            f"{chunks * _kernels.MIN_BLOCK_K} ({chunks} canonical chunks x 32), i.e. N % 256 == 0"
        )
    return _kernels.tree_gemm(dy_local, w_local, out_dtype=torch.float32, num_chunks=chunks)


class _ColumnParallelFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a2d, w_local, out_dtype, tp_size, da_reduce):
        ctx.save_for_backward(a2d, w_local)
        ctx.tp_size = tp_size
        ctx.da_reduce = da_reduce
        return _kernels.tree_gemm(a2d, w_local.t(), out_dtype=out_dtype)

    @staticmethod
    def backward(ctx, grad_y):
        a2d, w_local = ctx.saved_tensors
        dy = _grad_bf16(grad_y)
        da = dw = None
        if ctx.needs_input_grad[0]:
            partial = column_parallel_da_partial(dy, w_local, ctx.tp_size)
            root = _reduce_ranks(partial, ctx.tp_size, ctx.da_reduce, "column-parallel dA")
            da = root.to(a2d.dtype)  # single rounding after the full tree
        if ctx.needs_input_grad[1]:
            dw = _dw(dy, a2d, w_local, False)
        _record()
        return da, dw, None, None, None


def column_parallel_fast_det_gemm(
    a: torch.Tensor,
    w_local: torch.Tensor,
    *,
    tp_size: int,
    da_reduce: ReduceFn | None = None,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Column-parallel shard ``y_local = a @ w_local.T`` whose input gradient equals TP1 bitwise.

    ``a (..., K)`` is replicated on every rank, ``w_local (N/tp, K)`` holds this rank's
    consecutive rows of W (rank order).  Forward bits equal the corresponding TP1 columns.
    Backward: dA's reduction runs over N, split across ranks; this rank computes the fp32
    sub-tree partial over its ``8/tp`` canonical N chunks (:func:`column_parallel_da_partial`),
    ``da_reduce(partial)`` must return the fp32 canonical-tree root over all ranks' partials in
    rank order (rank-ordered all-gather + ``combine_tree``; see :func:`tree_reduce_over_ranks`),
    and the root is rounded once to bf16.  ``dW_local`` reduces over tokens only (ragged rule)
    and equals the TP1 rows.  ``N % 256 == 0`` is required (no ragged rule across ranks).
    """

    tp_size = int(tp_size)
    _tp_chunks(tp_size)
    k_size = _check(a, w_local, w_kn=False, out_dtype=out_dtype)
    lead = a.shape[:-1]
    y = _ColumnParallelFn.apply(_flatten(a, k_size), w_local, out_dtype, tp_size, da_reduce)
    return y.reshape(*lead, w_local.size(0))


class _RowParallelFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a2d, w_local, out_dtype, tp_size, y_reduce):
        ctx.save_for_backward(a2d, w_local)
        chunks = _tp_chunks(tp_size)
        partial = _kernels.tree_gemm(
            a2d, w_local.t(), out_dtype=torch.float32, num_chunks=chunks
        )
        root = _reduce_ranks(partial, tp_size, y_reduce, "row-parallel forward")
        return root.to(out_dtype)

    @staticmethod
    def backward(ctx, grad_y):
        a2d, w_local = ctx.saved_tensors
        dy = _grad_bf16(grad_y)  # replicated on every rank
        da = dw = None
        if ctx.needs_input_grad[0]:
            # dA_local[m, k] = sum_n dY[m, n] * W_local[n, k]: full N on every rank, no exchange.
            da = _kernels.tree_gemm(dy, w_local, out_dtype=a2d.dtype, allow_pad=True)
        if ctx.needs_input_grad[1]:
            dw = _dw(dy, a2d, w_local, False)
        _record()
        return da, dw, None, None, None


def row_parallel_fast_det_gemm(
    a_local: torch.Tensor,
    w_local: torch.Tensor,
    *,
    tp_size: int,
    y_reduce: ReduceFn | None = None,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Row-parallel shard (o_proj, down_proj) with autograd, bitwise equal to TP1.

    ``a_local (..., K/tp)`` and ``w_local (N, K/tp)`` are this rank's consecutive K slice (its
    ``8/tp`` canonical chunks, ``rank_chunk_ids``).  Forward: fp32 sub-tree partial ->
    ``y_reduce`` (fp32 canonical-tree root over all ranks, rank order) -> one rounding.
    Backward: ``dY`` is replicated, so ``dA_local = dY @ W_local`` and ``dW_local`` need no
    exchange and equal the TP1 columns.
    """

    tp_size = int(tp_size)
    chunks = _tp_chunks(tp_size)
    k_local = _check(a_local, w_local, w_kn=False, out_dtype=out_dtype)
    if k_local % (chunks * _kernels.MIN_BLOCK_K):
        raise ValueError(
            f"row-parallel K_local={k_local} must be a multiple of {chunks * _kernels.MIN_BLOCK_K}"
        )
    lead = a_local.shape[:-1]
    y = _RowParallelFn.apply(_flatten(a_local, k_local), w_local, out_dtype, tp_size, y_reduce)
    return y.reshape(*lead, w_local.size(0))


def tree_reduce_over_ranks(collective: Any) -> ReduceFn:
    """``partial -> canonical tree over all ranks' fp32 partials`` via a rank-ordered all-gather.

    ``collective.all_gather(x)`` must move bits only (no arithmetic) and return the ranks'
    tensors in rank order, either stacked along a new dim 0 or as a list (both conventions used
    in this repo).  The fold is ``runtime.tp_canonical_tree.combine_tree`` (the WS2 tree).
    """

    from rl_engine.models.muse_glimmer.runtime import tp_canonical_tree as _tct

    def reduce(partial: torch.Tensor) -> torch.Tensor:
        gathered = collective.all_gather(partial.unsqueeze(0).contiguous())
        if isinstance(gathered, torch.Tensor):
            leaves = list(gathered.reshape(-1, *partial.shape).unbind(0))
        else:
            leaves = [g.reshape(partial.shape) for g in gathered]
        return _tct.combine_tree(leaves)

    return reduce


# ----------------------------------------------------------------------------------------------
# Provenance
# ----------------------------------------------------------------------------------------------


def _runtime_env() -> dict[str, Any]:
    import triton

    env: dict[str, Any] = {"triton": triton.__version__, "torch": torch.__version__}
    # Never initialise CUDA just to build a provenance dict; once a GEMM ran it is initialised.
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        env["device_name"] = torch.cuda.get_device_name()
        env["device_capability"] = list(torch.cuda.get_device_capability())
    else:
        env["device_name"] = "cuda not initialised in this process"
    return env


def trace(
    m_size: int | None = None,
    out_dtype: torch.dtype = torch.bfloat16,
    k_chunks: int = CANONICAL_K_CHUNKS,
) -> dict[str, Any]:
    """Provenance dict (contract R4/R6): dtypes, reduction order, rules and tile constants.

    With ``m_size`` the ``tile_config`` entry is the one shape class that M selects; without it
    all shape classes are listed.  ``k_chunks`` records a sub-tree launch (canonical: 8).
    ``runtime`` names the GPU and Triton version: bit equality is promised within one GPU
    generation and one MMA lowering only.
    """

    if k_chunks not in _SUPPORTED_CHUNK_COUNTS:
        raise ValueError(f"k_chunks must be one of {_SUPPORTED_CHUNK_COUNTS}, got {k_chunks}")
    if m_size is None:
        tile_config: dict[str, Any] = {
            "decode16": asdict(_kernels.DECODE16_CONFIG),
            "decode64": asdict(_kernels.DECODE64_CONFIG),
            "prefill": asdict(_kernels.PREFILL_CONFIG),
        }
        shape_class = dict(_kernels.SHAPE_CLASSES)
    else:
        name, cfg = _kernels.select_config(int(m_size))
        tile_config = {name: asdict(cfg)}
        shape_class = {name: _kernels.SHAPE_CLASSES[name]}
    tree = TREE_ORDER if k_chunks == CANONICAL_K_CHUNKS else f"sub-tree of {k_chunks} chunks"
    return {
        "op": OP_NAME,
        "kernel_id": KERNEL_ID,
        "dtype_in": "bfloat16",
        "dtype_acc": "float32",
        "dtype_out": str(out_dtype).replace("torch.", ""),
        "reduction_order": (
            f"K split into {k_chunks} consecutive chunks [c*K/{k_chunks}, (c+1)*K/{k_chunks}); "
            "per chunk a dependent chain of bf16 mma.sync.m16n8k16 (fp32 accumulate, starting "
            "at +0) in ascending k over >= 2 BLOCK_K iterations; chunk partials combined in "
            f"registers as fp32 IEEE adds in the fixed tree {tree}; single rounding to dtype_out "
            "at the store"
        ),
        "k_chunks": int(k_chunks),
        "tree_order": TREE_ORDER,
        "k_rule": K_RULE,
        "block_k_rule": BLOCK_K_RULE,
        "min_chunk_iters": _kernels.MIN_ITERS,
        "da_chunk_rule": DA_CHUNK_RULE,
        "dw_chunk_rule": DW_CHUNK_RULE,
        "grad_cast": GRAD_CAST,
        "tree_ir_guard": (
            "enabled: TTGIR of every compiled variant must hold NUM_CHUNKS-1 fp32 tree adds"
            if _kernels.ir_guard_enabled()
            else f"DISABLED by {_kernels.IR_GUARD_ENV} (explicit opt-out)"
        ),
        "split_k": False,
        "atomics": False,
        "autotune": False,
        "allow_tf32": False,
        "shape_class": shape_class,
        "tile_config": tile_config,
        "runtime": _runtime_env(),
    }


# Provenance travels with the callable: consumers that discover it via getattr(gemm, "trace")
# (swiglu_ffn, the strict reference model) get the full dict (finding det_gemm:3).
fast_det_gemm.trace = trace  # type: ignore[attr-defined]
fast_det_gemm.name = OP_NAME  # type: ignore[attr-defined]


__all__ = [
    "KERNEL_ID",
    "OP_NAME",
    "column_parallel_da_partial",
    "column_parallel_fast_det_gemm",
    "combine_tree",
    "fast_det_gemm",
    "fast_det_gemm_partial",
    "forward",
    "forward_fp32",
    "rank_chunk_ids",
    "row_parallel_fast_det_gemm",
    "trace",
    "tree_reduce_over_ranks",
]
