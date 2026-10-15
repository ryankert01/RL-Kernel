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
  forward   y  = round_out( tree_c( sum_{k in chunk c} a[m,k] * w[n,k] ) ),  chunks over K
  backward  dY -> bf16 (declared cast; fp32 for the lm_head path),
            dA = tree over N   of dY_bf16[m,n] * w[n,k]            -> bf16  (K-rule on N)
            dW = tree over M   of dY_bf16[m,n] * a[m,k]            -> bf16  (ragged rule on M:
                 M padded to a multiple of 256, 8 chunks of M_pad/8 token rows, rows >= M masked
                 to zero -- repeatable; batch invariance of dW is not a contract requirement)
Every reduction length of the model (K in {6656, 4096, 19968}, N in {4096, 256, 19968, 6656,
202752}) is a multiple of 256, so forward and dA never pad (the forward raises for other K; dA
applies the same masked rule as dW for other N, which keeps dA batch-invariant).  BLOCK_K is 64
when the chunk length allows it and 32 otherwise (N = 256: chunk length 32).  BLOCK_K never
changes the bits.

Tensor-parallel partials
------------------------
``fast_det_gemm_partial(a_local, w_local, chunk_ids)`` returns the fp32 value of the sub-tree over
the consecutive canonical chunks a rank owns; ``combine_tree`` over the ranks' partials yields
bits equal to the TP1 output (``tests/models/muse_glimmer/test_det_gemm.py`` simulates TP 2/4/8 on
one GPU by slicing K).
"""

from __future__ import annotations

from collections.abc import Sequence
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


class _FastDetGemmFn(torch.autograd.Function):
    """Deterministic forward and backward; see the module docstring for the reduction rules."""

    @staticmethod
    def forward(ctx, a2d, w, out_dtype, w_kn, k_chunks):
        ctx.save_for_backward(a2d, w)
        ctx.w_kn = bool(w_kn)
        return _kernels.tree_gemm(a2d, _as_kn(w, w_kn), out_dtype=out_dtype, num_chunks=k_chunks)

    @staticmethod
    def backward(ctx, grad_y):
        a2d, w = ctx.saved_tensors
        # Declared cast: the upstream gradient enters the GEMMs as bf16 (fp32 for lm_head logits).
        dy = grad_y.contiguous().to(torch.bfloat16)
        b_kn = _as_kn(w, ctx.w_kn)  # (K, N)
        da = dw = None
        if ctx.needs_input_grad[0]:
            # dA[m, k] = sum_n dY[m, n] * B[k, n]; reduction over N with the canonical chunk rule
            # (every model N is a multiple of 256; other N fall back to the masked/padded rule,
            # which is still batch-invariant because the padding does not depend on M).
            da = _kernels.tree_gemm(dy, b_kn.t(), out_dtype=a2d.dtype, allow_pad=True)
        if ctx.needs_input_grad[1]:
            # Reduction over the M token rows: ragged rule (M padded to 256, rows >= M masked).
            if ctx.w_kn:
                # dW[k, n] = sum_m A[m, k] * dY[m, n]
                dw = _kernels.tree_gemm(a2d.t(), dy, out_dtype=w.dtype, allow_pad=True)
            else:
                # dW[n, k] = sum_m dY[m, n] * A[m, k]
                dw = _kernels.tree_gemm(dy.t(), a2d, out_dtype=w.dtype, allow_pad=True)
        record_backward(
            "det_gemm",
            kernel_id=KERNEL_ID,
            impl="muse_glimmer_fast_det_gemm",
            family="triton",
        )
        return da, dw, None, None, None


def fast_det_gemm(
    a: torch.Tensor,
    w: torch.Tensor,
    *,
    out_dtype: torch.dtype = torch.bfloat16,
    k_chunks: int = CANONICAL_K_CHUNKS,
    w_kn: bool = False,
) -> torch.Tensor:
    """Strict GEMM ``a @ w.T`` (or ``a @ w`` with ``w_kn=True``) with deterministic autograd.

    ``a`` is ``(..., K)`` bf16 (leading dims are flattened to M), ``w`` is ``(N, K)`` bf16 (an
    ``nn.Linear`` weight) or ``(K, N)`` when ``w_kn=True``.  ``out_dtype`` is bf16 (default) or
    fp32 (lm_head logits).  ``k_chunks`` is 8 under the contract; 1, 2 and 4 are accepted for
    sub-tree experiments only.  The result is ``(..., N)``.
    """

    if k_chunks not in _SUPPORTED_CHUNK_COUNTS:
        raise ValueError(f"k_chunks must be one of {_SUPPORTED_CHUNK_COUNTS}, got {k_chunks}")
    k_size = _check(a, w, w_kn=w_kn, out_dtype=out_dtype)
    n_size = w.size(1) if w_kn else w.size(0)
    lead = a.shape[:-1]
    a2d = a.reshape(-1, k_size)
    y = _FastDetGemmFn.apply(a2d, w, out_dtype, w_kn, k_chunks)
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
    bf16, equals the TP1 ``fast_det_gemm`` output bitwise.
    """

    ids = tuple(int(c) for c in chunk_ids)
    count = len(ids)
    if count not in _SUPPORTED_CHUNK_COUNTS:
        raise ValueError(f"chunk_ids must hold 1, 2, 4 or 8 ids, got {count}")
    if ids != tuple(range(ids[0], ids[0] + count)) or ids[0] % count or ids[-1] >= 8 or ids[0] < 0:
        raise ValueError(f"chunk_ids {ids} is not an aligned node of the 8-chunk tree")
    k_local = _check(a_local, w_local, w_kn=w_kn, out_dtype=torch.float32)
    if k_local % (count * _kernels.MIN_BLOCK_K):
        raise ValueError(
            f"local K {k_local} must be a multiple of {count * _kernels.MIN_BLOCK_K} for "
            f"{count} chunks"
        )
    lead = a_local.shape[:-1]
    out = _kernels.tree_gemm(
        a_local.reshape(-1, k_local),
        _as_kn(w_local, w_kn),
        out_dtype=torch.float32,
        num_chunks=count,
    )
    return out.reshape(*lead, -1)


def trace(m_size: int | None = None, out_dtype: torch.dtype = torch.bfloat16) -> dict[str, Any]:
    """Provenance dict (contract R4): dtypes, reduction order and the pinned tile constants.

    With ``m_size`` the ``tile_config`` entry is the one shape class that M selects; without it
    all shape classes are listed.
    """

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
    return {
        "op": OP_NAME,
        "kernel_id": KERNEL_ID,
        "dtype_in": "bfloat16",
        "dtype_acc": "float32",
        "dtype_out": str(out_dtype).replace("torch.", ""),
        "reduction_order": (
            f"K split into {CANONICAL_K_CHUNKS} consecutive chunks [c*K/8, (c+1)*K/8); "
            "per chunk a dependent chain of bf16 mma.sync.m16n8k16 (fp32 accumulate) in "
            "ascending k; chunk partials combined in registers as fp32 IEEE adds in the fixed "
            f"tree {TREE_ORDER}; single rounding to dtype_out at the store"
        ),
        "k_chunks": CANONICAL_K_CHUNKS,
        "tree_order": TREE_ORDER,
        "block_k_rule": "64 if (K/8) % 64 == 0 else 32; bits do not depend on BLOCK_K",
        "dw_chunk_rule": (
            "reduction over M tokens: M_pad = roundup(M, 256), 8 chunks of M_pad/8 rows, "
            "rows >= M masked to zero (repeatable; not batch-invariant by design)"
        ),
        "split_k": False,
        "atomics": False,
        "autotune": False,
        "allow_tf32": False,
        "shape_class": shape_class,
        "tile_config": tile_config,
    }


__all__ = [
    "KERNEL_ID",
    "OP_NAME",
    "combine_tree",
    "fast_det_gemm",
    "fast_det_gemm_partial",
    "forward",
    "forward_fp32",
    "rank_chunk_ids",
    "trace",
]
