# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Fast deterministic GEMM with the canonical 8-chunk K tree (Muse-Glimmer-30B strict POC, O3).

One Triton program owns one ``(BLOCK_M, BLOCK_N)`` output tile and walks the whole reduction
dimension itself: no split-K across programs, no atomics, no autotuning.  The reduction order is
therefore a pure function of the kernel constants and is independent of M (batch), of the tile a
row lands in, of the TP degree and of sequence chunking.

Reduction order (the arithmetic contract)
-----------------------------------------
* K is split into ``NUM_CHUNKS`` (8 canonically) consecutive chunks of ``CHUNK_LEN = K_pad / 8``
  columns; chunk ``c`` covers ``[c * CHUNK_LEN, (c + 1) * CHUNK_LEN)``.
* Inside a chunk the fp32 accumulator starts at +0 and is advanced by
  ``tl.dot(a, b, acc, allow_tf32=False)`` over ``BLOCK_K`` columns at a time in ascending k.  On
  sm80+ this is a dependent chain of bf16 ``mma.sync.m16n8k16`` instructions (fp32 accumulate)
  in ascending k; the chain does not depend on BLOCK_M, BLOCK_N, num_warps, num_stages or BLOCK_K
  (BLOCK_K only changes how many k16 steps one loop iteration issues) PROVIDED every chunk loop
  runs at least two iterations (``ITERS = CHUNK_LEN / BLOCK_K >= 2``, see below).  The
  bf16 x bf16 products are exact in fp32.
* The chunk partials are combined in registers as fp32 IEEE adds in the fixed tree
  ``((0+1)+(2+3))+((4+5)+(6+7))`` with at most four live accumulators
  (``s0123, s45, acc6, acc7`` while chunk 7 is being accumulated).
* The root is rounded ONCE to the output dtype (bf16 round-to-nearest-even, or stored as fp32).

Why ``ITERS >= 2`` is part of the arithmetic (review finding det_gemm:1, reproduced on Triton
3.7.1): when a chunk is a single ``BLOCK_K`` step, MLIR folds the 1-trip loop and Triton's TTIR
combine pass rewrites ``dot(a1, b1, 0) + acc0`` into ``dot(a1, b1, acc0)``: the tree add
``s01 = acc0 + acc1`` disappears and chunk 1 is MMA-accumulated on top of chunk 0 (likewise
s23/s45/s67), so the kernel no longer implements the declared tree (K=256/512: ~50 % of the
outputs differ from the tree in the last bits).  ``plan_chunks`` therefore halves BLOCK_K until
``ITERS >= 2`` (BLOCK_K = 16 for 32-wide chunks, 32 for 64-wide chunks), the kernel asserts it
statically, and :func:`tree_gemm` verifies once per compiled variant that the TTGIR still holds
exactly ``NUM_CHUNKS - 1`` fp32 tree adds (fail closed: :class:`TreeIRError`).

Cast points: inputs are bf16 (no cast), accumulation fp32, one rounding at the store.

Ragged reduction (weight gradients): when the reduction length is not a multiple of
``NUM_CHUNKS * 32`` the caller may opt into padding (``allow_pad=True``): ``K_pad`` is K rounded
up to a multiple of 256, ``CHUNK_LEN = K_pad / 8`` and loads beyond K are masked to zero, so
empty chunk partials are exact zeros and ``x + 0 == x`` leaves the tree bits unchanged.  This is
the chunk rule for dW (reduction over the M tokens): repeatable, not batch-invariant (dW depends
on every token anyway).  For the forward and dA GEMMs of the model every reduction length
(6656, 4096, 19968, 256, 202752, 25344) is a multiple of 256, so no padding is used there.

Device: the launch is pinned to the operands' CUDA device (``torch.cuda.device(a.device)``), so a
call on tensors that do not live on the current device launches on the right context/stream.
"""

from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass

import torch
import triton
import triton.language as tl

CANONICAL_K_CHUNKS = 8
# Alignment unit of a chunk: K_pad must be a multiple of NUM_CHUNKS * MIN_BLOCK_K (8 x 32 = 256).
MIN_BLOCK_K = 32
# Smallest BLOCK_K the kernel may use (one bf16 mma.sync k16 step); lets a 32-wide chunk still run
# ITERS = 2 loop iterations (module docstring).
MIN_MMA_BLOCK_K = 16
MIN_ITERS = 2
_SUPPORTED_CHUNK_COUNTS = (1, 2, 4, 8)
# Explicit opt-out of the compiled-IR tree check (recorded in trace()); never set in production.
IR_GUARD_ENV = "RL_KERNEL_MUSE_GLIMMER_SKIP_TREE_IR_GUARD"


class TreeIRError(RuntimeError):
    """The compiled kernel does not contain the declared fp32 tree adds (fail closed)."""


@dataclass(frozen=True)
class TileConfig:
    """Compile-time launch constants of one shape class (contract R6).  Performance-only."""

    block_m: int
    block_n: int
    block_k: int
    num_warps: int
    num_stages: int
    group_m: int = 8


@dataclass(frozen=True)
class ChunkPlan:
    """How a reduction of length ``k`` is split into the canonical chunks."""

    k: int
    k_pad: int
    num_chunks: int
    chunk_len: int
    block_k: int
    masked: bool


# Pinned per shape class (A100-80GB, Triton 3.7.1; picked from an offline sweep recorded in
# REPORT_det_gemm.md).  Tile constants do not change the bits as long as ITERS >= 2 (module doc);
# ``block_k`` is the preferred value, ``plan_chunks`` lowers it for short chunks.
PREFILL_CONFIG = TileConfig(block_m=64, block_n=128, block_k=64, num_warps=8, num_stages=3)
DECODE64_CONFIG = TileConfig(block_m=32, block_n=64, block_k=64, num_warps=4, num_stages=4)
DECODE16_CONFIG = TileConfig(block_m=16, block_n=64, block_k=64, num_warps=4, num_stages=4)

SHAPE_CLASSES: dict[str, str] = {
    "decode16": "M <= 16",
    "decode64": "16 < M <= 64",
    "prefill": "M > 64",
}

_LAST_LAUNCH: dict[str, object] = {}


def select_config(m_size: int) -> tuple[str, TileConfig]:
    """Return ``(shape_class, TileConfig)`` for a GEMM with ``m_size`` output rows."""

    if m_size <= 16:
        return "decode16", DECODE16_CONFIG
    if m_size <= 64:
        return "decode64", DECODE64_CONFIG
    return "prefill", PREFILL_CONFIG


def plan_chunks(
    k_size: int,
    num_chunks: int = CANONICAL_K_CHUNKS,
    *,
    allow_pad: bool = False,
    block_k: int = 64,
) -> ChunkPlan:
    """Split a reduction of length ``k_size`` into ``num_chunks`` canonical chunks.

    ``K_pad = roundup(K, num_chunks * 32)``; without ``allow_pad`` a K that needs padding raises
    (out of contract).  ``block_k`` is the preferred BLOCK_K; it is halved until it divides the
    chunk length AND the chunk loop runs at least ``MIN_ITERS = 2`` iterations (so no tree add can
    be folded into the MMA chain; module docstring), never below ``MIN_MMA_BLOCK_K = 16``.
    """

    if num_chunks not in _SUPPORTED_CHUNK_COUNTS:
        raise ValueError(f"num_chunks must be one of {_SUPPORTED_CHUNK_COUNTS}, got {num_chunks}")
    if k_size <= 0:
        raise ValueError(f"reduction length must be positive, got {k_size}")
    if block_k < MIN_BLOCK_K or block_k % MIN_BLOCK_K:
        raise ValueError(f"block_k must be a positive multiple of {MIN_BLOCK_K}, got {block_k}")
    unit = num_chunks * MIN_BLOCK_K
    k_pad = -(-k_size // unit) * unit
    if k_pad != k_size and not allow_pad:
        raise ValueError(
            f"reduction length {k_size} is not a multiple of {unit} "
            f"({num_chunks} chunks x {MIN_BLOCK_K}); out of contract for a reduction without the "
            "declared ragged (token-axis) padding rule"
        )
    chunk_len = k_pad // num_chunks
    effective_block_k = block_k
    while chunk_len % effective_block_k or chunk_len // effective_block_k < MIN_ITERS:
        effective_block_k //= 2
    if effective_block_k < MIN_MMA_BLOCK_K:  # unreachable: chunk_len >= 32 by construction
        raise ValueError(f"no BLOCK_K >= {MIN_MMA_BLOCK_K} gives {MIN_ITERS} chunk iterations")
    return ChunkPlan(
        k=k_size,
        k_pad=k_pad,
        num_chunks=num_chunks,
        chunk_len=chunk_len,
        block_k=effective_block_k,
        masked=k_pad != k_size,
    )


@triton.jit
def _chunk_partial(
    a_ptr,
    b_ptr,
    offs_m,
    offs_n,
    k_begin,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    ITERS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    MASK_K: tl.constexpr,
):
    """fp32 partial of one K chunk: ``ITERS`` dependent bf16 MMA steps over ascending k."""

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    offs_k = k_begin + tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :].to(tl.int64) * stride_ak
    b_ptrs = b_ptr + offs_k[:, None].to(tl.int64) * stride_bk + offs_n[None, :] * stride_bn
    for it in range(ITERS):
        if MASK_K:
            k_ok = (offs_k + it * BLOCK_K) < K
            a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & k_ok[None, :], other=0.0)
            b = tl.load(b_ptrs, mask=k_ok[:, None] & (offs_n[None, :] < N), other=0.0)
        else:
            if EVEN_M:
                a = tl.load(a_ptrs)
            else:
                a = tl.load(a_ptrs, mask=offs_m[:, None] < M, other=0.0)
            if EVEN_N:
                b = tl.load(b_ptrs)
            else:
                b = tl.load(b_ptrs, mask=offs_n[None, :] < N, other=0.0)
        acc = tl.dot(a, b, acc, allow_tf32=False)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    return acc


@triton.jit
def det_gemm_tree_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    CHUNK_LEN: tl.constexpr,
    NUM_CHUNKS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    MASK_K: tl.constexpr,
):
    """C[M, N] = A[M, K] @ B[K, N]; one program per output tile, canonical K tree in registers."""

    tl.static_assert(
        NUM_CHUNKS == 1 or NUM_CHUNKS == 2 or NUM_CHUNKS == 4 or NUM_CHUNKS == 8,
        "NUM_CHUNKS must be 1, 2, 4 or 8",
    )
    tl.static_assert(CHUNK_LEN % BLOCK_K == 0, "CHUNK_LEN must be a multiple of BLOCK_K")
    ITERS: tl.constexpr = CHUNK_LEN // BLOCK_K
    # A 1-trip chunk loop lets the compiler fold the tree adds into the MMA chain (det_gemm:1).
    tl.static_assert(ITERS >= 2, "each chunk must run >= 2 BLOCK_K iterations (tree adds)")

    # Grouped rasterisation (L2 reuse of the B operand).  Which program computes which tile has
    # no effect on the arithmetic of the tile.
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)

    # Chunks are visited in order 0..NUM_CHUNKS-1; partials are combined in the fixed tree.
    acc0 = _chunk_partial(
        a_ptr, b_ptr, offs_m, offs_n, 0 * CHUNK_LEN, M, N, K,
        stride_am, stride_ak, stride_bk, stride_bn,
        ITERS, BLOCK_M, BLOCK_N, BLOCK_K, EVEN_M, EVEN_N, MASK_K,
    )  # fmt: skip
    if NUM_CHUNKS == 1:
        root = acc0
    else:
        acc1 = _chunk_partial(
            a_ptr, b_ptr, offs_m, offs_n, 1 * CHUNK_LEN, M, N, K,
            stride_am, stride_ak, stride_bk, stride_bn,
            ITERS, BLOCK_M, BLOCK_N, BLOCK_K, EVEN_M, EVEN_N, MASK_K,
        )  # fmt: skip
        s01 = acc0 + acc1
        if NUM_CHUNKS == 2:
            root = s01
        else:
            acc2 = _chunk_partial(
                a_ptr, b_ptr, offs_m, offs_n, 2 * CHUNK_LEN, M, N, K,
                stride_am, stride_ak, stride_bk, stride_bn,
                ITERS, BLOCK_M, BLOCK_N, BLOCK_K, EVEN_M, EVEN_N, MASK_K,
            )  # fmt: skip
            acc3 = _chunk_partial(
                a_ptr, b_ptr, offs_m, offs_n, 3 * CHUNK_LEN, M, N, K,
                stride_am, stride_ak, stride_bk, stride_bn,
                ITERS, BLOCK_M, BLOCK_N, BLOCK_K, EVEN_M, EVEN_N, MASK_K,
            )  # fmt: skip
            s23 = acc2 + acc3
            s0123 = s01 + s23
            if NUM_CHUNKS == 4:
                root = s0123
            else:
                acc4 = _chunk_partial(
                    a_ptr, b_ptr, offs_m, offs_n, 4 * CHUNK_LEN, M, N, K,
                    stride_am, stride_ak, stride_bk, stride_bn,
                    ITERS, BLOCK_M, BLOCK_N, BLOCK_K, EVEN_M, EVEN_N, MASK_K,
                )  # fmt: skip
                acc5 = _chunk_partial(
                    a_ptr, b_ptr, offs_m, offs_n, 5 * CHUNK_LEN, M, N, K,
                    stride_am, stride_ak, stride_bk, stride_bn,
                    ITERS, BLOCK_M, BLOCK_N, BLOCK_K, EVEN_M, EVEN_N, MASK_K,
                )  # fmt: skip
                s45 = acc4 + acc5
                acc6 = _chunk_partial(
                    a_ptr, b_ptr, offs_m, offs_n, 6 * CHUNK_LEN, M, N, K,
                    stride_am, stride_ak, stride_bk, stride_bn,
                    ITERS, BLOCK_M, BLOCK_N, BLOCK_K, EVEN_M, EVEN_N, MASK_K,
                )  # fmt: skip
                acc7 = _chunk_partial(
                    a_ptr, b_ptr, offs_m, offs_n, 7 * CHUNK_LEN, M, N, K,
                    stride_am, stride_ak, stride_bk, stride_bn,
                    ITERS, BLOCK_M, BLOCK_N, BLOCK_K, EVEN_M, EVEN_N, MASK_K,
                )  # fmt: skip
                s67 = acc6 + acc7
                s4567 = s45 + s67
                root = s0123 + s4567

    # Single rounding to the output dtype (bf16 RNE, or fp32 stored as is).
    c = root.to(c_ptr.dtype.element_ty)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    if EVEN_M and EVEN_N:
        tl.store(c_ptrs, c)
    else:
        tl.store(c_ptrs, c, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def _check_operands(a: torch.Tensor, b: torch.Tensor) -> None:
    if a.dim() != 2 or b.dim() != 2:
        raise ValueError("tree_gemm expects two 2-D operands (M, K) and (K, N)")
    if a.size(1) != b.size(0):
        raise ValueError(f"tree_gemm K mismatch: a is {tuple(a.shape)}, b is {tuple(b.shape)}")
    if a.dtype != torch.bfloat16 or b.dtype != torch.bfloat16:
        raise TypeError(f"tree_gemm requires bf16 operands, got {a.dtype} and {b.dtype}")
    if not a.is_cuda or not b.is_cuda or a.device != b.device:
        raise RuntimeError("tree_gemm operands must share one CUDA device")


def tree_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    out_dtype: torch.dtype = torch.bfloat16,
    num_chunks: int = CANONICAL_K_CHUNKS,
    config: TileConfig | None = None,
    allow_pad: bool = False,
) -> torch.Tensor:
    """``a (M, K) @ b (K, N)`` with the canonical K tree; both operands may be strided views.

    ``b`` is addressed as ``B[k, n] = b_ptr + k * b.stride(0) + n * b.stride(1)``, so an
    ``nn.Linear`` weight ``w (N, K)`` is passed as ``w.t()`` with no copy.  ``out_dtype`` is bf16
    (default, one rounding) or fp32 (lm_head logits).  ``num_chunks`` < 8 computes the sub-tree
    over the chunks the local operands hold (TP partials).  ``config`` overrides the pinned tile
    constants (tests and the offline sweep only; bits do not depend on it).
    """

    _check_operands(a, b)
    if out_dtype not in (torch.bfloat16, torch.float32):
        raise TypeError(f"tree_gemm out_dtype must be bf16 or fp32, got {out_dtype}")
    m_size, k_size = a.shape
    n_size = b.size(1)
    shape_class, cfg = select_config(m_size)
    if config is not None:
        cfg = config
        shape_class = "override"
    plan = plan_chunks(k_size, num_chunks, allow_pad=allow_pad, block_k=cfg.block_k)
    out = torch.empty((m_size, n_size), dtype=out_dtype, device=a.device)
    if m_size == 0 or n_size == 0:
        return out
    grid = (triton.cdiv(m_size, cfg.block_m) * triton.cdiv(n_size, cfg.block_n),)
    # Triton launches on the *current* device/stream: pin it to the operands' device.
    with torch.cuda.device(a.device):
        compiled = det_gemm_tree_kernel[grid](
            a,
            b,
            out,
            m_size,
            n_size,
            k_size,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            out.stride(0),
            out.stride(1),
            CHUNK_LEN=plan.chunk_len,
            NUM_CHUNKS=plan.num_chunks,
            BLOCK_M=cfg.block_m,
            BLOCK_N=cfg.block_n,
            BLOCK_K=plan.block_k,
            GROUP_M=cfg.group_m,
            EVEN_M=m_size % cfg.block_m == 0,
            EVEN_N=n_size % cfg.block_n == 0,
            MASK_K=plan.masked,
            num_warps=cfg.num_warps,
            num_stages=cfg.num_stages,
        )
    _verify_tree_ir(compiled, plan.num_chunks, cfg.block_m, cfg.block_n)
    _LAST_LAUNCH.clear()
    _LAST_LAUNCH.update(
        shape=(m_size, n_size, k_size),
        shape_class=shape_class,
        config=asdict(cfg),
        plan=asdict(plan),
        grid=grid,
        n_regs=getattr(compiled, "n_regs", None),
        n_spills=getattr(compiled, "n_spills", None),
    )
    return out


_VERIFIED_IR: set[str] = set()
_TREE_ADD_CACHE: dict[tuple[int, int], re.Pattern[str]] = {}


def ir_guard_enabled() -> bool:
    """False only when the explicit opt-out env var is set (recorded in ``trace()``)."""

    return os.environ.get(IR_GUARD_ENV, "").strip().lower() not in ("1", "true", "yes")


def count_tree_adds(ttgir: str, block_m: int, block_n: int) -> int:
    """Number of fp32 ``arith.addf`` on ``(block_m, block_n)`` tensors in a TTGIR listing."""

    pattern = _TREE_ADD_CACHE.get((block_m, block_n))
    if pattern is None:
        pattern = re.compile(rf"arith\.addf [^\n]*: tensor<{block_m}x{block_n}xf32")
        _TREE_ADD_CACHE[(block_m, block_n)] = pattern
    return len(pattern.findall(ttgir))


def _verify_tree_ir(compiled: object, num_chunks: int, block_m: int, block_n: int) -> None:
    """Fail closed unless the compiled kernel keeps the ``num_chunks - 1`` declared tree adds.

    Checked once per compiled variant (keyed by Triton's kernel hash).  The only fp32 adds of
    ``(BLOCK_M, BLOCK_N)`` tensors in the kernel are the tree nodes, so a compiler that
    re-associates the tree (e.g. folds an add into ``tl.dot``'s accumulator, finding
    det_gemm:1) lowers the count and is rejected instead of silently changing the bits.
    """

    if not ir_guard_enabled():
        return
    key = str(getattr(compiled, "hash", id(compiled)))
    if key in _VERIFIED_IR:
        return
    asm = getattr(compiled, "asm", None)
    ttgir = asm.get("ttgir") if asm is not None and hasattr(asm, "get") else None
    if not ttgir:
        raise TreeIRError(
            "det_gemm_tree: cannot verify the compiled tree (no TTGIR in the Triton cache; unset "
            f"TRITON_STORE_BINARY_ONLY, or set {IR_GUARD_ENV}=1 to opt out explicitly)"
        )
    found = count_tree_adds(ttgir, block_m, block_n)
    if found != num_chunks - 1:
        raise TreeIRError(
            f"det_gemm_tree: compiled kernel holds {found} fp32 tree adds, expected "
            f"{num_chunks - 1} for {num_chunks} chunks; the compiler re-associated the canonical "
            "tree (refusing to produce out-of-contract bits)"
        )
    _VERIFIED_IR.add(key)


def last_launch() -> dict[str, object]:
    """Launch constants and ptxas register/spill counts of the most recent ``tree_gemm``."""

    return dict(_LAST_LAUNCH)


__all__ = [
    "CANONICAL_K_CHUNKS",
    "DECODE16_CONFIG",
    "DECODE64_CONFIG",
    "IR_GUARD_ENV",
    "MIN_BLOCK_K",
    "MIN_ITERS",
    "MIN_MMA_BLOCK_K",
    "PREFILL_CONFIG",
    "SHAPE_CLASSES",
    "ChunkPlan",
    "TileConfig",
    "TreeIRError",
    "count_tree_adds",
    "det_gemm_tree_kernel",
    "ir_guard_enabled",
    "last_launch",
    "plan_chunks",
    "select_config",
    "tree_gemm",
]
