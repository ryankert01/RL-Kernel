# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Triton kernels for contract O4 ``qknorm_scaled_rope`` (Muse-Glimmer-30B strict POC).

Fused, weightless per-head Q/K RMSNorm, the fp32 query scale (3.87) and rotate_half RoPE
over all 128 channels, with a single rounding to the output dtype at the very end.

Layout and block structure (contract R3: one logical row end-to-end inside one program)
    * ``q`` is ``(T, HQ, D)``, ``k`` is ``(T, HK, D)``, both contiguous; ``cos``/``sin`` are
      ``(T, D)`` fp32 rows already gathered per token.
    * One program per token ``t``. The program handles the token's HQ query heads as a
      ``[BLOCK_HQ, D]`` tile and its HK key heads as a ``[BLOCK_HK, D]`` tile, where
      ``BLOCK_H = next_pow2(H)`` and padded rows are masked. Every head row is normalized,
      scaled and rotated inside this one program with a block layout that depends only on
      the compile-time constants ``(BLOCK_HQ, BLOCK_HK, D, num_warps)`` - never on ``T`` or
      on the program id - so the bits of a row are the same at T=1 and T=4096 and in any
      batch position (batch invariance by construction).

Reduction order (fixed, recorded in ``trace()``, identical to the torch golden)
    The 128 channels are loaded as two ``[BLOCK_H, 64]`` halves ``x1 = x[:, :64]`` and
    ``x2 = x[:, 64:]``.  ``ss = tree64(x1*x1) + tree64(x2*x2)`` where ``tree64`` is the
    explicit binary tree pairing channels ``(2i, 2i+1)`` at each of 6 levels
    (``tl.reshape`` + ``tl.split`` + one add per pair); the 7th level adds the two half sums.
    This is exactly the golden's ``_tree_sum_last`` over 128 channels, so the kernel and
    ``qknorm_scaled_rope_fp32`` perform the same sequence of IEEE fp32 operations and the
    kernel output equals ``golden.to(bf16)`` bitwise (verified in the tests; ``tl.rsqrt`` and
    ``torch.pow(x, -0.5)`` both lower to the CUDA ``rsqrtf`` approximation).  The backward dot
    product ``sum(g * x_hat)`` uses the same tree.

Casts (contract R5)
    inputs bf16 -> fp32 once at load; ``rstd = rsqrt(ss / D + eps)`` fp32; ``x_hat = x * rstd``;
    ``q_hat *= qk_scale`` (fp32, Q only); rotation ``o1 = x1*c1 - x2*s1``,
    ``o2 = x2*c2 + x1*s2`` in fp32; one cast to the output dtype at the store.  The kernels
    are launched with ``enable_fp_fusion=False`` so that no mul+add is contracted into an FMA:
    the arithmetic is then the same sequence of IEEE fp32 operations as the torch golden.

Backward (row-local, deterministic, same block structure)
    ``g = dy -> fp32``; inverse rotation (orthogonal): ``g1 = dy1*c1 + dy2*s2``,
    ``g2 = dy2*c2 - dy1*s1``; ``g *= qk_scale`` (Q only); RMSNorm VJP
    ``dx = rstd * (g - x_hat * (sum(g * x_hat) / D))`` with ``rstd`` recomputed exactly as in
    the forward; one cast to the gradient dtype at the store.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

HEAD_DIM = 128
NUM_WARPS = 4
NUM_STAGES = 1  # no software pipelining: the kernels have no loops.

__all__ = [
    "HEAD_DIM",
    "NUM_STAGES",
    "NUM_WARPS",
    "qknorm_rope_backward_triton",
    "qknorm_rope_forward_triton",
    "tile_config",
]


@triton.jit
def _pair_add(x, R: tl.constexpr, N: tl.constexpr):
    """[R, N] -> [R, N // 2]: out[:, i] = x[:, 2i] + x[:, 2i + 1] (one fp32 add per pair)."""
    even, odd = tl.split(tl.reshape(x, [R, N // 2, 2]))
    return even + odd


@triton.jit
def _tree_sum_64(x, R: tl.constexpr):
    """Fixed binary tree over 64 channels pairing (2i, 2i+1) at each of the 6 levels.

    Identical to the golden's ``_tree_sum_last`` restricted to one 64-channel half.
    """
    x = _pair_add(x, R, 64)
    x = _pair_add(x, R, 32)
    x = _pair_add(x, R, 16)
    x = _pair_add(x, R, 8)
    x = _pair_add(x, R, 4)
    x = _pair_add(x, R, 2)
    return tl.reshape(x, [R])


@triton.jit
def _norm_scale_rope_rows_fwd(
    X,
    Y,
    row0,
    n_rows,
    c1,
    c2,
    s1,
    s2,
    scale,
    eps,
    BLOCK_ROWS: tl.constexpr,
    HALF: tl.constexpr,
    HAS_ROPE: tl.constexpr,
    APPLY_SCALE: tl.constexpr,
):
    """Normalize, scale and rotate ``n_rows`` consecutive head rows starting at ``row0``."""
    D: tl.constexpr = 2 * HALF
    rows = tl.arange(0, BLOCK_ROWS)
    cols = tl.arange(0, HALF)
    rmask = rows < n_rows
    mask2d = tl.broadcast_to(rmask[:, None], [BLOCK_ROWS, HALF])
    offs = (row0 + rows)[:, None] * D + cols[None, :]

    x1 = tl.load(X + offs, mask=mask2d, other=0.0).to(tl.float32)
    x2 = tl.load(X + offs + HALF, mask=mask2d, other=0.0).to(tl.float32)

    # mean(x^2): the golden's (2i, 2i+1) binary tree: 6 levels per 64-channel half, then the
    # 7th level adds the two half sums.
    ss = _tree_sum_64(x1 * x1, BLOCK_ROWS) + _tree_sum_64(x2 * x2, BLOCK_ROWS)
    rstd = tl.rsqrt(ss / D + eps)

    xh1 = x1 * rstd[:, None]
    xh2 = x2 * rstd[:, None]
    if APPLY_SCALE:
        xh1 = xh1 * scale
        xh2 = xh2 * scale

    if HAS_ROPE:
        # rotate_half: out = x_hat * cos + cat(-x2, x1) * sin  (no FMA contraction).
        o1 = xh1 * c1[None, :] - xh2 * s1[None, :]
        o2 = xh2 * c2[None, :] + xh1 * s2[None, :]
    else:
        o1 = xh1
        o2 = xh2

    out_ty = Y.dtype.element_ty
    tl.store(Y + offs, o1.to(out_ty), mask=mask2d)
    tl.store(Y + offs + HALF, o2.to(out_ty), mask=mask2d)


@triton.jit
def _norm_scale_rope_rows_bwd(
    X,
    DY,
    DX,
    row0,
    n_rows,
    c1,
    c2,
    s1,
    s2,
    scale,
    eps,
    BLOCK_ROWS: tl.constexpr,
    HALF: tl.constexpr,
    HAS_ROPE: tl.constexpr,
    APPLY_SCALE: tl.constexpr,
):
    """Row-local VJP of ``_norm_scale_rope_rows_fwd`` for ``n_rows`` rows starting at ``row0``."""
    D: tl.constexpr = 2 * HALF
    rows = tl.arange(0, BLOCK_ROWS)
    cols = tl.arange(0, HALF)
    rmask = rows < n_rows
    mask2d = tl.broadcast_to(rmask[:, None], [BLOCK_ROWS, HALF])
    offs = (row0 + rows)[:, None] * D + cols[None, :]

    x1 = tl.load(X + offs, mask=mask2d, other=0.0).to(tl.float32)
    x2 = tl.load(X + offs + HALF, mask=mask2d, other=0.0).to(tl.float32)
    dy1 = tl.load(DY + offs, mask=mask2d, other=0.0).to(tl.float32)
    dy2 = tl.load(DY + offs + HALF, mask=mask2d, other=0.0).to(tl.float32)

    # rstd recomputed with exactly the forward's operation order.
    ss = _tree_sum_64(x1 * x1, BLOCK_ROWS) + _tree_sum_64(x2 * x2, BLOCK_ROWS)
    rstd = tl.rsqrt(ss / D + eps)
    xh1 = x1 * rstd[:, None]
    xh2 = x2 * rstd[:, None]

    if HAS_ROPE:
        # Transpose of the rotation [[c, -s], [s, c]] applied to the output gradient.
        g1 = dy1 * c1[None, :] + dy2 * s2[None, :]
        g2 = dy2 * c2[None, :] - dy1 * s1[None, :]
    else:
        g1 = dy1
        g2 = dy2
    if APPLY_SCALE:
        g1 = g1 * scale
        g2 = g2 * scale

    # Weightless RMSNorm VJP: dx = rstd * (g - x_hat * mean(g * x_hat)).
    dot = _tree_sum_64(g1 * xh1, BLOCK_ROWS) + _tree_sum_64(g2 * xh2, BLOCK_ROWS)
    coef = dot / D
    dx1 = rstd[:, None] * (g1 - xh1 * coef[:, None])
    dx2 = rstd[:, None] * (g2 - xh2 * coef[:, None])

    out_ty = DX.dtype.element_ty
    tl.store(DX + offs, dx1.to(out_ty), mask=mask2d)
    tl.store(DX + offs + HALF, dx2.to(out_ty), mask=mask2d)


@triton.jit
def _qknorm_rope_fwd_kernel(
    Q,
    K,
    COS,
    SIN,
    OQ,
    OK,
    scale,
    eps,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    BLOCK_HQ: tl.constexpr,
    BLOCK_HK: tl.constexpr,
    HALF: tl.constexpr,
    HAS_ROPE: tl.constexpr,
):
    """One program per token: all HQ query heads and HK key heads of that token."""
    t = tl.program_id(0)
    D: tl.constexpr = 2 * HALF
    cols = tl.arange(0, HALF)
    if HAS_ROPE:
        c1 = tl.load(COS + t * D + cols)
        c2 = tl.load(COS + t * D + HALF + cols)
        s1 = tl.load(SIN + t * D + cols)
        s2 = tl.load(SIN + t * D + HALF + cols)
    else:
        c1 = tl.zeros([HALF], dtype=tl.float32)
        c2 = tl.zeros([HALF], dtype=tl.float32)
        s1 = tl.zeros([HALF], dtype=tl.float32)
        s2 = tl.zeros([HALF], dtype=tl.float32)

    _norm_scale_rope_rows_fwd(
        Q, OQ, t * HQ, HQ, c1, c2, s1, s2, scale, eps, BLOCK_HQ, HALF, HAS_ROPE, True
    )
    _norm_scale_rope_rows_fwd(
        K, OK, t * HK, HK, c1, c2, s1, s2, scale, eps, BLOCK_HK, HALF, HAS_ROPE, False
    )


@triton.jit
def _qknorm_rope_bwd_kernel(
    Q,
    K,
    DOQ,
    DOK,
    COS,
    SIN,
    DQ,
    DK,
    scale,
    eps,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    BLOCK_HQ: tl.constexpr,
    BLOCK_HK: tl.constexpr,
    HALF: tl.constexpr,
    HAS_ROPE: tl.constexpr,
):
    """One program per token: dQ for its HQ query heads and dK for its HK key heads."""
    t = tl.program_id(0)
    D: tl.constexpr = 2 * HALF
    cols = tl.arange(0, HALF)
    if HAS_ROPE:
        c1 = tl.load(COS + t * D + cols)
        c2 = tl.load(COS + t * D + HALF + cols)
        s1 = tl.load(SIN + t * D + cols)
        s2 = tl.load(SIN + t * D + HALF + cols)
    else:
        c1 = tl.zeros([HALF], dtype=tl.float32)
        c2 = tl.zeros([HALF], dtype=tl.float32)
        s1 = tl.zeros([HALF], dtype=tl.float32)
        s2 = tl.zeros([HALF], dtype=tl.float32)

    _norm_scale_rope_rows_bwd(
        Q, DOQ, DQ, t * HQ, HQ, c1, c2, s1, s2, scale, eps, BLOCK_HQ, HALF, HAS_ROPE, True
    )
    _norm_scale_rope_rows_bwd(
        K, DOK, DK, t * HK, HK, c1, c2, s1, s2, scale, eps, BLOCK_HK, HALF, HAS_ROPE, False
    )


# --------------------------------------------------------------------------------------
# Launchers
# --------------------------------------------------------------------------------------


def _check_inputs(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor | None, sin: torch.Tensor | None
) -> tuple[int, int, int, int]:
    if q.dim() != 3 or k.dim() != 3:
        raise ValueError(f"q and k must be (T, H, D); got {tuple(q.shape)} and {tuple(k.shape)}")
    T, HQ, D = q.shape
    TK, HK, DK = k.shape
    if TK != T or DK != D:
        raise ValueError(f"q {tuple(q.shape)} and k {tuple(k.shape)} disagree on T or D")
    if D != HEAD_DIM:
        raise ValueError(f"the strict kernel pins head_dim={HEAD_DIM} (contract O4); got {D}")
    if q.dtype != k.dtype:
        raise ValueError(f"q and k must share a dtype; got {q.dtype} and {k.dtype}")
    if q.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise TypeError(f"unsupported dtype {q.dtype}")
    if q.device.type != "cuda":
        raise RuntimeError("the strict qknorm_scaled_rope kernel requires CUDA tensors")
    if (cos is None) != (sin is None):
        raise ValueError("cos and sin must both be given (local layer) or both None (global)")
    if cos is not None:
        if cos.shape != (T, D) or sin.shape != (T, D):
            raise ValueError(
                f"cos/sin must be ({T}, {D}) rows gathered per token; "
                f"got {tuple(cos.shape)} and {tuple(sin.shape)}"
            )
        if cos.dtype != torch.float32 or sin.dtype != torch.float32:
            raise TypeError("cos/sin must be fp32 (contract O4)")
        if cos.device != q.device or sin.device != q.device:
            raise RuntimeError("cos/sin must live on the same device as q/k")
    return T, HQ, HK, D


def tile_config(HQ: int = 32, HK: int = 2, D: int = HEAD_DIM) -> dict[str, object]:
    """Compile-time launch configuration recorded in ``trace()`` (contract R6)."""
    return {
        "programs": "one per token",
        "BLOCK_HQ": triton.next_power_of_2(HQ),
        "BLOCK_HK": triton.next_power_of_2(HK),
        "HEAD_DIM": D,
        "HALF": D // 2,
        "num_warps": NUM_WARPS,
        "num_stages": NUM_STAGES,
        "enable_fp_fusion": False,
    }


def qknorm_rope_forward_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor | None,
    sin: torch.Tensor | None,
    *,
    eps: float,
    qk_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Strict forward. Returns ``(q_hat, k_hat)`` in the input dtype (bf16 on the contract path)."""
    T, HQ, HK, D = _check_inputs(q, k, cos, sin)
    q = q.contiguous()
    k = k.contiguous()
    oq = torch.empty_like(q)
    ok = torch.empty_like(k)
    if T == 0:
        return oq, ok
    has_rope = cos is not None
    if has_rope:
        cos = cos.contiguous()
        sin = sin.contiguous()
    else:
        cos = sin = oq  # never dereferenced: HAS_ROPE is a compile-time False.
    _qknorm_rope_fwd_kernel[(T,)](
        q,
        k,
        cos,
        sin,
        oq,
        ok,
        float(qk_scale),
        float(eps),
        HQ=HQ,
        HK=HK,
        BLOCK_HQ=triton.next_power_of_2(HQ),
        BLOCK_HK=triton.next_power_of_2(HK),
        HALF=D // 2,
        HAS_ROPE=has_rope,
        num_warps=NUM_WARPS,
        num_stages=NUM_STAGES,
        enable_fp_fusion=False,
    )
    return oq, ok


def qknorm_rope_backward_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    grad_q_hat: torch.Tensor,
    grad_k_hat: torch.Tensor,
    cos: torch.Tensor | None,
    sin: torch.Tensor | None,
    *,
    eps: float,
    qk_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Strict backward. Returns ``(dq, dk)`` in the input dtype; row-local and deterministic."""
    T, HQ, HK, D = _check_inputs(q, k, cos, sin)
    if grad_q_hat.shape != q.shape or grad_k_hat.shape != k.shape:
        raise ValueError("output gradients must match q/k shapes")
    q = q.contiguous()
    k = k.contiguous()
    doq = grad_q_hat.contiguous()
    dok = grad_k_hat.contiguous()
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    if T == 0:
        return dq, dk
    has_rope = cos is not None
    if has_rope:
        cos = cos.contiguous()
        sin = sin.contiguous()
    else:
        cos = sin = dq
    _qknorm_rope_bwd_kernel[(T,)](
        q,
        k,
        doq,
        dok,
        cos,
        sin,
        dq,
        dk,
        float(qk_scale),
        float(eps),
        HQ=HQ,
        HK=HK,
        BLOCK_HQ=triton.next_power_of_2(HQ),
        BLOCK_HK=triton.next_power_of_2(HK),
        HALF=D // 2,
        HAS_ROPE=has_rope,
        num_warps=NUM_WARPS,
        num_stages=NUM_STAGES,
        enable_fp_fusion=False,
    )
    return dq, dk
