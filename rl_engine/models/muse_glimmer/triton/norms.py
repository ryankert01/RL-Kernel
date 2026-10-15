# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Triton kernels for the Muse-Glimmer-30B strict norms (RFC #501 rows ``normalized_embedding``
and ``fused_add_rmsnorm``; contract O1/O2 in docs/contracts/muse-glimmer-30b-strict-contract.md).

Shared row arithmetic (every kernel in this file, forward and backward):

* One logical row of ``H`` elements is processed end-to-end by ONE program (contract R3), so the
  per-row bits never depend on the batch size, the row's position in the batch, or the grid.
* ``x`` (bf16) -> fp32, zero-padded to ``BLOCK_H = next_pow2(H)`` lanes.
* ``ss = tl.sum(x * x, axis=0)``: Triton's reduction tree over the ``(BLOCK_H,)`` fp32 block with
  the compile-time ``num_warps`` recorded in ``trace()``. The tree is fixed per
  ``(BLOCK_H, num_warps)``; padded lanes contribute exact zeros.
* ``ms = div_rn(ss, H)``; ``inv = div_rn(1, sqrt_rn(ms + eps))`` (IEEE round-to-nearest sqrt and
  division, so ``inv`` is uniquely determined by IEEE-754 given ``ss`` -- this is the
  "pick one rsqrt form" decision: NOT the approximate ``rsqrt``/``pow(-0.5)`` libdevice paths).
* ``xhat = x * inv``; ``y = xhat * (1 + w)`` (centered), ``xhat * w`` (weighted) or ``xhat``
  (weightless). ``w`` (bf16) -> fp32; ``1 + w`` is formed in fp32.
* One cast to the output dtype at the store.
* Kernels are launched with ``enable_fp_fusion=False`` so ``a * b + c`` is never contracted into
  an FMA: the written multiply/add sequence IS the executed one, and the fp32 golden
  (``rl_engine/models/muse_glimmer/ops/norms.py``) performs the same sequence of IEEE ops.

Backward (row-local fp32 VJP of y = (x * inv) * g, g in {1, 1+w, w}):

* ``dxhat = dy * g``; ``dot = tl.sum(dxhat * xhat, axis=0)``;
  ``dx = inv * (dxhat - xhat * div_rn(dot, H))`` -> output dtype.
* per-row weight partial ``pdw = dy * xhat`` (fp32, one row per program, no atomics); rows are
  reduced by :func:`colsum_fixed_order` -- a left fold over blocks of ``COLSUM_ROWS`` rows, then a
  left fold over the block partials (repeated until one row remains). The order depends only on
  the number of rows, never on the launch, so the result is repeat-deterministic.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

FORM_WEIGHTLESS = 0  # y = xhat             (normalized embedding)
FORM_CENTERED = 1  # y = xhat * (1 + w)   (decoder input/post/pre norms)
FORM_WEIGHTED = 2  # y = xhat * w         (final norm)

ROW_NUM_STAGES = 1
COLSUM_ROWS = 32
COLSUM_BLOCK_C = 256
COLSUM_NUM_WARPS = 4
COLSUM_NUM_STAGES = 3


def row_launch_config(hidden: int) -> dict[str, int]:
    """Compile-time launch constants for the one-row-per-program kernels (contract R6)."""

    if hidden <= 0:
        raise ValueError(f"hidden must be positive, got {hidden}")
    block_h = triton.next_power_of_2(int(hidden))
    if block_h > 65536:
        raise ValueError(f"hidden={hidden} exceeds the single-block row kernel limit")
    num_warps = 8 if block_h >= 4096 else 4
    return {"BLOCK_H": block_h, "num_warps": num_warps, "num_stages": ROW_NUM_STAGES}


def colsum_launch_config() -> dict[str, int]:
    return {
        "ROWS": COLSUM_ROWS,
        "BLOCK_C": COLSUM_BLOCK_C,
        "num_warps": COLSUM_NUM_WARPS,
        "num_stages": COLSUM_NUM_STAGES,
    }


# --------------------------------------------------------------------------------------------- #
# Shared row arithmetic
# --------------------------------------------------------------------------------------------- #


@triton.jit
def _row_inv_rms(x, H: tl.constexpr, EPS: tl.constexpr):
    """inv = 1 / sqrt(mean(x^2) + eps) with IEEE-RN sqrt and division (x: (BLOCK_H,) fp32)."""

    ss = tl.sum(x * x, axis=0)
    ms = tl.div_rn(ss, H * 1.0)
    return tl.div_rn(1.0, tl.sqrt_rn(ms + EPS))


@triton.jit
def _apply_gain(xhat, W, offs, mask, FORM: tl.constexpr):
    """y = xhat * g with g = 1 (FORM 0), 1 + w (FORM 1) or w (FORM 2); w -> fp32."""

    if FORM == 1:
        w = tl.load(W + offs, mask=mask, other=0.0).to(tl.float32)
        return xhat * (1.0 + w)
    elif FORM == 2:
        w = tl.load(W + offs, mask=mask, other=0.0).to(tl.float32)
        return xhat * w
    else:
        return xhat


@triton.jit
def _load_gain(W, offs, mask, FORM: tl.constexpr):
    if FORM == 1:
        w = tl.load(W + offs, mask=mask, other=0.0).to(tl.float32)
        return 1.0 + w
    elif FORM == 2:
        return tl.load(W + offs, mask=mask, other=0.0).to(tl.float32)
    else:
        return tl.full((offs.shape[0],), 1.0, tl.float32)


@triton.jit
def _row_vjp(dy, xhat, inv, g, H: tl.constexpr):
    """dx = inv * (dxhat - xhat * mean(dxhat * xhat)), dxhat = dy * g (all fp32)."""

    dxhat = dy * g
    dot = tl.sum(dxhat * xhat, axis=0)
    return inv * (dxhat - xhat * tl.div_rn(dot, H * 1.0))


# --------------------------------------------------------------------------------------------- #
# Forward kernels
# --------------------------------------------------------------------------------------------- #


@triton.jit
def _norm_fwd_kernel(
    X,
    IDS,
    W,
    Y,
    INV,
    H: tl.constexpr,
    V: tl.constexpr,
    EPS: tl.constexpr,
    FORM: tl.constexpr,
    GATHER: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """One row per program. GATHER: the row is X[IDS[row]] (embedding) else X[row]."""

    row = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BLOCK_H)
    mask = offs < H
    if GATHER:
        tok = tl.load(IDS + row).to(tl.int64)
        valid = (tok >= 0) & (tok < V)
        src = tl.where(valid, tok, 0)
        x = tl.load(X + src * H + offs, mask=mask & valid, other=0.0).to(tl.float32)
    else:
        x = tl.load(X + row * H + offs, mask=mask, other=0.0).to(tl.float32)
    inv = _row_inv_rms(x, H, EPS)
    y = _apply_gain(x * inv, W, offs, mask, FORM)
    tl.store(Y + row * H + offs, y.to(Y.dtype.element_ty), mask=mask)
    tl.store(INV + row, inv)


@triton.jit
def _fused_add_norm_fwd_kernel(
    B,
    R,
    WPOST,
    WPRE,
    RNEW,
    Y,
    INV_POST,
    INV_PRE,
    H: tl.constexpr,
    EPS_POST: tl.constexpr,
    EPS_PRE: tl.constexpr,
    PRE_FORM: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """r' = bf16(r + bf16(centered_norm(b, w_post))); y = pre_norm(r', w_pre). One row/program.

    The two intermediate casts are the contract-declared ones (O2: post_norm output -> bf16,
    residual stream kept in bf16); the pre-norm reads the bf16 r' exactly as the unfused
    ``centered_rmsnorm`` would, so this kernel is bit-equal to the composition of the parts.
    """

    row = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BLOCK_H)
    mask = offs < H

    b = tl.load(B + row * H + offs, mask=mask, other=0.0).to(tl.float32)
    inv_post = _row_inv_rms(b, H, EPS_POST)
    post = _apply_gain(b * inv_post, WPOST, offs, mask, 1)
    post = post.to(B.dtype.element_ty).to(tl.float32)  # declared cast: post_norm -> bf16

    r = tl.load(R + row * H + offs, mask=mask, other=0.0).to(tl.float32)
    rnew = (r + post).to(RNEW.dtype.element_ty)  # declared cast: residual stream -> bf16
    tl.store(RNEW + row * H + offs, rnew, mask=mask)

    x = rnew.to(tl.float32)
    inv_pre = _row_inv_rms(x, H, EPS_PRE)
    y = _apply_gain(x * inv_pre, WPRE, offs, mask, PRE_FORM)
    tl.store(Y + row * H + offs, y.to(Y.dtype.element_ty), mask=mask)
    tl.store(INV_POST + row, inv_post)
    tl.store(INV_PRE + row, inv_pre)


# --------------------------------------------------------------------------------------------- #
# Backward kernels
# --------------------------------------------------------------------------------------------- #


@triton.jit
def _norm_bwd_kernel(
    DY,
    X,
    IDS,
    W,
    INV,
    DX,
    PDW,
    H: tl.constexpr,
    V: tl.constexpr,
    FORM: tl.constexpr,
    GATHER: tl.constexpr,
    WRITE_PDW: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Row-local VJP. DX takes the dtype of its buffer; PDW rows are fp32 ``dy * xhat``."""

    row = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BLOCK_H)
    mask = offs < H
    dy = tl.load(DY + row * H + offs, mask=mask, other=0.0).to(tl.float32)
    if GATHER:
        tok = tl.load(IDS + row).to(tl.int64)
        valid = (tok >= 0) & (tok < V)
        src = tl.where(valid, tok, 0)
        x = tl.load(X + src * H + offs, mask=mask & valid, other=0.0).to(tl.float32)
    else:
        x = tl.load(X + row * H + offs, mask=mask, other=0.0).to(tl.float32)
    inv = tl.load(INV + row)
    xhat = x * inv
    g = _load_gain(W, offs, mask, FORM)
    dx = _row_vjp(dy, xhat, inv, g, H)
    tl.store(DX + row * H + offs, dx.to(DX.dtype.element_ty), mask=mask)
    if WRITE_PDW:
        tl.store(PDW + row * H + offs, dy * xhat, mask=mask)


@triton.jit
def _fused_add_norm_bwd_kernel(
    DRNEW,
    DPRE,
    B,
    RNEW,
    WPOST,
    WPRE,
    INV_POST,
    INV_PRE,
    DB,
    DR,
    PDW_POST,
    PDW_PRE,
    H: tl.constexpr,
    PRE_FORM: tl.constexpr,
    HAS_DRNEW: tl.constexpr,
    HAS_DPRE: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Fused VJP, one row per program, fp32 chain with no intermediate rounding:

    g_r = pre_norm_vjp(dpre) + drnew            (grad of the bf16 residual stream r')
    dr  = g_r -> bf16 ; db = post_norm_vjp(g_r) -> bf16
    pdw_pre = dpre * xhat_pre ; pdw_post = g_r * bhat   (fp32 rows, reduced by colsum_fixed_order)
    """

    row = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BLOCK_H)
    mask = offs < H

    g_r = tl.zeros((BLOCK_H,), tl.float32)
    if HAS_DPRE:
        x = tl.load(RNEW + row * H + offs, mask=mask, other=0.0).to(tl.float32)
        inv_pre = tl.load(INV_PRE + row)
        xhat = x * inv_pre
        dpre = tl.load(DPRE + row * H + offs, mask=mask, other=0.0).to(tl.float32)
        g_pre = _load_gain(WPRE, offs, mask, PRE_FORM)
        g_r = _row_vjp(dpre, xhat, inv_pre, g_pre, H)
        tl.store(PDW_PRE + row * H + offs, dpre * xhat, mask=mask)
    if HAS_DRNEW:
        drnew = tl.load(DRNEW + row * H + offs, mask=mask, other=0.0).to(tl.float32)
        g_r = g_r + drnew
    tl.store(DR + row * H + offs, g_r.to(DR.dtype.element_ty), mask=mask)

    b = tl.load(B + row * H + offs, mask=mask, other=0.0).to(tl.float32)
    inv_post = tl.load(INV_POST + row)
    bhat = b * inv_post
    g_post = _load_gain(WPOST, offs, mask, 1)
    db = _row_vjp(g_r, bhat, inv_post, g_post, H)
    tl.store(DB + row * H + offs, db.to(DB.dtype.element_ty), mask=mask)
    tl.store(PDW_POST + row * H + offs, g_r * bhat, mask=mask)


@triton.jit
def _colsum_kernel(
    P,
    OUT,
    n_rows,
    H: tl.constexpr,
    ROWS: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """OUT[i, c] = left fold over P[i*ROWS : (i+1)*ROWS, c] in ascending row order (fp32)."""

    pid_r = tl.program_id(0).to(tl.int64)
    pid_c = tl.program_id(1)
    cols = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    cmask = cols < H
    acc = tl.zeros((BLOCK_C,), tl.float32)
    start = pid_r * ROWS
    for i in range(ROWS):
        r = start + i
        v = tl.load(P + r * H + cols, mask=cmask & (r < n_rows), other=0.0)
        acc = acc + v
    tl.store(OUT + pid_r * H + cols, acc, mask=cmask)


# --------------------------------------------------------------------------------------------- #
# Python launchers
# --------------------------------------------------------------------------------------------- #


def norm_forward(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    *,
    eps: float,
    form: int,
    token_ids: torch.Tensor | None = None,
    out_dtype: torch.dtype = torch.bfloat16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (y, inv). ``token_ids`` switches to the gathered (embedding) row source.

    ``x`` is ``(T, H)`` (direct) or the ``(V, H)`` table (gather); ``weight`` is ``(H,)`` or None
    for the weightless form.
    """

    hidden = int(x.shape[-1])
    if token_ids is not None:
        rows = int(token_ids.numel())
        vocab = int(x.shape[0])
    else:
        rows = int(x.shape[0])
        vocab = 0
    y = torch.empty((rows, hidden), device=x.device, dtype=out_dtype)
    inv = torch.empty((rows,), device=x.device, dtype=torch.float32)
    if rows == 0:
        return y, inv
    cfg = row_launch_config(hidden)
    w_arg = weight if weight is not None else x  # unused pointer for FORM 0
    ids_arg = token_ids if token_ids is not None else inv  # unused pointer when not gathering
    _norm_fwd_kernel[(rows,)](
        x,
        ids_arg,
        w_arg,
        y,
        inv,
        H=hidden,
        V=vocab,
        EPS=float(eps),
        FORM=int(form),
        GATHER=token_ids is not None,
        BLOCK_H=cfg["BLOCK_H"],
        num_warps=cfg["num_warps"],
        num_stages=cfg["num_stages"],
        enable_fp_fusion=False,
    )
    return y, inv


def norm_backward(
    dy: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor | None,
    inv: torch.Tensor,
    *,
    form: int,
    token_ids: torch.Tensor | None = None,
    dx_dtype: torch.dtype = torch.bfloat16,
    need_dw: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Return (dx, pdw_rows). pdw_rows is fp32 ``(T, H)`` or None when ``need_dw`` is False."""

    hidden = int(x.shape[-1])
    rows = int(dy.shape[0])
    vocab = int(x.shape[0]) if token_ids is not None else 0
    dx = torch.empty((rows, hidden), device=x.device, dtype=dx_dtype)
    pdw = torch.empty((rows if need_dw else 0, hidden), device=x.device, dtype=torch.float32)
    if rows == 0:
        return dx, (pdw if need_dw else None)
    cfg = row_launch_config(hidden)
    w_arg = weight if weight is not None else x
    ids_arg = token_ids if token_ids is not None else inv
    _norm_bwd_kernel[(rows,)](
        dy,
        x,
        ids_arg,
        w_arg,
        inv,
        dx,
        pdw if need_dw else dx,
        H=hidden,
        V=vocab,
        FORM=int(form),
        GATHER=token_ids is not None,
        WRITE_PDW=bool(need_dw),
        BLOCK_H=cfg["BLOCK_H"],
        num_warps=cfg["num_warps"],
        num_stages=cfg["num_stages"],
        enable_fp_fusion=False,
    )
    return dx, (pdw if need_dw else None)


def fused_add_norm_forward(
    branch: torch.Tensor,
    resid: torch.Tensor,
    w_post: torch.Tensor,
    eps_post: float,
    w_pre: torch.Tensor,
    eps_pre: float,
    *,
    pre_form: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (resid_new bf16, pre_normed bf16, inv_post fp32, inv_pre fp32)."""

    rows, hidden = int(branch.shape[0]), int(branch.shape[1])
    rnew = torch.empty_like(resid)
    y = torch.empty_like(branch)
    inv_post = torch.empty((rows,), device=branch.device, dtype=torch.float32)
    inv_pre = torch.empty((rows,), device=branch.device, dtype=torch.float32)
    if rows == 0:
        return rnew, y, inv_post, inv_pre
    cfg = row_launch_config(hidden)
    _fused_add_norm_fwd_kernel[(rows,)](
        branch,
        resid,
        w_post,
        w_pre,
        rnew,
        y,
        inv_post,
        inv_pre,
        H=hidden,
        EPS_POST=float(eps_post),
        EPS_PRE=float(eps_pre),
        PRE_FORM=int(pre_form),
        BLOCK_H=cfg["BLOCK_H"],
        num_warps=cfg["num_warps"],
        num_stages=cfg["num_stages"],
        enable_fp_fusion=False,
    )
    return rnew, y, inv_post, inv_pre


def fused_add_norm_backward(
    d_rnew: torch.Tensor | None,
    d_pre: torch.Tensor | None,
    branch: torch.Tensor,
    rnew: torch.Tensor,
    w_post: torch.Tensor,
    w_pre: torch.Tensor,
    inv_post: torch.Tensor,
    inv_pre: torch.Tensor,
    *,
    pre_form: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (d_branch bf16, d_resid bf16, pdw_post fp32 rows, pdw_pre fp32 rows)."""

    rows, hidden = int(branch.shape[0]), int(branch.shape[1])
    d_branch = torch.empty_like(branch)
    d_resid = torch.empty_like(rnew)
    pdw_post = torch.empty((rows, hidden), device=branch.device, dtype=torch.float32)
    pdw_pre = torch.empty((rows, hidden), device=branch.device, dtype=torch.float32)
    if d_pre is None:
        pdw_pre.zero_()
    if rows == 0:
        return d_branch, d_resid, pdw_post, pdw_pre
    cfg = row_launch_config(hidden)
    _fused_add_norm_bwd_kernel[(rows,)](
        d_rnew if d_rnew is not None else branch,
        d_pre if d_pre is not None else branch,
        branch,
        rnew,
        w_post,
        w_pre,
        inv_post,
        inv_pre,
        d_branch,
        d_resid,
        pdw_post,
        pdw_pre,
        H=hidden,
        PRE_FORM=int(pre_form),
        HAS_DRNEW=d_rnew is not None,
        HAS_DPRE=d_pre is not None,
        BLOCK_H=cfg["BLOCK_H"],
        num_warps=cfg["num_warps"],
        num_stages=cfg["num_stages"],
        enable_fp_fusion=False,
    )
    return d_branch, d_resid, pdw_post, pdw_pre


def colsum_fixed_order(partials: torch.Tensor) -> torch.Tensor:
    """Sum fp32 ``(N, H)`` rows into ``(H,)`` in a fixed order independent of the launch.

    Level k folds consecutive blocks of ``COLSUM_ROWS`` rows left-to-right; the level outputs are
    folded the same way until one row remains (N=4096 -> 128 -> 4 -> 1). No atomics; the order is
    a pure function of N, so two runs are bit-identical.
    """

    if partials.ndim != 2 or partials.dtype != torch.float32:
        raise ValueError("colsum_fixed_order expects fp32 (N, H) rows")
    n_rows, hidden = int(partials.shape[0]), int(partials.shape[1])
    if n_rows == 0:
        return torch.zeros((hidden,), device=partials.device, dtype=torch.float32)
    cfg = colsum_launch_config()
    cur = partials.contiguous()
    while n_rows > 1:
        n_out = triton.cdiv(n_rows, cfg["ROWS"])
        out = torch.empty((n_out, hidden), device=cur.device, dtype=torch.float32)
        _colsum_kernel[(n_out, triton.cdiv(hidden, cfg["BLOCK_C"]))](
            cur,
            out,
            n_rows,
            H=hidden,
            ROWS=cfg["ROWS"],
            BLOCK_C=cfg["BLOCK_C"],
            num_warps=cfg["num_warps"],
            num_stages=cfg["num_stages"],
            enable_fp_fusion=False,
        )
        cur, n_rows = out, n_out
    return cur[0]


__all__ = [
    "COLSUM_BLOCK_C",
    "COLSUM_ROWS",
    "FORM_CENTERED",
    "FORM_WEIGHTED",
    "FORM_WEIGHTLESS",
    "colsum_fixed_order",
    "colsum_launch_config",
    "fused_add_norm_backward",
    "fused_add_norm_forward",
    "norm_backward",
    "norm_forward",
    "row_launch_config",
]
