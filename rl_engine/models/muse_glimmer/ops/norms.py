# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Muse-Glimmer-30B strict norms: fp32 goldens, strict bf16 ops with deterministic backward, trace.

RFC #501 rows ``normalized_embedding`` (contract O1) and ``fused_post_norm_residual_pre_norm``
(contract O2).

Public strict interface (bf16 in, bf16 out, fp32 compute, one output cast):

* ``normalized_embedding(weight (V,H), token_ids (T,), *, eps=1e-5)`` -> (T,H)
  lookup bf16 row -> fp32 -> weightless RMSNorm (no sqrt(H) scaling) -> bf16.
  Backward: row-local fp32 VJP, then the repo's deterministic sorted-segment embedding backward
  (``rl_engine.backends.shared.triton.embedding.embedding._embedding_grad_weight``) for dweight.
* ``centered_rmsnorm(x, w, *, eps)``  y = xhat * (1 + w)   (input/post_attention/pre_ffn/post_ffn)
* ``weighted_rmsnorm(x, w, *, eps)``  y = xhat * w         (final norm)
* ``residual_add(a, b)``              bf16(a -> fp32 + b -> fp32)
* ``fused_post_norm_residual_pre_norm(branch, resid, w_post, eps_post, w_pre, eps_pre,
  *, pre_form="centered")`` -> (resid_new, pre_normed): the fusion [F1] of
  ``resid_new = residual_add(resid, centered_rmsnorm(branch, w_post, eps_post))`` and
  ``pre_normed = {centered|weighted}_rmsnorm(resid_new, w_pre, eps_pre)``; bit-equal to that
  composition.

Every strict function has a ``*_fp32`` golden (hand-written fixed-order torch, fp32 output).

Numerics shared by golden and kernel (see ``rl_engine/models/muse_glimmer/triton/norms.py``):
x -> fp32; ``ss = sum(x*x)`` fixed-order; ``ms = ss / H``; ``inv = 1 / sqrt(ms + eps)`` with IEEE
round-to-nearest sqrt and division (the chosen "rsqrt form"; NOT approximate rsqrt/pow(-0.5));
``xhat = x * inv``; gain ``(1 + w)``/``w`` applied in fp32; cast once. The golden's ``ss`` is
summed as 32-wide chunk partials then over the H/32 partials (fixed shapes -> fixed torch
kernels); the kernel's ``ss`` is ``tl.sum`` over a zero-padded ``BLOCK_H`` block. They agree to
fp32 rounding, not bitwise; everything after ``ss`` is a sequence of IEEE-RN ops on both sides,
provided the torch side divides by a TENSOR (:func:`div_ieee`): in torch CUDA, ``t / python_float``
is ``t * (1 / python_float)`` (two roundings), not a division.

Backward (all strict ops): row-local fp32 VJP of y = xhat * g; weight gradients are fp32 per-row
partials ``dy * xhat`` reduced by ``colsum_fixed_order`` (left fold over 32-row blocks, then over
block partials; no atomics) and cast to the weight dtype once. The fused backward keeps the whole
chain (pre-norm VJP -> residual gradient -> post-norm VJP) in fp32 with no intermediate bf16
rounding; the unfused composition would round the residual gradient to bf16 between the two norms.
"""

from __future__ import annotations

from typing import Any

import torch

from rl_engine.backends.shared.triton.embedding.embedding import _embedding_grad_weight
from rl_engine.models.muse_glimmer.triton import norms as _k
from rl_engine.ops.autograd.backward_runtime import record_backward

HIDDEN = 6656
VOCAB = 202048
RMS_NORM_EPS = 1e-5  # input_layernorm, pre_feedforward_layernorm, final norm, embedding
POST_NORM_EPS = 1e-8  # post_attention_layernorm, post_feedforward_layernorm

_PRE_FORMS = {"centered": _k.FORM_CENTERED, "weighted": _k.FORM_WEIGHTED}
_KERNEL_MODULE = "rl_engine.models.muse_glimmer.triton.norms"


# --------------------------------------------------------------------------------------------- #
# fp32 goldens (hand-written, fixed-order torch; no matmul)
# --------------------------------------------------------------------------------------------- #


def _sum_squares_fp32(x32: torch.Tensor) -> torch.Tensor:
    """Fixed-order fp32 sum of squares over the last dim.

    Order: 32-wide chunk partial sums, then a sum over the H/32 partials (both torch fp32
    reductions over fixed-size inner dims, so the reduction kernels do not depend on the batch
    layout; see ``rl_engine.reference.norm.rms_norm.shape_invariant_rstd``). Falls back to a plain
    sum when H is not a multiple of 32.
    """

    hidden = x32.shape[-1]
    sq = x32 * x32
    if hidden % 32 != 0:
        return sq.sum(dim=-1)
    partial = sq.reshape(*sq.shape[:-1], hidden // 32, 32).sum(dim=-1)
    return partial.sum(dim=-1)


def div_ieee(t: torch.Tensor, divisor: float) -> torch.Tensor:
    """IEEE round-to-nearest ``t / divisor`` on every device (the kernels' ``div_rn``).

    The divisor is a 0-dim tensor on ``t``'s device, never a Python number: ATen's CUDA division
    by a CPU scalar is ``t * (1 / divisor)`` (two roundings; it differs from ``div_rn`` in ~46 %
    of the rows at H=6656), while CPU division by a scalar is a true division, so ``t / float``
    would make the golden device-dependent.
    """

    return torch.div(t, torch.full((), float(divisor), dtype=t.dtype, device=t.device))


def inv_rms_fp32(x32: torch.Tensor, eps: float) -> torch.Tensor:
    """inv = 1 / sqrt(sum(x^2) / H + eps) in fp32 with IEEE-RN sqrt and division. Shape (...,).

    ``ms = div_ieee(ss, H)`` (true division, see :func:`div_ieee`); ``1.0 / t`` is
    ``reciprocal(t) * 1.0`` in torch, i.e. one IEEE-RN division on both devices.
    """

    if x32.dtype != torch.float32:
        raise TypeError("inv_rms_fp32 expects an fp32 tensor")
    ms = div_ieee(_sum_squares_fp32(x32), x32.shape[-1])
    return 1.0 / torch.sqrt(ms + float(eps))


def _rmsnorm_fp32(x: torch.Tensor, weight: torch.Tensor | None, *, eps: float, form: int):
    x32 = x.float()
    inv = inv_rms_fp32(x32, eps).unsqueeze(-1)
    xhat = x32 * inv
    if form == _k.FORM_CENTERED:
        return xhat * (1.0 + weight.float())
    if form == _k.FORM_WEIGHTED:
        return xhat * weight.float()
    return xhat


def normalized_embedding_fp32(
    weight: torch.Tensor, token_ids: torch.Tensor, *, eps: float = RMS_NORM_EPS
) -> torch.Tensor:
    """Golden O1: rows = weight[token_ids] (exact gather) -> fp32 weightless RMSNorm -> fp32."""

    _check_embedding_inputs(weight, token_ids, strict_dtype=False)
    rows = weight.index_select(0, token_ids.reshape(-1).to(torch.long))
    out = _rmsnorm_fp32(rows, None, eps=eps, form=_k.FORM_WEIGHTLESS)
    return out.reshape(*token_ids.shape, weight.shape[1])


def centered_rmsnorm_fp32(x: torch.Tensor, w: torch.Tensor, *, eps: float) -> torch.Tensor:
    """Golden: y = (x * inv) * (1 + w) in fp32."""

    _check_norm_inputs(x, w, strict_dtype=False)
    return _rmsnorm_fp32(x, w, eps=eps, form=_k.FORM_CENTERED)


def weighted_rmsnorm_fp32(x: torch.Tensor, w: torch.Tensor, *, eps: float) -> torch.Tensor:
    """Golden (final norm): y = (x * inv) * w in fp32."""

    _check_norm_inputs(x, w, strict_dtype=False)
    return _rmsnorm_fp32(x, w, eps=eps, form=_k.FORM_WEIGHTED)


def residual_add_fp32(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Golden: a -> fp32 + b -> fp32 (single fp32 add)."""

    return a.float() + b.float()


def declared_cast_bf16(t32: torch.Tensor) -> torch.Tensor:
    """A contract-declared "-> bf16" rounding point inside an fp32 golden.

    Forward: the fp32 tensor holding exactly the bf16-rounded values (both fp32 ops below are
    exact). Backward: identity, so autograd through the golden keeps the VJP chain in fp32 like
    the strict kernels do, instead of rounding intermediate gradients to bf16 as autograd would
    through a plain ``.to(torch.bfloat16)``.
    """

    rounded = t32.to(torch.bfloat16).to(torch.float32)
    return t32 + (rounded - t32).detach()


def fused_post_norm_residual_pre_norm_fp32(
    branch: torch.Tensor,
    resid: torch.Tensor,
    w_post: torch.Tensor,
    eps_post: float,
    w_pre: torch.Tensor,
    eps_pre: float,
    *,
    pre_form: str = "centered",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Golden [F1] with fp32 outputs.

    The two contract-declared intermediate casts are kept because they define the residual
    stream's semantics (O2: post_norm(branch) -> bf16; r' = bf16(r + post)); only the final
    output cast is dropped. ``resid_new`` is therefore an fp32 tensor holding bf16-representable
    values, and ``pre_normed`` is the full-fp32 norm of that bf16 residual. The casts use
    :func:`declared_cast_bf16`, so autograd through this golden is the fp32 VJP chain the strict
    fused backward implements (no bf16 rounding of the residual-stream gradient).
    """

    form = _pre_form_code(pre_form)
    _check_norm_inputs(branch, w_post, strict_dtype=False)
    _check_norm_inputs(resid, w_pre, strict_dtype=False)
    post = declared_cast_bf16(centered_rmsnorm_fp32(branch, w_post, eps=eps_post))
    resid_new = declared_cast_bf16(residual_add_fp32(resid, post))
    pre = _rmsnorm_fp32(resid_new, w_pre, eps=eps_pre, form=form)
    return resid_new, pre


# --------------------------------------------------------------------------------------------- #
# Validation helpers
# --------------------------------------------------------------------------------------------- #


def _pre_form_code(pre_form: str) -> int:
    try:
        return _PRE_FORMS[pre_form]
    except KeyError:
        raise ValueError(f"pre_form must be one of {sorted(_PRE_FORMS)}, got {pre_form!r}")


def _check_norm_inputs(x: torch.Tensor, w: torch.Tensor | None, *, strict_dtype: bool) -> None:
    if x.ndim < 1:
        raise ValueError("norm input must have a hidden dimension")
    if w is not None and (w.ndim != 1 or w.shape[0] != x.shape[-1]):
        raise ValueError(
            f"weight must be 1-D of size x.shape[-1]={x.shape[-1]}, got {tuple(w.shape)}"
        )
    if strict_dtype:
        if x.dtype != torch.bfloat16:
            raise TypeError(f"strict norm input must be bf16, got {x.dtype}")
        if w is not None and w.dtype != torch.bfloat16:
            raise TypeError(f"strict norm weight must be bf16, got {w.dtype}")
        if not x.is_cuda or (w is not None and w.device != x.device):
            raise RuntimeError("strict norm requires CUDA tensors on one device")


def _check_embedding_inputs(weight: torch.Tensor, token_ids: torch.Tensor, *, strict_dtype: bool):
    if weight.ndim != 2:
        raise ValueError(f"embedding weight must be (V, H), got {tuple(weight.shape)}")
    if token_ids.is_floating_point() or token_ids.is_complex() or token_ids.dtype == torch.bool:
        raise TypeError("token_ids must use an integer dtype")
    if token_ids.device != weight.device:
        raise RuntimeError("token_ids and weight must share a device")
    if strict_dtype:
        if weight.dtype != torch.bfloat16:
            raise TypeError(f"strict embedding weight must be bf16, got {weight.dtype}")
        if not weight.is_cuda:
            raise RuntimeError("strict embedding requires CUDA tensors")


def _as_rows(x: torch.Tensor) -> torch.Tensor:
    return x.contiguous().reshape(-1, x.shape[-1])


# --------------------------------------------------------------------------------------------- #
# Strict autograd functions
# --------------------------------------------------------------------------------------------- #


class _StrictRMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x2d: torch.Tensor, w: torch.Tensor, eps: float, form: int) -> torch.Tensor:
        y, inv = _k.norm_forward(x2d, w, eps=eps, form=form)
        ctx.save_for_backward(x2d, w, inv)
        ctx.form = form
        return y

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        x2d, w, inv = ctx.saved_tensors
        need_dw = bool(ctx.needs_input_grad[1])
        dx, pdw = _k.norm_backward(
            dy.contiguous(), x2d, w, inv, form=ctx.form, dx_dtype=x2d.dtype, need_dw=need_dw
        )
        dw = _k.colsum_fixed_order(pdw).to(w.dtype) if need_dw else None
        record_backward(
            "centered_rmsnorm" if ctx.form == _k.FORM_CENTERED else "weighted_rmsnorm",
            kernel_id=f"{_KERNEL_MODULE}._norm_bwd_kernel+{_KERNEL_MODULE}._colsum_kernel",
            impl="triton_row_fp32_vjp_colsum_fixed_order_dw",
            family="triton",
        )
        return dx, dw, None, None


class _StrictNormalizedEmbedding(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight: torch.Tensor, ids: torch.Tensor, eps: float) -> torch.Tensor:
        y, inv = _k.norm_forward(
            weight, None, eps=eps, form=_k.FORM_WEIGHTLESS, token_ids=ids, out_dtype=weight.dtype
        )
        ctx.save_for_backward(weight, ids, inv)
        return y

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        weight, ids, inv = ctx.saved_tensors
        dweight = None
        if ctx.needs_input_grad[0]:
            # Row-local fp32 VJP of the weightless norm, rows kept in fp32 so the segment sum
            # rounds to the weight dtype exactly once.
            dx_rows, _ = _k.norm_backward(
                dy.contiguous(),
                weight,
                None,
                inv,
                form=_k.FORM_WEIGHTLESS,
                token_ids=ids,
                dx_dtype=torch.float32,
                need_dw=False,
            )
            dweight = _embedding_grad_weight(
                ids,
                dx_rows.contiguous(),
                weight_shape=(int(weight.shape[0]), int(weight.shape[1])),
                weight_dtype=weight.dtype,
            )
            record_backward(
                "normalized_embedding",
                kernel_id=(
                    f"{_KERNEL_MODULE}._norm_bwd_kernel"
                    "+rl_engine.backends.shared.triton.embedding.embedding._embedding_bwd"
                ),
                impl="triton_row_fp32_vjp_sorted_segment_dweight",
                family="triton",
            )
        return dweight, None, None


class _StrictFusedAddNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, branch, resid, w_post, eps_post, w_pre, eps_pre, form):
        rnew, y, inv_post, inv_pre = _k.fused_add_norm_forward(
            branch, resid, w_post, eps_post, w_pre, eps_pre, pre_form=form
        )
        ctx.save_for_backward(branch, rnew, w_post, w_pre, inv_post, inv_pre)
        ctx.form = form
        return rnew, y

    @staticmethod
    def backward(ctx, d_rnew: torch.Tensor | None, d_pre: torch.Tensor | None):
        branch, rnew, w_post, w_pre, inv_post, inv_pre = ctx.saved_tensors
        d_rnew = d_rnew.contiguous() if d_rnew is not None else None
        d_pre = d_pre.contiguous() if d_pre is not None else None
        d_branch, d_resid, pdw_post, pdw_pre = _k.fused_add_norm_backward(
            d_rnew, d_pre, branch, rnew, w_post, w_pre, inv_post, inv_pre, pre_form=ctx.form
        )
        dw_post = _k.colsum_fixed_order(pdw_post).to(w_post.dtype)
        dw_pre = _k.colsum_fixed_order(pdw_pre).to(w_pre.dtype)
        record_backward(
            "fused_post_norm_residual_pre_norm",
            kernel_id=(
                f"{_KERNEL_MODULE}._fused_add_norm_bwd_kernel+{_KERNEL_MODULE}._colsum_kernel"
            ),
            impl="triton_fused_row_fp32_vjp_colsum_fixed_order_dw",
            family="triton",
        )
        return d_branch, d_resid, dw_post, None, dw_pre, None, None


# --------------------------------------------------------------------------------------------- #
# Strict public API
# --------------------------------------------------------------------------------------------- #


def normalized_embedding(
    weight: torch.Tensor, token_ids: torch.Tensor, *, eps: float = RMS_NORM_EPS
) -> torch.Tensor:
    """O1 strict forward: (V,H) bf16 table, (...,) ids -> (..., H) bf16; dweight deterministic."""

    _check_embedding_inputs(weight, token_ids, strict_dtype=True)
    ids = token_ids.reshape(-1).to(torch.int64).contiguous()
    if ids.numel():
        valid = ((ids >= 0) & (ids < weight.shape[0])).all()
        torch._assert_async(valid, f"token_ids must be in [0, {weight.shape[0]})")
    y = _StrictNormalizedEmbedding.apply(weight.contiguous(), ids, float(eps))
    return y.reshape(*token_ids.shape, weight.shape[1])


def centered_rmsnorm(x: torch.Tensor, w: torch.Tensor, *, eps: float) -> torch.Tensor:
    """O2 strict forward: y = bf16((x * inv) * (1 + w)), fp32 compute."""

    _check_norm_inputs(x, w, strict_dtype=True)
    y = _StrictRMSNorm.apply(_as_rows(x), w.contiguous(), float(eps), _k.FORM_CENTERED)
    return y.reshape(x.shape)


def weighted_rmsnorm(x: torch.Tensor, w: torch.Tensor, *, eps: float) -> torch.Tensor:
    """Final-norm strict forward: y = bf16((x * inv) * w), fp32 compute."""

    _check_norm_inputs(x, w, strict_dtype=True)
    y = _StrictRMSNorm.apply(_as_rows(x), w.contiguous(), float(eps), _k.FORM_WEIGHTED)
    return y.reshape(x.shape)


def residual_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Strict residual add: both bf16 operands -> fp32, one add, -> bf16 (elementwise torch)."""

    if a.dtype != torch.bfloat16 or b.dtype != torch.bfloat16:
        raise TypeError("residual_add expects bf16 operands")
    if a.shape != b.shape:
        raise ValueError(f"residual_add shape mismatch {tuple(a.shape)} vs {tuple(b.shape)}")
    return (a.float() + b.float()).to(torch.bfloat16)


def fused_post_norm_residual_pre_norm(
    branch: torch.Tensor,
    resid: torch.Tensor,
    w_post: torch.Tensor,
    eps_post: float,
    w_pre: torch.Tensor,
    eps_pre: float,
    *,
    pre_form: str = "centered",
) -> tuple[torch.Tensor, torch.Tensor]:
    """[F1] strict forward -> (resid_new bf16, pre_normed bf16); autograd for all four tensors.

    Bit-equal to ``residual_add(resid, centered_rmsnorm(branch, w_post, eps_post))`` followed by
    ``centered_rmsnorm`` / ``weighted_rmsnorm`` of the result (same kernels, same block layout).
    """

    form = _pre_form_code(pre_form)
    devices = {t.device for t in (branch, resid, w_post, w_pre)}
    if len(devices) != 1:
        raise RuntimeError(
            "fused_post_norm_residual_pre_norm: branch, resid, w_post and w_pre must share one "
            f"device, got {sorted(str(d) for d in devices)}"
        )
    _check_norm_inputs(branch, w_post, strict_dtype=True)
    _check_norm_inputs(resid, w_pre, strict_dtype=True)
    if branch.shape != resid.shape:
        raise ValueError(
            f"branch/resid shape mismatch {tuple(branch.shape)} vs {tuple(resid.shape)}"
        )
    rnew, y = _StrictFusedAddNorm.apply(
        _as_rows(branch),
        _as_rows(resid),
        w_post.contiguous(),
        float(eps_post),
        w_pre.contiguous(),
        float(eps_pre),
        form,
    )
    return rnew.reshape(resid.shape), y.reshape(branch.shape)


# --------------------------------------------------------------------------------------------- #
# trace()  (contract R4 / R6)
# --------------------------------------------------------------------------------------------- #

_ROW_REDUCTION = (
    "one row per program; x->fp32 zero-padded to BLOCK_H; ss=tl.sum(x*x) (Triton fixed tree for "
    "(BLOCK_H,num_warps)); ms=div_rn(ss,H); inv=div_rn(1,sqrt_rn(ms+eps)); xhat=x*inv; "
    "{gain}; one cast to bf16 at the store; enable_fp_fusion=False"
)
_DW_REDUCTION = (
    "dw: fp32 rows dy*xhat per program (no atomics) -> colsum_fixed_order: left fold over "
    "blocks of {rows} rows, then over block partials until one row remains; cast to bf16 once"
)
_DX_REDUCTION = "dx: dxhat=dy*g; dot=tl.sum(dxhat*xhat); dx=inv*(dxhat-xhat*div_rn(dot,H)) -> bf16"


def _op_trace(op: str, row: str, gain: str, hidden: int, extra: dict[str, Any]) -> dict[str, Any]:
    cfg = _k.row_launch_config(hidden)
    out = {
        "op": op,
        "rfc_row": row,
        "dtype_in": "bfloat16",
        "dtype_acc": "float32",
        "dtype_out": "bfloat16",
        "reduction_order": _ROW_REDUCTION.format(gain=gain),
        "tile_config": {
            "BLOCK_H": cfg["BLOCK_H"],
            "num_warps": cfg["num_warps"],
            "num_stages": cfg["num_stages"],
            "enable_fp_fusion": False,
            "programs": "T (one per row)",
        },
        "backward": {
            "dx": _DX_REDUCTION,
            "dw": _DW_REDUCTION.format(rows=_k.COLSUM_ROWS),
            "colsum_config": _k.colsum_launch_config(),
        },
        "kernels": [f"{_KERNEL_MODULE}._norm_fwd_kernel", f"{_KERNEL_MODULE}._norm_bwd_kernel"],
    }
    out.update(extra)
    return out


def trace(op: str | None = None, *, hidden: int = HIDDEN) -> dict[str, Any]:
    """Provenance per contract R4: ``{op, dtype_in, dtype_acc, dtype_out, reduction_order,
    tile_config, ...}``. ``trace()`` returns all ops keyed by name; ``trace(name)`` one op."""

    traces = {
        "normalized_embedding": _op_trace(
            "normalized_embedding",
            "normalized_embedding",
            "y=xhat (weightless, no sqrt(H) scaling)",
            hidden,
            {
                "eps": RMS_NORM_EPS,
                "dtype_in": "bfloat16 table, int64 ids",
                "backward": {
                    "dx": _DX_REDUCTION.replace("-> bf16", "kept fp32"),
                    "dweight": (
                        "rl_engine.backends.shared.triton.embedding.embedding."
                        "_embedding_grad_weight: stable argsort by token id, one program per "
                        "(sorted position, 128-col block), segment start sums its fp32 rows in "
                        "sorted order, cast to bf16 once; no atomics"
                    ),
                },
            },
        ),
        "centered_rmsnorm": _op_trace(
            "centered_rmsnorm",
            "fused_post_norm_residual_pre_norm",
            "y=xhat*(1+w)",
            hidden,
            {"eps": {"input/pre_feedforward": RMS_NORM_EPS, "post_*": POST_NORM_EPS}},
        ),
        "weighted_rmsnorm": _op_trace(
            "weighted_rmsnorm",
            "fused_post_norm_residual_pre_norm",
            "y=xhat*w",
            hidden,
            {"eps": RMS_NORM_EPS},
        ),
        "residual_add": {
            "op": "residual_add",
            "rfc_row": "fused_post_norm_residual_pre_norm",
            "dtype_in": "bfloat16",
            "dtype_acc": "float32",
            "dtype_out": "bfloat16",
            "reduction_order": "elementwise: bf16(a->fp32 + b->fp32), torch eager",
            "tile_config": {"kernel": "torch elementwise"},
        },
        "fused_post_norm_residual_pre_norm": _op_trace(
            "fused_post_norm_residual_pre_norm",
            "fused_post_norm_residual_pre_norm",
            "post=bf16(bhat*(1+w_post)); r'=bf16(r+post); y=r'hat*(1+w_pre) or r'hat*w_pre",
            hidden,
            {
                "eps": {"post": POST_NORM_EPS, "pre": RMS_NORM_EPS},
                "kernels": [
                    f"{_KERNEL_MODULE}._fused_add_norm_fwd_kernel",
                    f"{_KERNEL_MODULE}._fused_add_norm_bwd_kernel",
                ],
                "backward": {
                    "chain": (
                        "g_r = pre_vjp(d_pre) + d_rnew (fp32, no intermediate rounding); "
                        "d_resid = bf16(g_r); d_branch = bf16(post_vjp(g_r))"
                    ),
                    "dw": _DW_REDUCTION.format(rows=_k.COLSUM_ROWS),
                    "colsum_config": _k.colsum_launch_config(),
                },
            },
        ),
    }
    if op is None:
        return traces
    try:
        return traces[op]
    except KeyError:
        raise KeyError(f"unknown op {op!r}; known: {sorted(traces)}")


__all__ = [
    "HIDDEN",
    "POST_NORM_EPS",
    "RMS_NORM_EPS",
    "VOCAB",
    "centered_rmsnorm",
    "centered_rmsnorm_fp32",
    "declared_cast_bf16",
    "div_ieee",
    "fused_post_norm_residual_pre_norm",
    "fused_post_norm_residual_pre_norm_fp32",
    "inv_rms_fp32",
    "normalized_embedding",
    "normalized_embedding_fp32",
    "residual_add",
    "residual_add_fp32",
    "trace",
    "weighted_rmsnorm",
    "weighted_rmsnorm_fp32",
]
