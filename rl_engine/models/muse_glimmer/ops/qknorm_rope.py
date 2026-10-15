# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Contract O4 ``qknorm_scaled_rope`` for the Muse-Glimmer-30B strict POC (RFC #501 row
``qknorm_scaled_rope``; trace nodes ``layers.i.q_norm``, ``layers.i.k_norm``,
``layers.i.rope_q``, ``layers.i.rope_k``).

Per head (fp32 arithmetic, bf16 in / bf16 out, ONE rounding at the end)::

    x_hat = x * (mean(x^2) + eps) ** -0.5        # weightless RMSNorm, eps = rms_norm_eps = 1e-5
    q_hat = q_hat * qk_scale                     # 3.87, fp32, queries only (keys are not scaled)
    local layers:  y = x_hat * cos + rotate_half(x_hat) * sin      # all 128 channels, fp32
    global layers: y = x_hat                                       # NoPE: cos is None
    -> bf16

Public interface (imported by the reference-model and attention agents)
    ``rope_table(positions, *, theta, dim, device) -> (cos, sin)`` fp32 ``(T, dim)``
    ``qknorm_scaled_rope(q, k, cos, sin, *, eps, qk_scale)`` strict Triton op, bf16 -> bf16,
    differentiable (deterministic row-local Triton backward)
    ``qknorm_scaled_rope_fp32(...)`` fixed-order torch golden, fp32 outputs (autograd gives the
    fp32 reference backward)
    ``qknorm_scaled_rope_fp64(...)`` the same golden in fp64 (independent high-precision check)
    ``trace()`` provenance dict (contract R4)

Reduction order of the golden: the per-head mean of squares is a binary tree over the 128
channels that pairs ``(2i, 2i+1)`` at every level (7 levels); every op is elementwise so the
golden is bitwise batch invariant on any device.  The Triton kernel implements the SAME tree
and the same op sequence without FMA contraction (see ``triton/qknorm_rope.py``), so on CUDA
``qknorm_scaled_rope(q, k, ...) == qknorm_scaled_rope_fp32(q, k, ...).to(bf16)`` bitwise; the
contract gate (G1) nevertheless only requires the ws1 elementwise tolerances.
"""

from __future__ import annotations

from typing import Any

import torch

from rl_engine.models.muse_glimmer.triton.qknorm_rope import (
    HEAD_DIM,
    qknorm_rope_backward_triton,
    qknorm_rope_forward_triton,
    tile_config,
)

ROPE_THETA = 500000.0
RMS_NORM_EPS = 1e-5
QK_SCALE = 3.87

__all__ = [
    "HEAD_DIM",
    "QK_SCALE",
    "RMS_NORM_EPS",
    "ROPE_THETA",
    "QKNormScaledRopeOp",
    "qknorm_scaled_rope",
    "qknorm_scaled_rope_backward_fp32",
    "qknorm_scaled_rope_fp32",
    "qknorm_scaled_rope_fp64",
    "rope_table",
    "rope_table_fp64",
    "trace",
]


# --------------------------------------------------------------------------------------
# RoPE table - THE definition for both runtimes
# --------------------------------------------------------------------------------------


def _rope_table_in(
    positions: torch.Tensor,
    *,
    theta: float,
    dim: int,
    device: torch.device | str,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    positions = torch.as_tensor(positions)
    if positions.dim() != 1:
        raise ValueError(f"positions must be 1-D (T,), got shape {tuple(positions.shape)}")
    if positions.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"positions must be int32 or int64, got {positions.dtype}")
    if dim % 2 != 0:
        raise ValueError(f"dim must be even, got {dim}")
    device = torch.device(device)
    # inv_freq[i] = theta ** (-(2i)/dim), i in [0, dim/2): the exponent is arange(0, dim, 2)/dim
    # in `dtype`, then a scalar-base torch.pow and a reciprocal (HF's exact formula).
    exponent = torch.arange(0, dim, 2, dtype=dtype, device=device) / dim
    inv_freq = 1.0 / torch.pow(theta, exponent)
    pos = positions.to(device=device, dtype=dtype)
    angle = pos[:, None] * inv_freq[None, :]  # elementwise outer product (T, dim/2)
    emb = torch.cat((angle, angle), dim=-1)  # (T, dim): channel d pairs with d +/- dim/2
    return emb.cos(), emb.sin()


def rope_table(
    positions: torch.Tensor,
    *,
    theta: float = ROPE_THETA,
    dim: int = HEAD_DIM,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """fp32 ``(cos, sin)`` rows of shape ``(T, dim)`` for absolute ``positions`` (int32|int64).

    Definition (all fp32, on ``device``)::

        inv_freq = 1 / theta ** (arange(0, dim, 2) / dim)      # (dim/2,)
        angle    = positions.float()[:, None] * inv_freq[None]  # (T, dim/2), elementwise
        emb      = cat(angle, angle)                             # (T, dim)
        cos, sin = emb.cos(), emb.sin()

    This is the table definition for BOTH runtimes (training and rollout); it is a pure
    function of ``positions`` and deterministic, so callers cache it (e.g. build it once for
    ``arange(max_positions)`` and gather rows with ``positions``).  Bits are specific to the
    device's fp32 ``cos``/``sin``/``pow`` implementation: build it on the device that runs the
    kernel, as the contract says (fixed fp32 torch-CUDA function).
    """
    return _rope_table_in(positions, theta=theta, dim=dim, device=device, dtype=torch.float32)


def rope_table_fp64(
    positions: torch.Tensor,
    *,
    theta: float = ROPE_THETA,
    dim: int = HEAD_DIM,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """fp64 version of :func:`rope_table` (reference only; NOT the runtime table)."""
    return _rope_table_in(positions, theta=theta, dim=dim, device=device, dtype=torch.float64)


# --------------------------------------------------------------------------------------
# Golden (fixed-order torch)
# --------------------------------------------------------------------------------------


def _tree_sum_last(x: torch.Tensor) -> torch.Tensor:
    """Fixed binary-tree sum over the last dim (power of two), pairing (2i, 2i+1) per level."""
    while x.shape[-1] > 1:
        x = x[..., 0::2] + x[..., 1::2]
    return x


def _golden(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor | None,
    sin: torch.Tensor | None,
    *,
    eps: float,
    qk_scale: float,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    if (cos is None) != (sin is None):
        raise ValueError("cos and sin must both be given (local layer) or both None (global)")
    outs = []
    for x, scale in ((q, qk_scale), (k, 1.0)):
        xf = x.to(dtype)
        D = xf.shape[-1]
        if D & (D - 1):
            raise ValueError(f"head_dim must be a power of two, got {D}")
        ms = _tree_sum_last(xf * xf) / D  # (..., 1)
        x_hat = xf * torch.pow(ms + eps, -0.5)
        if scale != 1.0:
            x_hat = x_hat * scale
        if cos is not None:
            c = cos.to(dtype).unsqueeze(-2)  # (T, 1, D) broadcast over heads
            s = sin.to(dtype).unsqueeze(-2)
            half = D // 2
            x1, x2 = x_hat[..., :half], x_hat[..., half:]
            rotated = torch.cat((-x2, x1), dim=-1)
            x_hat = x_hat * c + rotated * s
        outs.append(x_hat)
    return outs[0], outs[1]


def qknorm_scaled_rope_fp32(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor | None,
    sin: torch.Tensor | None,
    *,
    eps: float = RMS_NORM_EPS,
    qk_scale: float = QK_SCALE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """fp32 golden with the module-level semantics; fixed-order torch, fp32 outputs.

    Casts: ``q``/``k`` -> fp32 once; ``cos``/``sin`` used as given (fp32); no output cast.
    Autograd through this function is the fp32 reference backward.
    """
    return _golden(q, k, cos, sin, eps=eps, qk_scale=qk_scale, dtype=torch.float32)


def qknorm_scaled_rope_fp64(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor | None,
    sin: torch.Tensor | None,
    *,
    eps: float = RMS_NORM_EPS,
    qk_scale: float = QK_SCALE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """fp64 golden (pass fp64 tables from :func:`rope_table_fp64` for a fully independent check,
    or the fp32 runtime tables to isolate the arithmetic error of the kernel)."""
    return _golden(q, k, cos, sin, eps=eps, qk_scale=qk_scale, dtype=torch.float64)


def _golden_backward(
    x: torch.Tensor,
    dy: torch.Tensor,
    cos: torch.Tensor | None,
    sin: torch.Tensor | None,
    *,
    eps: float,
    scale: float,
    dtype: torch.dtype,
) -> torch.Tensor:
    xf = x.to(dtype)
    g = dy.to(dtype)
    D = xf.shape[-1]
    half = D // 2
    ms = _tree_sum_last(xf * xf) / D
    rstd = torch.pow(ms + eps, -0.5)
    x_hat = xf * rstd
    if cos is not None:
        c = cos.to(dtype).unsqueeze(-2)
        s = sin.to(dtype).unsqueeze(-2)
        dy1, dy2 = g[..., :half], g[..., half:]
        c1, c2 = c[..., :half], c[..., half:]
        s1, s2 = s[..., :half], s[..., half:]
        # Transpose of the rotation: d(out1)/d(x1) = c1, d(out2)/d(x1) = s2, d(out1)/d(x2) = -s1,
        # d(out2)/d(x2) = c2  (same op sequence as the kernel: mul, mul, add/sub).
        g = torch.cat((dy1 * c1 + dy2 * s2, dy2 * c2 - dy1 * s1), dim=-1)
    if scale != 1.0:
        g = g * scale
    coef = _tree_sum_last(g * x_hat) / D
    return rstd * (g - x_hat * coef)


def qknorm_scaled_rope_backward_fp32(
    q: torch.Tensor,
    k: torch.Tensor,
    grad_q_hat: torch.Tensor,
    grad_k_hat: torch.Tensor,
    cos: torch.Tensor | None,
    sin: torch.Tensor | None,
    *,
    eps: float = RMS_NORM_EPS,
    qk_scale: float = QK_SCALE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Hand-written fixed-order fp32 VJP (the formula the Triton backward implements).

    ``g = R^T dy`` (inverse rotation), ``g *= qk_scale`` (Q only),
    ``dx = rstd * (g - x_hat * mean(g * x_hat))`` with the (2i, 2i+1) tree for both reductions.
    fp32 outputs.  Autograd through :func:`qknorm_scaled_rope_fp32` is the independent check.
    """
    dq = _golden_backward(
        q, grad_q_hat, cos, sin, eps=eps, scale=qk_scale, dtype=torch.float32
    )
    dk = _golden_backward(k, grad_k_hat, cos, sin, eps=eps, scale=1.0, dtype=torch.float32)
    return dq, dk


# --------------------------------------------------------------------------------------
# Strict op (Triton) with deterministic backward
# --------------------------------------------------------------------------------------


class _QKNormScaledRopeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, cos, sin, eps, qk_scale):  # type: ignore[override]
        q_hat, k_hat = qknorm_rope_forward_triton(q, k, cos, sin, eps=eps, qk_scale=qk_scale)
        ctx.save_for_backward(q, k, cos, sin)
        ctx.eps = eps
        ctx.qk_scale = qk_scale
        return q_hat, k_hat

    @staticmethod
    def backward(ctx, grad_q_hat, grad_k_hat):  # type: ignore[override]
        q, k, cos, sin = ctx.saved_tensors
        if grad_q_hat is None:
            grad_q_hat = torch.zeros_like(q)
        if grad_k_hat is None:
            grad_k_hat = torch.zeros_like(k)
        dq, dk = qknorm_rope_backward_triton(
            q, k, grad_q_hat, grad_k_hat, cos, sin, eps=ctx.eps, qk_scale=ctx.qk_scale
        )
        try:  # provenance for the WS1 backward runtime snapshot (optional)
            from rl_engine.ops.autograd.backward_runtime import record_backward

            record_backward(
                "qknorm_scaled_rope",
                kernel_id=(
                    "rl_engine.models.muse_glimmer.triton.qknorm_rope._qknorm_rope_bwd_kernel"
                ),
                impl="triton_row_local_fp32_vjp_inverse_rotation_then_rmsnorm",
                family="triton",
            )
        except Exception:  # pragma: no cover - provenance must never break training
            pass
        if not ctx.needs_input_grad[0]:
            dq = None
        if not ctx.needs_input_grad[1]:
            dk = None
        return dq, dk, None, None, None, None


def qknorm_scaled_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor | None,
    sin: torch.Tensor | None,
    *,
    eps: float = RMS_NORM_EPS,
    qk_scale: float = QK_SCALE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Strict O4 op: ``(q (T,32,128) bf16, k (T,2,128) bf16, cos/sin (T,128) fp32 | None)``
    -> ``(q_hat bf16, k_hat bf16)``.  ``cos is None`` selects the global (NoPE) layer path.

    Differentiable w.r.t. ``q`` and ``k`` with a deterministic, batch-invariant Triton backward.
    """
    return _QKNormScaledRopeFunction.apply(q, k, cos, sin, float(eps), float(qk_scale))


def trace(HQ: int = 32, HK: int = 2, D: int = HEAD_DIM) -> dict[str, Any]:
    """Provenance dict (contract R4): op, dtypes, reduction order, tile constants."""
    return {
        "op": "qknorm_scaled_rope",
        "row": "qknorm_scaled_rope",
        "nodes": ["q_norm", "k_norm", "rope_q", "rope_k"],
        "dtype_in": "bfloat16",
        "dtype_acc": "float32",
        "dtype_out": "bfloat16",
        "dtype_table": "float32",
        "cast_points": [
            "bf16->fp32 at load",
            "fp32->bf16 once at store (after norm, scale and rotation)",
        ],
        "reduction_order": (
            "per head row inside one program: ss = binary tree over 128 channels pairing "
            "(2i, 2i+1) at each of 7 levels (6 levels per 64-channel half via tl.reshape+tl.split, "
            "then half0 + half1), identical to the golden's _tree_sum_last; "
            "rstd = rsqrt(ss/128 + eps); backward dot uses the same tree; "
            "no FMA contraction (enable_fp_fusion=False)"
        ),
        "rotation": "rotate_half over all 128 channels, fp32, cos/sin fp32 table rows",
        "scale": {"q": QK_SCALE, "k": 1.0, "applied": "fp32 after norm, before rotation"},
        "eps": RMS_NORM_EPS,
        "rope_theta": ROPE_THETA,
        "tile_config": tile_config(HQ, HK, D),
        "backward": (
            "row-local fp32 VJP: inverse rotation (orthogonal transpose), q-scale, "
            "weightless RMSNorm VJP dx = rstd*(g - x_hat*mean(g*x_hat)); rstd recomputed; "
            "-> bf16 once"
        ),
        "golden": (
            "qknorm_scaled_rope_fp32: fixed (2i,2i+1) binary tree over 128 channels, "
            "torch.pow(ms+eps,-0.5), elementwise ops only"
        ),
    }


class QKNormScaledRopeOp:
    """Op-object view of the module functions (forward / forward_fp32 / trace), R4 style."""

    # Repo convention for norm-like ops (operator_specs "qk_norm"); the tests nevertheless gate
    # on the tighter ws1 *elementwise* rows (bf16 atol 0.02 / rtol 0.016), which pass.
    op_class = "reduction"

    def __init__(self, *, eps: float = RMS_NORM_EPS, qk_scale: float = QK_SCALE) -> None:
        self.eps = float(eps)
        self.qk_scale = float(qk_scale)

    def __call__(self, q, k, cos, sin):
        return self.forward(q, k, cos, sin)

    def forward(self, q, k, cos, sin):
        return qknorm_scaled_rope(q, k, cos, sin, eps=self.eps, qk_scale=self.qk_scale)

    def forward_fp32(self, q, k, cos, sin):
        return qknorm_scaled_rope_fp32(q, k, cos, sin, eps=self.eps, qk_scale=self.qk_scale)

    def trace(self, HQ: int = 32, HK: int = 2, D: int = HEAD_DIM) -> dict[str, Any]:
        return trace(HQ, HK, D)
