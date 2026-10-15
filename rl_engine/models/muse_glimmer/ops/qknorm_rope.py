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
    ``rope_table(positions, *, theta, dim, device) -> (cos, sin)`` fp32 ``(T, dim)``; THE table
    definition; CUDA devices only (a CPU build differs in ~19% of the elements)
    ``canonical_rope_table(device)`` the cached ``(cos, sin)`` for ``arange(131072)`` (one
    object per device; sha256 pinned in ``CANONICAL_ROPE_TABLE_SHA256``)
    ``rope_rows(positions)`` rows gathered from the canonical table (bitwise == ``rope_table``)
    ``qknorm_scaled_rope(q, k, cos, sin, *, eps, qk_scale, local=None)`` strict Triton op,
    bf16 -> bf16 (other dtypes raise), differentiable (deterministic row-local Triton backward);
    ``local=True/False`` makes the layer kind explicit and is checked against ``cos``/``sin``
    ``qknorm_scaled_rope_at(q, k, positions, *, local | layer_idx)`` the contract's assembly
    signature ``qknorm_scaled_rope(q, k, pos, local?)``: explicit layer kind, canonical table
    ``layer_is_local(i)`` the frozen schedule (layer i is global iff i % 4 == 3, 52 layers)
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

import hashlib
import threading
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
NUM_LAYERS = 52
GLOBAL_LAYER_PERIOD = 4  # layer i is global (NoPE) iff i % 4 == 3
ROPE_TABLE_POSITIONS = 131072  # max_position_embeddings: rows of the canonical table
# sha256(cos.bytes || sin.bytes) of rope_table(arange(131072), device=cuda) (fp32, little
# endian).  Measured on A100 with torch 2.13.0+cu130 (vLLM env) and torch 2.13.0+cu129
# (Megatron env): identical, so both runtimes build the same table.
CANONICAL_ROPE_TABLE_SHA256 = "088dbb8c93a55ce79d189181c46ac06fbdb61dab6c3914218bbe40ed3ea56d9c"

__all__ = [
    "CANONICAL_ROPE_TABLE_SHA256",
    "GLOBAL_LAYER_PERIOD",
    "HEAD_DIM",
    "NUM_LAYERS",
    "QK_SCALE",
    "RMS_NORM_EPS",
    "ROPE_TABLE_POSITIONS",
    "ROPE_THETA",
    "QKNormScaledRopeOp",
    "canonical_rope_table",
    "layer_is_local",
    "qknorm_scaled_rope",
    "qknorm_scaled_rope_at",
    "qknorm_scaled_rope_backward_fp32",
    "qknorm_scaled_rope_fp32",
    "qknorm_scaled_rope_fp64",
    "rope_rows",
    "rope_table",
    "rope_table_fp64",
    "rope_table_sha256",
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

    This is the table definition for BOTH runtimes (training and rollout); it is a pure,
    deterministic and elementwise function of ``positions``, so the row of position ``p`` has
    the same bits whether it is built alone or as part of ``arange(N)``.  Prefer
    :func:`canonical_rope_table` / :func:`rope_rows` (built once per device, cached, fingerprint
    pinned).  Bits are specific to the fp32 ``cos``/``sin``/``pow`` implementation, so the
    contract fixes the builder to torch-CUDA: a non-CUDA ``device`` raises (fail closed; a CPU
    build differs from the CUDA build in ~19 % of the elements at T=4096 and flips ~0.1 % of
    the bf16 q_hat outputs).  Use :func:`rope_table_fp64` for references on any device.
    """
    dev = torch.device(device)
    if dev.type != "cuda":
        raise RuntimeError(
            "contract O4: the RoPE table is built by a fixed fp32 torch-CUDA function; got "
            f"device {dev}. CPU/other-device builds differ in the last bits (~19% of cos/sin "
            "elements). Build on the CUDA device that runs the kernel (canonical_rope_table)."
        )
    return _rope_table_in(positions, theta=theta, dim=dim, device=dev, dtype=torch.float32)


_CANONICAL_TABLES: dict[tuple[int, float, int], tuple[torch.Tensor, torch.Tensor]] = {}
_CANONICAL_SHA: dict[tuple[int, float, int], str] = {}
_CANONICAL_LOCK = threading.Lock()


def _cuda_index(device: torch.device | str | int | None) -> int:
    if device is None:
        return torch.cuda.current_device()
    dev = torch.device("cuda", device) if isinstance(device, int) else torch.device(device)
    if dev.type != "cuda":
        raise RuntimeError(
            f"contract O4: the canonical RoPE table lives on a CUDA device; got {dev}"
        )
    return dev.index if dev.index is not None else torch.cuda.current_device()


def canonical_rope_table(
    device: torch.device | str | int | None = None,
    *,
    theta: float = ROPE_THETA,
    dim: int = HEAD_DIM,
    verify: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """THE cached fp32 ``(cos, sin)`` for positions ``arange(ROPE_TABLE_POSITIONS)``.

    Built once per ``(cuda index, theta, dim)`` with :func:`rope_table` and returned as the SAME
    tensor objects on every call (``canonical_rope_table(d)[0] is canonical_rope_table(d)[0]``);
    callers must not modify them in place.  64 MiB each at the model's dim=128.
    ``verify=True`` checks the bits against the pinned :data:`CANONICAL_ROPE_TABLE_SHA256`
    (defined for the model's theta/dim only) and raises on a mismatch - call it once per process
    in each runtime to prove that both build the same table.
    """
    key = (_cuda_index(device), float(theta), int(dim))
    with _CANONICAL_LOCK:
        tables = _CANONICAL_TABLES.get(key)
        if tables is None:
            dev = torch.device("cuda", key[0])
            pos = torch.arange(ROPE_TABLE_POSITIONS, device=dev)
            cos, sin = rope_table(pos, theta=key[1], dim=key[2], device=dev)
            tables = (cos.contiguous(), sin.contiguous())
            _CANONICAL_TABLES[key] = tables
    if verify:
        if (key[1], key[2]) != (ROPE_THETA, HEAD_DIM):
            raise ValueError("verify=True is only defined for theta=5e5, dim=128")
        got = rope_table_sha256(key[0])
        if got != CANONICAL_ROPE_TABLE_SHA256:
            raise RuntimeError(
                f"canonical RoPE table sha256 {got} != pinned {CANONICAL_ROPE_TABLE_SHA256}: this "
                "torch/CUDA build computes different cos/sin bits; train/rollout parity is broken"
            )
    return tables


def rope_table_sha256(
    device: torch.device | str | int | None = None,
    *,
    theta: float = ROPE_THETA,
    dim: int = HEAD_DIM,
) -> str:
    """sha256 hex of ``cos.bytes || sin.bytes`` of :func:`canonical_rope_table` (cached).

    Log it in each runtime's provenance; equal digests == bitwise-identical tables."""
    key = (_cuda_index(device), float(theta), int(dim))
    digest = _CANONICAL_SHA.get(key)
    if digest is None:
        cos, sin = canonical_rope_table(key[0], theta=theta, dim=dim)
        h = hashlib.sha256()
        for t in (cos, sin):
            h.update(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
        digest = h.hexdigest()
        _CANONICAL_SHA[key] = digest
    return digest


def rope_rows(
    positions: torch.Tensor,
    *,
    device: torch.device | str | int | None = None,
    theta: float = ROPE_THETA,
    dim: int = HEAD_DIM,
) -> tuple[torch.Tensor, torch.Tensor]:
    """fp32 ``(cos, sin)`` rows ``(T, dim)`` for absolute ``positions``, gathered (exact copies,
    ``index_select``) from :func:`canonical_rope_table` on ``device`` (default: the positions'
    CUDA device, else the current one).  Bitwise equal to ``rope_table(positions)``.

    Positions must be int32/int64, 1-D and inside ``[0, ROPE_TABLE_POSITIONS)``; out-of-range
    positions raise ``ValueError`` (checked on the host, one sync; skipped during CUDA-graph
    capture, where ``index_select``'s device-side bounds check still fails loudly)."""
    positions = torch.as_tensor(positions)
    if positions.dim() != 1:
        raise ValueError(f"positions must be 1-D (T,), got shape {tuple(positions.shape)}")
    if positions.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"positions must be int32 or int64, got {positions.dtype}")
    if device is None and positions.device.type == "cuda":
        device = positions.device
    cos, sin = canonical_rope_table(device, theta=theta, dim=dim)
    capturing = positions.device.type == "cuda" and torch.cuda.is_current_stream_capturing()
    if positions.numel() and not capturing:
        lo, hi = int(positions.min()), int(positions.max())
        if lo < 0 or hi >= ROPE_TABLE_POSITIONS:
            raise ValueError(
                f"position out of range: [{lo}, {hi}] not inside [0, {ROPE_TABLE_POSITIONS}) "
                "(max_position_embeddings)"
            )
    idx = positions.to(device=cos.device, dtype=torch.long)
    return cos.index_select(0, idx), sin.index_select(0, idx)


def layer_is_local(layer_idx: int) -> bool:
    """Frozen layer schedule: layer ``i`` (0..51) is global/NoPE iff ``i % 4 == 3``, else local
    (sliding window + RoPE): 39 local and 13 global layers."""
    i = int(layer_idx)
    if not 0 <= i < NUM_LAYERS:
        raise ValueError(f"layer_idx must be in [0, {NUM_LAYERS}), got {layer_idx}")
    return i % GLOBAL_LAYER_PERIOD != GLOBAL_LAYER_PERIOD - 1


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


def _check_layer_kind(
    local: bool | None, cos: torch.Tensor | None, sin: torch.Tensor | None
) -> None:
    if local is None:
        return
    has_tables = cos is not None or sin is not None
    if bool(local) and not has_tables:
        raise ValueError("local=True (sliding-window RoPE layer) needs cos/sin rows; got None")
    if not bool(local) and has_tables:
        raise ValueError("local=False (global NoPE layer) must not be given cos/sin rows")


def qknorm_scaled_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor | None,
    sin: torch.Tensor | None,
    *,
    eps: float = RMS_NORM_EPS,
    qk_scale: float = QK_SCALE,
    local: bool | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Strict O4 op: ``(q (T,HQ,128) bf16, k (T,HK,128) bf16, cos/sin (T,128) fp32 | None)``
    -> ``(q_hat bf16, k_hat bf16)``.  HQ/HK are 32/2 at TP1 and the per-rank head shards
    16/8/4 and 1 at TP2/4/8 (bitwise equal to the slice of the full call).

    Layer kind: pass ``local=True`` (sliding layer, RoPE; ``cos``/``sin`` required) or
    ``local=False`` (global NoPE layer; ``cos``/``sin`` must be None) - a mismatch raises.
    ``local=None`` is the legacy implicit form (``cos is None`` selects NoPE).
    Non-bf16 ``q``/``k`` raise ``TypeError`` (no fallback).

    Differentiable w.r.t. ``q`` and ``k`` with a deterministic, batch-invariant Triton backward.
    """
    _check_layer_kind(local, cos, sin)
    return _QKNormScaledRopeFunction.apply(q, k, cos, sin, float(eps), float(qk_scale))


def qknorm_scaled_rope_at(
    q: torch.Tensor,
    k: torch.Tensor,
    positions: torch.Tensor | None,
    *,
    local: bool | None = None,
    layer_idx: int | None = None,
    eps: float = RMS_NORM_EPS,
    qk_scale: float = QK_SCALE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The contract's assembly signature ``qknorm_scaled_rope(q, k, pos, local?)``.

    The layer kind is mandatory: give ``local`` or ``layer_idx`` (or both, consistent).  Local
    layers gather ``cos``/``sin`` rows for ``positions`` from :func:`canonical_rope_table` on
    ``q``'s device (so the table bits cannot come from another builder); global layers ignore
    ``positions`` (may be None).  Same bits as :func:`qknorm_scaled_rope` with ``rope_table``
    rows."""
    if local is None and layer_idx is None:
        raise ValueError("give local=True/False or layer_idx: the layer kind must be explicit")
    if layer_idx is not None:
        from_idx = layer_is_local(layer_idx)
        if local is not None and bool(local) != from_idx:
            raise ValueError(
                f"local={local} contradicts layer_idx={layer_idx} (global iff i % 4 == 3)"
            )
        local = from_idx
    if not local:
        return qknorm_scaled_rope(q, k, None, None, eps=eps, qk_scale=qk_scale, local=False)
    if positions is None:
        raise ValueError("a local (RoPE) layer needs positions")
    positions = torch.as_tensor(positions)
    if positions.dim() != 1 or positions.shape[0] != q.shape[0]:
        raise ValueError(
            f"positions must be (T,) with T={q.shape[0]}; got shape {tuple(positions.shape)}"
        )
    cos, sin = rope_rows(positions, device=q.device)
    return qknorm_scaled_rope(q, k, cos, sin, eps=eps, qk_scale=qk_scale, local=True)


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
        "rope_table": {
            "builder": "rl_engine.models.muse_glimmer.ops.qknorm_rope.rope_table",
            "device_type": "cuda",
            "formula": (
                "inv_freq = 1 / torch.pow(theta, arange(0,128,2,fp32)/128); "
                "angle = float(p)[:,None] * inv_freq[None,:]; emb = cat(angle, angle); "
                "cos, sin = emb.cos(), emb.sin()  (fp32, torch-CUDA)"
            ),
            "cache": "canonical_rope_table(device): one (cos, sin) object per CUDA device",
            "rows": "rope_rows(positions) = index_select of the canonical table (exact copies)",
            "positions": ROPE_TABLE_POSITIONS,
            "sha256": CANONICAL_ROPE_TABLE_SHA256,
        },
        "layer_selection": (
            "explicit: local=True (RoPE, cos/sin required) | local=False (NoPE, cos/sin must "
            "be None), mismatch raises; qknorm_scaled_rope_at requires local or layer_idx; "
            "schedule: layer i is global iff i % 4 == 3 (52 layers); local=None = legacy "
            "implicit (cos is None -> NoPE)"
        ),
        "dtype_policy": "q/k bf16 only; fp16/fp32 raise TypeError (no fallback)",
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

    def __call__(self, q, k, cos, sin, *, local: bool | None = None):
        return self.forward(q, k, cos, sin, local=local)

    def forward(self, q, k, cos, sin, *, local: bool | None = None):
        return qknorm_scaled_rope(
            q, k, cos, sin, eps=self.eps, qk_scale=self.qk_scale, local=local
        )

    def forward_fp32(self, q, k, cos, sin):
        return qknorm_scaled_rope_fp32(q, k, cos, sin, eps=self.eps, qk_scale=self.qk_scale)

    def trace(self, HQ: int = 32, HK: int = 2, D: int = HEAD_DIM) -> dict[str, Any]:
        return trace(HQ, HK, D)
