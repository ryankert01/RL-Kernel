# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""SwiGLU and the composed SwiGLU FFN of the Muse-Glimmer-30B strict POC (RFC #501 row
``swiglu_ffn``; numerical contract O6 with the O3 GEMM rules).

Shapes: x (T, H=6656) bf16, W_gate / W_up (F=19968, H) bf16, W_down (H, F) bf16, y (T, H) bf16.

Forward (contract O6; casts happen only where "→ bf16" is written, contract R5)::

    g = gemm(x, W_gate)  → bf16          column-parallel GEMM, fp32 accumulate, one rounding
    u = gemm(x, W_up)    → bf16
    h = silu(g→fp32) * (u→fp32) → bf16    row-local Triton kernel, silu(x) = x / (1 + exp(-x)) fp32
    y = gemm(h, W_down)  → bf16          row-parallel GEMM: K=F walked in the 8 canonical chunks
                                          of 2496, fp32 partials summed ((0+1)+(2+3))+((4+5)+(6+7)),
                                          rounded once (contract O3)

Backward (contract O3/O6: det_gemm + row-local fp32 VJP, never cuBLAS)::

    dh        = gemm(dy, W_downᵀ)                   → bf16   (K = H, same chunk rule)
    (dg, du)  = fp32 VJP of silu*mul on (dh, g, u)  → bf16   (row-local kernel)
    dx        = gemm(dg, W_gateᵀ) + gemm(du, W_upᵀ)          both partials fp32, summed in fp32,
                                                    → bf16   one rounding (K = F, chunk rule)
    dW_gate   = gemm(dgᵀ, xᵀ), dW_up = gemm(duᵀ, xᵀ), dW_down = gemm(dyᵀ, hᵀ)   → bf16
                (K = T, the declared ragged token rule: T_pad = roundup(T, 256), 8 chunks of
                 T_pad/8 token rows, rows >= T masked to zero; every T >= 1 is accepted)

The GEMM is pluggable: ``gemm(a (M,K) bf16, w (N,K) bf16, *, out_dtype=torch.bfloat16,
ragged_k=False) -> a @ w.T`` where ``ragged_k=True`` selects the ragged token rule above (the
three dW GEMMs pass it; every other GEMM must have an in-contract K).  ``default_gemm()`` is
``rl_engine.models.muse_glimmer.ops.det_gemm.fast_det_gemm`` (no silent fallback: an import
failure propagates).  :class:`TritonDetGemmAdapter` is an explicit, opt-in twin that drives the
repo's fixed-order Triton GEMM kernel through the same arithmetic (8 chunk partials in fp32,
fixed tree, one rounding; bitwise equal to ``fast_det_gemm`` wherever both accept the shape).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
import triton
from torch import Tensor

from rl_engine.models.muse_glimmer.ops import det_gemm as _det_gemm
from rl_engine.models.muse_glimmer.triton.swiglu import (
    kernel_trace as _swiglu_kernel_trace,
)
from rl_engine.models.muse_glimmer.triton.swiglu import (
    launch_swiglu_bwd,
    launch_swiglu_fwd,
)
from rl_engine.ops.autograd.backward_runtime import record_backward

GemmFn = Callable[..., Tensor]

_CONTRACT = "docs/contracts/muse-glimmer-30b-strict-contract.md O6 (+O3 GEMM rules)"
_K_CHUNKS = 8
_TREE = "((0+1)+(2+3))+((4+5)+(6+7))"
# Ragged (token-axis) rule shared with fast_det_gemm: K padded to a multiple of 8 x 32.
_RAGGED_UNIT = 256
_DW_RULE = (
    "K = T: T_pad = roundup(T, 256), 8 chunks of T_pad/8 token rows, rows >= T masked to zero "
    "(declared ragged token rule, same as fast_det_gemm's dW; any T >= 1)"
)


# --------------------------------------------------------------------------------------------
# SwiGLU: golden + strict kernel with autograd
# --------------------------------------------------------------------------------------------
def swiglu_fp32(gate: Tensor, up: Tensor) -> Tensor:
    """Golden: ``silu(gate→fp32) * (up→fp32)`` in fp32, ``silu(x) = x / (1 + exp(-x))``.

    Elementwise, so there is no reduction order; ``torch.exp`` and the IEEE division run in fp32.
    """
    if gate.shape != up.shape:
        raise ValueError(
            f"gate and up must share shape, got {tuple(gate.shape)} vs {tuple(up.shape)}."
        )
    g = gate.to(torch.float32)
    u = up.to(torch.float32)
    silu = g / (1.0 + torch.exp(-g))
    return silu * u


def swiglu_backward_fp32(grad_h: Tensor, gate: Tensor, up: Tensor) -> tuple[Tensor, Tensor]:
    """Golden row-local VJP in fp32 (closed form used by the kernel): ``(dgate, dup)``."""
    dh = grad_h.to(torch.float32)
    g = gate.to(torch.float32)
    u = up.to(torch.float32)
    denom = 1.0 + torch.exp(-g)
    s = 1.0 / denom
    silu = g / denom
    dsilu = s * (1.0 + g * (1.0 - s))
    return (dh * u) * dsilu, dh * silu


class _SwiGLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, gate: Tensor, up: Tensor) -> Tensor:
        h = launch_swiglu_fwd(gate, up)
        ctx.save_for_backward(gate, up)
        return h

    @staticmethod
    def backward(ctx: Any, grad_h: Tensor) -> tuple[Tensor | None, Tensor | None]:
        gate, up = ctx.saved_tensors
        if not (ctx.needs_input_grad[0] or ctx.needs_input_grad[1]):
            return None, None
        dgate, dup = launch_swiglu_bwd(grad_h, gate, up)
        return (
            dgate if ctx.needs_input_grad[0] else None,
            dup if ctx.needs_input_grad[1] else None,
        )


def swiglu(gate: Tensor, up: Tensor) -> Tensor:
    """Strict SwiGLU: ``h = silu(gate→fp32) * (up→fp32) → gate.dtype`` with a deterministic,
    row-local fp32 backward. ``gate`` and ``up`` are ``(T, F)`` (any leading shape is flattened)."""
    if gate.shape != up.shape:
        raise ValueError(
            f"gate and up must share shape, got {tuple(gate.shape)} vs {tuple(up.shape)}."
        )
    if gate.dtype != up.dtype:
        raise TypeError(f"gate and up must share dtype, got {gate.dtype} and {up.dtype}.")
    if gate.dim() == 2:
        return _SwiGLUFunction.apply(gate, up)
    width = gate.shape[-1]
    h = _SwiGLUFunction.apply(gate.reshape(-1, width), up.reshape(-1, width))
    return h.reshape(gate.shape)


def swiglu_trace() -> dict[str, Any]:
    """Contract R4 provenance of the SwiGLU elementwise op."""
    return {
        "op": "swiglu",
        "contract": _CONTRACT,
        "dtype_in": "bfloat16",
        "dtype_acc": "float32",
        "dtype_out": "bfloat16",
        "reduction_order": (
            "none (elementwise); h = (g / (1 + exp(-g))) * u in fp32, one rounding at the store; "
            "backward dg = (dh*u) * (s*(1+g*(1-s))), du = dh*silu(g) in fp32, one rounding"
        ),
        "tile_config": _swiglu_kernel_trace(),
    }


# --------------------------------------------------------------------------------------------
# GEMM plug: default adapter around the repo's fixed-order Triton kernel
# --------------------------------------------------------------------------------------------
class TritonDetGemmAdapter:
    """``y = a @ w.T`` through the repo's fixed-order Triton GEMM kernel in the O3 K-chunk tree.

    Explicit, opt-in twin of ``fast_det_gemm`` (never selected implicitly).

    Arithmetic (contract O3): K is split into 8 canonical chunks of K/8 (K=19968 → 2496,
    K=6656 → 832, K=4096 → 512). For every chunk one program per (BLOCK_M, BLOCK_N) = (64, 64)
    output tile walks the chunk in ascending BLOCK_K=32 steps with a bf16 tensor-core ``tl.dot``
    into an fp32 accumulator starting at +0 (no split-K across programs, no atomics) and stores the
    fp32 partial. The 8 fp32 partials are summed in the fixed tree ((0+1)+(2+3))+((4+5)+(6+7))
    with fp32 elementwise adds and rounded once to ``out_dtype``. Because each chunk is a separate
    launch, this is the declared tree by construction (and bitwise equal to ``fast_det_gemm``
    wherever both accept K). K % 8 != 0 raises ``ValueError`` (out of contract) unless
    ``ragged_k=True``, which applies the declared ragged token rule of ``fast_det_gemm``:
    K_pad = roundup(K, 256), chunk c = columns [c*K_pad/8, (c+1)*K_pad/8) clipped to K, an empty
    chunk is an exact fp32 zero partial (``ragged_k`` also applies to K % 256 == 0, where it is the
    plain 8-chunk rule).

    ``a`` and ``w`` may have arbitrary positive strides: the weight is read in its native (N, K)
    layout through strides, so no transposed copy is cached and the transposed views needed by
    the backward GEMMs cost nothing. Triton specialises the compiled kernel on M/N/K/stride
    divisibility only, which does not change the per-element arithmetic (verified bitwise at
    M = 1 / 77 / 4096). Kernel: ``_det_gemm_fp32_kernel`` of
    ``rl_engine.backends.shared.triton.gemm.det_gemm`` with ``PROMOTE_INPUTS=False`` (the
    fp32-output path of ``TritonDetGemmOp``, bf16 operands).
    """

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32
    NUM_WARPS = 4
    NUM_STAGES = 3
    K_CHUNKS = _K_CHUNKS

    def __init__(self) -> None:
        from rl_engine.backends.shared.triton.gemm.det_gemm import _det_gemm_fp32_kernel

        self._kernel = _det_gemm_fp32_kernel

    def __call__(
        self,
        a: Tensor,
        w: Tensor,
        *,
        out_dtype: torch.dtype = torch.bfloat16,
        ragged_k: bool = False,
    ) -> Tensor:
        if a.dim() != 2 or w.dim() != 2:
            raise ValueError("TritonDetGemmAdapter expects a (M, K) and w (N, K).")
        if a.dtype != torch.bfloat16 or w.dtype != torch.bfloat16:
            raise TypeError(
                f"TritonDetGemmAdapter requires bf16 operands, got {a.dtype}/{w.dtype}."
            )
        if not a.is_cuda or a.device != w.device:
            raise RuntimeError("TritonDetGemmAdapter operands must share one CUDA device.")
        m_size, k_size = a.shape
        n_size, k_w = w.shape
        if k_w != k_size:
            raise ValueError(f"K mismatch: a is {tuple(a.shape)}, w is {tuple(w.shape)}.")
        if a.stride(1) == 0 or w.stride(1) == 0:
            raise ValueError("TritonDetGemmAdapter rejects broadcast (zero-stride) operands.")
        if out_dtype not in (torch.bfloat16, torch.float32):
            raise TypeError(f"TritonDetGemmAdapter out_dtype must be bf16 or fp32, got {out_dtype}.")
        if m_size == 0 or n_size == 0:
            return torch.zeros((m_size, n_size), dtype=out_dtype, device=a.device)
        if k_size == 0:
            return torch.zeros((m_size, n_size), dtype=out_dtype, device=a.device)
        if ragged_k:
            step = -(-k_size // _RAGGED_UNIT) * _RAGGED_UNIT // self.K_CHUNKS
        elif k_size % self.K_CHUNKS:
            raise ValueError(
                f"TritonDetGemmAdapter: K={k_size} is not a multiple of {self.K_CHUNKS} canonical "
                "chunks (out of contract); token-axis reductions must pass ragged_k=True"
            )
        else:
            step = k_size // self.K_CHUNKS

        def partial(i: int) -> Tensor:
            return self._partial(a, w, min(i * step, k_size), min((i + 1) * step, k_size))

        # Fixed tree ((0+1)+(2+3))+((4+5)+(6+7)); in-place fp32 adds keep <= 4 live partials.
        s01 = partial(0).add_(partial(1))
        s23 = partial(2).add_(partial(3))
        s0123 = s01.add_(s23)
        s45 = partial(4).add_(partial(5))
        s67 = partial(6).add_(partial(7))
        s4567 = s45.add_(s67)
        y = s0123.add_(s4567)
        return y if out_dtype == torch.float32 else y.to(out_dtype)

    def _partial(self, a: Tensor, w: Tensor, k0: int, k1: int) -> Tensor:
        a_c = a[:, k0:k1]
        w_c = w[:, k0:k1]
        m_size, k_size = a_c.shape
        n_size = w_c.shape[0]
        if k_size == 0:  # empty ragged chunk: exact zero partial (x + 0 == x in the tree)
            return torch.zeros((m_size, n_size), dtype=torch.float32, device=a.device)
        out = torch.empty((m_size, n_size), dtype=torch.float32, device=a.device)
        grid = (triton.cdiv(m_size, self.BLOCK_M), triton.cdiv(n_size, self.BLOCK_N))
        self._kernel[grid](
            a_c,
            w_c,
            out,
            m_size,
            n_size,
            k_size,
            a_c.stride(0),
            a_c.stride(1),
            w_c.stride(1),  # b[k, n] = w[n, k]
            w_c.stride(0),
            out.stride(0),
            out.stride(1),
            BLOCK_M=self.BLOCK_M,
            BLOCK_N=self.BLOCK_N,
            BLOCK_K=self.BLOCK_K,
            PROMOTE_INPUTS=False,
            num_warps=self.NUM_WARPS,
            num_stages=self.NUM_STAGES,
        )
        return out

    def trace(self) -> dict[str, Any]:
        return {
            "op": "det_gemm",
            "impl": "TritonDetGemmAdapter",
            "kernel": "rl_engine.backends.shared.triton.gemm.det_gemm._det_gemm_fp32_kernel",
            "dtype_in": "bfloat16",
            "dtype_acc": "float32",
            "dtype_out": "bfloat16 (fp32 on request)",
            "reduction_order": (
                f"K split in {self.K_CHUNKS} canonical chunks of K/8 (K % 8 != 0 raises unless "
                "ragged_k); per chunk one program per output tile walks K ascending in BLOCK_K "
                "steps (bf16 tl.dot, fp32 accumulate from +0, no split-K); fp32 partials summed "
                f"{_TREE}; one rounding to the output dtype"
            ),
            "k_rule": "K % 8 == 0 (chunk K/8); otherwise ValueError unless ragged_k=True",
            "ragged_k_rule": (
                "K_pad = roundup(K, 256), chunk c = columns [c*K_pad/8, (c+1)*K_pad/8) clipped "
                "to K, empty chunk = exact fp32 zero partial (== fast_det_gemm's dW rule)"
            ),
            "selection": "explicit only (never an implicit fallback)",
            "tile_config": {
                "BLOCK_M": self.BLOCK_M,
                "BLOCK_N": self.BLOCK_N,
                "BLOCK_K": self.BLOCK_K,
                "num_warps": self.NUM_WARPS,
                "num_stages": self.NUM_STAGES,
                "k_chunks": self.K_CHUNKS,
                "weight_layout": "(N, K) read through strides",
            },
        }


class _FastDetGemm:
    """Shim over the sibling ``fast_det_gemm``: forwards ``out_dtype``/``ragged_k`` and the
    operands as they are (the kernel reads strided/transposed views through their strides, so the
    backward's ``w.t()`` views cost no copy) and exposes the sibling's full ``trace()``."""

    name = "fast_det_gemm"

    def __init__(self, fn: GemmFn) -> None:
        self._fn = fn

    def __call__(
        self,
        a: Tensor,
        w: Tensor,
        *,
        out_dtype: torch.dtype = torch.bfloat16,
        ragged_k: bool = False,
    ) -> Tensor:
        return self._fn(a, w, out_dtype=out_dtype, ragged_k=ragged_k)

    def trace(self) -> dict[str, Any]:
        tracer = getattr(self._fn, "trace", None)
        if not callable(tracer):
            tracer = _det_gemm.trace
        return tracer()


_DEFAULT_GEMM: GemmFn | None = None


def default_gemm() -> GemmFn:
    """``fast_det_gemm`` (``rl_engine.models.muse_glimmer.ops.det_gemm``), resolved once per
    process. There is no implicit fallback; pass ``gemm=TritonDetGemmAdapter()`` explicitly to use
    the twin kernel."""
    global _DEFAULT_GEMM
    if _DEFAULT_GEMM is None:
        _DEFAULT_GEMM = _FastDetGemm(_det_gemm.fast_det_gemm)
    return _DEFAULT_GEMM


def _gemm_trace(gemm: GemmFn) -> dict[str, Any]:
    tracer = getattr(gemm, "trace", None)
    if callable(tracer):
        return tracer()
    return {"op": "det_gemm", "impl": getattr(gemm, "__name__", type(gemm).__name__)}


def _gemm_name(gemm: GemmFn) -> str:
    return str(getattr(gemm, "name", None) or getattr(gemm, "__name__", type(gemm).__name__))


# --------------------------------------------------------------------------------------------
# SwiGLU FFN: golden + strict composition with deterministic autograd
# --------------------------------------------------------------------------------------------
def swiglu_ffn_fp32(
    x: Tensor,
    w_gate: Tensor,
    w_up: Tensor,
    w_down: Tensor,
    *,
    compute_dtype: torch.dtype = torch.float64,
) -> Tensor:
    """Golden FFN: every operand promoted to ``compute_dtype`` (fp64 by default), no intermediate
    rounding, result returned in fp32. silu(x) = x / (1 + exp(-x)).

    The GEMMs use ``torch.matmul`` in fp64 whose accumulation order is not fixed; at fp64 the
    ordering noise (~1e-16 relative) is five orders of magnitude below the bf16 tolerances this
    golden is compared against (flagged in the report as a contract R4 clarification).
    """
    _validate_ffn(x, w_gate, w_up, w_down, require_cuda=False, require_bf16=False)
    xd = x.to(compute_dtype)
    g = torch.matmul(xd, w_gate.to(compute_dtype).t())
    u = torch.matmul(xd, w_up.to(compute_dtype).t())
    h = (g / (1.0 + torch.exp(-g))) * u
    y = torch.matmul(h, w_down.to(compute_dtype).t())
    return y.to(torch.float32)


def _validate_ffn(
    x: Tensor,
    w_gate: Tensor,
    w_up: Tensor,
    w_down: Tensor,
    *,
    require_cuda: bool,
    require_bf16: bool,
) -> None:
    if x.dim() < 1:
        raise ValueError("x must have at least one dimension (…, H).")
    hidden = x.shape[-1]
    for name, weight in (("w_gate", w_gate), ("w_up", w_up), ("w_down", w_down)):
        if weight.dim() != 2:
            raise ValueError(f"{name} must be 2-D, got shape {tuple(weight.shape)}.")
    ffn = w_gate.shape[0]
    expected = {
        "w_gate": (ffn, hidden),
        "w_up": (ffn, hidden),
        "w_down": (hidden, ffn),
    }
    for name, weight in (("w_gate", w_gate), ("w_up", w_up), ("w_down", w_down)):
        if tuple(weight.shape) != expected[name]:
            raise ValueError(f"{name} must have shape {expected[name]}, got {tuple(weight.shape)}.")
    tensors = {"x": x, "w_gate": w_gate, "w_up": w_up, "w_down": w_down}
    for name, tensor in tensors.items():
        if require_bf16 and tensor.dtype != torch.bfloat16:
            raise TypeError(f"{name} must be bf16, got {tensor.dtype}.")
        if require_cuda and not tensor.is_cuda:
            raise RuntimeError(f"{name} must be a CUDA tensor, got device '{tensor.device}'.")
        if tensor.device != x.device:
            raise RuntimeError(f"{name} must be on {x.device}, got {tensor.device}.")


class _SwiGLUFFNFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any, x: Tensor, w_gate: Tensor, w_up: Tensor, w_down: Tensor, gemm: GemmFn
    ) -> Tensor:
        g = gemm(x, w_gate)
        u = gemm(x, w_up)
        h = launch_swiglu_fwd(g, u)
        y = gemm(h, w_down)
        ctx.save_for_backward(x, g, u, h, w_gate, w_up, w_down)
        ctx.gemm = gemm
        return y

    @staticmethod
    def backward(ctx: Any, grad_y: Tensor) -> tuple[Any, ...]:
        x, g, u, h, w_gate, w_up, w_down = ctx.saved_tensors
        gemm: GemmFn = ctx.gemm
        needs_x, needs_wg, needs_wu, needs_wd = ctx.needs_input_grad[:4]
        dy = grad_y.to(torch.bfloat16).contiguous()
        dx = dw_gate = dw_up = dw_down = None
        # Weight gradients reduce over the T tokens: declared ragged token rule (any T >= 1).
        if needs_wd:
            dw_down = gemm(dy.t(), h.t(), ragged_k=True)
        if needs_x or needs_wg or needs_wu:
            dh = gemm(dy, w_down.t())
            dg, du = launch_swiglu_bwd(dh, g, u)
            del dh
            if needs_x:
                dx_gate = gemm(dg, w_gate.t(), out_dtype=torch.float32)
                dx_up = gemm(du, w_up.t(), out_dtype=torch.float32)
                dx = dx_gate.add_(dx_up).to(torch.bfloat16)
                del dx_gate, dx_up
            if needs_wg:
                dw_gate = gemm(dg.t(), x.t(), ragged_k=True)
            if needs_wu:
                dw_up = gemm(du.t(), x.t(), ragged_k=True)
        record_backward(
            "swiglu_ffn",
            kernel_id=f"{_gemm_name(gemm)}+rl_engine.models.muse_glimmer.triton.swiglu",
            impl="muse_glimmer_swiglu_ffn",
            family="triton",
        )
        return dx, dw_gate, dw_up, dw_down, None


def swiglu_ffn(
    x: Tensor,
    w_gate: Tensor,
    w_up: Tensor,
    w_down: Tensor,
    *,
    gemm: GemmFn | None = None,
) -> Tensor:
    """Strict SwiGLU FFN ``y = gemm(silu(gemm(x, W_gate)) * gemm(x, W_up), W_down)`` (module doc).

    ``x`` is ``(…, H)`` bf16 and the result ``(…, H)`` bf16; autograd end to end through
    deterministic GEMMs and the row-local fp32 SwiGLU VJP. ``gemm`` defaults to
    :func:`default_gemm`.
    """
    _validate_ffn(x, w_gate, w_up, w_down, require_cuda=True, require_bf16=True)
    if gemm is None:
        gemm = default_gemm()
    lead = x.shape[:-1]
    x2d = x.reshape(-1, x.shape[-1])
    y = _SwiGLUFFNFunction.apply(x2d, w_gate, w_up, w_down, gemm)
    return y.reshape(*lead, y.shape[-1])


def trace(gemm: GemmFn | None = None) -> dict[str, Any]:
    """Contract R4 provenance dict for ``swiglu_ffn`` (stable for a given ``gemm``)."""
    if gemm is None:
        gemm = default_gemm()
    return {
        "op": "swiglu_ffn",
        "contract": _CONTRACT,
        "nodes": ["mlp_gate_proj", "up_proj", "swiglu", "down_proj"],
        "dtype_in": "bfloat16",
        "dtype_acc": "float32",
        "dtype_out": "bfloat16",
        "reduction_order": (
            "g = gemm(x, W_gate) -> bf16; u = gemm(x, W_up) -> bf16; h = silu(g) * u in fp32 "
            "row-local -> bf16; y = gemm(h, W_down) with K=F in 8 canonical chunks of F/8, fp32 "
            f"partials {_TREE} -> bf16 once"
        ),
        "backward": (
            "dh = gemm(dy, W_down^T) -> bf16; (dg, du) = row-local fp32 VJP -> bf16; "
            "dx = gemm(dg, W_gate^T) + gemm(du, W_up^T) summed in fp32 -> bf16 once; "
            "dW_gate = gemm(dg^T, x^T), dW_up = gemm(du^T, x^T), dW_down = gemm(dy^T, h^T) -> bf16 "
            "with ragged_k=True"
        ),
        "dw_chunk_rule": _DW_RULE,
        "tile_config": {
            "swiglu": _swiglu_kernel_trace(),
            "gemm": _gemm_trace(gemm),
        },
    }


class SwiGLUFFNOp:
    """Contract R4 surface: ``forward_fp32`` / ``forward`` / autograd backward / ``trace``."""

    op_class = "reduction"

    def __init__(self, gemm: GemmFn | None = None) -> None:
        self.gemm = gemm

    def __call__(self, x: Tensor, w_gate: Tensor, w_up: Tensor, w_down: Tensor) -> Tensor:
        return self.forward(x, w_gate, w_up, w_down)

    def forward(self, x: Tensor, w_gate: Tensor, w_up: Tensor, w_down: Tensor) -> Tensor:
        return swiglu_ffn(x, w_gate, w_up, w_down, gemm=self.gemm)

    @staticmethod
    def forward_fp32(x: Tensor, w_gate: Tensor, w_up: Tensor, w_down: Tensor) -> Tensor:
        return swiglu_ffn_fp32(x, w_gate, w_up, w_down)

    def trace(self) -> dict[str, Any]:
        return trace(self.gemm)


forward = swiglu_ffn
forward_fp32 = swiglu_ffn_fp32

__all__ = [
    "SwiGLUFFNOp",
    "TritonDetGemmAdapter",
    "default_gemm",
    "forward",
    "forward_fp32",
    "swiglu",
    "swiglu_backward_fp32",
    "swiglu_ffn",
    "swiglu_ffn_fp32",
    "swiglu_fp32",
    "swiglu_trace",
    "trace",
]
