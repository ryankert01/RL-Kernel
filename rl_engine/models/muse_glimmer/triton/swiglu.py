# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Row-local SwiGLU Triton kernels for the Muse-Glimmer-30B strict POC (contract O6, R2, R3, R6).

Forward (one rounding, at the store)::

    silu(g) = g / (1 + exp(-g))          fp32, IEEE division (``tl.div_rn``), fp32 ``tl.exp``
    h       = silu(g→fp32) * (u→fp32)    fp32  → out dtype (bf16 in the strict chain)

Backward (row-local fp32 VJP of ``silu * mul``; inputs cast to fp32, one rounding at the store)::

    s   = 1 / (1 + exp(-g))              fp32
    dg  = (dh * u) * (s * (1 + g * (1 - s)))
    du  = dh * silu(g)                   silu(g) evaluated exactly as in the forward

Reduction order: there is none — the ops are elementwise. One program owns one logical row and
walks F in ascending ``BLOCK_F`` tiles (contract R3), so an element's bits depend only on its own
(g, u, dh) values and never on T, on the row's batch position or on the launch shape.
Launch configuration is a compile-time constant recorded by :func:`kernel_trace` (contract R6);
there is no autotuning.
"""

from __future__ import annotations

from typing import Any

import torch
import triton
import triton.language as tl
from torch import Tensor

#: Pinned launch configuration (contract R6). F=19968 = 19.5 * BLOCK_F: the last tile is masked.
BLOCK_F = 1024
NUM_WARPS = 4
NUM_STAGES = 3

_KERNEL_MODULE = "rl_engine.models.muse_glimmer.triton.swiglu"
_SUPPORTED_DTYPES = (torch.bfloat16, torch.float16, torch.float32)


@triton.jit
def _swiglu_fwd_kernel(
    g_ptr,
    u_ptr,
    h_ptr,
    F,
    stride_g,
    stride_u,
    stride_h,
    BLOCK_F: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    g_row = g_ptr + row * stride_g
    u_row = u_ptr + row * stride_u
    h_row = h_ptr + row * stride_h
    for start in range(0, F, BLOCK_F):
        offs = start + tl.arange(0, BLOCK_F)
        mask = offs < F
        g = tl.load(g_row + offs, mask=mask, other=0.0).to(tl.float32)
        u = tl.load(u_row + offs, mask=mask, other=0.0).to(tl.float32)
        silu = tl.div_rn(g, 1.0 + tl.exp(-g))
        h = silu * u
        tl.store(h_row + offs, h.to(h_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _swiglu_bwd_kernel(
    dh_ptr,
    g_ptr,
    u_ptr,
    dg_ptr,
    du_ptr,
    F,
    stride_dh,
    stride_g,
    stride_u,
    stride_dg,
    stride_du,
    BLOCK_F: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    dh_row = dh_ptr + row * stride_dh
    g_row = g_ptr + row * stride_g
    u_row = u_ptr + row * stride_u
    dg_row = dg_ptr + row * stride_dg
    du_row = du_ptr + row * stride_du
    for start in range(0, F, BLOCK_F):
        offs = start + tl.arange(0, BLOCK_F)
        mask = offs < F
        dh = tl.load(dh_row + offs, mask=mask, other=0.0).to(tl.float32)
        g = tl.load(g_row + offs, mask=mask, other=0.0).to(tl.float32)
        u = tl.load(u_row + offs, mask=mask, other=0.0).to(tl.float32)
        denom = 1.0 + tl.exp(-g)
        s = tl.div_rn(1.0, denom)
        silu = tl.div_rn(g, denom)
        dsilu = s * (1.0 + g * (1.0 - s))
        dg = (dh * u) * dsilu
        du = dh * silu
        tl.store(dg_row + offs, dg.to(dg_ptr.dtype.element_ty), mask=mask)
        tl.store(du_row + offs, du.to(du_ptr.dtype.element_ty), mask=mask)


def _row_major(t: Tensor, name: str) -> Tensor:
    if t.dim() != 2:
        raise ValueError(f"{name} must be 2-D (T, F), got shape {tuple(t.shape)}.")
    if t.dtype not in _SUPPORTED_DTYPES:
        raise TypeError(f"{name} must be bf16, fp16 or fp32, got {t.dtype}.")
    if not t.is_cuda:
        raise RuntimeError(f"{name} must be a CUDA tensor, got device '{t.device}'.")
    if t.shape[1] > 0 and t.stride(1) != 1:
        t = t.contiguous()
    return t


def _check_same(a: Tensor, b: Tensor, name_a: str, name_b: str) -> None:
    if a.shape != b.shape:
        raise ValueError(
            f"{name_a} and {name_b} must share shape, got {tuple(a.shape)} vs {tuple(b.shape)}."
        )
    if a.device != b.device:
        raise RuntimeError(
            f"{name_a} and {name_b} must share device, got {a.device} vs {b.device}."
        )


def launch_swiglu_fwd(
    gate: Tensor, up: Tensor, *, out_dtype: torch.dtype | None = None
) -> Tensor:
    """``h = silu(gate→fp32) * (up→fp32)``, rounded once to ``out_dtype`` (default: input dtype)."""
    gate = _row_major(gate, "gate")
    up = _row_major(up, "up")
    _check_same(gate, up, "gate", "up")
    rows, width = gate.shape
    out = torch.empty((rows, width), dtype=out_dtype or gate.dtype, device=gate.device)
    if rows == 0 or width == 0:
        return out
    _swiglu_fwd_kernel[(rows,)](
        gate,
        up,
        out,
        width,
        gate.stride(0),
        up.stride(0),
        out.stride(0),
        BLOCK_F=BLOCK_F,
        num_warps=NUM_WARPS,
        num_stages=NUM_STAGES,
    )
    return out


def launch_swiglu_bwd(
    grad_h: Tensor,
    gate: Tensor,
    up: Tensor,
    *,
    out_dtype: torch.dtype | None = None,
) -> tuple[Tensor, Tensor]:
    """Row-local fp32 VJP ``(dgate, dup)``, rounded once to ``out_dtype`` (default: input dtype)."""
    gate = _row_major(gate, "gate")
    up = _row_major(up, "up")
    grad_h = _row_major(grad_h, "grad_h")
    _check_same(gate, up, "gate", "up")
    _check_same(gate, grad_h, "gate", "grad_h")
    rows, width = gate.shape
    dtype = out_dtype or gate.dtype
    dgate = torch.empty((rows, width), dtype=dtype, device=gate.device)
    dup = torch.empty((rows, width), dtype=dtype, device=gate.device)
    if rows == 0 or width == 0:
        return dgate, dup
    _swiglu_bwd_kernel[(rows,)](
        grad_h,
        gate,
        up,
        dgate,
        dup,
        width,
        grad_h.stride(0),
        gate.stride(0),
        up.stride(0),
        dgate.stride(0),
        dup.stride(0),
        BLOCK_F=BLOCK_F,
        num_warps=NUM_WARPS,
        num_stages=NUM_STAGES,
    )
    return dgate, dup


def kernel_trace() -> dict[str, Any]:
    """Launch configuration and arithmetic provenance of the two kernels (contract R6)."""
    return {
        "kernel": _KERNEL_MODULE,
        "forward": "_swiglu_fwd_kernel",
        "backward": "_swiglu_bwd_kernel",
        "grid": "(T,) one program per logical row; F walked ascending in BLOCK_F tiles",
        "BLOCK_F": BLOCK_F,
        "num_warps": NUM_WARPS,
        "num_stages": NUM_STAGES,
        "exp": "tl.exp fp32",
        "div": "tl.div_rn fp32 (IEEE round-to-nearest)",
        "autotune": False,
    }


__all__ = [
    "BLOCK_F",
    "NUM_STAGES",
    "NUM_WARPS",
    "kernel_trace",
    "launch_swiglu_bwd",
    "launch_swiglu_fwd",
]
