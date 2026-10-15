# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Triton tile kernels for the Muse-Glimmer-30B soft-capped LM-head log-probability (O7).

The LM head is evaluated per canonical vocabulary tile by ``det_gemm`` (fp32 logits); these
kernels consume one fp32 logits tile ``z`` of shape ``(T, tile_size)`` at a time.

Arithmetic (all fp32, identical in every kernel below and in the torch golden):

* soft cap            ``zhat = T_cap * tanh((z * m) / T_cap)`` — RN multiply by ``m``
  (``output_multiplier``), IEEE round-to-nearest division by ``T_cap`` (``tl.div_rn``;
  Triton's default ``/`` is not IEEE), ``libdevice.tanh`` (bitwise equal to ``torch.tanh``
  fp32 on CUDA — verified on 5e7 samples, see the report), RN multiply by ``T_cap``.
  HF order (``mul_output_multiplier``, ``div_T``, ``tanh``, ``mul_T``) is preserved.
* padded columns      global column ``>= vocab_size`` contributes the identity (``-inf`` to
  the max, ``0`` to the sum, ``0`` to the gradient).
* tile statistics     one program per (row, tile).  Pass 1 walks the tile in ``BLOCK_V``
  chunks from the tile start and takes ``m_t = max zhat`` (exact, order free).  Pass 2 walks
  the same chunks in the same order and accumulates ``s_t += tl.sum(exp(zhat - m_t))``
  chunk by chunk (fixed sequential chunk order; the intra-chunk order is ``tl.sum``'s fixed
  tree for ``BLOCK_V``/``num_warps``).  Nothing depends on the row index or on ``T``, so a
  logical row's bits are batch invariant, and nothing depends on other tiles, so tiles may
  be computed in any order or on any rank.
* exp/log             ``tl.exp`` / ``tl.log`` fp32 (contract R2).
* backward            ``dz = (onehot - exp(zhat - logZ)) * dlogp * (m * (1 - tanh**2))`` in
  fp32, padded columns forced to ``0``, then ONE rounding to bf16 (the declared cast for the
  ``det_gemm`` operands of ``dh`` and ``dW``).

Launch configuration is a compile-time constant (``BLOCK_V=1024``, ``NUM_WARPS=4``); there is
no autotuning.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

BLOCK_V: int = 1024
NUM_WARPS: int = 4
KERNEL_ID = "rl_engine.models.muse_glimmer.triton.softcap_logprob"


@triton.jit
def _softcap(z, multiplier, softcap):
    # zhat = T * tanh((z * m) / T); see the module docstring for the rounding of each step.
    return softcap * libdevice.tanh(tl.div_rn(z * multiplier, softcap))


@triton.jit
def _softcap_tile_stats_kernel(
    z_ptr,  # [T, tile_size] fp32 logits of ONE vocabulary tile
    targets_ptr,  # [T] int64 global target ids
    m_ptr,  # [T] fp32 out: tile max of zhat over the real columns
    s_ptr,  # [T] fp32 out: sum exp(zhat - m) over the real columns
    tgt_logit_ptr,  # [T] fp32 out: zhat[target] if the target lies in this tile else 0
    tgt_in_ptr,  # [T] int8 out: 1 if the target lies in this tile
    tile_size,
    real_cols,  # number of non-padded columns in this tile (0..tile_size)
    col_begin,  # global column index of the tile start
    stride_row,
    multiplier,
    softcap,
    BLOCK_V: tl.constexpr,
):
    row = tl.program_id(0)
    row_base = row.to(tl.int64) * stride_row

    # Pass 1: tile max (exact; the max is order independent).
    m = tl.full((), float("-inf"), dtype=tl.float32)
    for start in range(0, tile_size, BLOCK_V):
        cols = start + tl.arange(0, BLOCK_V)
        mask = cols < real_cols
        z = tl.load(z_ptr + row_base + cols, mask=mask, other=0.0)
        zhat = tl.where(mask, _softcap(z, multiplier, softcap), float("-inf"))
        m = tl.maximum(m, tl.max(zhat))

    # Pass 2: sum exp(zhat - m), chunk by chunk in the same fixed order.
    m_safe = tl.where(m == float("-inf"), 0.0, m)
    s = tl.zeros((), dtype=tl.float32)
    for start in range(0, tile_size, BLOCK_V):
        cols = start + tl.arange(0, BLOCK_V)
        mask = cols < real_cols
        z = tl.load(z_ptr + row_base + cols, mask=mask, other=0.0)
        zhat = _softcap(z, multiplier, softcap)
        s += tl.sum(tl.where(mask, tl.exp(zhat - m_safe), 0.0))

    # Selected-token soft-capped logit (same formula -> same bits as in the stream above).
    target = tl.load(targets_ptr + row)
    local = target - col_begin
    in_tile = (local >= 0) & (local < real_cols)
    z_t = tl.load(z_ptr + row_base + local, mask=in_tile, other=0.0)
    zhat_t = tl.where(in_tile, _softcap(z_t, multiplier, softcap), 0.0)

    tl.store(m_ptr + row, m)
    tl.store(s_ptr + row, s)
    tl.store(tgt_logit_ptr + row, zhat_t)
    tl.store(tgt_in_ptr + row, in_tile.to(tl.int8))


@triton.jit
def _softcap_logprob_dz_kernel(
    z_ptr,  # [T, tile_size] fp32 logits of ONE tile (recomputed in backward)
    logz_ptr,  # [T] fp32 full-vocabulary logsumexp of zhat
    dlogp_ptr,  # [T] fp32 upstream gradient of the selected logprob
    targets_ptr,  # [T] int64 global target ids
    dz_ptr,  # [T, tile_size] bf16 out: gradient w.r.t. the raw fp32 logits z
    tile_size,
    real_cols,
    col_begin,
    stride_z_row,
    stride_dz_row,
    multiplier,
    softcap,
    BLOCK_V: tl.constexpr,
):
    row = tl.program_id(0)
    chunk = tl.program_id(1)
    cols = chunk * BLOCK_V + tl.arange(0, BLOCK_V)
    in_tile = cols < tile_size
    real = cols < real_cols
    row64 = row.to(tl.int64)

    z = tl.load(z_ptr + row64 * stride_z_row + cols, mask=real, other=0.0)
    th = libdevice.tanh(tl.div_rn(z * multiplier, softcap))
    zhat = softcap * th
    logz = tl.load(logz_ptr + row)
    g = tl.load(dlogp_ptr + row)
    target = tl.load(targets_ptr + row)

    p = tl.exp(zhat - logz)
    onehot = tl.where((cols.to(tl.int64) + col_begin) == target, 1.0, 0.0)
    dzhat = (onehot - p) * g
    dz = dzhat * (multiplier * (1.0 - th * th))
    dz = tl.where(real, dz, 0.0)
    tl.store(dz_ptr + row64 * stride_dz_row + cols, dz.to(tl.bfloat16), mask=in_tile)


@triton.jit
def _softcap_apply_kernel(z_ptr, out_ptr, n_elements, multiplier, softcap, BLOCK: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    z = tl.load(z_ptr + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, _softcap(z, multiplier, softcap), mask=mask)


def _check_logits_tile(z: torch.Tensor, tile_size: int) -> torch.Tensor:
    if z.device.type not in ("cuda", "hip"):
        raise RuntimeError(f"{KERNEL_ID} requires a GPU tensor, got device {z.device}")
    if z.dim() != 2 or z.shape[1] != tile_size:
        raise ValueError(f"logits tile must be [T, {tile_size}], got {tuple(z.shape)}")
    if z.dtype != torch.float32:
        raise TypeError(f"logits tile must be float32 (contract O7), got {z.dtype}")
    if z.stride(1) != 1:
        z = z.contiguous()
    return z


def _check_rows(name: str, t: torch.Tensor, rows: int, dtype: torch.dtype, device) -> None:
    if t.shape != (rows,) or t.dtype != dtype or t.device != device:
        raise ValueError(f"{name} must be a [{rows}] {dtype} tensor on {device}")


def launch_tile_stats(
    z: torch.Tensor,
    targets: torch.Tensor,
    *,
    tile_size: int,
    real_cols: int,
    col_begin: int,
    multiplier: float,
    softcap: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-row ``(m_t, s_t, zhat[target], target_in_tile)`` of one fp32 logits tile."""

    z = _check_logits_tile(z, tile_size)
    rows = z.shape[0]
    _check_rows("targets", targets, rows, torch.int64, z.device)
    m = torch.empty(rows, device=z.device, dtype=torch.float32)
    s = torch.empty(rows, device=z.device, dtype=torch.float32)
    tgt_logit = torch.empty(rows, device=z.device, dtype=torch.float32)
    tgt_in = torch.empty(rows, device=z.device, dtype=torch.int8)
    if rows == 0:
        return m, s, tgt_logit, tgt_in.bool()
    _softcap_tile_stats_kernel[(rows,)](
        z,
        targets.contiguous(),
        m,
        s,
        tgt_logit,
        tgt_in,
        int(tile_size),
        int(real_cols),
        int(col_begin),
        z.stride(0),
        float(multiplier),
        float(softcap),
        BLOCK_V=BLOCK_V,
        num_warps=NUM_WARPS,
    )
    return m, s, tgt_logit, tgt_in.bool()


def launch_dz(
    z: torch.Tensor,
    logz: torch.Tensor,
    dlogp: torch.Tensor,
    targets: torch.Tensor,
    *,
    tile_size: int,
    real_cols: int,
    col_begin: int,
    multiplier: float,
    softcap: float,
) -> torch.Tensor:
    """bf16 ``dz`` of one tile: ``(onehot - softmax) * dlogp * m * (1 - tanh^2)``, pads zero."""

    z = _check_logits_tile(z, tile_size)
    rows = z.shape[0]
    _check_rows("logz", logz, rows, torch.float32, z.device)
    _check_rows("dlogp", dlogp, rows, torch.float32, z.device)
    _check_rows("targets", targets, rows, torch.int64, z.device)
    dz = torch.empty((rows, tile_size), device=z.device, dtype=torch.bfloat16)
    if rows == 0:
        return dz
    grid = (rows, triton.cdiv(tile_size, BLOCK_V))
    _softcap_logprob_dz_kernel[grid](
        z,
        logz.contiguous(),
        dlogp.contiguous(),
        targets.contiguous(),
        dz,
        int(tile_size),
        int(real_cols),
        int(col_begin),
        z.stride(0),
        dz.stride(0),
        float(multiplier),
        float(softcap),
        BLOCK_V=BLOCK_V,
        num_warps=NUM_WARPS,
    )
    return dz


def launch_softcap(z: torch.Tensor, *, multiplier: float, softcap: float) -> torch.Tensor:
    """Elementwise ``zhat`` with exactly the kernel arithmetic (debug / sampling parity)."""

    if z.dtype != torch.float32:
        raise TypeError(f"softcap input must be float32, got {z.dtype}")
    z_c = z.contiguous()
    out = torch.empty_like(z_c)
    n = z_c.numel()
    if n == 0:
        return out
    _softcap_apply_kernel[(triton.cdiv(n, BLOCK_V),)](
        z_c, out, n, float(multiplier), float(softcap), BLOCK=BLOCK_V, num_warps=NUM_WARPS
    )
    return out


__all__ = [
    "BLOCK_V",
    "KERNEL_ID",
    "NUM_WARPS",
    "launch_dz",
    "launch_softcap",
    "launch_tile_stats",
]
