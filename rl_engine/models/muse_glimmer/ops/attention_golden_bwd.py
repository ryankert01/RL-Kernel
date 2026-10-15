# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Golden references for the Muse-Glimmer gated attention backward tests (contract O5).

Two references live here:

1. :func:`golden_forward_fixed_order` - a hand-written fixed-order fp32 forward with the
   contract's casts (p -> bf16 before PV, out -> bf16, y -> bf16, lse = m + log l). It follows
   the kernel's block structure (64-key blocks ascending from the first visible block, online
   softmax with m/l/acc updated in the contract's order) and evaluates every dot product as a
   sequential loop over the contraction index (no ``torch.matmul``). It is the forward the
   autograd Function falls back to while the sibling Triton forward is not importable, and the
   source of ``o``/``lse`` for the backward tests.

2. :func:`reference_backward` - the mathematical gradient of the un-quantised gated attention,
   computed by ``torch.autograd`` in fp64 (default; the contract allows an FP64 reference) or
   fp32. It uses ``torch.matmul`` because it is a tolerance reference, not a strict path and not
   the golden: its reduction order is irrelevant at the attention gradient tolerance.

Both work on the varlen layout: ``q (Tq,32,128)``, ``k/v (Tk,2,128)``, ``gate_logits
(Tq,4096)``, ``cu_seqlens_q/k`` int32 offsets, ``q_pos`` absolute query positions, keys of a
sequence at absolute positions ``0..Lk-1``.
"""

from __future__ import annotations

from typing import Optional

import torch

HEAD_DIM = 128
NUM_Q_HEADS = 32
NUM_KV_HEADS = 2
BLOCK_N = 64
SOFTMAX_SCALE = float(HEAD_DIM) ** -0.5


def _kv_index() -> torch.Tensor:
    """GQA map: q head h reads kv head h // 16."""
    return torch.arange(NUM_Q_HEADS) // (NUM_Q_HEADS // NUM_KV_HEADS)


def _visible(
    qpos: torch.Tensor, kpos: torch.Tensor, window: Optional[int]
) -> torch.Tensor:
    """Mask (Lq, Lk): global k <= q; local q - W < k <= q."""
    vis = kpos[None, :] <= qpos[:, None]
    if window is not None:
        vis = vis & ((qpos[:, None] - int(window)) < kpos[None, :])
    return vis


def _split(cu: torch.Tensor) -> list[tuple[int, int]]:
    cu_l = cu.to("cpu", torch.int64).tolist()
    return [(int(a), int(b)) for a, b in zip(cu_l[:-1], cu_l[1:])]


def golden_forward_fixed_order(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate_logits: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    q_pos: torch.Tensor,
    *,
    window: Optional[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fixed-order fp32 forward with the contract casts -> ``(y bf16, o bf16, lse fp32)``.

    Reduction order (per sequence, all heads together, rows independent):
      key blocks of 64 ascending from floor(max(0, p_min-W+1)/64) (0 on global layers) to
      floor(p_max/64); inside a block ``s = sum_{d ascending} q_d * k_d`` (sequential),
      ``m_new = max(m, rowmax)``, ``alpha = exp(m - m_new)``, ``p = exp(s - m_new)``,
      ``l = l*alpha + rowsum(p)``, ``acc = acc*alpha + sum_{j ascending} bf16(p_j) * v_j``.
    Casts: p -> bf16 before PV; out = acc/l -> bf16; y = out*sigmoid(g) fp32 -> bf16.
    """
    device = q.device
    tq = q.shape[0]
    kv_idx = _kv_index().to(device)
    y = torch.empty((tq, NUM_Q_HEADS * HEAD_DIM), dtype=torch.bfloat16, device=device)
    o = torch.empty_like(y)
    lse = torch.empty((tq, NUM_Q_HEADS), dtype=torch.float32, device=device)
    scale = SOFTMAX_SCALE
    neg_inf = float("-inf")

    for (q0, q1), (k0, k1) in zip(_split(cu_seqlens_q), _split(cu_seqlens_k)):
        if q1 == q0:
            continue
        qs = q[q0:q1].to(torch.float32)  # (Lq, Hq, D)
        ks = k[k0:k1].to(torch.float32).index_select(1, kv_idx)  # (Lk, Hq, D)
        vs = v[k0:k1].to(torch.float32).index_select(1, kv_idx)
        qpos = q_pos[q0:q1].to(torch.int64)
        lq, lk = q1 - q0, k1 - k0
        m = torch.full((lq, NUM_Q_HEADS), neg_inf, dtype=torch.float32, device=device)
        l = torch.zeros((lq, NUM_Q_HEADS), dtype=torch.float32, device=device)
        acc = torch.zeros((lq, NUM_Q_HEADS, HEAD_DIM), dtype=torch.float32, device=device)
        p_min = int(qpos.min())
        p_max = int(qpos.max())
        first = 0 if window is None else max(0, p_min - int(window) + 1) // BLOCK_N
        last = min(p_max // BLOCK_N, (lk - 1) // BLOCK_N) if lk > 0 else -1
        for kb in range(first, last + 1):
            j0, j1 = kb * BLOCK_N, min((kb + 1) * BLOCK_N, lk)
            kblk = ks[j0:j1].permute(1, 0, 2)  # (Hq, nk, D)
            vblk = vs[j0:j1]  # (nk, Hq, D)
            kpos = torch.arange(j0, j1, device=device)
            s = torch.zeros((lq, NUM_Q_HEADS, j1 - j0), dtype=torch.float32, device=device)
            for d in range(HEAD_DIM):
                s = s + qs[:, :, d, None] * kblk[None, :, :, d]
            s = s * scale
            vis = _visible(qpos, kpos, window)[:, None, :]
            s = torch.where(vis, s, neg_inf)
            m_new = torch.maximum(m, s.amax(dim=-1))
            alive = m_new > neg_inf
            alpha = torch.exp(torch.where(alive, m - m_new, 0.0))
            p = torch.where(vis, torch.exp(s - m_new[..., None]), 0.0)
            p = torch.where(alive[..., None], p, 0.0)
            l = l * alpha + p.sum(dim=-1)
            p_bf = p.to(torch.bfloat16).to(torch.float32)
            acc = acc * alpha[..., None]
            for j in range(j1 - j0):
                acc = acc + p_bf[:, :, j, None] * vblk[j][None, :, :]
            m = m_new
        out = (acc / l[..., None]).to(torch.bfloat16)
        o[q0:q1] = out.reshape(lq, -1)
        lse[q0:q1] = m + torch.log(l)
        g = gate_logits[q0:q1].to(torch.float32)
        y[q0:q1] = (out.to(torch.float32).reshape(lq, -1) * torch.sigmoid(g)).to(torch.bfloat16)
    return y, o, lse


def reference_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate_logits: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    q_pos: torch.Tensor,
    *,
    window: Optional[int],
) -> torch.Tensor:
    """Un-quantised differentiable forward in the dtype of ``q`` (fp32/fp64), ``y (Tq,4096)``.

    Uses ``torch.matmul``; this is the tolerance reference, not a strict path (see module doc).
    """
    device = q.device
    kv_idx = _kv_index().to(device)
    outs = []
    for (q0, q1), (k0, k1) in zip(_split(cu_seqlens_q), _split(cu_seqlens_k)):
        if q1 == q0:
            continue
        qs = q[q0:q1]  # (Lq, Hq, D)
        ks = k[k0:k1].index_select(1, kv_idx)  # (Lk, Hq, D)
        vs = v[k0:k1].index_select(1, kv_idx)
        qpos = q_pos[q0:q1].to(torch.int64)
        kpos = torch.arange(k1 - k0, device=device)
        s = torch.matmul(qs.permute(1, 0, 2), ks.permute(1, 2, 0)) * SOFTMAX_SCALE  # (Hq,Lq,Lk)
        vis = _visible(qpos, kpos, window)[None]
        s = torch.where(vis, s, torch.full_like(s, float("-inf")))
        p = torch.softmax(s, dim=-1)
        out = torch.matmul(p, vs.permute(1, 0, 2)).permute(1, 0, 2)  # (Lq, Hq, D)
        g = gate_logits[q0:q1].to(q.dtype)
        outs.append(out.reshape(q1 - q0, -1) * torch.sigmoid(g))
    return torch.cat(outs, dim=0)


def reference_backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate_logits: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    q_pos: torch.Tensor,
    dy: torch.Tensor,
    *,
    window: Optional[int],
    ref_dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(dq, dk, dv, dgate)`` in ``ref_dtype`` via torch.autograd of :func:`reference_forward`.

    Inputs are the bf16 tensors the strict path sees (so both differentiate the same function
    of the same bf16 values); the result is the exact-arithmetic gradient up to ``ref_dtype``.
    """
    qd = q.detach().to(ref_dtype).requires_grad_(True)
    kd = k.detach().to(ref_dtype).requires_grad_(True)
    vd = v.detach().to(ref_dtype).requires_grad_(True)
    gd = gate_logits.detach().to(ref_dtype).requires_grad_(True)
    y = reference_forward(qd, kd, vd, gd, cu_seqlens_q, cu_seqlens_k, q_pos, window=window)
    grads = torch.autograd.grad(y, (qd, kd, vd, gd), grad_outputs=dy.to(ref_dtype))
    return grads[0], grads[1], grads[2], grads[3]


__all__ = [
    "BLOCK_N",
    "SOFTMAX_SCALE",
    "golden_forward_fixed_order",
    "reference_backward",
    "reference_forward",
]
