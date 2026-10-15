# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Golden references for the Muse-Glimmer gated attention backward tests (contract O5).

Two references live here:

1. :func:`golden_forward_fixed_order` - a varlen ADAPTER over THE forward golden of the op,
   :func:`rl_engine.models.muse_glimmer.ops.gated_attention.gated_attention_forward_fp32`
   (fixed-order fp32 blocked online softmax with the contract casts). It loops over the packed
   sequences and returns ``(y bf16, o bf16, lse fp32)``, i.e. the golden rounded at the
   kernel's output cast points. There is exactly one O5 forward golden; this module does not
   re-implement it. The backward tests take ``o``/``lse`` from the strict forward kernel (the
   tensors a real training step saves) and use the goldens only as accuracy references.

2. :func:`reference_backward` - the mathematical gradient of the un-quantised gated attention,
   computed by ``torch.autograd`` in fp64 (default; the contract allows an FP64 reference) or
   fp32. It uses ``torch.matmul`` because it is a tolerance reference, not a strict path and not
   the golden: its reduction order is irrelevant at the attention gradient tolerance.

Both work on the varlen layout: ``q (Tq,H_Q,128)``, ``k/v (Tk,H_KV,128)``, ``gate_logits
(Tq,H_Q*128)``, ``cu_seqlens_q/k`` int32 offsets, ``q_pos`` absolute query positions, keys of a
sequence at absolute positions ``0..Lk-1``; GQA: q head ``h`` reads kv head ``h // (H_Q/H_KV)``.
"""

from __future__ import annotations

from typing import Optional

import torch

from rl_engine.models.muse_glimmer.ops.gated_attention import gated_attention_forward_fp32

HEAD_DIM = 128
NUM_Q_HEADS = 32
NUM_KV_HEADS = 2
BLOCK_N = 64
SOFTMAX_SCALE = float(HEAD_DIM) ** -0.5


def _kv_index(num_q_heads: int = NUM_Q_HEADS, num_kv_heads: int = NUM_KV_HEADS) -> torch.Tensor:
    """GQA map: q head h reads kv head h // (H_Q / H_KV)."""
    return torch.arange(num_q_heads) // (num_q_heads // num_kv_heads)


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
    """Varlen adapter over ``gated_attention_forward_fp32`` -> ``(y bf16, o bf16, lse fp32)``.

    Per sequence ``b``: ``gated_attention_forward_fp32(q_b, k_b, v_b, g_b, q_pos_b, window)`` in
    fp32 (reduction order and casts: see that function), then ``y`` and ``o`` rounded to bf16
    once (the kernel's output casts). Bitwise identical to calling the forward golden
    sequence by sequence (``test_golden_adapter_is_the_forward_golden``).
    """
    tq, hq = q.shape[0], q.shape[1]
    y = torch.zeros((tq, hq * HEAD_DIM), dtype=torch.bfloat16, device=q.device)
    o = torch.zeros_like(y)
    lse = torch.full((tq, hq), float("-inf"), dtype=torch.float32, device=q.device)
    for (q0, q1), (k0, k1) in zip(_split(cu_seqlens_q), _split(cu_seqlens_k)):
        if q1 == q0:
            continue
        yb, ob, lb = gated_attention_forward_fp32(
            q[q0:q1], k[k0:k1], v[k0:k1], gate_logits[q0:q1], q_pos[q0:q1], window
        )
        y[q0:q1] = yb.to(torch.bfloat16)
        o[q0:q1] = ob.to(torch.bfloat16)
        lse[q0:q1] = lb
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
    kv_idx = _kv_index(q.shape[1], k.shape[1]).to(device)
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
