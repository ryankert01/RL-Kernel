# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Deterministic Triton backward for the Muse-Glimmer gated GQA attention (contract O5).

Rows: ``global_gated_attention`` (full causal) and ``local_gated_attention_train``
(sliding window W=2048). Forward semantics (sibling kernel ``gated_attention_fwd``):

    s_ij = (q_i . k_j) * 128^-0.5      masked to -inf outside the causal / window mask
    p_ij = softmax_j(s_ij)             (online softmax over 64-key blocks, fp32)
    o_i  = sum_j p_ij v_j  -> bf16     (pre-gate output, saved)
    y_i  = o_i * sigmoid(g_i) -> bf16  (gate, row-local over 4096 channels)
    lse  = m + log l (fp32, saved)

Backward (FlashAttention-2 recompute style, no atomics):

  preprocess (row-local, one program per (token, head), fp32):
    sg      = sigmoid(g)                       fp32
    dgate   = dy * o * sg * (1 - sg)           fp32 -> bf16   (returned)
    do      = dy * sg                          fp32 -> bf16   (DECLARED CAST, feeds the dots)
    D_i     = sum_d do_bf16[d] * o_bf16[d]     fp32 (tl.sum over the 128-lane block)

  dK/dV kernel, one program per (64-key block, sequence, canonical Q-head group g of 4 heads):
    for h in heads of g (ascending):
      for query block qb ascending over the blocks that can see this key block:
        sT  = tl.dot(K_blk, Q_tile^T) * scale  (64k x 64q, fp32 acc)   masked -> -inf
        pT  = exp(sT - lse)                                             fp32
        dV += tl.dot(pT -> bf16, dO_tile)                               fp32 acc
        dPT = tl.dot(V_blk, dO_tile^T)                                  fp32 acc
        dST = pT * (dPT - D)                                            fp32
        dK += tl.dot(dST -> bf16, Q_tile)                               fp32 acc
    dK *= scale (fp32); emitted as fp32 PARTIALS dK_g / dV_g of shape (8, Tk, Hkv, 128).
    The final dK/dV is the canonical tree ((0+1)+(2+3))+((4+5)+(6+7)) over the 8 groups
    (fp32) -> bf16, evaluated identically at every TP (O3 tree / RFC #501 open question 2).

  dQ kernel, one program per (64-query block, sequence, head):
    for key block kb ascending within [first visible block, last visible block]:
        s   = tl.dot(Q_tile, K_blk^T) * scale  (64q x 64k) masked -> -inf
        p   = exp(s - lse)
        dP  = tl.dot(dO_tile, V_blk^T)
        dS  = p * (dP - D)
        dQ += tl.dot(dS -> bf16, K_blk)        fp32 acc
    dQ *= scale -> bf16.

Determinism: every reduction is a fixed-shape ``tl.dot`` (bf16 x bf16 -> fp32, no TF32) or a
fixed-shape ``tl.sum``; the cross-block accumulation order is the ascending block order
stated above; blocks are aligned at the sequence-local index 0 (= absolute position 0 for
keys), so a sequence's blocks are the same objects whatever its packing offset. Visiting a
fully masked block adds exact zeros (p = 0, dS = 0) and is bitwise neutral. No atomics,
no split-K, no autotune: the launch constants below are part of the arithmetic contract.

Cast points (p -> bf16 before the dV dot, dS -> bf16 before the dQ/dK dots, do -> bf16) are
the standard FA2 casts; they are declared here because contract O5 leaves them implicit.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import triton
import triton.language as tl

HEAD_DIM = 128
NUM_Q_HEADS = 32
NUM_KV_HEADS = 2
#: Canonical Q-head group (= TP8 shard): 4 heads, 8 groups per layer.
HEADS_PER_GROUP = 4
NUM_GROUPS = NUM_Q_HEADS // HEADS_PER_GROUP
SOFTMAX_SCALE = float(HEAD_DIM) ** -0.5

# Pinned launch constants (R6). Not tuned at runtime.
BLOCK_N = 64  # keys per block, aligned at absolute key position 0 (contract O5)
BLOCK_M = 64  # queries per tile in both backward kernels
_NUM_WARPS = 8
_NUM_STAGES = 2
_PRE_NUM_WARPS = 4

KERNEL_ID = "muse_glimmer.triton.gated_attention_bwd.v1"


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------


@triton.jit
def _bwd_preprocess_kernel(
    dy_ptr,
    o_ptr,
    g_ptr,
    dgate_ptr,
    do_ptr,
    delta_ptr,
    H: tl.constexpr,
    D: tl.constexpr,
):
    """Row-local gate VJP + do cast + D = rowsum(do * o); one program per (token, head)."""
    tok = tl.program_id(0)
    h = tl.program_id(1)
    offs = tok.to(tl.int64) * (H * D) + h * D + tl.arange(0, D)
    dy = tl.load(dy_ptr + offs).to(tl.float32)
    o = tl.load(o_ptr + offs).to(tl.float32)
    g = tl.load(g_ptr + offs).to(tl.float32)
    sg = 1.0 / (1.0 + tl.exp(-g))
    dgate = dy * o * (sg * (1.0 - sg))
    do = dy * sg
    do_bf = do.to(tl.bfloat16)
    tl.store(dgate_ptr + offs, dgate.to(tl.bfloat16))
    tl.store(do_ptr + offs, do_bf)
    delta = tl.sum(do_bf.to(tl.float32) * o, axis=0)
    tl.store(delta_ptr + tok.to(tl.int64) * H + h, delta)


@triton.jit
def _bwd_dkdv_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    do_ptr,
    lse_ptr,
    delta_ptr,
    cu_q_ptr,
    cu_k_ptr,
    qpos0_ptr,
    dk_part_ptr,
    dv_part_ptr,
    Tk,
    scale,
    WINDOW,
    H: tl.constexpr,
    HKV: tl.constexpr,
    HPG: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_LOCAL: tl.constexpr,
):
    """dK/dV partial of one 64-key block for one canonical Q-head group.

    Grid: (max key blocks, sequences, groups). Accumulation order: heads of the group
    ascending, then query tiles ascending; each tile contributes one fixed-shape tl.dot.
    """
    kb = tl.program_id(0)
    b = tl.program_id(1)
    g = tl.program_id(2)

    q_start = tl.load(cu_q_ptr + b)
    q_end = tl.load(cu_q_ptr + b + 1)
    k_start = tl.load(cu_k_ptr + b)
    k_end = tl.load(cu_k_ptr + b + 1)
    Lq = q_end - q_start
    Lk = k_end - k_start
    if kb * BLOCK_N >= Lk:
        return
    qpos0 = tl.load(qpos0_ptr + b)

    heads_per_kv: tl.constexpr = H // HKV
    kvh = (g * HPG) // heads_per_kv

    offs_n = tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)

    kpos = kb * BLOCK_N + offs_n  # absolute key position == index inside the sequence
    k_valid = kpos < Lk
    k_rows = (k_start + kpos).to(tl.int64) * (HKV * D) + kvh * D
    k_blk = tl.load(k_ptr + k_rows[:, None] + offs_d[None, :], mask=k_valid[:, None], other=0.0)
    v_blk = tl.load(v_ptr + k_rows[:, None] + offs_d[None, :], mask=k_valid[:, None], other=0.0)

    # Query tiles (sequence-local index t, position qpos0 + t) that can see this key block.
    lo_pos = kb * BLOCK_N
    t_lo = tl.maximum(lo_pos - qpos0, 0)
    if IS_LOCAL:
        hi_pos = kb * BLOCK_N + (BLOCK_N - 1) + (WINDOW - 1)
        t_hi = tl.minimum(hi_pos - qpos0, Lq - 1)
    else:
        t_hi = Lq - 1
    qb_lo = t_lo // BLOCK_M
    qb_hi = t_hi // BLOCK_M  # inclusive; empty range when t_lo > t_hi

    acc_dk = tl.zeros((BLOCK_N, D), dtype=tl.float32)
    acc_dv = tl.zeros((BLOCK_N, D), dtype=tl.float32)

    for hh in tl.static_range(HPG):
        h = g * HPG + hh
        for qb in range(qb_lo, qb_hi + 1):
            t = qb * BLOCK_M + offs_m
            q_valid = t < Lq
            qpos = qpos0 + t
            q_rows = (q_start + t).to(tl.int64) * (H * D) + h * D
            q_tile = tl.load(
                q_ptr + q_rows[:, None] + offs_d[None, :], mask=q_valid[:, None], other=0.0
            )
            do_tile = tl.load(
                do_ptr + q_rows[:, None] + offs_d[None, :], mask=q_valid[:, None], other=0.0
            )
            lse = tl.load(lse_ptr + (q_start + t).to(tl.int64) * H + h, mask=q_valid, other=0.0)
            delta = tl.load(
                delta_ptr + (q_start + t).to(tl.int64) * H + h, mask=q_valid, other=0.0
            )
            lse = tl.where(lse == float("-inf"), 0.0, lse)

            sT = tl.dot(k_blk, tl.trans(q_tile), allow_tf32=False) * scale
            visible = k_valid[:, None] & q_valid[None, :] & (kpos[:, None] <= qpos[None, :])
            if IS_LOCAL:
                visible = visible & ((qpos[None, :] - WINDOW) < kpos[:, None])
            sT = tl.where(visible, sT, float("-inf"))
            pT = tl.exp(sT - lse[None, :])

            acc_dv += tl.dot(pT.to(tl.bfloat16), do_tile, allow_tf32=False)
            dPT = tl.dot(v_blk, tl.trans(do_tile), allow_tf32=False)
            dST = pT * (dPT - delta[None, :])
            acc_dk += tl.dot(dST.to(tl.bfloat16), q_tile, allow_tf32=False)

    acc_dk = acc_dk * scale
    group_base = g.to(tl.int64) * (Tk.to(tl.int64) * HKV * D)
    out_rows = group_base + (k_start + kpos).to(tl.int64) * (HKV * D) + kvh * D
    tl.store(dk_part_ptr + out_rows[:, None] + offs_d[None, :], acc_dk, mask=k_valid[:, None])
    tl.store(dv_part_ptr + out_rows[:, None] + offs_d[None, :], acc_dv, mask=k_valid[:, None])


@triton.jit
def _bwd_dq_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    do_ptr,
    lse_ptr,
    delta_ptr,
    cu_q_ptr,
    cu_k_ptr,
    qpos0_ptr,
    dq_ptr,
    scale,
    WINDOW,
    H: tl.constexpr,
    HKV: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_LOCAL: tl.constexpr,
):
    """dQ of one 64-query tile of one head; key blocks ascending, fp32 accumulate -> bf16."""
    qb = tl.program_id(0)
    b = tl.program_id(1)
    h = tl.program_id(2)

    q_start = tl.load(cu_q_ptr + b)
    q_end = tl.load(cu_q_ptr + b + 1)
    k_start = tl.load(cu_k_ptr + b)
    k_end = tl.load(cu_k_ptr + b + 1)
    Lq = q_end - q_start
    Lk = k_end - k_start
    if qb * BLOCK_M >= Lq:
        return
    qpos0 = tl.load(qpos0_ptr + b)

    heads_per_kv: tl.constexpr = H // HKV
    kvh = h // heads_per_kv

    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)

    t = qb * BLOCK_M + offs_m
    q_valid = t < Lq
    qpos = qpos0 + t
    q_rows = (q_start + t).to(tl.int64) * (H * D) + h * D
    q_tile = tl.load(q_ptr + q_rows[:, None] + offs_d[None, :], mask=q_valid[:, None], other=0.0)
    do_tile = tl.load(
        do_ptr + q_rows[:, None] + offs_d[None, :], mask=q_valid[:, None], other=0.0
    )
    lse = tl.load(lse_ptr + (q_start + t).to(tl.int64) * H + h, mask=q_valid, other=0.0)
    delta = tl.load(delta_ptr + (q_start + t).to(tl.int64) * H + h, mask=q_valid, other=0.0)
    lse = tl.where(lse == float("-inf"), 0.0, lse)

    # Key block range of this tile (contract O5): first = floor(max(0, p_first-W+1)/64),
    # last = floor(p_last/64); extra blocks visited because a tile spans positions are no-ops.
    pos_first = qpos0 + qb * BLOCK_M
    pos_last = qpos0 + tl.minimum(qb * BLOCK_M + (BLOCK_M - 1), Lq - 1)
    n_kblocks = (Lk + BLOCK_N - 1) // BLOCK_N
    if IS_LOCAL:
        kb_lo = tl.maximum(pos_first - WINDOW + 1, 0) // BLOCK_N
    else:
        kb_lo = 0
    kb_hi = tl.minimum(pos_last // BLOCK_N, n_kblocks - 1)

    acc = tl.zeros((BLOCK_M, D), dtype=tl.float32)
    for kb in range(kb_lo, kb_hi + 1):
        kpos = kb * BLOCK_N + offs_n
        k_valid = kpos < Lk
        k_rows = (k_start + kpos).to(tl.int64) * (HKV * D) + kvh * D
        k_blk = tl.load(
            k_ptr + k_rows[:, None] + offs_d[None, :], mask=k_valid[:, None], other=0.0
        )
        v_blk = tl.load(
            v_ptr + k_rows[:, None] + offs_d[None, :], mask=k_valid[:, None], other=0.0
        )

        s = tl.dot(q_tile, tl.trans(k_blk), allow_tf32=False) * scale
        visible = q_valid[:, None] & k_valid[None, :] & (kpos[None, :] <= qpos[:, None])
        if IS_LOCAL:
            visible = visible & ((qpos[:, None] - WINDOW) < kpos[None, :])
        s = tl.where(visible, s, float("-inf"))
        p = tl.exp(s - lse[:, None])

        dP = tl.dot(do_tile, tl.trans(v_blk), allow_tf32=False)
        dS = p * (dP - delta[:, None])
        acc += tl.dot(dS.to(tl.bfloat16), k_blk, allow_tf32=False)

    acc = acc * scale
    tl.store(
        dq_ptr + q_rows[:, None] + offs_d[None, :],
        acc.to(tl.bfloat16),
        mask=q_valid[:, None],
    )


# ---------------------------------------------------------------------------
# Host side
# ---------------------------------------------------------------------------


def _check_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    q_pos: torch.Tensor,
    window: Optional[int],
) -> None:
    if q.dim() != 3 or q.shape[1:] != (NUM_Q_HEADS, HEAD_DIM):
        raise ValueError(f"q must be (Tq, {NUM_Q_HEADS}, {HEAD_DIM}), got {tuple(q.shape)}")
    for name, t in (("k", k), ("v", v)):
        if t.dim() != 3 or t.shape[1:] != (NUM_KV_HEADS, HEAD_DIM):
            raise ValueError(
                f"{name} must be (Tk, {NUM_KV_HEADS}, {HEAD_DIM}), got {tuple(t.shape)}"
            )
    if k.shape[0] != v.shape[0]:
        raise ValueError("k and v must have the same number of tokens")
    if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16 or v.dtype != torch.bfloat16:
        raise ValueError("q, k, v must be bf16")
    if not (q.is_cuda and k.is_cuda and v.is_cuda):
        raise ValueError("q, k, v must be CUDA tensors")
    if cu_seqlens_q.dtype != torch.int32 or cu_seqlens_k.dtype != torch.int32:
        raise ValueError("cu_seqlens must be int32")
    if cu_seqlens_q.numel() != cu_seqlens_k.numel() or cu_seqlens_q.numel() < 2:
        raise ValueError("cu_seqlens_q / cu_seqlens_k must have the same length >= 2")
    if q_pos.dtype != torch.int32 or q_pos.numel() != q.shape[0]:
        raise ValueError("q_pos must be int32 of length Tq")
    if window is not None and int(window) <= 0:
        raise ValueError("window must be None (global) or a positive int (local)")


def _sequence_meta(
    cu_seqlens_q: torch.Tensor, cu_seqlens_k: torch.Tensor, q_pos: torch.Tensor
) -> tuple[int, int, int, torch.Tensor]:
    """Return (B, max_q_blocks, max_k_blocks, qpos0) and validate contiguous query positions.

    The kernels derive each tile's position range from ``qpos0[b] + local index`` (training
    and prefill layouts); the exact mask is still evaluated per element, so this only bounds
    the block loops. Non-contiguous positions are rejected rather than silently mis-bounded.
    """
    cu_q = cu_seqlens_q.to("cpu", torch.int64)
    cu_k = cu_seqlens_k.to("cpu", torch.int64)
    lens_q = cu_q[1:] - cu_q[:-1]
    lens_k = cu_k[1:] - cu_k[:-1]
    if bool((lens_q < 0).any()) or bool((lens_k < 0).any()):
        raise ValueError("cu_seqlens must be non-decreasing")
    if int(cu_q[-1]) != q_pos.numel():
        raise ValueError("cu_seqlens_q[-1] must equal Tq")
    batch = int(lens_q.numel())
    pos_cpu = q_pos.to("cpu", torch.int64)
    qpos0_cpu = torch.zeros(batch, dtype=torch.int32)
    for b in range(batch):
        if int(lens_q[b]) == 0:
            continue
        seg = pos_cpu[int(cu_q[b]) : int(cu_q[b + 1])]
        expected = seg[0] + torch.arange(seg.numel(), dtype=torch.int64)
        if not torch.equal(seg, expected):
            raise ValueError(
                "gated_attention_bwd requires contiguous ascending q_pos inside every sequence"
            )
        qpos0_cpu[b] = int(seg[0])
    qpos0 = qpos0_cpu.to(q_pos.device)
    max_q_blocks = int(math.ceil(int(lens_q.max()) / BLOCK_M)) if batch else 0
    max_k_blocks = int(math.ceil(int(lens_k.max()) / BLOCK_N)) if batch else 0
    return batch, max_q_blocks, max_k_blocks, qpos0


def gated_attention_bwd_preprocess(
    dy: torch.Tensor, o: torch.Tensor, gate_logits: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Row-local part of the backward.

    Returns ``(dgate bf16 (Tq,4096), do bf16 (Tq,4096), delta fp32 (Tq,32))`` with
    ``dgate = dy*o*sg*(1-sg)``, ``do = dy*sg`` (fp32 -> bf16, declared cast) and
    ``delta = rowsum_d(do_bf16 * o)`` in fp32 per (token, head).
    """
    tq = dy.shape[0]
    width = NUM_Q_HEADS * HEAD_DIM
    for name, t in (("dy", dy), ("o", o), ("gate_logits", gate_logits)):
        if t.shape != (tq, width):
            raise ValueError(f"{name} must be (Tq, {width}), got {tuple(t.shape)}")
    dy = dy.to(torch.bfloat16).contiguous()
    o = o.contiguous()
    g = gate_logits.contiguous()
    dgate = torch.empty_like(o)
    do = torch.empty_like(o)
    delta = torch.empty((tq, NUM_Q_HEADS), dtype=torch.float32, device=dy.device)
    if tq == 0:
        return dgate, do, delta
    _bwd_preprocess_kernel[(tq, NUM_Q_HEADS)](
        dy, o, g, dgate, do, delta, H=NUM_Q_HEADS, D=HEAD_DIM, num_warps=_PRE_NUM_WARPS
    )
    return dgate, do, delta


def strict_gated_attention_bwd_partials(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    do: torch.Tensor,
    lse: torch.Tensor,
    delta: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    q_pos: torch.Tensor,
    *,
    window: Optional[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Core backward: ``(dq bf16 (Tq,32,128), dk_partials, dv_partials)``.

    ``dk_partials`` / ``dv_partials`` are fp32 ``(8, Tk, 2, 128)``: slice ``g`` holds the
    contribution of canonical Q-head group ``g`` (heads 4g..4g+3) to its KV head
    ``(4g)//16``; the other KV head's slice of ``g`` is exactly zero. They are NOT combined
    here so the WS2 ``kv_head_replication`` row can reduce them across ranks with the
    canonical tree; :func:`combine_kv_partials` is the TP1 reduction.
    """
    _check_inputs(q, k, v, cu_seqlens_q, cu_seqlens_k, q_pos, window)
    tq, tk = q.shape[0], k.shape[0]
    if do.shape != (tq, NUM_Q_HEADS * HEAD_DIM) or do.dtype != torch.bfloat16:
        raise ValueError("do must be bf16 (Tq, 4096)")
    if lse.shape != (tq, NUM_Q_HEADS) or lse.dtype != torch.float32:
        raise ValueError("lse must be fp32 (Tq, 32)")
    if delta.shape != (tq, NUM_Q_HEADS) or delta.dtype != torch.float32:
        raise ValueError("delta must be fp32 (Tq, 32)")
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    do = do.contiguous()
    lse = lse.contiguous()
    delta = delta.contiguous()
    cu_q = cu_seqlens_q.contiguous()
    cu_k = cu_seqlens_k.contiguous()
    batch, max_q_blocks, max_k_blocks, qpos0 = _sequence_meta(cu_q, cu_k, q_pos)

    dq = torch.empty_like(q)
    part_shape = (NUM_GROUPS, tk, NUM_KV_HEADS, HEAD_DIM)
    dk_part = torch.zeros(part_shape, dtype=torch.float32, device=q.device)
    dv_part = torch.zeros(part_shape, dtype=torch.float32, device=q.device)
    if tq == 0 or tk == 0 or batch == 0:
        dq.zero_()
        return dq, dk_part, dv_part

    is_local = window is not None
    win = int(window) if is_local else 0
    common = dict(
        H=NUM_Q_HEADS,
        HKV=NUM_KV_HEADS,
        D=HEAD_DIM,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        IS_LOCAL=is_local,
        num_warps=_NUM_WARPS,
        num_stages=_NUM_STAGES,
    )
    _bwd_dkdv_kernel[(max_k_blocks, batch, NUM_GROUPS)](
        q,
        k,
        v,
        do,
        lse,
        delta,
        cu_q,
        cu_k,
        qpos0,
        dk_part,
        dv_part,
        tk,
        SOFTMAX_SCALE,
        win,
        HPG=HEADS_PER_GROUP,
        **common,
    )
    _bwd_dq_kernel[(max_q_blocks, batch, NUM_Q_HEADS)](
        q,
        k,
        v,
        do,
        lse,
        delta,
        cu_q,
        cu_k,
        qpos0,
        dq,
        SOFTMAX_SCALE,
        win,
        **common,
    )
    return dq, dk_part, dv_part


def _combine_tree_fp32(partials: torch.Tensor) -> torch.Tensor:
    """Fixed tree ((0+1)+(2+3))+((4+5)+(6+7)) over the leading dim (8 leaves), fp32.

    Private fallback for ``rl_engine.models.muse_glimmer.ops.canonical_tree.combine_tree``
    (owned by the tp_canonical_tree agent); byte-identical association order.
    """
    if partials.shape[0] != 8 or partials.dtype != torch.float32:
        raise ValueError("combine tree expects 8 fp32 leaves on dim 0")
    p = partials
    return ((p[0] + p[1]) + (p[2] + p[3])) + ((p[4] + p[5]) + (p[6] + p[7]))


def _resolve_combine_tree():
    try:  # pragma: no cover - depends on a sibling agent's file
        from rl_engine.models.muse_glimmer.ops.canonical_tree import combine_tree

        return combine_tree, "rl_engine.models.muse_glimmer.ops.canonical_tree.combine_tree"
    except ImportError:
        return _combine_tree_fp32, "gated_attention_bwd._combine_tree_fp32"


def combine_kv_partials(partials: torch.Tensor) -> torch.Tensor:
    """(8, Tk, 2, 128) fp32 group partials -> (Tk, 2, 128) bf16 via the canonical tree."""
    combine, _ = _resolve_combine_tree()
    return combine(partials).to(torch.bfloat16)


def strict_gated_attention_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate_logits: torch.Tensor,
    o: torch.Tensor,
    lse: torch.Tensor,
    dy: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    q_pos: torch.Tensor,
    *,
    window: Optional[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Full strict backward: ``(dq, dk, dv, dgate)`` all bf16.

    ``dq (Tq,32,128)``, ``dk/dv (Tk,2,128)`` (tree-combined over the 8 group partials),
    ``dgate (Tq,4096)``.
    """
    dgate, do, delta = gated_attention_bwd_preprocess(dy, o, gate_logits)
    dq, dk_part, dv_part = strict_gated_attention_bwd_partials(
        q, k, v, do, lse, delta, cu_seqlens_q, cu_seqlens_k, q_pos, window=window
    )
    return dq, combine_kv_partials(dk_part), combine_kv_partials(dv_part), dgate


def trace() -> dict:
    """Provenance record (contract R4/R6)."""
    _, tree_impl = _resolve_combine_tree()
    return {
        "op": "muse_glimmer.gated_attention_bwd",
        "kernel_id": KERNEL_ID,
        "dtype_in": "bfloat16 (q,k,v,gate_logits,o,dy); float32 lse",
        "dtype_acc": "float32",
        "dtype_out": "bfloat16 (dq,dk,dv,dgate); float32 group partials",
        "reduction_order": (
            "dKdV: per (64-key block, group of 4 heads): heads ascending, query tiles of 64 "
            "ascending, tl.dot bf16->fp32; 8 group partials combined by the fixed tree "
            "((0+1)+(2+3))+((4+5)+(6+7)) in fp32 -> bf16. dQ: per 64-query tile, key blocks of "
            "64 ascending (first=floor(max(0,p-W+1)/64), last=floor(p/64)), tl.dot bf16->fp32 "
            "-> bf16. delta: tl.sum over 128 lanes per (token, head)."
        ),
        "cast_points": (
            "do=dy*sigmoid(g) fp32->bf16; p fp32->bf16 before the dV dot; dS fp32->bf16 before "
            "the dQ/dK dots; dgate fp32->bf16; dq/dk/dv fp32->bf16 once"
        ),
        "tile_config": {
            "BLOCK_M": BLOCK_M,
            "BLOCK_N": BLOCK_N,
            "HEAD_DIM": HEAD_DIM,
            "num_warps": _NUM_WARPS,
            "num_stages": _NUM_STAGES,
            "preprocess_num_warps": _PRE_NUM_WARPS,
            "allow_tf32": False,
            "autotune": False,
            "atomics": False,
        },
        "kv_grad_rule": "per canonical Q-head group (8 groups of 4 heads), tree at every TP",
        "combine_tree_impl": tree_impl,
        "softmax_scale": SOFTMAX_SCALE,
    }


__all__ = [
    "BLOCK_M",
    "BLOCK_N",
    "HEADS_PER_GROUP",
    "KERNEL_ID",
    "NUM_GROUPS",
    "SOFTMAX_SCALE",
    "combine_kv_partials",
    "gated_attention_bwd_preprocess",
    "strict_gated_attention_bwd",
    "strict_gated_attention_bwd_partials",
    "trace",
]
