# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Deterministic Triton backward for the Muse-Glimmer gated GQA attention (contract O5).

Rows: ``global_gated_attention`` (full causal) and ``local_gated_attention``
(sliding window W=2048). Forward semantics (sibling kernel ``gated_attention_fwd``):

    s_ij = (q_i . k_j) * 128^-0.5      masked to -inf outside the causal / window mask
    p_ij = softmax_j(s_ij)             (online softmax over 64-key blocks, fp32)
    o_i  = sum_j p_ij v_j  -> bf16     (pre-gate output, saved)
    y_i  = o_i * sigmoid(g_i) -> bf16  (gate, row-local over the H_Q*128 channels)
    lse  = m + log l (fp32, saved)

Backward (FlashAttention-2 recompute style, no atomics):

  preprocess (row-local, one program per (token, head), fp32):
    sg      = sigmoid_rn(g) = div_rn(1, 1 + exp(-g))   the forward's own sigmoid (shared helper)
    dgate   = dy * o * sg * (1 - sg)                   fp32 -> bf16   (returned)
    do      = dy * sg                                  fp32 -> bf16   (DECLARED CAST, feeds dots)
    D_i     = sum_d do_bf16[d] * o_bf16[d]             fp32 (tl.sum over the 128-lane block)

  dK/dV — query-chunk rule (CP-invariant; replaces "query blocks ascending" of contract v1):
    The queries of a sequence of Lk keys (absolute positions 0..Lk-1) are split into
    NUM_QUERY_CHUNKS = 8 canonical contiguous chunks on ABSOLUTE positions,
        chunk c = [floor(c*Lk/8), floor((c+1)*Lk/8))         (empty chunks allowed, Lk < 8)
    and every chunk into 64-query tiles aligned at the chunk start (the last tile of a chunk is
    cut at the chunk end; rows absent from the call are masked to exact zeros).
    Leaf (c, g) for canonical Q-head group g (4 heads = TP8 shard), one program per
    (64-key block, sequence, leaf):
      for h in the 4 heads of g (ascending):
        for tile of chunk c ascending (only tiles that can see the key block):
          sT  = tl.dot(K_blk, Q_tile^T) * scale                    masked -> -inf
          pT  = exp(sT - lse)
          dV += tl.dot(pT -> bf16, dO_tile)
          dPT = tl.dot(V_blk, dO_tile^T)
          dST = pT * (dPT - D)
          dK += tl.dot(dST -> bf16, Q_tile)
      leaf_dK = dK * scale (fp32), leaf_dV = dV (fp32)
    Group partial  P_g = tree_c(leaf (c, g)) = ((c0+c1)+(c2+c3))+((c4+c5)+(c6+c7)) fp32
    dK/dV (head h) = (P_g0+P_g1)+(P_g2+P_g3) over the 4 groups g0..g3 reading KV head h
                     (O3 tree, fp32) -> bf16 once.
    Why: a CP rank holding the contiguous chunk(s) [8i/c, 8(i+1)/c) of the sequence computes
    the same leaves for its chunks and exact zero leaves for the others, so its tree value is
    the sub-tree root of its chunks; the rank roots combined in the same tree reproduce the
    one-shot value bit for bit (CP1 == CP2 == CP4 == this kernel). A TP rank holding groups
    [8r/t, 8(r+1)/t) emits its P_g folded into its node of the per-KV-head tree; the t/2
    replicas of a KV head (TP4/TP8) fold their nodes in rank order and reproduce TP1.

  dQ, one program per (64-query tile, sequence, head), tiles aligned at the call's first query
  (per-row result independent of the tiling):
    for key block kb ascending within [first visible block, last visible block]:
        s   = tl.dot(Q_tile, K_blk^T) * scale masked -> -inf ; p = exp(s - lse)
        dP  = tl.dot(dO_tile, V_blk^T) ; dS = p * (dP - D)
        dQ += tl.dot(dS -> bf16, K_blk)        fp32 acc
    dQ *= scale -> bf16.

Head counts: ``num_q_heads`` / ``num_kv_heads`` (defaults 32 / 2) select a TP shard (16/1, 8/1,
4/1); the per-head arithmetic is unchanged, so a rank's dq/dgate rows and group partials equal
the matching slices of the TP1 call bit for bit.

Determinism: every reduction is a fixed-shape ``tl.dot`` (bf16 x bf16 -> fp32, no TF32) or a
fixed-shape ``tl.sum``; cross-tile/-block accumulation orders are stated above; the chunk and
group trees are exact fp32 adds in a fixed association. Visiting a fully masked tile adds
exact zeros. No atomics, no split-K, no autotune: the launch constants are part of the
arithmetic contract (``trace()``).

Cast points (p -> bf16 before the dV dot, dS -> bf16 before the dQ/dK dots, do -> bf16) are
the standard FA2 casts; they are declared here and in ``trace()`` because contract O5 v1 leaves
them implicit. ``dy`` must already be bf16 (no silent cast).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import triton
import triton.language as tl

from rl_engine.models.muse_glimmer.ops.canonical_tree import combine_tree
from rl_engine.models.muse_glimmer.triton.gated_attention_fwd import (
    SUPPORTED_HEAD_GEOMETRIES,
    check_head_geometry,
    sigmoid_rn,
)

HEAD_DIM = 128
NUM_Q_HEADS = 32
NUM_KV_HEADS = 2
#: Canonical Q-head group (= TP8 shard): 4 heads, 8 groups per layer at TP1.
HEADS_PER_GROUP = 4
NUM_GROUPS = NUM_Q_HEADS // HEADS_PER_GROUP
#: Canonical contiguous query chunks of a sequence (dK/dV reduction; CP-invariant).
NUM_QUERY_CHUNKS = 8
SOFTMAX_SCALE = float(HEAD_DIM) ** -0.5

# Pinned launch constants (R6). Not tuned at runtime.
BLOCK_N = 64  # keys per block, aligned at absolute key position 0 (contract O5)
BLOCK_M = 64  # queries per tile (dK/dV: aligned at the chunk start; dQ: at the first query)
_NUM_WARPS = 8
_NUM_STAGES = 2
_PRE_NUM_WARPS = 4

#: v2 = query-chunk rule for dK/dV (bits of dK/dV differ from v1; dq/dgate unchanged).
KERNEL_ID = "muse_glimmer.triton.gated_attention_bwd.v2"


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
    sg = sigmoid_rn(g)  # the forward's sigmoid (IEEE div_rn), not Triton's approximate `/`
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
    leaf_dk_ptr,
    leaf_dv_ptr,
    Tk,
    scale,
    WINDOW,
    H: tl.constexpr,
    HKV: tl.constexpr,
    HPG: tl.constexpr,
    G: tl.constexpr,
    NC: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_LOCAL: tl.constexpr,
):
    """dK/dV leaf (query chunk c, Q-head group g) of one 64-key block.

    Grid: (max key blocks, sequences, G * NC); leaf index = c * G + g. Accumulation order:
    heads of the group ascending, then the chunk's 64-query tiles ascending (tiles aligned at
    the chunk start). Leaves are written to ``(NC, G, Tk, D)`` fp32 (KV head implicit: g's).
    """
    kb = tl.program_id(0)
    b = tl.program_id(1)
    leaf = tl.program_id(2)
    c = leaf // G
    g = leaf % G

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

    # Canonical query chunk c of this sequence, absolute positions [c_lo, c_hi).
    c_lo = (c * Lk) // NC
    c_hi = ((c + 1) * Lk) // NC
    # Absolute positions that are in chunk c, present in this call and can see this key block.
    p_lo = tl.maximum(tl.maximum(c_lo, kb * BLOCK_N), qpos0)
    p_hi = tl.minimum(c_hi - 1, qpos0 + Lq - 1)
    if IS_LOCAL:
        p_hi = tl.minimum(p_hi, kb * BLOCK_N + (BLOCK_N - 1) + (WINDOW - 1))
    j_lo = (p_lo - c_lo) // BLOCK_M
    j_end = tl.where(p_hi >= p_lo, (p_hi - c_lo) // BLOCK_M + 1, j_lo)  # tiles [j_lo, j_end)

    acc_dk = tl.zeros((BLOCK_N, D), dtype=tl.float32)
    acc_dv = tl.zeros((BLOCK_N, D), dtype=tl.float32)

    for hh in tl.static_range(HPG):
        h = g * HPG + hh
        for j in range(j_lo, j_end):
            qpos = c_lo + j * BLOCK_M + offs_m  # absolute positions of the tile rows
            t = qpos - qpos0  # index inside this call's query rows
            q_valid = (t >= 0) & (t < Lq) & (qpos < c_hi)
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
    out_rows = (leaf.to(tl.int64) * Tk + (k_start + kpos).to(tl.int64)) * D
    tl.store(leaf_dk_ptr + out_rows[:, None] + offs_d[None, :], acc_dk, mask=k_valid[:, None])
    tl.store(leaf_dv_ptr + out_rows[:, None] + offs_d[None, :], acc_dv, mask=k_valid[:, None])


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
# Host side: validation and sequence metadata
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SequenceMeta:
    """Host-validated packing metadata of one backward call (see :func:`prepare_sequence_meta`).

    Computing it costs two device->host copies; a trainer calls :func:`prepare_sequence_meta`
    once per step and passes the object to every layer's backward (``seq_meta=``), which then
    performs no host synchronisation. The object remembers the exact tensors it was built from
    (identity + in-place version counters) and is rejected for any other tensors (fail closed).

    Precondition (not checkable without a sync): the tensors are only modified through PyTorch
    ops while the meta is in use. A write that does not bump ``_version`` (a Triton/CUDA kernel
    or an NCCL collective filling a persistent buffer, a ``.data`` write) leaves a stale meta
    that is accepted. The forward kernel re-checks the packing on the device on every call
    (``cu_seqlens``, grid coverage, ``q_pos`` range; device-side assert), so a stale meta cannot
    make the forward read or write out of bounds; the backward kernels rely on the meta's grid
    sizes and ``qpos0``. Build a new meta whenever such a buffer is refilled.
    """

    batch: int
    max_q_blocks: int
    max_k_blocks: int
    num_k_tokens: int
    #: ``max(len_q)`` over the sequences: the forward wrapper's ``max_seqlen_q`` (grid size), so
    #: an autograd step given this object performs no host synchronisation at all.
    max_seqlen_q: int
    qpos0: torch.Tensor
    cu_seqlens_q: torch.Tensor
    cu_seqlens_k: torch.Tensor
    q_pos: torch.Tensor
    versions: tuple[int, int, int]


def prepare_sequence_meta(
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    q_pos: torch.Tensor,
    *,
    num_k_tokens: int,
) -> SequenceMeta:
    """Validate the packing once (host sync) and return the reusable :class:`SequenceMeta`.

    Checks: int32 1-D offsets on one device, same batch size, start at 0, non-decreasing,
    ``cu_seqlens_q[-1] == len(q_pos)``, ``cu_seqlens_k[-1] == num_k_tokens``, contiguous
    ascending query positions inside every sequence and ``0 <= q_pos < len_k`` (queries lie
    inside their key sequence).
    """
    for name, t in (("cu_seqlens_q", cu_seqlens_q), ("cu_seqlens_k", cu_seqlens_k)):
        if t.dtype != torch.int32 or t.dim() != 1:
            raise ValueError(f"{name} must be a 1-D int32 tensor")
    if q_pos.dtype != torch.int32 or q_pos.dim() != 1:
        raise ValueError("q_pos must be a 1-D int32 tensor")
    if not (cu_seqlens_q.device == cu_seqlens_k.device == q_pos.device):
        raise ValueError("cu_seqlens_q, cu_seqlens_k and q_pos must be on one device")
    if cu_seqlens_q.numel() != cu_seqlens_k.numel() or cu_seqlens_q.numel() < 2:
        raise ValueError("cu_seqlens_q / cu_seqlens_k must have the same length >= 2")
    cu_q = cu_seqlens_q.to("cpu", torch.int64)
    cu_k = cu_seqlens_k.to("cpu", torch.int64)
    if int(cu_q[0]) != 0 or int(cu_k[0]) != 0:
        raise ValueError("cu_seqlens must start at 0")
    lens_q = cu_q[1:] - cu_q[:-1]
    lens_k = cu_k[1:] - cu_k[:-1]
    if bool((lens_q < 0).any()) or bool((lens_k < 0).any()):
        raise ValueError("cu_seqlens must be non-decreasing")
    if int(cu_q[-1]) != q_pos.numel():
        raise ValueError(f"cu_seqlens_q[-1]={int(cu_q[-1])} must equal Tq={q_pos.numel()}")
    if int(cu_k[-1]) != int(num_k_tokens):
        raise ValueError(
            f"cu_seqlens_k[-1]={int(cu_k[-1])} must equal the number of K/V rows {num_k_tokens}"
        )
    batch = int(lens_q.numel())
    pos_cpu = q_pos.to("cpu", torch.int64)
    qpos0_cpu = torch.zeros(batch, dtype=torch.int32, device="cpu")
    for b in range(batch):
        if int(lens_q[b]) == 0:
            continue
        seg = pos_cpu[int(cu_q[b]) : int(cu_q[b + 1])]
        expected = seg[0] + torch.arange(seg.numel(), dtype=torch.int64, device="cpu")
        if not torch.equal(seg, expected):
            raise ValueError(
                "gated_attention_bwd requires contiguous ascending q_pos inside every sequence"
            )
        if int(seg[0]) < 0 or int(seg[-1]) >= int(lens_k[b]):
            raise ValueError(
                f"sequence {b}: query positions [{int(seg[0])}, {int(seg[-1])}] must lie inside "
                f"its {int(lens_k[b])} keys"
            )
        qpos0_cpu[b] = int(seg[0])
    return SequenceMeta(
        batch=batch,
        max_q_blocks=int(math.ceil(int(lens_q.max()) / BLOCK_M)) if batch else 0,
        max_k_blocks=int(math.ceil(int(lens_k.max()) / BLOCK_N)) if batch else 0,
        num_k_tokens=int(num_k_tokens),
        max_seqlen_q=int(lens_q.max()) if batch else 0,
        qpos0=qpos0_cpu.to(q_pos.device),
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        q_pos=q_pos,
        versions=(cu_seqlens_q._version, cu_seqlens_k._version, q_pos._version),
    )


def _resolve_meta(
    seq_meta: Optional[SequenceMeta],
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    q_pos: torch.Tensor,
    num_k_tokens: int,
) -> SequenceMeta:
    if seq_meta is None:
        return prepare_sequence_meta(cu_seqlens_q, cu_seqlens_k, q_pos, num_k_tokens=num_k_tokens)
    same = (
        seq_meta.cu_seqlens_q is cu_seqlens_q
        and seq_meta.cu_seqlens_k is cu_seqlens_k
        and seq_meta.q_pos is q_pos
        and seq_meta.num_k_tokens == int(num_k_tokens)
    )
    if not same or seq_meta.versions != (
        cu_seqlens_q._version, cu_seqlens_k._version, q_pos._version
    ):
        raise ValueError(
            "seq_meta was prepared for different (or since modified) cu_seqlens/q_pos tensors; "
            "call prepare_sequence_meta on the tensors passed to this backward"
        )
    return seq_meta


def _check_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    q_pos: torch.Tensor,
    window: Optional[int],
    hq: int,
    hkv: int,
) -> None:
    if q.dim() != 3 or q.shape[1:] != (hq, HEAD_DIM):
        raise ValueError(f"q must be (Tq, {hq}, {HEAD_DIM}), got {tuple(q.shape)}")
    for name, t in (("k", k), ("v", v)):
        if t.dim() != 3 or t.shape[1:] != (hkv, HEAD_DIM):
            raise ValueError(f"{name} must be (Tk, {hkv}, {HEAD_DIM}), got {tuple(t.shape)}")
    if k.shape[0] != v.shape[0]:
        raise ValueError("k and v must have the same number of tokens")
    if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16 or v.dtype != torch.bfloat16:
        raise ValueError("q, k, v must be bf16")
    if not (q.is_cuda and k.is_cuda and v.is_cuda):
        raise ValueError("q, k, v must be CUDA tensors")
    for name, t in (
        ("k", k), ("v", v), ("cu_seqlens_q", cu_seqlens_q), ("cu_seqlens_k", cu_seqlens_k),
        ("q_pos", q_pos),
    ):
        if t.device != q.device:
            raise ValueError(f"{name} must be on q's device {q.device}, got {t.device}")
    if cu_seqlens_q.dtype != torch.int32 or cu_seqlens_k.dtype != torch.int32:
        raise ValueError("cu_seqlens must be int32")
    if cu_seqlens_q.numel() != cu_seqlens_k.numel() or cu_seqlens_q.numel() < 2:
        raise ValueError("cu_seqlens_q / cu_seqlens_k must have the same length >= 2")
    if q_pos.dtype != torch.int32 or q_pos.numel() != q.shape[0]:
        raise ValueError("q_pos must be int32 of length Tq")
    if window is not None and int(window) <= 0:
        raise ValueError("window must be None (global) or a positive int (local)")


# ---------------------------------------------------------------------------
# Host side: entry points
# ---------------------------------------------------------------------------


def gated_attention_bwd_preprocess(
    dy: torch.Tensor,
    o: torch.Tensor,
    gate_logits: torch.Tensor,
    *,
    num_q_heads: int = NUM_Q_HEADS,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Row-local part of the backward.

    Returns ``(dgate bf16 (Tq,H_Q*128), do bf16 (Tq,H_Q*128), delta fp32 (Tq,H_Q))`` with
    ``sg = sigmoid_rn(g)`` (the forward's sigmoid), ``dgate = dy*o*sg*(1-sg)``, ``do = dy*sg``
    (fp32 -> bf16, declared cast) and ``delta = rowsum_d(do_bf16 * o)`` in fp32 per (token,
    head). All three inputs must be bf16 (fail closed: no silent cast of ``dy``).
    """
    hq = int(num_q_heads)
    if hq not in {g[0] for g in SUPPORTED_HEAD_GEOMETRIES}:
        raise ValueError(f"num_q_heads={hq} is not a supported per-rank head count")
    tq = dy.shape[0]
    width = hq * HEAD_DIM
    for name, t in (("dy", dy), ("o", o), ("gate_logits", gate_logits)):
        if t.shape != (tq, width):
            raise ValueError(f"{name} must be (Tq, {width}), got {tuple(t.shape)}")
        if t.dtype != torch.bfloat16:
            raise ValueError(f"{name} must be bf16, got {t.dtype} (no implicit cast)")
        if t.device != dy.device or not t.is_cuda:
            raise ValueError(f"{name} must be a CUDA tensor on dy's device")
    dy = dy.contiguous()
    o = o.contiguous()
    g = gate_logits.contiguous()
    dgate = torch.empty_like(o)
    do = torch.empty_like(o)
    delta = torch.empty((tq, hq), dtype=torch.float32, device=dy.device)
    if tq == 0:
        return dgate, do, delta
    _bwd_preprocess_kernel[(tq, hq)](
        dy, o, g, dgate, do, delta, H=hq, D=HEAD_DIM, num_warps=_PRE_NUM_WARPS
    )
    return dgate, do, delta


def strict_gated_attention_bwd_chunk_leaves(
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
    num_q_heads: int = NUM_Q_HEADS,
    num_kv_heads: int = NUM_KV_HEADS,
    seq_meta: Optional[SequenceMeta] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Core backward: ``(dq bf16 (Tq,H_Q,128), dk_leaves, dv_leaves)``.

    ``dk_leaves`` / ``dv_leaves`` are fp32 ``(8 query chunks, H_Q/4 groups, Tk, 128)``: leaf
    ``(c, g)`` is the contribution of the queries of canonical chunk ``c`` through the 4 heads of
    group ``g`` to KV head ``(4g) // (H_Q/H_KV)`` (module docstring). Exposed for tests and for
    CP/TP plumbing; :func:`strict_gated_attention_bwd_partials` folds the chunk axis.
    """
    hq, hkv = check_head_geometry(num_q_heads, num_kv_heads)
    _check_inputs(q, k, v, cu_seqlens_q, cu_seqlens_k, q_pos, window, hq, hkv)
    tq, tk = q.shape[0], k.shape[0]
    if do.shape != (tq, hq * HEAD_DIM) or do.dtype != torch.bfloat16:
        raise ValueError(f"do must be bf16 (Tq, {hq * HEAD_DIM})")
    if lse.shape != (tq, hq) or lse.dtype != torch.float32:
        raise ValueError(f"lse must be fp32 (Tq, {hq})")
    if delta.shape != (tq, hq) or delta.dtype != torch.float32:
        raise ValueError(f"delta must be fp32 (Tq, {hq})")
    meta = _resolve_meta(seq_meta, cu_seqlens_q, cu_seqlens_k, q_pos, tk)
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    do = do.contiguous()
    lse = lse.contiguous()
    delta = delta.contiguous()
    cu_q = cu_seqlens_q.contiguous()
    cu_k = cu_seqlens_k.contiguous()

    n_groups = hq // HEADS_PER_GROUP
    dq = torch.empty_like(q)
    leaf_shape = (NUM_QUERY_CHUNKS, n_groups, tk, HEAD_DIM)
    if tq == 0 or tk == 0 or meta.batch == 0:
        dq.zero_()
        zeros = torch.zeros(leaf_shape, dtype=torch.float32, device=q.device)
        return dq, zeros, zeros.clone()
    leaf_dk = torch.empty(leaf_shape, dtype=torch.float32, device=q.device)
    leaf_dv = torch.empty(leaf_shape, dtype=torch.float32, device=q.device)

    is_local = window is not None
    win = int(window) if is_local else 0
    common = dict(
        H=hq,
        HKV=hkv,
        D=HEAD_DIM,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        IS_LOCAL=is_local,
        num_warps=_NUM_WARPS,
        num_stages=_NUM_STAGES,
    )
    _bwd_dkdv_kernel[(meta.max_k_blocks, meta.batch, n_groups * NUM_QUERY_CHUNKS)](
        q,
        k,
        v,
        do,
        lse,
        delta,
        cu_q,
        cu_k,
        meta.qpos0,
        leaf_dk,
        leaf_dv,
        tk,
        SOFTMAX_SCALE,
        win,
        HPG=HEADS_PER_GROUP,
        G=n_groups,
        NC=NUM_QUERY_CHUNKS,
        **common,
    )
    _bwd_dq_kernel[(meta.max_q_blocks, meta.batch, hq)](
        q,
        k,
        v,
        do,
        lse,
        delta,
        cu_q,
        cu_k,
        meta.qpos0,
        dq,
        SOFTMAX_SCALE,
        win,
        **common,
    )
    return dq, leaf_dk, leaf_dv


def fold_query_chunks(leaves: torch.Tensor, num_kv_heads: int) -> torch.Tensor:
    """``(8, G, Tk, 128)`` chunk leaves -> ``(G, Tk, H_KV, 128)`` fp32 group partials.

    ``P_g = ((c0+c1)+(c2+c3))+((c4+c5)+(c6+c7))`` (canonical O3 tree, exact fp32 adds) placed
    at group g's KV head; the other KV head's slice of ``P_g`` is exactly zero.
    """
    if leaves.dim() != 4 or leaves.shape[0] != NUM_QUERY_CHUNKS or leaves.shape[-1] != HEAD_DIM:
        raise ValueError(f"expected (8, G, Tk, 128) leaves, got {tuple(leaves.shape)}")
    folded = combine_tree(list(leaves.unbind(0)))  # (G, Tk, D)
    n_groups, tk = folded.shape[0], folded.shape[1]
    hkv = int(num_kv_heads)
    if hkv == 1:
        return folded.unsqueeze(2)
    groups_per_kv = n_groups // hkv
    out = torch.zeros((n_groups, tk, hkv, HEAD_DIM), dtype=torch.float32, device=leaves.device)
    for h in range(hkv):
        gs = slice(h * groups_per_kv, (h + 1) * groups_per_kv)
        out[gs, :, h] = folded[gs]
    return out


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
    num_q_heads: int = NUM_Q_HEADS,
    num_kv_heads: int = NUM_KV_HEADS,
    seq_meta: Optional[SequenceMeta] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(dq bf16 (Tq,H_Q,128), dk_partials, dv_partials)``.

    ``dk_partials`` / ``dv_partials`` are fp32 ``(H_Q/4, Tk, H_KV, 128)`` (``(8, Tk, 2, 128)``
    at TP1): slice ``g`` holds group ``g``'s contribution (already folded over the 8 query
    chunks) at its KV head; the other KV head's slice is exactly zero. They are NOT combined
    over groups so a TP rank can hand its groups to the canonical TP tree;
    :func:`combine_kv_partials` is the reduction when one rank holds all groups of a KV head.
    """
    dq, leaf_dk, leaf_dv = strict_gated_attention_bwd_chunk_leaves(
        q, k, v, do, lse, delta, cu_seqlens_q, cu_seqlens_k, q_pos, window=window,
        num_q_heads=num_q_heads, num_kv_heads=num_kv_heads, seq_meta=seq_meta,
    )
    return dq, fold_query_chunks(leaf_dk, num_kv_heads), fold_query_chunks(leaf_dv, num_kv_heads)


def kv_partials_subtree(partials: torch.Tensor) -> torch.Tensor:
    """``(G, Tk, H_KV, 128)`` fp32 group partials -> ``(Tk, H_KV, 128)`` fp32 node (no rounding).

    Per KV head ``h``, the canonical tree over the ``G / H_KV`` consecutive groups that read
    ``h`` (only those; the other head's exact-zero slices are never added): TP1 / TP2 rank ->
    the 4-leaf tree ``(g0+g1)+(g2+g3)`` (final), TP4 rank -> its 2-leaf node, TP8 rank -> its
    leaf. Replicas of a KV head (TP4 / TP8) fold their nodes in rank order with the same tree.
    """
    if partials.dtype != torch.float32 or partials.dim() != 4:
        raise ValueError("expected fp32 (G, Tk, H_KV, 128) group partials")
    n_groups, _, hkv, _ = partials.shape
    if hkv < 1 or n_groups % hkv:
        raise ValueError(f"{n_groups} group partials do not split over {hkv} KV heads")
    per_kv = n_groups // hkv
    if hkv == 1:
        return combine_tree(list(partials.unbind(0)))
    return torch.stack(
        [
            combine_tree([partials[g, :, h] for g in range(h * per_kv, (h + 1) * per_kv)])
            for h in range(hkv)
        ],
        dim=1,
    )


def combine_kv_partials(partials: torch.Tensor) -> torch.Tensor:
    """(G, Tk, H_KV, 128) fp32 partials -> (Tk, H_KV, 128) bf16 (per-KV-head tree, one cast)."""
    return kv_partials_subtree(partials).to(torch.bfloat16)


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
    num_q_heads: int = NUM_Q_HEADS,
    num_kv_heads: int = NUM_KV_HEADS,
    seq_meta: Optional[SequenceMeta] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Full strict backward: ``(dq, dk, dv, dgate)`` all bf16.

    ``dq (Tq,H_Q,128)``, ``dk/dv (Tk,H_KV,128)`` (tree over the group partials), ``dgate
    (Tq,H_Q*128)``. Only for geometries where this call holds every group of its KV heads
    (TP1 32/2, TP2 16/1); a TP4/TP8 rank holds a fraction of its KV head's groups and must use
    :func:`strict_gated_attention_bwd_partials` + the cross-rank canonical tree (raises here).
    """
    hq, hkv = check_head_geometry(num_q_heads, num_kv_heads)
    if hq // hkv != NUM_Q_HEADS // NUM_KV_HEADS:
        raise ValueError(
            f"strict_gated_attention_bwd: a ({hq}, {hkv}) shard holds only part of its KV "
            "head's Q-head groups, so its dK/dV are partials; call "
            "strict_gated_attention_bwd_partials and reduce across TP ranks with the canonical "
            "tree (runtime/tp_canonical_tree.combine_tree)"
        )
    dgate, do, delta = gated_attention_bwd_preprocess(dy, o, gate_logits, num_q_heads=hq)
    dq, dk_part, dv_part = strict_gated_attention_bwd_partials(
        q, k, v, do, lse, delta, cu_seqlens_q, cu_seqlens_k, q_pos, window=window,
        num_q_heads=hq, num_kv_heads=hkv, seq_meta=seq_meta,
    )
    return dq, combine_kv_partials(dk_part), combine_kv_partials(dv_part), dgate


def trace() -> dict:
    """Provenance record (contract R4/R6)."""
    return {
        "op": "muse_glimmer.gated_attention_bwd",
        "kernel_id": KERNEL_ID,
        "dtype_in": "bfloat16 (q,k,v,gate_logits,o,dy); float32 lse",
        "dtype_acc": "float32",
        "dtype_out": "bfloat16 (dq,dk,dv,dgate); float32 group partials / chunk leaves",
        "reduction_order": (
            "dKdV: leaf (query chunk c, group g of 4 heads) per 64-key block: heads ascending, "
            "then the chunk's 64-query tiles ascending (tiles aligned at the chunk start), "
            "tl.dot bf16->fp32, *scale (dK); chunks c = [floor(c*Lk/8), floor((c+1)*Lk/8)) on "
            "absolute positions; P_g = tree over the 8 chunk leaves ((0+1)+(2+3))+((4+5)+(6+7)); "
            "dK/dV of KV head h = the same tree over the 4 groups reading h -> bf16 once. dQ: "
            "per 64-query tile, key blocks "
            "of 64 ascending (first=floor(max(0,p-W+1)/64), last=floor(p/64)), tl.dot "
            "bf16->fp32 -> bf16. delta: tl.sum over 128 lanes per (token, head)."
        ),
        "query_chunk_rule": {
            "num_chunks": NUM_QUERY_CHUNKS,
            "boundaries": "floor(c * Lk / 8), Lk = keys of the sequence (absolute positions)",
            "tail_rule": (
                "any Lk >= 1: chunk lengths differ by at most 1, empty chunks when Lk < 8 are "
                "exact zero leaves; no alignment requirement"
            ),
            "tile_alignment": "64-query tiles aligned at each chunk start, cut at the chunk end",
            "cp_invariance": "CP rank i of c owns chunks [8i/c, 8(i+1)/c) when Lk % c == 0",
        },
        "cast_points": (
            "sg=div_rn(1,1+exp(-g)) fp32 (forward's sigmoid_rn); do=dy*sg fp32->bf16; p fp32->bf16 "
            "before the dV dot; dS fp32->bf16 before the dQ/dK dots; dgate fp32->bf16; "
            "dq/dk/dv fp32->bf16 once; dy must be bf16 (no implicit cast)"
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
        "head_geometries": [list(g) for g in SUPPORTED_HEAD_GEOMETRIES],
        "kv_grad_rule": (
            "per canonical Q-head group (4 heads = TP8 shard): P_g folded over 8 query chunks; "
            "per KV head its 4 groups folded by the canonical tree ((g0+g1)+(g2+g3)) at any TP; "
            "TP4/TP8 replicas fold their rank nodes in rank order (ranks hold partials only)"
        ),
        "combine_tree_impl": "rl_engine.models.muse_glimmer.ops.canonical_tree.combine_tree",
        "host_sync": "prepare_sequence_meta (2 D2H copies) unless seq_meta= is passed",
        "softmax_scale": SOFTMAX_SCALE,
    }


__all__ = [
    "BLOCK_M",
    "BLOCK_N",
    "HEADS_PER_GROUP",
    "KERNEL_ID",
    "NUM_GROUPS",
    "NUM_QUERY_CHUNKS",
    "SOFTMAX_SCALE",
    "SequenceMeta",
    "combine_kv_partials",
    "fold_query_chunks",
    "gated_attention_bwd_preprocess",
    "kv_partials_subtree",
    "prepare_sequence_meta",
    "strict_gated_attention_bwd",
    "strict_gated_attention_bwd_chunk_leaves",
    "strict_gated_attention_bwd_partials",
    "trace",
]
