# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Per-rank strict attention for the Megatron Bridge binding (contract O5, head-count params).

Head geometry at TP ``t``, rank ``r`` (``H = 32`` Q heads, ``H_KV = 2`` KV heads, ``D = 128``):

====  ===========  ============  =======================  ====================================
 t    Q heads      KV heads      KV head of rank r        K/V column shard of the rank
====  ===========  ============  =======================  ====================================
 1    32 (all)     2 (all)       -                        both heads (256 cols)
 2    16           1             r                        that head (128 cols)
 4    8            1 (replica)   (2r)//4: ranks {0,1}->0  half of the head (64 cols)
 8    4            1 (replica)   (2r)//8: ranks 0-3 -> 0  a quarter of the head (32 cols)
====  ===========  ============  =======================  ====================================

* Q heads ``[r*32/t, (r+1)*32/t)`` and the matching gate / ``y`` columns are exactly the rank's
  canonical column-parallel q / gate rows and the rank's row-parallel ``o_proj`` input columns,
  so q, gate and ``y`` never leave the rank.
* At ``t >= 4`` a KV head is replicated on ``t/2`` ranks. :class:`GatherKVHead` all-gathers the
  rank's k / v column shard over the TP group (rank-ordered data movement) and keeps the 128
  columns of the rank's KV head. Its backward returns ONLY the rank's own columns of the (already
  replica-reduced, identical on every replica) gradient, so a K/V projection row's gradient is
  produced by exactly one rank: no double counting.
* :class:`RankGatedAttention` runs the sibling strict kernels with
  ``num_q_heads = 32/t``, ``num_kv_heads = max(1, 2/t)``. Forward ``(y, o, lse)`` == the TP1
  slice bitwise (measured by the attention row). Backward: ``dgate``/``do``/``delta`` row-local,
  ``dq`` the TP1 slice; dK/dV: the rank's group partials (folded over the 8 canonical query
  chunks) -> ``kv_partials_subtree`` node; at ``t >= 4`` the ``t/2`` replica nodes of the KV head
  are all-gathered in rank order and folded with the canonical tree (no reducing collective)
  BEFORE the single bf16 rounding -> equals TP1 ``dk[:, kvh]`` bitwise on every replica.
  TP1/TP2: the node is final (== the sibling fused ``strict_gated_attention_bwd``).
"""

from __future__ import annotations

from typing import Any

import torch

from rl_engine.models.muse_glimmer.runtime.strict_kernels import AttentionKernels, TPComm

NUM_Q_HEADS = 32
NUM_KV_HEADS = 2
HEAD_DIM = 128


def rank_head_geometry(rank: int, world: int) -> dict[str, int]:
    """``{q_heads, kv_heads, q_head_start, kv_head, replicas}`` of TP rank ``rank``."""
    if world not in (1, 2, 4, 8) or not 0 <= rank < world:
        raise ValueError(f"rank {rank} of TP {world} is not a supported Muse Glimmer layout")
    hq = NUM_Q_HEADS // world
    hkv = max(1, NUM_KV_HEADS // world)
    kvh = (NUM_KV_HEADS * rank) // world if world >= NUM_KV_HEADS else 0
    replicas = max(1, world // NUM_KV_HEADS)
    return {
        "q_heads": hq,
        "kv_heads": hkv,
        "q_head_start": rank * hq,
        "kv_head": kvh,
        "replicas": replicas,
    }


def kv_replica_ranks(rank: int, world: int) -> list[int]:
    """Ranks holding the same (replicated) KV head as ``rank``, in rank order."""
    geo = rank_head_geometry(rank, world)
    if geo["replicas"] == 1:
        return [rank]
    first = geo["kv_head"] * geo["replicas"]
    return list(range(first, first + geo["replicas"]))


class GatherKVHead(torch.autograd.Function):
    """``(T, 256/t)`` k or v column shard -> ``(T, H_KV*128)`` the rank's whole KV head(s).

    TP1/TP2: identity (the shard already holds whole heads). TP4/TP8: rank-ordered all-gather of
    every rank's shard (data movement), keep KV head ``(2r)//t``. Backward keeps this rank's own
    columns of the head gradient (the head gradient is identical on every replica).
    """

    @staticmethod
    def forward(ctx, x_local: torch.Tensor, comm: TPComm) -> torch.Tensor:
        world, rank = comm.world, comm.rank
        ctx.passthrough = world <= NUM_KV_HEADS
        if ctx.passthrough:
            return x_local
        n_local = x_local.shape[1]
        if n_local * world != NUM_KV_HEADS * HEAD_DIM:
            raise RuntimeError(f"k/v shard width {n_local} at TP{world} is not 256/t")
        kvh = rank_head_geometry(rank, world)["kv_head"]
        full = torch.cat(comm.all_gather(x_local.contiguous()), dim=1)
        ctx.offset = rank * n_local - kvh * HEAD_DIM
        ctx.n_local = n_local
        if not 0 <= ctx.offset < HEAD_DIM:
            raise RuntimeError("rank's k/v columns are not inside its KV head")
        return full[:, kvh * HEAD_DIM : (kvh + 1) * HEAD_DIM].contiguous()

    @staticmethod
    def backward(ctx, grad_head: torch.Tensor):
        if ctx.passthrough:
            return grad_head, None
        return grad_head[:, ctx.offset : ctx.offset + ctx.n_local].contiguous(), None


class RankGatedAttention(torch.autograd.Function):
    """Strict gated attention on this rank's heads; see the module docstring for the rules."""

    @staticmethod
    def forward(ctx, q, k, v, gate, cu_q, cu_k, q_pos, window, comm, kernels: AttentionKernels):
        hq, hkv = int(q.shape[1]), int(k.shape[1])
        q_c, k_c, v_c, g_c = (t.contiguous() for t in (q, k, v, gate))
        y, o, lse = kernels.fwd(
            q_c, k_c, v_c, g_c, cu_q, cu_k, q_pos, window=window,
            num_q_heads=hq, num_kv_heads=hkv,
        )
        ctx.save_for_backward(q_c, k_c, v_c, g_c, o, lse, cu_q, cu_k, q_pos)
        ctx.window = window
        ctx.comm = comm
        ctx.kernels = kernels
        return y

    @staticmethod
    def backward(ctx, dy):
        q, k, v, g, o, lse, cu_q, cu_k, q_pos = ctx.saved_tensors
        kernels: AttentionKernels = ctx.kernels
        comm: TPComm = ctx.comm
        hq, hkv = int(q.shape[1]), int(k.shape[1])
        if dy.dtype != torch.bfloat16:
            raise TypeError(f"attention upstream gradient must be bf16, got {dy.dtype}")
        dgate, do, delta = kernels.preprocess(dy.contiguous(), o, g, num_q_heads=hq)
        dq, dk_part, dv_part = kernels.partials(
            q, k, v, do, lse, delta, cu_q, cu_k, q_pos, window=ctx.window,
            num_q_heads=hq, num_kv_heads=hkv,
        )
        node_k = kernels.subtree(dk_part)
        node_v = kernels.subtree(dv_part)
        del dk_part, dv_part
        replicas = kv_replica_ranks(comm.rank, comm.world)
        if len(replicas) > 1:
            # Replicated KV head: fold the replicas' fp32 nodes in rank order with the canonical
            # tree (all-gather = data movement), then round once.
            nodes_k = comm.all_gather(node_k.contiguous())
            nodes_v = comm.all_gather(node_v.contiguous())
            node_k = kernels.combine([nodes_k[r] for r in replicas])
            node_v = kernels.combine([nodes_v[r] for r in replicas])
        dk = node_k.to(torch.bfloat16)
        dv = node_v.to(torch.bfloat16)
        return dq, dk, dv, dgate, None, None, None, None, None, None


def rank_gated_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    cu_q: torch.Tensor,
    cu_k: torch.Tensor,
    q_pos: torch.Tensor,
    *,
    window: int | None,
    comm: TPComm,
    kernels: AttentionKernels,
) -> torch.Tensor:
    """``y (T, H_Q*128)`` bf16 of this rank's heads (strict kernels, or the labelled fallback)."""
    if kernels.strict:
        return RankGatedAttention.apply(q, k, v, gate, cu_q, cu_k, q_pos, window, comm, kernels)
    if kernels.fallback is None:
        raise RuntimeError("attention kernels are neither strict nor a fallback")
    if comm.world > NUM_KV_HEADS:
        raise RuntimeError(
            "the fallback attention cannot fold replicated KV-head gradients (TP > 2); "
            "strict kernels are required"
        )
    return kernels.fallback(q, k, v, gate, cu_q, cu_k, window)


def attention_core(
    q_local: torch.Tensor,
    k_local: torch.Tensor,
    v_local: torch.Tensor,
    g_local: torch.Tensor,
    *,
    ops: Any,
    comm: TPComm,
    cos: torch.Tensor | None,
    sin: torch.Tensor | None,
    cu_seqlens: torch.Tensor,
    positions: torch.Tensor,
    window: int | None,
    rms_eps: float,
    qk_scale: float,
) -> torch.Tensor:
    """Everything between the four column projections and ``o_proj`` for one rank.

    ``q_local``/``g_local`` ``(T, 4096/t)``, ``k_local``/``v_local`` ``(T, 256/t)`` (the rank's
    canonical column shards) -> ``y_local (T, 4096/t)`` bf16, the rank's ``o_proj`` input
    columns. ``cos``/``sin`` None selects a global (NoPE) layer.
    """
    t = q_local.shape[0]
    geo = rank_head_geometry(comm.rank, comm.world)
    hq, hkv = geo["q_heads"], geo["kv_heads"]
    k_head = GatherKVHead.apply(k_local, comm)
    v_head = GatherKVHead.apply(v_local, comm)
    q3 = q_local.view(t, hq, HEAD_DIM)
    k3 = k_head.view(t, hkv, HEAD_DIM)
    qh, kh = ops.qknorm_scaled_rope(q3, k3, cos, sin, eps=rms_eps, qk_scale=qk_scale)
    return rank_gated_attention(
        qh, kh, v_head.view(t, hkv, HEAD_DIM), g_local, cu_seqlens, cu_seqlens, positions,
        window=window, comm=comm, kernels=ops.attention,
    )


__all__ = [
    "GatherKVHead",
    "RankGatedAttention",
    "attention_core",
    "kv_replica_ranks",
    "rank_gated_attention",
    "rank_head_geometry",
]
