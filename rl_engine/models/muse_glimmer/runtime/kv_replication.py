# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""``kv_head_replication`` forward half (RFC #501, WS2): replicated-KV attention == TP1.

Muse-Glimmer has 32 Q heads and 2 KV heads; q head ``h`` reads kv head ``h // 16``. Under TP
``t`` rank ``r`` owns q heads ``[32r/t, 32(r+1)/t)`` and therefore needs exactly one KV head,
``floor(2r/t)``: TP2 needs no replication, TP4 / TP8 hold each KV head 2x / 4x. The declared
map lives in :class:`TPLayout.kv_ownership_map`; this module provides the forward proof.

Proof sketch (why TP t equals TP1 byte for byte): gated attention is separable per Q head.
``out_h = softmax(q_h k_{h//16}^T * s) v_{h//16}`` and ``y_h = out_h * sigmoid(g_h)`` read only
``q_h``, the one KV head and the 128 gate channels of head ``h``; no reduction crosses heads. A
rank runs the identical kernel on identical bits (the replicated K/V copies come from the same
deterministic ``k_proj`` / ``v_proj`` det_gemm, so every replica is byte-equal), hence
``y[:, cols(r)]`` at TP1 == rank r's output, and concatenating ranks in order == TP1. The only
requirement on the attention kernel is that a head's arithmetic does not depend on how many
heads share the launch -- which this module checks empirically for the kernel in use.

Attention kernel (fixed sibling interface, may be absent):
``rl_engine.models.muse_glimmer.ops.gated_attention.strict_gated_attention_fwd_varlen(q (T,Hq,128),
k (T,Hkv,128), v, gate_logits (T,Hq*128), cu_seqlens_q, cu_seqlens_k, q_pos, *, window) ->
(y, o, lse)``. Fallback: a labelled pure-torch fp32 implementation of the O5 arithmetic (online
softmax over 64-key blocks aligned at absolute position 0, ``p -> bf16`` before PV, ``acc / l ->
bf16``, gate in fp32 -> bf16) whose reductions are elementwise pairwise trees, so its per-head
bits are independent of the number of heads in the launch and of the batch.

Backward half (NOT implemented here; stated for provenance): dK/dV and the ``k_proj`` /
``v_proj`` weight gradients of KV head ``g`` are produced per canonical 4-head group (one per
canonical TP8 rank) and summed in the 4-leaf sub-tree ``(c0+c1)+(c2+c3)`` of the canonical tree
(canonical ranks 0-3 -> KV 0, 4-7 -> KV 1) at every physical TP including TP1. The sibling
kernel ``triton.gated_attention_bwd.strict_gated_attention_bwd_partials`` (same package)
already emits the fp32 ``(8, Tk, 2, 128)`` group partials this rule consumes; under TP t a rank
holds the ``8/t`` partials of its own groups and the same all-gather + tree as ``o_proj`` gives
TP1 bits (follow-up row).
"""

from __future__ import annotations

from typing import Any

import torch

from rl_engine.models.muse_glimmer.runtime.tp_layout import TPLayout
from rl_engine.models.muse_glimmer.workload import OFFICIAL_FINGERPRINT

Tensor = torch.Tensor

BLOCK_N = 64
BLOCK_M = 64
SLIDING_WINDOW = int(OFFICIAL_FINGERPRINT["sliding_window"])

_sibling_attention = None
ATTENTION_BACKEND = "fallback:pure_torch_fp32_blockwise_online_softmax"
for _module_name in (
    # Strict Triton forward (the gated_attention agent's kernel; also what
    # ops.gated_attention_autograd.strict_forward binds when importable).
    "rl_engine.models.muse_glimmer.triton.gated_attention_fwd",
    # Name used in the WS2 brief for the same fixed interface.
    "rl_engine.models.muse_glimmer.ops.gated_attention",
):
    try:
        import importlib

        _sibling_attention = importlib.import_module(_module_name).strict_gated_attention_fwd_varlen
        ATTENTION_BACKEND = f"{_module_name}.strict_gated_attention_fwd_varlen"
        break
    except (ImportError, AttributeError):  # pragma: no cover - depends on sibling branch
        continue


# --------------------------------------------------------------------------- #
# Fallback attention (O5 arithmetic in pure torch)
# --------------------------------------------------------------------------- #


def _tree_sum_last(x: Tensor) -> Tensor:
    """Pairwise-halves sum over the last dim (power of two); elementwise ops only."""

    while x.size(-1) > 1:
        half = x.size(-1) // 2
        x = x[..., :half] + x[..., half:]
    return x[..., 0]


def _tree_sum_dim(x: Tensor, dim: int) -> Tensor:
    x = x.movedim(dim, -1)
    return _tree_sum_last(x)


def _fallback_gated_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    gate_logits: Tensor,
    cu_seqlens: Tensor,
    positions: Tensor,
    *,
    window: int | None,
) -> Tensor:
    tokens, heads, head_dim = q.shape
    kv_heads = k.size(1)
    if heads % kv_heads:
        raise ValueError(f"{heads} q heads do not group over {kv_heads} kv heads")
    group = heads // kv_heads
    scale = float(head_dim) ** -0.5
    device = q.device
    out = torch.empty((tokens, heads, head_dim), dtype=torch.bfloat16, device=device)
    bounds = [int(b) for b in cu_seqlens.tolist()]
    pos_all = positions.to(torch.long)
    head_to_kv = torch.arange(heads, device=device) // group
    neg_inf = torch.tensor(float("-inf"), device=device)

    for seq in range(len(bounds) - 1):
        start, stop = bounds[seq], bounds[seq + 1]
        length = stop - start
        if length == 0:
            continue
        pos = pos_all[start:stop]
        first_pos = int(pos[0])
        if not bool(torch.equal(pos, torch.arange(first_pos, first_pos + length, device=device))):
            raise ValueError("fallback attention requires contiguous ascending positions per seq")
        qs = q[start:stop].float()  # (L, H, D)
        ks = k[start:stop].float()[:, head_to_kv]  # (L, H, D) replicated view per q head
        vs = v[start:stop].float()[:, head_to_kv]
        for q0 in range(0, length, BLOCK_M):
            q1 = min(length, q0 + BLOCK_M)
            q_pos = pos[q0:q1]  # (Mq,)
            tile_lo, tile_hi = int(q_pos[0]), int(q_pos[-1])
            q_tile = qs[q0:q1].permute(1, 0, 2)  # (H, Mq, D)
            m_run = torch.full((heads, q1 - q0), float("-inf"), device=device)
            l_run = torch.zeros((heads, q1 - q0), device=device)
            acc = torch.zeros((heads, q1 - q0, head_dim), device=device)
            if window is None:
                first_block = first_pos // BLOCK_N
            else:
                first_block = max(0, tile_lo - int(window) + 1) // BLOCK_N
                first_block = max(first_block, first_pos // BLOCK_N)
            last_block = tile_hi // BLOCK_N
            for block in range(first_block, last_block + 1):
                k_lo = max(block * BLOCK_N, first_pos)
                k_hi = min((block + 1) * BLOCK_N, first_pos + length)
                if k_hi <= k_lo:
                    continue
                k_idx0, k_idx1 = k_lo - first_pos, k_hi - first_pos
                k_pos = pos[k_idx0:k_idx1]
                k_blk = ks[k_idx0:k_idx1].permute(1, 0, 2)  # (H, Nk, D)
                v_blk = vs[k_idx0:k_idx1].permute(1, 0, 2)
                # scores = q . k over D (pairwise tree), fp32, scaled, masked
                scores = _tree_sum_last(q_tile[:, :, None, :] * k_blk[:, None, :, :]) * scale
                visible = k_pos[None, :] <= q_pos[:, None]
                if window is not None:
                    visible = visible & (q_pos[:, None] - k_pos[None, :] < int(window))
                scores = torch.where(visible[None], scores, neg_inf)
                m_new = torch.maximum(m_run, scores.amax(dim=-1))
                m_safe = torch.where(m_new == float("-inf"), torch.zeros_like(m_new), m_new)
                alpha = torch.exp(m_run - m_safe)
                p = torch.exp(scores - m_safe[..., None])
                l_run = l_run * alpha + _tree_sum_last(p)
                p_bf16 = p.to(torch.bfloat16).float()
                pv = _tree_sum_dim(p_bf16[..., None] * v_blk[:, None, :, :], dim=2)
                acc = acc * alpha[..., None] + pv
                m_run = m_new
            o_tile = (acc / l_run[..., None]).to(torch.bfloat16)  # (H, Mq, D)
            out[start + q0 : start + q1] = o_tile.permute(1, 0, 2)
    o_flat = out.reshape(tokens, heads * head_dim)
    gate = torch.sigmoid(gate_logits.float())
    return (o_flat.float() * gate).to(torch.bfloat16)


def gated_attention_fwd(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    gate_logits: Tensor,
    cu_seqlens: Tensor,
    positions: Tensor,
    *,
    window: int | None,
) -> Tensor:
    """Gated attention output ``y (T, Hq*128)`` bf16 (sibling strict kernel or the fallback)."""

    if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16 or v.dtype != torch.bfloat16:
        raise TypeError("gated attention requires bf16 q/k/v")
    if gate_logits.dtype != torch.bfloat16:
        raise TypeError("gate logits must be bf16 (attn_gate det_gemm output)")
    if q.dim() != 3 or k.dim() != 3 or v.dim() != 3 or k.shape != v.shape:
        raise ValueError("q (T, Hq, D), k/v (T, Hkv, D) expected")
    if gate_logits.shape != (q.size(0), q.size(1) * q.size(2)):
        raise ValueError(
            f"gate_logits must be (T, Hq*D) = {(q.size(0), q.size(1) * q.size(2))}; "
            f"got {tuple(gate_logits.shape)}"
        )
    if _sibling_attention is not None:
        y, _o, _lse = _sibling_attention(
            q, k, v, gate_logits, cu_seqlens, cu_seqlens, positions, window=window
        )
        return y
    return _fallback_gated_attention(q, k, v, gate_logits, cu_seqlens, positions, window=window)


# --------------------------------------------------------------------------- #
# Per-rank replicated-KV attention and the proof
# --------------------------------------------------------------------------- #


def rank_inputs(
    q: Tensor, k: Tensor, v: Tensor, gate_logits: Tensor, layout: TPLayout, rank: int
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Slice the rank's Q heads, its (replicated) KV head(s) and gate columns from full tensors.

    This is what a column-parallel ``q_proj`` / ``gate_proj`` and a replicated ``k_proj`` /
    ``v_proj`` would place on the rank; slicing full tensors stands in for them here.
    """

    heads = layout.q_heads(rank)
    kv = layout.kv_heads(rank)
    c0, c1 = layout.q_columns(rank)
    kv_index = torch.tensor(kv, device=k.device)
    return (
        q[:, heads.start : heads.stop].contiguous(),
        k.index_select(1, kv_index).contiguous(),
        v.index_select(1, kv_index).contiguous(),
        gate_logits[:, c0:c1].contiguous(),
    )


def rank_kv_replicated_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    gate_logits: Tensor,
    cu_seqlens: Tensor,
    positions: Tensor,
    layout: TPLayout,
    rank: int,
    *,
    window: int | None,
) -> Tensor:
    """``y (T, 32/tp * 128)`` bf16 for rank ``rank``: its Q heads against its replicated KV head."""

    q_r, k_r, v_r, g_r = rank_inputs(q, k, v, gate_logits, layout, rank)
    return gated_attention_fwd(q_r, k_r, v_r, g_r, cu_seqlens, positions, window=window)


def prove_kv_replication(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    gate_logits: Tensor,
    cu_seqlens: Tensor,
    positions: Tensor,
    *,
    window: int | None,
    tps: tuple[int, ...] = (1, 2, 4, 8),
    layout_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Single-device proof: for each TP, every rank's output equals the TP1 column slice.

    Returns ``{"tp1_sha256": ..., "per_tp": {tp: {"equal": bool, "ranks": [bool, ...],
    "concat_equal": bool}}, "attention_backend": ..., "kv_ownership": {tp: map}}``.
    """

    import hashlib

    kwargs = dict(layout_kwargs or {})
    y_tp1 = gated_attention_fwd(q, k, v, gate_logits, cu_seqlens, positions, window=window)
    report: dict[str, Any] = {
        "attention_backend": ATTENTION_BACKEND,
        "tp1_sha256": hashlib.sha256(y_tp1.contiguous().view(torch.uint8).cpu().numpy().tobytes())
        .hexdigest(),
        "per_tp": {},
        "kv_ownership": {},
    }
    for tp in tps:
        layout = TPLayout(tp=tp, **kwargs)
        parts = []
        ranks_equal = []
        for rank in range(tp):
            y_r = rank_kv_replicated_attention(
                q, k, v, gate_logits, cu_seqlens, positions, layout, rank, window=window
            )
            c0, c1 = layout.q_columns(rank)
            ranks_equal.append(bool(torch.equal(y_r, y_tp1[:, c0:c1])))
            parts.append(y_r)
        concat = torch.cat(parts, dim=1)
        report["per_tp"][tp] = {
            "equal": all(ranks_equal),
            "ranks": ranks_equal,
            "concat_equal": bool(torch.equal(concat, y_tp1)),
        }
        report["kv_ownership"][tp] = layout.kv_ownership_map()
    return report


def trace(layout: TPLayout) -> dict[str, Any]:
    return {
        "op": "kv_head_replication.forward",
        "dtype_in": "bfloat16",
        "dtype_acc": "float32",
        "dtype_out": "bfloat16",
        "reduction_order": (
            "per head: online softmax over 64-key blocks ascending from absolute position 0, "
            "p -> bf16 before PV, acc / l -> bf16, gate fp32 -> bf16; no cross-head reduction"
        ),
        "tile_config": {"BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N, "head_dim": layout.head_dim},
        "attention_backend": ATTENTION_BACKEND,
        "kv_ownership": layout.kv_ownership_map(),
        "backward_rule": (
            "dK/dV and k_proj/v_proj weight grads per canonical 4-head group, summed in the "
            "4-leaf sub-tree (c0+c1)+(c2+c3) per KV head at every physical TP (not implemented "
            "here; partials from triton.gated_attention_bwd.strict_gated_attention_bwd_partials)"
        ),
    }


__all__ = [
    "ATTENTION_BACKEND",
    "BLOCK_M",
    "BLOCK_N",
    "SLIDING_WINDOW",
    "gated_attention_fwd",
    "prove_kv_replication",
    "rank_inputs",
    "rank_kv_replicated_attention",
    "trace",
]
