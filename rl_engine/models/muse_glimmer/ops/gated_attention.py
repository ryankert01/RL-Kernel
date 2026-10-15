# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Strict gated attention for Muse-Glimmer-30B — forward wrappers and fp32 golden (O5).

Rows covered: ``global_gated_attention``, ``local_gated_attention_train`` and
``local_gated_attention_rollout`` of RFC #501. This module owns the FORWARD side only:

* :func:`strict_gated_attention_fwd_varlen` — packed (varlen) K/V: training, one-shot and
  chunked prefill.
* :func:`strict_gated_attention_fwd_paged` — paged K/V cache (page = 16 tokens): chunked
  prefill and decode in the rollout engine. Same kernel body, same bits.
* :func:`gated_attention_forward_fp32` — single-sequence fp32 (or fp64) golden written as an
  explicit fixed-order blocked online softmax (no SDPA, no matmul).
* :func:`trace` — contract R4 provenance dict.

The deterministic backward and the ``torch.autograd.Function`` live in a sibling module and
import the three functions above by name.

Numerics are defined in ``docs/contracts/muse-glimmer-30b-strict-contract.md`` section O5 and
restated in :mod:`rl_engine.models.muse_glimmer.triton.gated_attention_fwd`. Summary of the
reduction order and cast points (identical in kernel and golden):
KV blocks of 64 keys aligned at absolute position 0, ascending; fp32 scores ``q.k * 128**-0.5``;
online softmax ``m_new = max(m, rowmax)``, ``alpha = exp(m - m_new)``, ``p = exp(s - m_new)``,
``l = l*alpha + rowsum(p)``, ``acc = acc*alpha + (p -> bf16) @ V``; ``o = acc / l -> bf16``;
``y = (o_bf16 -> fp32) * sigmoid(g -> fp32) -> bf16``; ``lse = m + log(l)`` fp32.
"""

from __future__ import annotations

from typing import Any

import torch

from rl_engine.models.muse_glimmer.triton.gated_attention_fwd import (
    BLOCK_M_HEAD,
    BLOCK_M_TOKEN,
    BLOCK_N,
    HEAD_DIM,
    NUM_KV_HEADS,
    NUM_Q_HEADS,
    NUM_STAGES,
    NUM_WARPS_HEAD,
    NUM_WARPS_TOKEN,
    PAGE_SIZE,
    _gated_attention_fwd_kernel,
)

SLIDING_WINDOW = 2048
SOFTMAX_SCALE = HEAD_DIM**-0.5

#: Sequences whose longest query chunk is at most this long run on HEAD tiles (one program per
#: query position and KV head, 16 query heads per tile) instead of TOKEN tiles. Pure
#: performance choice: both tile modes are bitwise identical per row (test_gated_attention_fwd).
HEAD_TILE_MAX_Q = 16

TILE_MODES = ("token", "head")


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


def _validate_common(
    q: torch.Tensor, gate_logits: torch.Tensor, cu_seqlens_q: torch.Tensor, window: int | None
) -> int:
    _check(q.is_cuda, "q must be a CUDA tensor")
    _check(q.dtype == torch.bfloat16, f"q must be bf16, got {q.dtype}")
    _check(
        q.dim() == 3 and q.shape[1] == NUM_Q_HEADS and q.shape[2] == HEAD_DIM,
        f"q must be (Tq, {NUM_Q_HEADS}, {HEAD_DIM}), got {tuple(q.shape)}",
    )
    _check(q.is_contiguous(), "q must be contiguous")
    _check(gate_logits.dtype == torch.bfloat16, "gate_logits must be bf16")
    _check(
        gate_logits.shape == (q.shape[0], NUM_Q_HEADS * HEAD_DIM),
        f"gate_logits must be (Tq, {NUM_Q_HEADS * HEAD_DIM}), got {tuple(gate_logits.shape)}",
    )
    _check(gate_logits.is_contiguous() and gate_logits.device == q.device, "gate_logits layout")
    _check(
        cu_seqlens_q.dtype == torch.int32 and cu_seqlens_q.dim() == 1 and cu_seqlens_q.numel() >= 2,
        "cu_seqlens_q must be a 1-D int32 tensor with B+1 entries",
    )
    _check(cu_seqlens_q.device == q.device, "cu_seqlens_q must be on q's device")
    if window is not None:
        _check(int(window) > 0, "window must be a positive int or None")
    return int(cu_seqlens_q.numel() - 1)


def _resolve_max_seqlen_q(cu_seqlens_q: torch.Tensor, max_seqlen_q: int | None) -> int:
    if max_seqlen_q is None:
        lens = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
        max_seqlen_q = int(lens.max().item())
    _check(max_seqlen_q >= 1, "max_seqlen_q must be >= 1")
    return int(max_seqlen_q)


def _resolve_tile_mode(max_seqlen_q: int, tile_mode: str | None) -> str:
    if tile_mode is None:
        return "head" if max_seqlen_q <= HEAD_TILE_MAX_Q else "token"
    _check(tile_mode in TILE_MODES, f"tile_mode must be one of {TILE_MODES}")
    return tile_mode


def _launch(
    *,
    q: torch.Tensor,
    gate_logits: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_pos: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    max_pages: int,
    num_seqs: int,
    max_seqlen_q: int,
    window: int | None,
    paged: bool,
    tile_mode: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tq = q.shape[0]
    y = torch.empty((tq, NUM_Q_HEADS * HEAD_DIM), dtype=torch.bfloat16, device=q.device)
    o = torch.empty_like(y)
    lse = torch.empty((tq, NUM_Q_HEADS), dtype=torch.float32, device=q.device)
    if tq == 0:
        return y, o, lse

    local = window is not None
    if tile_mode == "head":
        grid = (max_seqlen_q, num_seqs, NUM_KV_HEADS)
        block_m, num_warps = BLOCK_M_HEAD, NUM_WARPS_HEAD
    else:
        grid = (triton_cdiv(max_seqlen_q, BLOCK_M_TOKEN), num_seqs, NUM_Q_HEADS)
        block_m, num_warps = BLOCK_M_TOKEN, NUM_WARPS_TOKEN

    _gated_attention_fwd_kernel[grid](
        q,
        gate_logits,
        k,
        v,
        y,
        o,
        lse,
        q_pos,
        cu_seqlens_q,
        cu_seqlens_k,
        seq_lens,
        block_table,
        float(SOFTMAX_SCALE),
        int(window) if local else 0,
        int(max_pages),
        H_Q=NUM_Q_HEADS,
        H_KV=NUM_KV_HEADS,
        D=HEAD_DIM,
        PAGE=PAGE_SIZE,
        BLOCK_M=block_m,
        BLOCK_N=BLOCK_N,
        LOCAL=local,
        PAGED=paged,
        HEAD_TILE=(tile_mode == "head"),
        num_warps=num_warps,
        num_stages=NUM_STAGES,
    )
    return y, o, lse


def triton_cdiv(a: int, b: int) -> int:
    return -(-a // b)


def strict_gated_attention_fwd_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate_logits: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    q_pos: torch.Tensor,
    *,
    window: int | None,
    max_seqlen_q: int | None = None,
    tile_mode: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Strict gated attention over packed sequences (training / prefill / chunked prefill).

    Args:
        q: ``(Tq, 32, 128)`` bf16 queries (after qk-norm, scale and RoPE).
        k, v: ``(Tk, 2, 128)`` bf16 keys/values. Keys of sequence ``i`` are
            ``k[cu_seqlens_k[i]:cu_seqlens_k[i+1]]`` at absolute positions ``0..len_k-1``.
        gate_logits: ``(Tq, 4096)`` bf16 output of the attention gate projection of the
            layer input (input_layernorm output).
        cu_seqlens_q, cu_seqlens_k: ``(B+1,)`` int32 packed offsets.
        q_pos: ``(Tq,)`` int32 absolute positions of the queries (chunked prefill: the queries
            are the last ``len_q`` positions of the sequence; ``q_pos`` says which).
        window: ``None`` for a global (full causal) layer, ``2048`` for a local layer.
        max_seqlen_q: optional host-side ``max(len_q)`` (avoids one device sync).
        tile_mode: ``"token"`` / ``"head"`` to force a tile shape; default picks by length.

    Returns:
        ``(y, o, lse)``: gated output bf16 ``(Tq, 4096)``, pre-gate output bf16 ``(Tq, 4096)``,
        ``lse = m + log(l)`` fp32 ``(Tq, 32)``.
    """
    num_seqs = _validate_common(q, gate_logits, cu_seqlens_q, window)
    for name, t in (("k", k), ("v", v)):
        _check(t.dtype == torch.bfloat16, f"{name} must be bf16")
        _check(
            t.dim() == 3 and t.shape[1] == NUM_KV_HEADS and t.shape[2] == HEAD_DIM,
            f"{name} must be (Tk, {NUM_KV_HEADS}, {HEAD_DIM}), got {tuple(t.shape)}",
        )
        _check(t.is_contiguous() and t.device == q.device, f"{name} layout/device")
    _check(k.shape == v.shape, "k and v must have the same shape")
    _check(
        cu_seqlens_k.dtype == torch.int32 and cu_seqlens_k.shape == cu_seqlens_q.shape,
        "cu_seqlens_k must be int32 with the same shape as cu_seqlens_q",
    )
    _check(cu_seqlens_k.device == q.device, "cu_seqlens_k must be on q's device")
    _check(
        q_pos.dtype == torch.int32 and q_pos.shape == (q.shape[0],) and q_pos.device == q.device,
        "q_pos must be int32 of shape (Tq,) on q's device",
    )
    max_seqlen_q = _resolve_max_seqlen_q(cu_seqlens_q, max_seqlen_q)
    mode = _resolve_tile_mode(max_seqlen_q, tile_mode)
    dummy = torch.zeros((1,), dtype=torch.int32, device=q.device)
    return _launch(
        q=q,
        gate_logits=gate_logits,
        k=k,
        v=v,
        q_pos=q_pos.contiguous(),
        cu_seqlens_q=cu_seqlens_q.contiguous(),
        cu_seqlens_k=cu_seqlens_k.contiguous(),
        seq_lens=dummy,
        block_table=dummy,
        max_pages=0,
        num_seqs=num_seqs,
        max_seqlen_q=max_seqlen_q,
        window=window,
        paged=False,
        tile_mode=mode,
    )


def paged_query_positions(
    cu_seqlens_q: torch.Tensor, seq_lens: torch.Tensor, total_q: int
) -> torch.Tensor:
    """Absolute positions of packed queries that are the last ``len_q`` tokens of each sequence.

    ``q_pos[t] = seq_lens[b] - len_q[b] + (t - cu_seqlens_q[b])`` for ``t`` in sequence ``b``.
    Pure device arithmetic (no host sync).
    """
    lens_q = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).to(torch.int64)
    num_seqs = lens_q.numel()
    seq_of_tok = torch.repeat_interleave(
        torch.arange(num_seqs, device=cu_seqlens_q.device), lens_q, output_size=total_q
    )
    starts = seq_lens.to(torch.int64) - lens_q
    tok = torch.arange(total_q, device=cu_seqlens_q.device)
    pos = starts[seq_of_tok] + tok - cu_seqlens_q[:-1].to(torch.int64)[seq_of_tok]
    return pos.to(torch.int32)


def strict_gated_attention_fwd_paged(
    q: torch.Tensor,
    gate_logits: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    *,
    window: int | None,
    max_seqlen_q: int | None = None,
    tile_mode: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Strict gated attention reading K/V from a paged cache (rollout prefill / decode).

    Args:
        q, gate_logits: as in :func:`strict_gated_attention_fwd_varlen`; the queries of
            sequence ``i`` are its last ``len_q_i`` positions (decode: ``len_q_i == 1``).
        k_cache, v_cache: ``(num_pages, 16, 2, 128)`` bf16.
        block_table: ``(B, max_pages)`` int32 physical page per logical page; ``-1`` marks a
            page that was never allocated or was evicted. Evicted pages are never read: the
            kernel masks every key at position ``<= q_pos - window`` before it touches the
            table, so eviction timing cannot change bits.
        seq_lens: ``(B,)`` int32 total keys per sequence including the new tokens, which the
            caller has already written to the cache.
        window: ``None`` (global) or ``2048`` (local).

    Returns:
        ``(y, o, lse)`` exactly as the varlen path, bitwise.
    """
    num_seqs = _validate_common(q, gate_logits, cu_seqlens_q, window)
    for name, t in (("k_cache", k_cache), ("v_cache", v_cache)):
        _check(t.dtype == torch.bfloat16, f"{name} must be bf16")
        _check(
            t.dim() == 4 and tuple(t.shape[1:]) == (PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM),
            f"{name} must be (num_pages, {PAGE_SIZE}, {NUM_KV_HEADS}, {HEAD_DIM}), "
            f"got {tuple(t.shape)}",
        )
        _check(t.is_contiguous() and t.device == q.device, f"{name} layout/device")
    _check(k_cache.shape == v_cache.shape, "k_cache and v_cache must have the same shape")
    _check(
        block_table.dtype == torch.int32 and block_table.dim() == 2
        and block_table.shape[0] == num_seqs and block_table.device == q.device,
        "block_table must be int32 (B, max_pages) on q's device",
    )
    _check(
        seq_lens.dtype == torch.int32 and seq_lens.shape == (num_seqs,)
        and seq_lens.device == q.device,
        "seq_lens must be int32 (B,) on q's device",
    )
    max_seqlen_q = _resolve_max_seqlen_q(cu_seqlens_q, max_seqlen_q)
    mode = _resolve_tile_mode(max_seqlen_q, tile_mode)
    cu_q = cu_seqlens_q.contiguous()
    seq_lens = seq_lens.contiguous()
    q_pos = paged_query_positions(cu_q, seq_lens, q.shape[0])
    return _launch(
        q=q,
        gate_logits=gate_logits,
        k=k_cache,
        v=v_cache,
        q_pos=q_pos,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_q,
        seq_lens=seq_lens,
        block_table=block_table.contiguous(),
        max_pages=int(block_table.shape[1]),
        num_seqs=num_seqs,
        max_seqlen_q=max_seqlen_q,
        window=window,
        paged=True,
        tile_mode=mode,
    )


# --------------------------------------------------------------------------------------------
# fp32 / fp64 golden
# --------------------------------------------------------------------------------------------


def gated_attention_forward_fp32(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate_logits: torch.Tensor,
    q_pos: torch.Tensor,
    window: int | None,
    *,
    dtype: torch.dtype = torch.float32,
    q_chunk: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Single-sequence golden: the O5 algorithm in ``dtype`` (fp32 default, fp64 allowed).

    Inputs are the bf16 tensors of one sequence: ``q (Tq, 32, 128)``, ``k/v (Tk, 2, 128)`` at
    absolute positions ``0..Tk-1``, ``gate_logits (Tq, 4096)``, ``q_pos (Tq,)`` int. Returns
    ``(y, o, lse)`` in ``dtype`` (unrounded; the kernel's bf16 outputs are compared to them
    with the WS1 attention tolerance).

    Reduction order: identical to the kernel — KV blocks of 64 keys aligned at absolute
    position 0, ascending, with the same online-softmax update order and the same casts
    (``p -> bf16`` before the PV product, ``o -> bf16`` before the gate). The two inner
    contractions (``sum over d`` of ``q*k`` and ``sum over the 64 keys`` of ``p*v``) are
    torch reductions over a fixed-size trailing axis: deterministic for a given shape and
    device, accumulated in ``dtype``. No matmul / SDPA.
    """
    _check(q.dim() == 3 and k.dim() == 3 and v.dim() == 3, "golden expects 3-D q/k/v")
    tq, hq, d = q.shape
    tk, hkv, _ = k.shape
    _check(hq % hkv == 0, "q heads must be a multiple of kv heads")
    group = hq // hkv
    dev = q.device
    scale = float(d) ** -0.5
    neg_inf = float("-inf")

    qf = q.to(dtype)
    kf = k.to(dtype)
    vf = v.to(dtype)
    gf = gate_logits.to(dtype).view(tq, hq, d)
    pos_all = q_pos.to(torch.int64)
    kv_of_h = torch.arange(hq, device=dev) // group

    y = torch.empty((tq, hq, d), dtype=dtype, device=dev)
    o = torch.empty_like(y)
    lse = torch.empty((tq, hq), dtype=dtype, device=dev)

    for q0 in range(0, tq, q_chunk):
        q1 = min(q0 + q_chunk, tq)
        qs = qf[q0:q1]  # (m, hq, d)
        pos = pos_all[q0:q1]
        p_min = int(pos.min().item())
        p_max = int(pos.max().item())
        first_blk = max(p_min - int(window) + 1, 0) // BLOCK_N if window is not None else 0
        last_blk = min(p_max, tk - 1) // BLOCK_N

        m_i = torch.full((q1 - q0, hq), neg_inf, dtype=dtype, device=dev)
        l_i = torch.zeros((q1 - q0, hq), dtype=dtype, device=dev)
        acc = torch.zeros((q1 - q0, hq, d), dtype=dtype, device=dev)

        for blk in range(first_blk, last_blk + 1):
            k0 = blk * BLOCK_N
            k1 = min(k0 + BLOCK_N, tk)
            kpos = torch.arange(k0, k1, device=dev)
            kb = kf[k0:k1][:, kv_of_h, :].permute(1, 0, 2)  # (hq, n, d)
            vb = vf[k0:k1][:, kv_of_h, :].permute(1, 0, 2)  # (hq, n, d)
            # scores[i, h, j] = sum_d q[i, h, d] * k[j, h // group, d]  (then * scale, fp32)
            scores = (qs[:, :, None, :] * kb[None, :, :, :]).sum(dim=-1) * scale
            keep = kpos[None, None, :] <= pos[:, None, None]
            if window is not None:
                keep = keep & (kpos[None, None, :] > (pos[:, None, None] - int(window)))
            scores = torch.where(keep, scores, torch.full_like(scores, neg_inf))

            m_blk = scores.amax(dim=-1)
            m_new = torch.maximum(m_i, m_blk)
            m_safe = torch.where(m_new == neg_inf, torch.zeros_like(m_new), m_new)
            alpha = torch.exp(m_i - m_safe)
            p = torch.exp(scores - m_safe[..., None])
            l_i = l_i * alpha + p.sum(dim=-1)
            p_b = p.to(torch.bfloat16).to(dtype)  # declared cast before the PV product
            pv = (p_b[..., None] * vb[None, :, :, :]).sum(dim=2)  # (m, hq, d)
            acc = acc * alpha[..., None] + pv
            m_i = m_new

        o_c = acc / l_i[..., None]
        o[q0:q1] = o_c
        lse[q0:q1] = m_i + torch.log(l_i)
        o_b = o_c.to(torch.bfloat16).to(dtype)  # declared cast before the gate
        y[q0:q1] = o_b * torch.sigmoid(gf[q0:q1])

    return y.reshape(tq, hq * d), o.reshape(tq, hq * d), lse


forward_fp32 = gated_attention_forward_fp32


def trace() -> dict[str, Any]:
    """Contract R4 provenance for the strict gated-attention forward."""
    return {
        "op": "gated_attention",
        "rows": [
            "global_gated_attention",
            "local_gated_attention_train",
            "local_gated_attention_rollout",
        ],
        "dtype_in": "bf16",
        "dtype_acc": "fp32",
        "dtype_out": "bf16",
        "dtype_lse": "fp32",
        "softmax_scale": SOFTMAX_SCALE,
        "window": SLIDING_WINDOW,
        "mask": "global: k_pos <= q_pos ; local: q_pos - W < k_pos <= q_pos (absolute positions)",
        "reduction_order": (
            "kv blocks of 64 keys aligned at absolute key position 0, ascending; per block: "
            "m_new=max(m,rowmax(s)); alpha=exp(m-m_new); p=exp(s-m_new); l=l*alpha+rowsum(p); "
            "acc=acc*alpha+dot(bf16(p),V); fully-masked blocks are bitwise no-ops"
        ),
        "casts": [
            "scores fp32 (tl.dot bf16xbf16->fp32, K=128), scaled in fp32",
            "p -> bf16 before the PV dot (K=64)",
            "o = div_rn(acc, l) fp32 -> bf16",
            "y = (o_bf16->fp32) * sigmoid(g->fp32) fp32 -> bf16 (elementwise over 4096)",
        ],
        "gqa": "q head h reads kv head h // 16",
        "tile_config": {
            "BLOCK_N": BLOCK_N,
            "token_tile": {"BLOCK_M": BLOCK_M_TOKEN, "num_warps": NUM_WARPS_TOKEN},
            "head_tile": {"BLOCK_M": BLOCK_M_HEAD, "num_warps": NUM_WARPS_HEAD},
            "num_stages": NUM_STAGES,
            "head_tile_max_q": HEAD_TILE_MAX_Q,
            "page_size": PAGE_SIZE,
        },
        "kernel": "rl_engine.models.muse_glimmer.triton.gated_attention_fwd."
        "_gated_attention_fwd_kernel",
        "autotune": False,
        "tf32": False,
        "exp": "tl.exp fp32",
        "split_kv": "disabled",
    }


__all__ = [
    "HEAD_TILE_MAX_Q",
    "SLIDING_WINDOW",
    "SOFTMAX_SCALE",
    "forward_fp32",
    "gated_attention_forward_fp32",
    "paged_query_positions",
    "strict_gated_attention_fwd_paged",
    "strict_gated_attention_fwd_varlen",
    "trace",
]
