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

Head counts (tensor parallelism): every wrapper takes ``num_q_heads`` / ``num_kv_heads``
(defaults 32 / 2 = the full Muse-Glimmer geometry, so existing callers are unchanged). A TP rank
passes its shard: TP2 ``16/1``, TP4 ``8/1`` and TP8 ``4/1`` (KV head replicated). Only the
geometries in :data:`SUPPORTED_HEAD_GEOMETRIES` are accepted (fail closed). Heads never interact
in O5, so a rank's ``(y, o, lse)`` equal the matching slices of the full call bit for bit
(``test_gated_attention_fwd.py::test_tp_head_shard_*``).

Numerics are defined in ``docs/contracts/muse-glimmer-30b-strict-contract.md`` section O5 and
restated in :mod:`rl_engine.models.muse_glimmer.triton.gated_attention_fwd`. Summary of the
reduction order and cast points (same order in kernel and golden; the golden is a tolerance
reference because the compiled exp/division/FMA differ, see ``trace()["compiled_arithmetic"]``):
KV blocks of 64 keys aligned at absolute position 0, ascending; fp32 scores ``q.k * 128**-0.5``;
online softmax ``m_new = max(m, rowmax)``, ``alpha = exp(m - m_new)``, ``p = exp(s - m_new)``,
``l = l*alpha + rowsum(p)``, ``acc = acc*alpha + (p -> bf16) @ V``; ``o = acc / l -> bf16``;
``y = (o_bf16 -> fp32) * sigmoid(g -> fp32) -> bf16``; ``lse = m + log(l)`` fp32.
"""

from __future__ import annotations

from typing import Any

import torch

from rl_engine.models.muse_glimmer.triton import gated_attention_bwd as _bwd
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
    SUPPORTED_HEAD_GEOMETRIES,
    _gated_attention_fwd_kernel,
    check_head_geometry,
)

SLIDING_WINDOW = 2048
SOFTMAX_SCALE = HEAD_DIM**-0.5

#: Sequences whose longest query chunk is at most this long run on HEAD tiles (one program per
#: query position and KV head, 16 query heads per tile) instead of TOKEN tiles. Pure
#: performance choice: both tile modes are bitwise identical per row (test_gated_attention_fwd).
HEAD_TILE_MAX_Q = 16

TILE_MODES = ("token", "head")

#: Message of the device-side assert (``torch._assert_async``) raised at the next
#: synchronisation when the kernel finds metadata that the wrapper could not check on the host
#: without a sync. A device-side assert poisons the CUDA context (fail closed, no recovery).
#: torch limits the message to 254 characters. Meaning: cu_seqlens not starting at 0 / decreasing
#: / not ending at the row count; max_seqlen_q below a real query length; q_pos outside
#: [0, len_k); seq_lens outside [len_q, max_pages*16]; a block-table page outside [0, num_pages)
#: under a key some query needs.
METADATA_ASSERT_MSG = (
    "strict gated attention: invalid metadata (cu_seqlens, max_seqlen_q < a query length, "
    "q_pos outside [0,len_k), seq_lens outside [len_q,max_pages*16] or a page id outside "
    "[0,num_pages) under a visible key)"
)


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


def _validate_common(
    q: torch.Tensor,
    gate_logits: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    window: int | None,
    hq: int,
) -> int:
    _check(q.is_cuda, "q must be a CUDA tensor")
    _check(q.dtype == torch.bfloat16, f"q must be bf16, got {q.dtype}")
    _check(
        q.dim() == 3 and q.shape[1] == hq and q.shape[2] == HEAD_DIM,
        f"q must be (Tq, {hq}, {HEAD_DIM}) for num_q_heads={hq}, got {tuple(q.shape)}",
    )
    _check(q.is_contiguous(), "q must be contiguous")
    _check(gate_logits.dtype == torch.bfloat16, "gate_logits must be bf16")
    _check(
        gate_logits.shape == (q.shape[0], hq * HEAD_DIM),
        f"gate_logits must be (Tq, {hq * HEAD_DIM}), got {tuple(gate_logits.shape)}",
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


# ------------------------------------------------------------------------------ host checks
def _host_int64(*tensors: torch.Tensor) -> list[torch.Tensor]:
    """One device->host copy (one synchronisation) of several 1-D int tensors, as int64."""
    sizes = [int(t.numel()) for t in tensors]
    flat = torch.cat([t.reshape(-1) for t in tensors]).to("cpu", torch.int64)
    return list(torch.split(flat, sizes))


def _check_offsets_host(cu: torch.Tensor, total: int, name: str) -> torch.Tensor:
    """``cu[0] == 0``, non-decreasing, ``cu[-1] == total``; returns the lengths."""
    _check(int(cu[0]) == 0, f"{name}[0] must be 0, got {int(cu[0])}")
    lens = cu[1:] - cu[:-1]
    if bool((lens < 0).any()):
        i = int(torch.nonzero(lens < 0)[0, 0])
        raise ValueError(
            f"{name} must be non-decreasing: {name}[{i}]={int(cu[i])} > "
            f"{name}[{i + 1}]={int(cu[i + 1])}"
        )
    _check(
        int(cu[-1]) == int(total),
        f"{name}[-1]={int(cu[-1])} must equal the number of rows {int(total)}",
    )
    return lens


def validate_varlen_packing(
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    q_pos: torch.Tensor,
    *,
    num_k_tokens: int,
) -> int:
    """Host validation of a varlen packing; returns ``max_seqlen_q``.

    Checks ``cu_seqlens_q`` / ``cu_seqlens_k`` (start at 0, non-decreasing, end at ``len(q_pos)``
    / ``num_k_tokens``) and ``0 <= q_pos < len_k`` of each query's sequence. Cost: one
    device->host copy of ``cu_seqlens_q``, ``cu_seqlens_k`` and ``q_pos`` (``4 * (2B + 2 + Tq)``
    bytes, one synchronisation) plus vectorised CPU int64 ops. Positions need not be
    contiguous here (the forward does not require it; the backward's
    ``prepare_sequence_meta`` does).
    """
    cu_q, cu_k, pos = _host_int64(cu_seqlens_q, cu_seqlens_k, q_pos)
    lens_q = _check_offsets_host(cu_q, q_pos.numel(), "cu_seqlens_q")
    lens_k = _check_offsets_host(cu_k, int(num_k_tokens), "cu_seqlens_k")
    seq_of_tok = torch.repeat_interleave(torch.arange(lens_q.numel(), device="cpu"), lens_q)
    bad = (pos < 0) | (pos >= lens_k[seq_of_tok])
    if bool(bad.any()):
        t = int(torch.nonzero(bad)[0, 0])
        b = int(seq_of_tok[t])
        raise ValueError(
            f"q_pos[{t}]={int(pos[t])} (sequence {b}) must lie in [0, {int(lens_k[b])}): a query "
            "position must be one of its sequence's keys"
        )
    return int(lens_q.max()) if lens_q.numel() else 0


def validate_paged_packing(
    cu_seqlens_q: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    num_q_tokens: int,
    max_pages: int,
) -> int:
    """Host validation of the paged path's packing; returns ``max_seqlen_q``.

    Checks ``cu_seqlens_q`` (start 0, non-decreasing, ends at ``num_q_tokens``) and
    ``len_q <= seq_lens <= max_pages * 16`` per sequence. Cost: one device->host copy of
    ``cu_seqlens_q`` and ``seq_lens`` (``4 * (2B + 1)`` bytes, one synchronisation). The
    block-table entries are checked by the kernel (device-side assert), not here.
    """
    cu_q, lens_k = _host_int64(cu_seqlens_q, seq_lens)
    lens_q = _check_offsets_host(cu_q, int(num_q_tokens), "cu_seqlens_q")
    bad = (lens_k < lens_q) | (lens_k > int(max_pages) * PAGE_SIZE)
    if bool(bad.any()):
        b = int(torch.nonzero(bad)[0, 0])
        raise ValueError(
            f"sequence {b}: seq_lens={int(lens_k[b])} must lie in [len_q={int(lens_q[b])}, "
            f"max_pages*{PAGE_SIZE}={int(max_pages) * PAGE_SIZE}]"
        )
    return int(lens_q.max()) if lens_q.numel() else 0


def _check_max_seqlen_q(max_seqlen_q: int, tq: int) -> int:
    max_seqlen_q = int(max_seqlen_q)
    _check(max_seqlen_q >= 1 or tq == 0, "max_seqlen_q must be >= 1")
    return max_seqlen_q


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
    num_k_rows: int,
    num_seqs: int,
    max_seqlen_q: int,
    window: int | None,
    paged: bool,
    tile_mode: str,
    hq: int,
    hkv: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Allocate outputs, launch the kernel and queue the metadata assert (no host sync).

    ``num_k_rows``: ``Tk`` (varlen) or ``num_pages`` (paged). Cost of the fail-closed status
    word: one 4-byte ``torch.ones`` fill and one ``torch._assert_async`` launch per call.
    """
    tq = q.shape[0]
    y = torch.empty((tq, hq * HEAD_DIM), dtype=torch.bfloat16, device=q.device)
    o = torch.empty_like(y)
    lse = torch.empty((tq, hq), dtype=torch.float32, device=q.device)
    if tq == 0:
        return y, o, lse

    local = window is not None
    if tile_mode == "head":
        # BLOCK_M stays 16 for every shard (group 16/8/4): same tile, same per-row bits
        grid = (max_seqlen_q, num_seqs, hkv)
        block_m, num_warps = BLOCK_M_HEAD, NUM_WARPS_HEAD
    else:
        grid = (triton_cdiv(max_seqlen_q, BLOCK_M_TOKEN), num_seqs, hq)
        block_m, num_warps = BLOCK_M_TOKEN, NUM_WARPS_TOKEN

    status = torch.ones((1,), dtype=torch.int32, device=q.device)
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
        status,
        float(SOFTMAX_SCALE),
        int(window) if local else 0,
        int(max_pages),
        int(tq),
        int(num_k_rows),
        H_Q=hq,
        H_KV=hkv,
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
    torch._assert_async(status, METADATA_ASSERT_MSG)
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
    num_q_heads: int = NUM_Q_HEADS,
    num_kv_heads: int = NUM_KV_HEADS,
    seq_meta: _bwd.SequenceMeta | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Strict gated attention over packed sequences (training / prefill / chunked prefill).

    Args:
        q: ``(Tq, H_Q, 128)`` bf16 queries (after qk-norm, scale and RoPE); ``H_Q = 32`` at TP1.
        k, v: ``(Tk, H_KV, 128)`` bf16 keys/values. Keys of sequence ``i`` are
            ``k[cu_seqlens_k[i]:cu_seqlens_k[i+1]]`` at absolute positions ``0..len_k-1``.
        gate_logits: ``(Tq, H_Q*128)`` bf16 output of the attention gate projection of the
            layer input (input_layernorm output).
        cu_seqlens_q, cu_seqlens_k: ``(B+1,)`` int32 packed offsets (start at 0,
            non-decreasing, end at ``Tq`` / ``Tk``).
        q_pos: ``(Tq,)`` int32 absolute positions of the queries, ``0 <= q_pos < len_k`` of
            the query's sequence (chunked prefill: the queries are the last ``len_q``
            positions of the sequence; ``q_pos`` says which).
        window: ``None`` for a global (full causal) layer, ``2048`` for a local layer.
        max_seqlen_q: optional host-side ``max(len_q)`` (avoids the host validation's sync).
        tile_mode: ``"token"`` / ``"head"`` to force a tile shape; default picks by length.
        num_q_heads, num_kv_heads: per-rank head counts (``SUPPORTED_HEAD_GEOMETRIES``).
        seq_meta: optional :class:`~rl_engine.models.muse_glimmer.triton.gated_attention_bwd.
            SequenceMeta` from ``prepare_sequence_meta`` on these exact ``cu_seqlens``/``q_pos``
            tensors: host-validated packing, no synchronisation here.

    Validation (fail closed): with ``seq_meta`` the packing was validated when the meta was
    built (identity/version check only, no sync); without it and without ``max_seqlen_q`` the
    wrapper validates on the host (:func:`validate_varlen_packing`, one sync, ``ValueError``);
    with only ``max_seqlen_q`` the kernel checks the same conditions and the coverage of the
    grid, and a violation raises a device-side assert (:data:`METADATA_ASSERT_MSG`) at the next
    synchronisation. The in-kernel checks run on every path (one 4-byte fill + one assert
    launch per call).

    Returns:
        ``(y, o, lse)``: gated output bf16 ``(Tq, H_Q*128)``, pre-gate output bf16
        ``(Tq, H_Q*128)``, ``lse = m + log(l)`` fp32 ``(Tq, H_Q)``.
    """
    hq, hkv = check_head_geometry(num_q_heads, num_kv_heads)
    num_seqs = _validate_common(q, gate_logits, cu_seqlens_q, window, hq)
    for name, t in (("k", k), ("v", v)):
        _check(t.dtype == torch.bfloat16, f"{name} must be bf16")
        _check(
            t.dim() == 3 and t.shape[1] == hkv and t.shape[2] == HEAD_DIM,
            f"{name} must be (Tk, {hkv}, {HEAD_DIM}), got {tuple(t.shape)}",
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
    if seq_meta is not None:
        meta = _bwd._resolve_meta(seq_meta, cu_seqlens_q, cu_seqlens_k, q_pos, k.shape[0])
        if max_seqlen_q is None:
            max_seqlen_q = meta.max_seqlen_q
        _check(
            int(max_seqlen_q) >= meta.max_seqlen_q,
            f"max_seqlen_q={int(max_seqlen_q)} is below the packing's {meta.max_seqlen_q}",
        )
    elif max_seqlen_q is None:
        max_seqlen_q = validate_varlen_packing(
            cu_seqlens_q, cu_seqlens_k, q_pos, num_k_tokens=k.shape[0]
        )
    max_seqlen_q = _check_max_seqlen_q(max_seqlen_q, q.shape[0])
    mode = _resolve_tile_mode(max_seqlen_q, tile_mode)
    cu_q = cu_seqlens_q.contiguous()
    return _launch(
        q=q,
        gate_logits=gate_logits,
        k=k,
        v=v,
        q_pos=q_pos.contiguous(),
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_seqlens_k.contiguous(),
        seq_lens=cu_q,  # unused on the varlen path (any int32 pointer)
        block_table=cu_q,  # unused on the varlen path
        max_pages=0,
        num_k_rows=k.shape[0],
        num_seqs=num_seqs,
        max_seqlen_q=max_seqlen_q,
        window=window,
        paged=False,
        tile_mode=mode,
        hq=hq,
        hkv=hkv,
    )


def paged_query_positions(
    cu_seqlens_q: torch.Tensor, seq_lens: torch.Tensor, total_q: int
) -> torch.Tensor:
    """Absolute positions of packed queries that are the last ``len_q`` tokens of each sequence.

    ``q_pos[t] = seq_lens[b] - len_q[b] + (t - cu_seqlens_q[b])`` for ``t`` in sequence ``b``.
    Pure device arithmetic (no host sync). The paged kernel derives the same positions itself;
    this helper is kept for callers that need them on the host side (e.g. RoPE tables, tests).
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
    num_q_heads: int = NUM_Q_HEADS,
    num_kv_heads: int = NUM_KV_HEADS,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Strict gated attention reading K/V from a paged cache (rollout prefill / decode).

    Args:
        q, gate_logits: as in :func:`strict_gated_attention_fwd_varlen`; the queries of
            sequence ``i`` are its last ``len_q_i`` positions (decode: ``len_q_i == 1``). The
            kernel derives their absolute positions ``seq_lens[i] - len_q_i + t``.
        k_cache, v_cache: ``(num_pages, 16, H_KV, 128)`` bf16 (``H_KV = 2`` at TP1).
        block_table: ``(B, max_pages)`` int32 physical page per logical page. Every page that
            holds a key some query of the call can see must be a live page id in
            ``[0, num_pages)``; ``-1`` (never allocated / evicted) is allowed only for pages
            no query can see (positions ``>= seq_lens``, or ``<= p - window`` for the smallest
            query position ``p`` of the sequence). Those entries are never read; a ``-1`` or
            out-of-range id under a visible key fails closed (device-side assert), it is not
            treated as "masked".
        seq_lens: ``(B,)`` int32 total keys per sequence including the new tokens, which the
            caller has already written to the cache; ``len_q <= seq_lens <= max_pages * 16``.
        cu_seqlens_q: ``(B+1,)`` int32 (start at 0, non-decreasing, end at ``Tq``).
        window: ``None`` (global) or ``2048`` (local).
        max_seqlen_q: optional host-side ``max(len_q)`` (vLLM: ``max_query_len``); without it
            the wrapper validates ``cu_seqlens_q``/``seq_lens`` on the host (one sync).
        num_q_heads, num_kv_heads: per-rank head counts (``SUPPORTED_HEAD_GEOMETRIES``).

    Validation cost on the hot path (``max_seqlen_q`` given): Python shape checks, one 4-byte
    fill and one ``torch._assert_async`` launch; the kernel checks ``cu_seqlens_q``, the grid
    coverage, ``seq_lens`` and every visible key's page id. No host synchronisation.

    Returns:
        ``(y, o, lse)`` exactly as the varlen path, bitwise.
    """
    hq, hkv = check_head_geometry(num_q_heads, num_kv_heads)
    num_seqs = _validate_common(q, gate_logits, cu_seqlens_q, window, hq)
    for name, t in (("k_cache", k_cache), ("v_cache", v_cache)):
        _check(t.dtype == torch.bfloat16, f"{name} must be bf16")
        _check(
            t.dim() == 4 and tuple(t.shape[1:]) == (PAGE_SIZE, hkv, HEAD_DIM),
            f"{name} must be (num_pages, {PAGE_SIZE}, {hkv}, {HEAD_DIM}), "
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
    max_pages = int(block_table.shape[1])
    if max_seqlen_q is None:
        max_seqlen_q = validate_paged_packing(
            cu_seqlens_q, seq_lens, num_q_tokens=q.shape[0], max_pages=max_pages
        )
    max_seqlen_q = _check_max_seqlen_q(max_seqlen_q, q.shape[0])
    mode = _resolve_tile_mode(max_seqlen_q, tile_mode)
    cu_q = cu_seqlens_q.contiguous()
    return _launch(
        q=q,
        gate_logits=gate_logits,
        k=k_cache,
        v=v_cache,
        q_pos=cu_q,  # unused on the paged path: the kernel derives the positions
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_q,  # unused on the paged path
        seq_lens=seq_lens.contiguous(),
        block_table=block_table.contiguous(),
        max_pages=max_pages,
        num_k_rows=int(k_cache.shape[0]),
        num_seqs=num_seqs,
        max_seqlen_q=max_seqlen_q,
        window=window,
        paged=True,
        tile_mode=mode,
        hq=hq,
        hkv=hkv,
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

    Reduction order: the kernel's — KV blocks of 64 keys aligned at absolute position 0,
    ascending, with the same online-softmax update order and the same casts (``p -> bf16``
    before the PV product, ``o -> bf16`` before the gate). The two inner contractions (``sum
    over d`` of ``q*k`` and ``sum over the 64 keys`` of ``p*v``) are torch reductions over a
    fixed-size trailing axis: deterministic for a given shape and device, accumulated in
    ``dtype``. No matmul / SDPA.

    Not the kernel's instruction sequence (tolerance reference, not bitwise): the kernel's
    ``tl.exp`` is ``ex2.approx(x * log2 e)`` (here ``torch.exp``), its divisions flush
    subnormals (``div.rn.ftz``), ``l*alpha + rowsum`` is one fused multiply-add, and the PV
    products accumulate into ``acc*alpha`` inside the MMA (here ``acc*alpha + sum(p*v)``).
    See ``trace()["compiled_arithmetic"]``.
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
            "o = div_rn(acc, l) fp32 (div.rn.ftz.f32) -> bf16",
            "y = (o_bf16->fp32) * sigmoid(g->fp32) fp32 -> bf16 (elementwise over 4096)",
        ],
        # What the compiled kernel executes (Triton 3.7.1, sm80, enable_fp_fusion=True default;
        # PTX/TTGIR inspected in wave B, review gated_attention_fwd:R5). Kernel-vs-kernel parity
        # (tiles, TP shards, paged/varlen, chunking) holds because every variant compiles the same
        # sequence; the torch golden differs in these points and is a tolerance reference only.
        "compiled_arithmetic": {
            "exp": "tl.exp(x) = ex2.approx.f32(x * log2(e)) (the exp2 form; golden: torch.exp)",
            "log": "tl.log = libdevice __nv_logf with FTZ (fma.rn.ftz polynomial)",
            "division": (
                "libdevice.div_rn -> div.rn.ftz.f32: IEEE round-to-nearest, subnormal operands "
                "and results flushed to zero (o = acc/l and the gate sigmoid)"
            ),
            "l_update": "l = l*alpha + rowsum(p) contracts to one fma.rn.f32 (single rounding)",
            "acc_update": (
                "acc*alpha is a separate mul.f32; the PV tl.dot accumulates into it inside the "
                "MMA (tt.dot %p_bf16, %v, %acc_scaled), not dot-then-add"
            ),
            "rowsum": "tl.sum over 64 keys: warps split along keys (warpsPerCTA=[1,4])",
        },
        "window_values": (
            "the wrappers accept window None or any positive int (tiny-model tests use small "
            "windows); Muse-Glimmer-30B bindings must pass exactly 2048 (= SLIDING_WINDOW)"
        ),
        "validation": (
            "fail closed: cu_seqlens (start 0, monotone, end = rows), grid coverage "
            "(max_seqlen_q), 0 <= q_pos < len_k, len_q <= seq_lens <= max_pages*16 and every "
            "visible key's page id in [0, num_pages): host ValueError when the wrapper syncs "
            "anyway (max_seqlen_q None) or a SequenceMeta is given, otherwise in-kernel checks "
            "+ torch._assert_async (one 4-byte fill + one assert launch per call, no sync)"
        ),
        "paged_positions": "derived in the kernel: seq_lens[b] - len_q + t",
        "gqa": "q head h reads kv head h // 16 (rank-local: h_local // (H_Q // H_KV))",
        "head_geometries": [list(g) for g in SUPPORTED_HEAD_GEOMETRIES],
        "head_tile_rows": (
            "BLOCK_M=16 rows hold one GQA group: 16 at TP1/TP2, 8 at TP4, 4 at TP8; unused "
            "rows masked (never stored)"
        ),
        "gate_sigmoid": "sigmoid_rn: div_rn(1, 1 + tl.exp(-g)) fp32 (shared with the backward)",
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
        "exp": "tl.exp fp32 (ex2.approx.f32(x * log2 e), see compiled_arithmetic)",
        "split_kv": "disabled",
    }


__all__ = [
    "HEAD_TILE_MAX_Q",
    "METADATA_ASSERT_MSG",
    "SLIDING_WINDOW",
    "SOFTMAX_SCALE",
    "SUPPORTED_HEAD_GEOMETRIES",
    "check_head_geometry",
    "forward_fp32",
    "gated_attention_forward_fp32",
    "paged_query_positions",
    "strict_gated_attention_fwd_paged",
    "strict_gated_attention_fwd_varlen",
    "trace",
    "validate_paged_packing",
    "validate_varlen_packing",
]
