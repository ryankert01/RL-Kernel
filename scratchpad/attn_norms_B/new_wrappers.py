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
METADATA_ASSERT_MSG = (
    "strict gated attention: invalid metadata (cu_seqlens not starting at 0 / decreasing / not "
    "ending at the row count, max_seqlen_q smaller than a real query length, q_pos outside "
    "[0, len_k), seq_lens outside [len_q, max_pages*16], or a block-table page outside "
    "[0, num_pages) for a key a query needs)"
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
    seq_of_tok = torch.repeat_interleave(torch.arange(lens_q.numel()), lens_q)
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
    _check(
        max_seqlen_q <= max(tq, 1),
        f"max_seqlen_q={max_seqlen_q} exceeds the number of query rows {tq}",
    )
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


