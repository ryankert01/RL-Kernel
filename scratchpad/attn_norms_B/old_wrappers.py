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
    hq: int,
    hkv: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Strict gated attention over packed sequences (training / prefill / chunked prefill).

    Args:
        q: ``(Tq, H_Q, 128)`` bf16 queries (after qk-norm, scale and RoPE); ``H_Q = 32`` at TP1.
        k, v: ``(Tk, H_KV, 128)`` bf16 keys/values. Keys of sequence ``i`` are
            ``k[cu_seqlens_k[i]:cu_seqlens_k[i+1]]`` at absolute positions ``0..len_k-1``.
        gate_logits: ``(Tq, H_Q*128)`` bf16 output of the attention gate projection of the
            layer input (input_layernorm output).
        cu_seqlens_q, cu_seqlens_k: ``(B+1,)`` int32 packed offsets.
        q_pos: ``(Tq,)`` int32 absolute positions of the queries (chunked prefill: the queries
            are the last ``len_q`` positions of the sequence; ``q_pos`` says which).
        window: ``None`` for a global (full causal) layer, ``2048`` for a local layer.
        max_seqlen_q: optional host-side ``max(len_q)`` (avoids one device sync).
        tile_mode: ``"token"`` / ``"head"`` to force a tile shape; default picks by length.
        num_q_heads, num_kv_heads: per-rank head counts (``SUPPORTED_HEAD_GEOMETRIES``).

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
        hq=hq,
        hkv=hkv,
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
    num_q_heads: int = NUM_Q_HEADS,
    num_kv_heads: int = NUM_KV_HEADS,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Strict gated attention reading K/V from a paged cache (rollout prefill / decode).

    Args:
        q, gate_logits: as in :func:`strict_gated_attention_fwd_varlen`; the queries of
            sequence ``i`` are its last ``len_q_i`` positions (decode: ``len_q_i == 1``).
        k_cache, v_cache: ``(num_pages, 16, H_KV, 128)`` bf16 (``H_KV = 2`` at TP1).
        block_table: ``(B, max_pages)`` int32 physical page per logical page; ``-1`` marks a
            page that was never allocated or was evicted. Evicted pages are never read: the
            kernel masks every key at position ``<= q_pos - window`` before it touches the
            table, so eviction timing cannot change bits.
        seq_lens: ``(B,)`` int32 total keys per sequence including the new tokens, which the
            caller has already written to the cache.
        window: ``None`` (global) or ``2048`` (local).
        num_q_heads, num_kv_heads: per-rank head counts (``SUPPORTED_HEAD_GEOMETRIES``).

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
        hq=hq,
        hkv=hkv,
    )


