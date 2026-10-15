# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Contract gates for the strict Muse-Glimmer gated-attention FORWARD (O5, RFC #501).

G1 correctness vs the fp32 golden (WS1 attention bf16 tolerance) and fp32 vs fp64 golden;
G2 batch invariance (alone vs packed, any batch slot) -> torch.equal;
G3 sequence-split invariance (one-shot vs chunked vs token-by-token paged decode, eager vs no
   window eviction, both layer kinds, lengths around the window edge) -> torch.equal;
G5 trace(); plus the BLOCK_M=64 (TOKEN) vs BLOCK_M=16 (HEAD) tile parity the contract asks
every agent to verify, the GQA head mapping, and the real Muse-Glimmer shapes.

GPU only: every strict call runs the Triton kernel. TF32 is disabled for the whole module.
"""

from __future__ import annotations

import json
import random

import pytest
import torch

from rl_engine.contracts.numerical import load_contract, resolve_tolerance
from rl_engine.models.muse_glimmer.ops import gated_attention as ga
from rl_engine.models.muse_glimmer.ops.paged_kv import PagedKVCache
from rl_engine.models.muse_glimmer.triton import gated_attention_fwd as kern

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

DEV = torch.device("cuda")
W = ga.SLIDING_WINDOW
HQ, HKV, D = kern.NUM_Q_HEADS, kern.NUM_KV_HEADS, kern.HEAD_DIM
LAYER_KINDS = pytest.mark.parametrize("window", [None, W], ids=["global", "local"])

_CONTRACT = load_contract()
_TOL = resolve_tolerance(
    _CONTRACT, judgment="forward_accuracy", op_class="attention", dtype=torch.bfloat16
)


# ------------------------------------------------------------------------------- helpers
def _inputs(n: int, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Realistic magnitudes: q is qk-normed and scaled by 3.87, k/v unit RMS, gate ~ N(0, 4)."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    q = (torch.randn(n, HQ, D, device=DEV, generator=g) * 3.87).to(torch.bfloat16)
    k = torch.randn(n, HKV, D, device=DEV, generator=g).to(torch.bfloat16)
    v = torch.randn(n, HKV, D, device=DEV, generator=g).to(torch.bfloat16)
    gate = (torch.randn(n, HQ * D, device=DEV, generator=g) * 2.0).to(torch.bfloat16)
    return q, k, v, gate


def _i32(values) -> torch.Tensor:
    return torch.tensor(list(values), dtype=torch.int32, device=DEV)


def _one_shot(q, k, v, gate, window, tile_mode=None):
    n = q.shape[0]
    cu = _i32([0, n])
    pos = torch.arange(n, dtype=torch.int32, device=DEV)
    return ga.strict_gated_attention_fwd_varlen(
        q, k, v, gate, cu, cu, pos, window=window, tile_mode=tile_mode
    )


def _chunked_varlen(q, k, v, gate, window, cuts, tile_mode=None):
    """Chunked prefill through the varlen path: keys 0..end-1 are visible, queries = chunk."""
    n = q.shape[0]
    bounds = [0, *sorted(cuts), n]
    ys, os_, ls = [], [], []
    for a, b in zip(bounds[:-1], bounds[1:], strict=True):
        if a == b:
            continue
        cu_q = _i32([0, b - a])
        cu_k = _i32([0, b])
        pos = torch.arange(a, b, dtype=torch.int32, device=DEV)
        y, o, lse = ga.strict_gated_attention_fwd_varlen(
            q[a:b].contiguous(), k[:b].contiguous(), v[:b].contiguous(), gate[a:b].contiguous(),
            cu_q, cu_k, pos, window=window, tile_mode=tile_mode,
        )
        ys.append(y)
        os_.append(o)
        ls.append(lse)
    return torch.cat(ys), torch.cat(os_), torch.cat(ls)


def _new_cache(n_tokens: int, num_seqs: int = 1, poison: bool = True) -> PagedKVCache:
    pages_per_seq = -(-n_tokens // kern.PAGE_SIZE)
    return PagedKVCache(
        num_pages=pages_per_seq * num_seqs + 8,
        num_seqs=num_seqs,
        max_pages_per_seq=pages_per_seq,
        device=DEV,
        poison_freed=poison,
    )


def _paged_step(cache, q_chunk, gate_chunk, window, tile_mode=None):
    cu_q = _i32([0, q_chunk.shape[0]])
    return ga.strict_gated_attention_fwd_paged(
        q_chunk.contiguous(), gate_chunk.contiguous(), cache.k_cache, cache.v_cache,
        cache.block_table, cache.seq_lens, cu_q, window=window, tile_mode=tile_mode,
    )


def _chunked_paged(q, k, v, gate, window, cuts, evict: bool, tile_mode=None):
    n = q.shape[0]
    cache = _new_cache(n)
    bounds = [0, *sorted(cuts), n]
    ys, os_, ls = [], [], []
    for a, b in zip(bounds[:-1], bounds[1:], strict=True):
        if a == b:
            continue
        cache.append(0, k[a:b], v[a:b])
        if evict and window is not None:
            # Eager: free everything invisible to the smallest query of this chunk.
            cache.evict_window(0, window, query_pos=a)
        y, o, lse = _paged_step(cache, q[a:b], gate[a:b], window, tile_mode)
        ys.append(y)
        os_.append(o)
        ls.append(lse)
    return torch.cat(ys), torch.cat(os_), torch.cat(ls)


def _decode_paged(q, k, v, gate, window, evict: bool, prefill: int = 0):
    """Optional one-shot paged prefill of ``prefill`` tokens, then token-by-token decode."""
    n = q.shape[0]
    cache = _new_cache(n)
    ys, os_, ls = [], [], []
    if prefill:
        cache.append(0, k[:prefill], v[:prefill])
        y, o, lse = _paged_step(cache, q[:prefill], gate[:prefill], window)
        ys.append(y)
        os_.append(o)
        ls.append(lse)
    for t in range(prefill, n):
        cache.append(0, k[t : t + 1], v[t : t + 1])
        if evict and window is not None:
            cache.evict_window(0, window)
        y, o, lse = _paged_step(cache, q[t : t + 1], gate[t : t + 1], window)
        ys.append(y)
        os_.append(o)
        ls.append(lse)
    return torch.cat(ys), torch.cat(os_), torch.cat(ls)


def _assert_equal(got, want, what: str) -> None:
    for name, a, b in zip(("y", "o", "lse"), got, want, strict=True):
        if not torch.equal(a, b):
            diff = (a.float() - b.float()).abs()
            bad = int((diff > 0).sum().item())
            raise AssertionError(
                f"{what}: {name} differs in {bad} elements, max |diff| = {diff.max().item():.3e}"
            )


def _assert_close_golden(got_bf16: torch.Tensor, golden: torch.Tensor, what: str) -> float:
    err = (got_bf16.float() - golden.float()).abs().max().item()
    assert torch.allclose(
        got_bf16.float(), golden.float(), atol=_TOL.atol, rtol=_TOL.rtol
    ), f"{what}: max |err| = {err:.3e} exceeds WS1 attention bf16 atol={_TOL.atol} rtol={_TOL.rtol}"
    return err


# ------------------------------------------------------------------------------- G1
@LAYER_KINDS
@pytest.mark.parametrize("n", [37, 300, 2300], ids=lambda n: f"n{n}")
def test_g1_kernel_vs_fp32_golden(window, n):
    q, k, v, gate = _inputs(n, seed=11 + n)
    pos = torch.arange(n, dtype=torch.int32, device=DEV)
    y, o, lse = _one_shot(q, k, v, gate, window)
    yg, og, lseg = ga.gated_attention_forward_fp32(q, k, v, gate, pos, window)
    assert not torch.isnan(y).any() and not torch.isnan(lse).any()
    ey = _assert_close_golden(y, yg, "y")
    eo = _assert_close_golden(o, og, "o")
    el = (lse - lseg).abs().max().item()
    assert el < 1e-4, f"lse max |err| = {el:.3e}"
    print(f"\nG1 window={window} n={n}: max|dy|={ey:.3e} max|do|={eo:.3e} max|dlse|={el:.3e}")


@LAYER_KINDS
def test_g1_fp32_golden_vs_fp64_golden(window):
    n = 200
    q, k, v, gate = _inputs(n, seed=5)
    pos = torch.arange(n, dtype=torch.int32, device=DEV)
    y32, o32, l32 = ga.gated_attention_forward_fp32(q, k, v, gate, pos, window)
    y64, o64, l64 = ga.gated_attention_forward_fp32(q, k, v, gate, pos, window, dtype=torch.float64)
    ey = (y32.double() - y64).abs().max().item()
    eo = (o32.double() - o64).abs().max().item()
    el = (l32.double() - l64).abs().max().item()
    # fp32 accumulation error plus the rare bf16 round flips of p at rounding boundaries.
    assert ey < 2e-2 and eo < 2e-2 and el < 1e-4
    print(
        f"\nfp32 vs fp64 golden window={window}: max|dy|={ey:.3e} max|do|={eo:.3e} "
        f"max|dlse|={el:.3e}"
    )
    # and the kernel is as close to fp64 as to fp32
    y, o, lse = _one_shot(q, k, v, gate, window)
    e64 = (y.double() - y64).abs().max().item()
    assert e64 <= _TOL.atol + _TOL.rtol * y64.abs().max().item()
    print(f"kernel vs fp64 golden: max|dy|={e64:.3e}")


# ------------------------------------------------------------------------------- tiles
@LAYER_KINDS
@pytest.mark.parametrize("n", [1, 16, 17, 64, 65, 2100], ids=lambda n: f"n{n}")
def test_token_vs_head_tiles_bitwise(window, n):
    """Contract O5: BLOCK_M=64 (prefill) and BLOCK_M=16 (decode) tiles give identical bits."""
    q, k, v, gate = _inputs(n, seed=101 + n)
    tok = _one_shot(q, k, v, gate, window, tile_mode="token")
    head = _one_shot(q, k, v, gate, window, tile_mode="head")
    auto = _one_shot(q, k, v, gate, window)
    _assert_equal(head, tok, "HEAD tile vs TOKEN tile")
    _assert_equal(auto, tok, "auto tile vs TOKEN tile")


def test_tile_layouts_share_warp_split():
    """The compiled MMA layouts must split warps along the key axis identically (see kernel doc).

    Compiles every variant the strict paths launch before scanning: TOKEN and HEAD tiles at
    every TP head geometry (32/2, 16/1, 8/1 and 4/1, the last two with masked HEAD-tile rows),
    varlen and paged, global and local.
    """
    assert kern.NUM_WARPS_HEAD == kern.NUM_WARPS_TOKEN
    n = 70
    for hq, hkv in kern.SUPPORTED_HEAD_GEOMETRIES:
        g = torch.Generator(device=DEV).manual_seed(hq)
        q = torch.randn(n, hq, D, device=DEV, generator=g).to(torch.bfloat16)
        k = torch.randn(n, hkv, D, device=DEV, generator=g).to(torch.bfloat16)
        v = torch.randn(n, hkv, D, device=DEV, generator=g).to(torch.bfloat16)
        gate = torch.randn(n, hq * D, device=DEV, generator=g).to(torch.bfloat16)
        cu = _i32([0, n])
        pos = torch.arange(n, dtype=torch.int32, device=DEV)
        cache = PagedKVCache(num_pages=8, num_seqs=1, max_pages_per_seq=5, device=DEV,
                             num_kv_heads=hkv)
        cache.append(0, k, v)
        heads = {"num_q_heads": hq, "num_kv_heads": hkv}
        for window in (None, W):
            for mode in ("token", "head"):
                ga.strict_gated_attention_fwd_varlen(
                    q, k, v, gate, cu, cu, pos, window=window, tile_mode=mode, **heads
                )
                ga.strict_gated_attention_fwd_paged(
                    q, gate, cache.k_cache, cache.v_cache, cache.block_table, cache.seq_lens,
                    cu, window=window, tile_mode=mode, **heads,
                )
    caches = kern._gated_attention_fwd_kernel.device_caches[torch.cuda.current_device()][0]
    splits = set()
    for compiled in caches.values():
        for line in compiled.asm["ttgir"].splitlines():
            if line.strip().startswith("#mma"):
                splits.add(line.split("warpsPerCTA = ")[1].split("]")[0] + "]")
    assert len(caches) >= 4 * 2 * 2 * 2, f"only {len(caches)} variants compiled"
    assert len(splits) == 1, f"different warp splits across tile variants: {splits}"
    print(f"\nMMA warpsPerCTA for all {len(caches)} compiled variants: {splits}")


# ------------------------------------------------------------------------------- G2
@LAYER_KINDS
@pytest.mark.parametrize("tile_mode", ["token", "head"])
def test_g2_batch_invariance_varlen(window, tile_mode):
    lens = [5, 130, 64, 2100, 1, 333]
    seqs = [_inputs(n, seed=200 + i) for i, n in enumerate(lens)]
    alone = []
    for q, k, v, gate in seqs:
        alone.append(_one_shot(q, k, v, gate, window, tile_mode=tile_mode))

    for order in ([0, 1, 2, 3, 4, 5], [5, 3, 1, 4, 2, 0], [3, 3, 0, 3]):
        q = torch.cat([seqs[i][0] for i in order])
        k = torch.cat([seqs[i][1] for i in order])
        v = torch.cat([seqs[i][2] for i in order])
        gate = torch.cat([seqs[i][3] for i in order])
        cu = _i32([0, *torch.tensor([lens[i] for i in order]).cumsum(0).tolist()])
        pos = torch.cat([torch.arange(lens[i], dtype=torch.int32, device=DEV) for i in order])
        y, o, lse = ga.strict_gated_attention_fwd_varlen(
            q, k, v, gate, cu, cu, pos, window=window, tile_mode=tile_mode
        )
        off = 0
        for slot, i in enumerate(order):
            n = lens[i]
            got = (y[off : off + n], o[off : off + n], lse[off : off + n])
            _assert_equal(got, alone[i], f"seq {i} (len {n}) at batch slot {slot} of {order}")
            off += n


@LAYER_KINDS
def test_g2_batch_invariance_paged_decode(window):
    """Decode batch with mixed context lengths vs each sequence decoded alone."""
    lens = [1, 17, 64, 2049, 700, 2048]
    seqs = [_inputs(n, seed=300 + i) for i, n in enumerate(lens)]
    alone = []
    for (q, k, v, gate), n in zip(seqs, lens, strict=True):
        cache = _new_cache(n)
        cache.append(0, k, v)
        alone.append(_paged_step(cache, q[-1:], gate[-1:], window))

    for order in ([0, 1, 2, 3, 4, 5], [3, 5, 2, 4, 1, 0]):
        cache = _new_cache(max(lens), num_seqs=len(order))
        for slot, i in enumerate(order):
            cache.append(slot, seqs[i][1], seqs[i][2])
        q = torch.cat([seqs[i][0][-1:] for i in order])
        gate = torch.cat([seqs[i][3][-1:] for i in order])
        cu_q = _i32(range(len(order) + 1))
        y, o, lse = ga.strict_gated_attention_fwd_paged(
            q, gate, cache.k_cache, cache.v_cache, cache.block_table, cache.seq_lens, cu_q,
            window=window,
        )
        for slot, i in enumerate(order):
            got = (y[slot : slot + 1], o[slot : slot + 1], lse[slot : slot + 1])
            _assert_equal(got, alone[i], f"decode seq {i} at slot {slot}")


# ------------------------------------------------------------------------------- G3
G3_LENGTHS = [64, 2047, 2048, 2049, 4096, 8192]


def _split_points(n: int, seed: int) -> list[list[int]]:
    rng = random.Random(seed)
    cuts = [c for c in (2047, 2048, 2049) if c < n]
    out: list[list[int]] = [[c] for c in cuts]
    if n > 2:
        out.append(sorted(rng.sample(range(1, n), min(3, n - 1))))
        out.append(sorted(rng.sample(range(1, n), min(7, n - 1))))
    if n >= 2:
        out.append([1, n - 1])
    return out


@LAYER_KINDS
@pytest.mark.parametrize("n", G3_LENGTHS, ids=lambda n: f"len{n}")
def test_g3_chunked_prefill_varlen_and_paged(window, n):
    q, k, v, gate = _inputs(n, seed=400 + n)
    ref = _one_shot(q, k, v, gate, window)
    for cuts in _split_points(n, seed=n):
        got = _chunked_varlen(q, k, v, gate, window, cuts)
        _assert_equal(got, ref, f"varlen chunked prefill cuts={cuts}")
        got_p = _chunked_paged(q, k, v, gate, window, cuts, evict=False)
        _assert_equal(got_p, ref, f"paged chunked prefill (no eviction) cuts={cuts}")
        got_pe = _chunked_paged(q, k, v, gate, window, cuts, evict=True)
        _assert_equal(got_pe, ref, f"paged chunked prefill (eager eviction) cuts={cuts}")


@LAYER_KINDS
@pytest.mark.parametrize("n", G3_LENGTHS, ids=lambda n: f"len{n}")
def test_g3_token_by_token_decode(window, n):
    """One-shot prefill == paged decode of every token (eager window eviction, NaN-poisoned)."""
    q, k, v, gate = _inputs(n, seed=500 + n)
    ref = _one_shot(q, k, v, gate, window)
    got = _decode_paged(q, k, v, gate, window, evict=True)
    _assert_equal(got, ref, "token-by-token paged decode")


@pytest.mark.parametrize("n", [2049, 2200, 4096], ids=lambda n: f"len{n}")
def test_g3_window_eviction_timing(n):
    """Local layer: never evicting vs evicting eagerly at the window edge -> equal bits."""
    q, k, v, gate = _inputs(n, seed=600 + n)
    ref = _one_shot(q, k, v, gate, W)
    prefill = n - 300
    never = _decode_paged(q, k, v, gate, W, evict=False, prefill=prefill)
    eager = _decode_paged(q, k, v, gate, W, evict=True, prefill=prefill)
    _assert_equal(never, ref, "no eviction")
    _assert_equal(eager, ref, "eager eviction")
    # Eviction actually happened and freed pages were poisoned/reused without changing bits.
    cache = _new_cache(n)
    cache.append(0, k, v)
    freed = cache.evict_window(0, W)
    expected_free = max(0, (n - 1 - W + 1)) // kern.PAGE_SIZE
    assert len(freed) == expected_free, (len(freed), expected_free)
    y, o, lse = _paged_step(cache, q[-1:], gate[-1:], W)
    _assert_equal((y, o, lse), tuple(t[-1:] for t in ref), "after eviction")


def test_g3_evicted_pages_reused_by_other_sequence():
    """Pages freed by one sequence are reused by another; neither sequence's bits change."""
    n_a, n_b = 2600, 400  # A frees floor((2600 - 2048) / 16) = 34 pages; B needs 25
    qa, ka, va, ga_ = _inputs(n_a, seed=700)
    qb, kb, vb, gb = _inputs(n_b, seed=701)
    ref_a = _one_shot(qa, ka, va, ga_, W)
    ref_b = _one_shot(qb, kb, vb, gb, W)
    pages_a = -(-n_a // kern.PAGE_SIZE)
    cache = PagedKVCache(
        num_pages=pages_a + 2, num_seqs=2, max_pages_per_seq=pages_a, device=DEV, poison_freed=True
    )
    cache.append(0, ka, va)
    freed = cache.evict_window(0, W)
    assert len(freed) * kern.PAGE_SIZE >= n_b, "test needs enough freed pages to host seq B"
    cache.append(1, kb, vb)  # reuses pages freed by A
    assert set(cache.allocated_pages(1)) <= set(freed) | {pages_a, pages_a + 1}
    q = torch.cat([qa[-1:], qb[-1:]])
    gate = torch.cat([ga_[-1:], gb[-1:]])
    y, o, lse = ga.strict_gated_attention_fwd_paged(
        q, gate, cache.k_cache, cache.v_cache, cache.block_table, cache.seq_lens,
        _i32([0, 1, 2]), window=W,
    )
    _assert_equal((y[:1], o[:1], lse[:1]), tuple(t[-1:] for t in ref_a), "seq A after reuse")
    _assert_equal((y[1:], o[1:], lse[1:]), tuple(t[-1:] for t in ref_b), "seq B on reused pages")


# ------------------------------------------------------------------------------- GQA
@LAYER_KINDS
def test_gqa_head_mapping(window):
    n = 150
    q, k, v, gate = _inputs(n, seed=800)
    pos = torch.arange(n, dtype=torch.int32, device=DEV)
    base = _one_shot(q, k, v, gate, window)
    k2 = k.clone()
    v2 = v.clone()
    k2[:, 1] = torch.randn_like(k2[:, 1].float()).to(torch.bfloat16)
    v2[:, 1] = torch.randn_like(v2[:, 1].float()).to(torch.bfloat16)
    y2, o2, lse2 = _one_shot(q, k2, v2, gate, window)
    half = (HQ // HKV) * D
    assert torch.equal(y2[:, :half], base[0][:, :half]), "kv head 1 leaked into q heads 0..15"
    assert torch.equal(lse2[:, : HQ // HKV], base[2][:, : HQ // HKV])
    assert not torch.equal(o2[:, half:], base[1][:, half:]), "q heads 16..31 ignore kv head 1"
    yg, og, lseg = ga.gated_attention_forward_fp32(q, k2, v2, gate, pos, window)
    _assert_close_golden(y2, yg, "gqa y")
    _assert_close_golden(o2, og, "gqa o")


# ------------------------------------------------------------------------------- TP shards
def _shard(t: torch.Tensor, lo: int, hi: int) -> torch.Tensor:
    return t[:, lo:hi].contiguous()


@LAYER_KINDS
@pytest.mark.parametrize("tp", [2, 4, 8])
def test_tp_head_shard_varlen_bitwise(window, tp):
    """Rank r of TP t (16/1, 8/1, 4/1 heads) == the TP1 call's head slice, bit for bit.

    Covers one-shot packed varlen (TOKEN and HEAD tiles, the HEAD tile holding a group of 8 or 4
    in a 16-row tile at TP4/TP8) and chunked prefill.
    """
    lens = [5, 130, 2100, 1, 333]
    seqs = [_inputs(n, seed=1200 + i) for i, n in enumerate(lens)]
    q = torch.cat([s[0] for s in seqs])
    k = torch.cat([s[1] for s in seqs])
    v = torch.cat([s[2] for s in seqs])
    gate = torch.cat([s[3] for s in seqs])
    cu = _i32([0, *torch.tensor(lens).cumsum(0).tolist()])
    pos = torch.cat([torch.arange(n, dtype=torch.int32, device=DEV) for n in lens])
    hq_r = HQ // tp
    for mode in ("token", "head"):
        full = ga.strict_gated_attention_fwd_varlen(
            q, k, v, gate, cu, cu, pos, window=window, tile_mode=mode
        )
        for r in range(tp):
            kvh = (r * hq_r) // (HQ // HKV)
            got = ga.strict_gated_attention_fwd_varlen(
                _shard(q, r * hq_r, (r + 1) * hq_r), _shard(k, kvh, kvh + 1),
                _shard(v, kvh, kvh + 1), _shard(gate, r * hq_r * D, (r + 1) * hq_r * D),
                cu, cu, pos, window=window, tile_mode=mode, num_q_heads=hq_r, num_kv_heads=1,
            )
            cols = slice(r * hq_r * D, (r + 1) * hq_r * D)
            want = (full[0][:, cols], full[1][:, cols], full[2][:, r * hq_r : (r + 1) * hq_r])
            _assert_equal(got, want, f"TP{tp} rank {r} {mode} tiles")
    # chunked prefill of the long sequence on rank 0's shard == its one-shot slice
    n = 2100
    q1, k1, v1, g1 = _inputs(n, seed=1300)
    ref = _one_shot(q1, k1, v1, g1, window)
    qs, gs = _shard(q1, 0, hq_r), _shard(g1, 0, hq_r * D)
    ks, vs = _shard(k1, 0, 1), _shard(v1, 0, 1)
    parts = []
    for a, b in ((0, 2047), (2047, 2049), (2049, n)):
        parts.append(
            ga.strict_gated_attention_fwd_varlen(
                qs[a:b].contiguous(), ks[:b].contiguous(), vs[:b].contiguous(),
                gs[a:b].contiguous(), _i32([0, b - a]), _i32([0, b]),
                torch.arange(a, b, dtype=torch.int32, device=DEV), window=window,
                num_q_heads=hq_r, num_kv_heads=1,
            )
        )
    got = tuple(torch.cat([p[i] for p in parts]) for i in range(3))
    _assert_equal(got, (ref[0][:, : hq_r * D], ref[1][:, : hq_r * D], ref[2][:, :hq_r]),
                  f"TP{tp} rank 0 chunked prefill")


@LAYER_KINDS
@pytest.mark.parametrize("tp", [2, 4, 8])
def test_tp_head_shard_paged_decode_bitwise(window, tp):
    """Paged decode batch on a replicated-KV shard cache == the TP1 decode's head slice."""
    lens = [1, 17, 2049, 700]
    seqs = [_inputs(n, seed=1400 + i) for i, n in enumerate(lens)]
    hq_r = HQ // tp
    full_cache = _new_cache(max(lens), num_seqs=len(lens))
    for i, (_, k, v, _) in enumerate(seqs):
        full_cache.append(i, k, v)
        if window is not None:
            full_cache.evict_window(i, window)
    q = torch.cat([s[0][-1:] for s in seqs])
    gate = torch.cat([s[3][-1:] for s in seqs])
    cu_q = _i32(range(len(lens) + 1))
    full = ga.strict_gated_attention_fwd_paged(
        q, gate, full_cache.k_cache, full_cache.v_cache, full_cache.block_table,
        full_cache.seq_lens, cu_q, window=window,
    )
    for r in range(tp):
        kvh = (r * hq_r) // (HQ // HKV)
        cache = PagedKVCache(
            num_pages=full_cache.k_cache.shape[0], num_seqs=len(lens),
            max_pages_per_seq=full_cache.max_pages_per_seq, device=DEV, num_kv_heads=1,
        )
        for i, (_, k, v, _) in enumerate(seqs):
            cache.append(i, k[:, kvh : kvh + 1], v[:, kvh : kvh + 1])
        got = ga.strict_gated_attention_fwd_paged(
            _shard(q, r * hq_r, (r + 1) * hq_r), _shard(gate, r * hq_r * D, (r + 1) * hq_r * D),
            cache.k_cache, cache.v_cache, cache.block_table, cache.seq_lens, cu_q,
            window=window, num_q_heads=hq_r, num_kv_heads=1,
        )
        cols = slice(r * hq_r * D, (r + 1) * hq_r * D)
        want = (full[0][:, cols], full[1][:, cols], full[2][:, r * hq_r : (r + 1) * hq_r])
        _assert_equal(got, want, f"TP{tp} rank {r} paged decode")


def test_rejects_unsupported_head_geometry():
    q, k, v, gate = _inputs(8, seed=1)
    cu = _i32([0, 8])
    pos = torch.arange(8, dtype=torch.int32, device=DEV)
    with pytest.raises(ValueError, match="TP shard"):  # 16 Q heads over 2 KV heads: not a shard
        ga.strict_gated_attention_fwd_varlen(
            q[:, :16].contiguous(), k, v, gate[:, : 16 * D].contiguous(), cu, cu, pos,
            window=None, num_q_heads=16, num_kv_heads=2,
        )
    with pytest.raises(ValueError, match=r"q must be \(Tq, 8, 128\)"):  # declared != actual
        ga.strict_gated_attention_fwd_varlen(
            q, k[:, :1].contiguous(), v[:, :1].contiguous(), gate, cu, cu, pos, window=None,
            num_q_heads=8, num_kv_heads=1,
        )


# ------------------------------------------------------------------------------- real shapes
@LAYER_KINDS
def test_real_shape_prefill_4096(window):
    n = 4096
    q, k, v, gate = _inputs(n, seed=900)
    y, o, lse = _one_shot(q, k, v, gate, window)
    assert y.shape == (n, HQ * D) and lse.shape == (n, HQ)
    assert not torch.isnan(y).any()
    # Golden on a spread of rows (incl. the window edge and the last rows): same positions.
    edge_rows = {0, 1, 63, 64, 2046, 2047, 2048, 2049, 2111, 3000, 4094, 4095}
    rows = torch.tensor(sorted(edge_rows | set(range(100, 4096, 397))), device=DEV)
    pos = rows.to(torch.int32)
    yg, og, lseg = ga.gated_attention_forward_fp32(
        q[rows].contiguous(), k, v, gate[rows].contiguous(), pos, window
    )
    ey = _assert_close_golden(y[rows], yg, "prefill-4096 y")
    eo = _assert_close_golden(o[rows], og, "prefill-4096 o")
    el = (lse[rows] - lseg).abs().max().item()
    assert el < 1e-4
    # Chunked prefill at the window edge and decode of the last 8 tokens reproduce the bits.
    got = _chunked_varlen(q, k, v, gate, window, [2048, 4088])
    _assert_equal(got, (y, o, lse), "4096 chunked [2048, 4088]")
    got_d = _decode_paged(q, k, v, gate, window, evict=True, prefill=4088)
    _assert_equal(got_d, (y, o, lse), "4096 paged prefill + 8 decode steps")
    print(
        f"\nreal prefill-4096 window={window}: max|dy|={ey:.3e} max|do|={eo:.3e} "
        f"max|dlse|={el:.3e}"
    )


@LAYER_KINDS
def test_real_shape_decode_b64_ctx4096(window):
    bsz, ctx = 64, 4096
    seqs = [_inputs(ctx, seed=1000 + i) for i in range(bsz)]
    pages = ctx // kern.PAGE_SIZE
    cache = PagedKVCache(
        num_pages=bsz * pages, num_seqs=bsz, max_pages_per_seq=pages, device=DEV, poison_freed=True
    )
    for i, (_, k, v, _) in enumerate(seqs):
        cache.append(i, k, v)
        if window is not None:
            cache.evict_window(i, window)
    q = torch.cat([s[0][-1:] for s in seqs])
    gate = torch.cat([s[3][-1:] for s in seqs])
    cu_q = _i32(range(bsz + 1))
    y, o, lse = ga.strict_gated_attention_fwd_paged(
        q, gate, cache.k_cache, cache.v_cache, cache.block_table, cache.seq_lens, cu_q,
        window=window,
    )
    assert y.shape == (bsz, HQ * D) and not torch.isnan(y).any()
    # Same decode through the varlen path with packed K/V (two different code paths, same bits).
    k_all = torch.cat([s[1] for s in seqs])
    v_all = torch.cat([s[2] for s in seqs])
    cu_k = _i32(range(0, bsz * ctx + 1, ctx))
    pos = torch.full((bsz,), ctx - 1, dtype=torch.int32, device=DEV)
    got_v = ga.strict_gated_attention_fwd_varlen(
        q, k_all, v_all, gate, cu_q, cu_k, pos, window=window
    )
    _assert_equal(got_v, (y, o, lse), "decode B=64: varlen vs paged")
    # TOKEN-tile decode (BLOCK_M=64 with one valid row) gives the same bits as the HEAD tile.
    got_t = ga.strict_gated_attention_fwd_varlen(
        q, k_all, v_all, gate, cu_q, cu_k, pos, window=window, tile_mode="token"
    )
    _assert_equal(got_t, (y, o, lse), "decode B=64: TOKEN tile vs HEAD tile")
    # Golden for three sequences.
    for i in (0, 31, 63):
        qi, ki, vi, gi = seqs[i]
        yg, og, lseg = ga.gated_attention_forward_fp32(qi[-1:], ki, vi, gi[-1:], pos[:1], window)
        _assert_close_golden(y[i : i + 1], yg, f"decode seq {i} y")
        _assert_close_golden(o[i : i + 1], og, f"decode seq {i} o")
        assert (lse[i : i + 1] - lseg).abs().max().item() < 1e-4


# ------------------------------------------------------------------------------- G5
def test_g5_trace():
    t1 = ga.trace()
    t2 = ga.trace()
    assert t1 == t2
    for key in ("op", "dtype_in", "dtype_acc", "dtype_out", "reduction_order", "tile_config"):
        assert key in t1, key
    assert t1["op"] == "gated_attention"
    assert t1["dtype_in"] == "bf16" and t1["dtype_acc"] == "fp32" and t1["dtype_out"] == "bf16"
    assert t1["tile_config"]["BLOCK_N"] == 64
    assert t1["tile_config"]["token_tile"]["BLOCK_M"] == 64
    assert t1["tile_config"]["head_tile"]["BLOCK_M"] == 16
    assert t1["autotune"] is False and t1["tf32"] is False
    assert [32, 2] in t1["head_geometries"] and [4, 1] in t1["head_geometries"]
    json.dumps(t1)  # serializable


def test_rejects_bad_inputs():
    q, k, v, gate = _inputs(8, seed=1)
    cu = _i32([0, 8])
    pos = torch.arange(8, dtype=torch.int32, device=DEV)
    with pytest.raises(ValueError):
        ga.strict_gated_attention_fwd_varlen(q.float(), k, v, gate, cu, cu, pos, window=None)
    with pytest.raises(ValueError):
        bad_gate = gate[:, :10].contiguous()
        ga.strict_gated_attention_fwd_varlen(q, k, v, bad_gate, cu, cu, pos, window=None)
    with pytest.raises(ValueError):
        bad_pos = pos.to(torch.int64)
        ga.strict_gated_attention_fwd_varlen(q, k, v, gate, cu, cu, bad_pos, window=None)
    with pytest.raises(ValueError):
        ga.strict_gated_attention_fwd_varlen(q, k, v, gate, cu, cu, pos, window=None, tile_mode="x")


# ------------------------------------------------------------------------------- fail closed
# Review wave A2 (gated_attention_fwd:R1-R4): bad packing metadata must raise, never read or
# write out of bounds, never return unwritten rows. Two mechanisms: host validation with a
# ValueError when the wrapper syncs anyway (max_seqlen_q=None), and in-kernel checks + a
# device-side assert (torch._assert_async) on the no-sync path (max_seqlen_q given), which
# poisons the CUDA context and is therefore tested in a subprocess.


def _varlen_case(n=64, seed=1):
    q, k, v, gate = _inputs(n, seed=seed)
    return q, k, v, gate, _i32([0, n]), torch.arange(n, dtype=torch.int32, device=DEV)


@pytest.mark.parametrize(
    "cu_q, cu_k, match",
    [
        ([0, 48], [0, 64], r"cu_seqlens_q\[-1\]=48 must equal the number of rows 64"),
        ([0, 80], [0, 64], r"cu_seqlens_q\[-1\]=80"),
        ([0, 64], [0, 128], r"cu_seqlens_k\[-1\]=128"),
        ([4, 64], [0, 64], r"cu_seqlens_q\[0\] must be 0"),
        ([0, 40, 30, 64], [0, 20, 40, 64], "non-decreasing"),
    ],
    ids=["q_end_short", "q_end_long", "k_end_long", "q_start", "q_decreasing"],
)
def test_host_validation_rejects_bad_offsets(cu_q, cu_k, match):
    q, k, v, gate, _cu, pos = _varlen_case()
    with pytest.raises(ValueError, match=match):
        ga.strict_gated_attention_fwd_varlen(
            q, k, v, gate, _i32(cu_q), _i32(cu_k), pos, window=None
        )


@LAYER_KINDS
def test_host_validation_rejects_q_pos_outside_keys(window):
    q, k, v, gate, cu, pos = _varlen_case()
    for bad in (pos + 1000, pos - 5, pos.clone().fill_(64)):
        with pytest.raises(ValueError, match="q_pos"):
            ga.strict_gated_attention_fwd_varlen(q, k, v, gate, cu, cu, bad, window=window)
    # positions need not be contiguous for the forward: any subset of the keys is valid
    rows = torch.tensor([0, 5, 63, 7], device=DEV)
    y, _o, _l = ga.strict_gated_attention_fwd_varlen(
        q[rows].contiguous(), k, v, gate[rows].contiguous(), _i32([0, 4]), cu,
        rows.to(torch.int32), window=window,
    )
    full, _o2, _l2 = ga.strict_gated_attention_fwd_varlen(q, k, v, gate, cu, cu, pos, window=window)
    assert torch.equal(y, full[rows])


def test_host_validation_paged_seq_lens():
    q, k, v, gate = _inputs(64, seed=2)
    cache = _new_cache(64, num_seqs=2)
    cache.append(0, k, v)
    cache.append(1, k, v)
    cu_q = _i32([0, 1, 2])
    qs, gs = q[-2:].contiguous(), gate[-2:].contiguous()
    lens = cache.seq_lens.clone()
    lens[0] = 96  # more keys than the 4 pages of its block-table row hold
    with pytest.raises(ValueError, match="seq_lens=96"):
        ga.strict_gated_attention_fwd_paged(
            qs, gs, cache.k_cache, cache.v_cache, cache.block_table, lens, cu_q, window=None
        )
    lens = cache.seq_lens.clone()
    lens[1] = 0  # fewer keys than queries
    with pytest.raises(ValueError, match="len_q=1"):
        ga.strict_gated_attention_fwd_paged(
            qs, gs, cache.k_cache, cache.v_cache, cache.block_table, lens, cu_q, window=None
        )
    with pytest.raises(ValueError, match=r"cu_seqlens_q\[-1\]"):
        ga.strict_gated_attention_fwd_paged(
            qs, gs, cache.k_cache, cache.v_cache, cache.block_table, cache.seq_lens,
            _i32([0, 1, 3]), window=None,
        )


def test_seq_meta_path_validates_without_sync():
    """A SequenceMeta is a host-validated packing: no sync, and max_seqlen_q may not undercut it."""
    from rl_engine.models.muse_glimmer.triton import gated_attention_bwd as bwd

    q, k, v, gate, cu, pos = _varlen_case(n=300, seed=3)
    ref = ga.strict_gated_attention_fwd_varlen(q, k, v, gate, cu, cu, pos, window=W)
    meta = bwd.prepare_sequence_meta(cu, cu, pos, num_k_tokens=k.shape[0])
    cache = _new_cache(300)
    cache.append(0, k, v)
    cu1 = _i32([0, 1])
    q1, g1 = q[-1:].contiguous(), gate[-1:].contiguous()
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        got = ga.strict_gated_attention_fwd_varlen(
            q, k, v, gate, cu, cu, pos, window=W, seq_meta=meta
        )
        # the hot paths with a host-side max_seqlen_q do not sync either
        got2 = ga.strict_gated_attention_fwd_varlen(
            q, k, v, gate, cu, cu, pos, window=W, max_seqlen_q=300
        )
        dec = ga.strict_gated_attention_fwd_paged(
            q1, g1, cache.k_cache, cache.v_cache, cache.block_table, cache.seq_lens, cu1,
            window=W, max_seqlen_q=1,
        )
    finally:
        torch.cuda.set_sync_debug_mode("default")
    _assert_equal(got, ref, "seq_meta path")
    _assert_equal(got2, ref, "max_seqlen_q path")
    _assert_equal(dec, tuple(t[-1:] for t in ref), "paged decode, max_seqlen_q path")
    with pytest.raises(ValueError, match="below the packing"):
        ga.strict_gated_attention_fwd_varlen(
            q, k, v, gate, cu, cu, pos, window=W, seq_meta=meta, max_seqlen_q=1
        )
    with pytest.raises(ValueError, match="seq_meta"):
        ga.strict_gated_attention_fwd_varlen(
            q, k, v, gate, cu.clone(), cu, pos, window=W, seq_meta=meta
        )


_CHILD = r'''
import sys, torch
from rl_engine.models.muse_glimmer.ops import gated_attention as ga
from rl_engine.models.muse_glimmer.ops.paged_kv import PagedKVCache
DEV = torch.device("cuda")
HQ, HKV, D = 32, 2, 128

def inputs(n, seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    q = (torch.randn(n, HQ, D, device=DEV, generator=g) * 3.87).bfloat16()
    k = torch.randn(n, HKV, D, device=DEV, generator=g).bfloat16()
    v = torch.randn(n, HKV, D, device=DEV, generator=g).bfloat16()
    gate = (torch.randn(n, HQ * D, device=DEV, generator=g) * 2).bfloat16()
    return q, k, v, gate

def i32(x):
    return torch.tensor(list(x), dtype=torch.int32, device=DEV)

def cache_of(k, v, num_pages, num_seqs=1, max_pages=None):
    c = PagedKVCache(num_pages=num_pages, num_seqs=num_seqs,
                     max_pages_per_seq=max_pages or -(-k.shape[0] // 16), device=DEV)
    for s in range(num_seqs):
        c.append(s, k, v)
    return c

case, window = sys.argv[1], (None if sys.argv[2] == "none" else int(sys.argv[2]))
n = 40
q, k, v, gate = inputs(n, 1)
cu = i32([0, n])
pos = torch.arange(n, dtype=torch.int32, device=DEV)
if case == "ok_varlen":
    ga.strict_gated_attention_fwd_varlen(q, k, v, gate, cu, cu, pos, window=window, max_seqlen_q=n)
elif case == "ok_paged":
    c = cache_of(k, v, 8)
    ga.strict_gated_attention_fwd_paged(q[-4:].contiguous(), gate[-4:].contiguous(), c.k_cache,
        c.v_cache, c.block_table, c.seq_lens, i32([0, 4]), window=window, max_seqlen_q=4)
elif case == "max_seqlen_short_auto":    # review p1: auto -> HEAD tile, rows 1.. never computed
    ga.strict_gated_attention_fwd_varlen(q, k, v, gate, cu, cu, pos, window=window, max_seqlen_q=1)
elif case == "max_seqlen_short_token":   # one 64-row TOKEN tile for a 100-query sequence
    q, k, v, gate = inputs(100, 2)
    cu = i32([0, 100])
    pos = torch.arange(100, dtype=torch.int32, device=DEV)
    ga.strict_gated_attention_fwd_varlen(q, k, v, gate, cu, cu, pos, window=window,
                                         max_seqlen_q=50, tile_mode="token")
elif case == "max_seqlen_short_head":
    ga.strict_gated_attention_fwd_varlen(q, k, v, gate, cu, cu, pos, window=window,
                                         max_seqlen_q=2, tile_mode="head")
elif case == "max_seqlen_short_paged":   # review p1b
    c = cache_of(k, v, 8)
    ga.strict_gated_attention_fwd_paged(q[-4:].contiguous(), gate[-4:].contiguous(), c.k_cache,
        c.v_cache, c.block_table, c.seq_lens, i32([0, 4]), window=window, max_seqlen_q=2)
elif case == "q_end_beyond_rows":        # review p8w: OOB write of y/o/lse
    ga.strict_gated_attention_fwd_varlen(q[:16], k, v, gate[:16], cu, cu, pos[:16].contiguous(),
                                         window=window, max_seqlen_q=n)
elif case == "k_end_beyond_rows":        # review p3: reads past v
    ga.strict_gated_attention_fwd_varlen(q, k[:24].contiguous(), v[:24].contiguous(), gate, cu,
                                         i32([0, n]), pos, window=window, max_seqlen_q=n)
elif case == "q_pos_beyond_keys":        # review p4
    ga.strict_gated_attention_fwd_varlen(q, k, v, gate, cu, cu, pos + 1000, window=window,
                                         max_seqlen_q=n)
elif case == "q_pos_negative":
    ga.strict_gated_attention_fwd_varlen(q, k, v, gate, cu, cu, pos - 5, window=window,
                                         max_seqlen_q=n, tile_mode="head")
elif case == "page_minus1_visible":      # review p5: -1 under a visible key is not "masked"
    c = cache_of(k, v, 8)
    bt = c.block_table.clone()
    bt[0, 1] = -1                        # keys 16..31, visible to the query at 39
    ga.strict_gated_attention_fwd_paged(q[-1:].contiguous(), gate[-1:].contiguous(), c.k_cache,
        c.v_cache, bt, c.seq_lens, i32([0, 1]), window=window, max_seqlen_q=1)
elif case == "page_out_of_range":        # review p7: OOB read of the cache
    c = cache_of(k, v, 4)
    bt = c.block_table.clone()
    bt[0, 1] = 1000
    ga.strict_gated_attention_fwd_paged(q[-1:].contiguous(), gate[-1:].contiguous(), c.k_cache,
        c.v_cache, bt, c.seq_lens, i32([0, 1]), window=window, max_seqlen_q=1)
elif case == "seq_lens_beyond_table":    # review p6: would read the next row of the table
    c = cache_of(k, v, 16, num_seqs=2, max_pages=3)
    lens = c.seq_lens.clone()
    lens[0] = 64
    ga.strict_gated_attention_fwd_paged(q[-2:].contiguous(), gate[-2:].contiguous(), c.k_cache,
        c.v_cache, c.block_table, lens, i32([0, 1, 2]), window=window, max_seqlen_q=1)
elif case == "seq_lens_below_len_q":
    c = cache_of(k, v, 8)
    lens = c.seq_lens.clone()
    lens[0] = 2
    ga.strict_gated_attention_fwd_paged(q[-4:].contiguous(), gate[-4:].contiguous(), c.k_cache,
        c.v_cache, c.block_table, lens, i32([0, 4]), window=window, max_seqlen_q=4)
else:
    raise SystemExit(f"unknown case {case}")
torch.cuda.synchronize()
print("CHILD-OK")
'''

_BAD_CASES = [
    "max_seqlen_short_auto", "max_seqlen_short_token", "max_seqlen_short_head",
    "max_seqlen_short_paged",
    "q_end_beyond_rows", "k_end_beyond_rows", "q_pos_beyond_keys", "q_pos_negative",
    "page_minus1_visible", "page_out_of_range", "seq_lens_beyond_table", "seq_lens_below_len_q",
]


def _run_child(case: str, window) -> tuple[int, str]:
    import os
    import subprocess
    import sys
    from pathlib import Path

    repo = str(Path(__file__).resolve().parents[3])
    env = dict(os.environ)
    env["PYTHONPATH"] = repo + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD, case, "none" if window is None else str(window)],
        capture_output=True, text=True, env=env, timeout=600,
    )
    return proc.returncode, proc.stdout + proc.stderr


@pytest.mark.parametrize("case", _BAD_CASES)
def test_device_checks_fail_closed(case):
    """No-sync path: the kernel detects the bad metadata and the queued assert fires."""
    window = W if case in ("page_minus1_visible", "q_pos_negative") else None
    code, out = _run_child(case, window)
    assert code != 0 and "CHILD-OK" not in out, f"{case}: no error raised\n{out[-2000:]}"
    assert "strict gated attention: invalid metadata" in out, out[-2000:]


@pytest.mark.parametrize("case", ["ok_varlen", "ok_paged"])
def test_device_checks_control(case):
    code, out = _run_child(case, W)
    assert code == 0 and "CHILD-OK" in out, out[-2000:]
