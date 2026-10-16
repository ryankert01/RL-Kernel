"""Fail-open probes for strict_gated_attention_fwd_{varlen,paged} (review2)."""
import torch
from rl_engine.models.muse_glimmer.ops import gated_attention as ga
from rl_engine.models.muse_glimmer.ops.paged_kv import PagedKVCache

torch.backends.cuda.matmul.allow_tf32 = False
DEV = torch.device("cuda")
HQ, HKV, D, W = 32, 2, 128, 2048


def inputs(n, seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    q = (torch.randn(n, HQ, D, device=DEV, generator=g) * 3.87).bfloat16()
    k = torch.randn(n, HKV, D, device=DEV, generator=g).bfloat16()
    v = torch.randn(n, HKV, D, device=DEV, generator=g).bfloat16()
    gate = (torch.randn(n, HQ * D, device=DEV, generator=g) * 2).bfloat16()
    return q, k, v, gate


def i32(x):
    return torch.tensor(list(x), dtype=torch.int32, device=DEV)


def run(name, fn):
    try:
        out = fn()
        torch.cuda.synchronize()
        print(f"[{name}] NO ERROR raised -> {out}")
    except Exception as e:  # noqa: BLE001
        print(f"[{name}] raised {type(e).__name__}: {str(e)[:160]}")


# P1: max_seqlen_q under-reported -> rows never computed, torch.empty garbage returned
def p1():
    n = 40
    q, k, v, gate = inputs(n, 1)
    cu = i32([0, n])
    pos = torch.arange(n, dtype=torch.int32, device=DEV)
    ref = ga.strict_gated_attention_fwd_varlen(q, k, v, gate, cu, cu, pos, window=None)
    res = []
    for _ in range(3):
        junk = torch.full((n * HQ * D * 4,), 7.0, device=DEV, dtype=torch.bfloat16)
        del junk
        y, o, lse = ga.strict_gated_attention_fwd_varlen(
            q, k, v, gate, cu, cu, pos, window=None, max_seqlen_q=1
        )
        bad = (~torch.eq(y, ref[0]).all(dim=1)).sum().item()
        res.append((bad, bool(torch.equal(y[:1], ref[0][:1]))))
    return f"rows of y differing from reference (out of {n}) per trial, row0 ok: {res}"


# P1b: same in paged mode with HEAD tiles (vLLM passes attn_metadata.max_query_len)
def p1b():
    n = 30
    q, k, v, gate = inputs(n, 2)
    cache = PagedKVCache(num_pages=8, num_seqs=1, max_pages_per_seq=2, device=DEV)
    cache.append(0, k, v)
    cu = i32([0, 4])
    ref = ga.strict_gated_attention_fwd_paged(
        q[-4:].contiguous(), gate[-4:].contiguous(), cache.k_cache, cache.v_cache,
        cache.block_table, cache.seq_lens, cu, window=None)
    junk = torch.full((4 * HQ * D * 4,), 7.0, device=DEV, dtype=torch.bfloat16)
    del junk
    y, o, lse = ga.strict_gated_attention_fwd_paged(
        q[-4:].contiguous(), gate[-4:].contiguous(), cache.k_cache, cache.v_cache,
        cache.block_table, cache.seq_lens, cu, window=None, max_seqlen_q=2)
    bad = (~torch.eq(lse, ref[2]).all(dim=1)).sum().item()
    return f"paged len_q=4 with max_seqlen_q=2: {bad}/4 lse rows differ from reference"


# P2: cu_seqlens_q[-1] < Tq -> trailing rows silently uninitialized
def p2():
    n = 100
    q, k, v, gate = inputs(n, 3)
    cu = i32([0, 80])
    pos = torch.arange(n, dtype=torch.int32, device=DEV)
    junk = torch.full((n * HQ * D * 4,), 7.0, device=DEV, dtype=torch.bfloat16)
    del junk
    y, o, lse = ga.strict_gated_attention_fwd_varlen(q, k, v, gate, cu, cu, pos, window=None)
    return f"rows 80..99 of lse (never written) sample: {lse[80:83, 0].tolist()}"


# P3: cu_seqlens_k end beyond k -> reads memory outside k (view into larger buffer)
def p3():
    n = 64
    q, k, v, gate = inputs(n, 4)
    bigk = torch.zeros(n + 64, HKV, D, device=DEV, dtype=torch.bfloat16)
    bigv = torch.zeros(n + 64, HKV, D, device=DEV, dtype=torch.bfloat16)
    bigk[:n] = k
    bigv[:n] = v
    kk, vv = bigk[:n], bigv[:n]
    cu_q = i32([0, n])
    cu_k = i32([0, n + 64])
    pos = torch.arange(64, 128, dtype=torch.int32, device=DEV)
    y1, _, _ = ga.strict_gated_attention_fwd_varlen(q, kk, vv, gate, cu_q, cu_k, pos, window=None)
    bigv[n:] = 100.0  # memory OUTSIDE the k/v tensors passed in
    y2, _, _ = ga.strict_gated_attention_fwd_varlen(q, kk, vv, gate, cu_q, cu_k, pos, window=None)
    return (f"k.shape={tuple(kk.shape)} cu_k[-1]={n + 64}: output changed after editing memory "
            f"outside v: {not torch.equal(y1, y2)} "
            f"(max|dy|={(y1.float() - y2.float()).abs().max().item():.3g})")


# P4: q_pos beyond len_k - 1 (varlen) -> silently truncated key set; negative -> NaN
def p4():
    n = 64
    q, k, v, gate = inputs(n, 5)
    cu = i32([0, n])
    pos = torch.arange(n, dtype=torch.int32, device=DEV) + 1000
    y, o, lse = ga.strict_gated_attention_fwd_varlen(q, k, v, gate, cu, cu, pos, window=None)
    ref_last, _, _ = ga.strict_gated_attention_fwd_varlen(
        q, k, v, gate, cu, cu, torch.full((n,), n - 1, dtype=torch.int32, device=DEV),
        window=None)
    posn = torch.arange(n, dtype=torch.int32, device=DEV) - 5
    _, _, l2 = ga.strict_gated_attention_fwd_varlen(q, k, v, gate, cu, cu, posn, window=None)
    return (f"q_pos=1000..1063 with len_k=64: finite={bool(torch.isfinite(y.float()).all())}, "
            f"== attending all 64 keys: {torch.equal(y, ref_last)}; negative q_pos -> NaN rows: "
            f"{int(torch.isnan(l2).any(dim=1).sum())} (lse -inf rows: "
            f"{int(torch.isinf(l2).any(dim=1).sum())})")


# P5: block table -1 for an IN-WINDOW page -> silently dropped keys
def p5():
    n = 200
    q, k, v, gate = inputs(n, 6)
    out = []
    for window in (None, W):
        cache = PagedKVCache(num_pages=20, num_seqs=1, max_pages_per_seq=13, device=DEV)
        cache.append(0, k, v)
        cu = i32([0, 1])
        ref = ga.strict_gated_attention_fwd_paged(
            q[-1:].contiguous(), gate[-1:].contiguous(), cache.k_cache, cache.v_cache,
            cache.block_table, cache.seq_lens, cu, window=window)
        bt = cache.block_table.clone()
        bt[0, 3] = -1  # keys 48..63, well inside the window
        got = ga.strict_gated_attention_fwd_paged(
            q[-1:].contiguous(), gate[-1:].contiguous(), cache.k_cache, cache.v_cache,
            bt, cache.seq_lens, cu, window=window)
        out.append((window, bool(torch.isfinite(got[0].float()).all()),
                    torch.equal(got[2], ref[2]),
                    round((got[0].float() - ref[0].float()).abs().max().item(), 4)))
    return f"(window, finite, lse_equal_to_full, max|dy|): {out}"


# P6: seq_lens > max_pages*16 -> reads the NEXT sequence's block-table row
def p6():
    n = 64
    qa, ka, va, ga_ = inputs(n, 7)
    qb, kb, vb, gb = inputs(n, 8)
    cache = PagedKVCache(num_pages=16, num_seqs=2, max_pages_per_seq=4, device=DEV)
    cache.append(0, ka, va)
    cache.append(1, kb, vb)
    seq_lens = cache.seq_lens.clone()
    seq_lens[0] = 96
    q = torch.cat([qa[-1:], qb[-1:]])
    g = torch.cat([ga_[-1:], gb[-1:]])
    cu = i32([0, 1, 2])
    y, o, lse = ga.strict_gated_attention_fwd_paged(
        q, g, cache.k_cache, cache.v_cache, cache.block_table, seq_lens, cu, window=None)
    return (f"seq 0 seq_lens=96 > 4 pages*16: finite={bool(torch.isfinite(y.float()).all())} "
            f"(keys 64..95 resolved via block_table[1] = seq 1's pages "
            f"{cache.block_table[1].tolist()})")


# P7: page id >= num_pages -> OOB read of k_cache (silent)
def p7():
    n = 32
    q, k, v, gate = inputs(n, 9)
    cache = PagedKVCache(num_pages=4, num_seqs=1, max_pages_per_seq=2, device=DEV)
    cache.append(0, k, v)
    bt = cache.block_table.clone()
    bt[0, 1] = 1000
    y, o, lse = ga.strict_gated_attention_fwd_paged(
        q[-1:].contiguous(), gate[-1:].contiguous(), cache.k_cache, cache.v_cache, bt,
        cache.seq_lens, i32([0, 1]), window=None)
    return f"page 1000 of 4: no error; lse={lse[0, :2].tolist()}"


# P8: cu_seqlens_q end beyond Tq -> OOB write of y/o/lse (run under compute-sanitizer)
def p8():
    n = 16
    q, k, v, gate = inputs(n, 10)
    cu = i32([0, n + 48])
    pos = torch.arange(n + 48, dtype=torch.int32, device=DEV)[:n].contiguous()
    kb = torch.cat([k, k, k, k])
    vb = torch.cat([v, v, v, v])
    cuk = i32([0, 4 * n])
    y, o, lse = ga.strict_gated_attention_fwd_varlen(q, kb, vb, gate, cu, cuk, pos, window=None,
                                                      max_seqlen_q=n + 48)
    return f"cu_seqlens_q[-1]={n + 48} > Tq={n}: no error raised by the wrapper"


if __name__ == "__main__":
    import sys
    which = sys.argv[1:] or ["p1", "p1b", "p2", "p3", "p4", "p5", "p6", "p7"]
    for name in which:
        run(name, globals()[name])


def p8w():
    """q/gate/q_pos are views into larger buffers (reads in bounds); y/o/lse are sized by Tq."""
    n, extra = 16, 48
    q, k, v, gate = inputs(n + extra, 11)
    qv, gv = q[:n], gate[:n]
    posb = torch.arange(n + extra, dtype=torch.int32, device=DEV)
    pv = posb[:n]
    cu = i32([0, n + extra])
    y, o, lse = ga.strict_gated_attention_fwd_varlen(qv, k, v, gv, cu, cu, pv, window=None,
                                                      max_seqlen_q=n + extra)
    return f"Tq={n}, cu_seqlens_q[-1]={n + extra}: wrapper accepted"


if __name__ == "__main__" and __import__("sys").argv[1:] == ["p8w"]:
    pass
