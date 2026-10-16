"""Review probe 2: extra batch/path compositions the author's tests do not cover.

1. Auto tile selection depends on max_seqlen_q of the whole batch: a 1-token sequence packed
   with a long sequence runs on TOKEN tiles, alone it runs on HEAD tiles. Bits must match.
2. Paged chunked prefill (TOKEN tile, PAGED=1) vs varlen chunked prefill at the same cut, with
   mixed batch (two sequences, different lengths, one of them a 1-token decode) - paged batch.
3. Repeatability: 5 identical calls -> identical bytes.
4. Non-monotonic q_pos inside one TOKEN tile (packed sequence whose queries are given in a
   permuted order) vs sorted order: per-row bits must match (contract: per-row arithmetic is a
   function of the row only).
"""
import torch
from rl_engine.models.muse_glimmer.ops import gated_attention as ga
from rl_engine.models.muse_glimmer.ops.paged_kv import PagedKVCache

DEV = torch.device("cuda")
HQ, HKV, D = 32, 2, 128
W = 2048


def inputs(n, seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    q = (torch.randn(n, HQ, D, device=DEV, generator=g) * 3.87).to(torch.bfloat16)
    k = torch.randn(n, HKV, D, device=DEV, generator=g).to(torch.bfloat16)
    v = torch.randn(n, HKV, D, device=DEV, generator=g).to(torch.bfloat16)
    gate = (torch.randn(n, HQ * D, device=DEV, generator=g) * 2.0).to(torch.bfloat16)
    return q, k, v, gate


def i32(x):
    return torch.tensor(list(x), dtype=torch.int32, device=DEV)


def eq(a, b):
    return all(torch.equal(x, y) for x, y in zip(a, b))


for window in (None, W):
    # 1. auto tile: decode row alone (HEAD) vs packed with a 3000-token prefill (TOKEN)
    qa, ka, va, ga_ = inputs(2500, 1)
    qb, kb, vb, gb = inputs(3000, 2)
    cu1 = i32([0, 1]); cuk1 = i32([0, 2500]); pos1 = i32([2499])
    alone = ga.strict_gated_attention_fwd_varlen(qa[-1:].contiguous(), ka, va, ga_[-1:].contiguous(), cu1, cuk1, pos1, window=window)
    q = torch.cat([qa[-1:], qb]); gate = torch.cat([ga_[-1:], gb]); k = torch.cat([ka, kb]); v = torch.cat([va, vb])
    cu_q = i32([0, 1, 3001]); cu_k = i32([0, 2500, 5500]); pos = torch.cat([pos1, torch.arange(3000, dtype=torch.int32, device=DEV)])
    packed = ga.strict_gated_attention_fwd_varlen(q, k, v, gate, cu_q, cu_k, pos, window=window)
    print(f"W={window} 1. decode row alone(HEAD auto) vs packed with 3000-prefill(TOKEN auto): equal={eq(alone, tuple(t[:1] for t in packed))}")
    # 3. repeatability
    reps = [ga.strict_gated_attention_fwd_varlen(q, k, v, gate, cu_q, cu_k, pos, window=window) for _ in range(5)]
    print(f"W={window} 3. repeat x5 identical: {all(eq(reps[0], r) for r in reps[1:])}")

    # 2. mixed paged batch: seq A chunked prefill of 300 queries at positions 2200..2499, seq B decode @2999
    cache = PagedKVCache(num_pages=400, num_seqs=2, max_pages_per_seq=188, device=DEV, poison_freed=True)
    cache.append(0, ka, va); cache.append(1, kb, vb)
    if window is not None:
        cache.evict_window(0, window, query_pos=2200); cache.evict_window(1, window)
    qp = torch.cat([qa[2200:], qb[-1:]]); gp = torch.cat([ga_[2200:], gb[-1:]])
    cu_qp = i32([0, 300, 301])
    paged = ga.strict_gated_attention_fwd_paged(qp, gp, cache.k_cache, cache.v_cache, cache.block_table, cache.seq_lens, cu_qp, window=window)
    cuq = i32([0, 300]); cuk = i32([0, 2500]); posA = torch.arange(2200, 2500, dtype=torch.int32, device=DEV)
    refA = ga.strict_gated_attention_fwd_varlen(qa[2200:].contiguous(), ka, va, ga_[2200:].contiguous(), cuq, cuk, posA, window=window)
    cuq2 = i32([0, 1]); cuk2 = i32([0, 3000]); posB = i32([2999])
    refB = ga.strict_gated_attention_fwd_varlen(qb[-1:].contiguous(), kb, vb, gb[-1:].contiguous(), cuq2, cuk2, posB, window=window)
    print(f"W={window} 2. mixed paged batch (300-chunk + decode): A equal={eq(tuple(t[:300] for t in paged), refA)} B equal={eq(tuple(t[300:] for t in paged), refB)}")

    # 4. permuted q_pos within a sequence (TOKEN tile with non-monotonic positions)
    n = 200
    q4, k4, v4, g4 = inputs(n, 44)
    cu = i32([0, n]); pos_sorted = torch.arange(n, dtype=torch.int32, device=DEV)
    base = ga.strict_gated_attention_fwd_varlen(q4, k4, v4, g4, cu, cu, pos_sorted, window=window, tile_mode="token")
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(3)).to(DEV)
    permd = ga.strict_gated_attention_fwd_varlen(q4[perm].contiguous(), k4, v4, g4[perm].contiguous(), cu, cu, pos_sorted[perm].contiguous(), window=window, tile_mode="token")
    print(f"W={window} 4. permuted q_pos inside TOKEN tiles vs sorted: equal={eq(tuple(t[perm] for t in base), permd)}")
