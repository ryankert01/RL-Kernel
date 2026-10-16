"""Verifier probes (wave B attn_norms): extra bad / legal metadata cases for the strict forward.

usage: python extra_child.py <case>
Prints CHILD-OK after a synchronize when no error was raised. Legal cases also print a
bitwise comparison against an unmodified reference call.
"""
import sys

import torch

from rl_engine.models.muse_glimmer.ops import gated_attention as ga
from rl_engine.models.muse_glimmer.ops.paged_kv import PagedKVCache
from rl_engine.models.muse_glimmer.triton import gated_attention_bwd as bwd

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


def eq(a, b):
    return all(torch.equal(x, y) for x, y in zip(a, b))


case = sys.argv[1]
q, k, v, gate = inputs(100, 1)

if case == "mid_cu_k_nonmono":
    cu_q = i32([0, 10, 20, 40])
    cu_k = i32([0, 30, 20, 40])
    pos = torch.cat([torch.arange(10), torch.arange(10), torch.arange(20)]).int().to(DEV)
    ga.strict_gated_attention_fwd_varlen(q[:40], k[:40], v[:40], gate[:40], cu_q, cu_k, pos,
                                         window=None, max_seqlen_q=20)
elif case == "cu_q_mid_beyond":
    cu_q = i32([0, 50, 40])
    cu_k = i32([0, 50, 100])
    pos = torch.arange(40, dtype=torch.int32, device=DEV)
    ga.strict_gated_attention_fwd_varlen(q[:40], k, v, gate[:40], cu_q, cu_k, pos,
                                         window=None, max_seqlen_q=64, tile_mode="token")
elif case == "token_qpos_bad_second_tile":
    pos = torch.arange(100, dtype=torch.int32, device=DEV)
    pos[70] = 100
    cu = i32([0, 100])
    ga.strict_gated_attention_fwd_varlen(q, k, v, gate, cu, cu, pos, window=None,
                                         max_seqlen_q=100, tile_mode="token")
elif case == "head_qpos_bad_other_seq":
    cu = i32([0, 50, 100])
    pos = torch.cat([torch.arange(50), torch.arange(50)]).int().to(DEV)
    pos[60] = 77  # sequence 1 has 50 keys
    ga.strict_gated_attention_fwd_varlen(q, k, v, gate, cu, cu, pos, window=2048,
                                         max_seqlen_q=50, tile_mode="head")
elif case == "paged_neg7_visible_global":
    c = cache_of(k, v, 8)
    bt = c.block_table.clone()
    bt[0, 0] = -7
    ga.strict_gated_attention_fwd_paged(q[-1:].contiguous(), gate[-1:].contiguous(), c.k_cache,
                                        c.v_cache, bt, c.seq_lens, i32([0, 1]), window=None,
                                        max_seqlen_q=1)
elif case == "paged_page_eq_num_pages":
    c = cache_of(k, v, 7)
    bt = c.block_table.clone()
    bt[0, 2] = 7
    ga.strict_gated_attention_fwd_paged(q[-3:].contiguous(), gate[-3:].contiguous(), c.k_cache,
                                        c.v_cache, bt, c.seq_lens, i32([0, 3]), window=None,
                                        max_seqlen_q=3, tile_mode="token")
elif case == "paged_seq_lens_negative":
    c = cache_of(k, v, 8)
    lens = c.seq_lens.clone()
    lens[0] = -5
    ga.strict_gated_attention_fwd_paged(q[-1:].contiguous(), gate[-1:].contiguous(), c.k_cache,
                                        c.v_cache, c.block_table, lens, i32([0, 1]),
                                        window=None, max_seqlen_q=1)
elif case == "paged_chunk_minus1_visible_token":
    # chunked prefill: 20 queries at the tail; a page visible only to the later rows
    c = cache_of(k, v, 8)
    bt = c.block_table.clone()
    bt[0, 6] = -1  # keys 96..99, visible to rows at 96..99 only
    ga.strict_gated_attention_fwd_paged(q[-20:].contiguous(), gate[-20:].contiguous(),
                                        c.k_cache, c.v_cache, bt, c.seq_lens, i32([0, 20]),
                                        window=None, max_seqlen_q=20, tile_mode="token")
elif case == "legal_evicted_and_garbage":
    # window 32, last query at 99 needs keys 68..99 (pages 4..6); pages 0..3 and padding
    # entries are never visible: -1 / out-of-range ids there must be accepted, bits unchanged
    c = cache_of(k, v, 8, max_pages=9)
    args = (q[-4:].contiguous(), gate[-4:].contiguous(), c.k_cache, c.v_cache)
    ref = ga.strict_gated_attention_fwd_paged(*args, c.block_table, c.seq_lens, i32([0, 4]),
                                              window=32, max_seqlen_q=4)
    bt = c.block_table.clone()
    bt[0, 0] = -1
    bt[0, 1] = -1
    bt[0, 2] = 123456
    bt[0, 3] = -99
    bt[0, 7] = 999999
    bt[0, 8] = -1
    got = ga.strict_gated_attention_fwd_paged(*args, bt, c.seq_lens, i32([0, 4]),
                                              window=32, max_seqlen_q=4)
    torch.cuda.synchronize()
    print("LEGAL-BITWISE", eq(ref, got))
elif case == "stale_meta_cu_q":
    # meta built, then cu_seqlens_q rewritten without a version bump (.data write)
    cu = i32([0, 100])
    pos = torch.arange(100, dtype=torch.int32, device=DEV)
    meta = bwd.prepare_sequence_meta(cu, cu.clone(), pos, num_k_tokens=100)
    cu_k = meta.cu_seqlens_k
    cu.data.copy_(i32([0, 140]))
    print("version unchanged:", meta.versions[0] == cu._version, flush=True)
    ga.strict_gated_attention_fwd_varlen(q, k, v, gate, cu, cu_k, pos, window=None,
                                         seq_meta=meta)
elif case == "stale_meta_q_pos":
    cu = i32([0, 100])
    pos = torch.arange(100, dtype=torch.int32, device=DEV)
    meta = bwd.prepare_sequence_meta(cu, cu.clone(), pos, num_k_tokens=100)
    pos.data.add_(500)
    ga.strict_gated_attention_fwd_varlen(q, k, v, gate, cu, meta.cu_seqlens_k, pos,
                                         window=None, seq_meta=meta)
elif case == "autograd_noncontig_qpos":
    # training path: non-contiguous q_pos with grads must raise in the forward (ValueError)
    from rl_engine.models.muse_glimmer.ops.gated_attention_autograd import gated_attention
    cu = i32([0, 4])
    cu_k = i32([0, 100])
    pos = i32([0, 5, 63, 7])
    qq = q[:4].clone().requires_grad_(True)
    try:
        gated_attention(qq, k, v, gate[:4].contiguous(), cu, cu_k, pos, None)
        print("NO-RAISE")
    except ValueError as e:
        print("RAISED", str(e)[:120])
else:
    raise SystemExit(f"unknown case {case}")
torch.cuda.synchronize()
print("CHILD-OK")
