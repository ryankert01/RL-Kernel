
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
