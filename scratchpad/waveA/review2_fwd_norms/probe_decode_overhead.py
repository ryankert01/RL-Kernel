"""Where does paged decode time go? wrapper (paged_query_positions etc.) vs kernel (review2)."""
import time

import torch
from rl_engine.models.muse_glimmer.ops import gated_attention as ga
from rl_engine.models.muse_glimmer.ops.paged_kv import PagedKVCache
from rl_engine.models.muse_glimmer.triton import gated_attention_fwd as kern

DEV = torch.device("cuda")
HQ, HKV, D = 32, 2, 128


def bench(fn, iters=50):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    wall = (time.perf_counter() - t0) / iters * 1e3
    return s.elapsed_time(e) / iters, wall


for bsz in (1, 64):
    ctx = 4096
    pages = ctx // 16
    cache = PagedKVCache(num_pages=bsz * pages, num_seqs=bsz, max_pages_per_seq=pages, device=DEV)
    k = torch.randn(ctx, HKV, D, device=DEV).bfloat16()
    for i in range(bsz):
        cache.append(i, k, k)
    q = torch.randn(bsz, HQ, D, device=DEV).bfloat16()
    g = torch.randn(bsz, HQ * D, device=DEV).bfloat16()
    cu = torch.arange(bsz + 1, dtype=torch.int32, device=DEV)
    for window in (None, 2048):
        full = bench(lambda: ga.strict_gated_attention_fwd_paged(
            q, g, cache.k_cache, cache.v_cache, cache.block_table, cache.seq_lens, cu,
            window=window, max_seqlen_q=1))
        qpos = ga.paged_query_positions(cu, cache.seq_lens, bsz)
        pos_only = bench(lambda: ga.paged_query_positions(cu, cache.seq_lens, bsz))

        def kernel_only():
            return ga._launch(q=q, gate_logits=g, k=cache.k_cache, v=cache.v_cache, q_pos=qpos,
                              cu_seqlens_q=cu, cu_seqlens_k=cu, seq_lens=cache.seq_lens,
                              block_table=cache.block_table, max_pages=pages, num_seqs=bsz,
                              max_seqlen_q=1, window=window, paged=True, tile_mode="head",
                              hq=HQ, hkv=HKV)
        kern_t = bench(kernel_only)
        print(f"B={bsz} window={window}: full wrapper gpu/wall ms = {full[0]:.3f}/{full[1]:.3f}; "
              f"paged_query_positions = {pos_only[0]:.3f}/{pos_only[1]:.3f}; "
              f"kernel launch only = {kern_t[0]:.3f}/{kern_t[1]:.3f}")
