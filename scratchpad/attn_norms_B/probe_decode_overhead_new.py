"""Paged decode cost after wave B: full wrapper vs the kernel launch alone (review fwd:R6).

The wrapper now derives positions in the kernel (no paged_query_positions) and adds a 4-byte
status fill + torch._assert_async. Writes JSON to argv[1].
"""
import json
import sys
import time

import torch

from rl_engine.models.muse_glimmer.ops import gated_attention as ga
from rl_engine.models.muse_glimmer.ops.paged_kv import PagedKVCache
from rl_engine.models.muse_glimmer.triton import gated_attention_fwd as kern

DEV = torch.device("cuda")
HQ, HKV, D = 32, 2, 128


def bench(fn, iters=200):
    for _ in range(20):
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
    return {"gpu_ms": s.elapsed_time(e) / iters, "wall_ms": wall}


out = []
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
        y = torch.empty((bsz, HQ * D), dtype=torch.bfloat16, device=DEV)
        o = torch.empty_like(y)
        lse = torch.empty((bsz, HQ), dtype=torch.float32, device=DEV)
        status = torch.ones((1,), dtype=torch.int32, device=DEV)

        def kernel_only():
            kern._gated_attention_fwd_kernel[(1, bsz, HKV)](
                q, g, cache.k_cache, cache.v_cache, y, o, lse, cu, cu, cu, cache.seq_lens,
                cache.block_table, status, float(ga.SOFTMAX_SCALE),
                0 if window is None else window, pages, bsz, bsz * pages,
                H_Q=HQ, H_KV=HKV, D=D, PAGE=16, BLOCK_M=16, BLOCK_N=64,
                LOCAL=window is not None, PAGED=True, HEAD_TILE=True, num_warps=4,
                num_stages=2)

        kern_t = bench(kernel_only)
        rec = {"B": bsz, "ctx": ctx, "window": window, "full_wrapper": full,
               "kernel_launch_only": kern_t}
        out.append(rec)
        print(json.dumps(rec))
json.dump({"records": out, "torch": torch.__version__}, open(sys.argv[1], "w"), indent=1)
