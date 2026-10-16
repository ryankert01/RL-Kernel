"""Decode-path latency: wall ms per call (synchronised) and host enqueue us per call.

Usage: PYTHONPATH=<tree> python bench_overhead.py <tag>
"""

import json
import sys
import time

import torch

from rl_engine.models.muse_glimmer.ops import det_gemm as dg
from rl_engine.models.muse_glimmer.ops import lm_head_softcap_logprob as lm

DEV, BF = "cuda", torch.bfloat16
tag = sys.argv[1]


def rnd(shape, seed, scale=1.0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    return (torch.randn(*shape, device=DEV, generator=g) * scale).to(BF)


def measure(fn, iters):
    with torch.inference_mode():
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        walls = []
        for _ in range(iters):
            t0 = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            walls.append((time.perf_counter() - t0) * 1e3)
        walls.sort()
        # host enqueue: back-to-back calls without sync
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        host = (time.perf_counter() - t0) / iters * 1e6
        torch.cuda.synchronize()
    return {"wall_ms_median": walls[len(walls) // 2], "host_us_per_call": host}


res = {"tag": tag, "module": dg.__file__}
wq = rnd((4096, 6656), 1, 0.02)
wk = rnd((256, 6656), 2, 0.02)
a1 = rnd((1, 6656), 3)
res["gemm_q_M1"] = measure(lambda: dg.fast_det_gemm(a1, wq), 300)
res["gemm_k_M1"] = measure(lambda: dg.fast_det_gemm(a1, wk), 300)
del wq, wk
w = rnd((lm.VOCAB_SIZE, lm.HIDDEN_SIZE), 4, 0.02)
wp = lm.pad_lm_head_weight(w)
del w
torch.cuda.empty_cache()
h1 = rnd((1, lm.HIDDEN_SIZE), 5)
t1 = torch.randint(0, lm.VOCAB_SIZE, (1,), device=DEV)
res["lm_head_T1"] = measure(lambda: lm.lm_head_softcap_logprob(h1, wp, t1), 60)
try:
    res["lm_head_T1_novalidate"] = measure(
        lambda: lm.lm_head_softcap_logprob(h1, wp, t1, validate_targets=False), 60
    )
except TypeError as exc:
    res["lm_head_T1_novalidate"] = f"n/a ({exc})"
print(json.dumps(res))
