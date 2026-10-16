"""Latency-bound calls (decode): median of 200 timed calls, CUDA events. Usage: <out_json>"""

import json
import statistics
import sys

import torch

from rl_engine.models.muse_glimmer.ops import det_gemm as dg
from rl_engine.models.muse_glimmer.ops import lm_head_softcap_logprob as lm
from rl_engine.models.muse_glimmer.ops import swiglu_ffn as sf

DEV, BF = "cuda", torch.bfloat16
g = torch.Generator(device=DEV).manual_seed(0)


def rnd(*s, scale=1.0):
    return (torch.randn(*s, device=DEV, generator=g) * scale).to(BF)


def t_ms(fn, iters=200, warmup=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(iters):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        out.append(s.elapsed_time(e))
    return statistics.median(out)


res = {}
H, F = 6656, 19968
with torch.no_grad():
    w = rnd(4096, H, scale=0.02)
    wk = rnd(256, H, scale=0.02)
    for m in (1, 16):
        a = rnd(m, H)
        res[f"gemm_q_proj_M{m}"] = t_ms(lambda: dg.fast_det_gemm(a, w))
        res[f"gemm_k_proj_M{m}"] = t_ms(lambda: dg.fast_det_gemm(a, wk))
    wg, wu, wd = rnd(F, H, scale=0.01), rnd(F, H, scale=0.01), rnd(H, F, scale=0.01)
    for t in (1, 16):
        x = rnd(t, H)
        res[f"ffn_fwd_T{t}"] = t_ms(lambda: sf.swiglu_ffn(x, wg, wu, wd), iters=100)
    del wg, wu, wd
    wl = lm.pad_lm_head_weight(rnd(lm.VOCAB_SIZE, H, scale=0.02))
    for t in (1, 16):
        h = rnd(t, H)
        tg = torch.randint(0, lm.VOCAB_SIZE, (t,), device=DEV, generator=g)
        res[f"lm_head_fwd_T{t}"] = t_ms(lambda: lm.lm_head_softcap_logprob(h, wl, tg), iters=100)
        try:
            res[f"lm_head_fwd_T{t}_novalidate"] = t_ms(
                lambda: lm.lm_head_softcap_logprob(h, wl, tg, validate_targets=False), iters=100)
        except TypeError:
            res[f"lm_head_fwd_T{t}_novalidate"] = None
print(json.dumps(res, indent=1))
with open(sys.argv[1], "w") as f:
    json.dump(res, f, indent=1)
