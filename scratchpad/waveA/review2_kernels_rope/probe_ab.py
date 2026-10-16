"""Old (6a61f6e, int32 offsets) vs new (e891ecc, int64) Triton launchers: bitwise fwd/bwd at
several T, both layer kinds, and GPU-bound timing (CUDA graph of 20 launches to remove host
launch overhead)."""
import importlib.util
import json
import sys

import torch

from rl_engine.models.muse_glimmer.ops import qknorm_rope as op
from rl_engine.models.muse_glimmer.triton import qknorm_rope as new

spec = importlib.util.spec_from_file_location("old_qk", sys.argv[1])
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)

DEV = torch.device("cuda")
EPS, SC = op.RMS_NORM_EPS, op.QK_SCALE
res = {"bitwise": [], "timing_us": []}
for T in (1, 3, 33, 256, 1000, 4096, 8192, 32768):
    g = torch.Generator(device=DEV).manual_seed(T)
    q = (torch.randn(T, 32, 128, device=DEV, generator=g) * 2).to(torch.bfloat16)
    k = (torch.randn(T, 2, 128, device=DEV, generator=g) * 2).to(torch.bfloat16)
    dq_ = torch.randn(T, 32, 128, device=DEV, generator=g).to(torch.bfloat16)
    dk_ = torch.randn(T, 2, 128, device=DEV, generator=g).to(torch.bfloat16)
    pos = torch.randint(0, 131072, (T,), device=DEV, generator=g)
    c, s = op.rope_rows(pos)
    for local in (True, False):
        cc, ss = (c, s) if local else (None, None)
        fo = old.qknorm_rope_forward_triton(q, k, cc, ss, eps=EPS, qk_scale=SC)
        fn = new.qknorm_rope_forward_triton(q, k, cc, ss, eps=EPS, qk_scale=SC)
        bo = old.qknorm_rope_backward_triton(q, k, dq_, dk_, cc, ss, eps=EPS, qk_scale=SC)
        bn = new.qknorm_rope_backward_triton(q, k, dq_, dk_, cc, ss, eps=EPS, qk_scale=SC)
        eq = all(torch.equal(a, b) for a, b in zip(fo + bo, fn + bn))
        res["bitwise"].append({"T": T, "local": local, "fwd_bwd_equal": eq})
    if T >= 4096:
        for name, mod in (("old", old), ("new", new)):
            fwd = lambda: mod.qknorm_rope_forward_triton(q, k, c, s, eps=EPS, qk_scale=SC)
            for _ in range(3):
                fwd()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(20):
                    fwd()
            times = []
            for _ in range(7):
                e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                e0.record()
                graph.replay()
                e1.record()
                torch.cuda.synchronize()
                times.append(e0.elapsed_time(e1) * 1000 / 20)
            times.sort()
            res["timing_us"].append({"T": T, "impl": name, "fwd_median_us": round(times[3], 2)})
print(json.dumps(res, indent=1))
