import os
import time

import torch

from rl_engine.models.muse_glimmer.ops import qknorm_rope as op
from rl_engine.models.muse_glimmer.triton import qknorm_rope as kk

dev = torch.device("cuda")
g = torch.Generator(device=dev).manual_seed(0)
print("loadavg", os.getloadavg())
for T in (1, 64, 256):
    q = torch.randn(T, 32, 128, device=dev, generator=g).to(torch.bfloat16)
    k = torch.randn(T, 2, 128, device=dev, generator=g).to(torch.bfloat16)
    c, s = op.rope_rows(torch.arange(T, device=dev))
    f = lambda: kk.qknorm_rope_forward_triton(q, k, c, s, eps=1e-5, qk_scale=3.87)
    api = lambda: op.qknorm_scaled_rope(q, k, c, s)
    for fn, name in ((f, "launcher"), (api, "api")):
        for _ in range(50):
            fn()
        torch.cuda.synchronize()
        res = []
        for _ in range(7):
            t0 = time.perf_counter()
            for _ in range(500):
                fn()
            torch.cuda.synchronize()
            res.append((time.perf_counter() - t0) / 500 * 1e6)
        res.sort()
        print(T, name, "median us/launch back-to-back:", round(res[3], 1), "min", round(res[0], 1))
print("loadavg", os.getloadavg())
