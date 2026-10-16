"""Count CUDA launches / syncs and CPU time of one lm_head T=1 forward (deterministic host work).

Usage: PYTHONPATH=<tree> python probe_launches.py
"""

import time

import torch
from torch.profiler import ProfilerActivity, profile

from rl_engine.models.muse_glimmer.ops import lm_head_softcap_logprob as lm

DEV, BF = "cuda", torch.bfloat16
g = torch.Generator(device=DEV).manual_seed(4)
w = (torch.randn(lm.VOCAB_SIZE, lm.HIDDEN_SIZE, device=DEV, generator=g) * 0.02).to(BF)
wp = lm.pad_lm_head_weight(w)
del w
h = torch.randn(1, lm.HIDDEN_SIZE, device=DEV, generator=g).to(BF)
t = torch.randint(0, lm.VOCAB_SIZE, (1,), device=DEV, generator=g)
with torch.inference_mode():
    for _ in range(5):
        lm.lm_head_softcap_logprob(h, wp, t)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        lm.lm_head_softcap_logprob(h, wp, t)
        torch.cuda.synchronize()
    names = [e.name for e in prof.events()]
    launches = sum(1 for n in names if n in ("cuLaunchKernel", "cudaLaunchKernel", "cuLaunchKernelEx"))
    syncs = sum(1 for n in names if n == "cudaStreamSynchronize")
    cpu = []
    for _ in range(30):
        t0 = time.process_time()
        lm.lm_head_softcap_logprob(h, wp, t)
        cpu.append((time.process_time() - t0) * 1e3)
    torch.cuda.synchronize()
    cpu.sort()
print({"module": lm.__file__[-60:], "launches": launches, "stream_syncs": syncs,
       "cpu_ms_median": round(cpu[len(cpu) // 2], 3), "cpu_ms_min": round(cpu[0], 3)})
