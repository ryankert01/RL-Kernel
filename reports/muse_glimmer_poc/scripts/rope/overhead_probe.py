# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
# ruff: noqa: E501  (kept as run in wave A; lines are measurement plumbing)
# Where does the ~300 us autograd fwd+bwd overhead come from?  Wall-clock (CPU) timing.
# usage: PYTHONPATH=<repo> python overhead_probe.py <out.json>
import json
import sys
import time

import torch

from rl_engine.models.muse_glimmer.ops import qknorm_rope as op

DEV = torch.device("cuda")
T = 64
q = torch.randn(T, 32, 128, device=DEV).to(torch.bfloat16)
k = torch.randn(T, 2, 128, device=DEV).to(torch.bfloat16)
cos, sin = op.rope_rows(torch.arange(T, device=DEV))
dyq, dyk = torch.ones_like(q), torch.ones_like(k)


def fwd_bwd():
    qg, kg = q.detach().requires_grad_(True), k.detach().requires_grad_(True)
    torch.autograd.grad(op.qknorm_scaled_rope(qg, kg, cos, sin), (qg, kg), (dyq, dyk))


def wall(fn, n=300):
    for _ in range(20):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1e6


res = {"T": T}
res["fwd_bwd_us"] = wall(fwd_bwd)
import rl_engine.ops.autograd.backward_runtime as br  # noqa: E402

orig = br.record_backward
br.record_backward = lambda *a, **kw: None
res["fwd_bwd_without_record_backward_us"] = wall(fwd_bwd)
br.record_backward = orig
res["fwd_only_us"] = wall(lambda: op.qknorm_scaled_rope(q, k, cos, sin))
pos = torch.arange(T, device=DEV)
res["rope_rows_us"] = wall(lambda: op.rope_rows(pos))
res["rope_rows_check_range_false_us"] = wall(lambda: op.rope_rows(pos, check_range=False))
res["qknorm_scaled_rope_at_local_us"] = wall(lambda: op.qknorm_scaled_rope_at(q, k, pos, local=True))
res["fwd_only_explicit_local_us"] = wall(lambda: op.qknorm_scaled_rope(q, k, cos, sin, local=True))
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
    for _ in range(50):
        fwd_bwd()
    torch.cuda.synchronize()
tab = prof.key_averages().table(sort_by="cpu_time_total", row_limit=15)
print(tab)
print(json.dumps(res, indent=1))
json.dump(res, open(sys.argv[1], "w"), indent=1)
