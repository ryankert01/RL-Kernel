"""Megatron binding's strict linears (TP1) vs ops.det_gemm autograd: forward and dX bitwise.

Usage: PYTHONPATH=<tree> python probe_megatron_vs_ops.py
"""

import json

import torch

from rl_engine.models.muse_glimmer.ops import det_gemm as dg
from rl_engine.models.muse_glimmer.runtime import megatron_lora as ml
from rl_engine.models.muse_glimmer.runtime.strict_kernels import TPComm

DEV, BF = "cuda", torch.bfloat16
comm = TPComm(None)
res = {"module": dg.__file__}


def rnd(shape, seed, scale=1.0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    return (torch.randn(*shape, device=DEV, generator=g) * scale).to(BF)


for name, (n, k) in (("k_proj", (256, 6656)), ("q_proj", (4096, 6656)), ("o_proj", (6656, 4096))):
    w = rnd((n, k), 1, 0.02)
    for m in (1, 64, 300):
        x = rnd((m, k), 2 + m)
        gy = rnd((m, n), 3 + m)
        x1 = x.clone().requires_grad_(True)
        y1 = dg.fast_det_gemm(x1, w)
        y1.backward(gy)
        x2 = x.clone().requires_grad_(True)
        if name == "o_proj":
            y2 = ml.StrictRowLinear.apply(x2, w, None, None, comm, k)
        else:
            y2 = ml.StrictColumnLinear.apply(x2, None, None, comm, n, w)
        y2.backward(gy)
        res[f"{name}/M{m}"] = {
            "fwd_equal": bool(torch.equal(y1, y2)),
            "dx_mismatch": int((x1.grad != x2.grad).sum()),
        }
        print(name, m, res[f"{name}/M{m}"], flush=True)
print(json.dumps(res))
