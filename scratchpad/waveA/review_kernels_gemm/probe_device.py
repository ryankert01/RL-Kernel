"""Reviewer probe (det_gemm:6): operands on cuda:1 while cuda:0 is current."""

import sys

import torch

from rl_engine.models.muse_glimmer.ops import det_gemm as dg

print("module", dg.__file__, "devices", torch.cuda.device_count(), flush=True)
torch.manual_seed(0)
a = (torch.randn(64, 512, device="cuda:0")).to(torch.bfloat16)
w = (torch.randn(256, 512, device="cuda:0") * 0.02).to(torch.bfloat16)
ref = dg.fast_det_gemm(a, w).cpu()
a1, w1 = a.to("cuda:1"), w.to("cuda:1")
try:
    with torch.cuda.device(0):
        y1 = dg.fast_det_gemm(a1, w1)
    torch.cuda.synchronize(1)
    print("device", y1.device, "equal", torch.equal(y1.cpu(), ref), flush=True)
except Exception as exc:  # noqa: BLE001
    print("ERROR", type(exc).__name__, str(exc)[:200], flush=True)
    sys.exit(1)
