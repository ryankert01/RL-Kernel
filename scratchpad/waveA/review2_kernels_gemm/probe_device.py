"""det_gemm:6 on a given tree: operands on cuda:1 while cuda:0 is current.

Usage: PYTHONPATH=<tree> python probe_device.py   (needs 2 visible GPUs)
"""

import torch

from rl_engine.models.muse_glimmer.ops import det_gemm as dg

print("module", dg.__file__)
g = torch.Generator(device="cuda:0").manual_seed(25)
a = torch.randn(64, 512, device="cuda:0", generator=g).to(torch.bfloat16)
w = (torch.randn(256, 512, device="cuda:0", generator=g) * 0.02).to(torch.bfloat16)
ref = dg.fast_det_gemm(a, w)
a1, w1 = a.to("cuda:1"), w.to("cuda:1")
torch.cuda.synchronize(1)
with torch.cuda.device(0):
    try:
        y1 = dg.fast_det_gemm(a1, w1)
        torch.cuda.synchronize(1)
        torch.cuda.synchronize(0)
        print("same device:", y1.device, "equal:", torch.equal(y1.cpu(), ref.cpu()))
    except Exception as exc:  # noqa: BLE001
        print("ERROR", type(exc).__name__, str(exc)[:300])
# A launch that is issued on device 0's stream while the producer of a1 runs on device 1's
# stream: write a1 on device 1 stream, then call without syncing.
s1 = torch.cuda.Stream(device="cuda:1")
with torch.cuda.device(1), torch.cuda.stream(s1):
    big = torch.randn(4096, 4096, device="cuda:1")
    for _ in range(20):
        big = big @ big * 1e-3
    a2 = (a1.float() + big[:64, :512] * 0).to(torch.bfloat16)
with torch.cuda.device(0):
    try:
        torch.cuda.current_stream(1).wait_stream(s1)
        y2 = dg.fast_det_gemm(a2, w1)
        torch.cuda.synchronize(1)
        torch.cuda.synchronize(0)
        print("stream-ordered equal:", torch.equal(y2.cpu(), ref.cpu()))
    except Exception as exc:  # noqa: BLE001
        print("ERROR2", type(exc).__name__, str(exc)[:300])
