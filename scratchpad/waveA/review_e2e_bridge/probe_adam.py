"""Probe: does torch.optim.Adam.step on CPU params create a CUDA context? Does the guard test's
monkeypatch of torch.cuda._lazy_init catch it? Does CPUSafeAdam avoid it?"""
import os
import subprocess
import sys

import torch

which = sys.argv[1] if len(sys.argv) > 1 else "driver"

if which == "driver":
    print("CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES"), "torch", torch.__version__)
    for case in ("plain", "plain_monkeypatched", "safe", "safe_monkeypatched"):
        out = subprocess.run([sys.executable, __file__, case], capture_output=True, text=True)
        print(case, "->", (out.stdout + out.stderr).strip().splitlines()[-1:])
    sys.exit(0)

from rl_engine.models.muse_glimmer.e2e.mocks import CPUSafeAdam  # noqa: E402

if which.endswith("monkeypatched"):
    def _blocked(*a, **k):
        raise AssertionError("BLOCKED _lazy_init")
    torch.cuda._lazy_init = _blocked

p = torch.nn.Parameter(torch.randn(4))
opt = (CPUSafeAdam if which.startswith("safe") else torch.optim.Adam)([p], lr=1e-3)
p.grad = torch.ones(4)
try:
    opt.step()
    print("step ok; cuda initialized:", torch.cuda.is_initialized())
except AssertionError as exc:
    print("step raised:", exc)
