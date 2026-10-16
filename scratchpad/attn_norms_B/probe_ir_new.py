"""Compiled arithmetic of the strict attention forward (review gated_attention_fwd:R5).

Counts f32 PTX mnemonics per compiled variant and prints the PV tl.dot lines of the TTGIR, plus
the norms forward for comparison. Writes a JSON summary to argv[1].
"""
import collections
import json
import re
import sys

import torch

from rl_engine.models.muse_glimmer.ops import gated_attention as ga
from rl_engine.models.muse_glimmer.ops import norms as N
from rl_engine.models.muse_glimmer.triton import gated_attention_fwd as kern
from rl_engine.models.muse_glimmer.triton import norms as kn

DEV = torch.device("cuda")
n = 100
q = torch.randn(n, 32, 128, device=DEV).bfloat16()
k = torch.randn(n, 2, 128, device=DEV).bfloat16()
v = torch.randn(n, 2, 128, device=DEV).bfloat16()
g = torch.randn(n, 4096, device=DEV).bfloat16()
cu = torch.tensor([0, n], dtype=torch.int32, device=DEV)
pos = torch.arange(n, dtype=torch.int32, device=DEV)
for window in (None, 2048):
    for mode in ("token", "head"):
        ga.strict_gated_attention_fwd_varlen(q, k, v, g, cu, cu, pos, window=window, tile_mode=mode)

PAT = r"\b((?:div|rcp|sqrt|ex2|lg2|fma|mul|add|sub|max|min|cvt)\.[a-z0-9.]+)"
out = {"attention": [], "norms": []}
caches = kern._gated_attention_fwd_kernel.device_caches[torch.cuda.current_device()][0]
for key, compiled in caches.items():
    ttgir = compiled.asm["ttgir"]
    ptx = compiled.asm["ptx"]
    dots = [ln.strip()[:200] for ln in ttgir.splitlines() if "tt.dot" in ln]
    ops = collections.Counter(re.findall(PAT, ptx))
    rec = {
        "enable_fp_fusion": getattr(compiled.metadata, "enable_fp_fusion", None),
        "dots": dots,
        "f32_ops": {k_: c for k_, c in sorted(ops.items()) if "f32" in k_},
        "mma_sync": len(re.findall(r"mma\.sync", ptx)),
        "ftz_div": len(re.findall(r"div\.rn\.ftz\.f32", ptx)),
    }
    out["attention"].append(rec)
    print(json.dumps(rec, indent=1))

x = torch.randn(4, 6656, device=DEV).bfloat16()
w = torch.randn(6656, device=DEV).bfloat16()
N.centered_rmsnorm(x, w, eps=1e-5)
nc = kn._norm_fwd_kernel.device_caches[torch.cuda.current_device()][0]
for key, compiled in nc.items():
    ops = collections.Counter(re.findall(PAT, compiled.asm["ptx"]))
    rec = {"enable_fp_fusion": getattr(compiled.metadata, "enable_fp_fusion", None),
           "f32_ops": {k_: c for k_, c in sorted(ops.items()) if "f32" in k_}}
    out["norms"].append(rec)
    print("norms", json.dumps(rec))
json.dump(out, open(sys.argv[1], "w"), indent=1)
