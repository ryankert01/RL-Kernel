# Dump instruction facts of the compiled strict attention forward (TOKEN tile, local) and the
# norm forward kernel: division / sqrt / exp lowering, FMA contraction, PV-dot accumulator.
from __future__ import annotations

import collections
import json
import re
import sys

import torch

from rl_engine.models.muse_glimmer.ops import gated_attention as ga
from rl_engine.models.muse_glimmer.ops import norms as N
from rl_engine.models.muse_glimmer.triton import gated_attention_fwd as kern
from rl_engine.models.muse_glimmer.triton import norms as nk

DEV = torch.device("cuda")
out_dir = sys.argv[1]

q = torch.randn(70, 32, 128, device=DEV).to(torch.bfloat16)
k = torch.randn(70, 2, 128, device=DEV).to(torch.bfloat16)
v = torch.randn(70, 2, 128, device=DEV).to(torch.bfloat16)
g = torch.randn(70, 4096, device=DEV).to(torch.bfloat16)
cu = torch.tensor([0, 70], dtype=torch.int32, device=DEV)
pos = torch.arange(70, dtype=torch.int32, device=DEV)
ga.strict_gated_attention_fwd_varlen(q, k, v, g, cu, cu, pos, window=2048, tile_mode="token")
x = torch.randn(4, 6656, device=DEV).to(torch.bfloat16)
w = torch.randn(6656, device=DEV).to(torch.bfloat16)
N.centered_rmsnorm(x, w, eps=1e-5)


def facts(compiled, tag):
    ptx = compiled.asm["ptx"]
    ttgir = compiled.asm["ttgir"]
    with open(f"{out_dir}/{tag}.ptx", "w") as f:
        f.write(ptx)
    with open(f"{out_dir}/{tag}.ttgir", "w") as f:
        f.write(ttgir)
    ops = collections.Counter(
        m.group(1) for m in re.finditer(r"^\s*((?:div|rcp|sqrt|rsqrt|ex2|lg2|fma|mul|add)\.[a-z0-9.]+)", ptx, re.M)
    )
    return {k: v for k, v in sorted(ops.items())}


res = {}
for c in kern._gated_attention_fwd_kernel.device_caches[torch.cuda.current_device()][0].values():
    res["attn_token_local"] = facts(c, "attn_token_local")
for c in nk._norm_fwd_kernel.device_caches[torch.cuda.current_device()][0].values():
    res["norm_fwd_centered"] = facts(c, "norm_fwd_centered")
print(json.dumps(res, indent=1))
