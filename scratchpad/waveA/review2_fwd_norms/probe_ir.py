"""Inspect compiled IR: dot-accumulator folding and FMA contraction (review2)."""
import re

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
ga.strict_gated_attention_fwd_varlen(q, k, v, g, cu, cu, pos, window=2048, tile_mode="token")
ga.strict_gated_attention_fwd_varlen(q, k, v, g, cu, cu, pos, window=2048, tile_mode="head")

caches = kern._gated_attention_fwd_kernel.device_caches[torch.cuda.current_device()][0]
for key, compiled in caches.items():
    ttgir = compiled.asm["ttgir"]
    ptx = compiled.asm["ptx"]
    dots = [ln.strip() for ln in ttgir.splitlines() if "tt.dot" in ln]
    print("variant options enable_fp_fusion=", getattr(compiled.metadata, "enable_fp_fusion", "?"))
    for d in dots:
        print("  ", d[:220])
    print("   ptx fma.rn.f32 count:", len(re.findall(r"fma\.rn\.f32", ptx)),
          " ex2.approx:", len(re.findall(r"ex2\.approx", ptx)),
          " lg2.approx:", len(re.findall(r"lg2\.approx", ptx)),
          " div.rn.f32:", len(re.findall(r"div\.rn\.f32", ptx)),
          " mma:", len(re.findall(r"mma\.sync", ptx)))

x = torch.randn(4, 6656, device=DEV).bfloat16()
w = torch.randn(6656, device=DEV).bfloat16()
N.centered_rmsnorm(x, w, eps=1e-5)
nc = kn._norm_fwd_kernel.device_caches[torch.cuda.current_device()][0]
for key, compiled in nc.items():
    ptx = compiled.asm["ptx"]
    print("norm fwd: enable_fp_fusion=", getattr(compiled.metadata, "enable_fp_fusion", "?"),
          "fma.rn.f32:", len(re.findall(r"fma\.rn\.f32", ptx)),
          "sqrt.rn:", len(re.findall(r"sqrt\.rn\.f32", ptx)),
          "div.rn:", len(re.findall(r"div\.rn\.f32", ptx)))

import collections  # noqa: E402

PAT = r"\b((?:div|rcp|sqrt|ex2|lg2|fma|mul|add|sub)\.[a-z0-9.]+)"
for key, compiled in list(caches.items())[:1]:
    ops = collections.Counter(re.findall(PAT, compiled.asm["ptx"]))
    print("attention fwd PTX f32 mnemonics:", {k: c for k, c in ops.items() if "f32" in k})
for key, compiled in list(nc.items())[:1]:
    ops = collections.Counter(re.findall(PAT, compiled.asm["ptx"]))
    print("norm fwd PTX f32 mnemonics:", {k: c for k, c in ops.items() if "f32" in k})
