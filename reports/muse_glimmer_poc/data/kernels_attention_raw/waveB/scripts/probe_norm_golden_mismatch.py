"""Strict norms vs golden.to(bf16) mismatch fractions at T=4096, H=6656 (wave B golden)."""
import json
import sys

import torch

from rl_engine.models.muse_glimmer.ops import norms as N

DEV = torch.device("cuda")
H = N.HIDDEN


def bf16(shape, seed, scale=1.0, shift=0.0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    return (torch.randn(*shape, device=DEV, generator=g) * scale + shift).to(torch.bfloat16)


def frac(a, b):
    return float((a != b).float().mean().item())


x = bf16((4096, H), 4106, 2.0)
w = bf16((H,), 11, 0.1)
wf = bf16((H,), 12, 0.1, 1.0)
out = {}
for eps in (N.RMS_NORM_EPS, N.POST_NORM_EPS):
    out[f"centered_eps{eps}"] = frac(N.centered_rmsnorm(x, w, eps=eps),
                                     N.centered_rmsnorm_fp32(x, w, eps=eps).to(torch.bfloat16))
out["weighted"] = frac(N.weighted_rmsnorm(x, wf, eps=N.RMS_NORM_EPS),
                       N.weighted_rmsnorm_fp32(x, wf, eps=N.RMS_NORM_EPS).to(torch.bfloat16))
table = bf16((50000, H), 7, 0.02)
gen = torch.Generator(device=DEV).manual_seed(8)
ids = torch.randint(0, 50000, (4096,), device=DEV, generator=gen)
out["embedding"] = frac(N.normalized_embedding(table, ids),
                        N.normalized_embedding_fp32(table, ids).to(torch.bfloat16))
b = bf16((4096, H), 21, 1.0)
r = bf16((4096, H), 22, 3.0)
rn, pre = N.fused_post_norm_residual_pre_norm(b, r, w, N.POST_NORM_EPS, w, N.RMS_NORM_EPS)
rn_g, pre_g = N.fused_post_norm_residual_pre_norm_fp32(b, r, w, N.POST_NORM_EPS, w,
                                                       N.RMS_NORM_EPS)
out["fused_pre"] = frac(pre, pre_g.to(torch.bfloat16))
out["fused_resid_new"] = frac(rn, rn_g.to(torch.bfloat16))
print(json.dumps(out, indent=1))
json.dump(out, open(sys.argv[1], "w"), indent=1)
