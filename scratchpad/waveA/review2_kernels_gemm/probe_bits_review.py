"""Independent bit ledger (reviewer): hash GEMM outputs on one code tree.

Usage: PYTHONPATH=<tree> python probe_bits_review.py <out_json>
Run once with the base tree (307fcf2) and once with the reviewed tree (b5b77b0).
"""

import hashlib
import json
import sys

import torch

from rl_engine.models.muse_glimmer.ops import det_gemm as dg

DEV, BF = "cuda", torch.bfloat16
out = {"module": dg.__file__}


def h(t):
    return hashlib.sha256(t.detach().contiguous().cpu().view(torch.uint8).numpy()).hexdigest()[:16]


def rnd(shape, seed, scale=1.0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    return (torch.randn(*shape, device=DEV, generator=g) * scale).to(BF)


SHAPES = {
    "q_proj": (4096, 6656),
    "k_proj": (256, 6656),
    "o_proj": (6656, 4096),
    "mlp_gate": (19968, 6656),
    "down_proj": (6656, 19968),
    "lm_head_tile": (25344, 6656),
}
for i, (name, (n, k)) in enumerate(SHAPES.items()):
    w = rnd((n, k), 500 + i, 0.02)
    for m in (1, 7, 16, 33, 64, 65, 256, 300):
        a = rnd((m, k), 900 + m)
        out[f"fwd_bf16/{name}/M{m}"] = h(dg.fast_det_gemm(a, w))
        out[f"fwd_fp32/{name}/M{m}"] = h(dg.fast_det_gemm(a, w, out_dtype=torch.float32))
    del w
    torch.cuda.empty_cache()

# backward paths
for name, (n, k), ms in (
    ("k_proj", (256, 6656), (1, 64, 300)),
    ("q_proj", (4096, 6656), (1, 64)),
):
    w = rnd((n, k), 700, 0.02)
    for m in ms:
        a = rnd((m, k), 710 + m).requires_grad_(True)
        gy = rnd((m, n), 720 + m)
        dg.fast_det_gemm(a, w).backward(gy)
        out[f"dA/{name}/M{m}"] = h(a.grad)

for t in (16, 200, 300, 512, 600, 1024, 4001):
    w = rnd((4096, 6656), 800, 0.02).requires_grad_(True)
    a = rnd((t, 6656), 810 + t)
    gy = rnd((t, 4096), 820 + t)
    dg.fast_det_gemm(a, w).backward(gy)
    out[f"dW/q_proj/T{t}"] = h(w.grad)

json.dump(out, open(sys.argv[1], "w"), indent=1)
print("done", len(out))
