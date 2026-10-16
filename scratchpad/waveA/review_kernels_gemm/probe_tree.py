"""Reviewer probe (det_gemm:1): kernel vs declared tree, and vs the 'folded' tree, base vs head.

Usage: PYTHONPATH=<tree> python probe_tree.py <label> <out.json>
"""

import hashlib
import json
import re
import sys

import torch
import triton

from rl_engine.models.muse_glimmer.ops import det_gemm as dg
from rl_engine.models.muse_glimmer.triton import det_gemm_tree as kt

BF16, FP32 = torch.bfloat16, torch.float32
label, out_path = sys.argv[1], sys.argv[2]
res = {"label": label, "module": dg.__file__, "triton": triton.__version__, "rows": []}


def rand(*shape, scale=1.0):
    return (torch.randn(*shape, device="cuda") * scale).to(BF16)


def leaf(a, b_kn, lo, hi):
    # single-chunk launch: no in-kernel tree adds that could be re-associated
    return kt.tree_gemm(a[:, lo:hi], b_kn[lo:hi], out_dtype=FP32, num_chunks=1)


def tree8(a, b_kn):
    c = a.size(1) // 8
    p = [leaf(a, b_kn, i * c, (i + 1) * c) for i in range(8)]
    return ((p[0] + p[1]) + (p[2] + p[3])) + ((p[4] + p[5]) + (p[6] + p[7]))


def folded(a, b_kn):
    # what the folded ITERS=1 kernel computes: chunk 2i+1 MMA-chained onto chunk 2i
    c = a.size(1) // 8
    q = [leaf(a, b_kn, 2 * i * c, (2 * i + 2) * c) for i in range(4)]
    return (q[0] + q[1]) + (q[2] + q[3])


def h16(t):
    return hashlib.sha256(t.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()[:16]


torch.manual_seed(0)
for k in (256, 512, 1024):
    for m in (1, 64, 256):
        a = rand(m, k)
        w = rand(384, k, scale=0.02)
        full = dg.fast_det_gemm(a, w, out_dtype=FP32)
        plan = kt.plan_chunks(k, block_k=kt.select_config(m)[1].block_k)
        t8 = tree8(a, w.t())
        fo = folded(a, w.t())
        res["rows"].append(
            dict(
                kind="fwd", K=k, M=m, N=384, numel=full.numel(),
                chunk_len=plan.chunk_len, block_k=plan.block_k,
                mismatch_vs_tree=int((full != t8).sum()),
                mismatch_vs_folded=int((full != fo).sum()),
                tree_vs_folded=int((t8 != fo).sum()),
            )
        )

# dA of k_proj (reduction over N=256) at M=256
torch.manual_seed(1)
w = rand(256, 6656, scale=0.02)
a = rand(256, 6656).requires_grad_(True)
gy = rand(256, 256)
dg.fast_det_gemm(a, w).backward(gy)
t8 = tree8(gy, w).to(BF16)
fo = folded(gy, w).to(BF16)
res["rows"].append(dict(kind="dA_kproj", M=256, numel=a.grad.numel(),
                        mismatch_vs_tree=int((a.grad != t8).sum()),
                        mismatch_vs_folded=int((a.grad != fo).sum())))

# dW at T=300 tokens (T_pad=512, chunks of 64)
torch.manual_seed(2)
x = rand(300, 1024)
w = rand(384, 1024, scale=0.02).requires_grad_(True)
gy = rand(300, 384)
dg.fast_det_gemm(x, w).backward(gy)
gp = torch.zeros(512, 384, device="cuda", dtype=BF16)
xp = torch.zeros(512, 1024, device="cuda", dtype=BF16)
gp[:300], xp[:300] = gy, x
t8 = tree8(gp.t(), xp).to(BF16)
fo = folded(gp.t(), xp).to(BF16)
res["rows"].append(dict(kind="dW_T300", numel=w.grad.numel(),
                        mismatch_vs_tree=int((w.grad != t8).sum()),
                        mismatch_vs_folded=int((w.grad != fo).sum())))

# IR of a K=512, M=256 launch with the shipped plan
a = rand(256, 512)
b = rand(512, 384, scale=0.02)
out = torch.empty(256, 384, device="cuda", dtype=FP32)
cfg = kt.PREFILL_CONFIG
plan = kt.plan_chunks(512, block_k=cfg.block_k)
grid = (triton.cdiv(256, cfg.block_m) * triton.cdiv(384, cfg.block_n),)
extra = {}
compiled = kt.det_gemm_tree_kernel[grid](
    a, b, out, 256, 384, 512, a.stride(0), a.stride(1), b.stride(0), b.stride(1),
    out.stride(0), out.stride(1), CHUNK_LEN=plan.chunk_len, NUM_CHUNKS=8,
    BLOCK_M=cfg.block_m, BLOCK_N=cfg.block_n, BLOCK_K=plan.block_k, GROUP_M=cfg.group_m,
    EVEN_M=True, EVEN_N=False, MASK_K=False, num_warps=cfg.num_warps, num_stages=cfg.num_stages,
)
ttgir = compiled.asm["ttgir"]
ttir = compiled.asm["ttir"]
ptx = compiled.asm["ptx"]
pat = rf"arith\.addf [^\n]*: tensor<{cfg.block_m}x{cfg.block_n}xf32"
res["ir_K512_M256"] = dict(
    block_k=plan.block_k,
    ttir_addf=len(re.findall(pat, ttir)),
    ttgir_addf=len(re.findall(pat, ttgir)),
    ttir_scf_for=ttir.count("scf.for"),
    ttir_dot=ttir.count("tt.dot"),
    ptx_add_f32=len(re.findall(r"\badd(\.rn)?\.f32\b", ptx)),
    ptx_mma=ptx.count("mma.sync"),
)

# forward bit hashes for a few model shapes (ledger spot check)
hashes = {}
for name, (n, k) in {"q_proj": (4096, 6656), "k_proj": (256, 6656), "o_proj": (6656, 4096),
                     "down_proj": (6656, 19968)}.items():
    for m in (1, 64, 300):
        torch.manual_seed(100 + m)
        a = rand(m, k)
        w = rand(n, k, scale=0.02)
        hashes[f"{name}/M{m}"] = h16(dg.fast_det_gemm(a, w))
res["fwd_hashes"] = hashes
json.dump(res, open(out_path, "w"), indent=1)
print(json.dumps(res, indent=1))
