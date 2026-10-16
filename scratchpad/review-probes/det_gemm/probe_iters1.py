"""Probe: does the compiled kernel still implement the 8-leaf fp32 tree when CHUNK_LEN == BLOCK_K
(ITERS == 1)?  Suspicion: MLIR folds the 1-trip scf.for, then Triton's combine pass rewrites
dot(a,b,0) + d -> dot(a,b,d), so the pairwise tree adds become chained MMA accumulation.
Also: column-parallel dA at simulated TP2 through the autograd backward vs TP1."""

import re
import torch

from rl_engine.models.muse_glimmer.ops import det_gemm as dg
from rl_engine.models.muse_glimmer.ops.canonical_tree import combine_tree
from rl_engine.models.muse_glimmer.triton import det_gemm_tree as kt

torch.manual_seed(0)
DEV = "cuda"
BF = torch.bfloat16


def rand(*s, scale=1.0):
    return (torch.randn(*s, device=DEV) * scale).to(BF)


def partial_tree(a, w):
    chunk = a.size(1) // 8
    parts = []
    for c in range(8):
        cols = slice(c * chunk, (c + 1) * chunk)
        parts.append(dg.fast_det_gemm_partial(a[:, cols], w[:, cols], [c]))
    return combine_tree(parts)


def dot_summary(tag):
    launch = kt.last_launch()
    print(f"  [{tag}] launch plan={launch['plan']} cfg_block_k={launch['config']['block_k']}")


def check(tag, m, n, k, cfgs=()):
    a, w = rand(m, k), rand(n, k, scale=0.02)
    full = dg.fast_det_gemm(a, w, out_dtype=torch.float32)
    plan = kt.last_launch()["plan"]
    tree = partial_tree(a, w)
    eq = torch.equal(full, tree)
    diff = (full - tree).abs().max().item()
    nd = (full != tree).sum().item()
    print(f"{tag}: M={m} N={n} K={k} chunk_len={plan['chunk_len']} block_k={plan['block_k']} "
          f"ITERS={plan['chunk_len']//plan['block_k']}  kernel==combine_tree(8 partials): {eq} "
          f"(max|diff|={diff:.3e}, {nd}/{full.numel()} elems differ)")
    for cfg in cfgs:
        var = kt.tree_gemm(a, w.t(), out_dtype=torch.float32, config=cfg)
        p2 = kt.last_launch()["plan"]
        print(f"    variant block_k={p2['block_k']} ITERS={p2['chunk_len']//p2['block_k']}: "
              f"== pinned {torch.equal(var, full)}  == tree {torch.equal(var, tree)}")
    return a, w


print("== forward, model-like K (ITERS>1) ==")
check("K=4096", 64, 256, 4096)
print("== forward, ITERS==1 shapes ==")
check("K=512", 64, 256, 512, cfgs=[kt.TileConfig(64, 64, 32, 4, 4)])
check("K=256", 64, 256, 256)
check("K=2048 (SMALL_SHAPES 257x320x2048)", 257, 320, 2048)

print("== backward paths of model shapes with ITERS==1 ==")
# dA for k_proj/v_proj: reduction over N=256 -> chunk 32, BLOCK_K 32, ITERS=1
m, n, k = 64, 256, 6656
w = rand(n, k, scale=0.02)
dy = rand(m, n)
da = kt.tree_gemm(dy, w, out_dtype=torch.float32, allow_pad=True)  # what backward does (fp32 for compare)
plan = kt.last_launch()["plan"]
da_tree = partial_tree(dy, w.t().contiguous())
print(f"dA k_proj (reduce N=256): chunk_len={plan['chunk_len']} block_k={plan['block_k']} "
      f"kernel==tree: {torch.equal(da, da_tree)} max|diff|={(da-da_tree).abs().max().item():.3e} "
      f"ndiff={(da!=da_tree).sum().item()}/{da.numel()}")
# dW for M=256 tokens: reduction over M -> chunk 32, ITERS=1
m, n, k = 256, 384, 1024
a = rand(m, k)
dy = rand(m, n)
dw = kt.tree_gemm(dy.t(), a, out_dtype=torch.float32, allow_pad=True)
plan = kt.last_launch()["plan"]
dw_tree = partial_tree(dy.t().contiguous(), a.t().contiguous())
print(f"dW M=256 tokens (reduce M): chunk_len={plan['chunk_len']} block_k={plan['block_k']} "
      f"kernel==tree: {torch.equal(dw, dw_tree)} max|diff|={(dw-dw_tree).abs().max().item():.3e} "
      f"ndiff={(dw!=dw_tree).sum().item()}/{dw.numel()}")
# dW for M=4096 tokens: chunk 512, ITERS 8 (control)
m, n, k = 4096, 384, 1024
a = rand(m, k)
dy = rand(m, n)
dw = kt.tree_gemm(dy.t(), a, out_dtype=torch.float32, allow_pad=True)
plan = kt.last_launch()["plan"]
dw_tree = partial_tree(dy.t().contiguous(), a.t().contiguous())
print(f"dW M=4096 tokens (control): chunk_len={plan['chunk_len']} block_k={plan['block_k']} "
      f"kernel==tree: {torch.equal(dw, dw_tree)}")

print("== TTIR inspection of an ITERS==1 compile ==")
a, w = rand(64, 512), rand(256, 512, scale=0.02)
cfg = kt.DECODE64_CONFIG
plan = kt.plan_chunks(512, 8, block_k=cfg.block_k)
grid = (1,)
out = torch.empty(64, 256, device=DEV, dtype=torch.float32)
compiled = kt.det_gemm_tree_kernel[grid](
    a, w.t(), out, 64, 256, 512, a.stride(0), a.stride(1), w.t().stride(0), w.t().stride(1),
    out.stride(0), out.stride(1), CHUNK_LEN=plan.chunk_len, NUM_CHUNKS=8, BLOCK_M=cfg.block_m,
    BLOCK_N=cfg.block_n, BLOCK_K=plan.block_k, GROUP_M=cfg.group_m, EVEN_M=True, EVEN_N=True,
    MASK_K=False, num_warps=cfg.num_warps, num_stages=cfg.num_stages)
ttir = compiled.asm["ttir"]
dots = re.findall(r"tt\.dot .*", ttir)
addfs = re.findall(r"arith\.addf .*tensor<64x64xf32", ttir)
fors = ttir.count("scf.for")
print(f"ITERS=1 kernel TTIR: scf.for={fors}  tt.dot={len(dots)}  addf(64x64xf32)={len(addfs)}")
for d in dots:
    print("   ", d.strip()[:110])
print("== control: ITERS=2 (BLOCK_K=32) TTIR ==")
plan2 = kt.plan_chunks(512, 8, block_k=32)
compiled2 = kt.det_gemm_tree_kernel[grid](
    a, w.t(), out, 64, 256, 512, a.stride(0), a.stride(1), w.t().stride(0), w.t().stride(1),
    out.stride(0), out.stride(1), CHUNK_LEN=plan2.chunk_len, NUM_CHUNKS=8, BLOCK_M=cfg.block_m,
    BLOCK_N=cfg.block_n, BLOCK_K=32, GROUP_M=cfg.group_m, EVEN_M=True, EVEN_N=True,
    MASK_K=False, num_warps=cfg.num_warps, num_stages=cfg.num_stages)
ttir2 = compiled2.asm["ttir"]
print(f"ITERS=2 kernel TTIR: scf.for={ttir2.count('scf.for')}  tt.dot={len(re.findall(r'tt.dot ', ttir2))}  "
      f"addf(64x64xf32)={len(re.findall(r'arith.addf .*tensor<64x64xf32', ttir2))}")

print("== column-parallel dA at simulated TP2 through autograd backward (q_proj N=4096,K=6656) ==")
m, n, k = 64, 4096, 6656
w = rand(n, k, scale=0.02)
x = rand(m, k)
gy = rand(m, n)
x1 = x.clone().requires_grad_(True)
dg.fast_det_gemm(x1, w).backward(gy)
da_tp1 = x1.grad
parts = []
for r in range(2):
    sl = slice(r * n // 2, (r + 1) * n // 2)
    xr = x.clone().requires_grad_(True)
    dg.fast_det_gemm(xr, w[sl]).backward(gy[:, sl])  # rank-local autograd, 8 chunks over N/2, bf16 out
    parts.append(xr.grad)
da_tp2_bf16sum = (parts[0].float() + parts[1].float()).to(BF)
print(f"autograd dA: TP1 vs TP2(sum of rank-local bf16 dA): equal={torch.equal(da_tp1, da_tp2_bf16sum)} "
      f"ndiff={(da_tp1 != da_tp2_bf16sum).sum().item()}/{da_tp1.numel()}")
# the correct way (not reachable through autograd): NUM_CHUNKS=4 fp32 partials over N halves + combine
good = []
for r in range(2):
    sl = slice(r * n // 2, (r + 1) * n // 2)
    good.append(kt.tree_gemm(gy[:, sl], w[sl], out_dtype=torch.float32, num_chunks=4))
da_tp2_good = combine_tree(good).to(BF)
print(f"manual TP2 (num_chunks=4 fp32 partials + combine_tree): equal to TP1 autograd dA = "
      f"{torch.equal(da_tp1, da_tp2_good)}")
