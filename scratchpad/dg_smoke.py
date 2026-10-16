import time

import torch

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

from rl_engine.models.muse_glimmer.ops import det_gemm as dg  # noqa: E402
from rl_engine.models.muse_glimmer.ops.canonical_tree import combine_tree, rank_chunk_ids  # noqa: E402
from rl_engine.models.muse_glimmer.triton import det_gemm_tree as kt  # noqa: E402

dev = "cuda"
torch.manual_seed(0)


def rand(*s, scale=1.0):
    return (torch.randn(*s, device=dev) * scale).to(torch.bfloat16)


def report(name, y, ref):
    err = (y.double() - ref.double()).abs()
    rel = err / ref.double().abs().clamp_min(1e-6)
    print(f"  {name}: max_abs={err.max().item():.3e} max_rel={rel.max().item():.3e}")


# 1. small shape, correctness and compile
M, N, K = 100, 192, 1024
a, w = rand(M, K), rand(N, K, scale=0.05)
t0 = time.time()
y = dg.fast_det_gemm(a, w)
torch.cuda.synchronize()
print(f"compile+run small: {time.time() - t0:.1f}s; launch={kt.last_launch()}")
ref = dg.forward_fp32(a, w)
report("small bf16", y, ref)
y32 = dg.fast_det_gemm(a, w, out_dtype=torch.float32)
report("small fp32", y32, ref)
print("  fp32->bf16 equals bf16 path:", torch.equal(y32.to(torch.bfloat16), y))

# 2. cross-config bitwise: decode16 vs decode64 vs prefill vs BLOCK_K=32, on the same rows
M, N, K = 4096, 4096, 6656
a, w = rand(M, K), rand(N, K, scale=0.02)
y_pref = dg.fast_det_gemm(a, w, out_dtype=torch.float32)
print("prefill launch:", kt.last_launch())
y_d16 = dg.fast_det_gemm(a[:16], w, out_dtype=torch.float32)
print("decode16 launch:", kt.last_launch())
y_d64 = dg.fast_det_gemm(a[:64], w, out_dtype=torch.float32)
print("decode64 launch:", kt.last_launch())
print("  decode16 == prefill rows:", torch.equal(y_d16, y_pref[:16]))
print("  decode64 == prefill rows:", torch.equal(y_d64, y_pref[:64]))
cfg32 = kt.TileConfig(64, 64, 32, 4, 3)
y_k32 = kt.tree_gemm(a[:256], w.t(), out_dtype=torch.float32, config=cfg32)
print("  BLOCK_K=32 == prefill rows:", torch.equal(y_k32, y_pref[:256]))
cfg_other = kt.TileConfig(64, 128, 64, 8, 2)
y_o = kt.tree_gemm(a[:256], w.t(), out_dtype=torch.float32, config=cfg_other)
print("  64x128 w8 s2 == prefill rows:", torch.equal(y_o, y_pref[:256]))
ref = dg.forward_fp32(a[:256], w)
report("real-shape fp32 out", y_pref[:256], ref)
report("real-shape bf16 out", y_pref[:256].to(torch.bfloat16), ref)

# 3. w_kn layout bitwise
wk = w.t().contiguous()
y_kn = dg.fast_det_gemm(a[:256], wk, w_kn=True, out_dtype=torch.float32)
print("  w_kn layout bitwise:", torch.equal(y_kn, y_pref[:256]))

# 4. TP simulation
chunk = K // 8
for tp in (2, 4, 8):
    parts = []
    for r in range(tp):
        ids = rank_chunk_ids(tp, r)
        lo, hi = ids[0] * chunk, (ids[-1] + 1) * chunk
        parts.append(dg.fast_det_gemm_partial(a[:256, lo:hi], w[:, lo:hi], ids))
    comb = combine_tree(parts)
    print(f"  TP{tp} fp32 bitwise:", torch.equal(comb, y_pref[:256]),
          " bf16 bitwise:", torch.equal(comb.to(torch.bfloat16), y_pref[:256].to(torch.bfloat16)))

# 5. backward
M, N, K = 256, 512, 1024
a = rand(M, K).requires_grad_(True)
w = rand(N, K, scale=0.05).requires_grad_(True)
g = rand(M, N)
y = dg.fast_det_gemm(a, w)
y.backward(g)
da1, dw1 = a.grad.clone(), w.grad.clone()
a.grad = None
w.grad = None
dg.fast_det_gemm(a, w).backward(g)
print("  backward repeat bitwise:", torch.equal(da1, a.grad), torch.equal(dw1, w.grad))
a32 = a.detach().float().requires_grad_(True)
w32 = w.detach().float().requires_grad_(True)
(a32 @ w32.t()).backward(g.float())
report("dA vs fp32 autograd", da1, a32.grad)
report("dW vs fp32 autograd", dw1, w32.grad)
# ragged M (16 tokens) dW
a = rand(16, K).requires_grad_(True)
w = rand(N, K, scale=0.05).requires_grad_(True)
dg.fast_det_gemm(a, w).backward(rand(16, N))
print("  ragged dW ok:", w.grad.shape, torch.isfinite(w.grad).all().item(), kt.last_launch()["plan"])
# fp32 out backward (lm_head path)
a = rand(64, K).requires_grad_(True)
y = dg.fast_det_gemm(a, w, out_dtype=torch.float32)
y.float().sum().backward()
print("  fp32-out backward ok:", a.grad.dtype, a.grad.shape)
print("trace:", dg.trace(4096))
