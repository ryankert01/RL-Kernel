"""Probe: which repo GEMM path should the swiglu_ffn default adapter use?

Measures, at the Muse-Glimmer gate GEMM shape (M=4096, K=6656, N=19968):
  (1) _triton_tree_gemm        bf16 leaf/node tree (TritonDetGemmOp.__call__ path)
  (2) _triton_gemm_fp32        fp32 FMA path (PROMOTE_INPUTS=True), fp32 out
  (3) _det_gemm_fp32_kernel    same kernel, bf16 operands (tensor-core MMA), fp32 acc/out,
                               w read in (N,K) layout via strides (no transpose copy)
  (4) torch bf16 matmul (native reference for the ratio)
and checks (3) for repeat determinism, batch invariance (M=1 / 77 / 4096) and fp64 accuracy.
"""

import time

import torch
import triton

from rl_engine.backends.shared.triton.gemm.det_gemm import (
    _BLOCK_K,
    _BLOCK_M,
    _BLOCK_N,
    _det_gemm_fp32_kernel,
    _triton_gemm_fp32,
    _triton_tree_gemm,
)

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
dev = torch.device("cuda")


def gemm3(a: torch.Tensor, w: torch.Tensor, promote: bool = False, num_warps=4, num_stages=3):
    """y = a @ w.T, fp32 out; a (M,K) any strides, w (N,K) any strides."""
    M, K = a.shape
    N = w.shape[0]
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    grid = (triton.cdiv(M, _BLOCK_M), triton.cdiv(N, _BLOCK_N))
    _det_gemm_fp32_kernel[grid](
        a, w, c, M, N, K,
        a.stride(0), a.stride(1),
        w.stride(1), w.stride(0),  # b[k, n] = w[n, k]
        c.stride(0), c.stride(1),
        BLOCK_M=_BLOCK_M, BLOCK_N=_BLOCK_N, BLOCK_K=_BLOCK_K,
        PROMOTE_INPUTS=promote, num_warps=num_warps, num_stages=num_stages,
    )
    return c


def timeit(fn, iters=3, warm=1):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / iters * 1e3


M, K, N = 4096, 6656, 19968
flops = 2.0 * M * K * N
g = torch.Generator(device=dev).manual_seed(0)
a = torch.randn((M, K), device=dev, generator=g).to(torch.bfloat16)
w = (torch.randn((N, K), device=dev, generator=g) / K**0.5).to(torch.bfloat16)
w_t = w.t().contiguous()  # (K, N)

print(f"shape M={M} K={K} N={N}  ({flops/1e12:.2f} TFLOP)")
ms = timeit(lambda: torch.matmul(a, w_t))
print(f"(4) torch bf16 matmul          : {ms:8.2f} ms  {flops/ms/1e9:7.1f} TFLOP/s")
ms = timeit(lambda: gemm3(a, w, promote=False))
print(f"(3) fp32_kernel bf16-MMA (N,K) : {ms:8.2f} ms  {flops/ms/1e9:7.1f} TFLOP/s")
ms = timeit(lambda: gemm3(a, w_t.t(), promote=False))
print(f"(3b) same, w given as (K,N)^T  : {ms:8.2f} ms  {flops/ms/1e9:7.1f} TFLOP/s")
for nw, ns in ((4, 2), (4, 4), (8, 3)):
    ms = timeit(lambda: gemm3(a, w, promote=False, num_warps=nw, num_stages=ns))
    print(f"(3) warps={nw} stages={ns}         : {ms:8.2f} ms  {flops/ms/1e9:7.1f} TFLOP/s")
ms = timeit(lambda: _triton_gemm_fp32(a, w_t), iters=1)
print(f"(2) _triton_gemm_fp32 promote  : {ms:8.2f} ms  {flops/ms/1e9:7.1f} TFLOP/s")
ms = timeit(lambda: _triton_tree_gemm(a, w_t), iters=1)
print(f"(1) _triton_tree_gemm bf16     : {ms:8.2f} ms  {flops/ms/1e9:7.1f} TFLOP/s")

# --- numerics of (3) ---
y_full = gemm3(a, w)
y_rep = gemm3(a, w)
print("repeat determinism (3):", torch.equal(y_full, y_rep))
y1 = gemm3(a[7:8], w)
y77 = gemm3(a[3:80], w)
print("batch invariance M=1 vs 4096:", torch.equal(y1[0], y_full[7]),
      " M=77 vs 4096:", torch.equal(y77, y_full[3:80]))
ref = a.double() @ w.double().t()
err = (y_full.double() - ref).abs()
print(f"fp64 accuracy (3): max_abs={err.max().item():.3e} max_rel={(err/ref.abs().clamp_min(1e-6)).max().item():.3e}")
# fp32-promoted path vs MMA path: do they agree bitwise? (not required; informative)
y_promote = gemm3(a[:256], w, promote=True)
d = (y_promote - y_full[:256]).abs()
print(f"promote(FMA) vs bf16-MMA: equal={torch.equal(y_promote, y_full[:256])} max_abs_diff={d.max().item():.3e}")
# 8-chunk tree vs single pass (informative: the tree changes bits, as designed)
c = K // 8
parts = [gemm3(a[:256, i*c:(i+1)*c], w[:, i*c:(i+1)*c]) for i in range(8)]
tree = ((parts[0]+parts[1])+(parts[2]+parts[3]))+((parts[4]+parts[5])+(parts[6]+parts[7]))
d = (tree - y_full[:256]).abs()
print(f"8-chunk tree vs single pass: max_abs_diff={d.max().item():.3e}")
err = (tree.double() - ref[:256]).abs()
print(f"8-chunk tree fp64 accuracy: max_abs={err.max().item():.3e}")
print("strided chunk slices (a[:, k0:k1], w[:, k0:k1]) worked without copies.")
