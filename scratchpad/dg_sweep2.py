"""Narrow-N decode tiles (k/v proj) and backward-layout timings."""

import torch

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

from rl_engine.models.muse_glimmer.ops import det_gemm as dg  # noqa: E402
from rl_engine.models.muse_glimmer.triton import det_gemm_tree as kt  # noqa: E402

dev = "cuda"


def timeit(fn, iters=10, warm=3):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


print("=== narrow N decode")
for M, N, K in [(16, 256, 6656), (64, 256, 6656), (16, 4096, 6656)]:
    a = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
    w = (torch.randn(N, K, device=dev) * 0.02).to(torch.bfloat16)
    wt = w.t()
    t_cublas = timeit(lambda: torch.matmul(a, wt))
    print(f"\n--- M={M} N={N} K={K} cuBLAS {t_cublas:.4f} ms")
    ref = None
    for cfg in [
        kt.TileConfig(16, 64, 64, 4, 4),
        kt.TileConfig(16, 32, 64, 2, 4),
        kt.TileConfig(16, 16, 64, 1, 4),
        kt.TileConfig(16, 16, 64, 2, 4),
        kt.TileConfig(16, 16, 32, 1, 5),
        kt.TileConfig(16, 32, 64, 1, 4),
        kt.TileConfig(32, 16, 64, 2, 4),
        kt.TileConfig(32, 32, 64, 2, 4),
        kt.TileConfig(32, 64, 64, 4, 4),
        kt.TileConfig(64, 16, 64, 2, 4),
    ]:
        if cfg.block_m > max(M, 16):
            continue
        try:
            y = kt.tree_gemm(a, wt, config=cfg)
            info = kt.last_launch()
            ms = timeit(lambda: kt.tree_gemm(a, wt, config=cfg))
        except Exception as exc:  # noqa: BLE001
            print(f"  {cfg}: FAILED {type(exc).__name__}: {str(exc)[:100]}")
            continue
        if ref is None:
            ref = y
        print(
            f"  BM={cfg.block_m:3d} BN={cfg.block_n:3d} BK={cfg.block_k} w={cfg.num_warps} "
            f"s={cfg.num_stages}: {ms:.4f} ms x{ms / t_cublas:.2f} grid={info['grid']} "
            f"regs={info['n_regs']} spills={info['n_spills']} bitwise={torch.equal(y, ref)}"
        )

print("\n=== backward layouts at M=4096 (pinned configs)")
for name, N, K in [("q_proj", 4096, 6656), ("k_proj", 256, 6656), ("down", 6656, 19968),
                   ("gate", 19968, 6656)]:
    M = 4096
    a = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
    w = (torch.randn(N, K, device=dev) * 0.02).to(torch.bfloat16)
    dy = torch.randn(M, N, device=dev, dtype=torch.bfloat16)
    flop = 2.0 * M * N * K
    t_fwd = timeit(lambda: kt.tree_gemm(a, w.t()))
    fwd_info = kt.last_launch()
    t_da = timeit(lambda: kt.tree_gemm(dy, w))  # reduction over N, b = w (N, K) row-major
    da_info = kt.last_launch()
    t_dw = timeit(lambda: kt.tree_gemm(dy.t(), a, allow_pad=True))  # reduction over M
    dw_info = kt.last_launch()
    t_dw_c = timeit(lambda: kt.tree_gemm(dy.t().contiguous(), a, allow_pad=True))
    t_cb_da = timeit(lambda: torch.matmul(dy, w))
    t_cb_dw = timeit(lambda: torch.matmul(dy.t(), a))
    print(
        f"{name:8s} fwd {t_fwd:7.3f} ms ({flop / t_fwd / 1e9:5.1f} TF) | "
        f"dA {t_da:7.3f} ms ({flop / t_da / 1e9:5.1f} TF, cuBLAS {t_cb_da:.3f}; "
        f"BK={da_info['plan']['block_k']} spills={da_info['n_spills']}) | "
        f"dW {t_dw:7.3f} ms ({flop / t_dw / 1e9:5.1f} TF, cuBLAS {t_cb_dw:.3f}; "
        f"spills={dw_info['n_spills']}; with .contiguous() dyT: {t_dw_c:.3f})"
    )

print("\n=== full autograd backward (M=4096 gate shape) strict vs torch bf16")
M, N, K = 4096, 19968, 6656
a = torch.randn(M, K, device=dev, dtype=torch.bfloat16, requires_grad=True)
w = (torch.randn(N, K, device=dev) * 0.02).to(torch.bfloat16).requires_grad_(True)
g = torch.randn(M, N, device=dev, dtype=torch.bfloat16)


def strict_fb():
    y = dg.fast_det_gemm(a, w)
    y.backward(g)


def torch_fb():
    y = torch.nn.functional.linear(a, w)
    y.backward(g)


print(f"strict fwd+bwd {timeit(strict_fb, iters=5):.3f} ms ; torch fwd+bwd {timeit(torch_fb, iters=5):.3f} ms")
