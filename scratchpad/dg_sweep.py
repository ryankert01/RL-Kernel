"""Offline tile-constant sweep (NOT runtime autotune): pick pinned configs per shape class."""

import sys

import torch

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

from rl_engine.models.muse_glimmer.triton import det_gemm_tree as kt  # noqa: E402

dev = "cuda"
mode = sys.argv[1] if len(sys.argv) > 1 else "prefill"


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


if mode == "prefill":
    shapes = [(4096, 19968, 6656), (4096, 6656, 19968), (4096, 4096, 6656), (4096, 6656, 4096)]
    configs = [
        kt.TileConfig(128, 64, 64, 8, 3),
        kt.TileConfig(128, 64, 32, 8, 3),
        kt.TileConfig(128, 64, 32, 8, 4),
        kt.TileConfig(128, 64, 64, 8, 2),
        kt.TileConfig(64, 128, 64, 8, 3),
        kt.TileConfig(64, 128, 32, 8, 4),
        kt.TileConfig(64, 64, 64, 4, 3),
        kt.TileConfig(64, 64, 64, 4, 4),
        kt.TileConfig(64, 64, 32, 4, 4),
        kt.TileConfig(64, 64, 32, 4, 5),
        kt.TileConfig(64, 64, 64, 8, 3),
        kt.TileConfig(128, 128, 64, 8, 3),
        kt.TileConfig(128, 128, 32, 8, 3),
        kt.TileConfig(128, 128, 32, 16, 3),
        kt.TileConfig(128, 128, 64, 16, 3),
        kt.TileConfig(128, 64, 64, 4, 3),
        kt.TileConfig(64, 32, 64, 4, 4),
        kt.TileConfig(32, 64, 64, 4, 4),
    ]
else:
    shapes = [(16, 4096, 6656), (16, 19968, 6656), (16, 6656, 19968), (16, 256, 6656),
              (64, 4096, 6656), (64, 19968, 6656), (64, 6656, 19968)]
    configs = [
        kt.TileConfig(16, 64, 64, 4, 4),
        kt.TileConfig(16, 64, 64, 4, 3),
        kt.TileConfig(16, 128, 64, 4, 4),
        kt.TileConfig(16, 32, 64, 4, 4),
        kt.TileConfig(16, 64, 32, 4, 5),
        kt.TileConfig(16, 64, 64, 2, 4),
        kt.TileConfig(64, 64, 64, 4, 4),
        kt.TileConfig(64, 64, 32, 4, 4),
        kt.TileConfig(64, 64, 64, 8, 3),
        kt.TileConfig(64, 32, 64, 4, 4),
        kt.TileConfig(32, 64, 64, 4, 4),
        kt.TileConfig(64, 128, 32, 8, 3),
    ]

for M, N, K in shapes:
    a = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
    w = (torch.randn(N, K, device=dev) * 0.02).to(torch.bfloat16)
    wt = w.t()
    flop = 2.0 * M * N * K
    t_cublas = timeit(lambda: torch.matmul(a, wt))
    print(f"\n=== M={M} N={N} K={K}  cuBLAS bf16 {t_cublas:.3f} ms = {flop / t_cublas / 1e9:.1f} TFLOP/s")
    ref = None
    for cfg in configs:
        if cfg.block_m > M and M > 16:
            continue
        try:
            y = kt.tree_gemm(a, wt, config=cfg)
            info = kt.last_launch()
            ms = timeit(lambda: kt.tree_gemm(a, wt, config=cfg))
        except Exception as exc:  # noqa: BLE001
            print(f"  {cfg}: FAILED {type(exc).__name__}: {str(exc)[:80]}")
            continue
        if ref is None:
            ref = y
        same = torch.equal(y, ref)
        print(
            f"  BM={cfg.block_m:3d} BN={cfg.block_n:3d} BK={cfg.block_k:2d} w={cfg.num_warps:2d} "
            f"s={cfg.num_stages}: {ms:8.3f} ms  {flop / ms / 1e9:6.1f} TFLOP/s  "
            f"x{ms / t_cublas:5.2f} cuBLAS  regs={info['n_regs']} spills={info['n_spills']} "
            f"bitwise={same}"
        )
