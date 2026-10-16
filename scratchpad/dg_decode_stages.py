import torch
torch.backends.cuda.matmul.allow_tf32 = False
from rl_engine.models.muse_glimmer.triton import det_gemm_tree as kt
dev = "cuda"
def timeit(fn, iters=20, warm=3):
    for _ in range(warm): fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters
for M, N, K in [(16, 4096, 6656), (16, 256, 6656), (16, 19968, 6656), (16, 6656, 19968), (64, 4096, 6656), (64, 19968, 6656)]:
    a = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
    w = (torch.randn(N, K, device=dev) * 0.02).to(torch.bfloat16); wt = w.t()
    tc = timeit(lambda: torch.matmul(a, wt))
    print(f"--- M={M} N={N} K={K} cuBLAS {tc:.4f}")
    ref = None
    for cfg in [kt.TileConfig(16, 64, 64, 4, 4), kt.TileConfig(16, 64, 64, 4, 6), kt.TileConfig(16, 64, 64, 4, 8),
                kt.TileConfig(16, 64, 128, 4, 4), kt.TileConfig(16, 128, 64, 4, 6), kt.TileConfig(16, 64, 128, 4, 6),
                kt.TileConfig(32, 64, 64, 4, 4), kt.TileConfig(32, 64, 64, 4, 6), kt.TileConfig(32, 64, 128, 4, 4)]:
        if cfg.block_m > M: continue
        try:
            y = kt.tree_gemm(a, wt, config=cfg); info = kt.last_launch()
            ms = timeit(lambda: kt.tree_gemm(a, wt, config=cfg))
        except Exception as exc:
            print(f"  {cfg}: FAILED {type(exc).__name__}: {str(exc)[:100]}"); continue
        ref = y if ref is None else ref
        print(f"  BM={cfg.block_m} BN={cfg.block_n} BK={cfg.block_k} w={cfg.num_warps} s={cfg.num_stages}: {ms:.4f} ms x{ms/tc:.2f} regs={info['n_regs']} spills={info['n_spills']} BK_eff={info['plan']['block_k']} bitwise={torch.equal(y, ref)}")
