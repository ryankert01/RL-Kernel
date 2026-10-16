import time

import torch

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
from rl_engine.backends.shared.triton.gemm.det_gemm import TritonDetGemmOp, _triton_gemm_fp32

op = TritonDetGemmOp()
dev = "cuda"


def bench(fn, *args, reps=3):
    fn(*args)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        out = fn(*args)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps * 1e3, out


g = torch.Generator(device="cpu").manual_seed(0)
shapes = {
    "o_proj_chunk": (256, 512, 6656),
    "down_chunk": (256, 2496, 6656),
    "vocab_tile": (64, 6656, 25344),
}
for name, (M, K, N) in shapes.items():
    a = torch.randn(M, K, generator=g).to(dev, torch.bfloat16)
    w = torch.randn(N, K, generator=g).to(dev, torch.bfloat16)
    b = w.t().contiguous()
    ms32, out32 = bench(_triton_gemm_fp32, a, b)
    ms16, out16 = bench(op, a, b)
    ms_ref, ref = bench(lambda a, b: (a.float() @ b.float()), a, b)
    err = (out32 - ref).abs().max().item()
    rep = _triton_gemm_fp32(a, b)
    row = _triton_gemm_fp32(a[3:4].contiguous(), b)
    print(
        f"{name}: fp32-det {ms32:.1f} ms | bf16-tree {ms16:.1f} ms | cublas-fp32 {ms_ref:.2f} ms"
        f" | max|fp32det-cublas| {err:.3e} | repeat_equal {torch.equal(rep, out32)}"
        f" | batch_inv {torch.equal(row[0], out32[3])}"
    )
