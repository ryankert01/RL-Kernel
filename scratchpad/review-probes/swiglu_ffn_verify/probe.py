import torch
from rl_engine.models.muse_glimmer.ops.swiglu_ffn import swiglu_ffn, default_gemm, TritonDetGemmAdapter
H, F = 512, 768  # small multiples of 256
torch.manual_seed(0)
dev = "cuda"
wg = (torch.randn(F, H, device=dev) * 0.02).bfloat16()
wu = (torch.randn(F, H, device=dev) * 0.02).bfloat16()
wd = (torch.randn(H, F, device=dev) * 0.02).bfloat16()
for label, gemm in (("fast_det_gemm", default_gemm()), ("adapter", TritonDetGemmAdapter())):
    for T in (64, 72, 77, 256, 4000):
        for wgrad in (True, False):
            x = torch.randn(T, H, device=dev).bfloat16().requires_grad_()
            leaves = [w.clone().requires_grad_(wgrad) for w in (wg, wu, wd)]
            try:
                y = swiglu_ffn(x, *leaves, gemm=gemm)
                y.backward(torch.randn_like(y))
                print(f"{label:14s} T={T:5d} wgrad={wgrad}: OK")
            except Exception as e:
                print(f"{label:14s} T={T:5d} wgrad={wgrad}: {type(e).__name__}: {e}")
print("default_gemm type:", type(default_gemm()).__name__)
