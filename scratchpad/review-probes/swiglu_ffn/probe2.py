import torch
from rl_engine.models.muse_glimmer.ops.swiglu_ffn import TritonDetGemmAdapter, default_gemm, swiglu_ffn
H, F = 6656, 19968
def randn(shape, seed, scale=1.0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return (torch.randn(*shape, generator=g, dtype=torch.float32) * scale).to("cuda", torch.bfloat16)
x = randn((256, H), 1); wg = randn((F, H), 2, H**-0.5); wu = randn((F, H), 3, H**-0.5); wd = randn((H, F), 4, F**-0.5)
with torch.no_grad():
    ya = swiglu_ffn(x, wg, wu, wd, gemm=TritonDetGemmAdapter())
    yf = swiglu_ffn(x, wg, wu, wd, gemm=default_gemm())
    ga = TritonDetGemmAdapter()(x, wg); gf = default_gemm()(x, wg)
print("FFN y adapter vs fast_det_gemm: equal =", torch.equal(ya, yf), "| mismatching elems =", (ya != yf).sum().item(), "of", ya.numel(),
      "| max abs diff =", (ya.float() - yf.float()).abs().max().item())
print("gate GEMM adapter vs fast: equal =", torch.equal(ga, gf), "| mismatching =", (ga != gf).sum().item())
