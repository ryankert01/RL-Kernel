"""P9: report clarification 1 -- theta**(-2i/128) vs 1/pow(theta, 2i/128): do they differ on CUDA?
P10: fp32 q/k input accepted by the strict op -> output dtype (declared cast skipped?)
P11: cos=None on a 'local' call is indistinguishable from global (interface hazard) -- just show it runs.
"""
import torch

from rl_engine.models.muse_glimmer.ops.qknorm_rope import qknorm_scaled_rope, rope_table

dev = torch.device("cuda")
exp = torch.arange(0, 128, 2, dtype=torch.float32, device=dev) / 128
a = 1.0 / torch.pow(500000.0, exp)
b = torch.pow(500000.0, -exp)
print(f"P9 inv_freq forms differ in {(a != b).sum().item()} of 64 elements; max rel {( (a-b).abs()/a ).max().item():.3e}")

T = 16
gen = torch.Generator(device=dev).manual_seed(1)
q32 = torch.randn(T, 32, 128, device=dev, generator=gen)
k32 = torch.randn(T, 2, 128, device=dev, generator=gen)
cos, sin = rope_table(torch.arange(T, device=dev), device=dev)
yq, yk = qknorm_scaled_rope(q32, k32, cos, sin)
print(f"P10 fp32 inputs accepted silently: out dtypes {yq.dtype} {yk.dtype} (contract output: bf16)")
yq16, _ = qknorm_scaled_rope(q32.half(), k32.half(), cos, sin)
print(f"P10 fp16 inputs accepted silently: out dtype {yq16.dtype}")
q = q32.to(torch.bfloat16)
k = k32.to(torch.bfloat16)
qg = q.clone().requires_grad_(True)
kg = k.clone().requires_grad_(True)
qh, kh = qknorm_scaled_rope(qg, kg, cos, sin)
yg, _ = qknorm_scaled_rope(q, k, None, None)
print(f"P11 local call with cos=None silently runs as NoPE: differs from rope path in {(yg != qh).sum().item()} elems, no error")
