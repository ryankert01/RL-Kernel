"""Review probes for O4 qknorm_scaled_rope.

P1: TP head-shard invariance (HQ=32 vs HQ in {16, 8, 4}; HK=2 replicated) bitwise, fwd + bwd.
P2: rope_table built on CPU vs CUDA: do the bits differ? (contract: fixed torch-CUDA function)
P3: denormal edge: kernel (Triton FTZ) vs golden bitwise on tiny inputs.
P4: int32 offset overflow guard: is there any validation for T*HQ*D >= 2**31? (shape only, no run)
P5: non-contiguous q/k (head-major view) accepted silently via .contiguous()?
P6: fp32 grads into backward accepted; dtype of dq.
"""
import torch

from rl_engine.models.muse_glimmer.ops.qknorm_rope import (
    qknorm_scaled_rope,
    qknorm_scaled_rope_fp32,
    rope_table,
)

dev = torch.device("cuda")
torch.backends.cuda.matmul.allow_tf32 = False
T = 4096
gen = torch.Generator(device=dev).manual_seed(123)
q = (torch.randn(T, 32, 128, device=dev, generator=gen) * 2).to(torch.bfloat16)
k = (torch.randn(T, 2, 128, device=dev, generator=gen) * 2).to(torch.bfloat16)
pos = torch.arange(T, device=dev)
cos, sin = rope_table(pos, device=dev)
dyq = torch.randn(q.shape, device=dev, generator=gen).to(torch.bfloat16)
dyk = torch.randn(k.shape, device=dev, generator=gen).to(torch.bfloat16)

print("== P1 TP head-shard invariance (forward + backward) ==")
qg = q.clone().requires_grad_(True)
kg = k.clone().requires_grad_(True)
qh, kh = qknorm_scaled_rope(qg, kg, cos, sin)
dq, dk = torch.autograd.grad((qh, kh), (qg, kg), (dyq, dyk))
for tp in (2, 4, 8):
    hq = 32 // tp
    ok_f = ok_b = True
    for r in range(tp):
        sl = slice(r * hq, (r + 1) * hq)
        qs = q[:, sl].contiguous().requires_grad_(True)
        ks = k.clone().requires_grad_(True)
        qhs, khs = qknorm_scaled_rope(qs, ks, cos, sin)
        dqs, dks = torch.autograd.grad((qhs, khs), (qs, ks), (dyq[:, sl].contiguous(), dyk))
        ok_f &= torch.equal(qhs, qh[:, sl]) and torch.equal(khs, kh)
        ok_b &= torch.equal(dqs, dq[:, sl]) and torch.equal(dks, dk)
    print(f"  TP{tp} (HQ={hq}, BLOCK_HQ={hq}): fwd bitwise={ok_f} bwd bitwise={ok_b}")
# also the global path
qh_g, kh_g = qknorm_scaled_rope(q, k, None, None)
ok = all(
    torch.equal(qknorm_scaled_rope(q[:, r * 4:(r + 1) * 4].contiguous(), k, None, None)[0], qh_g[:, r * 4:(r + 1) * 4])
    for r in range(8)
)
print(f"  TP8 global path fwd bitwise={ok}")

print("== P2 rope_table CPU vs CUDA ==")
c_cpu, s_cpu = rope_table(pos.cpu(), device="cpu")
c_gpu, s_gpu = cos, sin
dc = (c_cpu.to(dev) != c_gpu).sum().item()
ds = (s_cpu.to(dev) != s_gpu).sum().item()
print(f"  cos mismatching elements (of {c_gpu.numel()}): {dc}; sin: {ds}; "
      f"max|dcos|={(c_cpu.to(dev)-c_gpu).abs().max().item():.3e} max|dsin|={(s_cpu.to(dev)-s_gpu).abs().max().item():.3e}")
# inv_freq alone CPU vs CUDA (HF computes inv_freq on CPU at init)
exp_cpu = torch.arange(0, 128, 2, dtype=torch.float32) / 128
exp_gpu = exp_cpu.to(dev)
inv_cpu = 1.0 / torch.pow(500000.0, exp_cpu)
inv_gpu = 1.0 / torch.pow(500000.0, exp_gpu)
print(f"  inv_freq CPU vs CUDA mismatching: {(inv_cpu.to(dev) != inv_gpu).sum().item()} of 64")
# effect on the op: CPU-built table moved to GPU vs GPU-built table
qa, _ = qknorm_scaled_rope(q, k, c_cpu.to(dev), s_cpu.to(dev))
qb, _ = qknorm_scaled_rope(q, k, c_gpu, s_gpu)
print(f"  q_hat bits differ with CPU-built table moved to CUDA: {(qa != qb).sum().item()} of {qa.numel()}")

print("== P3 denormal edge (Triton FTZ vs torch) ==")
tiny = (torch.randn(8, 32, 128, device=dev, generator=gen) * 1e-20).to(torch.bfloat16)
tinyk = (torch.randn(8, 2, 128, device=dev, generator=gen) * 1e-20).to(torch.bfloat16)
yq, yk = qknorm_scaled_rope(tiny, tinyk, cos[:8], sin[:8])
gq, gk = qknorm_scaled_rope_fp32(tiny, tinyk, cos[:8], sin[:8])
print(f"  tiny inputs: kernel vs golden->bf16 mismatches q={(yq != gq.to(torch.bfloat16)).sum().item()} "
      f"k={(yk != gk.to(torch.bfloat16)).sum().item()}; max|q_hat|={yq.float().abs().max().item():.3e} golden max={gq.abs().max().item():.3e}")
# subnormal-range bf16 inputs (2^-130)
sub = torch.full((4, 32, 128), 2.0 ** -130, device=dev).to(torch.bfloat16)
subk = torch.full((4, 2, 128), 2.0 ** -130, device=dev).to(torch.bfloat16)
yq, yk = qknorm_scaled_rope(sub, subk, None, None)
gq, gk = qknorm_scaled_rope_fp32(sub, subk, None, None)
print(f"  subnormal inputs: kernel vs golden->bf16 mismatches q={(yq != gq.to(torch.bfloat16)).sum().item()}; "
      f"kernel max={yq.float().abs().max().item():.3e} golden max={gq.abs().max().item():.3e}")

print("== P5 non-contiguous inputs accepted silently ==")
q_hm = q.transpose(0, 1)  # (32, T, 128) view, non-contiguous when viewed back as (T,32,128)
q_nc = q_hm.contiguous().transpose(0, 1)  # (T,32,128) non-contiguous
assert not q_nc.is_contiguous()
y1, _ = qknorm_scaled_rope(q_nc, k, cos, sin)
print(f"  non-contiguous q accepted: bitwise equal to contiguous run: {torch.equal(y1, qh)}")

print("== P6 fp32 output-grad into backward ==")
qg = q.clone().requires_grad_(True)
kg = k.clone().requires_grad_(True)
qh2, kh2 = qknorm_scaled_rope(qg, kg, cos, sin)
dq32, dk32 = torch.autograd.grad((qh2, kh2), (qg, kg), (dyq.float(), dyk.float()), allow_unused=False) if False else (None, None)
try:
    dq32, dk32 = torch.autograd.grad((qh2, kh2), (qg, kg), (dyq.float(), dyk.float()))
    print(f"  fp32 grad_output accepted by autograd? dq dtype={dq32.dtype}")
except Exception as e:  # noqa: BLE001
    print(f"  fp32 grad_output -> {type(e).__name__}: {str(e)[:120]}")

print("== P7 repeat determinism forward (bitwise) ==")
a = qknorm_scaled_rope(q, k, cos, sin)
b = qknorm_scaled_rope(q, k, cos, sin)
print(f"  fwd repeat equal: {torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])}")

print("== P8 positions gather vs direct, int32, large positions (131071) ==")
big = torch.tensor([0, 131071, 65536, 4097], device=dev)
c1, s1 = rope_table(big, device=dev)
full_c, full_s = rope_table(torch.arange(131072, device=dev), device=dev)
print(f"  gather == direct at large positions: {torch.equal(c1, full_c[big]) and torch.equal(s1, full_s[big])}")
c2, s2 = rope_table(big.to(torch.int32), device=dev)
print(f"  int32 == int64: {torch.equal(c1, c2) and torch.equal(s1, s2)}")
