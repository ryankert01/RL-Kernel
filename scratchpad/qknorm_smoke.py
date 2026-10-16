import sys
import time

sys.path.insert(0, "/home/ubuntu/RL-Kernel/.claude/worktrees/agent-a1476a61a3b149700")
import torch  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

from rl_engine.models.muse_glimmer.ops import qknorm_rope as m  # noqa: E402

dev = torch.device("cuda")
T = int(sys.argv[1]) if len(sys.argv) > 1 else 4096
g = torch.Generator(device=dev).manual_seed(0)
q = (torch.randn(T, 32, 128, device=dev, generator=g) * 2.0).bfloat16()
k = (torch.randn(T, 2, 128, device=dev, generator=g) * 2.0).bfloat16()
pos = torch.arange(T, device=dev)
cos, sin = m.rope_table(pos, device=dev)
cos64, sin64 = m.rope_table_fp64(pos, device=dev)
print("table fp32 vs fp64 max abs:", (cos.double() - cos64).abs().max().item(),
      (sin.double() - sin64).abs().max().item())

t0 = time.time()
oq, ok = m.qknorm_scaled_rope(q, k, cos, sin)
torch.cuda.synchronize()
print("first launch (compile) s:", time.time() - t0, oq.dtype, oq.shape, ok.shape)
gq, gk = m.qknorm_scaled_rope_fp32(q, k, cos, sin)
gq64, gk64 = m.qknorm_scaled_rope_fp64(q, k, cos64, sin64)
gq64s, gk64s = m.qknorm_scaled_rope_fp64(q, k, cos, sin)


def rep(name, a, b):
    a = a.double()
    b = b.double()
    d = (a - b).abs()
    rel = d / b.abs().clamp_min(1e-6)
    print(f"{name}: max_abs={d.max().item():.3e} max_rel={rel.max().item():.3e} "
          f"frac_neq={(d > 0).double().mean().item():.3e}")


rep("kernel q vs golden fp32->bf16", oq, gq.bfloat16())
rep("kernel k vs golden fp32->bf16", ok, gk.bfloat16())
rep("kernel q vs golden fp32", oq, gq)
rep("kernel k vs golden fp32", ok, gk)
rep("kernel q vs golden fp64 (fp64 table)", oq, gq64)
rep("kernel q vs golden fp64 (fp32 table)", oq, gq64s)
rep("golden fp32 vs fp64 (fp64 table) q", gq, gq64)
rep("golden fp32 vs fp64 (fp32 table) q", gq, gq64s)
rep("golden fp32 vs fp64 (fp32 table) k", gk, gk64s)

# global path
oq_g, ok_g = m.qknorm_scaled_rope(q, k, None, None)
oq_i, ok_i = m.qknorm_scaled_rope(q, k, torch.ones_like(cos), torch.zeros_like(sin))
print("global == identity-table:", torch.equal(oq_g, oq_i), torch.equal(ok_g, ok_i))
gq_g, gk_g = m.qknorm_scaled_rope_fp32(q, k, None, None)
rep("global kernel q vs golden->bf16", oq_g, gq_g.bfloat16())

# batch invariance
ok_all = True
for t in (0, 1, 777, T - 1):
    sq, sk = m.qknorm_scaled_rope(q[t : t + 1], k[t : t + 1], cos[t : t + 1], sin[t : t + 1])
    ok_all &= torch.equal(sq[0], oq[t]) and torch.equal(sk[0], ok[t])
print("batch invariance T=1 vs T:", ok_all)

# backward
qg = q.clone().requires_grad_(True)
kg = q.new_empty(0)
kg = k.clone().requires_grad_(True)
yq, yk = m.qknorm_scaled_rope(qg, kg, cos, sin)
dyq = torch.randn_like(yq)
dyk = torch.randn_like(yk)
dq1, dk1 = torch.autograd.grad((yq, yk), (qg, kg), (dyq, dyk))
yq, yk = m.qknorm_scaled_rope(qg, kg, cos, sin)
dq2, dk2 = torch.autograd.grad((yq, yk), (qg, kg), (dyq, dyk))
print("backward repeat equal:", torch.equal(dq1, dq2), torch.equal(dk1, dk2))
qf = q.float().requires_grad_(True)
kf = k.float().requires_grad_(True)
yq, yk = m.qknorm_scaled_rope_fp32(qf, kf, cos, sin)
dqf, dkf = torch.autograd.grad((yq, yk), (qf, kf), (dyq.float(), dyk.float()))
rep("dq kernel vs fp32 autograd", dq1, dqf)
rep("dk kernel vs fp32 autograd", dk1, dkf)
rep("dq kernel vs fp32 autograd->bf16", dq1, dqf.bfloat16())
hdq, hdk = m.qknorm_scaled_rope_backward_fp32(q, k, dyq, dyk, cos, sin)
rep("dq kernel vs handwritten fp32 vjp->bf16", dq1, hdq.bfloat16())
rep("dk kernel vs handwritten fp32 vjp->bf16", dk1, hdk.bfloat16())
rep("handwritten vjp vs autograd (fp32) dq", hdq, dqf)
rep("handwritten vjp vs autograd (fp32) dk", hdk, dkf)
print("dq allclose(0.02,0.016):", torch.allclose(dq1.float(), dqf, atol=0.02, rtol=0.016),
      torch.allclose(dk1.float(), dkf, atol=0.02, rtol=0.016))
# backward batch invariance
t = 777
sq = q[t : t + 1].clone().requires_grad_(True)
sk = k[t : t + 1].clone().requires_grad_(True)
yq, yk = m.qknorm_scaled_rope(sq, sk, cos[t : t + 1], sin[t : t + 1])
sdq, sdk = torch.autograd.grad((yq, yk), (sq, sk), (dyq[t : t + 1], dyk[t : t + 1]))
print("backward batch invariance:", torch.equal(sdq[0], dq1[t]), torch.equal(sdk[0], dk1[t]))

# timing
def timeit(fn, n=50):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(n):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / n


print("fwd ms:", timeit(lambda: m.qknorm_scaled_rope(q, k, cos, sin)))
