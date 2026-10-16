"""Probe: fused backward batch invariance (not covered by tests) and the HAS_DRNEW/HAS_DPRE=False
launcher paths (never reached through autograd, which materializes zero grads)."""
import torch
from rl_engine.models.muse_glimmer.ops import norms as N
from rl_engine.models.muse_glimmer.triton import norms as K

H = N.HIDDEN


def bf16(shape, seed, scale=1.0, shift=0.0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return (torch.randn(*shape, device="cuda", generator=g) * scale + shift).to(torch.bfloat16)


T = 4096
b, r = bf16((T, H), 1), bf16((T, H), 2, scale=3.0)
wp, wq = bf16((H,), 3, scale=0.1), bf16((H,), 4, scale=0.1)
drn, dpre = bf16((T, H), 5), bf16((T, H), 6)


def grads(bb, rr, d1, d2):
    bb, rr = bb.clone().requires_grad_(True), rr.clone().requires_grad_(True)
    rnew, pre = N.fused_post_norm_residual_pre_norm(bb, rr, wp, 1e-8, wq, 1e-5)
    torch.autograd.backward([rnew, pre], [d1, d2])
    return bb.grad, rr.grad


db, dr = grads(b, r, drn, dpre)
for p in (0, 777, 4095):
    db1, dr1 = grads(b[p:p + 1], r[p:p + 1], drn[p:p + 1], dpre[p:p + 1])
    print(f"fused bwd row {p}: batch-1 vs batch-4096 d_branch equal={torch.equal(db1[0], db[p])} "
          f"d_resid equal={torch.equal(dr1[0], dr[p])}")

# None path vs materialized-zeros path of the launcher
rnew, y, ip, iq = K.fused_add_norm_forward(b, r, wp, 1e-8, wq, 1e-5, pre_form=K.FORM_CENTERED)
zeros = torch.zeros_like(drn)
a = K.fused_add_norm_backward(None, dpre, b, rnew, wp, wq, ip, iq, pre_form=K.FORM_CENTERED)
z = K.fused_add_norm_backward(zeros, dpre, b, rnew, wp, wq, ip, iq, pre_form=K.FORM_CENTERED)
print("d_rnew None vs zeros: d_branch", torch.equal(a[0], z[0]), "d_resid", torch.equal(a[1], z[1]),
      "pdw_post", torch.equal(a[2], z[2]), "pdw_pre", torch.equal(a[3], z[3]))
a = K.fused_add_norm_backward(drn, None, b, rnew, wp, wq, ip, iq, pre_form=K.FORM_CENTERED)
z = K.fused_add_norm_backward(drn, zeros, b, rnew, wp, wq, ip, iq, pre_form=K.FORM_CENTERED)
print("d_pre None vs zeros: d_branch", torch.equal(a[0], z[0]), "d_resid", torch.equal(a[1], z[1]),
      "pdw_post", torch.equal(a[2], z[2]), "pdw_pre", torch.equal(a[3], z[3]),
      "| signbit diffs d_resid:", (torch.signbit(a[1]) != torch.signbit(z[1])).sum().item())
# does autograd ever hand None to the Function? (materialize_grads default)
seen = {}
class Spy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.clone(), x.clone()
    @staticmethod
    def backward(ctx, g1, g2):
        seen["g1_is_none"] = g1 is None
        return g1 if g1 is not None else g2
x = torch.randn(4, device="cuda", requires_grad=True)
o1, o2 = Spy.apply(x)
o2.sum().backward()
print("autograd passes None for unused output:", seen["g1_is_none"])
