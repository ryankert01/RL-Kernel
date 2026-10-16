import torch

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
from rl_engine.models.muse_glimmer.ops import norms as N  # noqa: E402

dev = "cuda"
g = torch.Generator(device=dev).manual_seed(0)
T, H = 16, 6656
x = (torch.randn(T, H, device=dev, generator=g) * 2).to(torch.bfloat16)
w = (torch.randn(H, device=dev, generator=g) * 0.1).to(torch.bfloat16)
wf = (1 + torch.randn(H, device=dev, generator=g) * 0.1).to(torch.bfloat16)

y = N.centered_rmsnorm(x, w, eps=1e-5)
yg = N.centered_rmsnorm_fp32(x, w, eps=1e-5)
print("centered maxdiff", (y.float() - yg).abs().max().item(), "neq", (y != yg.to(torch.bfloat16)).sum().item())
y2 = N.weighted_rmsnorm(x, wf, eps=1e-5)
y2g = N.weighted_rmsnorm_fp32(x, wf, eps=1e-5)
print("weighted maxdiff", (y2.float() - y2g).abs().max().item(), "neq", (y2 != y2g.to(torch.bfloat16)).sum().item())

V = 1000
E = (torch.randn(V, H, device=dev, generator=g) * 0.02).to(torch.bfloat16)
ids = torch.randint(0, V, (T,), device=dev, generator=g)
e = N.normalized_embedding(E, ids, eps=1e-5)
eg = N.normalized_embedding_fp32(E, ids, eps=1e-5)
print("embed maxdiff", (e.float() - eg).abs().max().item(), "neq", (e != eg.to(torch.bfloat16)).sum().item())

b = (torch.randn(T, H, device=dev, generator=g)).to(torch.bfloat16)
r = (torch.randn(T, H, device=dev, generator=g) * 3).to(torch.bfloat16)
wp = (torch.randn(H, device=dev, generator=g) * 0.1).to(torch.bfloat16)
for form in ("centered", "weighted"):
    wpre = w if form == "centered" else wf
    rn, pre = N.fused_post_norm_residual_pre_norm(b, r, wp, 1e-8, wpre, 1e-5, pre_form=form)
    post = N.centered_rmsnorm(b, wp, eps=1e-8)
    rn2 = N.residual_add(r, post)
    pre2 = N.centered_rmsnorm(rn2, wpre, eps=1e-5) if form == "centered" else N.weighted_rmsnorm(rn2, wpre, eps=1e-5)
    print(form, "fused==composition:", torch.equal(rn, rn2), torch.equal(pre, pre2))
    rng, preg = N.fused_post_norm_residual_pre_norm_fp32(b, r, wp, 1e-8, wpre, 1e-5, pre_form=form)
    print(form, "vs golden:", (rn.float() - rng).abs().max().item(), (pre.float() - preg).abs().max().item())

# backward smoke
xb = x.clone().requires_grad_(True)
wb = w.clone().requires_grad_(True)
yb = N.centered_rmsnorm(xb, wb, eps=1e-5)
dy = torch.randn_like(yb)
yb.backward(dy)
xr = x.float().requires_grad_(True)
wr = w.float().requires_grad_(True)
yr = N.centered_rmsnorm_fp32(xr, wr, eps=1e-5)
yr.backward(dy.float())
print("dx maxdiff", (xb.grad.float() - xr.grad).abs().max().item(), "dw maxdiff", (wb.grad.float() - wr.grad).abs().max().item(), "dw max", wr.grad.abs().max().item())

Eb = E.clone().requires_grad_(True)
eb = N.normalized_embedding(Eb, ids, eps=1e-5)
de = torch.randn_like(eb)
eb.backward(de)
Er = E.float().requires_grad_(True)
er = N.normalized_embedding_fp32(Er, ids, eps=1e-5)
er.backward(de.float())
print("dE maxdiff", (Eb.grad.float() - Er.grad).abs().max().item(), "dE max", Er.grad.abs().max().item(), "nz rows", (Eb.grad.abs().sum(-1) > 0).sum().item(), ids.unique().numel())

bb, rb = b.clone().requires_grad_(True), r.clone().requires_grad_(True)
wpb, wqb = wp.clone().requires_grad_(True), w.clone().requires_grad_(True)
rn, pre = N.fused_post_norm_residual_pre_norm(bb, rb, wpb, 1e-8, wqb, 1e-5)
d1, d2 = torch.randn_like(rn), torch.randn_like(pre)
torch.autograd.backward([rn, pre], [d1, d2])
br, rr = b.float().requires_grad_(True), r.float().requires_grad_(True)
wpr, wqr = wp.float().requires_grad_(True), w.float().requires_grad_(True)
rng, preg = N.fused_post_norm_residual_pre_norm_fp32(br, rr, wpr, 1e-8, wqr, 1e-5)
torch.autograd.backward([rng, preg], [d1.float(), d2.float()])
for name, a, bq in (("db", bb.grad, br.grad), ("dr", rb.grad, rr.grad), ("dwpost", wpb.grad, wpr.grad), ("dwpre", wqb.grad, wqr.grad)):
    print(name, "maxdiff", (a.float() - bq).abs().max().item(), "ref max", bq.abs().max().item())
print(N.trace("centered_rmsnorm")["tile_config"])
