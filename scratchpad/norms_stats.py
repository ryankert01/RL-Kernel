import torch

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
from rl_engine.models.muse_glimmer.ops import norms as N  # noqa: E402

dev = "cuda"
g = torch.Generator(device=dev).manual_seed(123)
T, H, V = 4096, 6656, 202048
x = (torch.randn(T, H, device=dev, generator=g) * 2).to(torch.bfloat16)
r = (torch.randn(T, H, device=dev, generator=g) * 3).to(torch.bfloat16)
w = (torch.randn(H, device=dev, generator=g) * 0.1).to(torch.bfloat16)
w2 = (torch.randn(H, device=dev, generator=g) * 0.1).to(torch.bfloat16)
wf = (1 + torch.randn(H, device=dev, generator=g) * 0.1).to(torch.bfloat16)
table = torch.randn(V, H, device=dev, generator=g, dtype=torch.bfloat16) * 0.02
ids = torch.randint(0, V, (T,), device=dev, generator=g)


def rms64(x, w, eps, form):
    x64 = x.double()
    inv = torch.rsqrt((x64 * x64).mean(-1, keepdim=True) + eps)
    if form == "centered":
        return x64 * inv * (1.0 + w.double())
    if form == "weighted":
        return x64 * inv * w.double()
    return x64 * inv


def report(name, y, g32, g64):
    neq = (y != g32.to(torch.bfloat16)).float().mean().item()
    e32 = (y.float() - g32).abs().max().item()
    e64 = (y.double() - g64).abs().max().item()
    rel64 = ((y.double() - g64).abs() / (g64.abs() + 1e-6)).max().item()
    print(f"{name:40s} neq_vs_bf16(golden)={neq:.2e}  max|strict-fp32golden|={e32:.3e}  "
          f"max|strict-fp64|={e64:.3e}  max|golden32-fp64|={(g32.double()-g64).abs().max().item():.3e}")


report("centered_rmsnorm eps=1e-5", N.centered_rmsnorm(x, w, eps=1e-5), N.centered_rmsnorm_fp32(x, w, eps=1e-5), rms64(x, w, 1e-5, "centered"))
report("centered_rmsnorm eps=1e-8", N.centered_rmsnorm(x, w, eps=1e-8), N.centered_rmsnorm_fp32(x, w, eps=1e-8), rms64(x, w, 1e-8, "centered"))
report("weighted_rmsnorm", N.weighted_rmsnorm(x, wf, eps=1e-5), N.weighted_rmsnorm_fp32(x, wf, eps=1e-5), rms64(x, wf, 1e-5, "weighted"))
report("normalized_embedding V=202048", N.normalized_embedding(table, ids), N.normalized_embedding_fp32(table, ids), rms64(table[ids], None, 1e-5, "weightless"))
rn, pre = N.fused_post_norm_residual_pre_norm(x, r, w, 1e-8, w2, 1e-5)
rng, preg = N.fused_post_norm_residual_pre_norm_fp32(x, r, w, 1e-8, w2, 1e-5)
print("fused resid_new: equal to golden bf16 values:", torch.equal(rn.float(), rng), " neq frac:", (rn.float() != rng).float().mean().item())
report("fused pre_normed", pre, preg, rms64(rn, w2, 1e-5, "centered"))

# backward dw error vs fp64-ish reference (fp32 autograd golden)
xx, ww = x.clone().requires_grad_(True), w.clone().requires_grad_(True)
dy = torch.randn_like(x)
N.centered_rmsnorm(xx, ww, eps=1e-5).backward(dy)
x32, w32 = x.float().requires_grad_(True), w.float().requires_grad_(True)
N.centered_rmsnorm_fp32(x32, w32, eps=1e-5).backward(dy.float())
print("centered bwd T=4096: max|dx-ref|=%.3e (ref max %.2f)  max|dw-ref|=%.3e (ref max %.2f)" % (
    (xx.grad.float() - x32.grad).abs().max().item(), x32.grad.abs().max().item(),
    (ww.grad.float() - w32.grad).abs().max().item(), w32.grad.abs().max().item()))
# colsum fixed-order vs fp64 sum of the same rows
from rl_engine.models.muse_glimmer.triton.norms import colsum_fixed_order  # noqa: E402
rows = torch.randn(4096, H, device=dev, generator=g)
cs = colsum_fixed_order(rows)
print("colsum_fixed_order vs fp64 sum: max abs err %.3e (max |sum| %.1f); repeat equal: %s" % (
    (cs.double() - rows.double().sum(0)).abs().max().item(), rows.double().sum(0).abs().max().item(),
    torch.equal(cs, colsum_fixed_order(rows))))
