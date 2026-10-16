"""Probe 1: can the G1 tests detect an eps mix-up at their input magnitudes?
Probe 2: does the golden's `ss / H` on CUDA equal IEEE div_rn (it is a reciprocal multiply)?
Probe 3: golden CPU vs golden CUDA bitwise."""
import json
import torch
from rl_engine.models.muse_glimmer.ops import norms as N
from rl_engine.models.muse_glimmer.triton import norms as K

torch.backends.cuda.matmul.allow_tf32 = False
P = json.load(open("rl_engine/contracts/profiles/precision/ws1.json"))
TOL = P["accuracy"]["default"]["reduction"]["bfloat16"]
H = N.HIDDEN


def bf16(shape, seed, scale=1.0, shift=0.0, device="cuda"):
    g = torch.Generator(device=device).manual_seed(seed)
    return (torch.randn(*shape, device=device, generator=g) * scale + shift).to(torch.bfloat16)


def close(a, b):
    try:
        torch.testing.assert_close(a.float(), b.float(), atol=TOL["atol"], rtol=TOL["rtol"])
        return True
    except AssertionError:
        return False


# ---- Probe 1: eps sensitivity at the test's magnitudes (scale 2.0 norm, scale 1.0 fused branch)
x = bf16((4096, H), 10 + 4096, scale=2.0)
w = bf16((H,), 11, scale=0.1)
y5 = N.centered_rmsnorm(x, w, eps=1e-5)
y8 = N.centered_rmsnorm(x, w, eps=1e-8)
y0 = N.centered_rmsnorm(x, w, eps=0.0)
print("P1 scale=2.0: eps=1e-5 vs 1e-8 torch.equal:", torch.equal(y5, y8),
      "| within TOL:", close(y5, y8), "| eps=0 within TOL of eps=1e-5:", close(y5, y0))
# the fused test: b scale 1.0, r scale 3.0 ; swap eps_post<->eps_pre
b = bf16((4096, H), 20 + 4096)
r = bf16((4096, H), 21 + 4096, scale=3.0)
wp = bf16((H,), 22, scale=0.1)
wq = bf16((H,), 23, scale=0.1)
ok = N.fused_post_norm_residual_pre_norm(b, r, wp, 1e-8, wq, 1e-5)
sw = N.fused_post_norm_residual_pre_norm(b, r, wp, 1e-5, wq, 1e-8)
ze = N.fused_post_norm_residual_pre_norm(b, r, wp, 0.0, wq, 0.0)
print("P1 fused: eps swapped -> rnew equal:", torch.equal(ok[0], sw[0]), "pre equal:",
      torch.equal(ok[1], sw[1]), "| both within TOL:", close(ok[0], sw[0]) and close(ok[1], sw[1]),
      "| eps=0 within TOL:", close(ok[0], ze[0]) and close(ok[1], ze[1]))
# where eps actually matters: small-RMS rows
xs = bf16((64, H), 5, scale=1e-3)
print("P1 scale=1e-3: eps=1e-5 vs 1e-8 within TOL:",
      close(N.centered_rmsnorm(xs, w, eps=1e-5), N.centered_rmsnorm(xs, w, eps=1e-8)))

# ---- Probe 2: golden ms = ss / H on CUDA vs IEEE division
ss = (torch.rand(1_000_000, device="cuda") * 1e5).float()
ms_scalar = ss / float(H)                       # what inv_rms_fp32 does
ms_div = ss / torch.full_like(ss, float(H))     # IEEE div_rn
print("P2 CUDA: (ss / 6656.0) != div_rn(ss, 6656) in", (ms_scalar != ms_div).sum().item(),
      "of", ss.numel(), "elements")
ss_cpu = ss.cpu()
print("P2 CPU : (ss / 6656.0) != div_rn(ss, 6656) in",
      (ss_cpu / float(H) != ss_cpu / torch.full_like(ss_cpu, float(H))).sum().item())
# effect on inv: golden inv (CUDA) vs the same formula with true division, vs kernel inv
x32 = x.float()
inv_g = N.inv_rms_fp32(x32, 1e-5)
ss_g = N._sum_squares_fp32(x32)
inv_true = 1.0 / torch.sqrt(ss_g / torch.full_like(ss_g, float(H)) + 1e-5)
_, inv_k = K.norm_forward(x, w, eps=1e-5, form=K.FORM_CENTERED)
print("P2 inv: golden(CUDA) != golden(true div):", (inv_g != inv_true).sum().item(), "/ 4096 rows;",
      "kernel != golden(CUDA):", (inv_k != inv_g).sum().item(), "; kernel != golden(true div):",
      (inv_k != inv_true).sum().item())
# ---- Probe 3: golden CPU vs CUDA bitwise
yg_cuda = N.centered_rmsnorm_fp32(x, w, eps=1e-5)
yg_cpu = N.centered_rmsnorm_fp32(x.cpu(), w.cpu(), eps=1e-5)
print("P3 golden CPU vs CUDA bitwise equal:", torch.equal(yg_cuda.cpu(), yg_cpu),
      "| rows with any diff:", (yg_cuda.cpu() != yg_cpu).any(dim=1).sum().item())
inv_cpu = N.inv_rms_fp32(x32.cpu(), 1e-5)
print("P3 inv CPU vs CUDA differ in", (inv_cpu != inv_g.cpu()).sum().item(), "rows")
