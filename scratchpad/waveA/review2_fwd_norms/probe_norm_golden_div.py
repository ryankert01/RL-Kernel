"""Is the norms golden's `ss / float(H)` the same IEEE op as the kernel's div_rn(ss, H)? (review2)"""
import torch
from rl_engine.models.muse_glimmer.ops import norms as N
from rl_engine.models.muse_glimmer.triton import norms as K

DEV = torch.device("cuda")
H = N.HIDDEN
g = torch.Generator(device=DEV).manual_seed(3)
x = (torch.randn(65536, H, device=DEV, generator=g) * 2.0).bfloat16()
x32 = x.float()
ss = N._sum_squares_fp32(x32)  # golden ss
ms_golden = ss / float(H)  # what inv_rms_fp32 does (CPU-scalar divisor on CUDA)
ms_true = torch.div(ss, torch.full_like(ss, float(H)))  # tensor/tensor true division (div.rn)
ms_cpu = (ss.cpu() / float(H)).to(DEV)
print("ms: golden(CUDA, scalar) vs true division mismatches:",
      int((ms_golden != ms_true).sum()), "of", ss.numel())
print("ms: golden CUDA vs golden CPU mismatches:", int((ms_golden != ms_cpu).sum()))
inv_g = 1.0 / torch.sqrt(ms_golden + 1e-5)
inv_t = 1.0 / torch.sqrt(ms_true + 1e-5)
print("inv mismatches caused by the division form alone:", int((inv_g != inv_t).sum()))
# kernel inv vs golden inv (differences = ss tree + division form)
y, inv_k = K.norm_forward(x, None, eps=1e-5, form=K.FORM_WEIGHTLESS)
print("kernel inv vs golden inv mismatches:", int((inv_k != inv_g).sum()),
      " vs true-div golden:", int((inv_k != inv_t).sum()))
# How do output bits move? weightless norm -> bf16
yg = (x32 * inv_g[:, None]).bfloat16()
yt = (x32 * inv_t[:, None]).bfloat16()
print("bf16 output elements differing between golden-division and true-division:",
      int((yg != yt).sum()), "of", yg.numel())
