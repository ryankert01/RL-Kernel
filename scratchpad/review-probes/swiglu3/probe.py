import warnings, torch
from rl_engine.models.muse_glimmer.ops.swiglu_ffn import TritonDetGemmAdapter, default_gemm, swiglu_ffn
torch.manual_seed(0)
T,H,F = 72, 64, 96
x = torch.randn(T,H,device="cuda",dtype=torch.bfloat16,requires_grad=True)
wg = torch.randn(F,H,device="cuda",dtype=torch.bfloat16,requires_grad=True)
wu = torch.randn(F,H,device="cuda",dtype=torch.bfloat16,requires_grad=True)
wd = torch.randn(H,F,device="cuda",dtype=torch.bfloat16,requires_grad=True)
for name, gemm in [("adapter", TritonDetGemmAdapter()), ("default", default_gemm())]:
    print(name, type(gemm).__name__)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            y = swiglu_ffn(x,wg,wu,wd,gemm=gemm); y.float().sum().backward()
            print("  backward OK, warnings:", [str(i.message) for i in w])
        except Exception as e:
            print("  RAISED:", type(e).__name__, str(e)[:160])
    x.grad=wg.grad=wu.grad=wd.grad=None
# adapter single-chunk vs 8-chunk at T=72 vs T=64 — bits differ?
ad = TritonDetGemmAdapter()
a = torch.randn(96,72,device="cuda",dtype=torch.bfloat16); w = torch.randn(64,72,device="cuda",dtype=torch.bfloat16)
y1 = ad(a,w,out_dtype=torch.float32)
# emulate the 8-chunk tree on K=72 (9 per chunk) manually to see whether the single chunk differs
p=[ad._partial(a,w,i*9,(i+1)*9) for i in range(8)]
y8=((p[0]+p[1])+(p[2]+p[3]))+((p[4]+p[5])+(p[6]+p[7]))
print("single-chunk vs hypothetical 8-chunk tree bitwise equal:", torch.equal(y1,y8), "max abs diff", (y1-y8).abs().max().item())
