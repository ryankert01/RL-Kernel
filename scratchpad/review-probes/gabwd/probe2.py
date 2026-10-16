import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from rl_engine.models.muse_glimmer.ops import attention_golden_bwd as gold
from rl_engine.models.muse_glimmer.ops import gated_attention as fwd_op


@triton.jit
def _sig_kernel(g_ptr, a_ptr, b_ptr, N: tl.constexpr):
    offs = tl.program_id(0) * N + tl.arange(0, N)
    g = tl.load(g_ptr + offs).to(tl.float32)
    a = 1.0 / (1.0 + tl.exp(-g))  # bwd preprocess, gated_attention_bwd.py:105
    one = tl.full([N], 1.0, dtype=tl.float32)
    b = libdevice.div_rn(one, one + tl.exp(-g))  # fwd kernel, gated_attention_fwd.py:~214
    tl.store(a_ptr + offs, a)
    tl.store(b_ptr + offs, b)


g = (torch.randn(1 << 22, device="cuda") * 3).to(torch.bfloat16)
a = torch.empty(g.shape, dtype=torch.float32, device="cuda")
b = torch.empty_like(a)
_sig_kernel[(g.numel() // 1024,)](g, a, b, N=1024, num_warps=4)
nd = (a != b).sum().item()
print(f"sigmoid: plain '/' vs div_rn differ on {nd} / {g.numel()} bf16 inputs; max|d|={(a-b).abs().max().item():.3e}")
# does it change the bf16 'do' cast or dgate?
dy = torch.randn(g.shape, device="cuda").to(torch.bfloat16).float()
print("  do=dy*sg -> bf16 differs:", ((dy * a).to(torch.bfloat16) != (dy * b).to(torch.bfloat16)).sum().item())

# Two fp32 goldens for the same op: backward author's vs forward author's
import inspect
print("fwd golden sig:", inspect.signature(fwd_op.gated_attention_forward_fp32))
torch.manual_seed(0)
for window in (None, 2048):
    n = 2100
    gen = torch.Generator(device="cpu").manual_seed(5)
    q = torch.randn(n, 32, 128, generator=gen)
    q = (q * torch.rsqrt(q.pow(2).mean(-1, keepdim=True) + 1e-5) * 3.87).to("cuda", torch.bfloat16)
    k = torch.randn(n, 2, 128, generator=gen)
    k = (k * torch.rsqrt(k.pow(2).mean(-1, keepdim=True) + 1e-5)).to("cuda", torch.bfloat16)
    v = torch.randn(n, 2, 128, generator=gen).to("cuda", torch.bfloat16)
    gate = torch.randn(n, 4096, generator=gen).to("cuda", torch.bfloat16)
    cu = torch.tensor([0, n], dtype=torch.int32, device="cuda")
    pos = torch.arange(n, dtype=torch.int32, device="cuda")
    y1, o1, l1 = gold.golden_forward_fixed_order(q, k, v, gate, cu, cu, pos, window=window)
    out = fwd_op.gated_attention_forward_fp32(q, k, v, gate, pos, window)
    print("fwd golden returns:", type(out), [(tuple(t.shape), t.dtype) for t in out] if isinstance(out, tuple) else out.shape)
    y2, o2, l2 = out[:3]
    print(f"[window={window}] bwd-golden vs fwd-golden: y eq {torch.equal(y1, y2.to(y1.dtype))} "
          f"o eq {torch.equal(o1, o2.to(o1.dtype))} lse eq {torch.equal(l1, l2.to(l1.dtype))} "
          f"max|o d|={(o1.float()-o2.float()).abs().max().item():.3e} max|lse d|={(l1.float()-l2.float()).abs().max().item():.3e}")
    ys, os_, ls = fwd_op.strict_gated_attention_fwd_varlen(q, k, v, gate, cu, cu, pos, window=window)
    print(f"   strict fwd vs fwd-golden: o eq {torch.equal(os_, o2.to(os_.dtype))} lse eq {torch.equal(ls, l2.to(ls.dtype))}")
