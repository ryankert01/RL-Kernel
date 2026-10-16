"""Probe: do Triton libdevice.tanh / tl.exp / tl.div_rn agree bitwise with torch fp32?"""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


@triton.jit
def _probe(x_ptr, t_ptr, e_ptr, d_ptr, f_ptr, g_ptr, n, mult, cap, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m, other=0.0)
    tl.store(t_ptr + offs, libdevice.tanh(x), mask=m)
    tl.store(e_ptr + offs, tl.exp(x), mask=m)
    tl.store(d_ptr + offs, tl.div_rn(x * mult, cap), mask=m)
    tl.store(f_ptr + offs, (x * mult) / cap, mask=m)
    # full softcap chain as the kernel will do it
    u = tl.div_rn(x * mult, cap)
    tl.store(g_ptr + offs, cap * libdevice.tanh(u), mask=m)


def run(x, mult, cap):
    n = x.numel()
    outs = [torch.empty_like(x) for _ in range(5)]
    _probe[(triton.cdiv(n, 1024),)](x, *outs, n, mult, cap, BLOCK=1024)
    return outs


def ulp_diff(a, b):
    ai = a.view(torch.int32).to(torch.int64)
    bi = b.view(torch.int32).to(torch.int64)
    return (ai - bi).abs()


if __name__ == "__main__":
    dev = "cuda"
    g = torch.Generator(device=dev).manual_seed(0)
    mult, cap = 0.19611613513818404, 20.0
    xs = [
        torch.randn(1 << 24, device=dev, generator=g) * 3.0,  # typical u range
        torch.randn(1 << 24, device=dev, generator=g) * 60.0,  # typical raw logit range
        torch.rand(1 << 24, device=dev, generator=g) * 2e-3,  # near zero
        torch.linspace(-12, 12, 1 << 22, device=dev),
        torch.tensor([0.0, -0.0, 1e-30, 1e-8, 0.5, 1.0, 8.0, 9.01, 15.0, 88.0, -88.0, 1e4, -1e4],
                     device=dev),
    ]
    for i, x in enumerate(xs):
        t, e, d, f, gsc = run(x, mult, cap)
        tt = torch.tanh(x)
        te = torch.exp(x)
        td = x * mult / cap
        tg = cap * torch.tanh(td)
        print(f"-- sample {i} n={x.numel()} range=[{x.min().item():.3g},{x.max().item():.3g}]")
        for name, a, b in (("tanh", t, tt), ("exp", e, te), ("div_rn", d, td),
                           ("div_default", f, td), ("softcap_chain", gsc, tg)):
            fin = torch.isfinite(a) & torch.isfinite(b)
            u = ulp_diff(a[fin], b[fin])
            eq = torch.equal(a, b)
            print(f"   {name:14s} bitwise={eq} max_ulp={u.max().item() if u.numel() else 0} "
                  f"frac_neq={(u > 0).float().mean().item():.3e} max_abs={(a - b).abs()[fin].max().item():.3e}")
