import sys
import time

import torch

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

from rl_engine.models.muse_glimmer.ops import attention_golden_bwd as gold
from rl_engine.models.muse_glimmer.triton import gated_attention_bwd as bwd


def make(lens, seed=0, device="cuda"):
    g = torch.Generator(device="cpu").manual_seed(seed)
    T = sum(lens)
    q = torch.randn(T, 32, 128, generator=g)
    q = q * torch.rsqrt(q.pow(2).mean(-1, keepdim=True) + 1e-5) * 3.87
    k = torch.randn(T, 2, 128, generator=g)
    k = k * torch.rsqrt(k.pow(2).mean(-1, keepdim=True) + 1e-5)
    v = torch.randn(T, 2, 128, generator=g)
    gate = torch.randn(T, 4096, generator=g)
    dy = torch.randn(T, 4096, generator=g)
    cu = torch.tensor([0] + list(torch.tensor(lens).cumsum(0).tolist()), dtype=torch.int32)
    pos = torch.cat([torch.arange(L, dtype=torch.int32) for L in lens])
    to = lambda t: t.to(device)
    return (
        to(q).bfloat16(), to(k).bfloat16(), to(v).bfloat16(), to(gate).bfloat16(),
        to(dy).bfloat16(), to(cu), to(cu).clone(), to(pos),
    )


def run(lens, window):
    q, k, v, gate, dy, cu_q, cu_k, pos = make(lens)
    t0 = time.time()
    y, o, lse = gold.golden_forward_fixed_order(q, k, v, gate, cu_q, cu_k, pos, window=window)
    torch.cuda.synchronize()
    t1 = time.time()
    yref = gold.reference_forward(q.double(), k.double(), v.double(), gate.double(), cu_q, cu_k, pos, window=window)
    print(f"lens={lens} window={window} golden fwd {t1-t0:.2f}s  y vs fp64 max abs {(y.double()-yref).abs().max().item():.3e}")
    dq, dk, dv, dgate = bwd.strict_gated_attention_bwd(q, k, v, gate, o, lse, dy, cu_q, cu_k, pos, window=window)
    torch.cuda.synchronize()
    t2 = time.time()
    dq2, dk2, dv2, dgate2 = bwd.strict_gated_attention_bwd(q, k, v, gate, o, lse, dy, cu_q, cu_k, pos, window=window)
    print("  repeat equal:", torch.equal(dq, dq2), torch.equal(dk, dk2), torch.equal(dv, dv2), torch.equal(dgate, dgate2))
    rq, rk, rv, rg = gold.reference_backward(q, k, v, gate, cu_q, cu_k, pos, dy, window=window)
    for name, a, r in (("dq", dq, rq), ("dk", dk, rk), ("dv", dv, rv), ("dgate", dgate, rg)):
        a = a.float(); r = r.float()
        err = (a - r).abs()
        ok = torch.allclose(a, r, atol=5e-2, rtol=2e-2)
        print(f"  {name}: max|ref|={r.abs().max().item():.3e} max abs err={err.abs().max().item():.3e} "
              f"rel(max err/max ref)={err.max().item()/max(r.abs().max().item(),1e-30):.3e} allclose={ok} nan={torch.isnan(a).any().item()}")
    print(f"  bwd time (first call incl. compile) {t2-t1:.2f}s")


if __name__ == "__main__":
    run([64], None)
    run([64], 2048)
    run([200, 130], None)
    run([200, 130], 2048)
    if len(sys.argv) > 1 and sys.argv[1] == "big":
        run([2049], None)
        run([2049], 2048)
        run([4096], None)
        run([4096], 2048)
