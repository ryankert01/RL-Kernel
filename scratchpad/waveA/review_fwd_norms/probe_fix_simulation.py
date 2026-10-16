# Simulates the one-line import fix of gated_attention_autograd.py (copy in fixsim/, not the repo)
# and checks: forward is the strict kernel (bitwise), backward runs and is repeat-deterministic.
import json, sys
import torch
sys.path.insert(0, sys.argv[1])
import ga_autograd_fixed as A
from rl_engine.models.muse_glimmer.ops import gated_attention as ga
DEV = torch.device("cuda")
g = torch.Generator(device=DEV).manual_seed(0)
n = 300
q = (torch.randn(n, 32, 128, device=DEV, generator=g) * 3.87).to(torch.bfloat16)
k = torch.randn(n, 2, 128, device=DEV, generator=g).to(torch.bfloat16)
v = torch.randn(n, 2, 128, device=DEV, generator=g).to(torch.bfloat16)
gate = (torch.randn(n, 4096, device=DEV, generator=g) * 2).to(torch.bfloat16)
dy = torch.randn(n, 4096, device=DEV, generator=g).to(torch.bfloat16)
cu = torch.tensor([0, n], dtype=torch.int32, device=DEV)
pos = torch.arange(n, dtype=torch.int32, device=DEV)
out = {"forward_is_fallback": A.trace()["forward_is_fallback"], "FORWARD_IMPL": A.FORWARD_IMPL}
for window in (None, 2048):
    ref = ga.strict_gated_attention_fwd_varlen(q, k, v, gate, cu, cu, pos, window=window)[0]
    grads = []
    for _ in range(2):
        tq, tk, tv, tg = (t.clone().requires_grad_(True) for t in (q, k, v, gate))
        y = A.gated_attention(tq, tk, tv, tg, cu, cu, pos, window)
        y.backward(dy)
        grads.append((y.detach(), tq.grad, tk.grad, tv.grad, tg.grad))
    out[str(window)] = {
        "y_equal_strict_kernel": bool(torch.equal(grads[0][0], ref)),
        "grads_repeat_equal": all(torch.equal(a, b) for a, b in zip(grads[0][1:], grads[1][1:])),
        "grads_finite": all(bool(torch.isfinite(t.float()).all()) for t in grads[0][1:]),
    }
print(json.dumps(out))
