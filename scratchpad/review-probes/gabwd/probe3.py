"""Replay test_autograd_function_binds_strict_backward with the sibling forward actually wired in."""
import torch

from rl_engine.models.muse_glimmer.ops import attention_golden_bwd as gold
from rl_engine.models.muse_glimmer.ops import gated_attention as fwd_op
from rl_engine.models.muse_glimmer.ops import gated_attention_autograd as ag

# What the import at gated_attention_autograd.py:38 would resolve to if it pointed at ops.gated_attention
ag._strict_fwd = fwd_op.strict_gated_attention_fwd_varlen
ag.FORWARD_IMPL = "patched:ops.gated_attention.strict_gated_attention_fwd_varlen"
print("forward_is_fallback after patch:", ag.trace()["forward_is_fallback"])


def make_inputs(lens, *, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    tq = sum(lens)
    q = torch.randn(tq, 32, 128, generator=g)
    q = q * torch.rsqrt(q.pow(2).mean(-1, keepdim=True) + 1e-5) * 3.87
    k = torch.randn(tq, 2, 128, generator=g)
    k = k * torch.rsqrt(k.pow(2).mean(-1, keepdim=True) + 1e-5)
    v = torch.randn(tq, 2, 128, generator=g)
    gate = torch.randn(tq, 4096, generator=g)
    dy = torch.randn(tq, 4096, generator=g)
    cu = torch.tensor([0] + torch.tensor(lens).cumsum(0).tolist(), dtype=torch.int32)
    pos = torch.cat([torch.arange(0, n, dtype=torch.int32) for n in lens])
    dev = torch.device("cuda")
    return dict(q=q.to(dev, torch.bfloat16), k=k.to(dev, torch.bfloat16), v=v.to(dev, torch.bfloat16),
                gate=gate.to(dev, torch.bfloat16), dy=dy.to(dev, torch.bfloat16),
                cu_q=cu.to(dev), cu_k=cu.to(dev), pos=pos.to(dev))


for window in (None, 2048):
    inp = make_inputs([300, 64], seed=9)
    q = inp["q"].clone().requires_grad_(True)
    k = inp["k"].clone().requires_grad_(True)
    v = inp["v"].clone().requires_grad_(True)
    gate = inp["gate"].clone().requires_grad_(True)
    y = ag.gated_attention(q, k, v, gate, inp["cu_q"], inp["cu_k"], inp["pos"], window)
    y.backward(inp["dy"])
    y_ref, o, lse = gold.golden_forward_fixed_order(inp["q"], inp["k"], inp["v"], inp["gate"],
                                                    inp["cu_q"], inp["cu_k"], inp["pos"], window=window)
    dq, dk, dv, dgate = ag.gated_attention_backward(inp["q"], inp["k"], inp["v"], inp["gate"], o, lse, inp["dy"],
                                                    inp["cu_q"], inp["cu_k"], inp["pos"], window=window)
    print(f"[window={window}] test asserts (test_gated_attention_bwd.py:261-262): "
          f"q.grad==dq {torch.equal(q.grad, dq)}  k.grad==dk {torch.equal(k.grad, dk)}  "
          f"v.grad==dv {torch.equal(v.grad, dv)}  gate.grad==dgate {torch.equal(gate.grad, dgate)}")
    # repeat determinism of the autograd path itself with the real forward
    q2 = inp["q"].clone().requires_grad_(True); k2 = inp["k"].clone().requires_grad_(True)
    v2 = inp["v"].clone().requires_grad_(True); g2 = inp["gate"].clone().requires_grad_(True)
    y2 = ag.gated_attention(q2, k2, v2, g2, inp["cu_q"], inp["cu_k"], inp["pos"], window)
    y2.backward(inp["dy"])
    print(f"   autograd path repeat: y {torch.equal(y, y2)} dq {torch.equal(q.grad, q2.grad)} dk {torch.equal(k.grad, k2.grad)} "
          f"dv {torch.equal(v.grad, v2.grad)} dgate {torch.equal(gate.grad, g2.grad)}")
