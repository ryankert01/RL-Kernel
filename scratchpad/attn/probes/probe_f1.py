"""Probe findings gated_attention_bwd:1/2/3 (autograd forward impl, golden duplicate)."""
import json
import time

import torch

torch.backends.cuda.matmul.allow_tf32 = False
from rl_engine.models.muse_glimmer.ops import attention_golden_bwd as gold  # noqa: E402
from rl_engine.models.muse_glimmer.ops import gated_attention as ga  # noqa: E402
from rl_engine.models.muse_glimmer.ops import gated_attention_autograd as ag  # noqa: E402
from rl_engine.models.muse_glimmer.triton import gated_attention_bwd as bwd  # noqa: E402

out = {"forward_impl": ag.FORWARD_IMPL, "forward_is_fallback": ag.trace()["forward_is_fallback"]}
dev = "cuda"
g = torch.Generator(device="cpu").manual_seed(0)
L = 2100
q = (torch.randn(L, 32, 128, generator=g) * 3.87).to(dev, torch.bfloat16)
k = torch.randn(L, 2, 128, generator=g).to(dev, torch.bfloat16)
v = torch.randn(L, 2, 128, generator=g).to(dev, torch.bfloat16)
gate = torch.randn(L, 4096, generator=g).to(dev, torch.bfloat16)
dy = torch.randn(L, 4096, generator=g).to(dev, torch.bfloat16)
cu = torch.tensor([0, L], dtype=torch.int32, device=dev)
pos = torch.arange(L, dtype=torch.int32, device=dev)
for window in (None, 2048):
    rec = {}
    ag.gated_attention(q, k, v, gate, cu, cu, pos, window)
    torch.cuda.synchronize()
    t0 = time.time()
    y_ag = ag.gated_attention(q, k, v, gate, cu, cu, pos, window)
    torch.cuda.synchronize()
    rec["autograd_forward_ms"] = (time.time() - t0) * 1e3
    y_k, o_k, lse_k = ga.strict_gated_attention_fwd_varlen(
        q, k, v, gate, cu, cu, pos, window=window
    )
    torch.cuda.synchronize()
    t0 = time.time()
    ga.strict_gated_attention_fwd_varlen(q, k, v, gate, cu, cu, pos, window=window)
    torch.cuda.synchronize()
    rec["strict_kernel_forward_ms"] = (time.time() - t0) * 1e3
    y_g, o_g, lse_g = gold.golden_forward_fixed_order(q, k, v, gate, cu, cu, pos, window=window)
    rec["autograd_y_vs_kernel_y_mismatch"] = int((y_ag != y_k).sum())
    rec["golden_lse_vs_kernel_lse_mismatch"] = int((lse_g != lse_k).sum())
    rec["golden_o_vs_kernel_o_mismatch"] = int((o_g != o_k).sum())
    rec["max_abs_lse_diff"] = float((lse_g - lse_k).abs().max())
    gk = bwd.strict_gated_attention_bwd(q, k, v, gate, o_k, lse_k, dy, cu, cu, pos, window=window)
    gg = bwd.strict_gated_attention_bwd(q, k, v, gate, o_g, lse_g, dy, cu, cu, pos, window=window)
    for n, a, b in zip(("dq", "dk", "dv", "dgate"), gk, gg):
        rec[f"{n}_kernel_olse_vs_golden_olse_mismatch"] = int((a != b).sum())
        rec[f"{n}_max_abs_diff"] = float((a.float() - b.float()).abs().max())
    y32, o32, l32 = ga.gated_attention_forward_fp32(q, k, v, gate, pos, window)
    rec["bwd_golden_vs_fwd_golden_lse_mismatch"] = int((l32 != lse_g).sum())
    rec["bwd_golden_vs_fwd_golden_max_lse_diff"] = float((l32 - lse_g).abs().max())
    rec["bwd_golden_o_vs_bf16(fwd_golden_o)_mismatch"] = int((o32.to(torch.bfloat16) != o_g).sum())
    out[f"window={window}"] = rec
print(json.dumps(out, indent=1))
