"""Independent reviewer probe for kernels_rope (qknorm_scaled_rope follow-ups).

Run with PYTHONPATH=<HEAD worktree>.  Loads the OLD triton kernel module (6a61f6e) from a file
path given as argv[1] and compares it with the HEAD kernel.
"""
import importlib.util
import json
import sys

import torch
import triton

from rl_engine.models.muse_glimmer.ops import qknorm_rope as op
from rl_engine.models.muse_glimmer.triton import qknorm_rope as new_k

spec = importlib.util.spec_from_file_location("old_qk_triton", sys.argv[1])
old_k = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old_k)

out = {}
dev = torch.device("cuda")

# (1) CPU vs CUDA table, built with the same formula (private builder, since rope_table now
# refuses CPU)
pos = torch.arange(131072)
cc, sc = op._rope_table_in(pos, theta=op.ROPE_THETA, dim=128, device=dev, dtype=torch.float32)
c_cpu, s_cpu = op._rope_table_in(pos, theta=op.ROPE_THETA, dim=128, device="cpu", dtype=torch.float32)
c_cpu, s_cpu = c_cpu.to(dev), s_cpu.to(dev)
out["table_cpu_vs_cuda_first4096"] = {
    "cos": int((cc[:4096] != c_cpu[:4096]).sum()),
    "sin": int((sc[:4096] != s_cpu[:4096]).sum()),
    "elements": 4096 * 128,
}
out["table_cpu_vs_cuda_all"] = {
    "cos": int((cc != c_cpu).sum()),
    "sin": int((sc != s_cpu).sum()),
    "max_abs": float(torch.maximum((cc - c_cpu).abs().max(), (sc - s_cpu).abs().max())),
}
# canonical sha
out["sha256_head"] = op.rope_table_sha256(dev)
out["sha256_pinned"] = op.CANONICAL_ROPE_TABLE_SHA256
# op flips from a CPU table at T=4096 positions 0..4095
g = torch.Generator(device=dev).manual_seed(1)
T = 4096
q = torch.randn(T, 32, 128, device=dev, generator=g).to(torch.bfloat16)
k = torch.randn(T, 2, 128, device=dev, generator=g).to(torch.bfloat16)
a = op.qknorm_scaled_rope(q, k, cc[:T].contiguous(), sc[:T].contiguous())
b = op.qknorm_scaled_rope(q, k, c_cpu[:T].contiguous(), s_cpu[:T].contiguous())
out["op_flips_cpu_table_T4096"] = {"q_hat": int((a[0] != b[0]).sum()), "k_hat": int((a[1] != b[1]).sum())}

# (2) new API behaviour
res = {}
try:
    op.rope_table(torch.arange(4), device="cpu")
    res["rope_table_cpu"] = "accepted"
except Exception as e:  # noqa: BLE001
    res["rope_table_cpu"] = type(e).__name__
try:
    op.qknorm_scaled_rope(q.float(), k.float(), None, None)
    res["fp32"] = "accepted"
except Exception as e:  # noqa: BLE001
    res["fp32"] = type(e).__name__
# legacy path: cos=None on a local layer still silently runs NoPE
res["legacy_local_none_runs"] = True
try:
    op.qknorm_scaled_rope(q, k, None, None)
except Exception:  # noqa: BLE001
    res["legacy_local_none_runs"] = False
# T=0 through the new API
try:
    z = op.qknorm_scaled_rope_at(q[:0], k[:0], torch.arange(0, device=dev), local=True)
    res["T0_at"] = list(z[0].shape)
except Exception as e:  # noqa: BLE001
    res["T0_at"] = f"{type(e).__name__}: {e}"
# positions at the table edge
try:
    op.rope_rows(torch.tensor([131071, 131072], device=dev))
    res["edge"] = "accepted"
except Exception as e:  # noqa: BLE001
    res["edge"] = type(e).__name__
out["api"] = res

# (3) old vs new kernel: bits + GPU time, local layers, interleaved
eps, scale = op.RMS_NORM_EPS, op.QK_SCALE
ab = []
for T in (1, 64, 4096, 8192, 32768, 131072):
    q = torch.randn(T, 32, 128, device=dev, generator=g).to(torch.bfloat16)
    k = torch.randn(T, 2, 128, device=dev, generator=g).to(torch.bfloat16)
    p = torch.randint(0, 131072, (T,), device=dev)
    c, s = cc[p].contiguous(), sc[p].contiguous()
    o_old = old_k.qknorm_rope_forward_triton(q, k, c, s, eps=eps, qk_scale=scale)
    o_new = new_k.qknorm_rope_forward_triton(q, k, c, s, eps=eps, qk_scale=scale)
    dy = torch.randn_like(q, dtype=torch.float32).to(torch.bfloat16)
    dyk = torch.randn_like(k, dtype=torch.float32).to(torch.bfloat16)
    b_old = old_k.qknorm_rope_backward_triton(q, k, dy, dyk, c, s, eps=eps, qk_scale=scale)
    b_new = new_k.qknorm_rope_backward_triton(q, k, dy, dyk, c, s, eps=eps, qk_scale=scale)
    eq = all(torch.equal(x, y) for x, y in zip(o_old + b_old, o_new + b_new))
    times = {"old_fwd": [], "new_fwd": [], "old_bwd": [], "new_bwd": []}
    for _ in range(5):
        times["old_fwd"].append(triton.testing.do_bench(lambda: old_k.qknorm_rope_forward_triton(q, k, c, s, eps=eps, qk_scale=scale), warmup=10, rep=50))
        times["new_fwd"].append(triton.testing.do_bench(lambda: new_k.qknorm_rope_forward_triton(q, k, c, s, eps=eps, qk_scale=scale), warmup=10, rep=50))
        times["old_bwd"].append(triton.testing.do_bench(lambda: old_k.qknorm_rope_backward_triton(q, k, dy, dyk, c, s, eps=eps, qk_scale=scale), warmup=10, rep=50))
        times["new_bwd"].append(triton.testing.do_bench(lambda: new_k.qknorm_rope_backward_triton(q, k, dy, dyk, c, s, eps=eps, qk_scale=scale), warmup=10, rep=50))
    med = {kk: round(sorted(v)[len(v) // 2] * 1e3, 1) for kk, v in times.items()}
    ab.append({"T": T, "bits_equal": eq, **med})
    print(ab[-1], flush=True)
out["ab_us"] = ab
print(json.dumps(out, indent=1))
json.dump(out, open(sys.argv[2], "w"), indent=1)
