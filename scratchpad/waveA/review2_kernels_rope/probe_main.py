"""Reviewer probe for kernels_rope (e891ecc). Read-only; prints JSON lines."""
import hashlib
import json
import sys

import torch

from rl_engine.models.muse_glimmer.ops import qknorm_rope as op
from rl_engine.models.muse_glimmer.reference import golden_ops
from rl_engine.models.muse_glimmer.runtime import fallback_ops, strict_kernels

DEV = torch.device("cuda")
out = {"torch": torch.__version__, "cuda": torch.version.cuda}


def sha(c, s):
    h = hashlib.sha256()
    for t in (c, s):
        h.update(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


N = op.ROPE_TABLE_POSITIONS
pos = torch.arange(N, device=DEV)
cc, sc = op.canonical_rope_table(DEV)
out["canonical_sha_fresh"] = sha(cc, sc)
out["pinned"] = op.CANONICAL_ROPE_TABLE_SHA256
builders = {
    "golden_ops.rope_table": lambda d: golden_ops.rope_table(pos.to(d), theta=5e5, dim=128, device=d),
    "fallback_ops.rope_table": lambda d: fallback_ops.rope_table(pos.to(d), theta=5e5, dim=128, device=d),
    "strict_kernels._fb_rope_table": lambda d: strict_kernels._fb_rope_table(pos.to(d), theta=5e5, dim=128, device=torch.device(d)),
}
res = {}
for name, fn in builders.items():
    c, s = fn(DEV)
    res[name] = {
        "cuda_cos_mismatch": int((c.view(torch.int32) != cc.view(torch.int32)).sum()),
        "cuda_sin_mismatch": int((s.view(torch.int32) != sc.view(torch.int32)).sum()),
    }
    # do the other builders fail closed on CPU?
    try:
        c2, s2 = fn("cpu")
        res[name]["cpu_accepted"] = True
        res[name]["cpu_cos_mismatch_vs_cuda"] = int((c2.view(torch.int32) != cc.cpu().view(torch.int32)).sum())
    except Exception as e:  # noqa
        res[name]["cpu_accepted"] = False
        res[name]["cpu_error"] = f"{type(e).__name__}: {e}"[:200]
out["builders"] = res

# CPU vs CUDA in first 4096 positions (headline 97,294 cos / 99,316 sin)
p4 = torch.arange(4096)
c_cpu, s_cpu = op._rope_table_in(p4, theta=5e5, dim=128, device="cpu", dtype=torch.float32)
out["cpu_vs_cuda_first4096"] = {
    "cos": int((c_cpu.view(torch.int32) != cc[:4096].cpu().view(torch.int32)).sum()),
    "sin": int((s_cpu.view(torch.int32) != sc[:4096].cpu().view(torch.int32)).sum()),
}
# The strict op still accepts a CPU-built (then moved) table -- kernel cannot tell provenance
g = torch.Generator(device=DEV).manual_seed(0)
q = (torch.randn(4096, 32, 128, device=DEV, generator=g) * 2).to(torch.bfloat16)
k = (torch.randn(4096, 2, 128, device=DEV, generator=g) * 2).to(torch.bfloat16)
a = op.qknorm_scaled_rope(q, k, cc[:4096], sc[:4096], local=True)
b = op.qknorm_scaled_rope(q, k, c_cpu.to(DEV), s_cpu.to(DEV), local=True)
out["cpu_table_through_strict_op"] = {
    "raised": False,
    "q_hat_mismatch": int((a[0] != b[0]).sum()),
    "k_hat_mismatch": int((a[1] != b[1]).sum()),
}

# legacy implicit path (local=None) still silently NoPE on a sliding layer
nope = op.qknorm_scaled_rope(q, k, None, None)
out["legacy_local_none_silent_nope_q_mismatch"] = int((nope[0] != a[0]).sum())

# verify=True memoisation: mutate canonical table in place, verify still passes?
cmod, smod = op.canonical_rope_table(DEV)
saved = cmod[5].clone()
cmod[5, 0] += 1.0  # corrupt the shared cached object (callers can: it is a plain tensor)
try:
    op.canonical_rope_table(DEV, verify=True)
    out["verify_after_inplace_corruption"] = "PASSED (no error) -- digest is memoised"
except Exception as e:  # noqa
    out["verify_after_inplace_corruption"] = f"raised {type(e).__name__}"
out["sha_reported_after_corruption"] = op.rope_table_sha256(DEV)
out["sha_actual_after_corruption"] = sha(cmod, smod)
rows = op.rope_rows(torch.tensor([5], device=DEV))
out["rope_rows_returns_corrupted_row"] = bool(rows[0][0, 0] != saved[0])
cmod[5].copy_(saved)

# T=0
z = torch.empty(0, 32, 128, device=DEV, dtype=torch.bfloat16)
zk = torch.empty(0, 2, 128, device=DEV, dtype=torch.bfloat16)
zc = torch.empty(0, 128, device=DEV)
try:
    zq = z.clone().requires_grad_(True)
    zkk = zk.clone().requires_grad_(True)
    o = op.qknorm_scaled_rope(zq, zkk, zc, zc.clone(), local=True)
    (o[0].float().sum() + o[1].float().sum()).backward()
    out["T0"] = "ok"
except Exception as e:  # noqa
    out["T0"] = f"{type(e).__name__}: {e}"[:200]
try:
    r = op.qknorm_scaled_rope_at(z, zk, torch.empty(0, dtype=torch.long, device=DEV), local=True)
    out["T0_at"] = "ok"
except Exception as e:  # noqa
    out["T0_at"] = f"{type(e).__name__}: {e}"[:200]

# k on other device / cpu with q on cuda
try:
    op.qknorm_scaled_rope(q[:4], k[:4].cpu(), None, None)
    out["k_on_cpu"] = "accepted"
except Exception as e:  # noqa
    out["k_on_cpu"] = f"{type(e).__name__}: {e}"[:160]

# local passed as a non-bool truthy value
try:
    op.qknorm_scaled_rope_at(q[:4], k[:4], None, local="False")
    out["local_str_False"] = "accepted"
except Exception as e:  # noqa
    out["local_str_False"] = f"{type(e).__name__}: {e}"[:160]

# fp32 grad into backward via autograd (downstream fp32 consumer)
qq = q[:8].clone().requires_grad_(True)
kk = k[:8].clone().requires_grad_(True)
oq, ok = op.qknorm_scaled_rope(qq, kk, cc[:8], sc[:8], local=True)
try:
    (oq.float() * 2).sum().backward()
    out["fp32_downstream_backward"] = f"ok dq dtype {qq.grad.dtype}"
except Exception as e:  # noqa
    out["fp32_downstream_backward"] = f"{type(e).__name__}: {e}"[:200]

print(json.dumps(out, indent=1))
