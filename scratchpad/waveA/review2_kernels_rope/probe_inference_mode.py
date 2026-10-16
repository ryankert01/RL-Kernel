"""Canonical table first built under torch.inference_mode() (vLLM forward), then used by a
training forward+backward in the same process (colocated RL)."""
import json

import torch

from rl_engine.models.muse_glimmer.ops import qknorm_rope as op

DEV = torch.device("cuda")
out = {}
g = torch.Generator(device=DEV).manual_seed(1)
q = torch.randn(16, 32, 128, device=DEV, generator=g).to(torch.bfloat16)
k = torch.randn(16, 2, 128, device=DEV, generator=g).to(torch.bfloat16)
pos = torch.arange(16, device=DEV) + 100
with torch.inference_mode():
    a = op.qknorm_scaled_rope_at(q, k, pos, local=True)
c, s = op.canonical_rope_table(DEV)
out["table_is_inference"] = bool(c.is_inference())
try:
    qq = q.clone().requires_grad_(True)
    kk = k.clone().requires_grad_(True)
    oq, ok = op.qknorm_scaled_rope_at(qq, kk, pos, local=True)
    (oq.float().sum() + ok.float().sum()).backward()
    out["train_after_inference_build"] = "ok"
    out["fwd_equal"] = bool(torch.equal(oq.detach(), a[0]))
except Exception as e:  # noqa
    out["train_after_inference_build"] = f"{type(e).__name__}: {str(e)[:200]}"
try:
    op.canonical_rope_table(DEV, verify=True)
    out["verify_after_inference_build"] = "ok"
except Exception as e:  # noqa
    out["verify_after_inference_build"] = f"{type(e).__name__}: {str(e)[:200]}"
# in-place guard: can a caller mutate the shared inference-mode table outside inference mode?
try:
    c.add_(0)
    out["inplace_on_shared_table"] = "allowed"
except Exception as e:  # noqa
    out["inplace_on_shared_table"] = f"{type(e).__name__}: {str(e)[:120]}"
print(json.dumps(out, indent=1))
