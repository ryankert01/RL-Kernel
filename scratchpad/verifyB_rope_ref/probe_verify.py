"""Adversarial probes for the rope_ref wave B fixes (verifier, read-only)."""
import json
import os
import sys
import warnings

import torch

from rl_engine.models.muse_glimmer.ops import qknorm_rope as op

DEV = torch.device("cuda")
out = {}


def tryit(fn):
    try:
        r = fn()
        return {"raised": False, "ret": r}
    except Exception as e:  # noqa
        return {"raised": True, "err": f"{type(e).__name__}: {str(e)[:160]}"}


torch.manual_seed(0)
T = 4096
q = (torch.randn(T, 32, 128, device=DEV) * 2).bfloat16()
k = (torch.randn(T, 2, 128, device=DEV) * 2).bfloat16()
pos = torch.arange(T, device=DEV)
cos, sin = op.rope_rows(pos)

# ---- R1: legacy local=None
os.environ.pop(op.LAYER_KIND_ENV, None)
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    qa, _ = op.qknorm_scaled_rope(q, k, None, None)  # sliding layer given no table
    qb, _ = op.qknorm_scaled_rope(q, k, cos, sin, local=True)
out["R1_env_unset"] = {
    "raised": False,
    "q_hat_diff": int((qa.view(torch.int16) != qb.view(torch.int16)).sum()),
    "n": qa.numel(),
    "warnings": [str(x.message)[:80] for x in w],
    "counts": op.runtime_provenance(compute_digest=False)["layer_kind_calls"],
}
os.environ[op.LAYER_KIND_ENV] = "1"
out["R1_env_1_none"] = tryit(lambda: op.qknorm_scaled_rope(q, k, None, None) and None)
out["R1_env_1_cos"] = tryit(lambda: op.qknorm_scaled_rope(q, k, cos, sin) and None)
out["R1_env_true"] = None
os.environ[op.LAYER_KIND_ENV] = "true"
out["R1_env_true"] = op.layer_kind_required()
os.environ[op.LAYER_KIND_ENV] = " 0 "
out["R1_env_' 0 '"] = op.layer_kind_required()
os.environ[op.LAYER_KIND_ENV] = "off"
out["R1_env_off_required"] = op.layer_kind_required()  # 'off' -> treated as required?
os.environ.pop(op.LAYER_KIND_ENV, None)

# golden_ops qknorm_scaled_rope has local but no env switch
from rl_engine.models.muse_glimmer.reference import golden_ops

os.environ[op.LAYER_KIND_ENV] = "1"
out["R1_golden_env_1_none"] = tryit(lambda: golden_ops.qknorm_scaled_rope(q, k, None, None) and None)
os.environ.pop(op.LAYER_KIND_ENV, None)

# ---- R2 builders on CPU
from rl_engine.models.muse_glimmer.runtime import fallback_ops, strict_kernels

p4 = torch.arange(4096)
for name, fn in (
    ("ops.rope_table", lambda: op.rope_table(p4, device="cpu")),
    ("golden_ops.rope_table", lambda: golden_ops.rope_table(p4, theta=5e5, dim=128, device="cpu")),
    ("fallback_ops.rope_table", lambda: fallback_ops.rope_table(p4, theta=5e5, dim=128, device="cpu")),
    ("strict_kernels._fb_rope_table", lambda: strict_kernels._fb_rope_table(p4, theta=5e5, dim=128, device=torch.device("cpu"))),
):
    r = tryit(fn)
    if not r["raised"]:
        c, _ = r["ret"]
        r["ret"] = f"accepted; cos elems != cuda: {int((c.to(DEV) != cos).sum())}"
    out[f"R2_cpu_{name}"] = r

# do the strict bindings resolve rope_table to the canonical builder?
from rl_engine.models.muse_glimmer.reference.backend import resolve_ops as load_ops  # noqa
try:
    ops_t = load_ops("triton", allow_partial=True)
    out["R2_ref_triton_rope_table"] = f"{ops_t.rope_table.__module__}.{ops_t.rope_table.__name__}"
except Exception as e:
    out["R2_ref_triton_rope_table"] = f"ERR {e}"

# same object? rope_table returns copies
c1, _ = op.rope_table(pos, device=DEV)
cc, sc = op.canonical_rope_table(DEV)
out["R2_rope_table_returns_copy"] = c1.data_ptr() != cc.data_ptr()

# ---- R3: in-place detection (views, out=, detach)
key = (torch.cuda.current_device(), op.ROPE_THETA, op.HEAD_DIM)
cv = cc[100:105]
saved = cc[100:105].clone()
cv.mul_(1.0)  # view modification, identical values
out["R3_view_mul_detected"] = tryit(lambda: op.canonical_rope_table(DEV) and None)
# restore version bookkeeping for further probes (verifier only)
op._CANONICAL_VERSIONS[key] = op._versions(op._CANONICAL_TABLES[key])
d = cc.detach()
d[0, 0] = d[0, 0] + 0.0
out["R3_detach_write_detected"] = tryit(lambda: op.canonical_rope_table(DEV) and None)
op._CANONICAL_VERSIONS[key] = op._versions(op._CANONICAL_TABLES[key])
# write through untyped storage (bypasses counters): cheap check passes, verify catches
st = cc.untyped_storage()
orig = cc[7, 3].item()
cc.data[7, 3] = orig + 1.0
out["R3_data_write_cheap_check"] = tryit(lambda: op.canonical_rope_table(DEV) and None)
out["R3_data_write_verify"] = tryit(lambda: op.canonical_rope_table(DEV, verify=True) and None)
out["R3_sha_after_write_eq_pinned"] = op.rope_table_sha256(DEV) == op.CANONICAL_ROPE_TABLE_SHA256
out["R3_runtime_prov_after_write"] = op.runtime_provenance()["rope_table_equals_pinned"]
cc.data[7, 3] = orig
out["R3_verify_after_restore"] = tryit(lambda: op.canonical_rope_table(DEV, verify=True) and None)
# memo staleness: rope_table_sha256 after .data write without verify
op.rope_table_sha256(DEV)  # memo at current versions
cc.data[7, 3] = orig + 1.0
out["R3_sha256_memo_after_data_write_no_verify_eq_pinned"] = (
    op.rope_table_sha256(DEV) == op.CANONICAL_ROPE_TABLE_SHA256)
out["R3_runtime_prov_memo_after_data_write"] = op.runtime_provenance()["rope_table_equals_pinned"]
cc.data[7, 3] = orig
op.canonical_rope_table(DEV, verify=True)

# ---- R4: first build during capture, via rope_table (the bindings' entry point)
key32 = (torch.cuda.current_device(), op.ROPE_THETA, 32)
torch.cuda.synchronize()
g = torch.cuda.CUDAGraph()
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
res = None
with torch.cuda.stream(s):
    try:
        with torch.cuda.graph(g):
            op.rope_table(torch.arange(8, device=DEV), dim=32, device=DEV)
        res = {"raised": False}
    except Exception as e:
        res = {"raised": True, "err": f"{type(e).__name__}: {str(e)[:120]}"}
torch.cuda.current_stream().wait_stream(s)
torch.cuda.synchronize()
res["cached_after"] = key32 in op._CANONICAL_TABLES
out["R4_rope_table_first_build_in_capture"] = res
# canonical build under inference_mode then used normally
key16 = (torch.cuda.current_device(), 1e4, 16)
with torch.inference_mode():
    c16, _ = op.rope_table(torch.arange(3, device=DEV), theta=1e4, dim=16, device=DEV)
out["R4_inference_table_is_inference"] = op._CANONICAL_TABLES[key16][0].is_inference()

# ---- out of range / negative positions
out["range_neg"] = tryit(lambda: op.rope_table(torch.tensor([-1], device=DEV), device=DEV) and None)
out["range_big"] = tryit(lambda: op.rope_table(torch.tensor([131072], device=DEV), device=DEV) and None)

# ---- trace / provenance
tr = op.trace()["rope_table"]
out["R11_trace_keys"] = sorted(tr)
out["R11_prov"] = op.runtime_provenance()

print(json.dumps(out, indent=1, default=str))
