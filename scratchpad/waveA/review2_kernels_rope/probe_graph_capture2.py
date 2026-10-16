"""First use of qknorm_scaled_rope_at / rope_rows inside CUDA-graph capture: the lazy canonical
build is recorded, not executed, so the cached table holds uninitialised graph-pool memory
until the graph is replayed.  What does an EAGER caller (prefill) get right after capture?"""
import hashlib
import json

import torch

from rl_engine.models.muse_glimmer.ops import qknorm_rope as op

DEV = torch.device("cuda")


def sha(c, s):
    h = hashlib.sha256()
    for t in (c, s):
        h.update(t.cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


out = {}
B = 8
q = torch.randn(B, 32, 128, device=DEV).to(torch.bfloat16)
k = torch.randn(B, 2, 128, device=DEV).to(torch.bfloat16)
pos = torch.arange(B, device=DEV) + 10
c0, s0 = op.rope_table(pos, device=DEV)
ref = op.qknorm_scaled_rope(q, k, c0, s0, local=True)  # compile outside capture
torch.cuda.synchronize()
g = torch.cuda.CUDAGraph()
st = torch.cuda.Stream()
st.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(st):
    with torch.cuda.graph(g):
        gq, gk = op.qknorm_scaled_rope_at(q, k, pos, local=True)
torch.cuda.synchronize()
cos, sin = op.canonical_rope_table(DEV)
out["after_capture_table_sha_equals_pinned"] = sha(cos, sin) == op.CANONICAL_ROPE_TABLE_SHA256
# an eager prefill call right after capture (before any replay)
pos2 = torch.arange(64, device=DEV) + 1000
q2 = torch.randn(64, 32, 128, device=DEV).to(torch.bfloat16)
k2 = torch.randn(64, 2, 128, device=DEV).to(torch.bfloat16)
try:
    eq, ek = op.qknorm_scaled_rope_at(q2, k2, pos2, local=True)
    rq, rk = op.qknorm_scaled_rope(q2, k2, *op.rope_table(pos2, device=DEV), local=True)
    out["eager_after_capture_raised"] = False
    out["eager_after_capture_q_mismatch"] = int((eq != rq).sum())
    out["eager_after_capture_q_finite"] = bool(torch.isfinite(eq.float()).all())
except Exception as e:  # noqa
    out["eager_after_capture_raised"] = f"{type(e).__name__}: {str(e)[:150]}"
g.replay()
torch.cuda.synchronize()
out["after_replay_table_sha_equals_pinned"] = sha(cos, sin) == op.CANONICAL_ROPE_TABLE_SHA256
out["graph_output_equals_ref_after_replay"] = bool(torch.equal(gq, ref[0]))
print(json.dumps(out, indent=1))
