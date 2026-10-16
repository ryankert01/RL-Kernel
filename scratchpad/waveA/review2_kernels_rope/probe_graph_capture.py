"""First use of rope_rows / qknorm_scaled_rope_at inside CUDA-graph capture (vLLM decode graphs):
is the lazy 131072-row canonical-table build captured into the graph?"""
import json

import torch

from rl_engine.models.muse_glimmer.ops import qknorm_rope as op

DEV = torch.device("cuda")
out = {}
B = 8
q = torch.randn(B, 32, 128, device=DEV).to(torch.bfloat16)
k = torch.randn(B, 2, 128, device=DEV).to(torch.bfloat16)
pos = torch.arange(B, device=DEV) + 10
# warm the Triton compile outside capture with explicit tables (not the canonical cache)
c0, s0 = op.rope_table(pos, device=DEV)
op.qknorm_scaled_rope(q, k, c0, s0, local=True)
torch.cuda.synchronize()
out["canonical_built_before_capture"] = bool(op._CANONICAL_TABLES)
g = torch.cuda.CUDAGraph()
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
try:
    with torch.cuda.stream(s):
        with torch.cuda.graph(g):
            oq, ok = op.qknorm_scaled_rope_at(q, k, pos, local=True)
    torch.cuda.synchronize()
    out["capture"] = "ok (no error)"
    out["canonical_built_during_capture"] = bool(op._CANONICAL_TABLES)
    cos, sin = op.canonical_rope_table(DEV)
    snap = cos.clone()
    # corrupt the cache then replay: if the build was captured, replay rewrites the table
    cos.zero_()
    g.replay()
    torch.cuda.synchronize()
    out["table_restored_by_replay (build captured into graph)"] = bool(torch.equal(cos, snap))
    cos.copy_(snap)
    try:
        op.canonical_rope_table(DEV, verify=True)
        out["verify_after"] = "ok"
    except Exception as e:  # noqa
        out["verify_after"] = f"{type(e).__name__}: {str(e)[:150]}"
except Exception as e:  # noqa
    out["capture"] = f"{type(e).__name__}: {str(e)[:200]}"
print(json.dumps(out, indent=1))
