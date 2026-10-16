"""verify=True after an in-place corruption of the shared cached table, with the digest memo
already populated (the documented usage: verify once per process at start-up, then gather)."""
import hashlib
import json

import torch

from rl_engine.models.muse_glimmer.ops import qknorm_rope as op

DEV = torch.device("cuda")
out = {}
c, s = op.canonical_rope_table(DEV, verify=True)  # start-up check: populates the memo
out["startup_verify"] = "ok"
c.mul_(1.0 + 2**-20)  # any in-place op by any holder of the shared object
try:
    op.canonical_rope_table(DEV, verify=True)
    out["verify_after_corruption"] = "PASSED silently (memoised digest)"
except Exception as e:  # noqa
    out["verify_after_corruption"] = f"raised {type(e).__name__}"
h = hashlib.sha256()
for t in (c, s):
    h.update(t.cpu().contiguous().view(torch.uint8).numpy().tobytes())
out["reported_sha"] = op.rope_table_sha256(DEV)
out["actual_sha"] = h.hexdigest()
out["pinned"] = op.CANONICAL_ROPE_TABLE_SHA256
out["cos_version_counter"] = c._version
print(json.dumps(out, indent=1))
