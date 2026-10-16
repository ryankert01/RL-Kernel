"""Compare the owner's grads_tp{1,2,4}.pt bitwise and recompute the aggregate hash."""
import hashlib
import json
import sys

import torch

d = sys.argv[1]
g = {tp: torch.load(f"{d}/grads_tp{tp}.pt", map_location="cpu", weights_only=False) for tp in (1, 2, 4)}
for tp, v in g.items():
    print(tp, type(v), (list(v.keys())[:3] if isinstance(v, dict) else None))
out = {}


def flat(x):
    return x["grads"] if isinstance(x, dict) and "grads" in x else x


ref = flat(g[1])
print("tensors", len(ref), "elements", sum(t.numel() for t in ref.values()))
h = hashlib.sha256()
for k in sorted(ref):
    h.update(ref[k].contiguous().view(torch.int32).numpy().tobytes())
out["agg_sha_tp1_mine(keys sorted, fp32 bytes)"] = h.hexdigest()
for tp in (2, 4):
    other = flat(g[tp])
    assert set(other) == set(ref), (tp, set(other) ^ set(ref))
    diff = 0
    tens = 0
    for k in ref:
        a, b = ref[k], other[k]
        assert a.shape == b.shape and a.dtype == b.dtype == torch.float32, k
        n = int((a.view(torch.int32) != b.view(torch.int32)).sum())
        diff += n
        tens += n > 0
    out[f"tp{tp}_vs_tp1_diff_elems"] = diff
    out[f"tp{tp}_vs_tp1_diff_tensors"] = tens
    nz = sum(int((other[k] != 0).sum()) for k in other)
    out[f"tp{tp}_nonzero"] = nz
out["tp1_nonzero"] = sum(int((ref[k] != 0).sum()) for k in ref)
print(json.dumps(out, indent=1))
