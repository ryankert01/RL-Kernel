"""Compare two muse_glimmer_xrt_logprobs_v1 files bitwise (read-only)."""
import json
import sys

a = json.load(open(sys.argv[1]))
b = json.load(open(sys.argv[2]))
ha = {s["id"]: s["logprobs_hex"] for s in a["sequences"]}
hb = {s["id"]: s["logprobs_hex"] for s in b["sequences"]}
assert set(ha) == set(hb), (set(ha) ^ set(hb))
tot = sum(len(v) for v in ha.values())
mism = sum(x != y for k in ha for x, y in zip(ha[k], hb[k]))
lens_ok = all(len(ha[k]) == len(hb[k]) for k in ha)
print(sys.argv[1].rsplit("/", 1)[-1], "vs", sys.argv[2].rsplit("/", 1)[-1], "tokens", tot, "mismatched", mism, "lens_ok", lens_ok,
      "sequences_identical", a["sequences"] == b["sequences"])
