import json
import sys

from rl_engine.models.muse_glimmer.reference.xrt_scoring import compare_xrt

old = json.load(open(sys.argv[1]))
new = json.load(open("reports/muse_glimmer_poc/data/xrt_reference_triton.json"))
d = compare_xrt(old, new)
a = sorted(v for r in d["sequences"] for v in r["abs_diff"] if v > 0)
print(len(a), "max", a[-1], "median", a[len(a) // 2])
