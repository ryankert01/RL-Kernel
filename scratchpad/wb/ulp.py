import json
import subprocess
from collections import Counter

old = json.loads(subprocess.run(
    ["git", "show", "HEAD:reports/muse_glimmer_poc/data/xrt_reference_triton.json"],
    capture_output=True, text=True, check=True).stdout)
new = json.load(open("reports/muse_glimmer_poc/data/xrt_reference_triton.json"))


def key(h):
    i = int(h, 16)
    # monotone integer key for IEEE fp32 ordering
    return i if i < 0x80000000 else -(i - 0x80000000)


c = Counter()
for a, b in zip(old["sequences"], new["sequences"]):
    for x, y in zip(a["logprobs_hex"], b["logprobs_hex"]):
        c[abs(key(x) - key(y))] += 1
print(sorted(c.items()))
