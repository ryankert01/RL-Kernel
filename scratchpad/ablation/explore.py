import json
import math
import sys
from collections import defaultdict

d = json.load(open(sys.argv[1]))
recs = [r for r in d["records"] if r["engine"] == "hf"]
by = defaultdict(dict)
for r in recs:
    by[r["setting"]][r["sequence"]] = r
names = list(by)
seqs = list(by[names[0]])
print("pairwise: tokens with identical bits between HF settings (of 933)")
for a in names:
    row = []
    for b in names:
        n = 0
        for s in seqs:
            n += sum(x == y for x, y in zip(by[a][s]["logp_hex"], by[b][s]["logp_hex"]))
        row.append(n)
    print(f"{a[:26]:26s}", " ".join(f"{x:4d}" for x in row))

base = {p["sequence"]: p for p in d["baselines"]["hf"]["per_sequence"]}
# |d| vs baseline logp buckets for padded right
for name in ("padded_batch9_right", "kv_cache_decode", "softcap_fp32", "sdpa"):
    buckets = defaultdict(list)
    for s in seqs:
        for lp, dd in zip(base[s]["logp"], by[name][s]["dlogp"]):
            p = math.exp(lp)
            k = "p>0.9" if p > 0.9 else "0.5<p<=0.9" if p > 0.5 else "0.1<p<=0.5" if p > 0.1 else "p<=0.1"
            buckets[k].append(abs(dd))
    print(name, {k: (len(v), round(sum(v) / len(v), 4), round(max(v), 3)) for k, v in buckets.items()})

# top tokens by |d|
for name in ("prefill_L_minus_1", "padded_batch9_right"):
    top = []
    for s in seqs:
        for i, (lp, dd) in enumerate(zip(base[s]["logp"], by[name][s]["dlogp"])):
            top.append((abs(dd), s, i, lp, dd))
    top.sort(reverse=True)
    print(name, [(round(a, 3), s, i, round(lp, 3)) for a, s, i, lp, dd in top[:5]])
# baseline argmax == target fraction
n = sum(len(p["targets"]) for p in base.values())
hit = sum(sum(a == t for a, t in zip(p["argmax"], p["targets"])) for p in base.values())
print("baseline argmax==gold", hit, n)
lps = [x for p in base.values() for x in p["logp"]]
print("mean baseline logp", sum(lps) / len(lps), "frac p>0.9", sum(math.exp(x) > 0.9 for x in lps) / n)
