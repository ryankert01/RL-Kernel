import json
import math
import sys

d = json.load(open(sys.argv[1]))
a = d["analysis"]
pw = a["hf_pairwise_identical_tokens"]
names = pw["settings"]
print("n_tokens", pw["n_tokens"], "sequences", len(pw["sequences"]))
short = [n[:14] for n in names]
print(" " * 28 + " ".join("%5d" % i for i in range(len(names))))
for i, (n, row) in enumerate(zip(names, pw["matrix"])):
    print("%2d %-25s" % (i, n[:25]) + " ".join("%5d" % x for x in row))

print()
for k in ("hf/padded_batch9_right", "hf/softcap_fp32", "cross/strict_vs_hf_baseline",
          "strict/packed_batch9", "hf/kv_cache_decode"):
    print(k, json.dumps(a["abs_dlogp_by_baseline_prob"].get(k)))

# identical-token check: which tokens stay identical for the schedule factors?
base = {p["sequence"]: p for p in d["baselines"]["hf"]["per_sequence"]}
recs = {}
for r in d["records"]:
    recs.setdefault((r["engine"], r["setting"]), {})[r["sequence"]] = r
for setting in ("padded_batch9_right", "sdpa", "kv_cache_decode", "chunked_prefill_64"):
    same = []
    for sid, r in recs[("hf", setting)].items():
        for i, b in enumerate(r["bits_differ"]):
            if not b:
                same.append(round(base[sid]["logp"][i], 6))
    print(setting, "identical tokens logp:", same)

# nonfinite tokens anywhere
print("nonfinite:", sum(r["nonfinite_tokens"] for r in d["records"]))
# strict baseline finite and close to HF
sb = d["baselines"]["strict"]["per_sequence"]
print("strict baseline finite:", all(math.isfinite(x) for p in sb for x in p["logp"]),
      "n", sum(len(p["logp"]) for p in sb))
# argmax ties check: how many baseline tokens' argmax equals target
am = sum(1 for p in base.values() for a_, t in zip(p["argmax"], p["targets"]) if a_ == t)
print("baseline argmax==target", am, "of", sum(len(p["targets"]) for p in base.values()))
