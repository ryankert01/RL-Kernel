import json
import sys

d = json.load(open(sys.argv[1]))
print(d["modules"], d["env"], "wall", d["wall_s"])
bad = 0
for r in d["tp"]:
    tot = {}
    for rk in r["ranks"]:
        for k, v in rk.items():
            if k in ("rank", "q_heads", "kv_head") or k.startswith("numel"):
                continue
            tot[k] = tot.get(k, 0) + v
    extra = {k: r[k] for k in r if k.endswith("vs_tp1")}
    print(r["case"], "tp", r["tp"], "win", r["window"], tot, extra)
    bad += sum(v for v in tot.values() if v) + sum(v for v in extra.values() if v)
for r in d["cp"]:
    print("cp", r["cp"], r["T"], r["window"], r["dq"], r["dk"], r["dv"], r["dgate"])
    bad += r["dq"] + r["dk"] + r["dv"] + r["dgate"]
print("total mismatches", bad)
