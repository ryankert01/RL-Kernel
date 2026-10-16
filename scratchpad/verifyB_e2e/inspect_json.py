import json
import sys

d = json.load(open(sys.argv[1]))
what = sys.argv[2]


def brief(v):
    if isinstance(v, (dict, list)) and len(json.dumps(v)) > 600:
        return list(v.keys()) if isinstance(v, dict) else f"list[{len(v)}]"
    return v


if what == "smoke":
    for r in d["real_e2e_smoke"]:
        print(json.dumps({k: brief(v) for k, v in r.items()}, indent=1, default=str)[:5000])
        print("-----")
elif what == "findings":
    print(json.dumps(d["integration_findings"], indent=1)[:9000])
elif what == "overlap":
    print(json.dumps(d["startup_overlap"], indent=1)[:6000])
elif what == "path":
    o = d
    for p in sys.argv[3].split("/"):
        o = o[int(p)] if isinstance(o, list) else o[p]
    print(json.dumps(o, indent=1, default=str)[: int(sys.argv[4]) if len(sys.argv) > 4 else 6000])
