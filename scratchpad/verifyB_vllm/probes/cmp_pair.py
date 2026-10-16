import json
import sys


def load(f):
    d = json.load(open(f))
    return {s["id"]: s["logprobs_hex"] for s in d["sequences"]}, d


ref_files = sys.argv[1].split(",")
other_files = sys.argv[2].split(",")
for rf in ref_files:
    r, rd = load(rf)
    for of in other_files:
        o, od = load(of)
        assert set(r) == set(o), set(r) ^ set(o)
        tot = sum(len(v) for v in r.values())
        mis = sum(1 for k in r for a, b in zip(r[k], o[k]) if a != b)
        lens = all(len(r[k]) == len(o[k]) for k in r)
        print(f"{rf.split('/')[-1]} [{rd.get('commit', '')[:7]}] vs {of.split('/')[-1]} "
              f"[{od.get('commit', '')[:7]} tp{od.get('tp')}]: {mis}/{tot} lens_equal={lens}")
