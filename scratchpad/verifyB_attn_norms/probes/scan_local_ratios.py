"""Scan kernels_attention.json for local-layer rows whose SDPA ratio fields are still numbers."""
import json
import sys

d = json.load(open(sys.argv[1]))
bad = []
nulls = 0
locals_seen = 0


def is_local(o):
    for key in ("window", "layer", "kind", "layer_kind"):
        if key in o:
            val = o[key]
            if key == "window" and val not in (None, "none", "None", 0, "global"):
                return True
            if isinstance(val, str) and "local" in val.lower():
                return True
    return False


def walk(o, path):
    global nulls, locals_seen
    if isinstance(o, dict):
        if is_local(o):
            locals_seen += 1
            for kk, vv in o.items():
                lk = kk.lower()
                if "ratio" in lk and "flash" not in lk and "fa2" not in lk:
                    if vv is None:
                        nulls += 1
                    elif isinstance(vv, (int, float)):
                        bad.append((path, kk, vv, {x: o.get(x) for x in ("T", "B", "window", "baseline_not_comparable")}))
        for kk, vv in o.items():
            walk(vv, path + "." + kk)
    elif isinstance(o, list):
        for i, vv in enumerate(o):
            walk(vv, f"{path}[{i}]")


walk(d, "")
print("local rows", locals_seen, "null ratios", nulls, "non-null local ratio fields", len(bad))
for b in bad[:60]:
    print(b)
