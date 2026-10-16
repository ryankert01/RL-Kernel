"""Compare two digest JSONs from gated_attention_bench digest: per case, per output."""
import json
import sys

a = json.load(open(sys.argv[1]))
b = json.load(open(sys.argv[2]))
out = {"base": sys.argv[1], "new": sys.argv[2], "base_modules": a["modules"],
       "new_modules": b["modules"]}
for part in ("forward", "backward"):
    keys = sorted(set(a[part]) | set(b[part]))
    same = [k for k in keys if a[part].get(k) == b[part].get(k)]
    diff = [k for k in keys if a[part].get(k) != b[part].get(k)]
    out[part] = {"cases": len(keys), "identical": len(same), "different": diff}
    print(f"{part}: {len(same)}/{len(keys)} identical; different: {diff}")
if len(sys.argv) > 3:
    json.dump(out, open(sys.argv[3], "w"), indent=1)
