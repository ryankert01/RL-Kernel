import json
import sys

d = json.load(open(sys.argv[1]))
for r in d["checks"]["bench"]["rows"]:
    for k, v in r["errors"].items():
        print(r["tokens"], k, v[:260])
    print(r["tokens"], {k: [round(x) for x in v] for k, v in r["samples"].items()
                        if k.endswith("_ms") and "total" in k or k == "strict_forward_ms"})
