import json
import sys

R = sys.argv[1]
for n in ("before", "after"):
    d = json.load(open(f"{R}/probe_wave_b_{n}.json"))
    print("=====", n, d["module"])
    for k, v in d.items():
        if isinstance(v, dict):
            print(k, json.dumps(v)[:800])
