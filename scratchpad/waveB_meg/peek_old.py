import json
import sys

d = json.load(open(sys.argv[1]))
t = d["runs"]["tp2"]["checks"]["identity"]
print({k: v for k, v in t.items() if k != "per_tensor"})
print(json.dumps(d["tests"], indent=1)[:4000])
