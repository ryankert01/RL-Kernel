"""List cross_runtime_xrt pairs and the tests block of megatron_runtime.json."""
import json
import sys

d = json.load(open(sys.argv[1]))
for k, v in d["cross_runtime_xrt"]["pairs"].items():
    print(k, v["b"]["commit"], v["b"]["postdates_o7_fix"], v["bit_mismatches"], v["tokens"],
          v["both_postdate_o7_fix"])
print(json.dumps(d["tests"], indent=1)[:4000])
print(json.dumps(d["runs"]["tp1"], indent=None)[:1500])
