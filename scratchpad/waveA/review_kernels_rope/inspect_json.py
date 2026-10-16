import json, sys
from collections import Counter

d = json.load(open(sys.argv[1]))
t = d["tests_new"]
print({k: v for k, v in t.items() if k != "per_test"})
print(Counter((p["before_6a61f6e"], p["after_vllm_env"], p["after_megatron_env"]) for p in t["per_test"]))
for p in t["per_test"]:
    if p["before_6a61f6e"] != "passed":
        print(p)
e = d["experiments"]
print(json.dumps(e["bitwise_unchanged_by_fix"], indent=0)[:3000])
print(e["host_load_note"])
print(json.dumps(e["cpu_overhead_T64_us"]))
print(json.dumps(e["bench_repo_T4096"])[:2500])
