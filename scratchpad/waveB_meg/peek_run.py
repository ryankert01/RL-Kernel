import json
import sys

d = json.load(open(sys.argv[1]))
p = d.get("provenance", {})
print("status", d.get("status"), "commit", d.get("commit"), "dirty", d.get("dirty"))
print("load", d.get("load"))
for k in ("memory_gib", "vision_drop", "attention_heads_per_rank", "rope_table_sha256", "notes",
          "micro_batch_tokens", "packing_rule", "lora_grad_rule", "tp_collective"):
    print(k, "=", p.get(k))
print("ops all_strict", p.get("ops", {}).get("all_strict"))
C = d.get("checks", {})
for name, v in C.items():
    if isinstance(v, dict):
        brief = {k: x for k, x in v.items() if not isinstance(x, (list, dict))}
        print(name, brief)
if "identity" in C:
    print("identity mismatches", [m["name"] for m in C["identity"]["mismatches"]])
if "packing" in C:
    for b in C["packing"]["per_budget"]:
        print("packing", {k: x for k, x in b.items() if k != "memory"})
if "step" in C:
    print("step info", C["step"]["info"])
    print("sha", C["step"]["lora_sha256_before"], C["step"]["lora_sha256_after"])
if "grad_repeat" in C:
    print("grad agg", C["grad_repeat"]["aggregate_sha256"])
    for ps in C["grad_repeat"]["passes"]:
        print("  pass", ps["info"], ps["fwd_s"], ps["bwd_s"])
if "bench" in C:
    for r in C["bench"]["rows"]:
        print("bench", r["tokens"], {k: round(x, 1) for k, x in r["median"].items()},
              r["peak_gib"], list(r["errors"]))
