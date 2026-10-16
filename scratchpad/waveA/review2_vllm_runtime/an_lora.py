import json
import sys

r = json.load(open(sys.argv[1]))
print("status", r.get("status"), r.get("error"))
for g in r.get("generate", []):
    dv = g["decode_vs_prefill"]
    print("base gen T", g["temperature"], "tokens", dv["tokens"], "mismatched", dv["mismatched"])
lr = r.get("lora_random") or {}
print("hash_identity", lr.get("hash_identity"), "worker_hashes", lr.get("worker_hashes"))
s = lr.get("scores_vs_no_lora") or {}
print("scores_vs_no_lora", s.get("tokens"), s.get("mismatched"))
for rule in lr.get("unmerged_rule", []):
    print("rule layer", rule["layer"], "all_equal", rule["all_equal"],
          {k: (v["binding_equals_formula"], v["binding_equals_megatron"]) for k, v in rule["slots"].items()})
for g in lr.get("generate", []):
    dv = g["decode_vs_prefill"]
    lp = [x for row in g["logprobs"] for x in row]
    print("lora gen T", g["temperature"], "tokens", dv["tokens"], "mismatched", dv["mismatched"], "exact0", sum(x == 0 for x in lp),
          "tok/s %.1f" % g["tokens_per_s"])
w = (r.get("meta") or {}).get("workers", [])
for x in w:
    cc = x.get("call_counts", {})
    print("worker", x.get("all_ops_strict"), {k: cc.get(k) for k in ("lora.column", "lora.row", "load_lora")})
