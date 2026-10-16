import json
import sys

d = json.load(open(sys.argv[1]))
print(d["commits_measured"])
for c in d["commands"]:
    print("CMD", c)
for r in d["runs"]:
    print({k: v for k, v in r.items() if k != "command"})
print()
for g in d["generate"]:
    dv = g.get("decode_vs_prefill") or {}
    lora = g["lora"]
    print(
        g["run"], g["mode"], g["tp"], "lora_en=", g["engine_lora_enabled"],
        (lora or {}) if not isinstance(lora, dict) else {k: str(v)[:16] for k, v in lora.items()},
        "T=", g["temperature"], g["generated_tokens"], "tps=%.1f" % g["tokens_per_s"],
        "mism=", dv.get("mismatched"), dv.get("max_abs_diff"),
    )
    print("   keys:", list(g.keys()))
print()
for z in d["lora_zero"]:
    print("ZERO", z["mode"], z["tp"], z["hash_identity"], z["vs_no_lora"]["mismatched"], z["vs_no_lora"]["tokens"])
for z in d["lora_random"]:
    print("RAND", {k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk not in ("per_sequence", "abs_diff_per_token")}) for k, v in z.items() if k not in ("worker_hashes", "lora_replies")})
for t in d["throughput"]:
    print("TPUT", t["mode"], t["tp"], t["tokens_per_s"], t["ttft_ms"], t["decode_step_ms_median"], t.get("memory_per_rank"))
print(json.dumps(d["plot_hints"], indent=1))
print(json.dumps(d["tests"], indent=1))
print(json.dumps(d["tp4_engine"], indent=1))
print(json.dumps(d["findings"], indent=1))
