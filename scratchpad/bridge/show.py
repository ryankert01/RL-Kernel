import json
import sys

d = json.load(open(sys.argv[1]))
print(list(d))
print(d["commit"], d["commands"])
print(json.dumps(d["surrogate_exactness"], indent=0)[:700])
for e in d.get("bridge_equivalence", []):
    print(e["name"], e["server_torch"], round(e["startup_s"], 2), e["all_bit_exact"],
          [(r["logprob_bit_mismatches"], r["lora_element_mismatches"], r["lora_sha256_equal"],
            r["grad_norm_equal"]) for r in e["steps"]], e["last_export_rank_sha256"] is not None)
for r in d.get("transport", {}).get("payloads", []):
    if r["repeat"] == 1:
        print(r["payload_bytes"], round(r["rtt_ms"], 2), round(r["round_trip_MBps"], 1),
              r["bit_exact"])
lr = d.get("transport", {}).get("real_lora_rank16", {})
for r in lr.get("rows", []):
    print("lora", r["lora_bytes"], r["num_tensors"], round(r["rtt_ms"], 1),
          round(r["client_encode_ms"], 1), round(r["client_decode_ms"], 1),
          round(r["one_way_MBps"]), r["sha256_equal"], round(r["lora_sha256_ms"], 1))
print("sha before", lr.get("lora_sha256_ms_before_send"))
for r in d.get("failure_handling", []):
    print(r)
for name, r in d.get("mock_e2e", {}).get("runs", {}).items():
    t = r["timeline"]
    print(name, r["status"], r["drift_totals"]["bitwise_mismatch_count"],
          r["drift_totals"]["token_count"], r["weight_sync"], r["final_lora_sha256"][:12],
          {k: round(v, 3) for k, v in t["mean_s"].items()})
    if t.get("bridge"):
        print("   bridge", {op: (v["count"], round(v["rtt_s_mean"] * 1e3, 1),
                              round(v["overhead_s_mean"] * 1e3, 1)) for op, v in t["bridge"].items()})
print(json.dumps(d.get("gpu_smoke"), indent=0)[:3000])
for k in d:
    if k.startswith("real_trainer"):
        print(k, json.dumps(d[k])[:600])
