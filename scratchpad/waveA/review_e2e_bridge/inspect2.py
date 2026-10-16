import json
import sys

d = json.load(open(sys.argv[1]))
m = d["mock_e2e"]
print("config", m["config"])
for name, r in m["runs"].items():
    tl = r["timeline"]
    print("==", name, r["mode"], "bridge", r["trainer_bridge"], "exit", r["exit_code"], r["status"])
    print("   drift_totals", json.dumps(r["drift_totals"])[:300])
    print("   first_drift", r["first_drift"], "weight_sync", r["weight_sync"], "final", r["final_lora_sha256"][:12])
    print("   reward_traj", r["reward_trajectory"])
    print("   timeline keys", sorted(tl.keys()))
    print("   steps", tl["steps"], "profiled", tl["profiled_steps"])
    print("   step_s", [round(x, 4) for x in tl["step_s"]])
    for ph in tl["phase_order"]:
        print("     %-20s" % ph, [round(x, 4) for x in tl["phases_s"][ph]], "frac", round(tl["fraction_of_step"][ph] or 0, 3))
    if tl.get("bridge"):
        for op, b in tl["bridge"].items():
            print("   bridge", op, "n", b["count"], "rtt_ms", round(b["rtt_s_mean"] * 1e3, 2), "srv_ms",
                  round(b["server_compute_s_mean"] * 1e3, 2), "ovh_ms", round(b["overhead_s_mean"] * 1e3, 2))
    print("   mismatch", tl["bitwise_mismatch_count"], "tokens", tl["token_count"], "k3", tl["mismatch_k3_kl"], "maxabs", tl["max_abs_diff"])
    if "top_ops" in r:
        print("   top_ops keys", list(r["top_ops"]))
print("transport payloads")
for row in d["transport"]["payloads"]:
    print("  ", row["payload_bytes"], row["repeat"], round(row["rtt_ms"], 2), row["bit_exact"])
print("before_nodelay", d["transport_before_tcp_nodelay"])
print("gpu smoke")
for v in d["gpu_smoke"]["cuda_mock_trainer_through_bridge"]:
    print({k: v[k] for k in v if k != "steps"})
    for s in v["steps"]:
        print("   ", s["step"], s["logprob_bit_mismatches"], s["lora_element_mismatches"], s["grad_norm_equal"], s["lora_sha256_equal"])
print("incident", d["incident"])
print("status", d["status"])
for f in d["integration_findings"]:
    print(f["id"], "|", f["severity"], "|", f["what"][:200])
