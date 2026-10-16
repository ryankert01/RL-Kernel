import json
import statistics as st
import sys

d = json.load(open(sys.argv[1]))
print(sorted(d.keys()))
print("commit", d["commit"], d.get("cpu_sections_commit"))
print("surrogate", d["surrogate_exactness"]["cpu"]["bit_exact"])
for e in d["bridge_equivalence"]:
    print(e["name"], e["all_bit_exact"], round(e["startup_s"], 2), e["last_export_rank_sha256"])
tr = d["transport"]
by = {}
for r in tr["payloads"]:
    by.setdefault((r["tcp_nodelay"], r["payload_bytes"]), []).append(r["rtt_ms"])
for s in sorted({k[1] for k in by}):
    a, b = by[(True, s)], by[(False, s)]
    print(s, "nodelay med %.2f" % st.median(a), [round(x, 1) for x in a],
          "nagle med %.2f" % st.median(b), [round(x, 1) for x in b])
print("all bit exact", all(r["bit_exact"] for r in tr["payloads"]))
for r in tr["real_lora_rank16"]["rows"]:
    print("lora", r["tcp_nodelay"], r["repeat"], round(r["rtt_ms"], 1), round(r["one_way_MBps"]),
          r["sha256_equal"], round(r["lora_sha256_ms"]), round(r["client_encode_ms"]),
          round(r["client_decode_ms"]), round(r["server_ms"]))
print(tr["real_lora_rank16"]["lora_sha256_ms_before_send"])
for label, e in d.get("startup_overlap", {}).items():
    print(label, e.get("commit"))
    for r in e["rows"]:
        st_ = r["startup"] and {k: round(v, 2) for k, v in r["startup"].items()
                                if k != "launch_unix" and v is not None}
        print("  caller %.0f blocked %.2f launch_to_ready %.2f build %.2f" % (
            r["caller_work_s"], r["wait_ready_blocked_s"], r["launch_to_ready_s"],
            r["server_build_s"]), st_)
for r in d["failure_handling"]:
    print(r["scenario"], round(r["detect_ms"], 1), r.get("error", "")[:170],
          r.get("server_alive_after_error"), r.get("server_usable_after_error"),
          r.get("server_gone_after_close", r.get("server_gone")))
print("hash", d["hash_schema_check"]["all_equal"])
print(d["status"])
for x in d.get("real_e2e_smoke", []):
    print(x["mode"], x["drift_totals"]["bitwise_mismatch_count"], x["commit"][:7],
          x["trainer_server_log"]["start_to_client_connected_s"],
          x["trainer_server_log"]["start_to_trainer_built_s"], x["trainer_server_log"]["close"])
for c in d["commands"]:
    print("CMD", c[:200])
gs = d.get("gpu_smoke", {})
print("gpu", gs.get("commit"), [(v["name"], v["all_bit_exact_vs_inprocess_cuda"]) for v in gs.get("cuda_mock_trainer_through_bridge", [])])
me = d["mock_e2e"]["runs"]
for k, v in me.items():
    tl = v["timeline"]
    print(k, v["status"], v["drift_totals"]["bitwise_mismatch_count"], v["drift_totals"]["token_count"],
          v["weight_sync"], round(st.mean(tl["step_s"]), 3),
          {op: round(b["overhead_s_mean"] * 1e3, 1) for op, b in (tl["bridge"] or {}).items()})
