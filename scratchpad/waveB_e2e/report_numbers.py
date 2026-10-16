import json
import statistics as st
import sys

d = json.load(open(sys.argv[1]))
tr = d["transport"]
print("transport commit", tr.get("commit"), "load", tr["load_average_1_5_15"])
by = {}
for r in tr["payloads"]:
    by.setdefault((r["tcp_nodelay"], r["payload_bytes"]), []).append(r["rtt_ms"])
print("| payload | TCP_NODELAY on: median RTT ms (min-max) | Nagle on: median RTT ms (min-max) |")
for s in sorted({k[1] for k in by}):
    a, b = by[(True, s)], by[(False, s)]
    print(f"| {s} | {st.median(a):.1f} ({min(a):.1f}-{max(a):.1f}) | "
          f"{st.median(b):.1f} ({min(b):.1f}-{max(b):.1f}) |")
lr = tr["real_lora_rank16"]
for nd in (True, False):
    rows = [r for r in lr["rows"] if r["tcp_nodelay"] is nd]
    rtt = [r["rtt_ms"] for r in rows]
    print("lora", nd, "median rtt", round(st.median(rtt), 1), "min", round(min(rtt), 1), "max",
          round(max(rtt), 1), "one-way MB/s median", round(st.median([r["one_way_MBps"] for r in rows])),
          "sha ok", all(r["sha256_equal"] for r in rows),
          "hash ms median", round(st.median([r["lora_sha256_ms"] for r in rows])),
          "enc", round(st.median([r["client_encode_ms"] for r in rows])),
          "dec", round(st.median([r["client_decode_ms"] for r in rows])))
print("sha before send ms", lr["lora_sha256_ms_before_send"])
gs = d["gpu_smoke"]
print("gpu", gs["commit"], gs["lease"], gs["surrogate_exactness_cuda"]["bit_exact"],
      [(v["name"], v["placement_ok"], v["all_bit_exact_vs_inprocess_cuda"], v["server_torch"])
       for v in gs["cuda_mock_trainer_through_bridge"]])
print("surrogate cpu", d["surrogate_exactness"]["cpu"])
for e in d["bridge_equivalence"]:
    print(e["name"], e["launcher"], e["server_torch"], e["all_bit_exact"], len(e["steps"]),
          round(e["startup_s"], 2))
me = d["mock_e2e"]["runs"]
for k, v in me.items():
    tl = v["timeline"]
    b = tl["bridge"] or {}
    print(k, v["drift_totals"]["bitwise_mismatch_count"], "/", v["drift_totals"]["token_count"],
          v["weight_sync"], v["final_lora_sha256"][:12], "step mean", round(st.mean(tl["step_s"]), 2),
          {op: (round(x["rtt_s_mean"] * 1e3, 1), round(x["overhead_s_mean"] * 1e3, 1)) for op, x in b.items()})
print("env cpu load", d["env"]["client_cpu_run"].get("load_average_1_5_15_at_start"))
for label, e in d["startup_overlap"].items():
    print(label, e["commit"])
    for r in e["rows"]:
        print(f"  | {r['caller_work_s']:.0f} | {r['wait_ready_blocked_s']:.2f} | {r['launch_to_ready_s']:.2f} | {r['server_build_s']:.2f} |")
print("| scenario | launcher | detect ms | client error (first line) | server alive after | usable after |")
for r in d["failure_handling"]:
    err = r.get("error", "")
    print(f"| {r['scenario']} | {r['launcher']} | {r['detect_ms']:.0f} | {err[:90]} | "
          f"{r.get('server_alive_after_error', r.get('server_gone'))} | {r.get('server_usable_after_error')} |")
