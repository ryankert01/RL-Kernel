import json
import sys

d = json.load(open(sys.argv[1]))
for p in d["profile"]:
    for kind in ("prefill", "decode"):
        e = p.get(kind)
        if not isinstance(e, dict):
            print(p["mode"], p["tp"], kind, e)
            continue
        kern = [o for o in e["top_ops"] if o.get("is_kernel")]
        top = sorted(kern, key=lambda o: -o["self_cuda_ms"])[:3]
        print(p["mode"], p["tp"], kind, "step", e["step"], "kernel_ms %.1f" % e["kernel_cuda_ms"], "nkern", e["kernels"])
        for o in top:
            print("    ", o["name"][:70], "%.1f" % o["self_cuda_ms"], o["calls"])
for t in d["throughput"]:
    steps = [s for s in t["step_series_rank0"] if s.get("reqs") == t["batch"] and s.get("tokens") == s.get("reqs")]
    ms = sorted(s["ms"] for s in steps)
    print(t["mode"], t["tp"], "decode steps", len(steps), "median %.1f" % ms[len(ms) // 2], "loadavg", t["host_loadavg_1m_before_after"], "prefill", t["prefill_steps_rank0"])
