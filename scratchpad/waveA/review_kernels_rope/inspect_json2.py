import json, sys

d = json.load(open(sys.argv[1]))
e = d["experiments"]
b = e["table_cpu_vs_cuda_per_bucket"]
print("buckets", len(b["bucket_start"]), b["bucket_start"][:3], b["bucket_start"][-1])
print("cos frac min/max", min(b["cos_mismatch_frac"]), max(b["cos_mismatch_frac"]))
print("sin frac min/max", min(b["sin_mismatch_frac"]), max(b["sin_mismatch_frac"]))
print("first buckets cos", b["cos_mismatch_frac"][:4])
tp = e["tp_shards"]["cases"]
print("tp cases", len(tp), "total mismatch", sum(c["q_hat_mismatch"] + c["k_hat_mismatch"] + c["dq_mismatch"] + c["dk_mismatch"] for c in tp))
print(sorted({(c["T"]) for c in tp}), sorted({c["tp"] for c in tp}))
bs = e["bench_series"]
print("bench_series T", bs["T"])
print("bench_series desc", bs["description"])
for k, v in bs["plot"].items():
    print(k, v)
ab = e["bench_ab_old_vs_new_kernel"]
print("ab T", ab["T"], ab["old"], ab["new"], ab["loadavg_start"], ab["loadavg_end"], ab["all_bitwise_equal"])
print(ab["note"])
for s in ab["series"]:
    print(s["T"], s["local"], s["bitwise_equal_fwd"], s["bitwise_equal_bwd"], round(s["old_fwd_ms"] * 1e3, 1), round(s["new_fwd_ms"] * 1e3, 1), round(s["old_bwd_ms"] * 1e3, 1), round(s["new_bwd_ms"] * 1e3, 1), "rounds", len(s["old_fwd_rounds_ms"]))
# native at 8192
for r in bs["after"]:
    if r["T"] in (4096, 8192) and r["local"]:
        print("after", r["T"], r["strict_fwd_kernel_ms"], r["native_hf_fwd_ms"], r["strict_fwd_api_ms"], r["strict_fwd_bwd_autograd_ms"])
for r in bs["before"]:
    if r["T"] in (4096, 8192, 64) and r["local"]:
        print("before", r["T"], r["strict_fwd_kernel_ms"], r["native_hf_fwd_ms"], r["strict_fwd_api_ms"], r["strict_fwd_bwd_autograd_ms"])
for r in bs["after"]:
    if r["T"] in (64,) :
        print("after64", r["local"], r["strict_fwd_kernel_ms"], r["strict_fwd_api_ms"], r["strict_fwd_bwd_autograd_ms"])
print(json.dumps(d["findings"][0]["bits_changed"]))
