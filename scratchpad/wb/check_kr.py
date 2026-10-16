import json

d = json.load(open("reports/muse_glimmer_poc/data/kernels_rope.json"))
print(d["status"])
print(json.dumps(d["tests_new"]["before_failed_by_kind"]))
wb = d["wave_b"]["tests"]
print({k: v for k, v in wb.items() if k != "per_test"})
for r in wb["per_test"]:
    print(" ", r["test"][:80], "|", r["before"], "|", (r["before_failure"] or "")[:60], "|",
          (r["before_failure_kind"] or "")[:20], "|", r["after_vllm_env"], r["after_megatron_env"])
print(json.dumps(d["tests_summary_lines"], indent=1))
p = d["experiments"]["bench_graph"]["plot"]
print(p["T"])
print([round(x, 1) for x in p["local_new_fwd_gpu_us"]])
print([round(x, 1) for x in p["local_new_fwd_eager_us"]])
print(p["local_new_fwd_GBps_T_ge_8192"])
print(d["experiments"]["bench_graph"]["all_bitwise_equal_old_vs_new"])
print(list(d["experiments"]["bench_series"]["plot"]))
print([f["id"] + ": " + f["verdict"][:60] for f in d["findings"]])
