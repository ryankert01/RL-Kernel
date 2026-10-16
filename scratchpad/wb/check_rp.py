import json

d = json.load(open("reports/muse_glimmer_poc/data/reference_profile.json"))
print(d["status"], d["collected_at_commit"])
for k, v in d["measured_commits"].items():
    print(" ", k, (v["commit"] or {}).get("sha", "")[:9], (v["commit"] or {}).get("dirty"),
          v["includes_o7_fix"])
e = d["experiments"]
print(list(e))
for k in ("chain_triton_512_hf", "chain_triton_2100_hf", "chain_golden_2100"):
    print(k, e[k]["status"], e[k].get("includes_o7_fix"))
x = e["xrt_scoring"]
print({k: (v.get("status"), v.get("n_checks"), v.get("includes_o7_fix")) for k, v in x.items()
       if isinstance(v, dict) and "status" in v})
print("repeat", e["xrt_triton_repeat"])
print("gate", e["decode_batch_invariance"]["status"], e["decode_batch_invariance"].get("compare"))
pr = e["profile"]["strict_triton"]["runs"]
print(all(r["kernel_profile"]["by_group_gpu_ms"] == r["by_group_gpu_ms"] for r in pr))
print(e["profile"]["stacked_bars"]["strict_triton"]["peak_allocated_gib"])
print(list(d["notes"]))
g = json.load(open("reports/muse_glimmer_poc/data/xrt_reference_golden.json"))
print("golden file commit", g["commit"], g.get("commit_note", "")[:60])
for n in ("triton", "triton_repeat", "hf_bf16"):
    f = json.load(open(f"reports/muse_glimmer_poc/data/xrt_reference_{n}.json"))
    print(n, f["commit"])
