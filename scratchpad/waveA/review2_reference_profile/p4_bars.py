import json, sys
D = sys.argv[1]
d = json.load(open(f"{D}/reference_profile.json"))
P = d["experiments"]["profile"]
sb = P["stacked_bars"]
print("configs", sb["configs"])
print("views", sb.get("views"))
for t in ("strict_triton", "hf_bf16_sdpa"):
    print("==", t)
    for view in ("gpu_ms", "wall_ms"):
        v = sb[t][view]
        print("  ", view)
        for g, vals in v.items():
            print(f"     {g:26s}", vals)
    for k in ("clean_total_ms", "gpu_busy_ms", "peak_allocated_gib", "activation_peak_gib", "tokens_per_s_clean", "instrumented_total_ms"):
        print("  ", k, sb[t][k])
print("strict_over_hf_clean", sb["strict_over_hf_clean"])
# raw per-run by_group_gpu_ms for strict decode B=16
for r in P["strict_triton"]["runs"]:
    if r["phase"] == "decode":
        kp = r["kernel_profile"]
        print("raw by_group_gpu_ms", r["batch"], {k: round(v, 2) for k, v in kp["by_group_gpu_ms"].items()})
        print("raw by_kind_class kv_write", {k: round(kp["by_kind_class_gpu_ms"][k]["kv_write"], 2) for k in kp["by_kind_class_gpu_ms"]})
        print("raw by_group_ms (wall)", {k: round(v, 2) for k, v in r["by_group_ms"].items()})
print("strict run keys", list(P["strict_triton"]["runs"][0].keys()))
print("strict args", P["strict_triton"].get("args"), P["strict_triton"].get("method"))
