import json, sys
D = sys.argv[1]
d = json.load(open(f"{D}/reference_profile.json"))
P = d["experiments"]["profile"]
print("profile keys", list(P.keys()))
print("configs", P["configs"])
for tgt in ("strict_triton", "hf_bf16_sdpa"):
    T = P[tgt]
    print("==", tgt, [k for k in T.keys()])
    runs = T["runs"]
    for r in runs:
        kp = r.get("kernel_profile", {})
        top = kp.get("top_kernels", [])
        print(f"  {r['phase']:14s} T={r['T']} B={r['batch']} clean={r['clean_total_ms_median']:.1f} "
              f"instr={r['total_ms_median']:.1f} ovh={r['instrumentation_overhead_frac']*100:.1f}% "
              f"peak={r['peak_allocated_gib']:.2f} before={r['mem_allocated_before_gib']:.2f} act={r['activation_peak_gib']:.2f} "
              f"busy={kp.get('gpu_busy_ms', float('nan')):.1f} launches={kp.get('kernel_launches')} matched={kp.get('launch_events_matched')} "
              f"unattr={kp.get('unattributed_ms', float('nan')):.3f}")
        if top:
            t0 = top[0]
            print(f"      top0 {t0['name'][:60]} {t0['ms']:.1f}ms calls={t0['calls']} frac_busy={t0['ms']/kp['gpu_busy_ms']*100:.1f}%")
        if kp:
            g = kp["by_group_gpu_ms"]
            tot = kp["gpu_busy_ms"]
            print("      gpu groups %:", {k: round(v / tot * 100, 1) for k, v in g.items() if v})
            gc = kp["by_kind_class_gpu_ms"]
            print("      attention local/global gpu ms:", round(gc["local"]["attention"], 1), round(gc["global"]["attention"], 1),
                  " attn_glue l/g", round(gc["local"]["attn_glue"], 1), round(gc["global"]["attn_glue"], 1))
            gem = sum(kp["by_class_gpu_ms"][c] for c in ("qkvg_gemm", "o_gemm", "ffn_gemm", "lm_head_gemm"))
            print("      gemm classes gpu ms:", round(gem, 1), {c: round(kp["by_class_gpu_ms"][c], 1) for c in ("qkvg_gemm", "o_gemm", "ffn_gemm", "lm_head_gemm", "softcap_logprob", "kv_write", "kv_evict")})
        if "by_group_ms" in r:
            print("      wall groups ms:", {k: round(v, 1) for k, v in r["by_group_ms"].items() if v})
    print("  layers", T.get("layers"))
print("stacked_bars keys", list(P["stacked_bars"].keys()))
sb = P["stacked_bars"]
for t in ("strict_triton", "hf_bf16_sdpa"):
    print(t, list(sb[t].keys()))
    for view in sb[t]:
        v = sb[t][view]
        if isinstance(v, dict):
            k0 = list(v.keys())[0]
            print("   ", view, "groups", list(v.keys()), "len", len(v[k0]) if isinstance(v[k0], (list, dict)) else v[k0])
print(json.dumps(d["experiments"]["decode_batch_invariance"], indent=0)[:1500])
print(d["notes"]["decode_setup"])
print(d["notes"]["prefill_setup"])
print(d["notes"]["profile_views"])
