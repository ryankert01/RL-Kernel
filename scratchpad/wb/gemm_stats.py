import json
import sys

R = sys.argv[1]
s = json.load(open(f"{R}/profile_strict.json"))
h = json.load(open(f"{R}/profile_hf.json"))
flops = {512: None, 2048: None, 4096: None}
for t in flops:
    flops[t] = 2 * t * (52 * 483.9e6 + 202752 * 6656)
for name, x in (("strict", s), ("hf", h)):
    for r in x["runs"]:
        if r["phase"] not in ("prefill_train",):
            continue
        kp = r["kernel_profile"]
        c = kp["by_class_gpu_ms"]
        gemm = c["qkvg_gemm"] + c["o_gemm"] + c["ffn_gemm"] + c["lm_head_gemm"]
        non = kp["gpu_busy_ms"] - gemm
        top = kp["top_kernels"][0]
        print(f"{name} T={r['T']}: busy {kp['gpu_busy_ms']:.1f} gemm-classes {gemm:.1f} non-gemm {non:.1f} "
              f"top {top['name'][:30]} {top['ms']:.1f} ({100 * top['ms'] / kp['gpu_busy_ms']:.1f}%) "
              f"TFLOP/s(gemm classes) {flops[r['T']] / (gemm / 1e3) / 1e12:.1f} "
              f"TFLOP/s(top) {flops[r['T']] / (top['ms'] / 1e3) / 1e12:.1f}")
for r in s["runs"]:
    if r["phase"] == "prefill_paged":
        kp = r["kernel_profile"]
        top = kp["top_kernels"][0]
        print(f"strict paged T={r['T']}: top share {100 * top['ms'] / kp['gpu_busy_ms']:.1f}%")
