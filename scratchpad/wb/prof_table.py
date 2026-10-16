import json
import sys

R = sys.argv[1]
s = json.load(open(f"{R}/profile_strict.json"))
h = json.load(open(f"{R}/profile_hf.json"))


def key(r):
    return (r["phase"], r["T"], r["batch"])


hr = {key(r): r for r in h["runs"]}
print("| config | strict clean ms | HF clean ms | strict/HF | strict GPU-busy ms (3 profiler runs) | HF GPU-busy ms (3 runs) | strict peak GiB | HF peak GiB |")
print("|---|---|---|---|---|---|---|---|")
for r in s["runs"]:
    k = key(r)
    o = hr.get(k)
    reps = r["kernel_profile_repeats"]["gpu_busy_ms"]
    lab = (f"decode B={r['batch']}, ctx 2048 (per step)" if r["phase"] == "decode"
           else f"{'forward' if r['phase'] == 'prefill_train' else 'paged prefill'} T={r['T']}")
    if o:
        oreps = o["kernel_profile_repeats"]["gpu_busy_ms"]
        print(f"| {lab} | {r['clean_total_ms_median']:.1f} | {o['clean_total_ms_median']:.1f} | "
              f"{r['clean_total_ms_median'] / o['clean_total_ms_median']:.2f} | "
              f"{min(reps):.1f}–{max(reps):.1f} | {min(oreps):.1f}–{max(oreps):.1f} | "
              f"{r['peak_allocated_gib']:.2f} | {o['peak_allocated_gib']:.2f} |")
    else:
        print(f"| {lab} (strict only) | {r['clean_total_ms_median']:.1f} | – | – | "
              f"{min(reps):.1f}–{max(reps):.1f} | – | {r['peak_allocated_gib']:.2f} "
              f"(one KV cache {r['prepared_state_gib']:.3f}) | – |")
for r in s["runs"]:
    if r["phase"] == "decode":
        print(r["batch"], "wall kv_cache group",
              round(r["by_group_ms"]["kv_cache"] if "by_group_ms" in r else -1, 1),
              {k: round(v, 1) for k, v in r["by_group_ms"].items() if v > 1})
print("instr overhead strict", [round(r["instrumentation_overhead_frac"] * 100, 1) for r in s["runs"]])
print("instr overhead hf", [round(r["instrumentation_overhead_frac"] * 100, 1) for r in h["runs"]])
print("weights", s["weights_gib"], h["weights_gib"])
print("launches", [r["kernel_profile"]["kernel_launches"] for r in s["runs"]],
      [r["kernel_profile"]["kernel_launches"] for r in h["runs"]])
