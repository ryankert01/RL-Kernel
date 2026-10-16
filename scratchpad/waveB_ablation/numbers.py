"""Extra numbers for the report from the merged ablation.json (argv[1])."""
import json
import sys

d = json.load(open(sys.argv[1]))
print("records", len(d["records"]), "settings", len(d["settings"]))
print("baselines:", {k: (round(v["wall_ms"]), round(v["peak_mem_gib"], 1)) for k, v in d["baselines"].items()})
st = {(s["engine"], s["setting"]): s for s in d["settings"]}
hf1 = [s for s in d["analysis"]["one_factor_settings"] if s["engine"] == "hf"]
meas = [s for s in hf1 if not s.get("derived_from_stored_values") and s["n_bits_differ"]]
print("measured one-factor hf moving:", [(s["setting"], s["argmax_flips"], s["argmax_flips_unique"], s["argmax_flips_at_ties"]) for s in meas])
print("sum flips", sum(s["argmax_flips"] for s in meas), "unique", sum(s["argmax_flips_unique"] for s in meas),
      "ties", sum(s["argmax_flips_at_ties"] for s in meas))
print("KL range (measured, moving):", min(s["mean_kl_full"] for s in meas), max(s["mean_kl_full"] for s in meas))
for k in [("hf", "rollout_like_batched_decode"), ("hf", "rollout_like_batched_decode_sdpa"), ("cross", "strict_vs_hf_baseline")]:
    s = st[k]
    print(k, s["n_bits_differ"], s["max_abs_dlogp"], s["mean_abs_dlogp"], s["mean_k3"], s.get("mean_kl_full"), s.get("wall_ms"), s.get("peak_mem_gib"))
for s in d["settings"]:
    if s["engine"] == "strict":
        print("strict", s["setting"], round(s.get("wall_ms") or 0), round(s.get("peak_mem_gib") or 0, 1),
              s.get("n_bits_differ_all_copies"), s.get("n_tokens_all_copies"), s.get("max_decode_rows"), s.get("engine_steps"))
print("hf decode wall", round(st[("hf", "kv_cache_decode")]["wall_ms"]), "padded", round(st[("hf", "padded_batch9_right")]["wall_ms"]),
      "peak padded", st[("hf", "padded_batch9_right")]["peak_mem_gib"])
print("repeat", st[("hf", "repeat_immediate")]["wall_ms"])
for p in d["parts"]:
    print("part", p["key"], p["commit"][:8], p["worktree_dirty"], round(p["load_s"] or 0, 1), p["started"], p["finished"], p["status"])
b = d["analysis"]["abs_dlogp_by_baseline_prob"]
for k in ("hf/padded_batch9_right", "hf/kv_cache_decode", "hf/rope_inv_freq_cuda", "hf/softcap_fp32", "cross/strict_vs_hf_baseline"):
    print(k, {lab: (v["n"], round(v["mean_abs_dlogp"], 4)) for lab, v in b[k].items()})
lw = [r for r in d["records"] if r["setting"] == "chunked_prefill_256__vs__prefill_L_minus_1"]
print("chunk256 vs L-1 per seq", {r["sequence"]: r["n_bits_differ"] for r in lw})
print("status", d["status"], d["status_missing"])
print("selfcheck", d["packed_pass_selfcheck"])
