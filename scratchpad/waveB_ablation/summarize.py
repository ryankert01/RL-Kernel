"""Print the numbers the REPORT quotes, from the merged ablation.json (argv[1])."""
import json
import sys

d = json.load(open(sys.argv[1]))
print("status", d["status"], "commits", d["commits_all_parts"], "head", d["head_at_merge"],
      "one clean commit", d["all_parts_at_one_clean_commit"])
print("headline:", d["headline"])
print("inv_freq cpu vs cuda entries differ:", d["rope_inv_freq_cpu_vs_cuda_entries_differ"])
print("selfcheck:", json.dumps(d["packed_pass_selfcheck"]))
print("tf32 probe:", json.dumps(d["tf32_probe"]))
print()
print("| engine | setting | nf | ref | differ | seqs | max|d| | mean|d| | k3 | KL | flips (u/t) | wall |")
for s in d["settings"]:
    fl = s.get("argmax_flips")
    fls = "-" if fl is None else f"{fl} ({s.get('argmax_flips_unique')}/{s.get('argmax_flips_at_ties')})"
    kl = s.get("mean_kl_full")
    extra = ""
    if s.get("n_tokens_all_copies"):
        extra = f" all {s['n_bits_differ_all_copies']}/{s['n_tokens_all_copies']} maxrows {s.get('max_decode_rows')} steps {s.get('engine_steps')}"
    print(f"| {s['engine']} | {s['setting']} | {s.get('n_factors')} | {s.get('one_factor_ref') or s.get('reference_setting') or ''} | "
          f"{s['n_bits_differ']}/{s['n_tokens']} | {s.get('sequences_with_any_diff')} | {s['max_abs_dlogp']:.3g} | "
          f"{s['mean_abs_dlogp']:.3g} | {s['mean_k3']:.3g} | {'-' if kl is None else f'{kl:.3g}'} | {fls} | "
          f"{'-' if s.get('wall_ms') is None else round(s['wall_ms'])} |{extra}")
print()
seq_ids = [p["sequence"] for p in d["baselines"]["hf"]["per_sequence"]]
want = sys.argv[2].split(",") if len(sys.argv) > 2 else []
rows = {}
for r in d["records"]:
    rows.setdefault((r["engine"], r["setting"]), {})[r["sequence"]] = r
print("per-sequence n_bits_differ")
print("setting | " + " | ".join(s.replace("gsm8k_test_", "t").replace("gsm8k_long_window", "long") for s in seq_ids))
for (e, n), per in rows.items():
    if want and n not in want:
        continue
    print(f"{e}/{n} | " + " | ".join(str(per[s]["n_bits_differ"]) if s in per else "-" for s in seq_ids))
print()
a = d["analysis"]
print("ties:", a["hf_baseline_top1_ties"]["n"], a["hf_baseline_top1_ties"]["positions"])
lp = a["low_p_gold_tokens"]
print("low-p:", lp["status"], json.dumps(lp.get("by_bucket")))
if lp.get("low_p_tokens"):
    print(" low-p tokens in calc:", sum(t["in_calc"] for t in lp["low_p_tokens"]),
          "first tokens:", sum(t["index"] == 0 for t in lp["low_p_tokens"]))
    print(" first-token logps:", [round(t["logp"], 2) for t in lp["low_p_tokens"] if t["index"] == 0])
    print(" sample:", [(t["text"], round(t["logp"], 2)) for t in lp["low_p_tokens"][:40]])
print("buckets baseline:", {k: v for k, v in a["abs_dlogp_by_baseline_prob"].get("hf/padded_batch9_left", {}).items()})
print()
names = a["hf_pairwise_identical_tokens"]["settings"]
m = a["hf_pairwise_identical_tokens"]["matrix"]
print("pairwise identical tokens (non-trivial pairs > 25 and < 933):")
for i, x in enumerate(names):
    for j, y in enumerate(names):
        if j > i and 25 < m[i][j] < a["hf_pairwise_identical_tokens"]["n_tokens"]:
            print(f"  {x} ~ {y}: {m[i][j]}")
print()
for r in d["layer_records"]:
    L = {x["point"]: x for x in r["layers"]}
    first = r["first_point_with_diff"]
    pts = ["rope_cos", "layer_0.input_layernorm", "layer_0.self_attn.q_proj", "layer_0.self_attn.k_proj",
           "layer_0.self_attn.v_proj", "layer_0.self_attn.gate_proj", "layer_0.self_attn.o_proj",
           "layer_0.mlp.gate_proj", "layer_0.mlp.down_proj", "layer_0", "layer_2", "layer_12", "layer_25", "layer_51", "final_norm"]
    s = " ".join(f"{p.replace('layer_0.', 'L0.').replace('self_attn.', '').replace('layer_', 'L')}={L[p]['frac_elems_differ']:.3g}/{L[p]['rows_with_any_diff']}r"
                 for p in pts if p in L)
    print(f"{r['sequence']} {r['setting']}: first={first} | {s}")
