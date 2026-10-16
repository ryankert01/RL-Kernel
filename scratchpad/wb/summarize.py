import json
import pathlib
import statistics
import sys

R = pathlib.Path(sys.argv[1])


def load(name):
    p = R / name
    return json.loads(p.read_text()) if p.is_file() else None


for name in ("xrt_reference_triton.json", "xrt_reference_triton_repeat.json",
             "xrt_reference_golden.json"):
    x = load(name)
    if x is None:
        continue
    c = x["checks"]
    print(f"== {name}: commit {x['commit']} all_equal {c['all_byte_equal_to_train']} "
          f"n_checks {c['n_checks']} skipped {c['n_skipped']} tokens_compared "
          f"{c['n_tokens_compared']} seconds {x.get('seconds'):.1f} peak {x.get('peak_allocated_gib'):.2f}")
    for s in c["sequences"]:
        d = s["modes"].get("decode")
        if d and "skipped" not in d:
            steps = d["decode_step_seconds"]
            print(f"   {s['id']}: len {s['length']} cs {s['completion_start']} prompt_prefill "
                  f"{d['prompt_prefill_seconds']:.2f}s decode {d['decode_steps']} steps "
                  f"{d['decode_seconds_excl_prefill']:.1f}s median/step "
                  f"{statistics.median(steps):.3f}s (total {d['seconds']:.1f}s)")
        eqs = {m: (v.get('equal_tokens'), v.get('n_tokens')) for m, v in s["modes"].items()
               if "skipped" not in v}
        bad = {m: v for m, v in eqs.items() if v[0] != v[1]}
        if bad:
            print("   MISMATCH", s["id"], bad)

a, b = load("xrt_reference_triton.json"), load("xrt_reference_triton_repeat.json")
if a and b:
    eq = sum(x == y for p, q in zip(a["sequences"], b["sequences"])
             for x, y in zip(p["logprobs_hex"], q["logprobs_hex"]))
    print("triton repeat equal:", eq)

for name in ("chain_triton_512_hf.json", "chain_triton_2100_hf.json", "chain_golden_2100.json"):
    x = load(name)
    if x is None:
        continue
    ch = x["chain"]
    print(f"== {name}: commit {x.get('commit')} all_equal {ch['all_equal']}")
    for r in ch["results"]:
        print(f"   {r['spec']['name']:<28} {r['seconds']:7.2f}s peak {r['peak_alloc_gib']:.2f}")
    for e in ch["equalities"]:
        if not e["equal"]:
            print("   NOT EQUAL", e)
    hf = x.get("hf_compare")
    if hf:
        print("   HF: logits all/compl max", round(hf["logits_all_positions"]["max_abs"], 4),
              round(hf["logits_completion_positions"]["max_abs"], 4), "logp compl max/mean",
              round(hf["logp_completion"]["max_abs"], 5), round(hf["logp_completion"]["mean_abs"], 5),
              "logp all max/mean", round(hf["logp_all"]["max_abs"], 4),
              round(hf["logp_all"]["mean_abs"], 5),
              "argmax all/compl", round(hf["argmax_agreement_all"], 4),
              round(hf["argmax_agreement_completion"], 4))

for name in ("profile_strict.json", "profile_hf.json"):
    x = load(name)
    if x is None:
        continue
    print(f"== {name}: commit {x.get('commit')}")
    for r in x["runs"]:
        kp = r.get("kernel_profile", {})
        reps = r.get("kernel_profile_repeats", {})
        print(f"   {r['phase']:<14} T={r['T']:<5} B={r['batch']:<3} clean {r['clean_total_ms_median']:8.1f} "
              f"instr {r['total_ms_median']:8.1f} gpu {kp.get('gpu_busy_ms', float('nan')):8.1f} "
              f"reps {[round(v, 1) for v in reps.get('gpu_busy_ms', [])]} peak "
              f"{r['peak_allocated_gib']:.2f} act {r['activation_peak_gib']:.3f} "
              f"prep {r.get('prepared_state_gib')} before {r['mem_allocated_before_gib']:.2f}")
    g = x.get("decode_batch_invariance")
    if g:
        print("   gate:", g.get("summary"), g.get("compare"))
