import json
import statistics
import sys

sys.path.insert(0, ".")
from rl_engine.models.muse_glimmer.reference.xrt_scoring import compare_xrt  # noqa: E402

R = sys.argv[1]
OLD = "reports/muse_glimmer_poc/data"


def load(p):
    return json.load(open(p))


t = load(f"{R}/xrt_reference_triton.json")
g = load(f"{R}/xrt_reference_golden.json")
h = load(f"{R}/xrt_reference_hf_bf16.json")
for name, a, b in (("triton new vs old", t, load(f"{OLD}/xrt_reference_triton.json")),
                   ("golden new vs old", g, load(f"{OLD}/xrt_reference_golden.json")),
                   ("hf new vs old", h, load(f"{OLD}/xrt_reference_hf_bf16.json")),
                   ("golden vs triton", g, t), ("triton vs hf", t, h), ("golden vs hf", g, h)):
    d = compare_xrt(a, b)
    allabs = [v for r in d["sequences"] for v in r["abs_diff"]]
    print(f"{name}: equal {d['equal_tokens']}/{d['n_tokens']} max {max(allabs):.4f} "
          f"mean {sum(allabs) / len(allabs):.5f}")
    if name == "triton vs hf":
        for r in d["sequences"]:
            print(f"   {r['id']}: max {r['max_abs_diff']:.3f} mean {r['mean_abs_diff']:.4f}")
for name, x in (("triton", t), ("golden", g)):
    c = x["checks"]
    print(f"{name}: commit {x['commit']} all {c['all_byte_equal_to_train']} n_checks {c['n_checks']} "
          f"skipped {c['n_skipped']} compared {c['n_tokens_compared']} seconds {x['seconds']:.0f} "
          f"peak {x['peak_allocated_gib']:.2f}")
    meds = []
    for s in c["sequences"]:
        d = s["modes"]["decode"]
        meds.append(statistics.median(d["decode_step_seconds"]))
        if s["id"] == "gsm8k_long_window":
            print(f"   long: prompt prefill {d['prompt_prefill_seconds']:.2f}s, {d['decode_steps']} steps "
                  f"{d['decode_seconds_excl_prefill']:.1f}s -> {d['decode_seconds_excl_prefill'] / d['decode_steps']:.3f} s/step "
                  f"(median {statistics.median(d['decode_step_seconds']):.3f}); total decode {d['seconds']:.1f}s; train {s['train_seconds']:.2f}")
        for m, v in s["modes"].items():
            if "skipped" not in v and not v["byte_equal"]:
                print("   MISMATCH", s["id"], m)
    print(f"   per-sequence median s/step: {min(meds):.3f}..{max(meds):.3f}")
hits = [x for s in h["sequences"] for x in s["argmax_is_token"]]
print("hf argmax==token", sum(hits) / len(hits))
