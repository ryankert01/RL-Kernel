import json
import sys

d = json.load(open(sys.argv[1]))
p = d["experiments"]["profile"]
for tgt in ("strict_triton", "hf_bf16_sdpa"):
    r = [x for x in p[tgt]["runs"] if x["phase"] == "prefill_train" and x["T"] == 4096][0]
    kk = r["kernel_profile"]["by_kind_class_gpu_ms"]
    print(tgt, "local attn", kk["local"].get("attention"), "global attn", kk["global"].get("attention"))
    plt = r.get("per_layer_type_gpu_ms", {})
    print("  per_layer_type attention", {k: plt.get(k, {}).get("attention") for k in ("local", "global")})
    reps = r.get("kernel_profile_repeats", {})
    print("  rep keys", list(reps.keys()))
    gm = r["kernel_profile"]["by_class_gpu_ms"]
    print("  classes", {c: round(v, 1) for c, v in gm.items()})
    print("  groups", {c: round(v, 1) for c, v in r["kernel_profile"]["by_group_gpu_ms"].items()})
print("layers", p["strict_triton"].get("layers"), p["hf_bf16_sdpa"].get("layers"))
