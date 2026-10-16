import json
import sys

from rl_engine.models.muse_glimmer.reference.profiling import GROUPS, group_of

R = sys.argv[1]
for name in ("profile_strict.json", "profile_hf.json"):
    x = json.load(open(f"{R}/{name}"))
    print("==", name)
    for r in x["runs"]:
        kp = r["kernel_profile"]
        g = {k: 0.0 for k in GROUPS}
        for kind, row in kp["by_kind_class_gpu_ms"].items():
            for c, v in row.items():
                g[group_of(c, kind)] += v
        busy = kp["gpu_busy_ms"]
        tot = sum(g.values())
        top = kp["top_kernels"][:3]
        lab = f"{r['phase']} T={r['T']} B={r['batch']}"
        print(f"  {lab}: busy {busy:.1f} attributed {tot:.1f} launches {kp['kernel_launches']}")
        print("    " + ", ".join(f"{k} {v:.1f} ({100 * v / tot:.1f}%)" for k, v in g.items() if v > 0.05))
        for t in top:
            print(f"    top: {t['name'][:60]} calls {t['calls']} ms {t['ms']:.1f} ({100 * t['ms'] / busy:.1f}%)")
        pl = r.get("per_layer_type_ms") or {}
        bk = kp["by_kind_class_gpu_ms"]
        nl, ng = x["layers"]["local"], x["layers"]["global"]
        print(f"    gpu attention per layer: local {bk['local']['attention'] / nl:.3f} ms, "
              f"global {bk['global']['attention'] / ng:.3f} ms; wall clean {r['clean_total_ms_median']:.1f}")
