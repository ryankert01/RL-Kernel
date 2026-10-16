import json
import sys

for path in sys.argv[1:]:
    d = json.load(open(path))
    tp = d["tp"]
    rows = d["checks"].get("bench", {}).get("rows", [])
    for r in rows:
        m = r["median"]

        def f(key):
            v = m.get(key)
            return "OOM" if v is None else f"{v:.0f}"

        nat_step = None
        if "native_fwd_bwd_forward_ms" in m:
            nat_step = m["native_fwd_bwd_forward_ms"] + m["native_fwd_bwd_backward_ms"]
        ratio_f = (m["strict_forward_ms"] / m["native_forward_ms"]
                   if "native_forward_ms" in m else None)
        ratio_s = m["strict_step_total_ms"] / nat_step if nat_step else None
        pk = r["peak_gib"]
        print(f"| {tp} | {r['tokens']} | {f('strict_forward_ms')} | {f('native_forward_ms')} | "
              f"{'-' if ratio_f is None else f'{ratio_f:.1f}x'} | {f('strict_step_total_ms')} | "
              f"{f('strict_recompute_fwd_bwd_ms')} | {'OOM' if nat_step is None else f'{nat_step:.0f}'} | "
              f"{'-' if ratio_s is None else f'{ratio_s:.1f}x'} | {pk.get('strict_step', 0):.1f} | "
              f"{', '.join(r['errors']) or '-'} |")
