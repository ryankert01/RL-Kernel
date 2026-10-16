import json
import sys

d = json.load(open(sys.argv[1]))
print(d["modules"])
for r in d["records"]:
    if "old_error" in r:
        print(r["T"], r["window"], "OLD ERROR", r["old_error"][:60])
        continue
    parts = []
    for pre in ("full", "preprocess", "core_same_do", "defect_old_oneshot_vs_old_cp1"):
        names = [k[len(pre) + 1:-9] for k in r if k.startswith(pre + "/") and k.endswith("_mismatch")]
        if not names:
            continue
        s = " ".join(
            f"{n}={r[f'{pre}/{n}_mismatch']}/{r[f'{pre}/{n}_numel']}"
            f"(rel{r[f'{pre}/{n}_max_rel_diff']:.1e})" for n in names
        )
        parts.append(f"[{pre}] {s}")
    print(r["T"], r["window"], " ".join(parts))
