"""Recompute headline numbers of the real 30B smokes from their raw run dirs."""
import json
import sys
from pathlib import Path

for d in sys.argv[1:]:
    d = Path(d)
    s = json.loads((d / "summary.json").read_text())
    steps = [json.loads(l) for l in (d / "steps.jsonl").read_text().splitlines() if l.strip()]
    print("==", d.name, s.get("mode"), s.get("status"))
    print(" drift_totals", {k: s["drift_totals"].get(k) for k in ("bitwise_mismatch_count", "token_count", "max_abs_diff")})
    print(" first_drift", s.get("first_drift"))
    print(" weight_sync", {k: s["weight_sync"].get(k) for k in ("checks", "passed", "initial_sha256", "final_sha256", "transport")})
    print(" code", s.get("code"))
    for st in steps:
        rc = st.get("rollout_recheck") or st.get("recheck")
        keys = [k for k in st.keys()]
        print(" step keys", keys)
        print(" rewards", st.get("rewards"), "grad_norm", (st.get("trainer_step") or {}).get("grad_norm"))
        for k in keys:
            if "recheck" in k:
                v = st[k]
                print(" ", k, {kk: v.get(kk) for kk in v if not isinstance(v.get(kk), (list, dict))} if isinstance(v, dict) else v)
        tl = st.get("timeline") or {}
        print(" timeline", {k: v for k, v in tl.items() if isinstance(v, (int, float))})
    # trainer server log
    for log in sorted((d / "trainer_server").glob("*")):
        print(" log", log.name)
        txt = log.read_text(errors="replace").splitlines()
        for line in txt:
            if any(w in line for w in ("start pid", "client connected", "built in", "close id", "bye")):
                print("   ", line[:200])
