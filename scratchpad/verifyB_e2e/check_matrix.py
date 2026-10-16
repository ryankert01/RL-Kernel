import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
for mode in ("strict-strict", "strict-native", "native-strict", "native-native"):
    s = json.loads((root / mode / "summary.json").read_text())
    dt = s["drift_totals"]
    print(mode, s["status"], dt["bitwise_mismatch_count"], "/", dt["token_count"],
          "max", f"{dt['max_abs_diff']:.2e}", "sync", s["weight_sync"]["passed"], "/",
          s["weight_sync"]["checks"], "code", (s.get("code") or {}).get("git_commit", "")[:8],
          "dirty", (s.get("code") or {}).get("git_dirty"), "first", s.get("first_drift"))
