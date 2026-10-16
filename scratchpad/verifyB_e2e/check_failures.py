import json
import sys

d = json.load(open(sys.argv[1]))
for r in d["failure_handling"]:
    keys = {k: r.get(k) for k in ("scenario", "launcher", "detect_ms", "error_type", "server_group_gone",
                                  "bridge_retired", "server_usable_after_error", "ok", "pass")
            if k in r}
    err = str(r.get("error") or r.get("client_error") or "")[:110].replace("\n", " | ")
    print(json.dumps(keys, default=str)[:300], "|", err)
print("mock_e2e runs:", list(d["mock_e2e"]["runs"].keys()))
for name, run in d["mock_e2e"]["runs"].items():
    ws = run.get("weight_sync") or {}
    print(" ", name, {k: run.get(k) for k in ("bitwise_mismatch_count", "token_count", "final_sha256")},
          ws.get("transport"), ws.get("passed"), ws.get("checks"))
