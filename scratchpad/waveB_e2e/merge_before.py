"""Merge the pre-fix start-up overlap rows (run_overlap_before.py) into e2e_bridge.json."""

import json
import sys
from pathlib import Path

main_path, before_path = Path(sys.argv[1]), Path(sys.argv[2])
data = json.loads(main_path.read_text())
before = json.loads(before_path.read_text())
entry = before["startup_overlap"]["before_fix"]
entry["commit"] = "87a4cd2 (pre-fix bridge.py/trainer_server.py; mocks.py with build_delay_s)"
entry["bench_code_commit"] = "33a86ad (bench_bridge.py copied into the extracted tree)"
entry["how"] = (
    "git archive 87a4cd2 rl_engine extracted to <scratch>/tree_87a4cd2, bench_bridge.py of "
    "33a86ad copied in, then ~/envs/vllm/bin/python <scratch>/run_overlap_before.py (client and "
    "trainer server both import the extracted tree; asserted bridge.STATEFUL_OPS absent)"
)
entry["client_env"] = before["env"].get("client_cpu_run")
data["startup_overlap"]["before_fix"] = entry
data["startup_overlap"] = {k: data["startup_overlap"][k] for k in ("before_fix", "after_fix")}
cmd = before["commands"][-1]
if cmd not in data["commands"]:
    data["commands"].append(cmd)
main_path.write_text(json.dumps(data, indent=1, default=str) + "\n", encoding="utf-8")
print("merged", [r["wait_ready_blocked_s"] for r in entry["rows"]])
