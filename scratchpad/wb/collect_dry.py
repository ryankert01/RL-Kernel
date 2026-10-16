import json
import pathlib
import sys

from rl_engine.models.muse_glimmer.reference import collect_results as cr

S = pathlib.Path(
    "/tmp/claude-1000/-home-ubuntu-RL-Kernel--claude-worktrees-swift-oak-hsv7/"
    "80655dc5-1eae-4992-a93b-f6f87e8ab1d8/scratchpad"
)
runs = sys.argv[1]
out = S / "wb" / "collect_dry"
out.mkdir(parents=True, exist_ok=True)
cr.DATA = out
cr.main(["--runs", runs, "--out", str(out / "rp.json")])
d = json.loads((out / "rp.json").read_text())
print(list(d))
for k, v in d["measured_commits"].items():
    print(k, v)
for r in d["experiments"]["profile"]["strict_triton"].get("runs", []):
    kp = r.get("kernel_profile", {})
    print(r["config"], kp.get("by_group_gpu_ms") == r.get("by_group_gpu_ms"),
          r.get("activation_peak_gib"), r.get("prepared_state_gib"))
