import json
import re
import sys
from pathlib import Path

W = Path("/tmp/claude-1000/-home-ubuntu-RL-Kernel--claude-worktrees-swift-oak-hsv7/"
         "80655dc5-1eae-4992-a93b-f6f87e8ab1d8/scratchpad/wb")
sys.path.insert(0, "reports/muse_glimmer_poc/scripts/rope")
import update_json_wave_b as u  # noqa: E402

notes = json.loads((W / "notes.json").read_text())
review = json.loads((W / "review_ref.json").read_text())
before = u.outcomes(str(W / "logs/before_f91e8b6_full.log"))
errs = u.first_errors(str(W / "logs/before_f91e8b6_full.log"))
after = u.outcomes(str(W / "runs2/pytest_after_vllm.log"))
after_m = u.outcomes(str(W / "runs2/pytest_after_megatron.log"))
tiny = sorted(t for t in after if not t.startswith("Test"))
rows = []
for t in tiny:
    b = before.get(t, "PASSED")
    e = None if b == "PASSED" else errs.get(t)
    rows.append({"test": f"test_strict_model_tiny.py::{t}", "before_f91e8b6": b,
                 "before_failure": e, "before_failure_kind": None if e is None else u.classify(e),
                 "after_vllm_env": after.get(t, "NOT RUN"),
                 "after_megatron_env": after_m.get(t, "NOT RUN")})
review["tiny_tests"] = rows
review["tiny_tests_summary"] = {
    "count": len(rows),
    "before_failed": sum(r["before_f91e8b6"] != "PASSED" for r in rows),
    "after_passed_vllm_env": sum(r["after_vllm_env"] == "PASSED" for r in rows),
    "after_passed_megatron_env": sum(r["after_megatron_env"] == "PASSED" for r in rows),
}
real = (W / "runs/pytest_full_chain_triton.log").read_text()
review["real_weights_pytest"] = {
    "command": "MUSEGLIMMER_CHECKPOINT=~/models/Muse-Glimmer-30B MUSEGLIMMER_BACKEND=triton "
               "pytest tests/models/muse_glimmer/test_full_model_chain.py -q -s -rA",
    "commit": "39053b2",
    "passed": re.findall(r"^PASSED (\S+)", real, re.M),
    "summary": [x for x in real.splitlines() if x.strip()][-1],
}
notes["wave_b_review"] = review
(W / "notes_final.json").write_text(json.dumps(notes, indent=1))
print(json.dumps(review["tiny_tests_summary"]), review["real_weights_pytest"]["summary"])
for r in rows:
    if r["before_f91e8b6"] != "PASSED":
        print(r["test"], r["before_failure"])
