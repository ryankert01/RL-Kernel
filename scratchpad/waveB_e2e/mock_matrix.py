"""Four-mode mock matrix (8 steps x 8 prompts x 4 samples x 64 tokens, seed 1234) at HEAD.

Run from the worktree: ~/envs/vllm/bin/python <scratch>/mock_matrix.py <worktree> <out_dir>
"""

import json
import sys
from pathlib import Path

worktree, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
sys.path.insert(0, str(worktree))

from rl_engine.models.muse_glimmer.e2e import run_poc_grpo  # noqa: E402

rows = []
for mode in ("strict/strict", "strict/native", "native/strict", "native/native"):
    run_dir = out_dir / mode.replace("/", "-")
    argv = ["--mock", "--mode", mode, "--steps", "8", "--prompts-per-step", "8",
            "--samples-per-prompt", "4", "--max-tokens", "64", "--lora-rank", "16",
            "--lr", "1e-5", "--seed", "1234", "--threads", "8", "--run-dir", str(run_dir)]
    code = run_poc_grpo.main(argv)
    s = json.loads((run_dir / "summary.json").read_text())
    steps = [json.loads(x) for x in (run_dir / "steps.jsonl").read_text().splitlines()]
    internal = sum(r["rollout_decode_vs_prefill"]["bitwise_mismatch_count"] for r in steps)
    one = sum(r["grpo"]["ratio_exactly_one_frac"] * r["drift"]["token_count"] for r in steps)
    clip = sum(r["grpo"]["clip_frac"] * r["drift"]["token_count"] for r in steps)
    tok = s["drift_totals"]["token_count"]
    t = s["timings_mean_s"]
    rows.append({
        "mode": mode, "exit": code, "status": s["status"], "commit": s["code"]["git_commit"],
        "dirty": s["code"]["git_dirty"],
        "bitwise": s["drift_totals"]["bitwise_mismatch_count"], "tokens": tok,
        "max_abs": s["drift_totals"]["max_abs_diff"],
        "mean_abs": sum(r["drift"]["mean_abs_diff"] * r["drift"]["token_count"] for r in steps) / tok,
        "k3": sum(r["drift"]["mismatch_k3_kl"] * r["drift"]["token_count"] for r in steps) / tok,
        "decode_vs_prefill": internal, "first_drift": s["first_drift"],
        "ratio_one_frac": one / tok, "clip_frac": clip / tok,
        "reward_mean": s["reward"]["mean"],
        "sync": f"{s['weight_sync']['passed']}/{s['weight_sync']['checks']}",
        "transport": s["weight_sync"]["transport"],
        "step_s": t["step_s"], "rollout_s": t["rollout_s"], "train_logp_s": t["train_logp_s"],
        "train_step_s": t["train_step_s"], "sync_s": t["sync_s"],
        "command": "python -m rl_engine.models.muse_glimmer.e2e.run_poc_grpo " + " ".join(argv),
    })
(out_dir / "matrix.json").write_text(json.dumps(rows, indent=1) + "\n")
for r in rows:
    print(json.dumps({k: v for k, v in r.items() if k != "command"}))
