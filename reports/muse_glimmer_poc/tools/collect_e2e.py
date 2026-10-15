#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Collect the four-mode GRPO run directories into reports/muse_glimmer_poc/data/e2e_run.json.

Usage: collect_e2e.py RUN_DIR [RUN_DIR ...] [--out PATH]
Each RUN_DIR holds the loop's steps.jsonl + summary.json (rl_engine/models/muse_glimmer/e2e/loop.py).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"


def collect(run: Path) -> dict:
    summary = json.loads((run / "summary.json").read_text())
    steps = [json.loads(l) for l in (run / "steps.jsonl").read_text().splitlines() if l.strip()]
    rows = []
    for s in steps:
        d, r, g = s.get("drift", {}), s.get("rollout_decode_vs_prefill") or {}, s.get("grpo", {})
        rows.append({
            "step": s["step"],
            "samples": s.get("num_samples"),
            "tokens": d.get("token_count", s.get("num_completion_tokens")),
            "bitwise_mismatch": d.get("bitwise_mismatch_count"),
            "max_abs_diff": d.get("max_abs_diff"),
            "mean_abs_diff": d.get("mean_abs_diff"),
            "mismatch_k3_kl": d.get("mismatch_k3_kl"),
            "first_mismatch": d.get("first_mismatch"),
            "rollout_recheck_mismatch": r.get("bitwise_mismatch_count"),
            "rollout_recheck_tokens": r.get("token_count"),
            "reward_mean": s.get("reward_mean"),
            "completion_len_mean": s.get("completion_len_mean"),
            "ratio_exactly_one_frac": g.get("ratio_exactly_one_frac"),
            "clip_frac": g.get("clip_frac"),
            "ratio_min": g.get("ratio_min"),
            "ratio_max": g.get("ratio_max"),
            "loss": g.get("loss"),
            "grad_norm": (s.get("trainer_step") or {}).get("grad_norm"),
            "sync_identity": (s.get("sync") or {}).get("identity"),
            "phases_s": (s.get("timeline") or {}).get("phases_s", {}),
            "step_s": (s.get("timings") or {}).get("step_s"),
        })
    cfg = summary.get("config", {})
    return {
        "mode": summary.get("mode"),
        "status": summary.get("status"),
        "error": summary.get("error"),
        "run_dir": str(run),
        "config": {k: cfg.get(k) for k in ("steps", "prompts_per_step", "samples_per_prompt", "max_tokens", "temperature", "lora_rank",
                                           "lr", "seed", "rollout_tp", "rollout_gpus", "train_tp", "train_gpus", "task", "split", "mock")},
        "steps_completed": summary.get("steps_completed"),
        "drift_totals": summary.get("drift_totals"),
        "first_drift": summary.get("first_drift"),
        "reward": summary.get("reward"),
        "weight_sync": summary.get("weight_sync"),
        "timings_mean_s": summary.get("timings_mean_s"),
        "phase_order": (summary.get("timeline") or {}).get("phase_order"),
        "steps": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", default=str(DATA / "e2e_run.json"))
    a = ap.parse_args()
    modes = [collect(Path(r)) for r in a.runs]
    out = {"schema": "muse_glimmer_e2e_run_v1", "modes": modes}
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    for m in modes:
        dt = m["drift_totals"] or {}
        print(f"{m['mode']:15} status={m['status']} steps={m['steps_completed']} mismatch={dt.get('bitwise_mismatch_count')}/{dt.get('token_count')} "
              f"sync={(m['weight_sync'] or {}).get('passed')}/{(m['weight_sync'] or {}).get('checks')} reward={(m['reward'] or {}).get('trajectory')}")


if __name__ == "__main__":
    main()
