"""Markdown table of the four-mode mock matrix for the report."""

import csv
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for mode in ("strict-strict", "strict-native", "native-strict", "native-native"):
    run = root / f"mode_{mode}"
    summary = json.loads((run / "summary.json").read_text())
    with (run / "rounds.csv").open() as handle:
        steps = list(csv.DictReader(handle))
    mean_abs = sum(float(s["mean_abs_diff"]) * int(s["num_completion_tokens"]) for s in steps)
    tokens = sum(int(s["num_completion_tokens"]) for s in steps)
    k3 = sum(float(s["mismatch_k3_kl"]) * int(s["num_completion_tokens"]) for s in steps) / tokens
    internal = sum(int(s["rollout_decode_vs_prefill_bitwise_mismatch_count"]) for s in steps)
    ratio_one = sum(float(s["ratio_exactly_one_frac"]) for s in steps) / len(steps)
    clip = sum(float(s["clip_frac"]) for s in steps) / len(steps)
    t = summary["timings_mean_s"]
    first = summary["first_drift"]
    rows.append(
        f"| {mode.replace('-', '/')} | {summary['drift_totals']['bitwise_mismatch_count']} / {tokens} "
        f"| {summary['drift_totals']['max_abs_diff']:.2e} | {mean_abs / tokens:.2e} | {k3:.2e} "
        f"| {internal} / {tokens} "
        f"| {'none' if first is None else f'(0, 0, 0) abs {first['abs_diff']:.2e}'} "
        f"| {ratio_one:.3f} / {clip:.3f} | {summary['reward']['mean']:.4f} "
        f"| {summary['weight_sync']['passed']}/{summary['weight_sync']['checks']} "
        f"| {t['step_s']:.2f} ({t['rollout_s']:.2f} / {t['train_logp_s']:.2f} / "
        f"{t['train_step_s']:.2f} / {t['sync_s']:.3f}) |"
    )
print(
    "| mode (train/rollout) | bitwise mismatches / tokens | max abs diff | mean abs diff | k3 KL "
    "| rollout decode-vs-prefill mismatches | first drift (step, sample, token) "
    "| ratio==1 frac / clip frac | reward mean | weight-sync checks "
    "| s/step (rollout / train logp / train step / sync) |"
)
print("|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|")
print("\n".join(rows))
