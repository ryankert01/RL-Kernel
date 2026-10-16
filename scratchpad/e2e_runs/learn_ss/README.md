# Muse-Glimmer-30B strict POC — GRPO LoRA loop (strict/strict)

Row `final_e2e_lora` of RFC #501. Orchestrator: `rl_engine.models.muse_glimmer.e2e` (one
process, no VIME, no Ray). Mode `strict/strict` = `<train>/<rollout>`.

| Setting | Value |
|---|---|
| Rollout engine | MockRolloutEngine (TP2, GPUs [0, 1], strict=True) |
| Trainer engine | MockTrainer (TP4, GPUs [4, 5, 6, 7], strict=True) |
| Task | synthetic-arith |
| Batch | 30 steps × 8 prompts × 4 samples, max_tokens 16, T=1.0 |
| LoRA / optimizer | rank 16, lr 0.01, clip ε 0.2, KL coef 0.0 |
| Seed | 1234 |

| Metric | Value |
|---|---:|
| Status | ok |
| Steps completed | 30 |
| First drift (step, sample, token) | none (bitwise equal at every step) |
| Bitwise mismatches / completion tokens | 0 / 14840 |
| Max abs logprob diff (all steps) | 0.000e+00 |
| Reward mean first / last / all steps | 0.000 / 0.000 / 0.004 |
| Weight-sync identity checks | 31 / 31 passed |
| Mean time per step (s) | 2.43 (rollout 1.05, train logp 0.15, train step 0.40, sync 0.002) |

Files: `steps.jsonl` (one JSON object per step with rewards, advantages, GRPO stats, drift and
sync records), `rounds.csv` (one row per step), `summary.json` (this table as data plus the
formulas), `provenance.json` (both engines before and after the run), `config.json`,
`mismatch-sidecars/step*.pt` (train/rollout logprobs, masks and token ids per step; recompute
the drift metrics offline with `drift.metrics_from_sidecar`).
