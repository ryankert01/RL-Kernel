# Muse-Glimmer-30B strict POC — GRPO LoRA loop (strict/strict)

Row `final_e2e_lora` of RFC #501. Orchestrator: `rl_engine.models.muse_glimmer.e2e` (one
process, no VIME, no Ray). Mode `strict/strict` = `<train>/<rollout>`.

| Setting | Value |
|---|---|
| Rollout engine | MockRolloutEngine (TP1, GPUs [], strict=True) |
| Trainer engine | MockTrainer (TP2, GPUs [], strict=True) |
| Task | synthetic-arith |
| Batch | 4 steps × 4 prompts × 4 samples |
| Generation | max_tokens 32, temperature 1.0 |
| LoRA / optimizer | rank 4, lr 0.001, clip ε 0.2, KL coef 0.0 |
| Seed | 1234 |

| Metric | Value |
|---|---:|
| Status | ok |
| Steps completed | 4 |
| First drift (step, sample, token) | none (bitwise equal at every step) |
| Bitwise mismatches / completion tokens | 0 / 1951 |
| Max abs logprob diff (all steps) | 0.000e+00 |
| Reward mean first / last / all steps | 0.062 / 0.000 / 0.078 |
| Weight-sync identity checks | 5 / 5 passed |
| Mean time per step (s) | 9.59 |
| Rollout / train-logp / train-step / sync (s) | 8.05 / 0.29 / 0.87 |
| Sync (s) | 0.009 |

Files: `steps.jsonl` (one JSON object per step with rewards, advantages, GRPO stats, drift and
sync records), `rounds.csv` (one row per step), `summary.json` (this table as data plus the
formulas), `provenance.json` (both engines before and after the run), `config.json`,
`mismatch-sidecars/step*.pt` (train/rollout logprobs, masks and token ids per step; recompute
the drift metrics offline with `drift.metrics_from_sidecar`).
