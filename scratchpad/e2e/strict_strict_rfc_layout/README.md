# Muse-Glimmer-30B strict POC — GRPO LoRA loop (strict/strict)

Row `final_e2e_lora` of RFC #501. Orchestrator: `rl_engine.models.muse_glimmer.e2e` (one
process, no VIME, no Ray). Mode `strict/strict` = `<train>/<rollout>`.

| Setting | Value |
|---|---|
| Rollout engine | StrictRolloutEngine (TP2, GPUs [0, 1], strict=True) |
| Trainer engine | StrictTrainer (TP4, GPUs [2, 3, 4, 5], strict=True) |
| Task | gsm8k |
| Batch | 8 steps × 8 prompts × 4 samples |
| Generation | max_tokens 256, temperature 1.0 |
| LoRA / optimizer | rank 16, lr 1e-05, clip ε 0.2, KL coef 0.0 |
| Seed | 1234 |

| Metric | Value |
|---|---:|
| Status | ok |
| Steps completed | 8 |
| First drift (step, sample, token) | none (bitwise equal at every step) |
| Bitwise mismatches / completion tokens | 0 / 61139 |
| Max abs logprob diff (all steps) | 0.000e+00 |
| Reward mean first / last / all steps | 0.062 / 0.156 / 0.535 |
| Weight-sync identity checks | 9 / 9 passed |
| Mean time per step (s) | 193.96 |
| Rollout / train-logp / train-step / sync (s) | 107.28 / 14.88 / 59.96 |
| Sync (s) | 2.662 |

Files: `steps.jsonl` (one JSON object per step with rewards, advantages, GRPO stats, drift and
sync records), `rounds.csv` (one row per step), `summary.json` (this table as data plus the
formulas), `provenance.json` (both engines before and after the run), `config.json`,
`mismatch-sidecars/step*.pt` (train/rollout logprobs, masks and token ids per step; recompute
the drift metrics offline with `drift.metrics_from_sidecar`).
