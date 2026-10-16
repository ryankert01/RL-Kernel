# Muse-Glimmer-30B strict POC — GRPO LoRA loop (strict/native)

Row `final_e2e_lora` of RFC #501. Orchestrator: `rl_engine.models.muse_glimmer.e2e` (one
process, no VIME, no Ray). Mode `strict/native` = `<train>/<rollout>`.

| Setting | Value |
|---|---|
| Rollout engine | StrictRolloutEngine (TP1, GPUs [0], strict=False) |
| Trainer engine | StrictTrainer (TP2, GPUs [1, 2], strict=True) |
| Task | gsm8k |
| Batch | 8 steps × 8 prompts × 4 samples |
| Generation | max_tokens 256, temperature 1.0 |
| LoRA / optimizer | rank 16, lr 1e-05, clip ε 0.2, KL coef 0.0 |
| Seed | 1234 |

| Metric | Value |
|---|---:|
| Status | ok |
| Steps completed | 8 |
| First drift (step, sample, token) | (0, 0, 1) |Δ|=5.960e-07 |
| Bitwise mismatches / completion tokens | 42417 / 62097 |
| Max abs logprob diff (all steps) | 1.471e+00 |
| Reward mean first / last / all steps | 0.125 / 0.094 / 0.500 |
| Weight-sync identity checks | 9 / 9 passed |
| Mean time per step (s) | 92.98 |
| Rollout / train-logp / train-step / sync (s) | 42.83 / 10.36 / 33.02 |
| Sync (s) | 2.761 |

Files: `steps.jsonl` (one JSON object per step with rewards, advantages, GRPO stats, drift and
sync records), `rounds.csv` (one row per step), `summary.json` (this table as data plus the
formulas), `provenance.json` (both engines before and after the run), `config.json`,
`mismatch-sidecars/step*.pt` (train/rollout logprobs, masks and token ids per step; recompute
the drift metrics offline with `drift.metrics_from_sidecar`).
