# Muse-Glimmer-30B strict POC — GRPO LoRA loop (native/native)

Row `final_e2e_lora` of RFC #501. Orchestrator: `rl_engine.models.muse_glimmer.e2e` (one
process, no VIME, no Ray). Mode `native/native` = `<train>/<rollout>`.

| Setting | Value |
|---|---|
| Rollout engine | MockRolloutEngine (TP1, GPUs [0], strict=False) |
| Trainer engine | MockTrainer (TP2, GPUs [2, 3], strict=False) |
| Task | synthetic-arith |
| Batch | 3 steps × 4 prompts × 4 samples |
| Generation | max_tokens 24, temperature 1.0 |
| LoRA / optimizer | rank 16, lr 1e-05, clip ε 0.2, KL coef 0.0 |
| Seed | 1234 |

| Metric | Value |
|---|---:|
| Status | ok |
| Steps completed | 3 |
| First drift (step, sample, token) | (0, 0, 0) |Δ|=1.669e-03 |
| Bitwise mismatches / completion tokens | 1079 / 1079 |
| Max abs logprob diff (all steps) | 2.661e-02 |
| Reward mean first / last / all steps | 0.062 / 0.250 / 0.104 |
| Weight-sync identity checks | 4 / 4 passed |
| Mean time per step (s) | 0.21 |
| Rollout / train-logp / train-step / sync (s) | 0.08 / 0.01 / 0.09 |
| Sync (s) | 0.002 |

Files: `steps.jsonl` (one JSON object per step with rewards, advantages, GRPO stats, drift and
sync records), `rounds.csv` (one row per step), `summary.json` (this table as data plus the
formulas), `provenance.json` (both engines before and after the run), `config.json`,
`mismatch-sidecars/step*.pt` (train/rollout logprobs, masks and token ids per step; recompute
the drift metrics offline with `drift.metrics_from_sidecar`).
