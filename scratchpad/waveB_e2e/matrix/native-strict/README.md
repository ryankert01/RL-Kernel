# Muse-Glimmer-30B strict POC — GRPO LoRA loop (native/strict)

Row `final_e2e_lora` of RFC #501. Orchestrator: `rl_engine.models.muse_glimmer.e2e` (one
process, no VIME, no Ray). Mode `native/strict` = `<train>/<rollout>`.

| Setting | Value |
|---|---|
| Rollout engine | MockRolloutEngine (TP1, GPUs [], strict=True) |
| Trainer engine | MockTrainer (TP2, GPUs [], strict=False) |
| Task | synthetic-arith |
| Batch | 8 steps × 8 prompts × 4 samples |
| Generation | max_tokens 64, temperature 1.0 |
| LoRA / optimizer | rank 16, lr 1e-05, clip ε 0.2, KL coef 0.0 |
| Seed | 1234 |

| Metric | Value |
|---|---:|
| Status | ok |
| Steps completed | 8 |
| First drift (step, sample, token) | (0, 0, 0) |Δ|=5.608e-04 |
| Bitwise mismatches / completion tokens | 13948 / 13951 |
| Max abs logprob diff (all steps) | 1.390e-02 |
| Reward mean first / last / all steps | 0.062 / 0.000 / 0.031 |
| Weight-sync identity checks | 9 / 9 passed |
| Mean time per step (s) | 4.60 |
| Rollout / train-logp / train-step / sync (s) | 3.12 / 0.23 / 0.05 |
| Sync (s) | 0.002 |

Files: `steps.jsonl` (one JSON object per step with rewards, advantages, GRPO stats, drift and
sync records), `rounds.csv` (one row per step), `summary.json` (this table as data plus the
formulas), `provenance.json` (both engines before and after the run), `config.json`,
`mismatch-sidecars/step*.pt` (train/rollout logprobs, masks and token ids per step; recompute
the drift metrics offline with `drift.metrics_from_sidecar`).
