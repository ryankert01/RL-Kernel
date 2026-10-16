# Muse-Glimmer-30B strict POC — GRPO LoRA loop (strict/native)

Row `final_e2e_lora` of RFC #501. Orchestrator: `rl_engine.models.muse_glimmer.e2e` (one
process, no VIME, no Ray). Mode `strict/native` = `<train>/<rollout>`.

| Setting | Value |
|---|---|
| Rollout engine | MockRolloutEngine (TP2, GPUs [0, 1], strict=False) |
| Trainer engine | MockTrainer (TP4, GPUs [4, 5, 6, 7], strict=True) |
| Task | first-token-parity |
| Batch | 1 steps × 2 prompts × 4 samples |
| Generation | max_tokens 10, temperature 1.0 |
| LoRA / optimizer | rank 4, lr 0.001, clip ε 0.2, KL coef 0.0 |
| Seed | 1234 |

| Metric | Value |
|---|---:|
| Status | ok |
| Steps completed | 1 |
| First drift (step, sample, token) | (0, 0, 0) |Δ|=2.414e-03 |
| Bitwise mismatches / completion tokens | 75 / 75 |
| Max abs logprob diff (all steps) | 1.988e-02 |
| Reward mean first / last / all steps | 0.750 / 0.750 / 0.750 |
| Weight-sync identity checks | 2 / 2 passed |
| Mean time per step (s) | 0.17 |
| Rollout / train-logp / train-step / sync (s) | 0.03 / 0.04 / 0.08 |
| Sync (s) | 0.002 |

Files: `steps.jsonl` (one JSON object per step with rewards, advantages, GRPO stats, drift and
sync records), `rounds.csv` (one row per step), `summary.json` (this table as data plus the
formulas), `provenance.json` (both engines before and after the run), `config.json`,
`mismatch-sidecars/step*.pt` (train/rollout logprobs, masks and token ids per step; recompute
the drift metrics offline with `drift.metrics_from_sidecar`).
