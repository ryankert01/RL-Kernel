# Muse-Glimmer-30B strict POC — GRPO LoRA loop (native/native)

Row `final_e2e_lora` of RFC #501. Orchestrator: `rl_engine.models.muse_glimmer.e2e` (one
process, no VIME, no Ray). Mode `native/native` = `<train>/<rollout>`.

| Setting | Value |
|---|---|
| Rollout engine | MockRolloutEngine (TP2, GPUs [0, 1], strict=False) |
| Trainer engine | MockTrainer (TP4, GPUs [4, 5, 6, 7], strict=False) |
| Task | synthetic-arith |
| Batch | 8 steps × 8 prompts × 4 samples, max_tokens 64, T=1.0 |
| LoRA / optimizer | rank 16, lr 1e-05, clip ε 0.2, KL coef 0.0 |
| Seed | 1234 |

| Metric | Value |
|---|---:|
| Status | ok |
| Steps completed | 8 |
| First drift (step, sample, token) | (0, 0, 0) |Δ|=1.669e-03 |
| Bitwise mismatches / completion tokens | 13950 / 13950 |
| Max abs logprob diff (all steps) | 3.973e-02 |
| Reward mean first / last / all steps | 0.062 / 0.000 / 0.031 |
| Weight-sync identity checks | 9 / 9 passed |
| Mean time per step (s) | 0.42 (rollout 0.24, train logp 0.01, train step 0.08, sync 0.002) |

Files: `steps.jsonl` (one JSON object per step with rewards, advantages, GRPO stats, drift and
sync records), `rounds.csv` (one row per step), `summary.json` (this table as data plus the
formulas), `provenance.json` (both engines before and after the run), `config.json`,
`mismatch-sidecars/step*.pt` (train/rollout logprobs, masks and token ids per step; recompute
the drift metrics offline with `drift.metrics_from_sidecar`).
