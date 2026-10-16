# Muse-Glimmer-30B strict POC — GRPO LoRA loop (native/native)

Row `final_e2e_lora` of RFC #501. Orchestrator: `rl_engine.models.muse_glimmer.e2e` (one
process, no VIME, no Ray). Mode `native/native` = `<train>/<rollout>`.

| Setting | Value |
|---|---|
| Rollout engine | StrictRolloutEngine (TP1, GPUs [4], strict=False) |
| Trainer engine | StrictTrainer(native Bridge forward) (TP2, GPUs [5, 6], strict=False) |
| Task | gsm8k |
| Batch | 1 steps × 2 prompts × 2 samples |
| Generation | max_tokens 32, temperature 1.0 |
| LoRA / optimizer | rank 16, lr 1e-05, clip ε 0.2, KL coef 0.0 |
| Seed | 1234 |

| Metric | Value |
|---|---:|
| Status | ok |
| Steps completed | 1 |
| First drift (step, sample, token) | (0, 0, 3) |Δ|=1.192e-07 |
| Bitwise mismatches / completion tokens | 57 / 128 |
| Max abs logprob diff (all steps) | 4.337e-04 |
| Reward mean first / last / all steps | 0.000 / 0.000 / 0.000 |
| Weight-sync identity checks | 2 / 2 passed |
| Mean time per step (s) | 17.77 |
| Rollout / train-logp / train-step / sync (s) | 5.62 / 4.75 / 4.99 |
| Sync (s) | 2.071 |

Files: `steps.jsonl` (one JSON object per step with rewards, advantages, GRPO stats, drift and
sync records), `rounds.csv` (one row per step), `summary.json` (this table as data plus the
formulas), `provenance.json` (both engines before and after the run), `config.json`,
`mismatch-sidecars/step*.pt` (train/rollout logprobs, masks and token ids per step; recompute
the drift metrics offline with `drift.metrics_from_sidecar`).
