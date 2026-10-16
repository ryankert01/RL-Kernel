# Muse-Glimmer-30B POC (RFC #501) — session artifacts

Everything the POC session produced on the on-prem node that is not already in the code branch
`feat/museglimmer-poc` (same fork). Code, data, reports and the blog live on that branch; this branch
keeps the working material behind them.

| Path | What |
|---|---|
| `scratchpad/` | The session scratchpad: E2E run directories (incl. per-step mismatch sidecars), agent probe scripts, measurement logs and JSON, review/verify probes, the blog builder's screenshot/lint tooling scripts. |
| `workflows/` | The multi-agent workflow scripts (`*.js`) and each run's journal (`*_journal.jsonl`, one result per agent). |
| `claude_memory/` | The assistant's project memory notes for this work. |
| `machine/` | `gpu-lease` (flock GPU lease tool used by every agent), `pip freeze` of `~/envs/vllm` and `~/envs/megatron`, the Megatron env setup report, hardware summary. |
| `stale_worktree_agent-aff98d33dd3f64af6/` | Two uncommitted files of the first-session ablation agent (superseded by the wave A ablation row). |
| `patches/ci_add_fingerprint_test.patch` | The one CI change kept off the pushed branches (adds the fingerprint test to `.github/workflows/ci.yml`); local branch `feat/museglimmer-arch-fingerprint` = `feat/museglimmer-poc` history + this line. |
| `SKIPPED_MANIFEST.json` | What was left out: tensors > 25 MB (name, size, SHA-256; reproducible), code snapshots (repo commits), venvs/browsers/wheels/screenshots and third-party downloads. |

Model: `meta-models/Muse-Glimmer-30B` @ `a4e59da52a7bc87ae7251dd5545c0dd437c44b68` (Hugging Face). Data: GSM8K (public).
Paths inside files point to the node's original locations (`/tmp/claude-1000/...`, `/home/ubuntu/...`).
