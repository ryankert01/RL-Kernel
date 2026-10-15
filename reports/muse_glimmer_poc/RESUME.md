# Muse-Glimmer-30B POC — how to resume (state as of 2026-10-15 ~15:20 UTC)

Branch: `feat/museglimmer-arch-fingerprint` (worktree `/home/ubuntu/RL-Kernel/.claude/worktrees/swift-oak-hsv7`).
Never pushed. Target branch for the eventual PR: `test-museglimmer` (per the maintainer comment on #501).

## Environments (ready)
- `~/envs/vllm`: Python 3.12, vLLM 0.31.0, torch 2.13.0+cu130, triton 3.7.1, transformers 5.17.0, pytest.
  Run repo code with `PYTHONPATH=<worktree> ~/envs/vllm/bin/python`. The compiled `rl_engine._C` is NOT built.
- `~/envs/megatron`: Python 3.12, torch 2.13.0+cu129, triton 3.7.1, transformers 5.17.0, megatron-core 0.20.0,
  TransformerEngine 2.20.2, Megatron Bridge 0.7.0 editable from `~/src/Megatron-Bridge` @ d86a5c09 (import torch before TE).
  Details: ENV_MEGATRON_SETUP.md.
- Checkpoint: `~/models/Muse-Glimmer-30B` (pinned a4e59da5, shard SHA-256 verified, `REVISION` present).
- GPU lease tool for parallel work: `~/bin/gpu-lease -n <k> [--pool 0-5|6-7] [--contiguous] -- <cmd>`.
- HF token: `~/.cache/huggingface/token` (read token; the repo is public anyway).

## Merged on the POC branch (all WS1 rows; 316 tests pass: `pytest tests/models/muse_glimmer --ignore=tests/models/muse_glimmer/distributed`)
fingerprint/loader/trace · fast tree GEMM · norms + normalized embedding · qknorm_scaled_rope · gated attention fwd (varlen/paged) · gated attention bwd · swiglu_ffn · lm_head_softcap_logprob. See MERGE_LOG.md and the per-op `rl_engine/models/muse_glimmer/ops/REPORT_*.md`.

## Agent branches NOT yet merged (work saved in their worktrees even if the agent was cut off)
| Row | Branch (worktree under /home/ubuntu/RL-Kernel/.claude/worktrees/) | Expected files |
|---|---|---|
| full_model_chain (strict reference + real-weights harness + HF anchor) | `worktree-agent-a960b7764e809556b` | rl_engine/models/muse_glimmer/reference/*, tests/models/muse_glimmer/test_strict_model_tiny.py, test_full_model_chain.py |
| WS2 TP rows (tp_canonical_tree, vocab_parallel_softcap_logp, kv_head_replication) | `worktree-agent-a00b53d30b29892a9` | rl_engine/models/muse_glimmer/runtime/tp_*.py, vocab_parallel_softcap_logp.py, kv_replication.py, tests/models/muse_glimmer/distributed/test_tp_*.py |
| WS2 CP rows (cp_global/local_attention, TP×CP) | `worktree-agent-a215e705260dd6ffe` | rl_engine/models/muse_glimmer/runtime/cp_attention.py, tests/models/muse_glimmer/distributed/test_cp_*.py |
| vllm_runtime_adapter (vLLM 0.31 strict binding, StrictRolloutEngine) | `worktree-agent-a7944ab684b3629cf` | rl_engine/models/muse_glimmer/runtime/vllm_*.py, strict_rollout_engine.py, run_vllm_strict.py (+ one dispatch line in rl_engine/integrations/engines/rollout/vllm/runtime.py; it may have `uv pip install -e` its worktree into ~/envs/vllm — re-point to the POC worktree after merging) |
| Megatron Bridge strict binding + LoRA + weight_sync_identity (StrictTrainer) | `worktree-agent-a0a497dd82b29ff26` | rl_engine/models/muse_glimmer/runtime/megatron_*.py, strict_trainer.py, run_megatron_strict.py |
| final_e2e_lora orchestrator (GRPO loop, mocks, drift metrics) | `worktree-agent-a7f90c5ca29b03916` | rl_engine/models/muse_glimmer/e2e/{data,grpo,sync,drift,loop,mocks,run_poc_grpo}.py |
| museglimmer_ablation (one-factor probes on HF model) | `worktree-agent-aff98d33dd3f64af6` | rl_engine/models/muse_glimmer/e2e/ablation_hf.py, run_ablation.py |

Merge each with `git merge --no-edit <branch>` from the POC worktree (the branches were created from `main` and merged the POC branch themselves; conflicts are only possible on `tests/models/muse_glimmer/distributed/__init__.py` and `e2e/__init__.py`, both SPDX-only).

## Reviews
- RFC audit workflow (56 verified edits): RFC_AUDIT_DIGEST.md.
- Adversarial WS1 code review workflow was running at wrap-up: run id `wf_69e440c2-c46`, journal
  `/home/ubuntu/.claude/projects/-home-ubuntu-RL-Kernel--claude-worktrees-swift-oak-hsv7/80655dc5-1eae-4992-a93b-f6f87e8ab1d8/subagents/workflows/wf_69e440c2-c46/journal.jsonl`
  (one `{"type":"result"}` line per finished reviewer/verifier; apply surviving findings before the E2E).

## Remaining steps, in order
1. Merge the seven branches above; rerun `pytest tests/models/muse_glimmer` (single GPU) and the distributed tests with
   `~/bin/gpu-lease -n 4 --contiguous --pool 0-7` ; run TP8/CP tests on the idle node (`MUSEGLIMMER_TP8=1`).
2. Real-weights chain: `rl_engine/models/muse_glimmer/reference/run_chain.py --checkpoint ~/models/Muse-Glimmer-30B --backend triton --hf-compare`.
3. Strict logprob dumps from both runtimes on the same token ids (`run_vllm_strict.py`, `run_megatron_strict.py`) and byte-diff them (`logprobs_hex`).
4. E2E: `rl_engine/models/muse_glimmer/e2e/run_poc_grpo.py --rollout-tp 2 --rollout-gpus 0,1 --train-tp 4 --train-gpus 4,5,6,7 --mode strict/strict ...` (needs the whole node idle; stop other leases first), then the other three modes.
5. Fill REPORT.md sections 4.2–8 from the agent reports, MERGE_LOG.md, bench_ws1.log and the ablation table; compile section 6 from RFC_AUDIT_DIGEST.md + the "RFC feedback" sections of every REPORT_*.md.

## Update 2026-10-15 ~15:30 UTC (wrap-up at the usage-limit reset)
All seven agent branches except `ablation` (no commits yet) were merged into the POC branch in their then-current
state (some agents were still committing; re-run `git merge --no-edit <branch>` for each to pick up late commits).
Modules import; the per-row tests, the real-weights chain run, TP/CP multi-GPU tests and the E2E have NOT been
re-run by the integrator after these merges — do that first when resuming (step 1 above).
- WS2 TP hand-back merged at its final HEAD 5816b5a: TP2/TP4==TP1 bitwise for tp_canonical_tree and vocab_parallel_softcap_logp on NCCL ranks; NCCL all_reduce control diverges at TP4 (tree is load-bearing). OPEN FINDING: ops/gated_attention.py strict forward pins 32 Q / 2 KV heads, so a TP-sharded rank cannot call it — add head-count parameters to the kernel (or zero-pad shards to full geometry) before the vLLM/Megatron TP>1 runs. TP8 tests still unrun (MUSEGLIMMER_TP8=1).
