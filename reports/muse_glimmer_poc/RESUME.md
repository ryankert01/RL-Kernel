# Muse-Glimmer-30B POC — how to resume (state as of 2026-10-15 ~15:20 UTC)

Branch: `feat/museglimmer-poc` (pushed to the fork ryankert01/RL-Kernel; worktree `/home/ubuntu/RL-Kernel/.claude/worktrees/swift-oak-hsv7`). `feat/museglimmer-arch-fingerprint` = same history + one CI line, local only.
Target branch for the eventual PR: `test-museglimmer` (per the maintainer comment on #501).

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
- vLLM binding hand-back merged at its final HEAD e9558d2 (24 unit tests pass; engine runs (a)–(e) NOT completed — the strict TP1 launch failed on vLLM 0.31's `__init__` signature inspection, fixed in HEAD but untested). Before resuming: re-point the editable install `uv pip install --python ~/envs/vllm/bin/python --no-deps -e /home/ubuntu/RL-Kernel/.claude/worktrees/swift-oak-hsv7` (the agent installed ITS worktree), then run `run_vllm_strict.py --tp 1 --native`, `--tp 1 --strict --prefill-check --lora-zero-check`, and `--score` at TP1/TP2 per REPORT_vllm_runtime_adapter.md.
- WS2 CP hand-back merged at its final HEAD 9ee0ce3 (23 tests on the REAL kernels: CP2/CP4==CP1 bitwise fwd+bwd, global+local, T 2048–8192; TP2xCP2 == single GPU). CONTRACT DEFECT found: O5's backward rule "dK/dV accumulated over query blocks ascending" is not CP-invariant; the CP path uses 8 canonical query chunks of the full sequence combined in the O3 tree (CP1/2/4 equal) but then differs from the WS1 one-shot kernel at TP1 in the last bits of dK/dV (rel 1.6e-5). Decide: adopt the query-chunk rule inside triton/gated_attention_bwd.py (preferred) or declare the CP path canonical. Also: both WS1 attention wrappers must accept head shards (H_Q=16/H_KV=1) for TP>1.
- full_model_chain hand-back merged at its final HEAD 09eef9d — COMPLETE: real weights, 52 layers, train == prefill == 4 chunkings (incl. 2047/2048/2049 at prompt 2100) == decode, byte-equal on BOTH golden and full-Triton backends; HF-bf16 anchor max|Δlogp| 0.066 / argmax 98.9 %; strict paths closer to HF-fp32 than HF-bf16 is. Report committed at rl_engine/models/muse_glimmer/reference/REPORT_full_model_chain.md; run artifacts in the session scratchpad chain_agent/runs/.
- Megatron Bridge binding merged at its final HEAD e33d4c1 (code complete; 8/9 unit tests — `test_lm_head_tiles_tp2_equals_tp1_and_reference` fails at toy vocab, cause unknown; the 30B runs (a)–(f) NOT run; pytest is missing in ~/envs/megatron). FINDINGS: fused linear_qkv layout per KV group g = [q heads (2048) | gate heads (2048) | k (128) | v (128)], verified torch.equal on real weights; weight identity: everything equal EXCEPT `model.language_model.norm.weight` — Bridge stores w−1 in bf16 and bf16(w_meg+1) differs at 1120/6656 elements → strict path reloads the HF tensor; TP>2 fails closed (MCore replicates KV heads). Resume: REPORT_megatron_strict.md "Resume" section (TP1 `--checks identity,repeat,lora_noop,native,grad_repeat,step,bench`, then TP2).
- GRPO orchestrator merged at its final HEAD c60316d (37 CPU tests; mock strict/strict drift 0/13951, native modes ~100% mismatched, weight-sync identity 9/9; GSM8K jsonl at ~/data/gsm8k). lora_sha256 schema `muse_glimmer_lora_sha256_v1` — the vLLM/Megatron engines must match it (check their export/import). Real run command in REPORT_e2e_loop.md item 7.

## Update 2026-10-15 ~16:30 UTC: second session, Phase 0 done, Wave A running
- Post-merge regression (HEAD 78d7090): single-GPU 400 passed / 1 failed (`test_lm_head_tiles_tp2_equals_tp1_and_reference`,
  same in the megatron env: 120 passed / 1 failed); distributed 4 GPUs 85 passed / 4 skipped; TP8 (`MUSEGLIMMER_TP8=1`, 8 GPUs) 9 passed.
  pytest is now installed in ~/envs/megatron. The editable rl_engine install in both envs already points at this worktree
  (new subpackages need PYTHONPATH=<worktree>).
- Shared cross-runtime input: `reports/muse_glimmer_poc/data/xrt_tokens.json` (8 GSM8K + 1 window-crossing sequence, 933 scored
  tokens). Every runtime writes `xrt_<runtime>_<mode>_tp<k>.json` (muse_glimmer_xrt_logprobs_v1) for the byte-diff.
- NEW WORK ITEM found: the E2E orchestrator is single-process but vLLM (cu130 env) and Megatron (cu129 env) cannot share a
  process -> cross-env trainer bridge (RemoteTrainer + torchrun server, surrogate loss sum(lp*g)). Megatron TP>2 fails closed,
  so the E2E layout is rollout TP1/TP2 + train TP2, not train TP4.
- Decisions taken (user reviewed the plan): CP backward -> adopt the 8-query-chunk rule inside triton/gated_attention_bwd.py;
  attention wrappers get H_Q/H_KV parameters now; det_gemm findings verified before fixing; E2E 8 steps x 8 prompts x 4 samples
  x 256 tokens, all four modes; push to the fork after each phase; no upstream PR without asking. Final report = illustrated
  HTML blog (figures over prose).
- Wave A workflow `wf_7b064961-36f` (resume: Workflow scriptPath ...museglimmer-wave-a-wf_7b064961-36f.js, resumeFromRunId):
  owners vllm_runtime (GPU 0-1), megatron_runtime (2-3), e2e_bridge (6-7), kernels_gemm / kernels_attention / kernels_rope
  (4-5), ablation (6-7), reference_profile (6-7), review_fwd_norms (read-only); adversarial reviewer per code row.
  Findings inputs: scratchpad waveA/findings_{gemm,attention,rope}.json (35 findings from the dead wf_69e440c2-c46 review).
  Next: merge each owner branch, apply reviewer fixes, Wave B (switch bindings to per-rank heads, re-run chain + TP/CP suites,
  3-way xrt byte-diff), Wave C (E2E four modes on the whole node), Wave D (HTML blog report).
