---
name: museglimmer-poc-state
description: "State of the Muse-Glimmer-30B strict train/rollout POC (RFC #501) — which branch is pushed where, envs, what is merged, how to resume"
metadata:
  node_type: memory
  type: project
  originSessionId: 80655dc5-1eae-4992-a93b-f6f87e8ab1d8
  modified: 2026-10-15T16:08:12.496Z
---

As of 2026-10-15 ~17:30 UTC the Muse-Glimmer-30B POC for RFC RL-Align/RL-Kernel#501 lives on branch
`feat/museglimmer-poc` (HEAD 78d7090), pushed to Ryan's fork `origin` = github.com/ryankert01/RL-Kernel, and
checked out in worktree `/home/ubuntu/RL-Kernel/.claude/worktrees/swift-oak-hsv7`. PRs for #501 must target
upstream `test-museglimmer`. The resume plan, agent-branch list and remaining steps are committed at
`reports/muse_glimmer_poc/RESUME.md` (it still names the old branch; content is otherwise current).

`feat/museglimmer-arch-fingerprint` (d84be03, local only) is the same history plus one line in
`.github/workflows/ci.yml` adding the fingerprint test to CI. Ryan said "don't push CI stuff for now", and the
`gh` login lacks the `workflow` scope, so the CI-free copy was made with `git filter-branch` and pushed instead.
Work on `feat/museglimmer-poc`; never push `feat/museglimmer-arch-fingerprint`.

**Why:** Ryan asked for a whole-pipeline POC (all RFC rows, real vLLM 0.31 + real Megatron Bridge, a short LoRA
GRPO run) done ASAP with many parallel agents, then a report on results and on how the RFC must change; the
session's usage limit forced a wrap-up before the integration/E2E steps finished.

**How to apply:** resume from RESUME.md, not from scratch. Envs `~/envs/vllm` and `~/envs/megatron` (both
Triton 3.7.1, transformers 5.17.0) and the checkpoint `~/models/Muse-Glimmer-30B` are ready; use
`~/bin/gpu-lease` for GPU sharing. All rows are merged (WS1 316 tests; chain, TP, CP, vLLM/Megatron bindings,
GRPO orchestrator); ablation has no code. Not yet done: post-merge re-test, vLLM/Megatron 30B runs, two-runtime
logprob byte-diff, E2E GRPO, REPORT.md sections 4.2–8. The RFC-change list comes from
`reports/muse_glimmer_poc/RFC_AUDIT_DIGEST.md` plus the "RFC feedback" sections of each
`rl_engine/models/muse_glimmer/**/REPORT_*.md`. See [[ryan-rl-kernel-workflow]].

Second session (2026-10-15 afternoon): Wave A workflow `wf_7b064961-36f` (8 owners + reviewers) running; the final
report is the HTML blog built by `reports/muse_glimmer_poc/blog/build_blog.py` from `reports/muse_glimmer_poc/data/*.json`
(see [[report-format-html-blog]]); cross-runtime byte-diff = `tools/xrt_diff.py` over `data/xrt_*.json`; paste-ready
RFC edits = `RFC_CHANGES.md` from `tools/make_rfc_changes.py`. Screenshot QA venv: scratchpad `shotenv` + `pw`.


State 2026-10-16 ~02:30 UTC: POC COMPLETE on feat/museglimmer-poc (pushed to the fork). Real 30B GRPO strict/strict 0/61139 drift (also in the RFC layout rollout TP2 + train TP4, identical LoRA trajectory); report published as private Artifact https://claude.ai/artifact/Jizn9UfL54zENiYaMki6Xf (rebuild with reports/muse_glimmer_poc/blog/build_blog.py, republish the same file). No upstream PR yet — ask before opening one (target test-museglimmer).

Upstream PR policy: Ryan said the upstream PR scope is the Foundation row only. Opened RL-Align/RL-Kernel#558 as a DRAFT (Ryan asked to keep it draft) (base test-museglimmer, head ryankert01:feat/museglimmer-foundation, 8 foundation files, no CI). Later rows go as separate PRs, only when Ryan asks.