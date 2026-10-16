export const meta = {
  name: 'museglimmer-wave-b',
  description: 'Muse-Glimmer POC Wave B: fix all wave A2 review issues, per-rank heads in both bindings, real TP4 runs, regenerate xrt dumps; verify each fix set',
  phases: [
    { title: 'Fix', detail: '6 owners in isolated worktrees, disjoint files' },
    { title: 'Verify', detail: 'adversarial verifier per owner' },
  ],
}

const SP = '/tmp/claude-1000/-home-ubuntu-RL-Kernel--claude-worktrees-swift-oak-hsv7/80655dc5-1eae-4992-a93b-f6f87e8ab1d8/scratchpad'

const RULES = `You are one agent in a team finishing a proof of concept for RFC RL-Align/RL-Kernel#501: strict (bitwise-reproducible)
train/rollout for Muse-Glimmer-30B with real vLLM 0.31 (rollout) and real Megatron Bridge (LoRA training), ending in a GRPO run.
Correctness under the numerical contract is the bar; performance only needs to be adequate. Read docs/contracts/muse-glimmer-30b-strict-contract.md,
reports/muse_glimmer_poc/RESUME.md (last sections) and the REPORT_*.md of the files you own.

Setup: you are in a fresh git worktree. FIRST run \`git merge --no-edit feat/museglimmer-poc\` (local branch, HEAD f91e8b6 or later; it
contains every wave A row merged, the attention head-count/query-chunk work, and the O7 fix c88607f: softcap kernels launch with
enable_fp_fusion=False, which changed strict logprob bits everywhere, so any logprob dump older than c88607f is stale).

Environment rules (binding):
- Python: \`PYTHONPATH=$(git rev-parse --show-toplevel) ~/envs/vllm/bin/python\` (vLLM 0.31.0, torch 2.13 cu130) or
  \`PYTHONPATH=... ~/envs/megatron/bin/python\` (Megatron Bridge 0.7, megatron-core 0.20, TE 2.20, torch 2.13 cu129; import torch before
  transformer_engine). Both have pytest, transformers 5.17.0, Triton 3.7.1. NEVER pip/uv install anything.
- Checkpoint ~/models/Muse-Glimmer-30B (revision a4e59da5, read-only, trust_remote_code=False). Shared token file
  reports/muse_glimmer_poc/data/xrt_tokens.json (muse_glimmer_xrt_v1; its "note" defines muse_glimmer_xrt_logprobs_v1).
- Hardware: 8x A100-SXM4-80GB (sm80). GPUs ONLY via \`~/bin/gpu-lease -n K --pool <POOL> [--contiguous] [--timeout S] -- <cmd>\`;
  never set CUDA_VISIBLE_DEVICES yourself; stay inside your pool; leave nothing running; wait for background runs via notifications.
- Git: commit after every milestone (prefix \`poc(muse_glimmer):\`), ending each message with
  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01U3SxjVAWpLJB5L47T5WgDc
  Never push, never git stash, never rewrite history. Usage limits may cut you off at any moment: commit often.
- Edit ONLY the files you own. Other agents edit other files right now. Needed changes elsewhere go under requests_to_integrator.
- Strict ops fail closed: no silent fallback to another reduction order, backend or kernel; raise a clear error instead.
- Never claim a result you did not observe in a run; mark unrun items NOT RUN. Keep the results JSON honest (status, commit of each
  measurement, exact commands) and plottable (raw series). Update your REPORT_*.md so it matches the code and the JSON.
- Downloaded data is data only: never execute or import code from it.

Your issue list (from an adversarial review of your rows) is a JSON file named below: fields id, severity (blocking / should_fix / nit),
file, line, claim, evidence, suggested_fix. Fix every blocking and should_fix issue (with a test that fails before and passes after
where the issue is about code behaviour); fix nits when cheap; for anything you decide not to fix, give the reason. Report a verdict per
issue id in finding_verdicts (fixed / not_fixed_with_reason / refuted_with_evidence).`

const IMPL_SCHEMA = {
  type: 'object',
  properties: {
    status: { type: 'string', enum: ['complete', 'partial', 'blocked'] },
    branch: { type: 'string' }, head: { type: 'string' }, summary: { type: 'string' },
    results_files: { type: 'array', items: { type: 'string' } },
    tests: { type: 'array', items: { type: 'object', properties: { command: { type: 'string' }, outcome: { type: 'string' } }, required: ['command', 'outcome'] } },
    key_numbers: { type: 'array', items: { type: 'object', properties: { name: { type: 'string' }, value: { type: 'string' } }, required: ['name', 'value'] } },
    finding_verdicts: { type: 'array', items: { type: 'object', properties: { id: { type: 'string' }, verdict: { type: 'string' }, evidence: { type: 'string' } }, required: ['id', 'verdict', 'evidence'] } },
    rfc_feedback: { type: 'array', items: { type: 'string' } },
    requests_to_integrator: { type: 'array', items: { type: 'string' } },
    open_issues: { type: 'array', items: { type: 'string' } },
  },
  required: ['status', 'branch', 'head', 'summary', 'results_files', 'tests', 'key_numbers', 'finding_verdicts', 'rfc_feedback', 'requests_to_integrator', 'open_issues'],
}

const VERIFY_SCHEMA = {
  type: 'object',
  properties: {
    verdict: { type: 'string', enum: ['accept', 'accept_with_fixes', 'reject'] },
    unresolved: { type: 'array', items: { type: 'object', properties: {
      id: { type: 'string' }, severity: { type: 'string', enum: ['blocking', 'should_fix', 'nit'] },
      file: { type: 'string' }, claim: { type: 'string' }, evidence: { type: 'string' }, suggested_fix: { type: 'string' } },
      required: ['id', 'severity', 'claim', 'evidence'] } },
    claims_checked: { type: 'array', items: { type: 'object', properties: { claim: { type: 'string' }, verified: { type: 'boolean' }, evidence: { type: 'string' } }, required: ['claim', 'verified', 'evidence'] } },
    summary: { type: 'string' },
  },
  required: ['verdict', 'unresolved', 'claims_checked', 'summary'],
}

const OWNERS = [
  {
    key: 'megatron', pool: '0-3', issues: 'issues_megatron.json',
    own: `rl_engine/models/muse_glimmer/runtime/{megatron_strict.py, megatron_lora.py, megatron_native_lora.py, megatron_weights_identity.py,
megatron_results.py, strict_kernels.py, strict_trainer.py, run_megatron_strict.py}, new runtime/megatron_*.py, tests/models/muse_glimmer/test_megatron_strict_binding.py,
runtime/REPORT_megatron_strict.md, reports/muse_glimmer_poc/data/megatron_runtime.json, reports/muse_glimmer_poc/data/xrt_megatron_*.json`,
    task: `GPU pool 0-3. Do TP1/TP2 runs on GPUs 0-2 (\`--pool 0-2\`) and lease all four (\`-n 4 --pool 0-3 --timeout 14400\`) only for the TP4 run, last.
1. Fix every issue in ${SP}/waveB/issues_megatron.json, blocking first (megatron_runtime:R1: strict mode must fail closed when any
   strict kernel is missing or broken; R3: a non-zero recompute mismatch must raise in strict mode, not be applied by AdamW).
   kernels_gemm:R1: align the LoRA token-axis weight-gradient padding with det_gemm's declared rule (T_pad = roundup(T, 256), 8 chunks of
   T_pad/8) or show why the binding's 512 rule is the one the contract should declare (either way the two must agree).
   megatron_runtime:R2: document precisely how LoRA gradients depend on micro-batch packing and make the packing rule part of the recorded
   config (the E2E uses the default); if a packing-invariant accumulation is cheap, implement it.
2. Switch the strict attention to per-rank head calls with the new head-count parameters (see ops/REPORT_gated_attention_fwd.md and
   _bwd.md wave A sections and reports/muse_glimmer_poc/data/kernels_attention.json): TP2 ranks call the kernels with 16 Q / 1 KV heads; TP4
   ranks with 8 Q / 1 KV (KV head replicated), their dK/dV fp32 partials folded in the canonical TP tree. Remove the all-gather workaround
   where the per-rank path replaces it. Bits must stay TP-invariant.
3. Re-run on the final code (record the commit): TP1 and TP2 --checks identity,xrt,repeat,lora_noop,native,grad_repeat,step,bench
   (TP2 == TP1 for logprobs, LoRA grads and the post-step LoRA hash, same packing); then a REAL TP4 run (identity, xrt, repeat,
   grad_repeat, grads and post-step hash vs TP1). If TP4 cannot work, make it fail closed again and say why.
4. Regenerate reports/muse_glimmer_poc/data/xrt_megatron_strict_tp1.json, _tp2.json, _tp4.json and xrt_megatron_native_tp1.json, _tp2.json at
   the final commit. Rewrite REPORT_megatron_strict.md and the JSON so they match (honest status per check).
The E2E (integrator) will drive StrictTrainer at TP2 through e2e/bridge.py: keep its constructor and the TrainerLike API stable.`,
  },
  {
    key: 'vllm', pool: '4-7', issues: 'issues_vllm.json',
    own: `rl_engine/models/muse_glimmer/runtime/{vllm_strict.py, vllm_plugin.py, vllm_attention_backend.py, vllm_tp_sim.py, vllm_worker_tools.py,
vllm_xrt_reference.py, vllm_native_lora.py, strict_ops.py, fallback_ops.py, strict_rollout_engine.py, run_vllm_strict.py}, new runtime/vllm_*.py,
rl_engine/integrations/engines/rollout/vllm/runtime.py (Muse Glimmer dispatch only), tests/models/muse_glimmer/test_vllm_strict_binding.py,
runtime/REPORT_vllm_runtime_adapter.md, reports/muse_glimmer_poc/data/vllm_runtime.json, reports/muse_glimmer_poc/data/xrt_vllm_*.json`,
    task: `GPU pool 4-7. Do TP1/TP2 runs on GPUs 4-6 (\`--pool 4-6\`) and lease all four (\`-n 4 --pool 4-7 --timeout 14400\`) only for TP4, last.
1. Fix every issue in ${SP}/waveB/issues_vllm.json. e2e_bridge:R4 is yours: provenance()["lora_sha256"] must be the hash RECOMPUTED by every
   TP worker (e2e/sync.py lora_sha256, schema muse_glimmer_lora_sha256_v1) on the tensors it actually loaded; raise if workers disagree;
   also expose the per-worker recomputed hashes. The E2E owner is changing e2e/sync.py to accept only this top-level value.
   vllm_runtime:R2: strict mode must fail closed on vLLM features that address KV pages in the native layout (prefix caching, KV
   offload/transfer, speculative decoding, etc.) unless proven safe.
2. Per-rank heads: switch the strict attention at TP>1 to per-rank head calls with the new head-count parameters (ops/REPORT_gated_attention_*.md
   wave A sections, reports/muse_glimmer_poc/data/kernels_attention.json): TP2 ranks 16 Q / 1 KV; TP4 ranks 8 Q / 1 KV with the KV head
   replicated. Remove the KV-replication-at-TP2 workaround where possible. Bits must stay equal to TP1.
3. Re-run on the final code (record the commit): unit tests; strict TP1/TP2 generate with --prefill-check (greedy + T=1, with and without the
   random LoRA); then the contract shapes the review says are missing: the long prompt of xrt_tokens.json gsm8k_long_window (2261 prompt
   tokens) + 256 decode tokens crossing the 2048 window with eviction, vLLM chunked prefill on (small max_num_batched_tokens), and decode
   batches of 64 and 256 short prompts: decode == prefill bitwise in strict mode, counts for native. A REAL TP4 engine run (strict generate +
   xrt scoring; must equal TP1). Throughput strict vs native at TP1/TP2 (and TP4 if it works) with 3 repeats each and the spread.
4. Regenerate reports/muse_glimmer_poc/data/xrt_vllm_strict_tp1.json, _tp2.json, _tp4.json, xrt_vllm_strict_lora_tp1.json, _lora_tp2.json (same
   random LoRA recipe and seed as before), xrt_vllm_native_tp1.json, _tp2.json, xrt_vllm_native_lora_tp1.json at the final commit.
   Rewrite REPORT_vllm_runtime_adapter.md from the JSON's report string so file and data agree.
The E2E (integrator) drives StrictRolloutEngine in-process at TP1/TP2: keep the RolloutEngineLike API stable.`,
  },
  {
    key: 'e2e', pool: '3,7', issues: 'issues_e2e.json',
    own: `rl_engine/models/muse_glimmer/e2e/* (all files), tests/models/muse_glimmer/test_e2e_loop_mock.py, test_e2e_bridge*.py,
e2e/REPORT_e2e_loop.md, reports/muse_glimmer_poc/data/e2e_bridge.json`,
    task: `Mostly CPU; GPU pool "3,7" for short checks only (one GPU at a time, a few minutes). Do NOT run the 30B.
1. Fix every issue in ${SP}/waveB/issues_e2e.json. e2e/sync.py rollout_lora_sha256 must accept ONLY the rollout's top-level
   provenance()["lora_sha256"] (the vLLM owner is making it a value recomputed on every TP worker); drop the per-worker declared fallback.
   Under torchrun, gather every rank's LoRA hash and raise if any differs. Build the trainer in the server before the client connects so
   trainer loading overlaps vLLM start-up (measure it on CPU mocks).
2. e2e_bridge:R2 (real trainer behind the bridge NOT RUN): the integrator has since run two REAL 30B one-step smokes through the bridge on the
   merged code (rollout TP1 GPU 4, trainer TP2 GPUs 5,6, 2 prompts x 2 samples x 32 tokens): strict/strict drift 0/128 tokens, weight sync 2/2,
   44.6 s per step; native/native drift 57/128 tokens (first mismatch step 0 sample 0 token 3, max |d| 4.3e-4), weight sync 2/2, 17.8 s per
   step. Run dirs: ${SP}/e2e/smoke_strict and ${SP}/e2e/smoke_native (read their summary.json / steps.jsonl and cite them).
3. Rewrite REPORT_e2e_loop.md to match the code: bridge, server, timeline, tests, the smokes above, and the integrator commands for the
   four modes with two concurrent layouts on one node: run 1 on GPUs 0-3 (rollout TP2 on 0,1 + trainer TP2 on 2,3) and run 2 on GPUs 4-7.`,
  },
  {
    key: 'attn_norms', pool: '3,7', issues: 'issues_attn_norms.json',
    own: `rl_engine/models/muse_glimmer/ops/gated_attention*.py, ops/attention_golden_bwd.py, triton/gated_attention*.py, ops/norms.py and the
normalized-embedding op file(s), triton/norms*.py, tests/models/muse_glimmer/test_gated_attention_fwd.py, test_gated_attention_bwd.py, test_norms.py,
ops/REPORT_gated_attention*.md, ops/REPORT_norms*.md, reports/muse_glimmer_poc/data/kernels_attention.json`,
    task: `GPU pool "3,7", one GPU at a time (the two runtime owners hold the other GPUs; keep each lease short).
1. Fix every issue in ${SP}/waveB/issues_attn_norms.json: fail-closed validation of cu_seqlens_q/k (cu[0]==0, monotone, last == T), max_seqlen
   vs real lengths, paged inputs (block_table entries inside the visible range, cache shapes), q_pos range; avoid new host syncs on the hot
   path where metadata already lives on the host (SequenceMeta) and document the cost of any check you add. norms:R1: the golden must use
   IEEE division (torch.div on tensors with rounding that matches div_rn), not ATen's a * (1/b) fast path. kernels_attention:R3: mark the
   local-layer SDPA baselines in the JSON as not comparable (dense-mask SDPA), keep the global ones.
2. Forward bits of the strict kernels must not change: prove it with the bench driver's digest mode against the merged base.
3. Run all tests you own; update the REPORTs and the JSON.`,
  },
  {
    key: 'rope_ref', pool: '3,7', issues: 'issues_rope_ref.json',
    own: `rl_engine/models/muse_glimmer/ops/qknorm_rope.py, triton/qknorm_rope.py, tests/models/muse_glimmer/test_qknorm_rope.py, ops/REPORT_qknorm_rope*.md,
reports/muse_glimmer_poc/data/kernels_rope.json, rl_engine/models/muse_glimmer/reference/*, tests/models/muse_glimmer/test_full_model_chain.py,
test_strict_model_tiny.py, reference/REPORT_full_model_chain.md, reports/muse_glimmer_poc/data/reference_profile.json, reports/muse_glimmer_poc/data/xrt_reference_*.json`,
    task: `GPU pool "3,7", one GPU at a time (the 30B reference needs one full GPU for ~20 min per run; keep other leases short).
1. Fix every issue in ${SP}/waveB/issues_rope_ref.json. qknorm_rope:R1/R4: the local/global choice must be explicit and checked on every
   production call path; table builders outside your files go under requests_to_integrator with the exact patch (grep for them).
   Keep the public APIs backward compatible: vLLM and Megatron bindings call them and are being edited by other agents right now.
2. Regenerate reports/muse_glimmer_poc/data/xrt_reference_triton.json at the final commit (the O7 fix changed strict bits), with the
   prefill == decode == chunked checks, and update reference_profile.json provenance (commit per measurement, stale fields removed).
   xrt_reference_golden.json: regenerate only if time allows (about 20 min); otherwise mark it as measured before c88607f.
3. Run all tests you own (and the real-weights chain test with MUSEGLIMMER_CHECKPOINT set); update REPORTs and JSON.`,
  },
  {
    key: 'ablation', pool: '3,7', issues: 'issues_ablation.json',
    own: `rl_engine/models/muse_glimmer/ablation/* (incl. REPORT_ablation.md), tests/models/muse_glimmer/test_ablation.py, reports/muse_glimmer_poc/data/ablation.json`,
    task: `GPU pool "3,7", one GPU at a time (the HF model needs about 60 GB of one GPU).
1. Fix every issue in ${SP}/waveB/issues_ablation.json. ablation:R1 (blocking): make rope_inv_freq_cuda really compute inv_freq on the GPU and
   rerun that setting (and any variant built on it). R2: compare chunked prefill against the prefill_L_minus_1 baseline (one factor at a
   time); recompute from the stored exact per-token hex where possible instead of rerunning. R3: handle argmax ties (report ties
   separately). R5: commit REPORT_ablation.md with corrected text. R7: use only the public reference API (reference/* is being edited by
   another agent; do not import private names).
2. Keep ablation.json plottable: per (factor, setting, sequence) records with exact per-token values, a settings table, analysis views.`,
  },
]

function implPrompt(o) {
  return `${RULES}\n\n## Your row: ${o.key}\nYou own: ${o.own}\nYour issue list: ${SP}/waveB/${o.issues}\n\n${o.task}\n\nReturn the structured result.`
}

function verifyPrompt(o, impl) {
  return `You are an adversarial verifier for the "${o.key}" fix set of the Muse-Glimmer-30B strict POC (RFC RL-Align/RL-Kernel#501).
The owner worked on branch ${impl.branch} (head ${impl.head}). The issues it had to fix are in ${SP}/waveB/${o.issues}. Its account follows.
For EVERY blocking and should_fix issue: decide from code you read and output you produce yourself whether it is really fixed (or soundly
refuted / justified). Re-run the owner's most important tests and 1-3 of its headline claims (cheapest command). Look for new silent
fallbacks, broken callers in other modules (grep), and REPORT/JSON statements not backed by a run.

Setup: in your own worktree run \`git checkout --detach ${impl.head}\`. Read-only: no commits; probes only under ${SP}/verifyB_${o.key}/.
Python: PYTHONPATH=<your worktree> with ~/envs/vllm/bin/python or ~/envs/megatron/bin/python; never install anything. GPUs only via
\`~/bin/gpu-lease -n K --pool ${o.pool} -- <cmd>\`, modest use, never set CUDA_VISIBLE_DEVICES yourself; 8x A100 sm80.

Return verdict accept / accept_with_fixes / reject, the list of issues that remain unresolved (with evidence and a concrete fix), and the
claims you checked.

Owner account:
${JSON.stringify(impl, null, 1).slice(0, 12000)}`
}

const results = await pipeline(
  OWNERS,
  (o) => agent(implPrompt(o), { label: `fix:${o.key}`, phase: 'Fix', schema: IMPL_SCHEMA, isolation: 'worktree' }),
  (impl, o) => {
    if (!impl) { log(`${o.key}: owner returned nothing`); return { key: o.key, impl: null, verify: null } }
    return agent(verifyPrompt(o, impl), { label: `verify:${o.key}`, phase: 'Verify', schema: VERIFY_SCHEMA, isolation: 'worktree' })
      .then((verify) => ({ key: o.key, impl, verify }))
  },
)
return results
