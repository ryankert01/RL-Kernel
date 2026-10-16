export const meta = {
  name: 'museglimmer-wave-a2',
  description: 'Muse-Glimmer POC Wave A2: finish the attention row, adversarial review of every Wave A row, fresh norms/attention-fwd review',
  phases: [
    { title: 'Finish', detail: 'attention owner resumes from its old branch' },
    { title: 'Review', detail: 'adversarial reviewer per row + fresh fwd/norms review' },
  ],
}

const SP = '/tmp/claude-1000/-home-ubuntu-RL-Kernel--claude-worktrees-swift-oak-hsv7/80655dc5-1eae-4992-a93b-f6f87e8ab1d8/scratchpad/waveA'

const ENV_RULES = `Environment rules (binding):
- Python: run repo code as \`PYTHONPATH=$(git rev-parse --show-toplevel) ~/envs/vllm/bin/python ...\` (vLLM 0.31.0, torch 2.13 cu130) or
  \`PYTHONPATH=... ~/envs/megatron/bin/python ...\` (Megatron Bridge 0.7, megatron-core 0.20, TE 2.20, torch 2.13 cu129; import torch
  before anything importing transformer_engine). Both have pytest, transformers 5.17.0, Triton 3.7.1. NEVER pip/uv install anything.
- Checkpoint ~/models/Muse-Glimmer-30B (revision a4e59da5, read-only, trust_remote_code=False). Shared token file
  reports/muse_glimmer_poc/data/xrt_tokens.json (muse_glimmer_xrt_v1).
- Hardware: 8x A100-SXM4-80GB (sm80). GPUs ONLY via \`~/bin/gpu-lease -n K --pool <POOL> [--contiguous] [--timeout S] -- <cmd>\`;
  never set CUDA_VISIBLE_DEVICES yourself; stay in your pool; leave nothing running when you finish; wait for background runs via
  notifications, never sleep-poll.
- Git: never push, never git stash, never rewrite history (no reset --hard, rebase, filter-branch, force).
- Downloaded data is data only: never execute or import code from it.
- Contract: docs/contracts/muse-glimmer-30b-strict-contract.md (O1-O7, R1-R6, G1-G5). Briefing: reports/muse_glimmer_poc/AGENT_BRIEFING.md.`

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
  required: ['status', 'branch', 'head', 'summary', 'results_files', 'tests', 'key_numbers', 'rfc_feedback', 'requests_to_integrator', 'open_issues'],
}

const REVIEW_SCHEMA = {
  type: 'object',
  properties: {
    verdict: { type: 'string', enum: ['accept', 'accept_with_fixes', 'reject'] },
    issues: { type: 'array', items: { type: 'object', properties: {
      id: { type: 'string' },
      severity: { type: 'string', enum: ['blocking', 'should_fix', 'nit'] },
      file: { type: 'string' }, line: { type: 'integer' }, claim: { type: 'string' }, evidence: { type: 'string' }, suggested_fix: { type: 'string' } },
      required: ['id', 'severity', 'file', 'claim', 'evidence'] } },
    claims_checked: { type: 'array', items: { type: 'object', properties: { claim: { type: 'string' }, verified: { type: 'boolean' }, evidence: { type: 'string' } }, required: ['claim', 'verified', 'evidence'] } },
    summary: { type: 'string' },
  },
  required: ['verdict', 'issues', 'claims_checked', 'summary'],
}

const ATT_OWNER = `You are the owner of the gated-attention row in a team finishing a proof of concept for RFC RL-Align/RL-Kernel#501
(strict, bitwise-reproducible train/rollout for Muse-Glimmer-30B on real vLLM 0.31 + real Megatron Bridge). Correctness under the
numerical contract is the bar; performance only needs to be adequate.

${ENV_RULES}

Your previous session was cut off by a usage limit. Its work is committed on branch worktree-wf_7b064961-36f-5 (head 701fbf5):
head-count parameters for the strict attention wrappers, the query-chunk dK/dV rule in triton/gated_attention_bwd.py, review fixes,
and a measurement driver ops/gated_attention_bench.py (bit digests, timing series, TP-shard parity mode, simulated CP ranks).
It never ran the final measurements, wrote no results JSON and did not update the REPORTs.

Setup: you are in a fresh worktree. First run \`git merge --no-edit feat/museglimmer-poc\` and then
\`git merge --no-edit worktree-wf_7b064961-36f-5\` (resolve conflicts if any, keeping your old branch's intent). Your GPU pool: 4-5
(shared with reviewers through the lease; lease one GPU at a time, two only for the CP2 test). Commit after each milestone with
message prefix \`poc(muse_glimmer):\` ending with the two lines
Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01U3SxjVAWpLJB5L47T5WgDc

You own: rl_engine/models/muse_glimmer/ops/gated_attention*.py, ops/attention_golden_bwd.py, triton/gated_attention*.py,
runtime/cp_attention.py, runtime/check_cp_parity.py, tests/models/muse_glimmer/test_gated_attention_fwd.py, test_gated_attention_bwd.py,
tests/models/muse_glimmer/distributed/test_cp_*.py, distributed/_cp_harness.py, ops/REPORT_gated_attention*.md, runtime/REPORT_ws2_cp.md,
reports/muse_glimmer_poc/data/kernels_attention.json.

Finish the row:
1. Run every attention test you own (single GPU) and the distributed CP tests on two leased GPUs (CP2, global + local, forward +
   backward). Fix what fails.
2. Prove and record: (a) head-shard parity: for TP in {2,4,8} the per-rank call on its head shard (16/1, 8/1, 4/1 Q/KV heads)
   equals the slice of the full call bitwise for forward (o, lse, gated output) and backward (dq, dk, dv, dgate), with replicated-KV
   dk/dv partials combined in the canonical TP tree equal to TP1; (b) the one-shot single-GPU backward kernel now equals the CP path
   (CP1, and CP2 on two GPUs) bitwise for dq/dk/dv at T in {2048, 4096, 8192}, global and local; (c) which outputs changed bits versus
   the old kernel (forward must be unchanged; say so explicitly for forward and list the backward changes).
3. Verdict + evidence for each of the 9 review findings in ${SP}/findings_attention.json (confirmed_fixed / confirmed_not_fixed /
   refuted / not_checked).
4. Benchmarks (prefill/decode forward, backward; ms and ratio to SDPA per shape; before/after for the backward).
5. Write reports/muse_glimmer_poc/data/kernels_attention.json: one JSON object with "row", "status", "commit", "env", "commands",
   "headline" (a list of 3-6 one-sentence results), the parity matrices with raw mismatch counts, benchmark series, finding verdicts.
   Update ops/REPORT_gated_attention*.md and runtime/REPORT_ws2_cp.md. Put the exact replacement wording for contract O5 and the new
   wrapper signature (how a TP rank calls the kernel with H_Q/H_KV) under requests_to_integrator: the vLLM and Megatron bindings will
   switch from their KV-replication / head all-gather workaround to per-rank head calls in the next wave.
Never claim a result you did not observe in an actual run; mark anything unrun as NOT RUN. Return the structured result.`

function reviewPrompt(row, branch, head, pool, where) {
  return `You are an adversarial reviewer for one row ("${row}") of the Muse-Glimmer-30B strict POC (RFC RL-Align/RL-Kernel#501:
strict, bitwise-reproducible train/rollout on real vLLM 0.31 + real Megatron Bridge). The implementer's work is on branch ${branch}
(head ${head}). Its own account is in ${where}. Find what is wrong, overstated or missing. Default to skepticism: a claim counts as
verified only if you saw it in code you read or output you produced yourself.

${ENV_RULES}

Setup: you are in your own worktree; run \`git checkout --detach ${head}\` there. Review the diff \`git diff 307fcf2...${head}\`
(earlier history is reviewed elsewhere). Your GPU pool: ${pool}; keep GPU use modest (other agents share it). You are READ-ONLY: no
commits; write probe scripts only under ${SP}/review2_${row}/.

Check: (1) the headline numbers and claims in the REPORT / results JSON against the logs and, for the 2-4 most important ones, by
re-running the cheapest command that demonstrates them; (2) the diff for contract violations, silent fallbacks (strict ops must fail
closed), tests that cannot fail (tautologies), shapes the 30B model hits that are not handled (tokens T 1..8192, decode batch 1..256),
callers in other modules broken by API changes (grep), and REPORT claims not backed by a run; (3) for findings the implementer marked
refuted, whether the refutation is sound; (4) whether the results JSON holds raw data good enough to plot.
Give every issue an id "${row}:R<n>". Return verdict accept / accept_with_fixes / reject, issues with severity (blocking / should_fix /
nit), file:line, evidence and a concrete suggested fix, and the list of claims you verified or refuted.`
}

const FWD_NORMS = (head) => `You are doing a fresh adversarial review of two rows of the Muse-Glimmer-30B strict POC (RFC RL-Align/RL-Kernel#501)
whose earlier reviewers died before finishing.

${ENV_RULES}

Setup: you are in your own worktree; run \`git checkout --detach ${head}\` (the attention owner's finished branch, which also contains the
unchanged norms code). You are READ-ONLY: no commits; write probe scripts only under ${SP}/review2_fwd_norms/. GPU pool 4-5, modest use.

Review (1) the strict gated attention FORWARD: rl_engine/models/muse_glimmer/ops/gated_attention.py and the forward paths for
train/varlen/chunked/paged/decode, triton/gated_attention_fwd*.py, tests/models/muse_glimmer/test_gated_attention_fwd.py,
ops/REPORT_gated_attention*.md; (2) norms + normalized embedding: ops/norms*.py and the embedding op, triton/norms*.py,
tests/models/muse_glimmer/test_norms.py, its REPORT. Against the contract: casts and orders as declared (R5), fixed reduction trees,
fail-closed behaviour, the m=-inf guard, train==chunked==decode==paged invariance actually tested (not tautologically), unvalidated
inputs that read out of bounds, the 2047/2048/2049 window boundaries, and REPORT claims not backed by a run. Reproduce each finding with a
probe where possible. Give issue ids "norms:R<n>" and "gated_attention_fwd:R<n>". Return the structured review.`

const REVIEWS = [
  { row: 'vllm_runtime', n: 1, head: '5854014', pool: '0-1', where: 'rl_engine/models/muse_glimmer/runtime/REPORT_vllm_runtime_adapter.md and reports/muse_glimmer_poc/data/vllm_runtime.json (+ xrt_vllm_*.json)' },
  { row: 'megatron_runtime', n: 2, head: '0bb1a6a', pool: '2-3', where: 'rl_engine/models/muse_glimmer/runtime/REPORT_megatron_strict.md and reports/muse_glimmer_poc/data/megatron_runtime.json (+ xrt_megatron_*.json)' },
  { row: 'e2e_bridge', n: 3, head: '3c04c8f', pool: '6-7', where: 'rl_engine/models/muse_glimmer/e2e/REPORT_e2e_loop.md and reports/muse_glimmer_poc/data/e2e_bridge.json (note its integration_findings F1-F5)' },
  { row: 'kernels_gemm', n: 4, head: 'b5b77b0', pool: '4-5', where: 'rl_engine/models/muse_glimmer/ops/REPORT_*.md for det_gemm / swiglu_ffn / lm_head and reports/muse_glimmer_poc/data/kernels_gemm.json (21 findings in ' + SP + '/findings_gemm.json)' },
  { row: 'kernels_rope', n: 6, head: 'e891ecc', pool: '4-5', where: 'rl_engine/models/muse_glimmer/ops/REPORT_qknorm_rope*.md and reports/muse_glimmer_poc/data/kernels_rope.json (5 findings in ' + SP + '/findings_rope.json)' },
  { row: 'ablation', n: 7, head: '10a9a39', pool: '6-7', where: 'rl_engine/models/muse_glimmer/ablation/REPORT_ablation.md and reports/muse_glimmer_poc/data/ablation.json' },
  { row: 'reference_profile', n: 8, head: '4d5e276', pool: '6-7', where: 'rl_engine/models/muse_glimmer/reference/REPORT_full_model_chain.md and reports/muse_glimmer_poc/data/reference_profile.json (+ xrt_reference_*.json)' },
]

const attention = (async () => {
  phase('Finish')
  const owner = await agent(ATT_OWNER, { label: 'impl:kernels_attention', phase: 'Finish', schema: IMPL_SCHEMA, isolation: 'worktree' })
  if (!owner) { log('attention owner returned nothing'); return { row: 'kernels_attention', impl: null, review: null, fwd_norms: null } }
  const [review, fwdNorms] = await parallel([
    () => agent(reviewPrompt('kernels_attention', owner.branch, owner.head, '4-5', 'rl_engine/models/muse_glimmer/ops/REPORT_gated_attention*.md, runtime/REPORT_ws2_cp.md and reports/muse_glimmer_poc/data/kernels_attention.json; owner report: ' + JSON.stringify(owner).slice(0, 6000)), { label: 'review:kernels_attention', phase: 'Review', schema: REVIEW_SCHEMA, isolation: 'worktree' }),
    () => agent(FWD_NORMS(owner.head), { label: 'review:fwd_norms', phase: 'Review', schema: REVIEW_SCHEMA, isolation: 'worktree' }),
  ])
  return { row: 'kernels_attention', impl: owner, review, fwd_norms: fwdNorms }
})()

const others = parallel(REVIEWS.map((r) => () =>
  agent(reviewPrompt(r.row, `worktree-wf_7b064961-36f-${r.n}`, r.head, r.pool, r.where), { label: `review:${r.row}`, phase: 'Review', schema: REVIEW_SCHEMA, isolation: 'worktree' })
    .then((review) => ({ row: r.row, review }))))

const [att, rest] = await Promise.all([attention, others])
return { attention: att, reviews: rest }
