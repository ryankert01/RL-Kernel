export const meta = {
  name: 'museglimmer-ws1-review',
  description: 'Adversarial contract review of the merged Muse Glimmer WS1 strict ops',
  phases: [
    { title: 'Review', detail: 'one reviewer per op vs the numerical contract' },
    { title: 'Verify', detail: 'refute or confirm each finding with evidence' },
    { title: 'Critic', detail: 'cross-op consistency' },
  ],
}

const REPO = '/home/ubuntu/RL-Kernel/.claude/worktrees/swift-oak-hsv7'
const CONTRACT = REPO + '/docs/contracts/muse-glimmer-30b-strict-contract.md'
const PROBES = '/tmp/claude-1000/-home-ubuntu-RL-Kernel--claude-worktrees-swift-oak-hsv7/80655dc5-1eae-4992-a93b-f6f87e8ab1d8/scratchpad/review-probes'
const COMMON = [
  'CONTEXT: ' + REPO + ' is the POC branch (feat/museglimmer-arch-fingerprint, HEAD 88a56a4) of RL-Kernel implementing RFC RL-Align/RL-Kernel#501 (Muse-Glimmer-30B strict train==rollout kernels). Seven single-GPU strict operators were just written in parallel by different engineers against ONE numerical contract: ' + CONTRACT + ' (read it fully first; the "Global rules" R1-R6 and the per-op sections are binding). Frozen model semantics: ' + REPO + '/rl_engine/models/muse_glimmer/workload.py (OFFICIAL_FROZEN_SEMANTICS) and museglimmer_manifest.json. Each op has: rl_engine/models/muse_glimmer/ops/<op>.py (golden + wrapper + autograd + trace), rl_engine/models/muse_glimmer/triton/<kernel>.py, tests/models/muse_glimmer/test_<op>.py, benchmarks/models/muse_glimmer/bench_<op>.py, and the author\'s rl_engine/models/muse_glimmer/ops/REPORT_<op>.md (self-reported deviations: check them, do not trust them).',
  'RULES: read-only on the repo (never edit, never commit, never git stash). You MAY run small GPU probes to confirm a suspicion, only via: PYTHONPATH=' + REPO + ' ~/bin/gpu-lease -n 1 -- ~/envs/vllm/bin/python <script in your own scratch dir under ' + PROBES + '/<your-label>/> ; keep each probe under 2 minutes (other agents share the GPUs). Never set CUDA_VISIBLE_DEVICES yourself. Cite file:line for everything.',
  'WHAT COUNTS AS A FINDING (severity blocking / should_fix / nit): (1) contract violations: a cast not at a declared point, a reduction whose order can depend on batch size/position/chunking/TP, autotune or shape-dependent launch constants not recorded in trace(), tl.dot without allow_tf32=False, atomics/split-K, fast-math/exp2 shortcuts not mirrored in the golden, torch.matmul/SDPA/cuBLAS inside the STRICT path (goldens may use fp64 matmul), silent fallbacks (a strict function that quietly computes something else when a sibling is missing or a shape is unsupported); (2) determinism hazards: anything that can make the same logical row produce different bytes across calls, batch compositions, sequence splits, decode vs prefill, or GPUs of the same model (non-deterministic torch ops like index_add_/scatter_add_ with duplicates, atomics inside torch reductions, Python set/dict ordering feeding kernels, uninitialized memory read under a mask); (3) tests that cannot fail: comparing an op to itself, tolerances so loose they pass anything, skipped-by-default real-shape tests, a "golden" that is the kernel, batch-invariance tests that regenerate weights per batch; (4) interface drift vs the fixed sibling interfaces named in the op docstrings/reports (names, shapes, dtypes, kwargs); (5) integration hazards for the consumers (strict reference model, vLLM 0.31 binding, Megatron Bridge binding): hidden host syncs, requirements like contiguity/alignment/T%8 not validated with a clear error, CPU-side state, module-level caches keyed wrongly (device, dtype), memory blowups at real shapes (T=4096, V=202752). Not findings: style, performance ideas, missing CUDA twin.',
].join('\n')

const OPS = [
  { key: 'det_gemm', files: 'ops/det_gemm.py ops/canonical_tree.py triton/det_gemm_tree.py tests/test_det_gemm.py ops/REPORT_det_gemm.md', section: 'O3 (+ the performance note); also check that combine_tree and rank_chunk_ids match the tree order every other op assumes and that fast_det_gemm_partial sub-trees are exactly tree nodes' },
  { key: 'norms', files: 'ops/norms.py triton/norms.py tests/test_norms.py ops/REPORT_norms.md', section: 'O1 and O2 (embedding, centered/weighted norms, fused post-norm->residual->pre-norm, residual casts, dw fold, embedding dweight path through the repo sorted-segment kernel)' },
  { key: 'qknorm_rope', files: 'ops/qknorm_rope.py triton/qknorm_rope.py tests/test_qknorm_rope.py ops/REPORT_qknorm_rope.md', section: 'O4 (weightless per-head norm, scale on Q only, rope_table definition and caching, rotation in fp32, one cast, backward)' },
  { key: 'gated_attention_fwd', files: 'ops/gated_attention.py ops/paged_kv.py triton/gated_attention_fwd.py tests/test_gated_attention_fwd.py ops/REPORT_gated_attention_fwd.md', section: 'O5 forward (block order, online-softmax update order, p->bf16, gate, masked-block no-op claim, decode tile vs prefill tile bit equality and the num_warps pin, paged path null blocks / block_table -1 / seq_lens semantics, window eviction)' },
  { key: 'gated_attention_bwd', files: 'ops/gated_attention_autograd.py ops/attention_golden_bwd.py triton/gated_attention_bwd.py tests/test_gated_attention_bwd.py ops/REPORT_gated_attention_bwd.md', section: 'O5 backward (per-group dK/dV partials and tree, dQ pass, dgate, declared casts, repeatability, batch invariance, the autograd Function binding to the forward incl. the fallback path and what trace() reports)' },
  { key: 'swiglu_ffn', files: 'ops/swiglu_ffn.py triton/swiglu.py tests/test_swiglu_ffn.py ops/REPORT_swiglu_ffn.md', section: 'O6 (silu form, casts, the FFN composition over the pluggable gemm incl. the TritonDetGemmAdapter fallback and the switch to fast_det_gemm, dW chunk rule T%8, dx fp32 sum)' },
  { key: 'lm_head_softcap_logprob', files: 'ops/lm_head_softcap_logprob.py triton/softcap_logprob.py tests/test_lm_head_softcap_logprob.py ops/REPORT_lm_head_softcap_logprob.md', section: 'O7 (tile stats order, tanh convention, padded-column exclusion, combine_tile_stats tree, fp32 logits never rounded, backward dz cast, the gemm resolver incl. the K%256 legacy fallback the integrator added, memory at T=4096)' },
]

const FINDINGS = {
  type: 'object',
  properties: {
    op: { type: 'string' },
    findings: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          id: { type: 'string' },
          severity: { type: 'string', enum: ['blocking', 'should_fix', 'nit'] },
          category: { type: 'string', enum: ['contract', 'determinism', 'test_cannot_fail', 'interface_drift', 'integration_hazard', 'self_report_wrong'] },
          file: { type: 'string' },
          line: { type: 'integer' },
          claim: { type: 'string' },
          evidence: { type: 'string' },
          failure_scenario: { type: 'string' },
          suggested_fix: { type: 'string' },
        },
        required: ['id', 'severity', 'category', 'file', 'line', 'claim', 'evidence', 'failure_scenario', 'suggested_fix'],
      },
    },
    checked_ok: { type: 'array', items: { type: 'string' } },
  },
  required: ['op', 'findings', 'checked_ok'],
}

const VERDICT = {
  type: 'object',
  properties: {
    id: { type: 'string' },
    verdict: { type: 'string', enum: ['confirmed', 'refuted', 'downgraded'] },
    severity: { type: 'string', enum: ['blocking', 'should_fix', 'nit'] },
    reason: { type: 'string' },
    probe_ran: { type: 'boolean' },
  },
  required: ['id', 'verdict', 'severity', 'reason', 'probe_ran'],
}

const CRITIC = {
  type: 'object',
  properties: {
    cross_op_issues: {
      type: 'array',
      items: {
        type: 'object',
        properties: { severity: { type: 'string', enum: ['blocking', 'should_fix', 'nit'] }, claim: { type: 'string' }, evidence: { type: 'string' }, suggested_fix: { type: 'string' } },
        required: ['severity', 'claim', 'evidence', 'suggested_fix'],
      },
    },
    assembly_ok: { type: 'boolean' },
    summary: { type: 'string' },
  },
  required: ['cross_op_issues', 'assembly_ok', 'summary'],
}

phase('Review')
const results = await pipeline(
  OPS,
  (op) => agent('You are reviewing the strict operator "' + op.key + '" (files, relative to ' + REPO + '/rl_engine/models/muse_glimmer/ and ' + REPO + '/tests/models/muse_glimmer/: ' + op.files + '). Contract section(s): ' + op.section + '.\n' + COMMON + '\nRead the contract section, then the op wrapper, the Triton kernel, the tests and the author REPORT (treat the report claims as hypotheses to check). Hunt for real findings per the definition above; be concrete (file:line, the exact code, the exact scenario that breaks). If you suspect a determinism or parity hole and can confirm it with a small GPU probe (e.g. run the op twice with different batch compositions and compare bytes), do it and include the probe output in evidence. List what you checked and found OK in checked_ok (one line each) so the integrator knows coverage. Return ids as "' + op.key + ':1", "' + op.key + ':2", ...', { label: 'review:' + op.key, phase: 'Review', schema: FINDINGS }),
  (rev, op) => {
    if (!rev) return null
    log(op.key + ': ' + rev.findings.length + ' findings -> verifying')
    return parallel(rev.findings.map((f) => () =>
      agent('Adversarially verify this code-review finding about the strict operator "' + op.key + '" in ' + REPO + '. Re-derive it from the code yourself (open ' + f.file + ' around line ' + f.line + ' and the related files: ' + op.files + '); run a small GPU probe if that is the only way to settle it. Default to "refuted" if the failure scenario cannot actually happen given how the op is called (check caller constraints in the ops/ docstrings and the contract); use "downgraded" when real but less severe than claimed.\nFINDING: ' + JSON.stringify(f) + '\n' + COMMON, { label: 'verify:' + f.id, phase: 'Verify', schema: VERDICT, effort: 'high' })
        .then((v) => ({ ...f, verdict: v }))
    )).then((verified) => ({ op: op.key, checked_ok: rev.checked_ok, findings: verified.filter(Boolean) }))
  },
)

const ops = results.filter(Boolean)
const confirmed = ops.flatMap((o) => o.findings.filter((f) => f.verdict && f.verdict.verdict !== 'refuted'))
log(confirmed.length + ' findings survive verification across ' + ops.length + ' ops')

phase('Critic')
const record = ops.map((o) => ({ op: o.op, checked_ok: o.checked_ok, surviving: o.findings.filter((f) => f.verdict && f.verdict.verdict !== 'refuted').map((f) => ({ id: f.id, severity: f.verdict.severity, claim: f.claim })) }))
const critic = await agent('You are the cross-operator consistency critic for the Muse Glimmer strict op set in ' + REPO + '. Seven ops were written by seven engineers against one contract; each was reviewed individually (record below). Your job is the seams: (a) do the ops compose into the contract "Model assembly" without hidden re-casts or layout copies: trace one decoder layer by hand through ops/norms.py -> ops/det_gemm.py (four GEMMs) -> ops/qknorm_rope.py -> ops/gated_attention.py (+ gated_attention_autograd.py) -> det_gemm (o_proj, row-parallel partials + canonical_tree) -> norms fused op -> ops/swiglu_ffn.py -> norms -> ... -> ops/lm_head_softcap_logprob.py, checking tensor shapes/dtypes/contiguity expectations at every hand-off (q (T,32,128) vs (T,4096) views, gate_logits layout, cu_seqlens/q_pos dtype int32 vs int64, positions device); (b) are the rsqrt/division/tanh/exp conventions consistent across ops where the contract requires identical arithmetic in golden and kernel (norms chose IEEE 1/sqrt; qknorm chose rsqrt; does any consumer assume they match?); (c) is the fp32 tree order identical everywhere (det_gemm combine_tree, attention bwd private tree, lm_head private tree, swiglu adapter tree, CP/TP code if present under runtime/); (d) are the fixed sibling interfaces actually identical to what consumers under rl_engine/models/muse_glimmer/{reference,runtime}/ import (if those directories exist yet); (e) anything in two REPORT_*.md files that contradicts each other or the contract. Report only real issues with file:line evidence; set assembly_ok=false only for a blocking seam.\nREVIEW RECORD: ' + JSON.stringify(record, null, 1) + '\n' + COMMON, { label: 'critic', phase: 'Critic', schema: CRITIC, effort: 'high' })

return { ops, confirmed, critic }