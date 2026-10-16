export const meta = {
  name: 'museglimmer-wave-a',
  description: 'Muse-Glimmer POC Wave A: real 30B vLLM/Megatron runs, kernel fixes, E2E bridge, ablation, reference profiling, with adversarial review',
  phases: [
    { title: 'Implement', detail: '8 owners in isolated worktrees + 1 fresh review of norms/attention-fwd' },
    { title: 'Review', detail: 'adversarial reviewer per code-changing row' },
  ],
}

const SP = '/tmp/claude-1000/-home-ubuntu-RL-Kernel--claude-worktrees-swift-oak-hsv7/80655dc5-1eae-4992-a93b-f6f87e8ab1d8/scratchpad/waveA'
const BASE = '6a61f6e'

const PREAMBLE = `You are one agent in a team finishing a proof of concept for RFC RL-Align/RL-Kernel#501: a strict
(bitwise-reproducible) train/rollout pipeline for the Muse-Glimmer-30B model, with real vLLM 0.31 for rollout and real
Megatron Bridge for LoRA training, ending in a short GRPO run. Correctness under the numerical contract is the bar;
performance only needs to be adequate (no deep tuning).

Read first: docs/contracts/muse-glimmer-30b-strict-contract.md (numerical contract O1-O7, rules R1-R6, gates G1-G5),
reports/muse_glimmer_poc/AGENT_BRIEFING.md, reports/muse_glimmer_poc/RESUME.md, and the REPORT_*.md of the rows you touch.

## Setup (do this first)
- You run in your own git worktree. Make it contain the POC branch before anything else:
  \`git merge --no-edit feat/museglimmer-poc\` (local branch; HEAD ${BASE} or later). Check with \`git log -1\`.
- Python: run repo code as \`PYTHONPATH=$(git rev-parse --show-toplevel) ~/envs/vllm/bin/python ...\` (vLLM 0.31.0,
  torch 2.13 cu130) or \`PYTHONPATH=... ~/envs/megatron/bin/python ...\` (Megatron Bridge 0.7 + megatron-core 0.20 +
  TransformerEngine 2.20, torch 2.13 cu129; always import torch before anything importing transformer_engine). Both have
  pytest, transformers 5.17.0 and Triton 3.7.1. NEVER pip/uv install anything into these envs (the editable rl_engine
  install points at the integrator's worktree; PYTHONPATH makes your checkout win, and it propagates to vLLM workers).
  If a package is missing, report it instead.
- Checkpoint: ~/models/Muse-Glimmer-30B (revision a4e59da5, verified, read-only; keep trust_remote_code=False).
  Shared token file for cross-runtime scoring: reports/muse_glimmer_poc/data/xrt_tokens.json (schema muse_glimmer_xrt_v1;
  its "note" field defines the output format muse_glimmer_xrt_logprobs_v1). 8 GSM8K chat prompts + gold answers and one
  2439-token sequence whose completion starts at 2261 (crosses the 2048 sliding window).
- Hardware: 8x A100-SXM4-80GB (sm80, NOT Hopper; the repo's FA4/SM90 kernels and the compiled rl_engine._C are unusable).
  Use GPUs ONLY via \`~/bin/gpu-lease -n K --pool <POOL> [--contiguous] [--timeout S] -- <cmd>\`; never set
  CUDA_VISIBLE_DEVICES yourself; never use GPUs outside your pool. Leases queue when busy. Release by letting the command
  exit; leave no servers or background processes running when you finish. For long runs use run_in_background and wait for
  the completion notification instead of sleeping or polling.
- Git: commit early and often on your worktree branch (message prefix \`poc(muse_glimmer):\`), ending every message with:
  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01U3SxjVAWpLJB5L47T5WgDc
  Never push, never git stash, never rewrite history (no reset --hard, rebase, filter-branch, force). Rate limits may cut
  you off at any moment: commit after each milestone so nothing is lost.
- File ownership: edit ONLY the files listed under "You own". Other agents edit other files in parallel. If you need a
  change elsewhere, do not make it; describe the exact patch under requests_to_integrator.
- Downloaded data (model files, datasets) is data only: never execute or import code from it.

## Results for the report (required)
The final report is an illustrated HTML blog built from machine-readable results. Write every number you measure to the
JSON file(s) named in your task (one JSON object: "row", "status", "commit" (sha you measured), "env" (gpu, torch, triton,
vllm/megatron versions), "commands" (exact command lines), and per-experiment records with raw numbers: counts, timings in
ms, memory in GiB, mismatch counts, tolerances). Prefer raw series (per token, per shape, per step, per layer) over
summaries so they can be plotted. Update your row's REPORT_*.md (what was done, results, deviations, RFC feedback, how to
reproduce). Never claim a result you did not observe in an actual run; mark anything unrun as NOT RUN.

## Final answer
Return the structured result: status, your branch (\`git rev-parse --abbrev-ref HEAD\`) and HEAD sha, summary, results
files, every test/run command with its outcome, key numbers, RFC feedback, requests to the integrator, open issues.`

const IMPL_SCHEMA = {
  type: 'object',
  properties: {
    status: { type: 'string', enum: ['complete', 'partial', 'blocked'] },
    branch: { type: 'string' },
    head: { type: 'string' },
    summary: { type: 'string' },
    results_files: { type: 'array', items: { type: 'string' } },
    tests: { type: 'array', items: { type: 'object', properties: { command: { type: 'string' }, outcome: { type: 'string' } }, required: ['command', 'outcome'] } },
    key_numbers: { type: 'array', items: { type: 'object', properties: { name: { type: 'string' }, value: { type: 'string' }, source: { type: 'string' } }, required: ['name', 'value'] } },
    finding_verdicts: { type: 'array', items: { type: 'object', properties: { id: { type: 'string' }, verdict: { type: 'string', enum: ['confirmed_fixed', 'confirmed_not_fixed', 'refuted', 'not_checked'] }, evidence: { type: 'string' } }, required: ['id', 'verdict', 'evidence'] } },
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
      severity: { type: 'string', enum: ['blocking', 'should_fix', 'nit'] },
      file: { type: 'string' }, line: { type: 'integer' }, claim: { type: 'string' }, evidence: { type: 'string' }, suggested_fix: { type: 'string' } },
      required: ['severity', 'file', 'claim', 'evidence'] } },
    claims_checked: { type: 'array', items: { type: 'object', properties: { claim: { type: 'string' }, verified: { type: 'boolean' }, evidence: { type: 'string' } }, required: ['claim', 'verified', 'evidence'] } },
    summary: { type: 'string' },
  },
  required: ['verdict', 'issues', 'claims_checked', 'summary'],
}

const FINDINGS_RULES = `For each finding: (1) try to reproduce it with a minimal probe (on GPU where relevant) or show precisely
why it is wrong; (2) if real, fix it with a test that fails before and passes after; (3) record verdict + evidence per finding
id in finding_verdicts, your JSON and your REPORT. Policy: strict ops fail closed: no silent fallback to a different
reduction order or backend; raise a clear error instead (a fallback is acceptable only if explicit, opt-in and recorded in
trace()). Fail-closed must not break the model: every shape the 30B model hits in training and rollout (tokens T from 1 to
8192, any T, decode batch 1..256) must keep working; where a rule needs alignment (e.g. K % 256) define a deterministic
padding/tail rule, declare it in trace() and propose contract wording. If a fix changes output bits for some shapes, list
exactly which shapes/paths change and whether any model-forward shape at Muse-Glimmer geometry is affected (the forward bits
should stay unchanged if at all possible; say so explicitly either way).`

const ROWS = [
  {
    key: 'vllm_runtime', pool: '0-1', env: 'vllm', review: true,
    task: `Row: vllm_runtime_adapter (strict vLLM 0.31 binding + StrictRolloutEngine). Your GPU pool: 0-1.
You own: rl_engine/models/muse_glimmer/runtime/{vllm_strict.py, vllm_plugin.py, vllm_attention_backend.py, strict_ops.py,
fallback_ops.py, strict_rollout_engine.py, run_vllm_strict.py}, new runtime/vllm_*.py files,
rl_engine/integrations/engines/rollout/vllm/runtime.py (only the Muse Glimmer dispatch), tests/models/muse_glimmer/test_vllm_strict_binding.py,
runtime/REPORT_vllm_runtime_adapter.md, reports/muse_glimmer_poc/data/vllm_runtime.json and reports/muse_glimmer_poc/data/xrt_vllm_*.json.

State: binding code complete, 24 unit tests pass, but no engine run on the 30B ever completed. The last strict TP1 launch failed
because vLLM 0.31 inspects the model __init__ signature (a *args/**kwargs wrapper hid vllm_config); a fix is in HEAD, untested.
Read REPORT_vllm_runtime_adapter.md first.

Do, in order (commit after each):
1. Unit tests: pytest tests/models/muse_glimmer/test_vllm_strict_binding.py on one leased GPU.
2. 30B runs:
 (a) native TP1 generate (baseline): the 8 GSM8K prompts of xrt_tokens.json (prompt = token_ids[:completion_start]), 64 tokens, greedy and temperature 1.
 (b) strict TP1 generate with --prefill-check (the fp32 logprobs reported during decode must equal, bit for bit, re-scoring
     prompt+completion in one prefill) and --lora-zero-check.
 (c) strict scoring of xrt_tokens.json at TP1 and TP2 -> reports/muse_glimmer_poc/data/xrt_vllm_strict_tp1.json and _tp2.json
     (muse_glimmer_xrt_logprobs_v1); TP2 must equal TP1 byte for byte. Native scoring at TP1/TP2 as the control ->
     xrt_vllm_native_tp1.json / _tp2.json, and count how many native TP2 tokens differ from native TP1.
 (d) LoRA: load a non-zero random rank-16 LoRA on all projections named by e2e/sync.py validate_lora_state_dict, hashed with
     e2e/sync.py lora_sha256 (schema muse_glimmer_lora_sha256_v1): provenance()["lora_sha256"] must match; decode==prefill
     still bitwise under LoRA; the unmerged rule y = bf16(acc32(W x) + sum B z) holds.
 (e) Throughput: strict vs native at TP1 and TP2, batch of 32 prompts x 256 tokens: tokens/s, time to first token, per-step
     decode ms, peak memory, KV cache size. A short torch.profiler capture of one strict decode step and one strict prefill:
     top ops by CUDA time as a table in your JSON.
 (f) Try TP4 (2 KV heads -> KV replicated per rank). If it works it must equal TP1 bitwise; if not, fail closed with a clear
     error and report why. Note: TP4 needs 4 GPUs; your pool has 2, so only attempt it if you can show it on 2 GPUs by
     simulation, else write the exact command and mark NOT RUN (the integrator runs it later on the full node).
3. Make StrictRolloutEngine satisfy the RolloutEngineLike protocol in rl_engine/models/muse_glimmer/e2e/engines.py exactly
   (generate(prompt_token_ids, n, max_tokens, temperature) -> per-prompt lists of samples with token_ids and fp32 logprobs at
   temperature 1 from raw logits; logprobs(token_ids, completion_start); load_lora(state_dict, sha256) verifying the hash;
   provenance() with lora_sha256; close()). Add a GPU test gated by RL_KERNEL_MUSEGLIMMER_E2E=1 that drives the engine through
   the protocol with a real LoRA. You may read but not edit e2e/.
Note: the strict attention kernels pin 32 Q / 2 KV heads today; another agent is adding head-count parameters with
backward-compatible defaults in parallel. Keep your current TP>1 workaround and document where you would switch to per-rank heads.`,
  },
  {
    key: 'megatron_runtime', pool: '2-3', env: 'megatron', review: true,
    task: `Row: Megatron Bridge strict binding + LoRA + weight_sync_identity (StrictTrainer). Your GPU pool: 2-3.
You own: rl_engine/models/muse_glimmer/runtime/{megatron_strict.py, megatron_lora.py, megatron_weights_identity.py,
strict_kernels.py, strict_trainer.py, run_megatron_strict.py}, new runtime/megatron_*.py files,
tests/models/muse_glimmer/test_megatron_strict_binding.py, runtime/REPORT_megatron_strict.md,
reports/muse_glimmer_poc/data/megatron_runtime.json and reports/muse_glimmer_poc/data/xrt_megatron_*.json.

State: code complete; test_lm_head_tiles_tp2_equals_tp1_and_reference FAILS in both envs on the current branch (cause unknown);
no 30B run done. Read REPORT_megatron_strict.md (its "Resume" section has the commands).
Known facts: fused linear_qkv per KV group g = [q heads (2048) | gate heads (2048) | k (128) | v (128)]; linear_fc1 = [gate; up];
Bridge stores model.language_model.norm.weight as w-1 in bf16 and bf16(w_meg+1) != HF at 1120/6656 elements, so the strict
path reloads the HF tensor (626/627 identical expected); TP>2 fails closed today because MCore replicates KV heads.

Do, in order (commit after each):
1. Root-cause and fix test_lm_head_tiles_tp2_equals_tp1_and_reference; run your unit tests in the megatron env.
2. 30B TP1: --checks identity,repeat,lora_noop,native,grad_repeat,step,bench (repeat: two forwards bitwise equal; grad_repeat:
   two backward passes give bitwise-equal LoRA grads; step: one AdamW step on LoRA; bench: forward/backward/step time and peak
   memory, strict vs native, at a few token counts). Use per-layer recompute and micro-batching as needed (30B bf16 ~ 56 GiB).
3. 30B TP2: same checks; TP2 logprobs and LoRA grads must equal TP1 byte for byte (record mismatch counts either way).
4. Score xrt_tokens.json with zero LoRA in strict mode at TP1 and TP2 -> xrt_megatron_strict_tp1.json / _tp2.json
   (muse_glimmer_xrt_logprobs_v1), plus native mode at TP1 and TP2 -> xrt_megatron_native_tp1.json / _tp2.json.
5. TP4 via the kv_head_replication row (runtime/kv_replication.py, read-only for you): needs 4 GPUs, your pool has 2, so design
   and unit-test it (toy config, 2 GPUs or CPU simulation) only if it fits in ~1.5 h; write the exact 4-GPU command and mark the
   real run NOT RUN. Otherwise keep failing closed and explain.
6. Make StrictTrainer satisfy TrainerLike in rl_engine/models/muse_glimmer/e2e/engines.py exactly: token_logprobs(token_ids,
   completion_start) -> fp32 tensors with autograd history; step(loss) for ANY scalar loss built from those tensors (the E2E bridge
   will call it from a server process with a surrogate loss sum_i (lp_i * g_i).sum()); export_lora() -> (bf16 state dict keyed
   layers.{i}.{proj}.lora_{A,B}, hash) using e2e/sync.py lora_sha256 (check key names with validate_lora_state_dict); provenance();
   close(). Document in your REPORT exactly how StrictTrainer is constructed and how TP2 is launched (torchrun? which env vars? which
   rank returns results?) so the bridge agent, who runs the trainer in a subprocess, can drive it. Add a GPU test gated by
   RL_KERNEL_MUSEGLIMMER_E2E=1 for the protocol.`,
  },
  {
    key: 'e2e_bridge', pool: '6-7', env: 'both', review: true,
    task: `Row: final_e2e_lora: cross-environment trainer bridge + timeline/profiling instrumentation for the real E2E. Your GPU pool:
6-7, one GPU at a time (two other agents share this pool through the lease; keep GPU use short).
You own: rl_engine/models/muse_glimmer/e2e/* (all files), tests/models/muse_glimmer/test_e2e_loop_mock.py, new
tests/models/muse_glimmer/test_e2e_bridge*.py, e2e/REPORT_e2e_loop.md, reports/muse_glimmer_poc/data/e2e_bridge.json.

Problem: the GRPO orchestrator (e2e/loop.py, e2e/run_poc_grpo.py) is single-process and imports both StrictRolloutEngine (needs
vLLM, only in ~/envs/vllm) and StrictTrainer (needs Megatron Bridge/TE, only in ~/envs/megatron). The envs cannot be merged (torch
cu130 vs cu129 builds). Build a bridge so the orchestrator runs in the vLLM env and the trainer runs in a subprocess started with
~/envs/megatron/bin/python:
- RemoteTrainer (client, implements TrainerLike) + a trainer server module. Transport: dependency-free (multiprocessing.connection
  Listener/Client on localhost with a random authkey, or a Unix socket); tensor payloads via torch.save to bytes.
  token_logprobs returns CPU fp32 leaf tensors with requires_grad=True; step(loss) runs loss.backward() locally, ships each lp.grad
  to the server, and the server applies them with the surrogate loss sum_i (lp_i * g_i).sum() passed to StrictTrainer.step. The
  gradient of that surrogate w.r.t. lp_i is exactly g_i bit for bit: prove it with a test. export_lora/provenance/close are
  forwarded; LoRA tensors cross the boundary bit-exact and the hash is re-verified on the client with e2e/sync.py.
- TP>1 trainer: the server runs under torchrun (nproc = train TP) with rank 0 as the RPC endpoint, broadcasting each call
  (torch.distributed broadcast_object_list) so all ranks execute the same trainer method; results come from rank 0. Read
  runtime/strict_trainer.py and runtime/REPORT_megatron_strict.md (read-only; another agent is changing them right now; rely
  only on the TrainerLike protocol plus constructor arguments, and make the server's trainer construction a single small function
  that is easy to adapt).
- run_poc_grpo.py: add --trainer-python (default ~/envs/megatron/bin/python) and use the bridge for the real engines. GPU placement
  stays explicit (--rollout-gpus / --train-gpus): the integrator leases the whole set for the orchestrator, and the orchestrator
  sets CUDA_VISIBLE_DEVICES only for the child processes it launches, from those flags (the only place anything sets it).
- Timeline instrumentation for the report: per step, wall time of rollout / trainer scoring / backward+step / LoRA
  export+transfer+load / drift metrics, tokens generated, reward mean, drift metrics; written to steps.jsonl and summarized in
  summary.json. Optional --profile-step N: torch.profiler capture of the rollout of step N in the orchestrator and of the train step
  N on the server; save top-ops tables as JSON.
- Tests: the bridge with a CPU mock trainer server in the megatron env (spawned subprocess, real transport), surrogate-gradient
  exactness, LoRA bit-exactness across the boundary, failure handling (server crash -> clear error, no hang: timeouts on every
  call). Then a small GPU smoke if StrictTrainer can be built with a toy config (see tests/models/muse_glimmer/test_megatron_strict_binding.py);
  otherwise mark it NOT RUN.
- Do NOT run the real 30B E2E (the integrator runs it on the whole node later). Write the exact command lines for the four modes
  (strict/strict, native/native, strict-train/native-rollout, native-train/strict-rollout) into your REPORT for two layouts:
  A) rollout TP1 on GPU 0 + train TP2 on GPUs 2,3; B) rollout TP2 on GPUs 0,1 + train TP2 on GPUs 2,3.`,
  },
  {
    key: 'kernels_gemm', pool: '4-5', env: 'vllm', review: true,
    task: `Row: GEMM correctness follow-ups (fast tree GEMM + its adapters in the FFN and LM-head ops). Your GPU pool: 4-5 (shared with
two other kernel agents through the lease; lease one GPU at a time).
You own: rl_engine/models/muse_glimmer/ops/det_gemm.py, rl_engine/models/muse_glimmer/triton/det_gemm_tree.py (and other
triton/det_gemm*.py), ops/swiglu_ffn.py, triton/swiglu*.py, ops/lm_head_softcap_logprob.py, triton/softcap_logprob.py, their tests
(tests/models/muse_glimmer/test_det_gemm.py, test_swiglu_ffn.py, test_lm_head_softcap_logprob.py), their ops/REPORT_*.md, and
reports/muse_glimmer_poc/data/kernels_gemm.json.

An adversarial review of these files produced 21 findings, in ${SP}/findings_gemm.json (fields: id, severity, file, line, claim,
evidence, failure_scenario, suggested_fix; prior_verification where one exists). Only swiglu_ffn:2 and swiglu_ffn:5 were
independently confirmed; the rest are UNVERIFIED. Highest priority: det_gemm:1 (when CHUNK_LEN == BLOCK_K, i.e. ITERS == 1, Triton
may fold the tree adds into chained MMA accumulation so the kernel does not implement the declared 8-leaf tree; affects K=256/512 and
the dA/dW paths) and det_gemm:2 (the autograd backward is not TP-capable for column-parallel GEMMs). For det_gemm:1 inspect the
compiled TTGIR/PTX, not just the Python.
${FINDINGS_RULES}
For det_gemm:2: provide a TP-capable path in which each rank produces its dA partial in fp32 and the partials are combined in the
canonical TP tree (rl_engine/models/muse_glimmer/runtime/tp_canonical_tree.py, read-only for you) before the single bf16 rounding,
with an autograd-capable entry and a test that TP2/TP4/TP8 (simulated by splitting N and combining with the tree function on one GPU)
equals TP1 bitwise.
After fixes: run all test files you own, and re-run the GEMM/FFN/LM-head benchmarks so the report has before/after numbers
(TFLOP/s and ms per shape; series in your JSON).`,
  },
  {
    key: 'kernels_attention', pool: '4-5', env: 'vllm', review: true,
    task: `Row: gated attention follow-ups. Your GPU pool: 4-5 (shared with two other kernel agents through the lease; lease one GPU at a
time, two only for the CP2 test).
You own: rl_engine/models/muse_glimmer/ops/gated_attention*.py, ops/attention_golden_bwd.py, triton/gated_attention*.py,
runtime/cp_attention.py, runtime/check_cp_parity.py, tests/models/muse_glimmer/test_gated_attention_fwd.py, test_gated_attention_bwd.py,
tests/models/muse_glimmer/distributed/test_cp_*.py, distributed/_cp_harness.py, ops/REPORT_gated_attention*.md, runtime/REPORT_ws2_cp.md,
reports/muse_glimmer_poc/data/kernels_attention.json.

Tasks (commit after each):
1. Head-count parameters: the strict attention wrappers (forward incl. varlen/chunked/paged/decode, and backward) pin 32 Q heads / 2 KV
   heads, so a TP-sharded rank (TP2: 16 Q / 1 KV; TP4 or TP8 with KV replication: 8 or 4 Q / 1 KV) cannot call them. Add H_Q / H_KV
   parameters with defaults that keep every current caller working, and prove for TP in {2,4,8}: the per-rank call on its head shard
   equals the matching slice of the full-geometry call bitwise (forward o, lse and gated output; backward dq, dk, dv, dgate; for TP4/8
   the replicated-KV ranks' dk/dv partials, combined in the canonical TP tree from runtime/tp_canonical_tree.py (read-only for you),
   must equal the TP1 dk/dv). Heads are independent, so any difference means a per-head reduction depends on the head count: a bug.
2. CP backward contract defect: contract O5's backward rule "dK/dV accumulated over query blocks ascending" is not CP-invariant. The
   CP path (runtime/cp_attention.py) accumulates dK/dV per canonical query chunk (8 canonical query chunks of the full sequence) and
   combines the chunk partials in the O3 tree, so CP1/CP2/CP4 agree, but it differs from the one-shot TP1 kernel in the last bits
   (rel 1.6e-5). Decision already made: adopt the query-chunk rule INSIDE triton/gated_attention_bwd.py so the one-shot kernel equals
   the CP path bitwise. Implement, test (single-GPU kernel == CP1 path == CP2 on two leased GPUs), and write the replacement contract
   wording for O5 under requests_to_integrator (do not edit docs/contracts). CP4 needs 4 GPUs: write the command and mark NOT RUN.
3. The 9 unverified review findings for the backward, in ${SP}/findings_attention.json (gated_attention_bwd:1 is marked blocking: the
   autograd binding may import the strict forward from the wrong module and silently run the slow fp32 golden).
${FINDINGS_RULES}
4. Re-run all attention tests you own and the attention benchmarks (prefill/decode forward, backward; ms and ratio to SDPA per
   shape) so the report has before/after numbers as series in your JSON.`,
  },
  {
    key: 'kernels_rope', pool: '4-5', env: 'vllm', review: true,
    task: `Row: qknorm_scaled_rope follow-ups. Your GPU pool: 4-5 (shared with two other kernel agents through the lease; lease one GPU at
a time).
You own: rl_engine/models/muse_glimmer/ops/qknorm_rope.py, triton/qknorm_rope.py, tests/models/muse_glimmer/test_qknorm_rope.py,
ops/REPORT_qknorm_rope*.md, reports/muse_glimmer_poc/data/kernels_rope.json.
The 5 unverified review findings are in ${SP}/findings_rope.json (contract O4 RoPE-table build/cache identity across runtimes,
non-bf16 inputs silently accepted, no TP-parity test for head shards (BLOCK_HQ 16/8/4 vs 32), implicit local/global selection,
int32 offset overflow).
${FINDINGS_RULES}
Keep the public API backward compatible (callers in reference/, runtime/vllm_*, runtime/megatron_* must keep working unchanged;
grep for them). For the TP-parity finding, prove per-rank head shards (16/8/4 Q heads, 1 KV head) equal the slice of the full call
bitwise. Re-run the tests and the rope benchmark (series in your JSON).`,
  },
  {
    key: 'ablation', pool: '6-7', env: 'vllm', review: false,
    task: `Row: museglimmer_ablation: where does train/rollout drift come from in the native stacks? Your GPU pool: 6-7 (shared with two
other agents through the lease; one GPU per process; budget about 2 GPU-hours).
You own: a new directory rl_engine/models/muse_glimmer/ablation/ (ablation_hf.py, run_ablation.py, REPORT_ablation.md, __init__.py with
the SPDX header used elsewhere in the package), tests/models/muse_glimmer/test_ablation.py, reports/muse_glimmer_poc/data/ablation.json.
Goal: one-factor-at-a-time probes on the HF transformers 5.17 Muse-Glimmer model (bf16, one GPU, ~60 GB) scoring the 9 sequences of
reports/muse_glimmer_poc/data/xrt_tokens.json. Flip each factor against a fixed baseline and measure per-token fp32 logprob differences
over the completion tokens: number of tokens whose fp32 bits differ, max and mean |dlogp|, mean k3-KL (exp(d)-d-1), argmax flips.
Factors (at least): (1) batch composition: each sequence alone vs inside a padded batch of all 9 (right padding and left padding);
(2) full prefill vs teacher-forced incremental decode with the KV cache over the completion; (3) chunked prefill (chunks of 64, 256, 1024);
(4) attention implementation sdpa vs eager (and flash-attention 2 only if already installed in the env; else skip and say so);
(5) TF32 matmul on vs off; (6) logits + soft-cap computed in bf16 vs fp32; (7) RoPE table construction: the transformers 5.17
matmul-based build (TF32-sensitive) vs an fp32 elementwise build as in 5.19, done by swapping the cos/sin computation; (8) a second GPU
vs the first (null control); (9) repeat run (null control).
Then show the strict reference path (rl_engine/models/muse_glimmer/reference, read-only for you: chain_harness.py / run_chain.py /
strict_model.py, backend triton) under factors (1)-(3): it must stay at 0 differing tokens.
Output: ablation.json with one record per (factor, setting, sequence) carrying per-token arrays (so the report can draw a factor x sequence
heatmap and bar charts) and REPORT_ablation.md with the table and conclusions. Load the model once per process.`,
  },
  {
    key: 'reference_profile', pool: '6-7', env: 'vllm', review: false,
    task: `Row: full_model_chain follow-up: reference scoring of the shared token file + single-GPU profiling. Your GPU pool: 6-7 (shared
with two other agents through the lease; one GPU per process).
You own: rl_engine/models/muse_glimmer/reference/*, tests/models/muse_glimmer/test_full_model_chain.py, test_strict_model_tiny.py,
reference/REPORT_full_model_chain.md, reports/muse_glimmer_poc/data/reference_profile.json, reports/muse_glimmer_poc/data/xrt_reference_*.json.
1. Add a scoring mode to reference/run_chain.py: --score-file reports/muse_glimmer_poc/data/xrt_tokens.json --score-out <path> writing
   muse_glimmer_xrt_logprobs_v1 from the train-forward path, for --backend golden and --backend triton -> xrt_reference_golden.json and
   xrt_reference_triton.json. In the same run, check prefill == decode for these sequences (bitwise) and record it.
2. Re-run the real-weights chain checks on the current branch (run_chain.py --backend triton --hf-compare, plus the 2100-token window
   case from REPORT_full_model_chain.md) and confirm byte equality still holds; keep the HF anchor numbers (max |dlogp|, argmax agreement)
   and per-token series.
3. Profile the strict single-GPU model (triton backend) at prefill T in {512, 2048, 4096} and decode (batch 1 and 16, context 2048): time
   per op class (embedding, norms, QKV/gate/O GEMMs, qknorm+rope, attention global vs local, FFN GEMMs, LM head+softcap+logprob) per layer
   type, in ms and as fractions of the total; the same breakdown for HF bf16 (sdpa) as the native baseline; peak memory. Use CUDA events
   or torch.profiler (record which). Save as JSON series for stacked-bar charts.
Kernel fixes are happening in parallel in other worktrees; your numbers are for the current branch state: record the commit sha.`,
  },
  {
    key: 'review_fwd_norms', pool: '4-5', env: 'vllm', review: false, readonly: true,
    task: `Row: fresh adversarial review (the earlier reviewers of these two rows died before finishing). You are READ-ONLY: do not edit or
commit repo files; write probe scripts only under ${SP}/review_fwd_norms/. Your GPU pool: 4-5 (shared through the lease).
Review (1) the strict gated attention FORWARD: rl_engine/models/muse_glimmer/ops/gated_attention.py (and siblings for varlen/paged/decode),
triton/gated_attention_fwd*.py / triton/gated_attention*.py forward kernels, tests/models/muse_glimmer/test_gated_attention_fwd.py,
ops/REPORT_gated_attention*.md; (2) norms + normalized embedding: ops/norms*.py / ops/*embedding*.py, triton/norms*.py,
tests/models/muse_glimmer/test_norms.py, its REPORT. Against docs/contracts/muse-glimmer-30b-strict-contract.md: casts and orders as
declared (R5), fixed reduction trees, fail-closed behaviour, the m=-inf guard, train==chunked==decode==paged invariance actually tested,
tests that cannot fail, unvalidated inputs that read out of bounds, and claims in the REPORTs not backed by a run. Reproduce each finding
with a probe where possible.
Return your findings in the structured result: put each finding in key_numbers? No: put each finding as an entry of open_issues formatted
"<id> | <severity blocking/should_fix/nit> | <file>:<line> | <claim> | <evidence> | <suggested fix>", ids norms:N and gated_attention_fwd:N.
Set branch to "none", head to the commit you reviewed, results_files to the probe scripts.`,
  },
]

function implPrompt(r) {
  return `${PREAMBLE}\n\n## Your task\n${r.task}`
}

function reviewPrompt(r, impl) {
  return `You are an adversarial reviewer for one row ("${r.key}") of the Muse-Glimmer-30B strict POC (RFC RL-Align/RL-Kernel#501).
The implementer worked on branch ${impl.branch} (HEAD ${impl.head}) and returned the report below. Find what is wrong, overstated or
missing. Default to skepticism: a claim counts as verified only if you saw it in code you read or output you produced yourself.

Setup: you are in your own worktree; run \`git checkout --detach ${impl.head}\` there (the branch itself is checked out elsewhere).
Review the diff \`git diff ${BASE}...${impl.head}\`. Environment rules: run code with PYTHONPATH=<your worktree> and ~/envs/vllm/bin/python
or ~/envs/megatron/bin/python; never install packages; GPUs only via \`~/bin/gpu-lease -n K --pool ${r.pool} -- <cmd>\` (never set
CUDA_VISIBLE_DEVICES yourself); 8x A100 sm80. You are READ-ONLY: no commits, no pushes, no stash; write probe scripts only under
${SP}/review_${r.key}/. Keep GPU use modest (other agents share the pool).

Check: (1) each key number/claim against the logs/JSON files and, for the 2-4 most important claims, by re-running the cheapest command
that demonstrates it; (2) the diff for contract violations (docs/contracts/muse-glimmer-30b-strict-contract.md), silent fallbacks, tests
that cannot fail (tautologies), unhandled shapes the 30B model hits (T 1..8192, decode batch 1..256), broken callers in other modules
(grep), and REPORT claims not backed by a run; (3) for finding verdicts the implementer marked "refuted": are the refutations sound?
(4) whether the results JSON holds raw data good enough to plot.

Implementer report:
${JSON.stringify(impl, null, 1)}`
}

const results = await pipeline(
  ROWS,
  (r) => agent(implPrompt(r), {
    label: `impl:${r.key}`, phase: 'Implement', schema: IMPL_SCHEMA,
    ...(r.readonly ? {} : { isolation: 'worktree' }),
  }),
  (impl, r) => {
    if (!impl) { log(`${r.key}: implementer returned nothing (died?)`); return { row: r.key, impl: null, review: null } }
    if (!r.review) return { row: r.key, impl, review: null }
    return agent(reviewPrompt(r, impl), { label: `review:${r.key}`, phase: 'Review', schema: REVIEW_SCHEMA, isolation: 'worktree' })
      .then((review) => ({ row: r.key, impl, review }))
  },
)
return results
