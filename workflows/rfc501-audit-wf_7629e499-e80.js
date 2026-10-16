export const meta = {
  name: 'rfc501-audit',
  description: 'Audit RFC #501 text against the pinned Muse Glimmer sources and propose verified edits',
  phases: [
    { title: 'Audit', detail: 'one auditor per RFC section vs primary sources' },
    { title: 'Refute', detail: 'adversarial per-edit verification' },
    { title: 'Critic', detail: 'what was not checked' },
  ],
}

const S = '/tmp/claude-1000/-home-ubuntu-RL-Kernel--claude-worktrees-swift-oak-hsv7/80655dc5-1eae-4992-a93b-f6f87e8ab1d8/scratchpad'
const REPO = '/home/ubuntu/RL-Kernel/.claude/worktrees/swift-oak-hsv7'

const SOURCES = `
PRIMARY SOURCES (read-only; all already on disk):
- RFC #501 body (the text under audit): ${S}/rfc501.md   (one maintainer comment: ${S}/rfc501_comments.md)
- Pinned checkpoint meta-models/Muse-Glimmer-30B @ revision a4e59da52a7bc87ae7251dd5545c0dd437c44b68:
    ${S}/hf/config.json                      (authoritative hyperparameters)
    ${S}/hf/model.safetensors.index.json     (weight_map + metadata total_size / total_parameters)
    ${S}/hf/h1.json, ${S}/hf/h2.json          (raw safetensors JSON headers of shard 1 and 2: every tensor's dtype + shape)
    ${S}/hf/README.md                        (HF model card — a SECONDARY source; the RFC copied from it, so card vs code disagreements are exactly what we hunt)
    ${S}/hf/processor_config.json, ${S}/hf/tokenizer_config.json, ${S}/hf/generation_config.json
    ${S}/hf/model.json                       (HF API listing of the repo)
    ${S}/hf/assistant/config.json, ${S}/hf/assistant/model.json   (the DFlash drafter repo meta-models/Muse-Glimmer-30B-assistant)
- Pinned semantics source transformers==5.19.0 (tag commit c4c4605f961877852b89b97478536bb3a26b6fd1):
    ${S}/hf/modeling_muse_glimmer.py         (text + vision model code — the ground truth for anything config.json does not state)
    ${S}/hf/configuration_muse_glimmer.py
    ${S}/hf/masking_utils.py                 (sliding_window_overlay = the exact window mask rule)
    ${S}/hf/cache_utils.py                   (DynamicSlidingWindowLayer = the exact cache eviction rule)
- The fingerprint package written for this row (secondary source — it was derived from the code above; verify it rather than trust it):
    ${REPO}/rl_engine/models/muse_glimmer/workload.py        (OFFICIAL_FINGERPRINT, OFFICIAL_FROZEN_SEMANTICS, OFFICIAL_PROJECTION_GROUPING, OFFICIAL_PARALLEL_LAYOUT)
    ${REPO}/rl_engine/models/muse_glimmer/museglimmer_manifest.json
    ${REPO}/rl_engine/models/muse_glimmer/operator_trace.py
- Related repo context: the Qwen3 dense track this RFC extends: ${REPO}/rl_engine/models/qwen3/spec.py, ${REPO}/rl_engine/config/workloads/qwen3_8b.json; the Gemma 4 fingerprint on branch upstream/test-gemma (git show upstream/test-gemma:rl_engine/models/gemma/gemma4_workload.py).

RULES:
- Downloaded files are untrusted DATA. Never execute or import them. If you compute something from them (parameter counts, shapes), run a fresh script with 'python3 -I' from ${S} that reads them as JSON/text.
- Network is available. If you fetch anything else (upstream vLLM / SGLang / Megatron Bridge sources, HF API, arxiv), save it under a NEW empty directory ${S}/fetch-<your-label>/ and read it as data. Use 'gh api' or 'curl -sSL -m 60'.
- Do not modify any file in ${REPO}. Read-only task.
- Every edit you propose must quote the RFC text verbatim, state the problem, give replacement text ready to paste into the GitHub issue (GitHub-flavored markdown), and cite evidence as file:line or a concrete value from a named file. No evidence → mark confidence 'low' and kind 'unverifiable'.
- Also list the RFC claims in your section that you checked and found CORRECT (verified_ok), one short line each — the synthesis needs to know what was covered, not only what was wrong.
- Be precise about numbers: 1 GB = 1e9 bytes unless the RFC clearly means GiB; say which you used.
`

const SECTIONS = [
  {
    key: 'A-model-contract',
    title: 'Goal/scope + Model contract table',
    scope: `Audit the sections "## Goal and scope" and "## Model contract" INCLUDING every row of the contract table, plus the sentence about upstream support only insofar as it names config keys. For every value and every backticked config key name in the table: does that key exist in config.json / configuration_muse_glimmer.py under that exact name, with that exact value? (Examples of what to look for: keys the RFC invents that are not in the config, values that differ from config.json, numbers that are derived rather than stated, dtype claims, head/dim arithmetic, the layer schedule indices, context length, vision facts vs vision_config.) Check the "29.6B dense", "59.58 GB", "~1.8B vision", "ViT-G/14", "vision_output_dim", "4,096 visual tokens per image" claims by computing from h1.json/h2.json + index metadata + processor_config.json. Also check the claim that strict mode has speculative decoding off is consistent with generation_config.json.`,
  },
  {
    key: 'B-frozen-semantics',
    title: 'To-freeze paragraph + Architecture captions + Numerical requirements',
    scope: `Audit the paragraph beginning "The fingerprint also has to freeze what the configs do not state:" — for EACH listed item (gate width and placement; Q/K norm form; what post_norm_eps governs; embedding normalization form and dtype; soft-cap formula, dtype, order vs multiplier; RoPE rotated-channel count; local-layer cache eviction rule; Q/K/V/gate projection grouping in Megatron Bridge vs vLLM) read modeling_muse_glimmer.py (plus masking_utils.py / cache_utils.py where relevant) and write the FROZEN ANSWER with line citations, as replacement RFC text (the RFC should now state the frozen answers, not list them as unknowns). Then audit the "### Architecture" captions ("[G] boxes reuse the deterministic GEMM", "[F*] fused strict operators", "output gate is part of [F3] and [F4]") and the whole "## Numerical requirements" section (fixed reduction order, batch invariance, sequence-split invariance "any eviction timing around the 2,048 window", one projection grouping, fail-closed list, parity/correctness definitions, Platforms paragraph) for statements that the code contradicts or that need a precise definition (e.g. exactly which keys a query at absolute position p attends to on a local layer; how many past tokens the HF cache retains; whether the gate reads the normed layer input or the attention output; in which dtype each cast happens). Compare with OFFICIAL_FROZEN_SEMANTICS in workload.py and flag any place where that pin itself is wrong.`,
  },
  {
    key: 'C-ws1-rows',
    title: 'WS1 work items table + Suggested order + operator-PR acceptance',
    scope: `Audit the "### WS1: single-GPU kernels" table row by row (normalized_embedding, fused_add_rmsnorm, dense_gemm_qualification, qknorm_scaled_rope, global_gated_attention, local_gated_attention_train, local_gated_attention_rollout, swiglu_ffn, lm_head_softcap_logprob), the "Foundation" row, the "Suggested order" paragraph, and "### Operator PRs" acceptance. For each row: is the Scope text an accurate description of what the code does (e.g. is [F1] really "residual add + RMSNorm" given the decoder's four norms per layer with two different eps values and the (1+w) centered form? is the Q/K scale applied to both Q and K? where does the sigmoid gate get its input and at what width? does the lm_head row list the operations in the order the code applies them? does the embedding row need the no-sqrt(hidden) fact?). Enumerate the exact GEMM shapes (N,K) per layer from h1.json/h2.json and the number of nodes per row from operator_trace.py; propose any missing row or row-split (e.g. does the attention gate GEMM belong to dense_gemm_qualification or to the attention rows? is a separate row needed for the four-norm sandwich?). Check the "Reuse" column PR numbers exist in RL-Align/RL-Kernel via 'gh pr view <n> --repo RL-Align/RL-Kernel --json title,state' or 'gh issue view' and that their titles plausibly match the stated reuse (report title+state for each).`,
  },
  {
    key: 'D-ws2-upstream',
    title: 'WS2 section + upstream runtime claims',
    scope: `Audit "### WS2: multi-GPU" (the intro paragraph, the four bullets Row-parallel sums / Vocabulary / 2 KV heads / CP, and the table) and the sentence "Upstream support: vLLM native model (0.27.0+, vllm#51655), SGLang (sglang#34262), and Megatron Bridge bridge.models.muse_glimmer with an 8x H100 LoRA recipe" plus the to-freeze claim "Megatron Bridge fuses all four, vLLM fuses Q/K/V only". Verify the arithmetic (4096/8=512 and 4 heads each; 19968/8=2496; 202048 -> 202752 with 128 x canonical TP 8; which KV head each Q head uses per repeat_kv in modeling_muse_glimmer.py and therefore the replication pattern at TP2/TP4/TP8; whether TP2 needs replication at all). Then FETCH upstream to verify: (1) vLLM: 'gh pr view 51655 --repo vllm-project/vllm --json title,state,mergedAt,files' and fetch the vLLM model file for muse_glimmer from the main branch (find its path via the PR files list; raw.githubusercontent.com/vllm-project/vllm/main/<path>) — confirm whether Q/K/V are fused (QKVParallelLinear) and how gate_proj is handled, and whether a release tag >= v0.27.0 contains it ('gh api repos/vllm-project/vllm/tags --paginate -q .[].name | head'); (2) SGLang: 'gh pr view 34262 --repo sgl-project/sglang --json title,state,mergedAt'; (3) Megatron Bridge: look for muse_glimmer under NVIDIA-NeMo/Megatron-Bridge via 'gh api repos/NVIDIA-NeMo/Megatron-Bridge/contents/src/megatron/bridge/models' (adjust path if needed) and check whether the attention projection is a single linear_qkv including the gate, and whether an 8xH100 LoRA recipe exists. Report exactly what you could and could not verify. Save fetched files under ${S}/fetch-D/.`,
  },
  {
    key: 'E-validation-openq-refs',
    title: 'Validation table + acceptance + experiments + open questions + follow-up + references',
    scope: `Audit "### Validation and integration" table, "### WS1 and WS2" acceptance, "### Experiments", "## Open questions" (1-8), "## Follow-up: speculative rollout", and "## References". For the open questions: which are now ANSWERED or constrained by the fingerprint pins (projection grouping is pinned as four separate GEMMs in workload.py; the HF cache eviction rule in cache_utils.py gives a baseline for Q5; etc.) — propose the status/decision text to add. For the follow-up: verify the drafter claims ("5 layers, 16-token blocks, reads target hidden states at layers {1, 13, 25, 37, 49}", repo name) against ${S}/hf/assistant/config.json and model.json. For references: resolve each URL ('curl -sSIL -m 30 <url> | grep -m1 HTTP' is enough; for arxiv use the API 'curl -sSL http://export.arxiv.org/api/query?id_list=<id>' and report the paper title to confirm it matches the link text). For the validation rows: check 'hybrid_attention_regression' lengths 2047/2048/2049 against the exact mask rule in masking_utils.py (sliding_window_overlay) and the cache retention in cache_utils.py — are these the right boundary lengths, and is anything missing (e.g. decode step right at eviction)? Check cited PR/issue numbers in these sections exist in RL-Align/RL-Kernel ('gh pr view <n> --repo RL-Align/RL-Kernel --json title,state' / 'gh issue view'). Save fetched files under ${S}/fetch-E/.`,
  },
]

const EDITS_SCHEMA = {
  type: 'object',
  properties: {
    section: { type: 'string' },
    edits: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          id: { type: 'string', description: 'section letter + number, e.g. A3' },
          kind: { type: 'string', enum: ['correction', 'clarification', 'frozen_answer', 'answered_open_question', 'new_item', 'unverifiable'] },
          rfc_text: { type: 'string', description: 'verbatim quote of the RFC text being changed (or the heading where new text goes)' },
          problem: { type: 'string' },
          proposed_text: { type: 'string', description: 'ready-to-paste GitHub markdown replacement' },
          evidence: { type: 'string', description: 'file:line citations and/or concrete values' },
          confidence: { type: 'string', enum: ['high', 'medium', 'low'] },
        },
        required: ['id', 'kind', 'rfc_text', 'problem', 'proposed_text', 'evidence', 'confidence'],
      },
    },
    verified_ok: { type: 'array', items: { type: 'string' } },
    could_not_check: { type: 'array', items: { type: 'string' } },
  },
  required: ['section', 'edits', 'verified_ok', 'could_not_check'],
}

const VERDICT_SCHEMA = {
  type: 'object',
  properties: {
    verdicts: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          id: { type: 'string' },
          verdict: { type: 'string', enum: ['confirmed', 'confirmed_with_fix', 'refuted', 'unverifiable'] },
          reason: { type: 'string', description: 'evidence with file:line; for refuted, what the source actually says' },
          corrected_proposed_text: { type: 'string', description: 'only for confirmed_with_fix: the corrected replacement text' },
        },
        required: ['id', 'verdict', 'reason'],
      },
    },
    missed_in_section: { type: 'array', items: { type: 'string' }, description: 'problems in this section the auditor did not report, with evidence' },
  },
  required: ['verdicts', 'missed_in_section'],
}

const CRITIC_SCHEMA = {
  type: 'object',
  properties: {
    missing: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          area: { type: 'string' },
          detail: { type: 'string' },
          evidence: { type: 'string' },
          severity: { type: 'string', enum: ['blocking', 'should_fix', 'nice_to_have'] },
        },
        required: ['area', 'detail', 'evidence', 'severity'],
      },
    },
    rfc_claims_never_checked: { type: 'array', items: { type: 'string' } },
    overall: { type: 'string' },
  },
  required: ['missing', 'rfc_claims_never_checked', 'overall'],
}

const auditPrompt = (sec) => `You are auditing one section of GitHub RFC issue RL-Align/RL-Kernel#501 ("[RFC][Muse-Glimmer-30B][CUDA/ROCm] ROADMAP") against primary sources, so the RFC author can update the issue text. The architecture-fingerprint row has just been implemented and pins the checkpoint and semantics; the RFC was written from the model card before that, so some statements are wrong, imprecise, or now answerable.

YOUR SECTION: ${sec.title}
${sec.scope}

Read the whole RFC once for context (${S}/rfc501.md), then audit ONLY your section exhaustively — every number, every backticked key, every claim. Prefer the modeling code and config.json over the model card. Be skeptical of the fingerprint package too: if workload.py disagrees with modeling_muse_glimmer.py, report that as an edit with kind 'correction' and say which is wrong.
${SOURCES}
Return the structured result. Put ids as "${sec.key.charAt(0)}1", "${sec.key.charAt(0)}2", ... Proposed text must be paste-ready GitHub markdown that fits where the quoted text sits (same table column format if it is a table cell).`

const refutePrompt = (sec, audit) => `You are an adversarial verifier. Another agent audited one section of GitHub RFC RL-Align/RL-Kernel#501 against primary sources and proposed the edits below. Your job is to try to REFUTE each edit independently: re-derive the fact from the primary sources yourself (open the file, find the line, compute the number) and decide whether the edit's claim and its proposed replacement text are exactly right. Default to 'refuted' when the evidence does not support the specific claim; use 'confirmed_with_fix' when the direction is right but a number, key name, line reference, or wording in proposed_text is wrong (and supply the corrected text); use 'unverifiable' only for claims that need information not in the sources and not fetchable. Also, read the section yourself and list any REAL problem the auditor missed (missed_in_section), with evidence — be concrete, not speculative.

SECTION: ${sec.title}
SECTION SCOPE GIVEN TO THE AUDITOR: ${sec.scope}

AUDITOR OUTPUT (JSON):
${JSON.stringify(audit, null, 2)}
${SOURCES}
Also re-check at least three items from the auditor's verified_ok list at random and report in missed_in_section if any is actually wrong. If the auditor fetched upstream files they are under ${S}/fetch-*/; you may re-fetch into ${S}/fetch-${sec.key.charAt(0)}-verify/ if needed.`

phase('Audit')
log(`Auditing ${SECTIONS.length} RFC sections against pinned sources`)

const results = await pipeline(
  SECTIONS,
  (sec) => agent(auditPrompt(sec), { label: `audit:${sec.key}`, phase: 'Audit', schema: EDITS_SCHEMA }),
  (audit, sec) => {
    if (!audit) return null
    log(`${sec.key}: ${audit.edits.length} edits proposed, ${audit.verified_ok.length} claims verified ok → refuting`)
    return agent(refutePrompt(sec, audit), { label: `refute:${sec.key}`, phase: 'Refute', schema: VERDICT_SCHEMA, effort: 'high' })
      .then((verdict) => ({ section: sec, audit, verdict }))
  },
)

const sections = results.filter(Boolean)
const dropped = SECTIONS.filter((s) => !sections.some((r) => r.section.key === s.key)).map((s) => s.key)
if (dropped.length) log(`WARNING: sections with no result (agent failed/skipped): ${dropped.join(', ')}`)

// Barrier is justified: the critic needs every section's edits + verified lists to say what was never checked.
const summaryForCritic = sections.map((r) => ({
  section: r.section.title,
  edits: r.audit.edits.map((e) => {
    const v = (r.verdict && r.verdict.verdicts.find((x) => x.id === e.id)) || null
    return { id: e.id, kind: e.kind, rfc_text: e.rfc_text, problem: e.problem, verdict: v ? v.verdict : 'no-verdict' }
  }),
  verified_ok: r.audit.verified_ok,
  could_not_check: r.audit.could_not_check,
  refuter_missed_in_section: r.verdict ? r.verdict.missed_in_section : [],
}))

phase('Critic')
const critic = await agent(
  `You are the completeness critic for an audit of GitHub RFC RL-Align/RL-Kernel#501 against the pinned Muse-Glimmer-30B sources. Five auditors each covered one section and five refuters verified their edits. Below is the full record of what was proposed, what verdict each edit got, and which RFC claims each auditor says it verified as correct.

Your job: read the ENTIRE RFC (${S}/rfc501.md) line by line and find what was NOT covered — RFC claims (numbers, key names, PR/issue numbers, architectural statements, process statements like which branch PRs target per ${S}/rfc501_comments.md, status markers in the tables, the Foundation row's own status now that the fingerprint exists) that appear in neither any edit nor any verified_ok list. For each such claim, check it yourself against the sources and report only real issues or genuinely unchecked claims. Also judge whether the set of edits, taken together, would leave the RFC internally consistent (e.g. if the gate is elementwise, do all places that say "per-head" get fixed? if qk_scale_factor changes, does every mention change? does the operator trace node accounting in the RFC match operator_trace.py?). Do NOT re-litigate verdicts already given unless you have hard contrary evidence.

AUDIT RECORD (JSON):
${JSON.stringify(summaryForCritic, null, 2)}
${SOURCES}`,
  { label: 'critic', phase: 'Critic', schema: CRITIC_SCHEMA, effort: 'high' },
)

return {
  sections: sections.map((r) => ({
    key: r.section.key,
    title: r.section.title,
    edits: r.audit.edits.map((e) => ({
      ...e,
      verdict: (r.verdict && r.verdict.verdicts.find((x) => x.id === e.id)) || { verdict: 'no-verdict', reason: 'refuter returned nothing for this id' },
    })),
    verified_ok: r.audit.verified_ok,
    could_not_check: r.audit.could_not_check,
    refuter_missed_in_section: r.verdict ? r.verdict.missed_in_section : [],
  })),
  dropped_sections: dropped,
  critic,
}