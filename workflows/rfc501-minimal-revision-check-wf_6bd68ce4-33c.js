export const meta = {
  name: 'rfc501-minimal-revision-check',
  description: 'Check the minimal RFC #501 revision for coverage, accuracy and brevity',
  phases: [{ title: 'Check', detail: 'coverage, accuracy, brevity checkers in parallel' }],
}

const WT = '/home/ubuntu/RL-Kernel/.claude/worktrees/swift-oak-hsv7'
const CTX = `RFC RL-Align/RL-Kernel#501 (Muse-Glimmer-30B strict train/rollout) is being revised after a proof of concept.
Original: ${WT}/reports/muse_glimmer_poc/rfc/RFC_501_original.md. Revised draft: ${WT}/reports/muse_glimmer_poc/rfc/RFC_501_revised.md.
The required changes: ${WT}/reports/muse_glimmer_poc/data/rfc_findings.json (20 implementation findings F1-F20, use body_en) and
${WT}/reports/muse_glimmer_poc/data/rfc_audit.json (56 verified line-by-line audit items; each has rfc (original line), problem,
proposed/corrected). The owner's instruction: change the original RFC as LITTLE as possible and keep it concise (no long contract
detail; the fingerprint PR #558 carries OFFICIAL_FROZEN_SEMANTICS and docs/contracts/muse-glimmer-30b-strict-contract.md carries the
numerics), but every needed change must be covered. Repo facts: ${WT} (branch feat/museglimmer-poc): rl_engine/models/muse_glimmer/
workload.py and museglimmer_manifest.json, docs/contracts/muse-glimmer-30b-strict-contract.md, reports/muse_glimmer_poc/data/*.json,
rl_engine/models/muse_glimmer/**/REPORT_*.md. You are read-only: no edits, no commits. Be concise.`

const COVER = {
  type: 'object',
  properties: {
    uncovered: { type: 'array', items: { type: 'object', properties: { id: { type: 'string' }, what_is_missing: { type: 'string' }, minimal_edit: { type: 'string' } }, required: ['id', 'what_is_missing', 'minimal_edit'] } },
    covered_count: { type: 'integer' },
    fine_to_omit: { type: 'array', items: { type: 'object', properties: { id: { type: 'string' }, why: { type: 'string' } }, required: ['id', 'why'] } },
  },
  required: ['uncovered', 'covered_count', 'fine_to_omit'],
}
const ACC = {
  type: 'object',
  properties: {
    wrong: { type: 'array', items: { type: 'object', properties: { text: { type: 'string' }, evidence: { type: 'string' }, fix: { type: 'string' } }, required: ['text', 'evidence', 'fix'] } },
    verified_count: { type: 'integer' },
  },
  required: ['wrong', 'verified_count'],
}
const BREV = {
  type: 'object',
  properties: {
    edits: { type: 'array', items: { type: 'object', properties: { current: { type: 'string' }, shorter: { type: 'string' }, keeps: { type: 'string' } }, required: ['current', 'shorter', 'keeps'] } },
    unnecessary: { type: 'array', items: { type: 'object', properties: { current: { type: 'string' }, restore_to: { type: 'string' }, why: { type: 'string' } }, required: ['current', 'restore_to', 'why'] } },
  },
  required: ['edits', 'unnecessary'],
}

const [cover, acc, brev] = await parallel([
  () => agent(`${CTX}\n\nTask: COVERAGE. For every finding F1-F20 and every audit item (56), decide whether the revised draft covers its substance
(possibly in a shorter form, or by pointing to the fingerprint/contract). List what is NOT covered with a minimal edit (exact text to add or
replace, as short as possible). Also list items that are fine to omit from the RFC (pure detail already carried by #558 or the contract doc),
with the reason. Note: audit items were written before the semantics pin moved to transformers 5.17.0; the draft must use 5.17.0.`,
    { label: 'check:coverage', phase: 'Check', schema: COVER, effort: 'high' }),
  () => agent(`${CTX}\n\nTask: ACCURACY. Diff original vs revised (e.g. python difflib) and check every changed or added statement against the
repo facts and data (numbers, key names, formulas, head counts, TP/CP claims, PR numbers, versions, what the POC actually showed). List each
wrong or unsupported statement with evidence and the corrected wording (keep it short).`,
    { label: 'check:accuracy', phase: 'Check', schema: ACC, effort: 'high' }),
  () => agent(`${CTX}\n\nTask: BREVITY. The revised draft is ~30% longer than the original. For each changed or added passage, propose a shorter
wording that keeps the required substance (say what it keeps), and list edits that are not needed at all (restore the original text) with the
reason. Do not drop anything a finding or audit item actually requires; prefer moving detail to "frozen by the fingerprint (#558)" pointers.`,
    { label: 'check:brevity', phase: 'Check', schema: BREV, effort: 'high' }),
])
return { cover, acc, brev }
