from pathlib import Path

p = Path("/tmp/claude-1000/-home-ubuntu-RL-Kernel--claude-worktrees-swift-oak-hsv7/80655dc5-1eae-4992-a93b-f6f87e8ab1d8/scratchpad/vllmB/make_extra.py")
s = p.read_text()
old = '''    "Decode step = rank-0 step wall time with a CUDA synchronize per step (measurement mode). "
    "The host CPU was heavily loaded by other agents' jobs during these runs (1-minute load "
    "average per repeat in the last column; the node has 240 hardware threads); eager decode is "
    "CPU-launch bound, so the absolute numbers and the strict/native ratio carry an "
    "uncontrolled bias from that load, and the spread column only measures run-to-run noise "
    "inside one engine. Native eager and native CUDA-graph rows are both given; production vLLM "
    "runs CUDA graphs."
)'''
new = '''    "Decode step = rank-0 step wall time with a CUDA synchronize per step (measurement mode). "
    "Other agents' jobs loaded the host CPU during several of these runs (1-minute load average "
    "before each repeat in the last column; the node has 240 hardware threads, and the strict "
    "TP1 row stayed within 1.2 % between loads of 333 and 92); eager decode is CPU-launch "
    "bound, so rows measured under different loads carry an uncontrolled bias, and the min-max "
    "only measures run-to-run noise inside one engine. Native eager and native CUDA-graph rows "
    "are both given; production vLLM runs CUDA graphs, against which strict is 0.29x (TP1) to "
    "0.09x (TP4): strict decode does not scale with TP here because every layer adds rank-ordered "
    "all-gathers (row-parallel partials, vocab-tile stats and logits) to an eager, launch-bound "
    "step."
)
static["profile_notes"] = (
    "At TP>1 the rank-0 all-gather kernel time also absorbs waiting for the slower ranks (the "
    "collective blocks until every rank arrives), so those rows are not a clean breakdown. The "
    "strict decode step spends ~48 ms in kernels (TP1) inside a ~139 ms step: the rest is eager "
    "Python launch overhead (four separate q/k/v/gate GEMMs, 8 lm_head tiles, per-layer trees)."
)'''
assert old in s
s = s.replace(old, new, 1)
old = '''* Preemption/recompute under memory pressure was not exercised.
"""'''
new = '''* Preemption with recompute was exercised only in the 1600-block KV-pool runs at TP1 (13
  preemptions, decode == prefill and tokens == the normal pool); not at TP>1.
* The LoRA rule check on real weights (`worker_check_lora_rule`) runs at TP1 only; at TP2/TP4 the
  LoRA path is covered by decode == prefill, xrt TP == TP1 and the simulation.
"""'''
assert old in s
s = s.replace(old, new, 1)
old = '''   (`attention_config={"backend": ...}` + `register_backend`); null/evicted pages are id 0; the
   loader inspects `__init__` signatures;'''
new = '''   (`attention_config={"backend": ...}` + `register_backend`); null (never-allocated) pages are
   id 0, freed sliding-window pages keep stale ids (item 7); the loader inspects `__init__`
   signatures;'''
assert old in s
s = s.replace(old, new, 1)
old = '''  `StrictRolloutEngine.__init__` gained optional `enforce_eager` (native only) and
  `max_num_batched_tokens`; the `RolloutEngineLike` methods are unchanged.'''
new = '''  `StrictRolloutEngine.__init__` gained optional `enforce_eager` (native only),
  `max_num_batched_tokens` and `num_gpu_blocks_override`; the `RolloutEngineLike` methods are
  unchanged. Native `load_lora` now force-loads the adapter on every worker (one extra RPC).
* `rl_engine.models.muse_glimmer.e2e.sync.rollout_lora_sha256` may drop its
  `workers[*].lora_sha256` fallback: the top-level value is now always the workers' recomputed
  hash.'''
assert old in s
s = s.replace(old, new, 1)
old = '''result = {
    "tp4_engine": tp4,'''
new = '''result = {
    "commit_note": (
        "Binding numerics are unchanged since `a0a3ad4` (the first measured commit): later "
        "commits only add opt-in measurement code (evicted-page counter, preemption counters), "
        "run-script options, provenance fields, the collector and this report (the diff "
        "a0a3ad4..d1a1ec4 of runtime/vllm_strict.py adds only provenance lines, "
        "runtime/strict_ops.py and runtime/fallback_ops.py are unchanged, and the backend diff "
        "is the opt-in counter)."
    ),
    "tp4_engine": tp4,'''
assert old in s
s = s.replace(old, new, 1)
p.write_text(s)
print("ok")
