"""Build the --extra JSON for vllm_collect_results (tests, simulation, TP4 summary, verdicts, prose).

usage: make_extra.py <out.json> <code_commit> <tests.json> <tp_sim.json|-> <status> <status_line>
"""

import json
import sys
from pathlib import Path

out, commit, tests_path, sim_path, status, status_line = sys.argv[1:7]
tests = json.loads(Path(tests_path).read_text())
sim = json.loads(Path(sim_path).read_text()) if sim_path != "-" else "NOT RUN"

WT = "/home/ubuntu/RL-Kernel/.claude/worktrees/wf_65f5032c-579-2"
D = "reports/muse_glimmer_poc/data"
M = "rl_engine.models.muse_glimmer.runtime"

finding_verdicts = [
    {
        "id": "vllm_runtime:R1",
        "verdict": "fixed",
        "evidence": "REPORT_vllm_runtime_adapter.md is now rendered from vllm_runtime.json by "
        "runtime/vllm_report.py (same text stored in report_markdown); it has an Integration "
        "requests section and a status line computed from the data.",
    },
    {
        "id": "vllm_runtime:R2",
        "verdict": "fixed",
        "evidence": "vllm_strict.check_strict_engine_config rejects prefix caching, "
        "kv_transfer_config (connectors/offload), kv_offloading_size, speculative decoding, "
        "kv_sharing_fast_prefill, decode/prefill CP, PP and processed logprobs modes at model "
        "init; copy_kv_cache_blocks_inplace (V1+V2 runners), KVBlockZeroer.zero_block_ids and "
        "GPUModelRunner._init_kv_zero_meta raise; batch-sharded sampling rejected; the false "
        "'same byte range' docstring is rewritten. Tests TestFailClosedEngineFeatures (11) fail "
        "on the pre-fix tree (AttributeError) and pass now; every engine run records the "
        "per-feature verdicts in worker_provenance[*].checks.",
    },
    {
        "id": "vllm_runtime:R3",
        "verdict": "fixed",
        "evidence": "compare_rows reports nonzero_tokens / tokens_lp_below_1e-3 / "
        "tokens_lp_below_0.1 / positive_tokens next to every mismatch count (REPORT headline "
        "shows them); contract_shapes holds the long-window (2261-token prompt + 256 decode "
        "tokens, positions 2261-2516, chunked prefill at max_num_batched_tokens=512), decode "
        "batch 64 and decode batch 256 runs at TP1/TP2 (strict and native) and TP4 strict, plus "
        "a 1600-block KV pool run (page reuse, 13 preemptions with recompute); the opt-in E2E "
        "test test_strict_engine_decode_equals_prefill now decodes the 2261-token prompt too.",
    },
    {
        "id": "vllm_runtime:R4",
        "verdict": "fixed",
        "evidence": "run_bench repeats every bench 3x in the same warm engine (median, "
        "min-max, spread %, host loadavg per repeat); a native CUDA-graph baseline "
        "(--cuda-graphs, enforce_eager=False) is measured next to native eager; the '+-15 %' "
        "claim is gone. The host was heavily loaded by other agents during every bench (loadavg "
        "recorded per repeat), which the report states; an idle host was not available.",
    },
    {
        "id": "vllm_runtime:R5",
        "verdict": "fixed",
        "evidence": "MuseGlimmerForCausalLM.compute_logits now returns the strict fp32 zhat of all "
        "202048 columns (8 canonical det-GEMM tiles + the O7 kernel's own soft cap "
        "launch_softcap, exact TP all-gather); the native cuBLAS lm_head is no longer run; the "
        "sampled-token logp reuses the same fp32 tiles. Unit test "
        "TestStrictSamplingLogits (logits[target] - logsumexp == strict logp; batch invariant; "
        "logp(argmax) <= 0) and the TP simulation (logits and argmax at TP2/4/8 == TP1) pass; "
        "engine evidence: token_selection_invariance (across TP and across 64- vs 256-wide "
        "batches) in vllm_runtime.json: native token choices diverge across TP, batch "
        "composition and KV pressure, strict ones never do.",
    },
    {
        "id": "vllm_runtime:R6",
        "verdict": "fixed",
        "evidence": "check_sampling_request raises on top-k/top-p, logprobs>0, logprob_token_ids "
        "and processed logprobs modes; check_prompt_logprobs_request raises on prompt_logprobs "
        "!= 0; PromptLogprobsWorker.compute_prompt_token_id_logprobs raises when token-id "
        "scoring is requested; FlashInfer/XPU fused samplers are disabled. Tests "
        "TestFailClosedSampling (9) fail on the pre-fix tree and pass now.",
    },
    {
        "id": "vllm_runtime:R7",
        "verdict": "fixed",
        "evidence": "compare_xrt labels come from the dump (xrt_label: mode, LoRA marker, TP, "
        "reference-chain mode) or the comparison key; decode_vs_prefill keeps prefill_logprobs "
        "as floats; LoRA generate records use the same schema (ids, prompt_lens, "
        "finish_reasons, logprobs_hex); kv_ownership.kv_heads_replicated = tp > 2 (TP4/8 only). "
        "Test TestCollectLabels fails on the pre-fix tree and passes now.",
    },
    {
        "id": "vllm_runtime:R8",
        "verdict": "fixed",
        "evidence": "worker_check_lora_rule's formula calls ops.det_gemm.fast_det_gemm directly "
        "on unpadded shapes (not the binding's op table or _padded_partial) with a Python rank "
        "loop, plus an fp32-golden (fallback_ops.det_gemm) tolerance view. Test "
        "TestLoRARuleCheckIsIndependent breaks _padded_partial by 1 ulp-scale and requires "
        "binding_equals_formula == False: it fails on the pre-fix tree (the old formula reused "
        "_padded_partial) and passes now.",
    },
    {
        "id": "vllm_runtime:R9",
        "verdict": "fixed",
        "evidence": "The protocol test asserts v <= 0.0 again (c88607f launches the softcap "
        "kernels with enable_fp_fusion=False). Every run now reports positive_tokens; see the "
        "decode_vs_prefill and xrt positive_logprobs counts in vllm_runtime.json.",
    },
    {
        "id": "vllm_runtime:R10",
        "verdict": "fixed",
        "evidence": "Native load_lora force-loads the adapter (LLMEngine.add_lora) and every TP "
        "worker rebuilds the canonical state dict from vLLM's LoRAModel tensors "
        "(vllm_worker_tools.canonical_state_from_lora_model: qkv_proj/gate_up_proj unpacked, "
        "output_gate_proj -> gate_proj, scaling must be 1) and recomputes lora_sha256; "
        "provenance reports that recomputed value plus slots_nonzero_b (416 expected). Tests "
        "TestNativeLoRARecomputed fail on the pre-fix tree and pass now.",
    },
    {
        "id": "vllm_runtime:R11",
        "verdict": "fixed",
        "evidence": "vllm_xrt_reference.py records commit, git_dirty and command; it was re-run "
        "at the final code and both outputs (train, prefill) are embedded in vllm_runtime.json "
        "(xrt.reference_chain_*.document) and compared with every vLLM strict dump. Every run "
        "file records git_dirty + git_status_porcelain; StrictRolloutEngine fails at construction "
        "unless every worker imports rl_engine from the driver's file (check_worker_paths, test "
        "TestWorkerPaths). The stale editable install itself is outside this row (integration "
        "request).",
    },
    {
        "id": "vllm_runtime:R12",
        "verdict": "fixed (reporting) / not_fixed_with_reason (kernel fusion)",
        "evidence": "The bench is repeated with the rank-16 LoRA loaded (--bench-lora): tokens/s "
        "and peak memory per rank are in the throughput table. Fusing the rank loop was not "
        "done: the loop is strict_kernels.small_k_fp32, shared with the trainer (megatron row "
        "owns it) and its fixed order is the cross-runtime contract, so a fused kernel must be "
        "introduced there for both runtimes at once (integration request).",
    },
    {
        "id": "vllm_runtime:R13",
        "verdict": "fixed",
        "evidence": "A real 4-GPU TP4 engine run (strict generate + prefill check + xrt + LoRA "
        "xrt + contract shapes) is in vllm_runtime.json (tp4_engine, xrt "
        "strict_tp4_vs_strict_tp1). The simulation now calls the binding's own "
        "strict_attention_forward / strict_mlp_forward / strict_logits on stand-in modules.",
    },
    {
        "id": "e2e_bridge:R4",
        "verdict": "fixed",
        "evidence": "worker_load_lora recomputes muse_glimmer_lora_sha256_v1 over the bf16 tensors "
        "the rank read and shards, refuses a mismatch with the declared hash (previous adapter "
        "kept) and stores the RECOMPUTED value; StrictRolloutEngine.provenance()['lora_sha256'] "
        "is the value all TP workers recomputed (aggregate_worker_lora_hashes raises "
        "WeightSyncError on disagreement among workers or with the driver's own hash), with "
        "lora_sha256_per_worker and lora_sha256_source; native mode does the same from vLLM's "
        "loaded LoRA tensors. Tests TestLoRAHashRecomputed fail on the pre-fix tree and pass now.",
    },
]

static = {}
static["throughput_notes"] = (
    "Each row: one engine, warm (2-token warm-up), batch 32 x 256 generated tokens at T=1 with "
    "ignore_eos, repeated 3x; tokens/s = generated / wall, median of the repeats (min-max in "
    "brackets). TTFT = wall time of generate(max_tokens=1) for the whole batch (min of 3). "
    "Decode step = rank-0 step wall time with a CUDA synchronize per step (measurement mode). "
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
)
static["what_changed"] = """## What changed in wave B (issue list -> verdicts in `vllm_runtime.json` -> `finding_verdicts`)
* **Per-rank attention heads.** TP>1 attention now runs on the rank's own heads: vLLM's native
  `Attention` (32/t query heads, max(1, 2/t) KV heads: TP2 16/1, TP4 8/1 with the KV head
  replicated on two ranks) and the strict op's new `num_q_heads`/`num_kv_heads`. The
  KV-replicating `Attention` factory and the four q/k/v/gate all-gathers are deleted; KV memory
  per rank and attention work drop by t. Bits are unchanged (heads never interact in O5): xrt
  scores and generated tokens at TP2 and TP4 equal TP1 byte for byte (headline table).
* **Strict token selection (R5).** `compute_logits` returns the strict fp32 `zhat` of the 202048
  real columns (8 canonical det-GEMM tiles, the O7 kernel's own soft cap, exact TP all-gather);
  vLLM V2's seeded Gumbel sampler / greedy argmax picks the token from it; the native cuBLAS
  lm_head is no longer run; the sampled-token logprob reuses the same fp32 tiles.
* **Fail closed (R2, R6).** Engine features that address KV pages in the native layout
  (prefix caching, KV connectors / transfer / offloading, speculative decoding, KV sharing,
  decode/prefill context parallelism, KV-block zeroing) are rejected at model init and the
  native page writers raise; requests that would return native numbers (top-k/top-p,
  `logprobs>0`, `prompt_logprobs>0`, token-id scoring, processed logprob modes) raise.
* **LoRA identity (e2e_bridge:R4, R10).** `provenance()["lora_sha256"]` is the hash every TP
  worker recomputed on the tensors it actually loaded (strict: the full tensors it read and
  sharded; native: the tensors vLLM's LoRA manager holds); disagreement raises.
* **Evidence (R3, R4, R7, R8, R11, R13).** Contract shapes (2261-token prompt + 256 decode
  tokens across the window with eviction and chunked prefill; decode batches 64 and 256),
  informative-token counts next to every mismatch count, 3-repeat benches with a CUDA-graph
  native baseline, independent LoRA-rule formula, dirty-tree flags, worker import-path check,
  reference chain re-run and embedded, real TP4 engine.
"""
static["design_notes"] = """## Design notes
Hooks: FLASH_ATTN enum override -> strict backend (owns the KV write; passes the rank's head
counts); `ParallelLMHead` padded to 202752 rows (rank r of t owns vocab tiles [8r/t, 8(r+1)/t));
`MuseGlimmerForCausalLM.__init__` (keeps the `(*, vllm_config, prefix)` signature vLLM inspects)
binds and checks (native class, config pins, eager, bf16 KV, KV-page features, per-rank head
geometry, all 15 ops strict); `compute_logits` -> strict sampling logits; V2 `Sampler.__call__`
(fail-closed request checks, strict selected-token logp from the stashed fp32 tiles) and
`Sampler.__init__` (Gumbel path only); `compute_prompt_logprobs_with_chunking` (prompt-token
column = strict O7, ranks from the strict logits) and `PromptLogprobsWorker.
compute_prompt_token_id_logprobs` (raises); `embed_input_ids` (O1, exact TP gather); decoder layer
(O2 incl. fused [F1]); attention (`strict_attention_forward`: four det GEMMs on the rank's
canonical slices of fused `qkv_proj` + `output_gate_proj`, O4 on the rank's heads, gate side
channel into O5, row-parallel `o_proj` tree); MLP (`strict_mlp_forward`); final norm; native KV
page writers (raise). Row-parallel GEMMs: fp32 sub-tree partial per rank ->
`all_gather_into_tensor` (data movement) -> canonical tree -> one cast.

KV layout: vLLM 0.31 hands each layer a logical `(B, H, N, 2D)` view of an `LBNHC` buffer; the
backend re-views the dense per-layer region as `(2, B, N, H, D)` (K pages then V pages) and owns
the write. A strict page is NOT the native page with the same id, which is why every native
page-addressed feature is rejected (above).

Sliding-window eviction in vLLM 0.31 (measured, not assumed): the scheduler frees a request's
pages whose keys are all at positions <= (first position computed this step) - 2048
(`SlidingWindowManager.remove_skipped_blocks`) and may hand them to other requests, but the
worker block table of a running request is append-only, so the freed pages keep their stale ids
there; null id 0 only marks never-allocated slots. The opt-in counter
(`--count-null-pages`) saw 0 in-range null entries during the long-window runs
(`contract_shapes[*].null_pages_read`). This is safe because the strict kernel never reads a
block-table entry or a K/V slot at a position <= p_min - 2048 (p_min = the smallest query
position of the tile; masked loads, `k_pos > win_lo`), which is exactly the set vLLM may have
freed; keys between p_min - 2048 and a later row's window start are still owned by the request
and are masked per row. The tiny-KV-pool runs (1600 blocks: freed pages are reallocated to other
requests and requests are preempted and recomputed) check this end to end.
"""
static["contract_deviations"] = """## Contract deviations
* The sampler kernels are vLLM's (temperature division, Gumbel noise from the per-request seed and
  position, per-row argmax over 1024-column blocks): row-local and batch invariant, but not a
  contract op. `selected_token_ranks` is vLLM's comparison count over the strict logits.
* `DeterministicCollective` needs the absent `_C` extension -> `torch.distributed.all_gather_into_tensor`
  (bit-exact data movement) + local canonical tree.
* `enforce_eager=True` is required (strict ops are plain Python/Triton calls).
"""
static["rfc_feedback"] = """## RFC feedback
1. **State the LoRA arithmetic exactly** (cross-runtime contract): `z = bf16(acc32(A x))`,
   `y = bf16(acc32(W x) + sum_{k<r} B[:,k] z[k])` with a sequential fp32 rank loop; A replicated /
   B row-sharded for column-parallel, A column-sharded (+ canonical K tree) / B replicated for
   row-parallel. `bf16(bf16(Wx) + bf16(BAx))` is a different function.
2. **The rollout binding must own the KV write and the physical KV layout**, and the RFC should
   list the engine features that address KV pages by block id (prefix caching copy-on-write,
   KV connectors / offloading, KV-block zeroing, speculative decoding, context parallelism) as
   forbidden in strict mode unless the binding re-implements them on its own layout.
3. **Token selection must read strict logits.** Without it the sampled token depends on cuBLAS
   lm_head bits at near-ties (batch composition, TP). This POC samples from strict fp32 `zhat`;
   the sampler's own kernels (temperature, Gumbel noise from (seed, position), argmax) should be
   named in the contract as row-local and batch invariant, or replaced.
4. **Fail closed on any non-strict number a caller can request** (top-k columns, token-id
   scoring, processed logprobs), and on any non-strict op; list every op implementation in
   provenance. vLLM builds the model under a non-default device and dtype; binding code that
   creates tensors must pass both.
5. **Weight-sync identity is a worker-side recomputation**: every TP rank hashes what it actually
   loaded and the engine reports the agreed value; an engine echo of the declared hash is not
   evidence. vLLM's RPC channel does not carry nested tensors (file + hash transport).
6. **O5 head counts**: per-rank `H_Q`/`H_KV` (TP2 16/1, TP4 8/1, TP8 4/1 with the KV head
   replicated) are sufficient and bit-identical to TP1; the KV-replication workaround is not needed.
7. **O5 eviction wording**: "pages whose every key position <= p-W may be freed; the kernel masks
   them anyway" is not enough for vLLM 0.31: freed pages keep their stale ids in the running
   request's block table and can be reallocated to other requests. The requirement must be that
   the kernel never reads a block-table entry or a K/V slot at a position <= p_min - W (the
   strict kernel satisfies it with masked loads); masking the scores alone would read reused
   memory.
8. vLLM facts for the RFC text: Model Runner V2 is the CUDA default; `VLLM_ATTENTION_BACKEND` is gone
   (`attention_config={"backend": ...}` + `register_backend`); null (never-allocated) pages are
   id 0, freed sliding-window pages keep stale ids (item 7); the loader inspects `__init__`
   signatures; vLLM fuses gate/up as well as q/k/v (the RFC says
   "Q/K/V only"); requests reach the engine core asynchronously, so batch composition varies run
   to run (harmless for strict ops, a source of native drift).
"""
static["integration_requests"] = """## Integration requests
* `~/envs/vllm`'s editable `rl_kernel` finder maps `rl_engine` to an old agent worktree
  (`worktrees/agent-a7944ab684b3629cf`); runs only work because `PYTHONPATH` wins. Re-point it to
  the POC worktree (or remove it) when installs are allowed; the engine now fails closed if a
  worker imports `rl_engine` from a different tree than the driver.
* Fusing the LoRA rank loop (`strict_kernels.small_k_fp32`, owned by the Megatron row and shared
  with the trainer) into one Triton kernel with the same fixed order would remove most of the
  strict-LoRA slowdown; it must change in both runtimes at once.
* E2E: `StrictRolloutEngine.provenance()` now raises `WeightSyncError` when TP workers disagree
  and reports `lora_sha256_source`; `e2e/sync.py` can accept only the top-level value.
  `StrictRolloutEngine.__init__` gained optional `enforce_eager` (native only),
  `max_num_batched_tokens` and `num_gpu_blocks_override`; the `RolloutEngineLike` methods are
  unchanged. Native `load_lora` now force-loads the adapter on every worker (one extra RPC).
* `rl_engine.models.muse_glimmer.e2e.sync.rollout_lora_sha256` may drop its
  `workers[*].lora_sha256` fallback: the top-level value is now always the workers' recomputed
  hash.
"""
static["open_issues"] = """## Open issues
* Performance is untuned (eager, launch-bound decode); next steps: CUDA graphs over the strict
  ops, one det GEMM for the four attention projections with per-slice canonical trees, a fused
  LoRA rank loop.
* TP8 engine not run (needs 8 GPUs); the simulation covers the binding arithmetic at TP8.
* Preemption with recompute was exercised only in the 1600-block KV-pool runs at TP1 (13
  preemptions, decode == prefill and tokens == the normal pool); not at TP>1.
* The LoRA rule check on real weights (`worker_check_lora_rule`) runs at TP1 only; at TP2/TP4 the
  LoRA path is covered by decode == prefill, xrt TP == TP1 and the simulation.
"""

tests_lines = ["## Tests (final code)", "```"]
for t in tests:
    tests_lines.append(t["command"])
    tests_lines.append(f"  -> {t['outcome']}")
tests_lines.append("```")
static["tests_text"] = "\n".join(tests_lines) + "\n"
static["how_to_reproduce"] = f"""## How to reproduce
```
WT=<worktree>; D=reports/muse_glimmer_poc/data; M={M}
R="~/bin/gpu-lease -n <t> --pool 4-6 [--contiguous] -- env PYTHONPATH=$WT ~/envs/vllm/bin/python -m $M.run_vllm_strict"
# main (per mode, t = 1, 2, 4): generate T=0/1 + prefill check, LoRA checks, xrt, decode batch 256
$R --tp t --strict --prompt-file $D/xrt_tokens.json --temperatures 0,1 --max-tokens 64 --prefill-check \\
   --lora-zero-check --lora-random-check --xrt-score $D/xrt_tokens.json --xrt-out $D/xrt_vllm_strict_tp<t>.json \\
   --xrt-lora-out $D/xrt_vllm_strict_lora_tp<t>.json --max-num-seqs 256 --decode-batches 256 --out strict_tp<t>.json
$R --tp t --native ... --xrt-out $D/xrt_vllm_native_tp<t>.json --max-num-seqs 256 --decode-batches 256 --out native_tp<t>.json
$R --tp 1 --native --native-lora ... --lora-random-check --xrt-lora-out $D/xrt_vllm_native_lora_tp1.json
# contract shapes with chunked prefill: long window (2261 + 256 tokens) and decode batch 64
$R --tp t --strict --prompt-file $D/xrt_tokens.json --no-generate --max-num-batched-tokens 512 --max-num-seqs 256 \\
   --long-window --decode-batches 64 --contract-lora --out strict_tp<t>_chunked.json
# isolated benches (3 repeats each), one engine at a time
$R --tp t --strict --prompt-file $D/xrt_tokens.json --no-generate --bench --bench-lora --profile --out bench_strict_tp<t>.json
$R --tp t --native [--cuda-graphs] --prompt-file $D/xrt_tokens.json --no-generate --bench [--profile] --out bench_native_tp<t>.json
# reference chain (embedded in vllm_runtime.json), TP simulation, collect, render
python -m $M.vllm_xrt_reference $D/xrt_tokens.json ref/xrt_reference.json train,prefill
python -m $M.vllm_tp_sim > tp_sim.json
python -m $M.vllm_collect_results --runs runs --data $D --reference ref/xrt_reference_train.json \\
   --reference ref/xrt_reference_prefill.json --extra extra.json --out $D/vllm_runtime.json
python -m $M.vllm_report $D/vllm_runtime.json rl_engine/models/muse_glimmer/runtime/REPORT_vllm_runtime_adapter.md
```
The exact command of every run is in `vllm_runtime.json` -> `runs[*].command` (with its commit and
`git_dirty`).
"""

RUNS = Path(tests_path).parent / "runs"
tp4 = {"status": "NOT RUN", "runs": []}
for name in ("strict_tp4", "strict_tp4_chunked", "bench_strict_tp4", "bench_native_tp4",
             "bench_native_cg_tp4"):
    path = RUNS / f"{name}.json"
    if not path.exists():
        continue
    run = json.loads(path.read_text())
    entry = {"run": f"runs/{name}.json", "status": run.get("status"), "commit": run.get("commit"),
             "command": run.get("command"), "error": run.get("error")}
    for g in run.get("generate", []):
        c = g.get("decode_vs_prefill") or {}
        entry.setdefault("decode_vs_prefill", []).append(
            {"temperature": g["temperature"], "mismatched": c.get("mismatched"),
             "tokens": c.get("tokens")}
        )
    for c in run.get("contract_shapes", []):
        d = c["decode_vs_prefill"]
        entry.setdefault("contract_shapes", []).append(
            {"label": c["label"], "mismatched": d["mismatched"], "tokens": d["tokens"],
             "max_decode_batch": c["engine_steps"]["max_decode_batch"]}
        )
    if run.get("lora_random"):
        entry["lora_hash_identity"] = run["lora_random"].get("hash_identity")
        entry["lora_sha256_per_worker"] = run["lora_random"].get("lora_sha256_per_worker")
    tp4["runs"].append(entry)
if tp4["runs"]:
    ok = all(r["status"] == "completed" for r in tp4["runs"])
    tp4["status"] = "completed" if ok else "partial"
    tp4["pass_criterion"] = (
        "xrt_vllm_strict_tp4.json 'sequences' == xrt_vllm_strict_tp1.json 'sequences' (and the "
        "_lora_ pair); decode_vs_prefill mismatched == 0; generated tokens == TP1 "
        "(token_selection_invariance.across_tp); see xrt.comparisons.strict_tp4_vs_strict_tp1"
    )
    tp4["lease"] = "~/bin/gpu-lease -n 4 --pool 4-7 --contiguous --timeout 14400"

result = {
    "commit_note": (
        "Binding numerics are unchanged since `a0a3ad4` (the first measured commit): later "
        "commits only add opt-in measurement code (evicted-page counter, preemption counters), "
        "run-script options, provenance fields, the collector and this report (the diff "
        "a0a3ad4..d62650b of runtime/vllm_strict.py adds only provenance lines, "
        "runtime/strict_ops.py and runtime/fallback_ops.py are unchanged, and the backend diff "
        "is the opt-in counter)."
    ),
    "tp4_engine": tp4,
    "status": status,
    "status_line": status_line,
    "commit": commit,
    "tests": tests,
    "tp_simulation": sim,
    "finding_verdicts": finding_verdicts,
    "report_static": static,
    "plot_hints": [
        "Bar: mismatched / compared tokens for native vs strict (decode vs prefill, TP2/TP4 vs "
        "TP1, contract shapes) -> generate[*].decode_vs_prefill, contract_shapes[*]."
        "decode_vs_prefill, xrt.comparisons.*",
        "Per-token |delta logp| lines (native TP2 vs TP1; native decode vs prefill in the long "
        "window): xrt.comparisons.native_tp2_vs_native_tp1.per_sequence[*].abs_diff_per_token, "
        "contract_shapes[label=long_window_no_lora, mode=native].decode_vs_prefill."
        "abs_diff_per_token",
        "Grouped bars with error bars (min-max of 3 repeats): throughput[*].tokens_per_s_repeats "
        "by label (strict/native eager/native CUDA graphs/strict+LoRA, TP1/TP2/TP4)",
        "Line: decode batch width per engine step: contract_shapes[*].engine_steps.series[*].reqs",
        "Line: per-step decode ms: throughput[*].step_series_rank0[*].ms where tokens == reqs",
        "Stacked bar: top CUDA kernels of one decode / prefill step: profile[*].decode.top_ops",
        "Table: token_selection_invariance.across_tp / across_batch identical sequences, strict "
        "vs native",
    ],
}
Path(out).write_text(json.dumps(result, indent=1))
print("ok", out)
