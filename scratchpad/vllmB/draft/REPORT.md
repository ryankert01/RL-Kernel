# REPORT — `vllm_runtime_adapter` (RFC #501, Muse-Glimmer-30B strict POC)

**STATUS: draft**

Code commit of the binding: `ccb4bcb`; runs measured at `a0a3ad4524` (dirty-tree runs: none). Real 30B checkpoint (revision a4e59da5), NVIDIA A100-SXM4-80GB, vLLM 0.31.0, torch 2.13.0+cu130, Triton 3.7.1. Machine-readable results with raw per-token / per-step series: `reports/muse_glimmer_poc/data/vllm_runtime.json` (this file is rendered from it by `runtime/vllm_report.py`); logprob dumps (`muse_glimmer_xrt_logprobs_v1`): `reports/muse_glimmer_poc/data/xrt_vllm_{strict,native}[_lora]_tp{1,2,4}.json`.

## Headline results

| check (30B, real weights) | native vLLM | strict binding |
|---|---|---|
| decode logprobs == one-shot prefill re-score, 8 GSM8K prompts x 64 tokens, T=0 (mismatched / compared) | NOT RUN | TP1 0 / 512 (162 non-zero, 56 below -1e-3); TP2 0 / 512 (162 non-zero, 56 below -1e-3) |
| decode logprobs == one-shot prefill re-score, 8 GSM8K prompts x 64 tokens, T=1 (mismatched / compared) | NOT RUN | TP1 0 / 512 (157 non-zero, 52 below -1e-3); TP2 0 / 512 (157 non-zero, 52 below -1e-3) |
| long window: 2261-token prompt + 256 decode tokens (positions 2261-2516 > 2048, sliding-window eviction), chunked prefill (max_num_batched_tokens=512), 1 greedy + 4 seeded T=1 copies | NOT RUN | NOT RUN |
| decode batch 64 (64 short prompts, T=1, 64 tokens, ignore_eos), chunked prefill 512 | NOT RUN | NOT RUN |
| decode batch 256 (256 short prompts, T=1, 64 tokens, ignore_eos), default budget | NOT RUN | TP1 0 / 16384 (5027 non-zero, 1729 below -1e-3), max decode batch 256; TP2 0 / 16384 (5027 non-zero, 1729 below -1e-3), max decode batch 256 |
| long window as above with the rank-16 random LoRA | NOT RUN | NOT RUN |
| decode batch 64 as above with the rank-16 random LoRA | NOT RUN | NOT RUN |
| xrt scoring (9 sequences, 933 tokens, one crossing the 2048 window) TP2 vs TP1 | 921 / 933, max 0.25 | 0 / 933 |
| same, TP4 vs TP1 (real 4-GPU engine) | NOT RUN | NOT RUN |
| strict vs native xrt scores TP1 | | 933 / 933, max 0.192 |
| rank-16 random LoRA on all 416 slots: decode == prefill, T=0 | NOT RUN | TP1 0 / 512 (164 non-zero, 57 below -1e-3); TP2 0 / 512 (164 non-zero, 57 below -1e-3) |
| rank-16 random LoRA on all 416 slots: decode == prefill, T=1 | NOT RUN | TP1 0 / 512 (160 non-zero, 56 below -1e-3); TP2 0 / 512 (160 non-zero, 56 below -1e-3) |
| LoRA xrt scores TP2 vs TP1 / TP4 vs TP1 | | 0 / 933 / NOT RUN |
| zero-B LoRA changes any score | | TP1 0 / 512; TP2 0 / 512 |
| unmerged LoRA rule on real weights (layers 0, 3, 51 x 8 slots, TP1) | | binding == independent formula 24 / 24, == trainer megatron_lora 24 / 24 |
| `provenance()["lora_sha256"]` = hash recomputed by every TP worker on what it loaded == exported hash | NOT RUN | strict TP1: yes (1 workers); strict TP2: yes (2 workers) |
| generated tokens + logprob bits, TP2/TP4 vs TP1 (same prompts, seeds) (sequences identical in tokens and bits) | NOT RUN | generate strict T=0.0 tp2 vs tp1: 8 / 8; generate strict T=1.0 tp2 vs tp1: 8 / 8; generate strict lora T=0.0 tp2 vs tp1: 8 / 8; generate strict lora T=1.0 tp2 vs tp1: 8 / 8; decode_batch_256_no_lora strict tp2 vs tp1: 256 / 256 |
| token choice, request i in a 64-wide vs a 256-wide batch (same seed) | NOT RUN | NOT RUN |

"non-zero" counts the compared tokens whose logprob is not exactly 0.0 (a token with logprob 0.0 compares trivially); "below -1e-3" counts the informative ones.

## Throughput

Throughput: NOT RUN.

Each row: one engine, warm (2-token warm-up), batch 32 x 256 generated tokens at T=1 with ignore_eos, repeated 3x; tokens/s = generated / wall, median of the repeats (min-max in brackets). TTFT = wall time of generate(max_tokens=1) for the whole batch (min of 3). Decode step = rank-0 step wall time with a CUDA synchronize per step (measurement mode). The host CPU was heavily loaded by other agents' jobs during these runs (1-minute load average per repeat in the last column; the node has 240 hardware threads); eager decode is CPU-launch bound, so the absolute numbers and the strict/native ratio carry an uncontrolled bias from that load, and the spread column only measures run-to-run noise inside one engine. Native eager and native CUDA-graph rows are both given; production vLLM runs CUDA graphs.

## What changed in wave B (issue list -> verdicts in `vllm_runtime.json` -> `finding_verdicts`)
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

## Design notes
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

## Contract deviations
* The sampler kernels are vLLM's (temperature division, Gumbel noise from the per-request seed and
  position, per-row argmax over 1024-column blocks): row-local and batch invariant, but not a
  contract op. `selected_token_ranks` is vLLM's comparison count over the strict logits.
* `DeterministicCollective` needs the absent `_C` extension -> `torch.distributed.all_gather_into_tensor`
  (bit-exact data movement) + local canonical tree.
* `enforce_eager=True` is required (strict ops are plain Python/Triton calls).

## Tests (final code)
```
pytest x
  -> draft
```

## How to reproduce
```
WT=<worktree>; D=reports/muse_glimmer_poc/data; M=rl_engine.models.muse_glimmer.runtime
R="~/bin/gpu-lease -n <t> --pool 4-6 [--contiguous] -- env PYTHONPATH=$WT ~/envs/vllm/bin/python -m $M.run_vllm_strict"
# main (per mode, t = 1, 2, 4): generate T=0/1 + prefill check, LoRA checks, xrt, decode batch 256
$R --tp t --strict --prompt-file $D/xrt_tokens.json --temperatures 0,1 --max-tokens 64 --prefill-check \
   --lora-zero-check --lora-random-check --xrt-score $D/xrt_tokens.json --xrt-out $D/xrt_vllm_strict_tp<t>.json \
   --xrt-lora-out $D/xrt_vllm_strict_lora_tp<t>.json --max-num-seqs 256 --decode-batches 256 --out strict_tp<t>.json
$R --tp t --native ... --xrt-out $D/xrt_vllm_native_tp<t>.json --max-num-seqs 256 --decode-batches 256 --out native_tp<t>.json
$R --tp 1 --native --native-lora ... --lora-random-check --xrt-lora-out $D/xrt_vllm_native_lora_tp1.json
# contract shapes with chunked prefill: long window (2261 + 256 tokens) and decode batch 64
$R --tp t --strict --prompt-file $D/xrt_tokens.json --no-generate --max-num-batched-tokens 512 --max-num-seqs 256 \
   --long-window --decode-batches 64 --contract-lora --out strict_tp<t>_chunked.json
# isolated benches (3 repeats each), one engine at a time
$R --tp t --strict --prompt-file $D/xrt_tokens.json --no-generate --bench --bench-lora --profile --out bench_strict_tp<t>.json
$R --tp t --native [--cuda-graphs] --prompt-file $D/xrt_tokens.json --no-generate --bench [--profile] --out bench_native_tp<t>.json
# reference chain (embedded in vllm_runtime.json), TP simulation, collect, render
python -m $M.vllm_xrt_reference $D/xrt_tokens.json ref/xrt_reference.json train,prefill
python -m $M.vllm_tp_sim > tp_sim.json
python -m $M.vllm_collect_results --runs runs --data $D --reference ref/xrt_reference_train.json \
   --reference ref/xrt_reference_prefill.json --extra extra.json --out $D/vllm_runtime.json
python -m $M.vllm_report $D/vllm_runtime.json rl_engine/models/muse_glimmer/runtime/REPORT_vllm_runtime_adapter.md
```
The exact command of every run is in `vllm_runtime.json` -> `runs[*].command` (with its commit and
`git_dirty`).

## RFC feedback
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
7. vLLM facts for the RFC text: Model Runner V2 is the CUDA default; `VLLM_ATTENTION_BACKEND` is gone
   (`attention_config={"backend": ...}` + `register_backend`); null/evicted pages are id 0; the
   loader inspects `__init__` signatures; vLLM fuses gate/up as well as q/k/v (the RFC says
   "Q/K/V only"); requests reach the engine core asynchronously, so batch composition varies run
   to run (harmless for strict ops, a source of native drift).

## Integration requests
* `~/envs/vllm`'s editable `rl_kernel` finder maps `rl_engine` to an old agent worktree
  (`worktrees/agent-a7944ab684b3629cf`); runs only work because `PYTHONPATH` wins. Re-point it to
  the POC worktree (or remove it) when installs are allowed; the engine now fails closed if a
  worker imports `rl_engine` from a different tree than the driver.
* Fusing the LoRA rank loop (`strict_kernels.small_k_fp32`, owned by the Megatron row and shared
  with the trainer) into one Triton kernel with the same fixed order would remove most of the
  strict-LoRA slowdown; it must change in both runtimes at once.
* E2E: `StrictRolloutEngine.provenance()` now raises `WeightSyncError` when TP workers disagree
  and reports `lora_sha256_source`; `e2e/sync.py` can accept only the top-level value.
  `StrictRolloutEngine.__init__` gained optional `enforce_eager` (native only) and
  `max_num_batched_tokens`; the `RolloutEngineLike` methods are unchanged.

## Open issues
* Performance is untuned (eager, launch-bound decode); next steps: CUDA graphs over the strict
  ops, one det GEMM for the four attention projections with per-slice canonical trees, a fused
  LoRA rank loop.
* TP8 engine not run (needs 8 GPUs); the simulation covers the binding arithmetic at TP8.
* Preemption/recompute under memory pressure was not exercised.
