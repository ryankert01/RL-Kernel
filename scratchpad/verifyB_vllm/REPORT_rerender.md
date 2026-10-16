# REPORT — `vllm_runtime_adapter` (RFC #501, Muse-Glimmer-30B strict POC)

**STATUS: complete — every requested run done on the final binding code (TP1, TP2 and a real 4-GPU TP4 engine; strict and native); TP8 engine NOT RUN (needs 8 GPUs; the single-GPU simulation covers TP8)**

Code commit of the binding: `d62650b`; runs measured at `040ec76241`, `0aabaf1162`, `122742ed18`, `3409cf6fa3`, `38ed927131`, `39739860b9`, `4325ba2cb7`, `988bacfcfd`, `a0a3ad4524`, `a719350b61`, `ca7c26fd20` (dirty-tree runs: none). Real 30B checkpoint (revision a4e59da5), NVIDIA A100-SXM4-80GB, vLLM 0.31.0, torch 2.13.0+cu130, Triton 3.7.1. Machine-readable results with raw per-token / per-step series: `reports/muse_glimmer_poc/data/vllm_runtime.json` (this file is rendered from it by `runtime/vllm_report.py`); logprob dumps (`muse_glimmer_xrt_logprobs_v1`): `reports/muse_glimmer_poc/data/xrt_vllm_{strict,native}[_lora]_tp{1,2,4}.json`.

Binding numerics are unchanged since `a0a3ad4` (the first measured commit): later commits only add opt-in measurement code (evicted-page counter, preemption counters), run-script options, provenance fields, the collector and this report (the diff a0a3ad4..d62650b of runtime/vllm_strict.py adds only provenance lines, runtime/strict_ops.py and runtime/fallback_ops.py are unchanged, and the backend diff is the opt-in counter).

## Headline results

| check (30B, real weights) | native vLLM | strict binding |
|---|---|---|
| decode logprobs == one-shot prefill re-score, 8 GSM8K prompts x 64 tokens, T=0 (mismatched / compared) | TP1 144 / 512 (263 non-zero, 56 below -1e-3); TP2 155 / 512 (263 non-zero, 56 below -1e-3) | TP1 0 / 512 (162 non-zero, 56 below -1e-3); TP2 0 / 512 (162 non-zero, 56 below -1e-3); TP4 0 / 512 (162 non-zero, 56 below -1e-3) |
| decode logprobs == one-shot prefill re-score, 8 GSM8K prompts x 64 tokens, T=1 (mismatched / compared) | TP1 157 / 512 (264 non-zero, 52 below -1e-3); TP2 148 / 512 (262 non-zero, 52 below -1e-3) | TP1 0 / 512 (157 non-zero, 52 below -1e-3); TP2 0 / 512 (157 non-zero, 52 below -1e-3); TP4 0 / 512 (157 non-zero, 52 below -1e-3) |
| long window: 2261-token prompt + 256 decode tokens (positions 2261-2516 > 2048; vLLM frees the pages before the window), chunked prefill (max_num_batched_tokens=512), 1 greedy + 4 seeded T=1 copies | TP1 779 / 1280 (1042 non-zero, 525 below -1e-3), max decode batch 5; TP2 792 / 1280 (1053 non-zero, 511 below -1e-3), max decode batch 5 | TP1 0 / 1280 (853 non-zero, 520 below -1e-3), max decode batch 5; TP2 0 / 1280 (853 non-zero, 520 below -1e-3), max decode batch 5; TP4 0 / 1280 (853 non-zero, 520 below -1e-3), max decode batch 5 |
| decode batch 64 (64 short prompts, T=1, 64 tokens, ignore_eos), chunked prefill 512 | TP1 1050 / 4096 (2085 non-zero, 425 below -1e-3), max decode batch 64; TP2 1025 / 4096 (2094 non-zero, 421 below -1e-3), max decode batch 64 | TP1 0 / 4096 (1246 non-zero, 426 below -1e-3), max decode batch 64; TP2 0 / 4096 (1246 non-zero, 426 below -1e-3), max decode batch 64; TP4 0 / 4096 (1246 non-zero, 426 below -1e-3), max decode batch 64 |
| decode batch 256 (256 short prompts, T=1, 64 tokens, ignore_eos), default budget | TP1 4517 / 16384 (8316 non-zero, 1741 below -1e-3), max decode batch 256; TP2 4557 / 16384 (8288 non-zero, 1734 below -1e-3), max decode batch 256 | TP1 0 / 16384 (5027 non-zero, 1729 below -1e-3), max decode batch 256; TP2 0 / 16384 (5027 non-zero, 1729 below -1e-3), max decode batch 256; TP4 0 / 16384 (5027 non-zero, 1729 below -1e-3), max decode batch 256 |
| long window as above with the rank-16 random LoRA | NOT RUN | TP1 0 / 1280 (863 non-zero, 432 below -1e-3), max decode batch 5; TP2 0 / 1280 (863 non-zero, 432 below -1e-3), max decode batch 5; TP4 0 / 1280 (863 non-zero, 432 below -1e-3), max decode batch 5 |
| decode batch 64 as above with the rank-16 random LoRA | NOT RUN | TP1 0 / 4096 (1261 non-zero, 413 below -1e-3), max decode batch 64; TP2 0 / 4096 (1261 non-zero, 413 below -1e-3), max decode batch 64; TP4 0 / 4096 (1261 non-zero, 413 below -1e-3), max decode batch 64 |
| long window, 1 greedy + 8 T=1 copies, in a 1600-block KV pool (freed pages reused, preemption + recompute) | TP1 1400 / 2304 (1843 non-zero, 916 below -1e-3), max decode batch 2, preempted 0 / resumed 0 | TP1 0 / 2304 (1505 non-zero, 920 below -1e-3), max decode batch 2, preempted 0 / resumed 0 |
| decode batch 64 in the same 1600-block KV pool | TP1 1073 / 4096 (2082 non-zero, 425 below -1e-3), max decode batch 44, preempted 13 / resumed 0 | TP1 0 / 4096 (1246 non-zero, 426 below -1e-3), max decode batch 44, preempted 13 / resumed 0 |
| xrt scoring (9 sequences, 933 tokens, one crossing the 2048 window) TP2 vs TP1 | 921 / 933, max 0.214 | 0 / 933 |
| same, TP4 vs TP1 (real 4-GPU engine) | NOT RUN | 0 / 933 |
| strict tp1 vs reference chain prefill | | 0 / 933 |
| strict tp1 vs reference chain train | | 0 / 933 |
| strict tp2 vs reference chain prefill | | 0 / 933 |
| strict tp2 vs reference chain train | | 0 / 933 |
| strict tp4 vs reference chain prefill | | 0 / 933 |
| strict tp4 vs reference chain train | | 0 / 933 |
| strict vs native xrt scores TP1 | | 933 / 933, max 0.192 |
| rank-16 random LoRA on all 416 slots: decode == prefill, T=0 | TP1 168 / 512 (272 non-zero, 59 below -1e-3) | TP1 0 / 512 (164 non-zero, 57 below -1e-3); TP2 0 / 512 (164 non-zero, 57 below -1e-3); TP4 0 / 512 (164 non-zero, 57 below -1e-3) |
| rank-16 random LoRA on all 416 slots: decode == prefill, T=1 | TP1 158 / 512 (272 non-zero, 57 below -1e-3) | TP1 0 / 512 (160 non-zero, 56 below -1e-3); TP2 0 / 512 (160 non-zero, 56 below -1e-3); TP4 0 / 512 (160 non-zero, 56 below -1e-3) |
| LoRA xrt scores TP2 vs TP1; TP4 vs TP1 | | TP2 0 / 933; TP4 0 / 933 |
| zero-B LoRA changes any score | | TP1 0 / 512; TP2 0 / 512 |
| unmerged LoRA rule on real weights (layers 0, 3, 51 x 8 slots, TP1) | | binding == independent formula 24 / 24, == trainer megatron_lora 24 / 24 |
| `provenance()["lora_sha256"]` = hash recomputed by every TP worker on what it loaded == exported hash | native TP1: yes (workers recomputing: 1) | strict TP1: yes (workers recomputing: 1); strict TP2: yes (workers recomputing: 2); strict TP4: yes (workers recomputing: 4) |
| generated tokens + logprob bits, TP2/TP4 vs TP1 (same prompts, seeds) (identical sequences: same tokens; same tokens and logprob bits) | generate native T=0.0 tp2 vs tp1: tokens 8, tokens+bits 0 / 8; generate native T=1.0 tp2 vs tp1: tokens 8, tokens+bits 0 / 8; decode_batch_256_no_lora native tp2 vs tp1: tokens 244, tokens+bits 0 / 256; decode_batch_64_no_lora native tp2 vs tp1: tokens 63, tokens+bits 0 / 64; long_window_no_lora native tp2 vs tp1: tokens 0, tokens+bits 0 / 5 | generate strict T=0.0 tp2 vs tp1: tokens 8, tokens+bits 8 / 8; generate strict T=1.0 tp2 vs tp1: tokens 8, tokens+bits 8 / 8; generate strict lora T=0.0 tp2 vs tp1: tokens 8, tokens+bits 8 / 8; generate strict lora T=1.0 tp2 vs tp1: tokens 8, tokens+bits 8 / 8; generate strict T=0.0 tp4 vs tp1: tokens 8, tokens+bits 8 / 8; generate strict T=1.0 tp4 vs tp1: tokens 8, tokens+bits 8 / 8; generate strict lora T=0.0 tp4 vs tp1: tokens 8, tokens+bits 8 / 8; generate strict lora T=1.0 tp4 vs tp1: tokens 8, tokens+bits 8 / 8; decode_batch_256_no_lora strict tp2 vs tp1: tokens 256, tokens+bits 256 / 256; decode_batch_256_no_lora strict tp4 vs tp1: tokens 256, tokens+bits 256 / 256; decode_batch_64_no_lora strict tp2 vs tp1: tokens 64, tokens+bits 64 / 64; decode_batch_64_no_lora strict tp4 vs tp1: tokens 64, tokens+bits 64 / 64; decode_batch_64_random_lora strict tp2 vs tp1: tokens 64, tokens+bits 64 / 64; decode_batch_64_random_lora strict tp4 vs tp1: tokens 64, tokens+bits 64 / 64; long_window_no_lora strict tp2 vs tp1: tokens 5, tokens+bits 5 / 5; long_window_no_lora strict tp4 vs tp1: tokens 5, tokens+bits 5 / 5; long_window_random_lora strict tp2 vs tp1: tokens 5, tokens+bits 5 / 5; long_window_random_lora strict tp4 vs tp1: tokens 5, tokens+bits 5 / 5 |
| token choice, request i in a 64-wide vs a 256-wide batch (same seed) (identical sequences: same tokens; same tokens and logprob bits) | native tp1_no_lora: tokens 62, tokens+bits 0 / 64; native tp2_no_lora: tokens 62, tokens+bits 0 / 64 | strict tp1_no_lora: tokens 64, tokens+bits 64 / 64; strict tp2_no_lora: tokens 64, tokens+bits 64 / 64; strict tp4_no_lora: tokens 64, tokens+bits 64 / 64 |
| token choice, tiny KV pool (reuse, preemption) vs normal pool (same seed) (identical sequences: same tokens; same tokens and logprob bits) | native tp1 long_window_no_lora: tokens 0, tokens+bits 0 / 5; native tp1 decode_batch_64_no_lora: tokens 62, tokens+bits 0 / 64 | strict tp1 long_window_no_lora: tokens 5, tokens+bits 5 / 5; strict tp1 decode_batch_64_no_lora: tokens 64, tokens+bits 64 / 64 |

"non-zero" counts the compared tokens whose logprob is not exactly 0.0 (a token with logprob 0.0 compares trivially); "below -1e-3" counts the informative ones.

## Throughput

| config | tokens/s median (min-max, n) | TTFT whole batch (ms) | decode step median (ms) | peak alloc / rank (GiB) | KV cache / rank (GiB, pages) | host load 1m |
|---|---|---|---|---|---|---|
| native tp1 cuda-graphs | 720.1 (720.1-720.5, n=3) | 901 | 42.2 | 70.3 | 17.5, 88346 | 1-2 |
| native tp2 cuda-graphs | 1275.4 (1273.1-1275.5, n=3) | 508 | 24.2 | 70.6 | 44.1, 444272 | 4-4 |
| native tp4 cuda-graphs | 1894.2 (1893.5-1900.3, n=3) | 340 | 16.6 | 70.6 | 56.9, 573513 | 7-7 |
| native tp1 eager | 409.3 (400.4-411.4, n=3) | 1071 | 78.8 | 70.5 | 17.6, 88967 | 46-80 |
| native tp2 eager | 287.4 (227.0-287.4, n=3) | 683 | 95.1 | 70.8 | 44.1, 444430 | 45-296 |
| native tp4 eager | 367.0 (286.4-391.6, n=3) | 505 | 80.0 | 70.8 | 57.0, 575167 | 6-6 |
| strict tp1 eager | 208.5 (206.2-208.6, n=3) | 1819 | 139.1 | 70.3 | 17.3, 86968 | 92-333 |
| strict tp1 eager +LoRA r16 | 62.4 (57.6-68.8, n=3) | 4948 | 429.1 | 71.2 | 17.3, 86968 | 58-256 |
| strict tp2 eager | 175.9 (163.1-177.2, n=3) | 1118 | 191.5 | 69.9 | 42.8, 431394 | 31-66 |
| strict tp2 eager +LoRA r16 | 55.1 (51.3-61.4, n=3) | 3209 | 540.1 | 70.1 | 42.8, 431394 | 23-115 |
| strict tp4 eager | 169.3 (155.1-178.6, n=3) | 999 | 226.6 | 69.5 | 55.0, 554156 | 5-7 |

Ratios of the medians: TP1 strict / native eager = 0.51x; TP1 strict / native cuda-graphs = 0.29x; TP1 strict +LoRA / strict = 0.30x; TP2 strict / native eager = 0.61x; TP2 strict / native cuda-graphs = 0.14x; TP2 strict +LoRA / strict = 0.31x; TP4 strict / native eager = 0.46x; TP4 strict / native cuda-graphs = 0.09x.

Each row: one engine, warm (2-token warm-up), batch 32 x 256 generated tokens at T=1 with ignore_eos, repeated 3x; tokens/s = generated / wall, median of the repeats (min-max in brackets). TTFT = wall time of generate(max_tokens=1) for the whole batch (min of 3). Decode step = rank-0 step wall time with a CUDA synchronize per step (measurement mode). Other agents' jobs loaded the host CPU during several of these runs (1-minute load average before each repeat in the last column; the node has 240 hardware threads, and the strict TP1 row stayed within 1.2 % between loads of 333 and 92); eager decode is CPU-launch bound, so rows measured under different loads carry an uncontrolled bias, and the min-max only measures run-to-run noise inside one engine. Native eager and native CUDA-graph rows are both given; production vLLM runs CUDA graphs, against which strict is 0.29x (TP1) to 0.09x (TP4): strict decode does not scale with TP here because every layer adds rank-ordered all-gathers (row-parallel partials, vocab-tile stats and logits) to an eager, launch-bound step.

## Profiles

| config | step (tokens / requests) | kernel time (ms) | top kernels (self CUDA ms) |
|---|---|---|---|
| native tp1 prefill | 2279 / 18 | 584.6 | `ampere_bf16_s16816gemm_bf16_256x128_ldg8_f2f_sta` 450.3, `void at::native::elementwise_kernel<128, 2, at::` 36.8, `void at::native::unrolled_elementwise_kernel<at:` 22.1 |
| native tp1 decode | 32 / 32 | 52.5 | `ampere_bf16_s16816gemm_bf16_256x64_ldg8_f2f_stag` 26.8, `void at::native::reduce_kernel<512, 1, at::nativ` 4.6, `ampere_bf16_s16816gemm_bf16_128x64_ldg8_f2f_stag` 4.4 |
| native tp2 prefill | 3853 / 30 | 605.0 | `ampere_bf16_s16816gemm_bf16_256x128_ldg8_f2f_sta` 356.6, `void at::native::elementwise_kernel<128, 2, at::` 58.5, `ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDev` 38.7 |
| native tp2 decode | 32 / 32 | 35.5 | `ampere_bf16_s16816gemm_bf16_128x64_ldg8_f2f_stag` 11.4, `void at::native::reduce_kernel<512, 1, at::nativ` 4.5, `ampere_bf16_s16816gemm_bf16_64x64_sliced1x2_ldg8` 4.2 |
| strict tp1 prefill | 3731 / 29 | 1608.3 | `det_gemm_tree_kernel` 1521.4, `_gated_attention_fwd_kernel` 16.4, `void at::native::unrolled_elementwise_kernel<at:` 13.9 |
| strict tp1 decode | 32 / 32 | 48.0 | `det_gemm_tree_kernel` 40.2, `_swiglu_fwd_kernel` 1.1, `_gated_attention_fwd_kernel` 0.7 |
| strict tp2 prefill | 3853 / 30 | 1024.1 | `det_gemm_tree_kernel` 757.8, `ncclDevKernel_AllGather_RING_LL(ncclDevKernelArg` 87.2, `nccl:_all_gather_base` 87.2 |
| strict tp2 decode | 32 / 32 | 44.0 | `det_gemm_tree_kernel` 25.1, `ncclDevKernel_AllGather_RING_LL(ncclDevKernelArg` 5.7, `nccl:_all_gather_base` 5.7 |
| strict tp4 prefill | 3443 / 27 | 762.8 | `det_gemm_tree_kernel` 319.7, `ncclDevKernel_AllGather_RING_LL(ncclDevKernelArg` 167.1, `nccl:_all_gather_base` 167.1 |
| strict tp4 decode | 32 / 32 | 157.9 | `ncclDevKernel_AllGather_RING_LL(ncclDevKernelArg` 65.9, `nccl:_all_gather_base` 65.9, `det_gemm_tree_kernel` 18.5 |

One step under torch.profiler on rank 0 (kernel rows only; the profiled step's wall time is inflated by the profiler and is not reported). Compare the decode kernel time with the unprofiled decode step median above: the gap is host-side (eager Python launches).

At TP>1 the rank-0 all-gather kernel time also absorbs waiting for the slower ranks (the collective blocks until every rank arrives), so those rows are not a clean breakdown. The strict decode step spends ~48 ms in kernels (TP1) inside a ~139 ms step: the rest is eager Python launch overhead (four separate q/k/v/gate GEMMs, 8 lm_head tiles, per-layer trees).

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

## Contract deviations
* The sampler kernels are vLLM's (temperature division, Gumbel noise from the per-request seed and
  position, per-row argmax over 1024-column blocks): row-local and batch invariant, but not a
  contract op. `selected_token_ranks` is vLLM's comparison count over the strict logits.
* `DeterministicCollective` needs the absent `_C` extension -> `torch.distributed.all_gather_into_tensor`
  (bit-exact data movement) + local canonical tree.
* `enforce_eager=True` is required (strict ops are plain Python/Triton calls).

## Tests (final code)
```
~/bin/gpu-lease -n 1 --pool 4-6 -- env PYTHONPATH=$WT ~/envs/vllm/bin/python -m pytest tests/models/muse_glimmer/test_vllm_strict_binding.py -q -p no:cacheprovider
  -> 65 passed, 2 skipped (the 2 opt-in 30B engine tests) in 30.23s at commit d62650b (final code)
RL_KERNEL_MUSEGLIMMER_E2E=1 ~/bin/gpu-lease -n 1 --pool 4-6 -- env PYTHONPATH=$WT ~/envs/vllm/bin/python -m pytest tests/models/muse_glimmer/test_vllm_strict_binding.py -q -p no:cacheprovider -k engine
  -> 13 passed, 54 deselected in 281.88s at commit 122742e (the -k filter selects the 2 opt-in 30B engine tests plus 11 unit tests whose names contain 'engine'): test_strict_engine_protocol_with_real_lora (RolloutEngineLike signatures, transfer_lora via file, worker-recomputed lora_sha256 per worker, wrong-hash rejection, n=2 sampling, decode == prefill bit for bit under a rank-16 LoRA on 416 slots with logp <= 0, second sync replaces the adapter) and test_strict_engine_decode_equals_prefill (short prompt + the 2261-token long-window prompt, 24 decode tokens at T=0 and T=1, bits equal, logp <= 0)
new test file on the pre-fix tree (git archive 021ea61^ rl_engine): PYTHONPATH=<old tree> ~/envs/vllm/bin/python -m pytest <new test file> -k 'TestFailClosedEngineFeatures or TestFailClosedSampling or TestLoRAHashRecomputed or TestNativeLoRARecomputed or TestWorkerPaths or TestCollectLabels or TestPerRankHeads'
  -> 34 failed, 1 passed (the pre-existing tampered-hash rejection) - the new tests fail before the fixes
same, GPU tests on the pre-fix tree: -k 'TestStrictSamplingLogits or TestLoRARuleCheckIsIndependent'
  -> 3 failed (no strict_logits; the old LoRA-rule formula reuses _padded_partial, so a broken helper still reports binding_equals_formula=True) - fail before, pass after
~/bin/gpu-lease -n 1 --pool 4-6 -- env PYTHONPATH=$WT ~/envs/vllm/bin/python -m rl_engine.models.muse_glimmer.runtime.vllm_tp_sim
  -> all_equal: true at commit 122742e (TP2/4/8 x LoRA on/off x local/global layer: attention out, MLP out, hidden, logp, sampler-path logp, strict logits and argmax equal TP1 bit for bit on every rank)
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
  `StrictRolloutEngine.__init__` gained optional `enforce_eager` (native only),
  `max_num_batched_tokens` and `num_gpu_blocks_override`; the `RolloutEngineLike` methods are
  unchanged. Native `load_lora` now force-loads the adapter on every worker (one extra RPC).
* `rl_engine.models.muse_glimmer.e2e.sync.rollout_lora_sha256` may drop its
  `workers[*].lora_sha256` fallback: the top-level value is now always the workers' recomputed
  hash.

## Open issues
* Performance is untuned (eager, launch-bound decode); next steps: CUDA graphs over the strict
  ops, one det GEMM for the four attention projections with per-slice canonical trees, a fused
  LoRA rank loop.
* TP8 engine not run (needs 8 GPUs); the simulation covers the binding arithmetic at TP8.
* Preemption with recompute was exercised only in the 1600-block KV-pool runs at TP1 (13
  preemptions, decode == prefill and tokens == the normal pool); not at TP>1.
* The LoRA rule check on real weights (`worker_check_lora_rule`) runs at TP1 only; at TP2/TP4 the
  LoRA path is covered by decode == prefill, xrt TP == TP1 and the simulation.
