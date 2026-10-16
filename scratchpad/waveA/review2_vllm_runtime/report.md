# REPORT — `vllm_runtime_adapter` (RFC #501, Muse-Glimmer-30B strict POC)

**STATUS: complete.** TP4 on the real engine is NOT RUN (needs 4 GPUs; the binding arithmetic at
TP2/4/8 is shown equal to TP1 by a single-GPU simulation). Every number below comes from a run on
the real 30B checkpoint (revision a4e59da5), A100-SXM4-80GB, vLLM 0.31.0, torch 2.13.0+cu130,
Triton 3.7.1. Machine-readable results with raw per-token / per-step series:
`reports/muse_glimmer_poc/data/vllm_runtime.json`; logprob dumps (`muse_glimmer_xrt_logprobs_v1`):
`reports/muse_glimmer_poc/data/xrt_vllm_{strict,native}[_lora]_tp{1,2}.json`.

## Headline results

| check (30B, real weights) | native vLLM | strict binding |
|---|---|---|
| decode logprobs == one-shot prefill re-score (8 GSM8K prompts x 64 tokens), T=0 | 161 / 512 differ (TP1), 152 / 512 (TP2) | **0 / 512** (TP1 and TP2) |
| same at T=1 sampling | 157 / 512 (TP1), 148 / 512 (TP2) | **0 / 512** (TP1 and TP2) |
| xrt scoring (9 sequences, 933 tokens, one crossing the 2048 window) TP2 vs TP1 | **921 / 933 differ**, max 0.25 | **0 / 933, `sequences` byte-identical** |
| strict TP1 and TP2 vs the in-repo strict reference chain (train and prefill modes) | 933 / 933 differ | **0 / 933** |
| strict vs native | max 0.19 (TP1), 0.21 (TP2) | |
| rank-16 random LoRA on all 416 slots: decode == prefill (T=0, T=1), TP1 and TP2 | 169 / 512 differ (vLLM's own LoRA) | **0 / 512** each |
| LoRA xrt scores TP2 vs TP1 | — | **0 / 933** |
| zero-B LoRA changes any score (TP1, TP2) | — | **0 / 512** |
| unmerged LoRA rule on real weights (layers 0, 3, 51 x 8 slots) | — | binding == written-out formula == trainer's `megatron_lora` **24 / 24** |
| `provenance()["lora_sha256"] == e2e.sync.lora_sha256(state_dict)` | yes | yes, every TP rank re-hashes what it loaded |
| generated tokens and their logprob bits, TP2 vs TP1 (T=0, T=1, with/without LoRA) | — | identical |

Throughput (isolated runs on GPUs 0-1; other agents loaded the host CPU, so treat as +-15 %),
batch 32 x 256 tokens, T=1, ignore_eos, eager mode:

| | tokens/s | TTFT (whole batch) | decode step median | peak alloc / rank | KV cache / rank |
|---|---|---|---|---|---|
| native TP1 | 406.6 | 1067 ms | 74.2 ms | 70.5 GiB | 17.6 GiB (88,966 pages) |
| strict TP1 | 186.4 (0.46x) | 1782 ms | 137.9 ms | 70.3 GiB | 17.2 GiB (86,919 pages) |
| native TP2 | 358.2 | 698 ms | 89.1 ms | 70.8 GiB | 44.1 GiB (444,430 pages) |
| strict TP2 | 145.1 (0.41x) | 1181 ms | 182.6 ms | 69.9 GiB | 42.8 GiB (215,648 pages: KV replicated) |

Profiles (one step under torch.profiler, kernel time only): strict TP1 decode (32 rows) = 49.5 ms of
kernels, 40.1 ms of it `det_gemm_tree_kernel`, inside a ~138 ms step -> the decode step is CPU/launch
bound (eager Python: four separate q/k/v/gate GEMMs, 8-tile fp32 lm_head, per-layer tree combines);
native TP1 decode = 52.1 ms of kernels in a ~74 ms step. Strict TP1 prefill (3204 tokens): 1385 ms of
kernels, 1308 ms det_gemm (~95 %). At TP2 the strict path adds rank-ordered all-gathers (~98 ms of all-gather kernels in a
~836 ms prefill kernel total); in the profiled TP2 decode step the all-gather kernels also absorb the wait for the
non-profiled rank, so that row is not a clean breakdown.

## What was done (this session)
1. Unit tests (see Tests).
2. Fixed what blocked every engine run; each was found on the real engine and is now tested:
   * **KV layout**: vLLM 0.31 hands each layer a *logical* `(B, H, N, 2D)` view whose strides follow the
     resolved `KVCacheLayout` (default `LBNHC` = physical `[B, N, H, 2D]`), so it is never contiguous.
     `strict_kv_views` re-views the dense per-layer region as `(2, B, N, H, D)` (K pages, then V pages)
     and still fails closed on a non-dense layout.
   * **The op probe silently demoted every strict op to its fallback inside the vLLM worker**: model
     construction runs under a CUDA default device and a bf16 default dtype, so the probe's
     `torch.randn(..., generator=cpu_gen)` raised or produced bf16; the attention probe always failed
     (int64 `q_pos`). Probe fixed, and **strict mode now refuses to start unless all 14 ops are the strict
     sibling kernels** (`RL_KERNEL_MUSEGLIMMER_ALLOW_FALLBACK=1` overrides and is recorded).
   * **Final norm** (`MuseGlimmerModel.norm`, `y = x_hat * w`) was still vLLM's `torch.mean` RMSNorm; now
     the strict `weighted_rmsnorm`.
   * **LoRA rule** now equals the trainer's (`megatron_lora.py`): `z = bf16(acc32(A x))`,
     `y = bf16(acc32(W x) + sum_k B[:,k] z[k])` (sequential fp32 rank loop, one rounding). The previous
     binding computed `bf16(bf16(Wx) + bf16(B z))`, which can never match the trainer bit for bit.
   * **LoRA transport**: tensors nested in `LLM.collective_rpc` arguments arrive as Python lists through
     vLLM's msgpack channel; the driver writes the bf16 state dict to a node-local file, every rank loads
     it (`weights_only=True`) and re-hashes it before installing its shard.
   * `collective_rpc` with module-level callables needs `VLLM_ALLOW_INSECURE_SERIALIZATION=1` (set by the
     engine; local processes only).
3. 30B runs (a)-(e) at TP1 and TP2, strict and native; vLLM-native LoRA (PEFT folder); scoring of the
   xrt file by the reference chain (`vllm_xrt_reference.py`) — the vLLM strict dumps equal it bitwise.
4. `StrictRolloutEngine` implements `e2e.engines.RolloutEngineLike` exactly (method names, parameter
   names and kinds asserted by the E2E test): returns `e2e.engines.RolloutSample`; fp32 logprobs at
   temperature 1 from raw logits; `logprobs()` returns `len - completion_start` values;
   `load_lora` verifies `muse_glimmer_lora_sha256_v1` + the key schema and fails closed (state unchanged)
   on a wrong hash; `provenance()["lora_sha256"]` is `None` before the first load and reported only if all
   TP ranks hold that hash; reproducible per-(seed, call, prompt) sampling seeds. Native mode supports
   `load_lora` too (vLLM punica LoRA from a PEFT folder, `vllm_native_lora.py`), so all four E2E modes can
   run; strict-LoRA vs native-LoRA xrt scores differ by the usual strict-vs-native noise (mean |d| 0.015)
   while the LoRA moves scores by mean 0.061, so the native mapping applies the same adapter.
5. TP4/TP8: `vllm_tp_sim.py` runs the binding's per-rank arithmetic (real shapes, one decoder layer +
   padded lm_head, with and without LoRA) as 2/4/8 threads with a rank-ordered in-process all-gather on
   one GPU: q, k, attention out, MLP out, final hidden and logp all equal TP1 bitwise. Engine TP4 NOT RUN
   (command in `vllm_runtime.json` -> `tp4_engine`). TP sizes that do not divide 8 fail closed at init.

## Design notes
Hooks: FLASH_ATTN enum override -> strict backend (owns the KV write); `ParallelLMHead` padded to 202752
rows (rank r of t owns vocab tiles [8r/t, 8(r+1)/t)); `MuseGlimmerForCausalLM.__init__` (keeps the
`(*, vllm_config, prefix)` signature vLLM inspects) binds and checks (native class, config pins, eager,
bf16 KV, all ops strict); `compute_logits` stays native for SAMPLING and publishes the hidden states;
V2 `Sampler.__call__` and `compute_prompt_logprobs_with_chunking` overwrite the selected-token column
with strict O7; `embed_input_ids` (O1, exact TP gather); decoder layer (O2 incl. fused [F1]); attention
(four det GEMMs on canonical slices of fused `qkv_proj` + `output_gate_proj`, O4, gate side channel into
O5); MLP (canonical gate/up slices, O6, row-parallel tree); final norm. Row-parallel GEMMs: fp32
sub-tree partial per rank -> `all_gather_into_tensor` (data movement) -> canonical tree -> one cast.

TP>1 workaround (kept): the strict attention kernel pins 32 Q / 2 KV heads, so every rank's vLLM
attention layer is built with all 32/2 heads (KV replicated in each rank's cache), q/k/v/gate are
all-gathered exactly, the full attention runs redundantly, and each rank keeps its own heads for o_proj.
**Switch to per-rank heads** once the kernel takes `H_Q`/`H_KV`: delete `_patch_attention_factory`
(vLLM's native 32/t and max(1, 2/t) heads per rank) and, in `attention_forward`, drop the four
`gather_heads` calls and the final head slice, passing the rank's q heads and its KV head (local q head j
reads KV head `(r*32/t + j) // 16`, offset by the rank's first KV head). Bits stay equal (per-head
arithmetic is independent of other heads) while KV memory and attention work drop by t.

## Contract deviations
* Token selection is native (vLLM samples from its own bf16 soft-capped logits); only selected-token
  logprob values are strict. Sampled tokens were nevertheless identical TP1 vs TP2 in these runs.
* KV-replicated, redundant attention at TP>1 (above).
* `DeterministicCollective` needs the absent `_C` extension -> `torch.distributed.all_gather_into_tensor`
  (bit-exact data movement) + local canonical tree.
* `enforce_eager=True` is required.

## Tests (final commit)
```
~/bin/gpu-lease -n 1 --pool 0-1 -- env PYTHONPATH=$WT ~/envs/vllm/bin/python -m pytest tests/models/muse_glimmer/test_vllm_strict_binding.py -q
  -> 27 passed, 2 skipped (the 2 opt-in 30B engine tests)
RL_KERNEL_MUSEGLIMMER_E2E=1 ~/bin/gpu-lease -n 1 --pool 0-1 -- env PYTHONPATH=$WT ~/envs/vllm/bin/python -m pytest tests/models/muse_glimmer/test_vllm_strict_binding.py -q -k engine
  -> 2 passed (166 s): protocol test with a real rank-16 LoRA + decode == prefill
```
New this session: LoRA rule == `megatron_lora`; probe keeps all ops strict under a CUDA default device +
bf16 default dtype; TP2/4/8 simulation == TP1; 30B protocol test with a real LoRA.

## How to reproduce
```
WT=<worktree>; D=reports/muse_glimmer_poc/data; M=rl_engine.models.muse_glimmer.runtime
~/bin/gpu-lease -n 1 --pool 0-1 -- env PYTHONPATH=$WT ~/envs/vllm/bin/python -m $M.run_vllm_strict --tp 1 --strict \
  --prompt-file $D/xrt_tokens.json --temperatures 0,1 --max-tokens 64 --prefill-check --lora-zero-check \
  --lora-random-check --xrt-score $D/xrt_tokens.json --xrt-out $D/xrt_vllm_strict_tp1.json \
  --xrt-lora-out $D/xrt_vllm_strict_lora_tp1.json --bench --profile --out runs/strict_tp1.json
# --tp 2 with `gpu-lease -n 2 --contiguous` -> *_tp2; --native without the LoRA flags -> xrt_vllm_native_tp{1,2}.json
# native LoRA: --native --native-lora --lora-random-check --xrt-score $D/xrt_tokens.json --xrt-lora-out $D/xrt_vllm_native_lora_tp1.json
# isolated perf: --no-generate --bench --profile (one config at a time)
# reference chain: python -m $M.vllm_xrt_reference $D/xrt_tokens.json runs/xrt_reference.json train,prefill
# TP simulation:   python -m $M.vllm_tp_sim
# collect:         python -m $M.vllm_collect_results --runs runs --bench-runs bench --reference runs/xrt_reference_train.json \
#                    --reference runs/xrt_reference_prefill.json --extra extra.json --out $D/vllm_runtime.json
```

## RFC feedback
1. **State the LoRA arithmetic exactly** (it is a cross-runtime contract): `z = bf16(acc32(A x))`,
   `y = bf16(acc32(W x) + sum_{k<r} B[:,k] z[k])` with a sequential fp32 rank loop; A replicated / B
   row-sharded for column-parallel, A column-sharded (+ canonical K tree) / B replicated for row-parallel.
   "Add the LoRA delta to the layer output" (`bf16(bf16(Wx) + bf16(BAx))`) is a different function.
2. **The rollout binding must own the KV write and the physical KV layout.** vLLM 0.31 passes a permuted
   logical view (`LBNHC`) with K|V packed in the last dim. Require a layer-compact layout; forbid prefix
   caching, KV connectors and cascade attention in strict mode.
3. **Fail closed on any non-strict op**, and list every op implementation in provenance. vLLM builds the
   model under a non-default device and dtype; binding code that creates tensors must pass both.
4. **Name the final norm and the sampler / prompt-logprob paths as hook points** (separate modules in
   vLLM: `MuseGlimmerModel.norm`, the V2 `Sampler`, `compute_prompt_logprobs_with_chunking`).
5. **Weight-sync transport**: specify "file or shm path + canonical hash, every TP rank re-hashes";
   vLLM's RPC channel does not carry nested tensors.
6. **O5 needs runtime `H_Q`/`H_KV`**, or the RFC must bless KV replication at TP>1 (t x KV memory and
   attention work; strict TP2 decode is slower than strict TP1 here).
7. **Token selection**: say whether the sampled token must come from strict logits. The logprob of a
   given token is TP/batch/decode-invariant, but the choice uses native cuBLAS logits and could differ
   across TP at near-ties (it did not in these runs).
8. **O7 sign invariant**: the strict lm_head tile kernel can return logp = +1..2 ulp for a near-certain
   argmax (48 of 1024 generated tokens; synthetic probe 44/2048 rows; the max tile's `s_t` falls below 1,
   i.e. `exp(zhat - m)` of the max element is not exactly 1 — likely FP contraction or the approximate
   `tl.exp`). Deterministic and TP-invariant, but consumers that assume `logp <= 0` should clamp, and
   contract R2 should forbid FP contraction in strict kernels (`enable_fp_fusion=False`) or the kernel
   should reuse pass-1 `zhat`.
9. vLLM facts for the RFC text: Model Runner V2 is the CUDA default; `VLLM_ATTENTION_BACKEND` is gone
   (`attention_config={"backend": ...}` + `register_backend`); null/evicted pages are id 0; the loader
   inspects `__init__` signatures; vLLM fuses gate/up as well as q/k/v (the RFC says "Q/K/V only");
   requests reach the engine core asynchronously, so batch composition varies run to run (harmless for
   strict ops, a source of native drift).

## Open issues
* TP4/TP8 engine runs (need 4/8 GPUs): exact command in `vllm_runtime.json` -> `tp4_engine`.
* Eviction beyond `max_model_len`, preemption/recompute and the `prompt_logprob_token_ids` path are
  untested / unpatched.
* Performance untuned (eager, launch-bound decode; next step: CUDA graphs over the strict ops or fusing
  the four attention projections into one det GEMM with canonical per-slice trees).

## Time spent
~4.5 h (previous session) + ~3 h (this session).

-----NOTE
content of rl_engine/models/muse_glimmer/runtime/REPORT_vllm_runtime_adapter.md (the agent harness blocks subagents from writing REPORT files; the integrator writes it from here)
