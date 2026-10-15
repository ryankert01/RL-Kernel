# REPORT — megatron_runtime (Megatron Bridge strict binding + LoRA + weight_sync_identity, StrictTrainer)

**STATUS (wave B, code commit `926da65`): TP1 and TP2 complete — every check passes, and TP2 equals TP1
bitwise for logprobs, LoRA gradients and the post-step LoRA hash. TP4: PENDING (see "TP4" below).**
Machine-readable results: `reports/muse_glimmer_poc/data/megatron_runtime.json` (raw series, exact
commands, producing commit and sha256 of every xrt file). Hardware A100-SXM4-80GB; `~/envs/megatron`
(torch 2.13.0+cu129, Triton 3.7.1, TE 2.20.2, megatron-core 0.20.0, Megatron Bridge 0.7.0); checkpoint
`~/models/Muse-Glimmer-30B` @ a4e59da5. All logprob dumps of this report postdate the O7 fix `c88607f`.

## Summary (real 30B model, `run_megatron_strict`, commit 926da65)
| check | TP1 | TP2 |
|---|---|---|
| weight_sync_identity (627 HF text tensors, sha256 + `torch.equal`) | 626/627 (final norm, by design) | 626/627 |
| xrt strict scoring, 933 scored tokens (LoRA rank 16, B = 0) | written | written, **0/933 bits vs TP1** |
| repeat (second scoring) | 0/933 | 0/933 |
| lora_noop (LoRA off vs B = 0) | 0/933 | 0/933 |
| native Bridge forward vs strict | max abs 0.314, mean 0.0160 | max abs 0.282, mean 0.0154 |
| grad_repeat (832 LoRA tensors, 104,792,064 elements, B ~ N(0, 1e-3)) | 0 differing elements | 0 |
| LoRA grads vs TP1 (bitwise) | - | **0 / 104,792,064** (aggregate sha256 `1df6d60b…` at both) |
| packing (micro_batch_tokens 512 / 4096 / 8192 = 6 / 2 / 1 micro-batches) | grads + logprobs bitwise identical | identical |
| step (AdamW lr 1e-5): grad norm | 7.0176005363464355 | 7.0176005363464355 |
| post-step LoRA hash (`muse_glimmer_lora_sha256_v1`) | `69968212be47…` | **equal**; post-step logprobs 0/933 vs TP1 |
| recompute check inside the strict step | 0 mismatches | 0 |
| bench 512..8192 tokens (1 warm-up + 3 timed per phase, raw samples) | complete; native OOM at 4096 fwd+bwd and 8192 | complete |
| TrainerLike protocol test (opt-in 30B, 2 steps, `sync.export_lora_verified`) | (wave A) | strict PASS, native PASS |

Every strict op resolves to its sibling strict kernel (`provenance.ops.all_strict = true`); strict
mode cannot select a fallback. TP8 has no real-model run: it is gated
(`MUSE_MEGATRON_ALLOW_UNTESTED_TP=8` required) and covered by simulated-TP unit tests only.

### TP4
PENDING: the 4-GPU run (`--tp 4 --checks identity,xrt,repeat,lora_noop,grad_repeat,packing,step`) and
the 4-process NCCL test are queued on GPUs 0-3; results replace this paragraph when they land.

## Files (all under `rl_engine/models/muse_glimmer/runtime/` unless noted)
| file | what |
|---|---|
| `megatron_strict.py` | `install_museglimmer_megatron_strict(model)`: re-drives the 52-layer text forward of Bridge's native `MuseGlimmerModel` from its parameters through the strict ops (no class patches); fused `linear_qkv` / `linear_fc1` views, canonical vocab tiles, HF final-norm reload, vision-tower release (`drop_vision_modules`), provenance; fails closed on a non-native class, config drift, CP > 1, SP, an untested TP size |
| `megatron_attention.py` (new) | per-rank attention: `rank_head_geometry`, `GatherKVHead`, `RankGatedAttention` (sibling forward + backward entries with head counts, replica dK/dV fold), `attention_core` |
| `megatron_lora.py` | unmerged strict LoRA linears (`StrictColumnLinear`, `StrictRowLinear`), packing-invariant LoRA gradient rule (`LORA_GRAD_RULE`), `LayerLoRA`, export/import (hash = `e2e.sync.lora_sha256`) |
| `megatron_native_lora.py` | NATIVE trainer mode: Bridge TE forward + the same LoRA via forward hooks (TP1/2) |
| `megatron_weights_identity.py` | `weight_sync_identity`: the 627 HF text tensors rebuilt from Bridge params (un-fused, TP-gathered) vs the safetensors bytes |
| `strict_kernels.py` | fail-closed op resolution (`resolve_ops`), GEMM backend incl. the token-axis dW rule, `TreeAccumulator`, `TPComm` (rank-ordered all-gather only), LM-head tile ops |
| `strict_trainer.py` | `StrictTrainer` (TrainerLike; rank 0 in-process, TP workers spawned), recompute-on-step with the fail-closed recompute check, `set_micro_batch_tokens` |
| `run_megatron_strict.py` | CLI: identity, xrt, repeat, lora_noop, native, grad_repeat, packing, step, bench |
| `megatron_results.py` | assembles `megatron_runtime.json` (raw-bit comparisons, per-file sha256 + producing commit, `--cross-runtime` diff with O7 staleness flags) |
| `tests/models/muse_glimmer/test_megatron_strict_binding.py` | unit, simulated-TP and real-NCCL tests (below) |

## Design

### Binding (Bridge module -> strict op), per layer `L = decoder.layers[i]`
`embedding.word_embeddings` -> O1 `normalized_embedding` (full table; gathered at TP>1) ·
`L.input_layernorm` -> O2 centered rmsnorm (fused into the previous layer's post-FFN norm) ·
`L.self_attention.linear_qkv` -> FOUR column GEMMs on views of the fused weight (per KV group
`[q 16 heads | gate 16 heads | k | v]`, verified on real weights) · O4 `qknorm_scaled_rope` on the
rank's heads · O5 per-rank gated attention · `linear_proj` -> row-parallel partial + rank-ordered
all-gather + canonical tree · post-attention / pre-FFN norms -> O2 fused · `linear_fc1` -> two column
GEMMs on its halves · O6 `swiglu` · `linear_fc2` -> row-parallel tree · `decoder.final_norm` -> O2
weighted rmsnorm with the HF tensor (Bridge stores `w-1` in bf16; 1120/6656 elements do not round-trip)
· `output_layer` -> O7 on the 8 canonical vocab tiles (Bridge's unpadded shards re-tiled; rank owns
tiles `[8r/t, 8(r+1)/t)`), tile statistics merged in the canonical tree. At TP4/8 MCore splits the
fused `linear_qkv` rows evenly (mid-group); the binding re-shards them once at bind time (all-gather,
data movement) to canonical per-projection row chunks. LoRA on the eight projections:
`y = bf16(acc32(Wx) + sum_k B[:,k] * bf16(Ax)[k])` (fp32 loop over the rank, one rounding);
`B = 0` is a bitwise no-op (`lora_noop` 0/933).

### Per-rank attention heads (wave B; the head all-gather workaround is removed)
| TP | Q heads / rank | KV heads / rank | KV head of rank r | k/v column shard of the rank |
|---|---|---|---|---|
| 1 | 32 | 2 | - | 256 cols (both heads) |
| 2 | 16 | 1 | r | 128 cols = that head |
| 4 | 8 | 1 (replicated on 2 ranks) | (2r)//4 | 64 cols = half of the head |
| 8 | 4 | 1 (replicated on 4 ranks) | (2r)//8 | 32 cols |

q, gate and `y` never leave the rank: its Q heads `[32r/t, 32(r+1)/t)` are exactly its canonical
q/gate rows and its `o_proj` input columns. `GatherKVHead` (TP4/8 only) all-gathers the k/v column
shards (rank-ordered data movement) and keeps the 128 columns of the rank's KV head; its backward
returns only the rank's own columns, so each K/V projection row's gradient comes from exactly one rank
(no double counting). `RankGatedAttention` calls `strict_gated_attention_fwd_varlen(...,
num_q_heads=32/t, num_kv_heads=max(1, 2/t))` and, in backward, `gated_attention_bwd_preprocess`,
`strict_gated_attention_bwd_partials` (8 canonical query chunks folded per 4-head group) and
`kv_partials_subtree`; at TP4/8 the `t/2` replica nodes of a KV head are all-gathered in rank order
and folded with `ops.canonical_tree.combine_tree` (never a reducing collective) BEFORE the single bf16
rounding. At TP1/TP2 the node is final and equals the sibling fused backward bit for bit.

### Fail-closed strict mode (review R1)
`strict_kernels.resolve_ops(allow_fallback=False)` resolves every op (norms, embedding, RoPE rows,
qk-norm, the attention entries, swiglu, det-GEMM partial, det-GEMM token-axis dW, LM-head tile stats) to
its sibling strict kernel or raises `StrictOpsUnavailable`, naming each non-strict op and the import
error behind it. `MuseGlimmerStrictBinding` (hence `StrictTrainer(strict=True)`) calls it first, before
touching the model. Unknown `MUSE_STRICT_ATTN` (`auto|strict|fallback`) / `MUSE_STRICT_GEMM` values
raise `ValueError`; `fallback`, `cublas` and `triton_repo` are still refused in strict mode. The labelled
fallbacks are reachable only with `allow_fallback=True` (unit tests). TP collectives are
`torch.distributed.all_gather` into rank-ordered slots (bitwise data movement) when the repo's
deterministic collective extension is absent; there is no reducing collective in the strict path.

### LoRA gradient rule and packing (review R2, kernels_gemm:R1)
`megatron_lora.LORA_GRAD_RULE`: for each sequence `s` of a packed micro-batch, `g_s` = the det_gemm dW
rule over its `T_s` tokens (`T_pad = roundup(T_s, 256)`, 8 chunks of `T_pad/8` rows, masked tail, fp32
tree, via `fast_det_gemm(..., ragged_k=True)`, i.e. `ops.det_gemm.DW_CHUNK_RULE`); the step gradient is
the LEFT FOLD `((g_0 + g_1) + g_2) + ...` in fp32 over the request's sequences in sample order, which
the op accumulates directly into the fp32 master's `.grad`. The gradient bits therefore do not depend
on `micro_batch_tokens` (how sequences are packed) nor on the TP size; they do depend on the sample
order, which is part of the data. `micro_batch_tokens` (default 4096) is a memory knob only; it and the
packing rule (`micro_batches`: greedy, order-preserving, a longer sequence is its own pack) are
recorded in the provenance (`micro_batch_tokens`, `packing_rule`, `lora_grad_rule`). The wave A rule
(the whole micro-batch padded to a multiple of 512, 8 chunks) both disagreed with det_gemm's 256 rule and
made the gradients packing-dependent. Cost: one small GEMM pair per sequence per projection per layer;
the xrt-set backward (9 sequences, 4216 tokens) takes 17.7 s at TP1 (2 micro-batches) versus 16.4 s for
the wave A rule in one micro-batch.

### StrictTrainer step (review R3)
`token_logprobs` = strict forward without a graph (micro-batched) -> fp32 values exposed as views of one
leaf; `step(loss)`: `g = dloss/dlogp` on rank 0, broadcast, every rank recomputes each micro-batch WITH
the graph (per-layer activation checkpointing), compares the recomputed logprobs bitwise with the
returned ones and only then back-propagates `g`. In strict mode any mismatch raises
`StrictRecomputeMismatch` before that micro-batch's backward; all LoRA gradients of the step are
discarded, so no optimizer step can use them (workers keep serving: the compared values are the
TP-gathered logprobs, identical on every rank). AdamW (`foreach=False`) on the fp32 masters; the grad
norm is computed from TP-gathered gradients in canonical key order. Constructor and TrainerLike API
unchanged (new optional method `set_micro_batch_tokens`).

### Memory (reviews R4, R7, R10)
- Vision tower: Bridge's `hook_hf_module_setattr_for_tp_grad_sync` leaves a bound `__setattr__` in every
  vision submodule (a reference cycle), so `delattr` alone freed nothing. `drop_vision_modules` releases
  the parameter storages, breaks the cycles and collects: measured `memory_allocated` drop 3.580 GiB =
  the 3.580 GiB of vision parameters (TP1 and TP2 runs).
- `small_k_fp32` accumulates in place (two `(M, N)` fp32 buffers instead of three); the column-parallel
  and LM-head backward build the canonical tree while the leaves arrive (`TreeAccumulator`: same bits,
  at most log2(n)+1 live partials).
- RoPE rows are gathered from the canonical cached table (`ops.qknorm_rope.rope_rows`, fixed 128 MiB per
  device; sha256 `088dbb8c…` logged, equal to the pinned `CANONICAL_ROPE_TABLE_SHA256`) instead of a
  per-packing cache that grew without bound.
- Result: an 8192-token sequence now trains at TP1 (strict step 40.8 s, peak 68.5 GiB); in wave A it ran
  out of memory.

## Contract deviations / declarations (not in contract v1)
1. **LoRA** (RFC open question 3): the unmerged form above and the LoRA weight-gradient rule
   (`LORA_GRAD_RULE`). The contract should state both. "Independent of batch size and sequence chunking"
   holds for the packing; a sum over samples is necessarily ordered by sample.
2. **Column-parallel input gradient**: tree over the 8 canonical output-row chunks of the full `N`
   (single-chunk leaves, rank sub-trees, rank-ordered all-gather, same tree); equal bit for bit to
   `ops.det_gemm.fast_det_gemm` autograd dA at TP1 and simulated TP2/4/8
   (`test_binding_dx_equals_ops_fast_det_gemm_autograd`, M in {1, 64, 300}, k/q/o_proj shapes).
3. **Final norm**: the HF tensor is reloaded (Bridge's `w-1` bf16 storage is lossy at 1120/6656
   elements); `weight_sync_identity` therefore reports 626/627 by design.
4. **TP4/8 weights and KV heads**: the fused `linear_qkv` is re-sharded to canonical rows at bind time;
   the KV head is all-gathered among its replicas per layer (data movement) and its gradient folded in
   the canonical tree before rounding (the attention row's replica rule).
5. **Embedding / LM head at TP>1**: the full embedding table is gathered on every rank; Bridge's
   unpadded vocab shards are re-tiled to the 8 canonical tiles (padded vocab 202752).
6. **Training step = two forwards** (no-grad scoring + recompute with the graph, bitwise-checked), with
   per-layer activation checkpointing.

## Test commands and results (final code)
Unit / simulated-TP tests, one GPU, megatron env (verbatim tail):
```
~/bin/gpu-lease -n 1 --pool 0-2 -- env PYTHONPATH=$WT ~/envs/megatron/bin/python -m pytest \
    tests/models/muse_glimmer/test_megatron_strict_binding.py -q -rs
SKIPPED [1] tests/models/muse_glimmer/test_megatron_strict_binding.py:724: needs 2 GPUs
SKIPPED [1] tests/models/muse_glimmer/test_megatron_strict_binding.py:724: needs 4 GPUs
SKIPPED [1] tests/models/muse_glimmer/test_megatron_strict_binding.py:1007: set RL_KERNEL_MUSEGLIMMER_E2E=1
45 passed, 3 skipped, 1 warning in 13.62s
```
The same file plus the vLLM binding's LoRA cross-check (`test_vllm_strict_binding.py::TestLoRA`, which
calls `StrictColumnLinear` / `StrictRowLinear` through their unchanged positional signature), vllm env:
`47 passed, 3 skipped in 45.31s`.
Real NCCL (`test_nccl_tp_equals_tp1_lm_head_column_row_attention[2]` / `[4]`: LM head, column/row LoRA
linears and the per-rank attention block incl. the TP4 replica fold on real collectives): PENDING (run
with the 4-GPU lease).
30B TrainerLike protocol test at TP2 (`RL_KERNEL_MUSEGLIMMER_E2E=1 MUSEGLIMMER_TRAIN_TP=2`, 2 steps,
lr 1e-4, any-scalar loss): strict `1 passed in 248.96s` (step 1 loss 117.5417, grad norm 423.0179,
recompute 0/231 mismatches; step 2 loss -0.68622, grad norm 0.54787); native
(`MUSEGLIMMER_TRAIN_STRICT=0`) `1 passed in 286.94s` (step 1 loss 116.9697, grad norm 335.11).

New regression tests; each fails on the wave A tree f91e8b6 and passes now (same conditions run on both
trees by the scratch probe `waveB_meg/probe_before_after.py`):
| test | wave A tree | now |
|---|---|---|
| `test_resolve_ops_fails_closed_when_a_strict_module_is_missing` | no exception, `all_strict=False` | `StrictOpsUnavailable` |
| `test_strict_mode_env_values_are_validated` (`MUSE_STRICT_ATTN=off`) | no exception (torch fallback attention) | `ValueError` |
| `test_recompute_mismatch_fails_closed_in_strict_mode` | count 1 reported, gradients kept | `StrictRecomputeMismatch`, gradients dropped |
| `test_lora_grads_are_packing_invariant` (3 sequences, column linear) | 14,860 / 15,034 of 20,480 grad elements differ for 2 / 3 micro-batches | 0 |
| `test_lora_token_grad_uses_the_det_gemm_dw_rule` (T = 200 / 300 / 513) | 10,581 / 0 / 16,803 elements differ from the dW kernel | 0 |
| `test_drop_vision_modules_releases_memory_held_by_setattr_cycles` | no helper (`delattr` freed 0 GiB on the 30B model) | freed = param bytes |

Other new tests: per-rank head geometry; `GatherKVHead` gradient ownership (TP1/2/4/8); per-rank
attention at TP1 == the sibling autograd op bitwise and within the attention row's normalized bound
(max|err|/max|ref| <= 1e-2) of the FP64 golden `attention_golden_bwd.reference_backward`; attention block
(four column LoRA projections + per-rank attention + row LoRA `o_proj`) at simulated TP2/4/8 == TP1
bitwise for output, dX and every LoRA gradient (local and global layers); `TreeAccumulator` ==
`combine_tree`; binding dX == ops autograd dA; a pinned `muse_glimmer_lora_sha256_v1` digest of a fixed
state recomputed from the documented byte stream (replaces the tautological alias check).

## Parity results (raw bits; `megatron_runtime.json`)
- **TP2 vs TP1** (commit 926da65, same packing, and also across packings): strict logprobs 0/933; LoRA
  gradients 0 of 104,792,064 elements in 832 tensors (aggregate sha256 `1df6d60b…` at both); LoRA
  logprobs with random B 0/933; grad norm equal; post-step LoRA hash equal (`69968212…`); post-step
  logprobs 0/933.
- **TP4 vs TP1**: PENDING.
- **Repeat / packing**: repeat 0/933 and grad_repeat 0 elements at every TP run; packing budgets 512 /
  4096 / 8192 (6 / 2 / 1 micro-batches) give identical gradients (same aggregate sha256) and logprobs at
  TP1 and TP2.
- **Native vs strict** (Bridge TE forward, bf16 logits, no LoRA): TP1 max |d logp| 0.314, mean 0.0160;
  TP2 0.282 / 0.0154 (not a parity target).
- **Cross-runtime** (`megatron_results --cross-runtime` on the files in this worktree): the vLLM and
  reference strict dumps on disk predate the O7 fix (`postdates_o7_fix = false`), so their mismatch
  counts (194/933 against the reference triton dump) are not evidence either way. The wave A Megatron
  dump (17dafef, pre-O7) matched that reference dump 0/933 and differs from the new Megatron dump in the
  same 194 tokens, consistent with the O7 fix being the only change. The final 3-way diff needs the
  regenerated vLLM / reference dumps (integrator, `tools/xrt_diff.py`).

## Benchmark (median of 3 timed runs after 1 warm-up per phase and size; one run per TP at 926da65)
Single sequence of `tokens` tokens; strict step = no-grad scoring + recompute forward/backward + AdamW;
native fwd+bwd = Bridge forward + backward of activation gradients through all layers (full recompute,
frozen weights, no LoRA). Times in ms; peak = strict-step peak GiB (max over ranks). Raw samples per
phase are in the JSON (`runs.tp<k>.checks.bench.rows[].samples`).

| TP | tokens | strict fwd | native fwd | ratio | strict step | of which recompute fwd+bwd | native fwd+bwd | ratio | peak GiB | errors |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 512 | 809 | 153 | 5.3x | 4402 | 3442 | 454 | 9.7x | 54.7 | - |
| 1 | 1024 | 1333 | 305 | 4.4x | 6289 | 4855 | 937 | 6.7x | 55.6 | - |
| 1 | 2048 | 2535 | 683 | 3.7x | 10922 | 8277 | 2130 | 5.1x | 57.2 | - |
| 1 | 4096 | 4940 | 1707 | 2.9x | 20774 | 15721 | OOM | - | 60.8 | native fwd+bwd OOM |
| 1 | 8192 | 9747 | OOM | - | 40770 | 30855 | OOM | - | 68.5 | native fwd and fwd+bwd OOM |
| 2 | 512 | 801 | 156 | 5.1x | 3431 | 2606 | 459 | 7.5x | 33.1 | - |
| 2 | 1024 | 829 | 182 | 4.6x | 4179 | 3158 | 543 | 7.7x | 33.9 | - |
| 2 | 2048 | 1499 | 384 | 3.9x | 7008 | 5327 | 1189 | 5.9x | 35.4 | - |
| 2 | 4096 | 2871 | 930 | 3.1x | 13074 | 10017 | 2972 | 4.4x | 38.6 | - |
| 2 | 8192 | 5651 | 2585 | 2.2x | 25888 | 20044 | 8410 | 3.1x | 45.3 | - |

The native OOMs are Bridge's own forward with a dense `(T, T)` attention mask at TP1 (8 GiB / 2 GiB
allocations), not the strict path.

## Review findings (wave A2) — verdicts
| id | verdict | evidence |
|---|---|---|
| megatron_runtime:R1 (blocking) | fixed | fail-closed `resolve_ops`; tests above; probe before/after |
| megatron_runtime:R2 | fixed (packing-invariant accumulation implemented) | per-sequence dW + left fold; 30B packing check 0 differing bits at TP1 and TP2; unit test |
| megatron_runtime:R3 | fixed | `StrictRecomputeMismatch` before backward, gradients dropped; unit test |
| megatron_runtime:R4 | fixed | 8192-token strict step at TP1: 40.8 s, 68.5 GiB peak; 8192 rows at TP1 and TP2 |
| megatron_runtime:R5 | fixed | this report rewritten from the JSON |
| megatron_runtime:R6 | fixed | JSON regenerated: per-check status, per-file sha256 + producing commit, every row of a TP from one run |
| megatron_runtime:R7 | fixed | 3.580 GiB freed = vision params; unit test with the setattr cycle |
| megatron_runtime:R8 | PENDING (TP4 run queued) | TP8 gated behind `MUSE_MEGATRON_ALLOW_UNTESTED_TP=8` |
| megatron_runtime:R9 | fixed | warm-up per phase and size, 3 repeats, samples + median, one run per TP |
| megatron_runtime:R10 | fixed | canonical `rope_rows`, no cache |
| megatron_runtime:R11 | fixed | pinned-hash test, FP64 golden comparison, tests for R1/R2/R3 |
| megatron_runtime:R12 | fixed | per-file sha256/commit, in-repo `--cross-runtime`, hex note in the JSON; stale dumps flagged |
| kernels_gemm:R1 | fixed | token-axis rule = det_gemm 256 rule (`fast_det_gemm(ragged_k=True)`); test at T = 1/200/300/513 |
| kernels_gemm:R2 | not fixed here (other owners' files); regression test added | `test_binding_dx_equals_ops_fast_det_gemm_autograd` (0 mismatches) supports the finding; report wording -> integrator |

## Integration requests
1. `ops/REPORT_det_gemm.md` (kernels_gemm:R2): state that `column_parallel_fast_det_gemm` /
   `row_parallel_fast_det_gemm` / `tree_reduce_over_ranks` are reference implementations not wired into
   any runtime; the Megatron binding uses its own `StrictColumnLinear` / `StrictRowLinear`, whose dX equals
   ops `fast_det_gemm` autograd dA bitwise (TP1 and simulated TP2/4/8, M in {1, 64, 300}); the parity
   gain was det_gemm:1. Add a list-returning fake gather to `test_det_gemm.py`'s TP tests.
2. `ops/REPORT_swiglu_ffn.md:258` ("both runtimes agree"): now true; the Megatron LoRA token-axis
   reduction uses the 256 rule.
3. Contract: add the LoRA form and `LORA_GRAD_RULE` (declaration 1); the per-rank O5 geometry and the
   KV-replica fold are in `kernels_attention.json` `o5_replacement_text`.
4. E2E: `StrictTrainer.step` can now raise `StrictRecomputeMismatch` in strict mode; the bridge should
   surface it as a hard failure of the step (it propagates through `trainer.step`).
5. Final cross-runtime byte-diff once the vLLM and reference dumps are regenerated after `c88607f`.

## Open issues
- TP8: no real-model run (needs 8 GPUs); gated behind `MUSE_MEGATRON_ALLOW_UNTESTED_TP=8`.
- Strict step cost: 3.1x (TP2, 8192 tokens) to 9.7x (TP1, 512 tokens) the native forward+backward; the
  scoring forward is repeated inside the step (two forwards per step), and the strict forward is
  latency-bound at short lengths (about 0.8 s per forward up to 1024 tokens at TP2).
- The native reference runs out of memory at TP1 for 4096 (fwd+bwd) and 8192 tokens (Bridge's dense mask).
- The final-norm identity exception is structural (Bridge's lossy `w-1` storage).

## RFC feedback
- `megatron_vime_provider`: "re-drive the forward from Bridge parameters" (not "patch modules"); attention
  per rank with head-count parameters; KV heads replicated when TP > num_kv_heads, with a canonical replica
  fold of dK/dV before rounding; Bridge's unpadded vocab shards re-tiled canonically.
- Training determinism needs a declared rule for every reduction over TOKENS (LoRA dA/dB): per-sequence
  canonical partials folded in sample order make the step independent of micro-batching and TP.
- `weight_sync_identity`: 626/627 plus the final-norm exception; hash the HF-form tensors the trainer
  computes with (`muse_glimmer_lora_sha256_v1`).
- Strict mode must fail closed on missing kernels, on unknown mode values and on a recompute mismatch.

## Resume
```
WT=$(git rev-parse --show-toplevel); P=~/envs/megatron/bin/python; S=<scratch dir>
~/bin/gpu-lease -n 1 --pool 0-2 -- env PYTHONPATH=$WT $P -m rl_engine.models.muse_glimmer.runtime.run_megatron_strict \
    --tp 1 --checks identity,xrt,repeat,lora_noop,native,grad_repeat,packing,step,bench \
    --results $S/run_tp1.json --grads-out $S/grads_tp1.pt
~/bin/gpu-lease -n 2 --pool 0-2 --contiguous -- ... --tp 2 (same checks) --results $S/run_tp2.json --grads-out $S/grads_tp2.pt
~/bin/gpu-lease -n 4 --pool 0-3 --contiguous -- ... --tp 4 --checks identity,xrt,repeat,lora_noop,grad_repeat,packing,step \
    --results $S/run_tp4.json --grads-out $S/grads_tp4.pt
$P -m rl_engine.models.muse_glimmer.runtime.megatron_results --run 1=$S/run_tp1.json --run 2=$S/run_tp2.json \
    --run 4=$S/run_tp4.json --grads 1=$S/grads_tp1.pt --grads 2=$S/grads_tp2.pt --grads 4=$S/grads_tp4.pt --cross-runtime
```
