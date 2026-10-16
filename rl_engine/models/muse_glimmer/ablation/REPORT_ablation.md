# REPORT: museglimmer_ablation, where train/rollout drift comes from in a native bf16 stack

STATUS: complete. Measurements were taken on 2026-10-15 between 22:26 and 23:14 UTC. Five parts ran at commit `d74733b` and the HF second-GPU baseline ran at `b71de7a`, both from clean trees; between them only `merge_parts` and one constant in `run_ablation.py` changed. The merge ran at `b71de7a`.

Setup: A100-SXM4-80GB (sm80), GPUs 3 and 7, one GPU per process; torch 2.13.0+cu130, Triton 3.7.1, transformers 5.17.0, in the vLLM 0.31.0 env.

Data: `reports/muse_glimmer_poc/data/ablation.json`, schema `muse_glimmer_ablation_v2`. It holds 405 records, one per (engine, setting, sequence), with exact per-token values (`logp_hex`, `dlogp`, `kl_full`, `var_argmax`, `top1_tie`). It also holds 45 setting summaries, `settings_table`, the `analysis` views, `layer_records`, `previous_runs` and `commands[]`.

**Scope.** This row ran two things on one A100 at a time: HF transformers 5.17.0 (`MuseGlimmerTextModel`, eager and SDPA, bf16) and the strict reference. It did NOT run vLLM 0.31 or Megatron. Anything below about those engines is a hypothesis for the xrt byte-diff rows to test.

This report replaces the wave-A text and applies the adversarial review (R1-R12; verdicts at the end).

## Summary

1. **Native HF bf16, one factor at a time: 13 of 16 one-factor comparisons change logprob bits.**
   - Factors: batch padding (right or left), KV-cache decode, the one-shot L-1 rollout prefill, SDPA instead of eager, the cuBLAS `allow_bf16_reduced_precision_reduction` flag, and building RoPE `inv_freq` on the GPU.
   - Each changes 918-927 of the 933 gold-token logprobs.
   - Size: mean |dlogp| 0.0167-0.0204, max 0.447-1.119, mean full-vocabulary KL(base\|\|variant) 5.6e-4 to 7.3e-4 nats per token.
2. **Chunking only matters for sequences longer than the chunk.** Compared against the one-shot L-1 prefill (one factor):
   - chunk 64: 925/933;
   - chunk 256: 571/933 (the 5 sequences shorter than 256 tokens are bitwise identical);
   - chunk 1024: 176/933 (only the 2439-token sequence).
   The wave-A table compared chunking against the L-token baseline, which changes two factors at once.
3. **RoPE `inv_freq` built by torch on the GPU** differs from HF's CPU-built table in 4 of 64 fp32 entries. That changes 924/933 tokens (max 0.820, mean 0.0167). The wave-A null result for this setting was a harness bug: the setting copied the CPU table to the GPU.
4. **TF32 on, the transformers-5.19 elementwise RoPE build, and TF32 under that build: 0/933 each.** The only fp32 GEMM in this bf16 model (the RoPE product, K = 1) is bitwise the same with TF32 on or off (`tf32_probe`).
5. **Logits dtype.** Soft cap in fp32 over the bf16 GEMM output: 922/933, mean 0.0112, KL 1.9e-4. Also keeping the lm_head GEMM output in fp32, against the soft-cap-only setting: 909/933, mean 0.0031.
6. **Argmax.**
   - The bf16 soft-capped logits tie exactly at the top at 16 of 933 baseline positions, and `argmax` then returns the lowest tied id.
   - The 8 measured one-factor settings that move give 45 raw flips. 39 of them sit at a top-1 tie in one of the two runs.
   - Flips where both maxima are unique: 0-2 per setting, at most 0.2% of tokens.
7. **Where it starts** (`layer_records`, gsm8k_test_0 and the long sequence). The embedding is always exact. The first differing point depends on the factor:
   - padding, KV decode, reduction flag: the first GEMM of layer 0 (`q_proj`; `k_proj` on the long row, whose `q_proj` does not change);
   - L-1 prefill and SDPA: the attention core output (the input of `o_proj`), with q/k/v/gate projections bitwise equal;
   - GPU `inv_freq`: the RoPE `sin` table (also `cos` past position 500);
   - KV decode on the long sequence: the eager RMSNorm (`input_layernorm`) on 4 decode rows.
   Depth then spreads a few changed elements to 80-96% of the final-norm elements. For example, the L-1 prefill on gsm8k_test_0 starts with 3 changed layer-0 output rows (frac 8.4e-4) and ends at 0.797 at the final norm.
8. **Strict reference** (triton backend, kernels after c88607f): 0 of 27057 token scorings differ across 15 settings:
   - packed, reversed, right-padded and left-padded varlen batches;
   - paged batch prefill, prefill alone, decode alone;
   - continuous batching of 9 sequences, and of 72 slots (8 staggered copies of each sequence, up to 72 decode rows per step) through the public API;
   - the same 72 slots as vLLM-style mixed prefill+decode steps;
   - chunked prefill 64/256/1024;
   - a repeat and a second GPU.
9. **Null controls: all 0/933 against the HF baseline.** Immediate repeat; repeat after all factors; a new process on the same GPU; a second GPU (3 vs 7); the wave-A run at 84f30f8 two hours earlier; round 1 at 467d4f6. Strict repeat and second GPU (7 vs 3): 0/933. The wave-A strict baseline differs from the current one in 194/933 tokens, as expected, because it predates the O7 fix c88607f.
10. **Drift grows with uncertainty.** Mean |dlogp| for padded-right by baseline gold-token probability: 0.0011 for p > 0.9 (535 tokens), 0.0133, 0.0387, and 0.0836 for p <= 0.1 (123 tokens). Of the 123 low-p tokens, 9 are the first completion token (logp -30.3 to -34.1), 13 overlap a GSM8K `<<...>>` annotation, and the rest are wording choices (' sells', ' duck', ' came'). 205 of the 535 tokens with p > 0.9 are inside annotations.

## What was measured

- **Workload.** `xrt_tokens.json` (muse_glimmer_xrt_v1): 8 GSM8K chat prompts with gold answers (39-156 completion tokens each) and one 2439-token sequence (completion 2261..2438, past the 2048 window). 933 scored tokens.
- **HF model.** transformers 5.17 `MuseGlimmerTextModel`, built-in classes (`trust_remote_code=False`). Only the text weights (51.9 GiB bf16) are filled from the shards. `inv_freq` comes from HF's `compute_default_rope_parameters` (pow on the CPU, fp32).
- **Baseline `hf_baseline`.** Eager attention, TF32 off, each sequence alone over all L tokens (trainer style), one shot, 5.17 matmul RoPE, HF head (bf16 GEMM, then `*m`, `/T`, `tanh`, `*T` in bf16), fp32 `log_softmax`.
- **Metrics per gold token**, with d = logp_var - logp_base:
  - `bits_differ` and |d|;
  - `k3 = exp(d)-d-1`, a mismatch size on gold tokens, not a KL;
  - `kl_full` = KL(base\|\|var) over the full vocabulary: an fp64 sum of the fp32 log_softmax rows, measured settings only, computed outside the timed region;
  - argmax ids of both runs and top-1 minus top-2 logit gaps of both runs. A flip counts as 'at a tie' when either gap is 0.
- **One factor at a time.** Each setting that flips more than one factor names a reference setting that differs from it in exactly one knob (`Setting.one_factor_ref`). The merge derives that comparison from the stored exact values: records named `<setting>__vs__<ref>`, `derived_from_stored_values: true`, no KL. `analysis.one_factor_settings` lists every one-factor comparison.
- **HF settings** (`settings_table.hf`): padded_batch9_right/left, kv_cache_decode, prefill_L_minus_1, chunked_prefill_64/256/1024 (reference: prefill_L_minus_1), sdpa, tf32_on, softcap_fp32, logits_fp32 (reference: softcap_fp32), rope_elementwise, rope_elementwise_tf32_on (reference: rope_elementwise), rope_inv_freq_cuda, bf16_reduced_reduction_off, rollout_like_batched_decode, rollout_like_batched_decode_sdpa (reference: rollout_like_batched_decode), plus the repeats. flash_attention_2 is SKIPPED because flash_attn is not installed.
- **Strict.** Baseline `strict_train_alone` = `MuseGlimmerStrictModel.logprobs(mode=train)` per sequence. Settings are in `settings_table.strict`.
  - Single-sequence modes call `logprobs(mode=...)`.
  - Continuous batching without 'mixed' uses only `model.prefill(Session(cache, slot))` and one `model.decode_batch` call per step. In the 72-slot runs, copy c of each sequence is admitted at step 4c, so the decode batch grows to 72 rows, shrinks, and rows change position.
  - Forms that put several multi-token sequences in one call have no public entry point and run through `ablation_strict.packed_pass`: packed varlen, paged batch prefill, mixed steps. Before any of them runs, `selfcheck_packed_pass` proves `packed_pass` byte-equal to the reference's public `forward_train`, `prefill` and `decode_batch` on the model being measured, and the run refuses the packed settings otherwise. On real weights all three checks were equal (`packed_pass_selfcheck`).

## Results (933 tokens; magnitudes from `settings[]`)

One factor (HF, against hf_baseline unless a reference is named):

| factor | setting | tokens differ | max abs dlogp | mean abs dlogp | mean k3 | mean KL | argmax flips (unique / at ties) |
|---|---|---|---|---|---|---|---|
| 1 batch | padded_batch9_right | 927 | 0.602 | 0.0190 | 1.43e-3 | 7.29e-4 | 6 (0 / 6) |
| 1 batch | padded_batch9_left | 926 | 0.633 | 0.0204 | 1.78e-3 | 6.49e-4 | 4 (0 / 4) |
| 2 decode | kv_cache_decode | 923 | 0.447 | 0.0175 | 1.16e-3 | 6.68e-4 | 7 (1 / 6) |
| 3 prefill length | prefill_L_minus_1 | 918 | 1.119 | 0.0186 | 1.57e-3 | 5.87e-4 | 5 (2 / 3) |
| 3 chunking vs L-1 | chunk 64 | 925 | 0.517 | 0.0175 | 1.23e-3 | - | 5 (2 / 3) |
| 3 chunking vs L-1 | chunk 256 | 571 | 0.602 | 0.0118 | 9.7e-4 | - | 2 (0 / 2) |
| 3 chunking vs L-1 | chunk 1024 | 176 | 0.241 | 0.0031 | 1.8e-4 | - | 1 (0 / 1) |
| 4 attention | sdpa | 924 | 0.711 | 0.0173 | 1.40e-3 | 5.86e-4 | 7 (1 / 6) |
| 4 attention, within rollout-like | sdpa vs rollout_like_batched_decode | 923 | 0.526 | 0.0177 | 1.19e-3 | - | 8 (0 / 8) |
| 5 tf32 | tf32_on; tf32 under elementwise RoPE | 0 | 0 | 0 | 0 | 0 | 0 |
| 6 logits | softcap_fp32 | 922 | 0.148 | 0.0112 | 2.74e-4 | 1.91e-4 | 4 (0 / 4) |
| 6 logits | logits_fp32 vs softcap_fp32 | 909 | 0.040 | 0.0031 | 2.2e-5 | - | 4 (0 / 4) |
| 7 rope | rope_elementwise | 0 | 0 | 0 | 0 | 0 | 0 |
| 7 rope | rope_inv_freq_cuda | 924 | 0.820 | 0.0167 | 1.21e-3 | 5.60e-4 | 6 (1 / 5) |
| 10 gemm reduction | bf16_reduced_reduction_off | 927 | 0.497 | 0.0193 | 1.42e-3 | 7.13e-4 | 6 (1 / 5) |

Multi-factor settings against hf_baseline:
- chunked_prefill 64 / 256 / 1024 (L-1 plus chunking): 926 / 924 / 922.
- logits_fp32: 922, mean 0.0118, KL 2.07e-4.
- rollout_like_batched_decode (left-padded prefill of the 9 prompts plus batched teacher-forced decode): 922, max 0.602, mean 0.0188, KL 8.37e-4; 25.3 s, peak 68.1 GiB.
- The same with SDPA: 923, mean 0.0177, KL 5.52e-4.

Strict: 0 tokens differ in every setting. Wall times:
- packed 1.84 s; right/left pad 9.4 / 9.3 s (57.0 GiB); paged batch prefill 1.98 s; prefill alone 2.71 s;
- decode alone 108.8 s; batched decode of 9: 28.4 s (178 steps);
- 72 slots: 99.4 s through the public API and 91.7 s as mixed steps (206 engine steps, 61.2-61.5 GiB, 0/7464 token scorings across all copies);
- chunks 64/256/1024: 8.7 / 3.6 / 2.8 s.

Cross context, strict_vs_hf_baseline: 933/933, max 0.679, mean 0.0192, k3 1.11e-3.

Per sequence: `run_ablation.py merge` prints the setting x sequence matrices. Chunk 256 vs L-1 is [0,0,0,0,91,148,0,155,177]. The schedule factors change 38-156 of each short sequence's 39-156 tokens and 172-178 of the long sequence's 178.

Pairwise identical tokens (`analysis.hf_pairwise_identical_tokens`):
- The baseline, tf32_on, the elementwise RoPE builds and all repeats are mutually identical (933).
- prefill_L_minus_1 shares 757 tokens with chunk 1024 and 362 with chunk 256; chunk 256 and chunk 1024 share 361.
- padded right, padded left and bf16_reduced_reduction_off share 183-184.

Layer 0 in detail (`layer_records[].detail_points`): frac of bf16 elements that differ / rows.
- gsm8k_test_0, padded right: q_proj 0.28, o_proj input 0.74, layer_0 output 0.649 / 190 rows, final norm 0.954.
- gsm8k_test_0, L-1 prefill: q/k/v/gate 0; o_proj input 9 rows from row 43; layer_0 output 3 rows (frac 8.4e-4); final norm 0.797 / 160 rows.
- gsm8k_test_0, GPU inv_freq: rope_sin 8 rows from position 43; layer_0 output 4 rows; final norm 0.250 / 51 rows.
- Long sequence, KV decode: input_layernorm 4 rows from row 2269; q/k/v differ only on the 177 decode rows; the attention output also differs on prompt rows from row 17, because the prompt prefill (M = 2261) and the training forward (M = 2439) give different eager attention shapes.

## RFC feedback (scope: HF transformers 5.17 bf16 on A100)

1. Any change to the shape or schedule of a bf16 GEMM or attention kernel changes about 99% of logprob bits, with one common magnitude. Only end-to-end batch- and split-invariant kernels remove this; fixing one knob does not.
2. The device and function that build the RoPE table are a real drift source (924/933 tokens). This supports O4: one fixed torch-CUDA table, the same object on both runtimes. HF's CPU-built table differs from a CUDA-built one.
3. TF32 is not a drift source for this bf16 model. Keep the TF32-off rule for fp32 GEMMs.
4. Chunk size only matters for sequences longer than the chunk. The L-1 vs L prefill is its own factor. Whether vLLM or Megatron show either one is NOT MEASURED here.
5. Report |dlogp|, full-vocabulary KL and a breakdown by probability bucket, because the bit count saturates. Make argmax comparisons tie-aware, or run them on fp32 logits.
6. The first-drift gate should expect layer 0 but not always its first GEMM: depending on the factor it is the attention core, the RoPE table or an eager RMSNorm. This supports contract R3.
7. This row measures distance to HF-bf16 only. For accuracy against HF-fp32 see REPORT_full_model_chain.
8. Strict data older than c88607f is stale (194/933 baseline tokens changed).

## Deviations

- flash_attention_2 is NOT RUN: flash_attn is not installed, and installs are forbidden.
- Strict padding has no key mask. Right padding appends pads inside a segment; left padding is a separate pad segment in front, with positions starting at 0.
- Strict records have no argmax and no KL, because the reference API returns selected-token logprobs only. Derived records have no KL.
- The attention core and the RoPE application are functions, not modules. They are observed through the `o_proj` input and the `rotary_emb` outputs. Sub-layer taps cover layer 0 only.
- First-sequence timings include warm-up. Analysis metrics are excluded from `wall_ms` and reported as `metrics_ms`.

## How to reproduce

The exact command lines are in `ablation.json` `commands[]`. Run them from the worktree root through `~/bin/gpu-lease -n 1`, one GPU per process, then run `run_ablation.py merge` with `--tokenizer`, `--previous` and `--commands-file`. Tests: `pytest tests/models/muse_glimmer/test_ablation.py` gives 26 passed (5 tests need CUDA).

## Figures for the blog (all from ablation.json)

- (a) Bars from `analysis.one_factor_settings`: n_bits_differ, mean_abs_dlogp, mean_kl_full; HF vs strict.
- (b) Heatmap of setting x sequence `n_bits_differ` from `records[]`, including chunked vs L-1.
- (c) Per-token dlogp strip for gsm8k_long_window: padded, decode, GPU inv_freq, softcap, and strict as a flat zero line.
- (d) `analysis.abs_dlogp_by_baseline_prob`.
- (e) Per-layer `layers[].frac_elems_differ`, 54 residual points, unchanged x-axis.
- (f) The layer-0 `detail_points` waterfall: which sub-op differs first.
- (g) Pairwise matrix from `analysis.hf_pairwise_identical_tokens`.
- (h) Raw flips vs unique flips (`argmax_flips_unique` / `argmax_flips_at_ties`).
- (i) Strict `decode_rows_per_step` for the 72-slot run.

## Review verdicts

- R1, R2, R3, R4, R6, R7, R8, R9, R10, R11, R12: fixed; evidence is in the code, tests and data.
- R5: this text is the fix; the subagent harness could not write it.
