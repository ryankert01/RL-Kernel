Please write rl_engine/models/muse_glimmer/ablation/REPORT_ablation.md with the content below; the harness blocked this subagent from writing report .md files.

# REPORT — museglimmer_ablation: where does train/rollout drift come from in the native stacks?

STATUS: complete. Measured at commit 84f30f8 (main HF and strict parts; later parts at 4105697 / 106ba05 / 97c6855 add only reporting, layer taps and the rollout-like driver). A100-SXM4-80GB sm80, torch 2.13.0+cu130, Triton 3.7.1, transformers 5.17.0, vLLM 0.31.0 env, GPUs 6 and 7. Data: reports/muse_glimmer_poc/data/ablation.json (schema muse_glimmer_ablation_v1, 324 per-sequence records with per-token arrays dlogp / k3 / bits_differ / argmax_flip / logp_hex, 36 setting summaries, layer_records, tf32_probe, analysis).

## Summary
1. Native HF bf16: every factor that changes the shape or schedule of a bf16 GEMM or attention kernel flips the fp32 logp bits of 98-99% of the 933 completion tokens. The size is the same for all of them: mean |dlogp| 0.017-0.020, max 0.45-1.12, mean k3 1.2-1.8e-3, 4-8 argmax flips. The factors are padded batch (right/left), KV-cache decode, chunked prefill 64/256/1024, rollout-style prefill of L-1 tokens, SDPA vs eager, and the cuBLAS bf16 reduced-precision reduction flag. A rollout-like combination (left-padded batched decode, with or without SDPA) is no larger.
2. Where it starts: the embedding is exact. 56-67% of layer-0 output elements already differ, rising to 93-96% at the final norm, and mean |delta| grows about 100x with depth. A seed of only a few elements (M = 190 vs 191) reaches 68% by layer 12.
3. Logits/soft-cap dtype is smaller and not amplified by depth: 922/933 tokens differ, mean 0.011-0.012, max 0.15-0.16, k3 about 3e-4.
4. Null results (0/933): TF32 on; RoPE 5.19 elementwise; elementwise RoPE with TF32 on; inv_freq computed on CUDA; immediate repeat; repeat after all factors; new process on the same GPU; second GPU (6 vs 7). The TF32 probe shows TF32 is really active on a square fp32 GEMM (1,048,489/1,048,576 elements change), but the model's only fp32 GEMM (RoPE inv_freq@positions, K=1) is unchanged and equals the elementwise build.
5. The strict reference (triton) stays at 0/933 under all 11 settings of factors 1-3, under repeat and on the second GPU. The settings include continuous-batching decode of all 9 and the 2439-token window-crossing sequence.
6. Strict vs HF baseline (context): 933/933 differ, mean 0.0192, max 0.68. That is the size of HF's own batching noise.
7. Drift lives on uncertain tokens. Mean |d| is about 0.001 for gold-token p > 0.9 (535 tokens) and about 0.08 for p <= 0.1 (123 tokens; GSM8K <<calc>> annotations).

## What was measured
Workload: xrt_tokens.json, 8 GSM8K chat prompts with gold answers plus one 2439-token sequence (completion 2261..2438, past the 2048 window), 933 scored tokens. Model: the built-in transformers 5.17 MuseGlimmerTextModel (trust_remote_code=False). Text weights only (51.9 GiB bf16) are built on the meta device and filled from the shards. inv_freq is recomputed by HF compute_default_rope_parameters on CPU in fp32 and kept fp32.

Baseline hf_baseline: eager, TF32 off, each sequence alone over all L tokens (trainer style), one-shot, 5.17 matmul RoPE, HF head (bf16 GEMM, then *m, /T, tanh, *T in bf16), fp32 log_softmax.

Metrics per token, with d = logp_var - logp_base: bits_differ, |d|, k3 = exp(d)-d-1, argmax_flip. Tokens are gold answers, so mean k3 is a mismatch size, not an unbiased KL.

Factors (factor_id):
1. Batch: padded_batch9_right / _left.
2. kv_cache_decode.
3. prefill_L_minus_1, chunked_prefill_64 / 256 / 1024.
4. sdpa (flash_attention_2 SKIPPED: flash_attn not installed).
5. tf32_on.
6. softcap_fp32, logits_fp32.
7. rope_elementwise, rope_elementwise_tf32_on, rope_inv_freq_cuda.
8. Second GPU.
9. Repeat (immediate, after all factors, new process).
10. bf16_reduced_reduction_off.
11. Rollout-like combination of 1+2 (+4).

Strict (backend triton, all 12 sibling ops), against its own baseline mode=train per sequence: packed_batch9, packed_batch9_reversed, right_pad_batch9 (21951 rows), left_pad_batch9, paged_prefill_batch9, prefill_alone, decode_alone, batched_decode9 (continuous batching), chunked_prefill_64 / 256 / 1024, repeat, second GPU.

## Results (933 tokens)

| engine | setting | tokens differ | max abs dlogp | mean abs dlogp | mean k3 | argmax flips | wall ms |
|---|---|---|---|---|---|---|---|
| hf | repeat_immediate / repeat_after_all / new_process_same_gpu_7 / second_gpu_6_vs_7 | 0 | 0 | 0 | 0 | 0 | 1812 / 1839 / 2664 / 3821 |
| hf | padded_batch9_right | 927 | 0.602 | 0.0190 | 1.43e-3 | 6 | 9011 |
| hf | padded_batch9_left | 926 | 0.633 | 0.0204 | 1.78e-3 | 4 | 8532 |
| hf | kv_cache_decode | 923 | 0.447 | 0.0175 | 1.16e-3 | 7 | 79105 |
| hf | prefill_L_minus_1 | 918 | 1.119 | 0.0186 | 1.57e-3 | 5 | 1799 |
| hf | chunked_prefill_64 | 926 | 0.602 | 0.0187 | 1.36e-3 | 8 | 7446 |
| hf | chunked_prefill_256 | 924 | 0.602 | 0.0179 | 1.45e-3 | 7 | 2220 |
| hf | chunked_prefill_1024 | 922 | 1.119 | 0.0188 | 1.60e-3 | 4 | 1809 |
| hf | sdpa | 924 | 0.711 | 0.0173 | 1.40e-3 | 7 | 1512 |
| hf | tf32_on | 0 | 0 | 0 | 0 | 0 | 1831 |
| hf | softcap_fp32 | 922 | 0.148 | 0.0112 | 2.74e-4 | 4 | 1837 |
| hf | logits_fp32 | 922 | 0.163 | 0.0118 | 3.06e-4 | 8 | 2111 |
| hf | rope_elementwise / _tf32_on / rope_inv_freq_cuda | 0 | 0 | 0 | 0 | 0 | 1832-1836 |
| hf | bf16_reduced_reduction_off | 927 | 0.497 | 0.0193 | 1.42e-3 | 6 | 1939 |
| hf | rollout_like_batched_decode | 922 | 0.602 | 0.0188 | 1.40e-3 | 7 | 24537 |
| hf | rollout_like_batched_decode_sdpa | 923 | 0.633 | 0.0177 | 1.29e-3 | 5 | 22142 |
| strict | every factor 1-3 setting, repeat, second_gpu_7_vs_6 | 0 | 0 | 0 | 0 | - | packed 1845, right_pad 9406, paged batch 2098, decode 107818, batched decode 27531, chunk64 8461 |
| cross | strict_vs_hf_baseline | 933 | 0.679 | 0.0192 | 1.11e-3 | - | - |

Per-sequence counts are in records[] (`run_ablation.py merge` prints the setting x sequence matrices). Every HF schedule factor gives 38-55 of 39-55 tokens differing on short sequences and 172-177 of 178 on the long sequence. Every strict row is 0 for all 9 sequences.

Pairwise (analysis.hf_pairwise_identical_tokens):
- Baseline, tf32_on, the RoPE variants and all repeats are mutually identical (933).
- prefill_L_minus_1 and chunked_prefill_1024 share 757 tokens: identical on the 8 short sequences, which fit in one chunk.
- padded right, padded left and bf16_reduced_reduction_off share about 184 tokens: the long row is identical in all three, which points to the same non-split-K kernel choice at large M.
- Any other pair shares fewer than 25 tokens. The tokens that stay identical sit at saturation (logp about -1e-6 or the soft-cap floor at about -30).

## Per-layer drift (layer_records; frac bf16 elements differ / mean |delta|)
- gsm8k_test_0, padded right: emb 0; L0 0.65 / 2.4e-3; L12 0.88 / 2.1e-2; L25 0.92 / 5.8e-2; L51 0.95 / 0.27 (rms 8.7); final norm 0.95 / 0.10.
- Decode, SDPA and rr_off look the same on test_0.
- prefill_L_minus_1 and chunked_256 on test_0: L0 about 0 (mean 1.1e-6), then 0.68 at L12 and 0.80 at the final norm.
- Long sequence, KV decode: L0 0.04 (rows from 17), L2 0.67, final 0.93.
- tf32_on and the repeat show 0 at every layer.

## Conclusions
In the native bf16 stack, train/rollout drift comes from kernel schedule (GEMM M-dependence and split-K, attention implementation, cache/chunk shapes). It enters in layer 0 and is amplified by depth to one common noise floor: about 1.4e-3 k3 per token, and about 0.5% greedy argmax flips. Fixing one knob leaves the floor unchanged. A smaller, depth-independent part comes from bf16 logits/soft cap. TF32 and RoPE table construction contribute nothing here. The strict path removes all of it: 0/933 differing tokens across batching, padding, paging, decode, continuous batching, chunking, repeats and GPUs. Its offset from HF equals HF's own batching noise.

## Deviations
- The strict batch layouts are run by ablation_strict.packed_pass. It calls the reference's own OpSet in MuseGlimmerStrictModel._run order over several cu_seqlens segments or paged slots, because the public API takes one sequence per call. It imports the private strict_model._row_parallel. The single-sequence modes call the reference API unchanged. The tiny CUDA test proves the packed runner is byte-equal to the reference.
- Strict padding has no key mask. Right padding appends pads inside a segment; left padding is a separate pad segment in front, with positions from 0.
- Strict records have no argmax.
- The HF baseline scores L tokens; the L-1 rollout prefill is its own setting.
- First-sequence timings include warm-up.
- flash_attention_2: not installed, so not run.

## How to reproduce
Run from the worktree root; S is the session scratchpad. The exact command lines are in ablation.json commands[].
```
T="~/bin/gpu-lease -n 1 --pool 6-7 --timeout 7200 -- env PYTHONPATH=$PWD ~/envs/vllm/bin/python rl_engine/models/muse_glimmer/ablation/run_ablation.py"
$T hf --out $S/parts/part_hf.json
$T strict --out $S/parts/part_strict.json
$T hf-layers --out $S/parts/part_hf_layers.json
# null controls: hf-baseline and strict --settings repeat_immediate, each pinned with --pool <the other GPU>
# combination: hf --settings rollout_like_batched_decode rollout_like_batched_decode_sdpa
PYTHONPATH=$PWD ~/envs/vllm/bin/python rl_engine/models/muse_glimmer/ablation/run_ablation.py merge --parts $S/parts/*.json --out reports/muse_glimmer_poc/data/ablation.json
~/bin/gpu-lease -n 1 --pool 6-7 -- env PYTHONPATH=$PWD ~/envs/vllm/bin/python -m pytest tests/models/muse_glimmer/test_ablation.py -q   # 13 passed
```

## Figures for the HTML blog (all from ablation.json)
- (a) Bar chart of settings[].frac_bits_differ and mean_k3 (log scale), HF vs strict side by side. Strict is all zero.
- (b) Heatmap of setting x sequence n_bits_differ / n_tokens from records[].
- (c) Per-token dlogp strip for gsm8k_long_window: padded_right, kv_cache_decode, sdpa, softcap_fp32 and strict, where strict is a flat zero line.
- (d) Bars of mean |dlogp| by gold-token probability bucket (analysis.abs_dlogp_by_baseline_prob).
- (e) Per-layer growth lines: frac_elems_differ, and mean_abs / ref_rms on a log scale (layer_records[].layers[]). The headline image is 'one ulp in layer 0 becomes 95% of the residual stream'.
- (f) Pairwise identical-token matrix (analysis.hf_pairwise_identical_tokens).
- (g) TF32 probe: square GEMM 1,048,489 changed elements vs RoPE K=1 GEMM 0.
- (h) Wall-time bars: baseline, decode and batched decode for HF vs strict.

## Time spent
About 1.5 h of agent time and about 17 GPU-minutes.