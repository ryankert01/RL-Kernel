# REPORT — RFC #501 row `full_model_chain` (Muse-Glimmer-30B strict POC)

STATUS: complete. Chain gate byte-equal on real weights for both backends; HF bf16 and HF fp32 anchors
measured; tiny gates 15/15; real-weights pytest gate passed (triton backend). (Committed by the integrator
from the implementing agent's hand-back; the agent's harness blocked it from writing REPORT files.)

## Summary
1. Strict 52-layer reference: ONE op composition for train-forward (varlen attention), one-shot/chunked prefill
   and decode via paged KV (page 16, 64-key blocks, window eviction); loads the real checkpoint after
   `verify_checkpoint` (627 bf16 tensors, 51.9 GiB, 11 s).
2. `backend="golden"` = pure-torch fp32 fixed-order fallbacks + repo `TritonDetGemmOp.forward_fp32` GEMM;
   `backend="triton"` = all 12 sibling ops (attention forward included).
3. REAL WEIGHTS, prompt 512 + 32 completion: train == prefill == 4 chunkings (prompt-boundary, halves, 64-aligned,
   ragged incl. 1-token chunks) == decode, **byte-equal 32/32 on BOTH backends**. Golden timings: train 4.7 s,
   prefill 4.1 s, decode 28 s (31 steps); full-Triton: train 1.2 s, prefill 3.0 s (incl. first compile), decode
   3.9 s (0.13 s/step); peak 52.0–52.6 GiB.
4. Prompt 2100 + 32 (golden, W=2048): 10 paths incl. chunk boundaries 2047/2048/2049 all byte-equal (train 36 s,
   decode 89 s, peak 54.3 GiB). Pages are really evicted on local layers (tiny test asserts it).
5. HF anchor (transformers 5.17.0 `MuseGlimmerForConditionalGeneration`, eager, bf16, same 544 tokens): golden
   max|Δlogp| 0.064 / mean 0.022 on the completion, argmax agreement 98.5 % (100 % on the completion);
   full-Triton 0.066 / 0.025 / 98.9 %. max|Δlogit| 0.53 (golden), 0.84 (Triton) — single bf16 outlier channels
   (ulp 2–8).
6. HF fp32 (CPU) cross-check: the strict paths are CLOSER to fp32 than HF-bf16 at every layer: completion
   max|Δlogp| golden 0.041 / Triton 0.046 / HF-bf16 0.046; mean 0.0125 / 0.016 / 0.019; argmax 99.63 / 99.63 /
   98.53 %. Layer-50 residual mean|Δ| 0.040 (golden), 0.040 (Triton), 0.050 (HF-bf16) at rms 7.3; the error grows
   ≈ linearly with depth, steeper after layer 40. A "≤1e-1 logits" bar is met by NO bf16 path, HF's own included
   (0.55 vs fp32) — logp is the robust metric.
7. Golden vs Triton per path: max|Δlogp| 0.048, mean 0.013 (identical for every path, since each backend is
   internally byte-equal).
8. Tiny config ([L,L,L,G], H=256, 8/2 heads, D=32, FFN 512, vocab 1024/1000, W=100): 15/15 tests —
   train == prefill == chunked == decode == per-token `torch.equal`, backward deterministic for all 51 parameters,
   drift localizer verified (injected error → `layers.0.attn`), HF tiny tolerance.

## Design
Files under `rl_engine/models/muse_glimmer/reference/`: `golden_ops.py` (fp32 fixed-order fallbacks with the
sibling names/signatures; `GoldenGemm`; Linear/LmHead holders), `paged_kv.py` (per-layer paged KV, page 16, block
tables, window eviction, reserved zero page), `backend.py` (`resolve_ops("golden"|"triton", allow_partial)` →
`OpSet` with per-node provenance), `weights.py` (`StrictConfig` official/tiny; `MuseGlimmerWeights.load_checkpoint`
/ `.random`), `strict_model.py` (`forward_train`, `prefill(chunk_sizes)`, `decode_step`,
`logprobs(prompt, completion, mode)`, node capture), `chain_harness.py` (`compare_paths`, `localize_first_drift`,
`compare_to_hf`, tiny HF helpers), `run_chain.py` (CLI `--checkpoint --backend --prompt-len --completion-len
--hf-compare --save-dir`); tests in `tests/models/muse_glimmer/`.

Reduction orders / casts (golden backend): row reductions = adjacent-pair halving then sequential (elementwise
only → batch/chunk/decode invariance by construction) [O1, O2, O4, O7]; GEMM = `TritonDetGemmOp.forward_fp32`
(one program per 64×64 tile, K in BLOCK_K=32 `tl.dot(allow_tf32=False)` steps, fp32 accumulate, no split-K, one
rounding; row-parallel = 8 fp32 K-chunk partials combined as ((0+1)+(2+3))+((4+5)+(6+7)) → bf16 once) [O3];
attention = 64-key blocks aligned at 0, ascending from floor(max(0, p−W+1)/64), every block padded to 64 keys,
fixed trees over D=128 and 64 keys, the contract's update order, p→bf16 before PV, o = acc/l → bf16,
y = bf16(fp32(o)·sigmoid(fp32(g))) with sigmoid = 1/(1+exp(−x)), lse = m + log l, m = −inf guard (α = 1, p = 0),
paged gather of 4 pages per block with −1 → zero page [O5]; Q/K norm pow −0.5, q *= 3.87 in fp32, fp32 rotate_half
RoPE with a cached fp32 CUDA table, one cast [O4]; centered norms rsqrt·(1+w), final pow −0.5·w, residual
bf16(fp32 + fp32), fused [F1] inside layers [O2]; SwiGLU bf16(g/(1+exp(−g))·u); lm_head 8 vocab tiles, fp32 logits,
T·tanh((z·m)/T), padded columns −inf, (M_t, S_t) tree [O6, O7].

## Contract deviations
1. Golden GEMM is the repo fp32 FMA Triton kernel (not per-chunk MMA, not a torch loop).
2. Golden attention uses explicit trees, not `tl.dot`.
3. The Triton backend applies the 8-chunk tree to column-parallel GEMMs too (`fast_det_gemm` does it for every K).
4. The tiny-config embedding backward is a local sorted-segment function (the repo canonical one is session-bound).
5. The deliberate fp32-vs-HF differences (single final rounding in the Q/K path, fp32 logits) are why HF is a
   tolerance anchor only.

## Sibling wiring facts the consumers need
- `fast_det_gemm` does the K tree internally → the row-parallel call is a plain call at TP1.
- The sibling lm_head rejects −1 targets → the model masks targetless rows before calling it.
- Paged `seq_lens` and varlen `q_pos` must be int32.
- The sibling attention kernels are hard-wired to 32/2/128 heads → tiny-config tests are golden-only.

## Test commands and results
```
~/bin/gpu-lease -n 1 -- env PYTHONPATH=$WT ~/envs/vllm/bin/python -m pytest \
    tests/models/muse_glimmer/test_strict_model_tiny.py tests/models/muse_glimmer/test_museglimmer_fingerprint.py -q
# → 47 passed, 1 skipped in 27.57s
MUSEGLIMMER_CHECKPOINT=/home/ubuntu/models/Muse-Glimmer-30B MUSEGLIMMER_BACKEND=triton \
~/bin/gpu-lease -n 1 -- env PYTHONPATH=$WT ~/envs/vllm/bin/python -m pytest tests/models/muse_glimmer/test_full_model_chain.py -q -s
# → | decode | - | yes | 32/32 | 0.000e+00 | 3.80 | 52.05 | … 1 passed in 21.76s
~/envs/vllm/bin/python rl_engine/models/muse_glimmer/reference/run_chain.py --checkpoint ~/models/Muse-Glimmer-30B \
    --backend golden|triton --prompt-len 512 --completion-len 32 --hf-compare --json-out out.json
```

## Parity results (real weights)
Golden 512+32 (all equal 32/32): train 4.70 s / 52.55 GiB; prefill 4.13; chunked prompt_boundary [512,31] 4.36;
halves [271,272] 4.93; block64 (9 chunks) 6.66; ragged [1,63,100,7,129,30,…] (11 chunks) 9.46; decode 28.27.
Triton (all sibling ops) 512+32 (all equal 32/32): train 1.20; prefill 2.97 (first compile); prompt_boundary 0.43;
halves 0.44; block64 1.11; ragged 2.60; decode 3.92; peak 52.0–52.1 GiB. Sibling ops + golden attention
(intermediate branch state): also 7/7 equal.
Golden 2100+32 (all equal 32/32): train 35.8; prefill 31.7; prompt_boundary 33.8; halves 36.2; block64 (34) 43.3;
ragged (39) 64.1; window_boundary_2047 [2047,84] 32.6; _2048 [2048,83] 33.5; _2049 [2049,82] 36.1; decode 89.4;
peak 54.3 GiB.
First drift never triggered on real weights; the localizer is verified on the tiny config (injected error →
`layers.0.attn` / attention_local, row 0). Golden vs Triton: max|Δlogp| 0.048, mean 0.013 for every path.

## HF comparison (544 tokens; completion = tokens 511..542)
max|Δlogit| all / completion, max|Δlogp| completion, mean|Δlogp|, argmax agreement (all):
golden vs HF-bf16 0.529 / 0.293, 0.064, 0.022, 0.985 | Triton vs HF-bf16 0.836 / 0.257, 0.066, 0.025, 0.989 |
golden vs HF-fp32 0.444 / 0.173, 0.041, 0.0125, 0.9963 | Triton vs HF-fp32 0.474 / 0.226, 0.046, 0.016, 0.9963 |
HF-bf16 vs HF-fp32 0.550 / 0.263, 0.046, 0.019, 0.9853 | golden vs Triton 0.415 / –, 0.048, 0.013, 0.9963.
Per-layer residual vs fp32 (mean/max |Δ|: golden | Triton | HF-bf16 | rms): emb 8.7e-4/0.062 (all) | 1.00;
L0 2.5e-3/0.29 | 2.5e-3/0.29 | 3.5e-3/0.33 | 1.12; L5 7.8e-3/0.75 | same | 1.4e-2/0.75 | 1.62; L10 1.16e-2/4.2 |
same | 2.0e-2/4.2 | 2.03; L20 1.54e-2/1.6 | 1.54e-2/1.6 | 2.4e-2/1.8 | 2.83; L30 1.64e-2/2.5 | 1.62e-2/2.0 |
2.2e-2/2.9 | 2.81; L40 1.87e-2/4.4 | 1.86e-2/4.4 | 2.3e-2/4.4 | 3.33; L45 2.59e-2/10.5 | 2.57e-2/10.5 |
3.3e-2/10.5 | 4.43; L50 4.02e-2/6.8 | 3.99e-2/5.4 | 5.0e-2/5.8 | 7.29; final norm 1.14e-2/0.65 | 1.16e-2/0.59 |
1.47e-2/0.80 | 1.01. Max diffs are single outlier channels (bf16 ulp 2–8); the mean grows ≈ linearly with depth.

## Benchmark / memory
Weights 51.9 GiB (11 s load); peaks 52.0–54.3 GiB; HF bf16 load 13–15 s, forward 0.48 s; HF fp32 on CPU load
7.6 s, forward 27.5 s (64 threads); golden train 4.7 s @544 / 35.8 s @2132, decode 0.9 s/step; Triton train 1.2 s,
decode 0.13 s/step.

## Integration requests
None required. Optional ws1.json row `muse_glimmer_full_model_hf_anchor`:
`{"logp_max_abs": 0.1, "logp_mean_abs": 0.03, "argmax_agreement_min": 0.97}` (observed 0.066 / 0.025 / 0.985).

## Open issues
Sibling attention kernels hard-wired to 32/2/128 heads (tiny tests golden-only); the full-model backward does not
fit on one GPU (tiny only); the golden decode is slow at long context (Python block loop).

## RFC feedback
- `full_model_chain`: define the gate as byte-equal selected-token logp across train / prefill / chunked
  (prompt-boundary, 64-aligned, ragged, W−1/W/W+1) / decode on ONE op set, plus a tolerance anchor vs HF-bf16 AND
  vs fp32 (HF-bf16 alone cannot distinguish a correct strict path from HF's own rounding); require node-named
  first-drift reports; state that lm_head must accept or mask targetless rows.
- `hybrid_attention_regression`: run T > W with chunk boundaries W−1/W/W+1 and assert real eviction on local
  layers; O5 must spell out the m = −inf guard (α = 1, p = 0) and the exact-zero contribution of masked/evicted
  (possibly NaN) keys.

Time spent ≈ 6 h.
