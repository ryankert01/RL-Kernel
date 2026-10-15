# REPORT — RFC #501 rows `normalized_embedding` + `fused_add_rmsnorm` (Muse-Glimmer-30B strict POC)

## Wave B update (review wave A2: norms:R1, norms:R2) — supersedes the text below where it disagrees

- **Golden division is now IEEE (norms:R1).** The golden computed `ms = ss / float(H)`. In torch
  CUDA a division by a CPU scalar is `ss * (1/H)` (two roundings), on the CPU it is a true
  division, so the golden was not the kernels' `div_rn(ss, H)` and differed between devices.
  `inv_rms_fp32` now uses `div_ieee(ss, H)` = `torch.div` by a 0-dim tensor on `ss`'s device (a
  true IEEE-RN division on both devices). New test `test_golden_mean_square_is_ieee_division`
  (cpu, cuda): the golden's `inv` equals `1/sqrt(rn32(ss/H) + eps)` computed from the golden's
  own `ss` with a correctly rounded division, bit for bit, at 4096 x 6656; the CUDA case fails on
  f91e8b6 and passes now. The strict kernels are unchanged (they already used `div_rn`); only the
  tolerance reference moved. The remaining kernel-vs-golden differences come from the `ss`
  reduction tree only.
- **Fail closed, pinned by tests (norms:R2).** `test_embedding_out_of_range_ids_fail_closed`
  (subprocess: the device-side assert poisons the context) checks that ids `[3, 100]` on a
  100-row table raise with `token_ids must be in [0, 100)` while `[3, 99]` pass.
  `fused_post_norm_residual_pre_norm` now requires `branch`, `resid`, `w_post`, `w_pre` on one
  device (it used to check each tensor only against its own weight):
  `test_fused_rejects_mixed_devices` (meta vs cpu tensors, runs anywhere) and
  `test_fused_rejects_two_gpus` (cuda:0 vs cuda:1; SKIPPED here: one GPU per lease, NOT RUN).
- Tests: `tests/models/muse_glimmer/test_norms.py` 47 passed, 1 skipped (the two-GPU case) on one
  leased A100 of pool 3,7 (see the Test commands section for the command).

Branch `worktree-agent-ae93c2f69853aebf8` (merged from `feat/museglimmer-arch-fingerprint`).

Files:
- `rl_engine/models/muse_glimmer/triton/norms.py` — Triton kernels (row forward/backward, fused
  forward/backward, fixed-order column reduction) + launchers.
- `rl_engine/models/muse_glimmer/ops/norms.py` — fp32 goldens, strict bf16 ops (autograd),
  `trace()`.
- `tests/models/muse_glimmer/test_norms.py` — G1/G2/G4/G5 + fusion-equals-composition.
- `benchmarks/models/muse_glimmer/bench_norms.py` — strict vs torch-eager at T=4096.

## Summary

All five strict ops of the two rows are implemented and pass every gate at the real shape
(H=6656, T∈{1,16,4096}, V=202048, real checkpoint embedding table):

| op | G1 vs fp32/fp64 golden | G2 batch inv. (`torch.equal`) | G4 bwd repeat / vs fp32 | fusion == parts |
|---|---|---|---|---|
| `normalized_embedding` | pass (max abs 1.56e-2 = ½ bf16 ulp) | pass | pass (dweight via repo sorted-segment) | n/a |
| `centered_rmsnorm` (eps 1e-5, 1e-8) | pass | pass | pass (dx also batch-invariant) | n/a |
| `weighted_rmsnorm` (final norm) | pass | pass | pass | n/a |
| `residual_add` | exact | exact | native | n/a |
| `fused_post_norm_residual_pre_norm` (centered / weighted pre) | pass | pass | pass (both / pre-only / rnew-only grads) | **`torch.equal` on both outputs** |

43/43 tests pass in 8 s. Strict kernels are 3–8× faster than the torch-eager HF formulas
(forward), 1.4–5× for fwd+bwd. No contract deviations except the rsqrt-form clarification below.

## Design

Shared row arithmetic (forward and backward, every kernel; one logical row per program, R3):

1. `x` bf16 → fp32, zero-padded to `BLOCK_H = next_pow2(H) = 8192`.
2. `ss = tl.sum(x*x, axis=0)` — Triton's fixed reduction tree for `(BLOCK_H=8192, num_warps=8)`;
   padded lanes add exact zeros. Same block layout in every kernel, so the fused kernel and the
   standalone norm produce the same `ss` bits.
3. `ms = div_rn(ss, H)`; `inv = div_rn(1.0, sqrt_rn(ms + eps))` — **the chosen "rsqrt form"**:
   IEEE round-to-nearest sqrt and division. The result is uniquely determined by IEEE-754 given
   `ss`, so any implementation that really performs these IEEE operations reproduces it bitwise;
   the libdevice `rsqrt`/`pow(-0.5)` approximations are not reproducible across toolchains.
   Pitfall (review norms:R1): in torch CUDA, `t / python_float` is `t * (1 / python_float)`
   (ATen's CPU-scalar fast path, two roundings), so a torch implementation must divide by a
   tensor. The golden does `ms = div_ieee(ss, H)` (0-dim tensor divisor) and
   `1.0 / torch.sqrt(ms + eps)` (`reciprocal`, one IEEE-RN division) — the same two IEEE-RN ops.
4. `xhat = x * inv`; gain in fp32: `(1 + w)` (centered), `w` (weighted), none (embedding);
   `w` bf16 → fp32, `1 + w` formed in fp32.
5. One cast to bf16 at the store (R5). Launches use `enable_fp_fusion=False`, so no FMA
   contraction: the written multiply/add sequence is the executed one on both sides.

Golden reduction order: `ss` = sum of 32-wide chunk partials then sum over the H/32 partials
(torch fp32, fixed inner shapes). Golden and kernel differ only in the `ss` tree; everything
after `ss` is identical IEEE-RN arithmetic (true since wave B: before, the golden's `ss / H`
was a reciprocal multiply on CUDA, see the Wave B update). Measured consequence at T=4096 (with
the pre-wave-B golden): strict output equals `golden.to(bf16)` for all but ~5e-6 of elements
(rounding-boundary flips), max abs error 1.56e-2 (½ bf16 ulp at |y|∈[4,8)); golden-vs-fp64
max 1e-6. Wave B re-measurement, same script on both trees (T=4096, H=6656,
`kernels_attention_raw/waveB/norm_golden_mismatch_{base,new}.json`), strict vs
`golden.to(bf16)` mismatch fraction, f91e8b6 golden -> IEEE golden: centered eps 1e-5
4.95e-6 -> 2.31e-6, centered eps 1e-8 4.84e-6 -> 2.24e-6, weighted 5.03e-6 -> 2.20e-6, fused pre
7.23e-6 -> 3.41e-6, fused resid_new 1.83e-6 -> 0.84e-6, embedding (table scale 0.02) 1.10e-6 ->
3.41e-6. All remaining flips come from the `ss` tree.

Fused kernel `[F1]` (one row per program): `post = bf16((b*inv_post)*(1+w_post))`
(declared cast), `r' = bf16(r→fp32 + post→fp32)` (declared cast, stored), then the pre-norm of
`r'` with `(1+w_pre)` (centered) or `w_pre` (weighted, used before the final norm). Because it
reuses the same jit helpers and block layout, it is bitwise equal to
`residual_add(resid, centered_rmsnorm(branch))` → `centered|weighted_rmsnorm` (tested at
T∈{1,16,4096}, both pre forms).

Embedding: the row source is `weight[ids[row]]` (int64 gather, out-of-range ids masked to a zero
row + `torch._assert_async`), then the weightless norm. No sqrt(H) scaling. Output bf16.

Backward (row-local fp32 VJP of `y = xhat*g`): `dxhat = dy*g`; `dot = tl.sum(dxhat*xhat)`;
`dx = inv*(dxhat − xhat*div_rn(dot,H))` → bf16. Weight grads: fp32 per-row partials `dy*xhat`
written by the row program (no atomics), then `colsum_fixed_order`: left fold over blocks of 32
rows, then over the block partials, until one row remains (4096 → 128 → 4 → 1); the order is a
pure function of T, so repeats are bit-identical; cast to bf16 once (measured: fp32-level error
4.5e-5 on sums of magnitude 242 vs fp64). Fused backward keeps the chain in fp32:
`g_r = pre_vjp(d_pre) + d_rnew` (no intermediate bf16 rounding), `d_resid = bf16(g_r)`,
`d_branch = bf16(post_vjp(g_r))`, `dw_pre = Σ d_pre*xhat_pre`, `dw_post = Σ g_r*bhat`.
Embedding dweight: fp32 `dx` rows (not rounded) → the repo's deterministic sorted-segment
backward `rl_engine.backends.shared.triton.embedding.embedding._embedding_grad_weight` (stable
argsort by token, one program per sorted position × 128-column block, segment start sums its
rows in sorted order, single cast to bf16). Reused, not re-implemented.

Tiles (R6, in `trace()`): row kernels `BLOCK_H=8192, num_warps=8, num_stages=1` for H=6656
(`next_pow2(H)`, 4 warps below 4096); column reduction `ROWS=32, BLOCK_C=256, num_warps=4,
num_stages=3`. No autotune; no runtime int args to the row kernels (T is the grid only), so
Triton's divisibility specialization cannot create per-batch-size variants.

## Contract deviations

- **rsqrt form (clarification, allowed by the brief).** O1 says `(ms+eps)^(-1/2)`, O2 says
  `rsqrt`, `workload.py` records HF's `pow_neg_half`/`rsqrt`. The strict path uses
  `div_rn(1, sqrt_rn(ms+eps))` (IEEE-RN) in both golden and kernel — see Design §3 for why.
  Difference from libdevice `rsqrt` is ≤2 fp32 ulp on `inv`, far below bf16 output rounding;
  the contract text should name this form so other agents (O4 qk-norm) match it.
- **Fused golden keeps the declared intermediate casts.** `fused_post_norm_residual_pre_norm_fp32`
  returns `resid_new` as fp32 holding bf16-rounded values and `pre_normed` as the fp32 norm of
  that bf16 residual, because the contract declares both casts as semantics of the residual
  stream. The casts are implemented as `declared_cast_bf16` (round forward, identity backward)
  so autograd through the golden is the fp32 VJP chain the fused kernel implements. A plain
  `.to(bf16)` in the golden rounds the residual-stream *gradient* to bf16 and its noise
  random-walks into `dw_post` (3% of columns outside tolerance at T=4096) — that was a reference
  artefact, not a kernel error, and is how an unfused autograd composition would behave.
- No sequence-split gate (G3): all ops are row-local; batch invariance covers it.

## Test commands and results

```
PYTHONPATH=$WT ~/bin/gpu-lease -n 1 -- ~/envs/vllm/bin/python -m pytest \
    tests/models/muse_glimmer/test_norms.py -q -p no:cacheprovider
[gpu-lease] using GPUs [1]
...........................................                              [100%]
43 passed in 8.05s
```
(includes `test_embedding_real_checkpoint_weights`: `~/models/Muse-Glimmer-30B/REVISION` exists;
`model.language_model.embed_tokens.weight` loaded from shard 1 via safetensors is bf16
(202048, 6656); strict output finite, unit RMS on non-zero rows, matches golden, batch-invariant.)

Tolerances: `ws1.json accuracy.default.reduction.bfloat16` (atol 0.05, rtol 0.02) for norms and
their grads; `elementwise.bfloat16` (0.02, 0.016) for the embedding forward (its op_class in the
repo); `reduction.float32` for golden-vs-fp64. Observed forward error is ½ bf16 ulp (1.56e-2 at
|y|<8), i.e. 3× below the atol alone.

## Parity results

- Batch: same row at batch 1 vs 4 (position 2) vs 16 (position 15) vs 4096 (positions
  0/1/777/4095): `torch.equal` for centered (both eps), weighted, embedding (small and real V),
  fused (both outputs, both pre forms). dx of `centered_rmsnorm` row 777 at batch 1 equals its
  batch-4096 bits.
- Repeat: two backward runs bit-identical for all ops (dx, dw, embedding dweight at V=202048).
- Fusion: `fused_post_norm_residual_pre_norm` ≡ unfused composition, `torch.equal`, T∈{1,16,4096}.
- Strict vs `golden.to(bf16)` mismatch fraction at T=4096 (pre-wave-B golden, see above): 4.8e-6 / 5.0e-6 / 4.6e-6 / 3.9e-6 /
  7.7e-6 (centered 1e-5, centered 1e-8, weighted, embedding, fused pre); fused `resid_new`
  differs from the golden's bf16 residual in 2.3e-6 of elements (one ulp, from `post` flips).

## Benchmark (A100-SXM4-80GB, T=4096, H=6656, V=202048, median of 50, torch 2.13 / Triton 3.7.1)

Native = HF formula in torch eager (bf16 in, fp32 compute, bf16 out).

| op | strict ms | native ms | strict/native |
|---|---:|---:|---:|
| centered_rmsnorm fwd | 0.141 | 0.804 | 0.18x |
| weighted_rmsnorm fwd | 0.137 | 0.798 | 0.17x |
| normalized_embedding fwd | 0.212 | 0.686 | 0.31x |
| fused_post_norm_residual_pre_norm fwd | 0.213 | 1.588 | 0.13x |
| centered_rmsnorm fwd+bwd | 0.627 | 2.620 | 0.24x |
| fused_post_norm_residual_pre_norm fwd+bwd | 1.123 | 5.486 | 0.20x |
| normalized_embedding fwd+bwd (V=202048) | 5.139 | 7.182 | 0.72x |

Row kernels move 2×T×H bf16 ≈ 109 MB in 0.14 ms ≈ 0.8 TB/s (near A100 bandwidth). Obvious next
optimisation (not done): fuse the `dy*xhat` partial rows into a multi-row backward program to
skip the (T,H) fp32 partial buffer; and `zeros(V,H)` in the embedding backward (2.7 GB memset)
dominates its 5 ms.

## Integration requests

None required: the ops are plain importable functions
(`from rl_engine.models.muse_glimmer.ops.norms import normalized_embedding, centered_rmsnorm,
weighted_rmsnorm, residual_add, fused_post_norm_residual_pre_norm` + `*_fp32`, `trace`).
Optional, for the integrator:
- `rl_engine/contracts/profiles/precision/ws1.json`: if a row is added for
  `normalized_embedding`, use `op_class="reduction"` (it contains a reduction; the generic
  `embedding` row is `elementwise`).
- `record_backward` kinds emitted: `centered_rmsnorm`, `weighted_rmsnorm`,
  `normalized_embedding`, `fused_post_norm_residual_pre_norm` (family `triton`).

## Open issues

- Backward provenance of the unfused composition differs from the fused op: autograd through
  `residual_add` accumulates the residual-stream gradient in bf16, the fused backward keeps it in
  fp32. The reference-model agent should use the fused op in the layer (as the contract's model
  assembly says) so training uses one gradient path; the unfused ops are for the final norm and
  for the composition test.
- `normalized_embedding` validates ids with `torch._assert_async` (no host sync); the kernel
  itself maps an out-of-range id to a zero row, so rejection rests on that assert (pinned by
  `test_embedding_out_of_range_ids_fail_closed` since wave B). The assert poisons the CUDA
  context: the process cannot continue after it (fail closed, no recovery).
- Triton compiles one variant per `(H, eps, form)`; first-call latency ~1–2 s each (shared
  `~/.triton` cache).

## RFC feedback (how the rows should be re-scoped)

- **`fused_add_rmsnorm` is the wrong name and the wrong unit.** Muse Glimmer has a four-norm
  sandwich per layer and the fusable pattern is *post-norm → residual add → pre-norm*
  (`r' = r + post_norm(branch); x = pre_norm(r')`), not the Llama-style `add → norm`. The row
  should be renamed `post_norm_residual_pre_norm` (or `sandwich_norm`) and specify: (a) the
  centered `(1 + w)` gain — it is NOT `w`; (b) **two eps values** in one op (post: 1e-8, pre:
  1e-5) — a single `eps` argument cannot express the layer; (c) the bf16 cast of the post-norm
  output *before* the add and the bf16 residual stream; (d) the pre-norm form switch (centered
  inside the layer, weighted `x̂·w` for the final norm after layer 51). The first `input_norm`
  of layer 0 is the only unfused centered norm in the model; everything else is the fused op.
- **`normalized_embedding` should state the weightless norm explicitly**: lookup bf16 row →
  fp32 → RMSNorm with no weight, eps = `rms_norm_eps` (1e-5), **no `sqrt(H)` scaling** (unlike
  Gemma), → bf16; and its backward is "row-local fp32 VJP, then the canonical sorted-segment
  dweight" — the repo already has that path, so the row is a wrapper plus a norm VJP, not a new
  kernel family.
- Both rows should name the rsqrt form (`1/sqrt` IEEE-RN) and that the dw reduction is a
  fixed-order column fold, so O4's qk-norm (same statistic per 128-wide head) can share the
  helper and the TP agents know dw is already order-fixed before any cross-rank tree.
- Correction (wave B, review norms:R1): an earlier version of this section said any torch
  CPU/CUDA implementation of this form reproduces it bitwise. That holds only when torch really
  divides: `t / python_float` on CUDA is a multiply by the reciprocal. The contract text should
  say "IEEE-RN division (`div_rn`; in torch, `torch.div` by a tensor)".

## Time spent

~2.5 h: reading (contract, workload, repo norm/embedding prior art) 35 min; kernels + ops 50 min;
tests/benchmark 40 min (one reference-artefact debug: bf16 gradient rounding in the golden's
casts); report 20 min.
