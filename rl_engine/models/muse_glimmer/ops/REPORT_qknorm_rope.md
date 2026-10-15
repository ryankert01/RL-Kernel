# REPORT — RFC #501 row `qknorm_scaled_rope` (contract O4)

Branch `worktree-agent-a1476a61a3b149700` (merged `feat/museglimmer-arch-fingerprint`).
Agent: qknorm_scaled_rope. Hardware: A100-SXM4-80GB, torch 2.13.0+cu130, Triton 3.7.1.

## Summary

- Fused Triton op for the per-head weightless Q/K RMSNorm, the fp32 query scale 3.87 and the
  rotate_half RoPE over all 128 channels (local layers; `cos=None` selects the NoPE path of the
  13 global layers), bf16 in → fp32 → ONE rounding to bf16 at the end. Deterministic row-local
  Triton backward (`torch.autograd.Function`). Fixed-order torch goldens in fp32 and fp64 and a
  hand-written fixed-order fp32 VJP.
- All contract gates pass (39 tests, `tests/models/muse_glimmer/test_qknorm_rope.py`):
  G1 within ws1 **elementwise** bf16 tolerances vs fp32 and fp64 goldens; G2 batch invariance,
  G3 position/sequence-split invariance, G4 backward repeat determinism + batch invariance,
  all by `torch.equal` at the real shape T=4096; G5 trace.
- Stronger than required: the kernel output equals `qknorm_scaled_rope_fp32(...).to(bf16)`
  **bitwise** (0 of 16.8 M q̂ and 1.05 M k̂ elements differ at T=4096), and the Triton backward
  equals the hand-written fp32 VJP → bf16 bitwise. The golden therefore literally defines the
  strict numerics.
- HF semantics measured at the real shape: HF's multi-rounding bf16 path differs from the
  contract by up to **0.25 abs** on q̂ (mean 8.0e-3; 47 % of elements differ; 0.14 % lie
  beyond the ws1 bf16 elementwise tolerance, worst err/tol 4.8) and up to 0.031 on k̂. Against
  the fp64 truth the strict op has max error 6.2e-2 (= one bf16 half-ulp at |q̂| ∈ [16, 32)),
  mean 4.4e-3; HF has max 0.22, mean 9.1e-3 (2.1× the mean, 3.5× the max error).
- Benchmark T=4096: strict forward 0.076 ms vs torch-native HF-style 0.96 ms (12.6× faster,
  ~990 GB/s effective); forward+backward 0.42 ms vs 2.39 ms (5.7×).

## Design

Files
- `rl_engine/models/muse_glimmer/triton/qknorm_rope.py` — kernels + launchers.
- `rl_engine/models/muse_glimmer/ops/qknorm_rope.py` — `rope_table`, `rope_table_fp64`,
  `qknorm_scaled_rope` (strict, autograd), `qknorm_scaled_rope_fp32/fp64` (goldens),
  `qknorm_scaled_rope_backward_fp32` (hand-written VJP), `trace()`, `QKNormScaledRopeOp`.
- `tests/models/muse_glimmer/test_qknorm_rope.py`,
  `benchmarks/models/muse_glimmer/bench_qknorm_rope.py`.

Public interface (as fixed in the brief)
```python
rope_table(positions: (T,) int32|int64, *, theta=500000.0, dim=128, device) -> (cos, sin) fp32 (T, 128)
qknorm_scaled_rope(q (T,32,128) bf16, k (T,2,128) bf16, cos (T,128) fp32 | None, sin | None,
                   *, eps=1e-5, qk_scale=3.87) -> (q_hat bf16, k_hat bf16)      # differentiable
qknorm_scaled_rope_fp32(...) -> fp32 outputs    # golden; autograd through it = fp32 reference backward
trace() -> dict
```
Usage: `cos, sin = rope_table(torch.arange(max_pos, device=dev), device=dev)` once per device,
then `qknorm_scaled_rope(q, k, cos[pos], sin[pos])` (local) / `qknorm_scaled_rope(q, k, None, None)`
(global). Gathered rows are bitwise identical to `rope_table(pos)` (tested).

Table definition (fp32, on the given device; `rope_table` is THE definition for both runtimes)
```
exponent = arange(0, 128, 2, fp32) / 128 ;  inv_freq = 1 / pow(theta, exponent)     # HF's form
angle = positions.float()[:, None] * inv_freq[None, :]  (elementwise) ; emb = cat(angle, angle)
cos, sin = emb.cos(), emb.sin()
```
This equals HF `MuseGlimmerTextRotaryEmbedding`'s fp32 cos/sin bitwise when HF's `inv_freq` is
computed on the same device (tested with HF's K=1 matmul form, TF32 off).

Kernel structure (R3/R6)
- One program per token; the token's 32 query heads form a `[32, 128]` tile and its 2 key heads
  a `[2, 128]` tile (`BLOCK_H = next_pow2(H)`, padded rows masked). Block layout depends only on
  the constexprs `(BLOCK_HQ=32, BLOCK_HK=2, HEAD_DIM=128, num_warps=4)`, never on T or the
  program id → batch invariance by construction. `num_stages=1` (no loops),
  `enable_fp_fusion=False` (no FMA contraction, so the kernel performs the same IEEE fp32 op
  sequence as the torch golden).
- Reduction order: channels loaded as halves `x1 = x[:, :64]`, `x2 = x[:, 64:]`;
  `ss = tree64(x1*x1) + tree64(x2*x2)` where `tree64` is the explicit binary tree pairing
  `(2i, 2i+1)` at each of 6 levels (`tl.reshape` + `tl.split` + one add per pair); the 7th level
  adds the two half sums. Identical to the golden's `_tree_sum_last` over 128 channels.
- Casts: bf16 → fp32 at load; `rstd = rsqrt(ss/128 + eps)` fp32 (`tl.rsqrt` and
  `torch.pow(x, -0.5)` both lower to CUDA `rsqrtf`); `x_hat = x * rstd`; `q_hat *= 3.87` (fp32,
  Q only); `o1 = x1*c1 − x2*s1`, `o2 = x2*c2 + x1*s2` fp32; one cast to bf16 at the store.
- Backward (same block structure, one program per token): `g = dy → fp32`; inverse rotation
  `g1 = dy1*c1 + dy2*s2`, `g2 = dy2*c2 − dy1*s1` (transpose of the rotation, orthogonal);
  `g *= 3.87` (Q only); `dx = rstd * (g − x_hat * (tree(g*x_hat)/128))` with `rstd` recomputed
  with the forward's exact op order; one cast to bf16. No atomics, no reductions across rows.
- Golden: fixed `(2i, 2i+1)` tree, `torch.pow(ms + eps, -0.5)`, elementwise ops only → the
  golden itself is bitwise batch invariant on any device.

trace() (G5): `{op, row, nodes, dtype_in=bf16, dtype_acc=fp32, dtype_out=bf16, dtype_table=fp32,
cast_points, reduction_order, rotation, scale, eps, rope_theta, tile_config{BLOCK_HQ=32,
BLOCK_HK=2, HEAD_DIM=128, HALF=64, num_warps=4, num_stages=1, enable_fp_fusion=False}, backward,
golden}` — stable across calls, JSON-serialisable.

## Contract deviations

None in the numerics. Clarifications the contract text leaves open (chosen and documented):
1. O4 writes the frequencies as `θ^(-2i/128)`; the implementation uses HF's algebraically equal
   form `1 / pow(θ, 2i/128)` so that `inv_freq` matches HF bit for bit (the two forms can differ
   by an fp32 ulp). The contract should spell out the formula.
2. The strict kernel pins `head_dim = 128` (contract value); other head dims raise. HQ/HK are
   generic (power-of-two padded).
3. The table is "built once and cached" by the caller (gather by position); `rope_table(pos)`
   and `rope_table(arange)[pos]` are bitwise equal, so both patterns satisfy the contract.
4. The op also accepts fp16/fp32 q/k (output dtype = input dtype); the contract path is bf16.
5. `QKNormScaledRopeOp.op_class = "reduction"` follows the repo's `qk_norm` OperatorSpec row;
   the tests gate on the tighter ws1 elementwise rows (bf16 0.02/0.016), which pass.

## Test commands and results

```
cd <worktree>
~/bin/gpu-lease -n 1 -- ~/envs/vllm/bin/python -m pytest tests/models/muse_glimmer/test_qknorm_rope.py -q -s -p no:cacheprovider
```
Verbatim tail:
```
[gpu-lease] using GPUs [2]
...
[rope_table] fp32 vs fp64 table max abs diff at T=4096: 2.815e-04
..........
[G1] fp32 golden vs fp64 golden (fp64 table) max abs: q 3.671e-03 k 7.543e-04
........................
[HF] q_hat: strict-vs-HF max_abs=0.2500 mean_abs=8.040e-03 frac_neq=0.4711 frac_beyond_ws1_bf16_tol=1.395e-03 worst_err/tol=4.77 | err vs fp64 truth: strict max=6.2351e-02 mean=4.360e-03 rms=6.431e-03; HF max=2.1964e-01 mean=9.104e-03 rms=1.364e-02
[HF] k_hat: strict-vs-HF max_abs=0.0312 mean_abs=1.713e-03 frac_neq=0.4118 frac_beyond_ws1_bf16_tol=9.537e-07 worst_err/tol=1.01 | err vs fp64 truth: strict max=1.4874e-02 mean=1.128e-03 rms=1.665e-03; HF max=2.5945e-02 mean=2.038e-03 rms=3.047e-03
..
39 passed in 3.53s
```

Gate-by-gate (tolerances from `ws1.json accuracy.default.elementwise` / `gradient_accuracy.elementwise`):
| Gate | Test(s) | Result |
|---|---|---|
| G1 fwd vs fp32 golden | T ∈ {1, 5, 64, 4096}, input RMS ∈ {0.05, 1, 2} | PASS (bf16 0.02/0.016); additionally **bitwise** equal to golden→bf16 |
| G1 fwd vs fp64 golden (fp64 table) | T=4096 | PASS (bf16 tol); max abs 6.2e-2 = one bf16 half-ulp |
| G1 fp32 golden vs fp64 golden (fp32 table) | T=4096 | PASS fp32 1e-5/1e-5 (max abs q 3.9e-6, k 7.1e-7) |
| G1 global path | `cos=None` ≡ identity table (bitwise), ≡ golden→bf16 (bitwise), RMS(k̂)=1, RMS(q̂)=3.87 | PASS |
| G2 batch invariance | rows at T=1 vs T=4096 (7 rows); same row at every batch position for batch ∈ {2,4,16} with random neighbours/positions; global path; backward | PASS (`torch.equal`) |
| G3 positions | one-shot vs 4 chunks vs per-token (T=97, offset 1000); packed sequences with logical positions; same (q,k,p) at different indices inside T ∈ {3, 64, 4096} sequences | PASS (`torch.equal`) |
| G4 backward | repeat determinism (T=4096, local & global); vs fp32 autograd of the golden (bf16 grad tol) for T ∈ {3, 64, 4096}; **bitwise** vs hand-written fp32 VJP→bf16; hand-written VJP vs autograd within fp32 1e-5 (max abs 1.9e-6); partial-grad path | PASS |
| G5 trace | keys, constants, stability, JSON | PASS |
| rope_table | shape/dtype, duplicated halves, int32 ≡ int64, gather consistency, fp64 bound, HF fp32 bitwise, input validation | PASS |

## Parity results

- Batch: `torch.equal` for the same token row at T=1 vs T=4096 and at every batch position
  (batch 2/4/16) with random neighbours and random neighbour positions, forward and backward.
- Sequence split: `torch.equal` one-shot vs chunked ([0:13],[13:50],[50:64],[64:97]) vs
  per-token; packed two-sequence case; the result depends only on the absolute position.
- Repeat: backward twice → `torch.equal` (local and global paths); forward is bitwise equal to
  the golden, so trivially repeatable.
- Kernel vs golden bitwise: 0 mismatches in q̂ (16,777,216 el.) and k̂ (1,048,576 el.) at T=4096
  after switching the kernel to the golden's pairwise tree (with Triton's native `tl.sum` tree the
  mismatch rate was 3.8e-6, all single bf16 ulps — recorded for the RFC: reduction-order
  differences alone flip ~1 in 260k bf16 outputs).

## Accuracy facts worth stating in the RFC

- fp32 table vs fp64 table: max |Δcos|, |Δsin| = 2.8e-4 at p < 4096 (angle error ≈ p·2⁻²⁴ for
  the inv_freq=1 channel pair). It grows linearly with position: ≈ 7.8e-3 rad at
  p = 131072 (max_position_embeddings). Both HF and the contract define the table in fp32, so
  this is "the semantics", not an error — but it means the fp32 RoPE is not an exact rotation
  at long context and any fp64 reference must use the fp32 table to be meaningful.
- With the fp32 table, the fp32 golden's arithmetic error vs fp64 is ≤ 3.9e-6 abs (q̂ ≤ ~17).
- HF (norm→bf16, ×3.87→bf16, bf16 cos/sin, bf16 q·cos, bf16 rot·sin, bf16 sum) vs contract
  (one rounding) at T=4096, inputs N(0, 2²):

| tensor | max abs Δ | mean abs Δ | frac differing | frac beyond ws1 bf16 tol | worst err/tol | strict err vs fp64 (max / mean / rms) | HF err vs fp64 (max / mean / rms) |
|---|---|---|---|---|---|---|---|
| q̂ | 0.250 | 8.0e-3 | 47.1 % | 1.40e-3 (23.4k el.) | 4.77 | 6.2e-2 / 4.4e-3 / 6.4e-3 | 0.220 / 9.1e-3 / 1.36e-2 |
| k̂ | 0.031 | 1.7e-3 | 41.2 % | 9.5e-7 (1 el.) | 1.01 | 1.5e-2 / 1.1e-3 / 1.7e-3 | 0.026 / 2.0e-3 / 3.0e-3 |

  The numbers are independent of the input scale (checked at RMS 0.05, 1, 2, 20). The large
  HF deviations occur where the rotation cancels (|q̂·cos| ≈ |rot(q̂)·sin| ≈ 10 while the result
  is ≈ 0.2): bf16 cos/sin (rel 2⁻⁹) and the two bf16 product roundings leave an absolute error
  of ~0.1 on a value of ~0.2. Consequence: HF/vLLM-native bf16 RoPE is NOT a bitwise or even
  ws1-tolerance target for this op; the rollout runtime must use this op (or an identical
  single-rounding fp32 kernel) for train–rollout parity.

## Benchmark

```
~/bin/gpu-lease -n 1 -- ~/envs/vllm/bin/python benchmarks/models/muse_glimmer/bench_qknorm_rope.py
device: NVIDIA A100-SXM4-80GB  T=4096 q(4096, 32, 128) k(4096, 2, 128) bf16
pass                strict Triton ms  torch-native HF ms  native/strict
forward                       0.0763              0.9595          12.57x
forward+backward              0.4189              2.3921           5.71x
strict forward effective bandwidth: 989 GB/s (75.5 MB moved)
```
Both sides get precomputed cos/sin (fp32 rows for strict, bf16 for HF); table building excluded.
The strict forward is a single memory-bound kernel at ~2/3 of A100 HBM bandwidth; the
forward+backward figure includes two `clone()`s and autograd overhead (the backward kernel alone
is ≈ 0.1 ms). First-call Triton compile ≈ 0.8 s (cached in `~/.triton`).
Obvious next optimisation (not done): wider vector loads / `tl.multiple_of` hints and a 2-tokens-
per-program variant to approach 1.3 TB/s; or fusing the op into the Q/K GEMM epilogue.

## Integration requests

- No shared file needs a change for the POC chain: `operator_trace.py` already maps
  `q_norm`/`k_norm`/`rope_q`/`rope_k` to row `qknorm_scaled_rope`; tolerances used are the
  existing ws1 elementwise rows.
- Optional, if the integrator wants the op in the WS1 harness
  (`rl_engine/validation/operators/operator_specs.py`; the harness will need an input builder for
  the `(q, k, cos, sin)` signature):
```python
    "qknorm_scaled_rope": OperatorSpec(
        name="qknorm_scaled_rope",
        op_class="reduction",
        gold_path="rl_engine.models.muse_glimmer.ops.qknorm_rope.QKNormScaledRopeOp",
        gold_method="forward_fp32",
        candidate_paths={
            "pytorch": "rl_engine.models.muse_glimmer.ops.qknorm_rope.QKNormScaledRopeOp",
            "triton": "rl_engine.models.muse_glimmer.ops.qknorm_rope.QKNormScaledRopeOp",
        },
        grad_input_names=("q", "k"),
    ),
```
- Reference-model / attention agents: import `rope_table`, `qknorm_scaled_rope`,
  `qknorm_scaled_rope_fp32` from `rl_engine.models.muse_glimmer.ops.qknorm_rope`; q from the
  det_gemm output is `q.view(T, 32, 128)` (contiguous), k is `k.view(T, 2, 128)`.
- Rollout (vLLM) agent: do NOT use vLLM's `RotaryEmbedding`/bf16 `cos_sin_cache` for this model;
  call `qknorm_scaled_rope` with `rope_table` rows gathered by the int32 positions (int32 and
  int64 positions give identical bits).

## Open issues

1. Bitwise kernel ≡ golden relies on `tl.rsqrt` and `torch.pow(x, -0.5)` both lowering to CUDA
   `rsqrtf` (verified on A100 / Triton 3.7.1 / torch 2.13). `test_matches_fp32_golden_bitwise`
   and `test_matches_handwritten_fp32_vjp_bitwise` will flag a regression; the contract gate
   (tolerance) does not depend on it.
2. The `(2i, 2i+1)` tree in Triton costs 6 `reshape+split` levels per 64-channel half; it did not
   change the measured forward time (0.073 ms before, 0.076 ms after) at T=4096.
3. The positional error of the fp32 table at p ≫ 4096 (see above) is inherent to the fp32
   definition shared with HF; if the RFC ever wants exact long-context rotation it must change
   the table definition on both runtimes simultaneously (e.g. fp64 angle → fp32 cos/sin).
4. Decode shape (T = batch, e.g. 1–256 programs) under-fills the GPU; irrelevant for parity, and
   the op is < 10 µs there.

## RFC feedback (what the row text should say)

- "Scaled Q/K norm" is wrong: **only Q is scaled**. HF: `query_states = qk_norm(q) * qk_scale_factor;
  key_states = qk_norm(k)` (`k_scale: None` in the frozen semantics).
- The scale is the config value **`qk_scale_factor = 3.87`, not 43.78**. 43.78 = 3.87 × √128, i.e.
  the RFC multiplied by √head_dim where the softmax scaling divides by it; the effective logit
  multiplier is 3.87 × 128^-0.5 = 0.342 (`attention.effective_q_scale` in `workload.py`).
- The RoPE rotates **all 128 channels** (`rotated_channels: 128`, rotate_half pairing d ↔ d+64;
  no partial rotary), θ = 5e5, on the 39 sliding-window layers only; the 13 global layers
  (i % 4 == 3) are NoPE.
- The row should also pin the table: fp32 `inv_freq = 1/θ^(arange(0,128,2)/128)`,
  `angle = p·inv_freq` elementwise, `cos/sin` fp32, built once on the device and shared by both
  runtimes; and state that HF applies the norm in fp32 but rounds to bf16 four times
  (norm, scale, two products + sum with bf16 cos/sin) whereas the strict op rounds once —
  measured max |Δ| 0.25 on q̂ (0.14 % of elements beyond the ws1 bf16 tolerance), so HF/vLLM
  native RoPE is a tolerance reference, not a parity target.
- Proposed row text: *"qknorm_scaled_rope — per-head weightless RMSNorm of Q and K (eps 1e-5,
  fp32), Q multiplied by qk_scale_factor = 3.87 (K unscaled), rotate_half RoPE over all 128
  channels (θ = 5e5, fp32 table) on sliding-window layers only (NoPE on global layers), one bf16
  rounding at the end; batch/position-invariant Triton kernel with row-local deterministic
  backward."*

## Time spent

≈ 2.5 h (reading 25 min, kernel + golden 40 min, tree/bitwise work 25 min, tests 35 min,
HF quantification + benchmark 15 min, report 20 min).
