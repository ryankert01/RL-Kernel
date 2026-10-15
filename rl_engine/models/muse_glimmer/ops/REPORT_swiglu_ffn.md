# REPORT — RFC #501 row `swiglu_ffn` (contract O6 + O3)

> **Review follow-up (2026-10-15, GEMM correctness row, branch `worktree-wf_7b064961-36f-4`).**
> Six review findings on this op (swiglu_ffn:1–6) were reproduced and fixed; the original report
> below is kept and statements that are no longer true are marked **[superseded]**. Raw numbers:
> `reports/muse_glimmer_poc/data/kernels_gemm.json`.

## Review follow-up: findings, fixes, evidence

| id | verdict | evidence before | fix | test (fails before, passes after) |
|---|---|---|---|---|
| swiglu_ffn:1 | **confirmed, fixed** | default-GEMM backward with trainable weights raised `ValueError: reduction length 64 is not a multiple of 256 ... forward GEMM` for every T % 256 != 0 (T = 1, 64, 255, 300, 4001 measured) | the three dW GEMMs pass `ragged_k=True`: the declared token rule of `fast_det_gemm` (T_pad = roundup(T, 256), 8 chunks of T_pad/8, rows >= T masked); every T >= 1 works | `test_g4_default_gemm_backward_any_token_count[1,64,255,300]` (repeat, fp32 reference, dW == dW of zero-padded tokens, adapter twin bitwise), `test_g4_default_gemm_backward_real_shape_ragged_tokens` (T=300 at H=6656/F=19968) |
| swiglu_ffn:2 | **confirmed, fixed** | `trace()["tile_config"]["gemm"]` was the 3-key stub `{op, impl, kernel}` | `_FastDetGemm.trace()` returns `det_gemm.trace()` (all shape classes, BLOCK_M/N/K, warps, stages, chunk rules, runtime) | `test_g5_default_gemm_trace_carries_launch_constants` |
| swiglu_ffn:3 | **confirmed, fixed** | `TritonDetGemmAdapter` silently used one chunk for K % 8 != 0 | raises `ValueError` unless `ragged_k=True`, which applies the same ragged rule as `fast_det_gemm` (bitwise equal, below) | `test_adapter_gemm_accuracy_invariance_and_strides`, `test_adapter_and_fast_gemm_are_bitwise_twins[36,256,300,512,4001,6656]` |
| swiglu_ffn:4 | **confirmed, fixed** | with `ops.det_gemm` blocked, `default_gemm()` silently returned `TritonDetGemmAdapter` (subprocess printed `SILENT TritonDetGemmAdapter`) | `det_gemm` imported at module level, no fallback; the adapter is explicit-only | `test_default_gemm_has_no_silent_fallback` (subprocess) |
| swiglu_ffn:5 | **confirmed (coverage gap), fixed** | G2 at the real shape pinned the adapter | G2 parametrised over (adapter, default) with T = 1 / 16 (decode16), 64 (decode64), 65 (prefill) slices and gathered rows; adapter == default bitwise at T = 256 | `test_g2_ffn_batch_invariance_real_shape[adapter/default]` (passes before too: invariance held, it was untested) |
| swiglu_ffn:6 | **confirmed, fixed** | `_FastDetGemm` called `.contiguous()` on both operands: 3 transposed 266 MB weight copies + 2 activation copies per layer backward | operands passed as strided views (the kernel reads them through strides); bits identical | `test_fast_gemm_shim_passes_strided_views_without_copies` (data_ptr/stride pass-through + bitwise vs contiguous) |

**Bits.** Forward: unchanged (same kernel, same launch for K = 6656 / 19968). Backward at the
real shape: dx unchanged; dW changes only where the GEMM kernel fix (det_gemm:1) applies, i.e.
32 < T <= 512 tokens, and T % 256 != 0 is new (it used to raise). The explicit adapter's dW at
T % 256 != 0 now follows the same ragged rule (it used T/8-wide chunks or a single chunk).
With both fixes, the adapter and `fast_det_gemm` are bitwise twins for every K tested (36, 256,
300, 512, 4001, 6656; fp32 and bf16 out) and for the whole FFN forward + backward at T = 1, 64,
255, 300 (H=256, F=512) and the forward at the real shape.

**Performance (A100, base 6a61f6e -> 87cba55, median ms, default GEMM, real shape):**

| pass | T | before | after | note |
|---|---|---|---|---|
| forward | 1 / 16 / 64 | 0.668 / 0.605 / 0.751 | 0.685 / 0.634 / 0.769 | launch-bound, within node noise |
| forward | 256 / 1024 / 4096 | 1.82 / 6.93 / 26.42 | 1.84 / 6.97 / 26.46 | unchanged (123.5 TFLOP/s at 4096) |
| fwd+bwd | 64 / 300 / 4001 | ValueError | 5.04 / 11.76 / 78.14 | swiglu_ffn:1 |
| fwd+bwd | 256 | 12.20 | 7.67 | no transposed copies (0.064 -> 0.052 GiB peak) |
| fwd+bwd | 4096 | 87.93 (111 TFLOP/s) | 79.28 (124 TFLOP/s) | no transposed copies (1.02 -> 0.81 GiB peak) |

Branch `worktree-agent-ad5c4d3b6ebfb920f`, based on `feat/museglimmer-arch-fingerprint` (e2f4e64,
the POC scaffold with the contract/manifest; `main` is **not** an ancestor of that branch, so the
integrator should merge this branch into the scaffold branch, not into `main`).

Files:

| path | role |
|---|---|
| `rl_engine/models/muse_glimmer/triton/swiglu.py` | row-local SwiGLU fwd/bwd Triton kernels, pinned launch config, `kernel_trace()` |
| `rl_engine/models/muse_glimmer/ops/swiglu_ffn.py` | `swiglu`, `swiglu_fp32`, `swiglu_ffn`, `swiglu_ffn_fp32`, `trace()`, `TritonDetGemmAdapter`, `default_gemm()`, `SwiGLUFFNOp` |
| `tests/models/muse_glimmer/test_swiglu_ffn.py` | gates G1/G2/G4/G5, real shape, real layer-0 weights (20 tests) |
| `benchmarks/models/muse_glimmer/bench_swiglu_ffn.py` | strict vs torch native at T=4096, GEMM/elementwise split |

## Summary

- `swiglu(gate, up)`: one Triton program per logical row (contract R3), `h = (g / (1 + exp(-g))) * u`
  in fp32 (`tl.exp` fp32, IEEE `tl.div_rn`), one rounding at the store; deterministic row-local fp32
  VJP. **G1/G2/G4/G5 pass.** 0.364 ms at 4096×19968 (1.35 TB/s), 0.73× of torch `F.silu(g)*u`.
- `swiglu_ffn(x, w_gate, w_up, w_down, *, gemm=None)`: gate GEMM, up GEMM, swiglu, row-parallel down
  GEMM over a pluggable `gemm(a (M,K), w (N,K), *, out_dtype) -> a @ w.T`, with an explicit
  `autograd.Function` backward assembled from the same gemm + the row-local VJP. **All gates pass**,
  including batch invariance at the real shape (T=1 / 4 gathered / 16 vs T=4096, `torch.equal`),
  repeat determinism of forward+backward at H=6656, F=19968, T=256 (y, dx, dW×3 `torch.equal`),
  correctness vs fp64 at T=64 (max abs 1.1e-2, mean 1.7e-3 on |y|~0.5) and one forward with the real
  layer-0 MLP weights (shapes/dtypes confirmed; max abs 1.8e-2 vs fp64, *lower* than HF-style bf16
  cuBLAS at 2.4e-2).
- **[superseded: the default is now `fast_det_gemm`, no fallback; the adapter is explicit-only]**
  Default GEMM (`TritonDetGemmAdapter`): the repo's fixed-order kernel driven with bf16 operands,
  fp32 accumulation, 8 canonical K-chunks summed in the fixed tree, one rounding; 91–109 TFLOP/s on
  A100 — already above the contract's 80 TFLOP/s note. It auto-switches to the sibling's
  `rl_engine.models.muse_glimmer.ops.det_gemm.fast_det_gemm` when that module is importable (it did
  not exist in this worktree; the shim path is exercised only structurally).
- FFN forward at T=4096: **33.2 ms strict vs 12.75 ms cuBLAS (2.60×)**; the three GEMMs are 99.0% of
  the time, the SwiGLU kernel 1.0%. The fast GEMM is the whole story for this row.

## Design

Shapes: x (T, H=6656) bf16, W_gate/W_up (F=19968, H) bf16, W_down (H, F) bf16 → y (T, H) bf16.

**Forward and cast points (contract O6/R5):**
```
g = gemm(x, W_gate) → bf16      u = gemm(x, W_up) → bf16        (fp32 accumulate, one rounding each)
h = silu(g→fp32) * (u→fp32) → bf16                              (one rounding, in the kernel store)
y = gemm(h, W_down) → bf16      K=F in 8 chunks of 2496, fp32 partials ((0+1)+(2+3))+((4+5)+(6+7)),
                                rounded once (contract O3 row-parallel rule)
```
**Backward (contract O3/O6: det_gemm + row-local fp32 VJP, no cuBLAS):**
```
dy  → bf16 (declared input cast; autograd already delivers bf16)
dh  = gemm(dy, W_downᵀ) → bf16                                   K=H, chunk rule
dg  = (dh*u) * (s*(1+g*(1-s))),  du = dh * (g/(1+exp(-g)))      fp32 row-local, → bf16 (one rounding)
dx  = gemm(dg, W_gateᵀ, fp32) + gemm(du, W_upᵀ, fp32) → bf16     two fp32 partials, ONE rounding
dW_gate = gemm(dgᵀ, xᵀ), dW_up = gemm(duᵀ, xᵀ), dW_down = gemm(dyᵀ, hᵀ) → bf16   K=T, chunk rule
```
dW GEMMs are skipped when the weight does not require grad (`needs_input_grad`), so a LoRA run with
frozen base weights pays only dh/dg/du/dx.

**SwiGLU kernel tiles (R6):** grid `(T,)`, each program walks F in ascending `BLOCK_F=1024` tiles
(F=19968 = 19.5 tiles, last one masked), `num_warps=4`, `num_stages=3`; no autotune. Elementwise, so
there is no reduction order; batch invariance holds by construction and was verified bitwise.

**Default GEMM adapter tiles:** `_det_gemm_fp32_kernel` of `rl_engine/backends/shared/triton/gemm/
det_gemm.py` with `PROMOTE_INPUTS=False` (bf16 tensor-core `tl.dot`, fp32 accumulator), `BLOCK_M=64,
BLOCK_N=64, BLOCK_K=32, num_warps=4, num_stages=3`, one program per output tile walking its K chunk
ascending (no split-K, no atomics). K is split into 8 chunks of K/8 (6656 → 832, 19968 → 2496,
T → T/8) for **every** GEMM; the 8 fp32 partials are combined with in-place fp32 adds in the fixed
tree and rounded once. The weight is read in its native (N, K) layout through strides, so nothing is
transposed or cached (the brief's "transpose w to (K,N) once, cache it" would double the 60 GB of
weights across 52 layers), and the transposed views in the backward are free. Triton specialises on
M/N/K/stride divisibility only; bitwise equality across M = 1 / 77 / 4096 was verified.

**Why not `TritonDetGemmOp.__call__`:** measured at the gate shape (4096×6656×19968): bf16 leaf/node
tree GEMM 561 ms (1.9 TFLOP/s) and it rounds to bf16 at every 32-wide leaf and every tree node,
which contradicts R1's "one rounding"; fp32-promoted FMA path 78 ms (14 TFLOP/s); the chosen bf16-MMA
path 11.0 ms (99 TFLOP/s), repeat-deterministic, batch-invariant, fp64 max-abs error 7e-5 single pass
/ 5e-6 with the 8-chunk tree. cuBLAS bf16: 4.7 ms (232 TFLOP/s).

`trace()` returns `{op, contract, nodes, dtype_in, dtype_acc, dtype_out, reduction_order, backward,
tile_config={swiglu: kernel_trace(), gemm: gemm.trace()}}`; it is stable and JSON-serialisable (G5).

## Contract deviations

1. **[superseded]** ~~GEMM intra-chunk order is implementation-defined ... the adapter and the
   sibling's fast kernel will not be bit-identical.~~ Measured after the review fixes: both run the
   same ascending k16 MMA chain from +0 per chunk and the same fp32 tree, and are **bitwise equal**
   (`test_adapter_and_fast_gemm_are_bitwise_twins`, FFN fwd/bwd twin checks).
2. **The 8-chunk tree is applied to all GEMMs** (gate, up, down, and all five backward GEMMs), not
   only the row-parallel ones named in O3. The `gemm` callable has no row-parallel flag and the fast
   kernel chunks always; cost is nil (gate GEMM 11.49 → 11.55 ms) or negative (down 15.30 → 10.06 ms).
3. **[superseded]** ~~dW GEMMs chunk over K=T, so T must be a multiple of 8 ... the adapter falls
   back to a single chunk.~~ dW GEMMs use the declared ragged token rule (T_pad = roundup(T, 256),
   8 chunks of T_pad/8, rows >= T masked) on both GEMMs; any T >= 1; no silent single-chunk path.
4. **dx = fp32 sum of the two dX partials, one rounding.** The contract is silent; plain autograd
   would add two bf16 tensors. Flagged for the RFC text.
5. **Golden FFN uses fp64 `torch.matmul`** (not fixed-order); fp64 ordering noise (~1e-16) is five
   orders below the bf16 tolerances. A fixed-order fp32 torch GEMM without matmul is impractical at
   H=6656 — R4 should allow an fp64 cuBLAS golden for GEMM-containing ops.
6. `tl.exp` is Triton's fp32 exp (ex2.approx·log2e on NVIDIA), not libm; the golden uses `torch.exp`.
   fp32-output kernel vs golden passes 1e-5/1e-5; after bf16 rounding the difference is invisible.
   Division is IEEE (`tl.div_rn`). Permitted by R2 as written.
7. Convenience beyond the contract: `swiglu` accepts fp16/fp32 inputs (output dtype = input dtype)
   and leading dims are flattened; `swiglu_ffn` accepts `(…, H)`.

## Test commands and results

```
PYTHONPATH=$PWD ~/bin/gpu-lease -n 1 -- ~/envs/vllm/bin/python -m pytest \
    tests/models/muse_glimmer/test_swiglu_ffn.py -v -s -p no:cacheprovider
```
Verbatim tail (A100-SXM4-80GB, torch 2.13.0+cu130, triton 3.7.1, TF32 disabled):
```
tests/models/muse_glimmer/test_swiglu_ffn.py::test_swiglu_fp32_golden_formula PASSED
tests/models/muse_glimmer/test_swiglu_ffn.py::test_swiglu_backward_fp32_matches_autograd_of_golden PASSED
tests/models/muse_glimmer/test_swiglu_ffn.py::test_g5_trace_is_emitted_and_stable PASSED
tests/models/muse_glimmer/test_swiglu_ffn.py::test_ffn_shape_validation PASSED
tests/models/muse_glimmer/test_swiglu_ffn.py::test_g1_swiglu_forward_vs_golden[shape0] PASSED
tests/models/muse_glimmer/test_swiglu_ffn.py::test_g1_swiglu_forward_vs_golden[shape1] PASSED
tests/models/muse_glimmer/test_swiglu_ffn.py::test_g1_swiglu_forward_vs_golden[shape2] [G1 swiglu bf16 vs fp64, T=64 x F] max_abs=1.672e-01 mean_abs=2.691e-03 max_rel=3.890e-03
PASSED
tests/models/muse_glimmer/test_swiglu_ffn.py::test_g1_swiglu_saturation_corners PASSED
tests/models/muse_glimmer/test_swiglu_ffn.py::test_g2_swiglu_batch_invariance_real_width PASSED
tests/models/muse_glimmer/test_swiglu_ffn.py::test_g2_ffn_batch_invariance_real_shape PASSED
tests/models/muse_glimmer/test_swiglu_ffn.py::test_adapter_gemm_accuracy_invariance_and_strides PASSED
tests/models/muse_glimmer/test_swiglu_ffn.py::test_ffn_forward_small_vs_golden[shape0] PASSED
tests/models/muse_glimmer/test_swiglu_ffn.py::test_ffn_forward_small_vs_golden[shape1] PASSED
tests/models/muse_glimmer/test_swiglu_ffn.py::test_ffn_forward_small_vs_golden[shape2] PASSED
tests/models/muse_glimmer/test_swiglu_ffn.py::test_ffn_forward_real_shape_vs_fp64 [FFN real shape T=64 bf16 vs fp64] max_abs=1.101e-02 mean_abs=1.653e-03 max_rel=5.705e+00
PASSED
tests/models/muse_glimmer/test_swiglu_ffn.py::test_g4_swiglu_backward_repeat_and_reference [G4 swiglu dgate bf16 vs fp32 autograd] max_abs=3.098e-02 mean_abs=4.852e-04 max_rel=3.890e-03
PASSED
tests/models/muse_glimmer/test_swiglu_ffn.py::test_g4_ffn_backward_small_repeat_and_reference[shape0] PASSED
tests/models/muse_glimmer/test_swiglu_ffn.py::test_g4_ffn_backward_small_repeat_and_reference[shape1] PASSED
tests/models/muse_glimmer/test_swiglu_ffn.py::test_g4_ffn_forward_backward_real_shape_repeat_determinism PASSED
tests/models/muse_glimmer/test_swiglu_ffn.py::test_real_layer0_mlp_weights_forward [real layer-0 MLP weights, T=8, bf16 vs fp64] max_abs=1.796e-02 mean_abs=3.449e-04 max_rel=1.261e+00
[real layer-0 MLP weights, HF-style bf16 cuBLAS vs fp64 (reference)] max_abs=2.385e-02 mean_abs=4.859e-04 max_rel=1.759e+00
PASSED

============================= 20 passed in 16.25s ==============================
```
(`max_rel` is relative to |ref| clamped at 1e-3, so it is dominated by near-zero outputs; the
meaningful metrics are max_abs/mean_abs. The swiglu max_abs 0.167 is one bf16 ulp at |h|≈75 with
the test's g scale of 3.)

Tolerances used (from `ws1.json`, unchanged): swiglu forward/grad — `elementwise/bfloat16`
(atol 2e-2, rtol 1.6e-2) and `elementwise/float32` (1e-5/1e-5) for the fp32-output kernel variant;
FFN forward — `forward_accuracy/reduction/bfloat16` (5e-2/2e-2); FFN grads —
`gradient_accuracy/reduction/bfloat16` (0.1/2e-2) vs an fp32 `torch.matmul` autograd reference
(TF32 off) at (T,H,F) = (32,64,96) and (24,40,56).

Gate status: **G1 pass, G2 pass, G3 n/a (stateless op; batch invariance covers it), G4 pass, G5 pass.**

## Parity results

| axis | what | result |
|---|---|---|
| batch (swiglu fwd) | rows 0/1/2047/4095 at T=1 vs T=4096; 4 rows gathered from positions {4095,7,2047,1}; windows T=16, T=2 | `torch.equal` ✓ |
| batch (swiglu bwd) | same rows, `launch_swiglu_bwd` T=1 vs T=4096 | `torch.equal` ✓ |
| batch (FFN fwd, real shape) | rows 0/1/2048/4095 at T=1 vs T=4096; 4 rows gathered at other positions; 16-row window at offset 37 | `torch.equal` ✓ |
| batch (adapter GEMM) | M=1 and M=77 slices vs M=100; strided (transposed-view) operands vs contiguous | `torch.equal` ✓ |
| repeat (swiglu bwd) | two autograd runs at T=64×F | `torch.equal` ✓ |
| repeat (FFN fwd+bwd) | small shapes and real shape T=256: y, dx, dW_gate, dW_up, dW_down | `torch.equal` ✓ |
| frozen weights | dx with `requires_grad=False` weights equals dx of the full backward | `torch.equal` ✓ |
| sequence split | not applicable (no state across tokens) | — |
| TP | not run (1 GPU). The tree partials are exposed (`TritonDetGemmAdapter._partial`), so TP-t ranks can reduce their 8/t chunks in tree order as O3 prescribes | open |

## Benchmark

`PYTHONPATH=$PWD ~/bin/gpu-lease -n 1 -- ~/envs/vllm/bin/python benchmarks/models/muse_glimmer/bench_swiglu_ffn.py --tokens 4096`
(A100-SXM4-80GB, median of 5, CUDA events, TF32 off, gemm = `TritonDetGemmAdapter`):

| case | strict ms | native ms | strict / native |
|---|---:|---:|---:|
| swiglu (4096 × 19968) fwd | 0.364 (1350 GB/s) | 0.497 (`F.silu(g)*u`) | 0.73× |
| swiglu_ffn fwd (T=4096) | 33.157 (99 TFLOP/s) | 12.751 (cuBLAS bf16, 256 TFLOP/s) | 2.60× |
| GEMM share of strict FFN fwd | 33.49 (gate 11.92 + up 11.55 + down 10.02) | – | 99.0% |
| swiglu share of strict FFN fwd | 0.353 | – | 1.0% |
| adapter gate GEMM: single pass vs 8-chunk tree | 11.49 vs 11.55 | – | tree overhead ≈ 0 |
| adapter down GEMM: single pass vs 8-chunk tree | 15.30 vs 10.06 | – | chunking is faster |

What the fast GEMM buys: at best 33.2 → ~12.8 ms per FFN forward (×52 layers, ×3 for fwd+bwd); the
elementwise kernel is already below torch and is not worth further work. Obvious next optimisation
(not done): larger tiles / more stages (128×128 or 128×256, 3–4 stages) in the GEMM — the fp32
partial traffic of the out-of-kernel tree costs nothing measurable, so the in-register tree of the
contract's performance note is not where the 2.6× is.

## Integration requests

No edits to shared files are needed. For the reference-model agent:
```python
from rl_engine.models.muse_glimmer.ops.swiglu_ffn import swiglu, swiglu_fp32, swiglu_ffn, trace
# optional: swiglu_ffn_fp32 (fp64 golden), swiglu_backward_fp32, TritonDetGemmAdapter, default_gemm
```
Node mapping for `operator_trace` (suffixes of layer nodes covered by this row):
`mlp_gate_proj`, `up_proj` → the two gemm calls, `swiglu` → `swiglu`, `down_proj` → the row-parallel
gemm call; all four are one `swiglu_ffn(...)` call per layer.

**[superseded]** `default_gemm()` is `fast_det_gemm` (no fallback) and `_FastDetGemm` passes the
strided views as they are (the `.contiguous()` calls are gone). A pluggable `gemm` must accept
`ragged_k=` (the dW calls pass `ragged_k=True`); raw `fast_det_gemm` (what the strict reference
model passes) does.

Marker used: `pytest.mark.ws1` (already declared in `pyproject.toml`).

## Open issues

- `_det_gemm_fp32_kernel` indexes with int32: the adapter requires M·N and M·K < 2^31 per GEMM
  (T ≲ 100k tokens at F=19968). Fine for the POC; the fast GEMM should use int64 offsets.
- Adapter peak memory: ≤ 4 live fp32 partials of (M, N) — 1.3 GB at T=4096 for the gate GEMM, 2.1 GB
  for a dW GEMM; the fast kernel removes this.
- Triton compiles one variant per divisibility class of (M, N, K, strides): ~1 s each on first use
  (a dozen in the test run); the shared `~/.triton` cache absorbs it afterwards.
- **[resolved]** Validated against the sibling's `fast_det_gemm`: bitwise twins (see follow-up).
- **[resolved]** The backward now calls `record_backward("swiglu_ffn", ...)`.

## RFC feedback (what the `swiglu_ffn` row text should say)

1. Spell out the GEMM arithmetic class once ("bf16 operands, fp32 accumulation, one rounding at the
   declared cast; the K-chunk tree is pinned, the intra-chunk order is pinned per kernel in
   `trace()`") and say the 8-chunk tree applies to **all** GEMMs of the FFN, forward and backward,
   including the K=T weight-gradient GEMMs. **[updated]** Do not require padded micro-batches:
   declare the ragged token rule (T_pad = roundup(T, 256), 8 chunks, masked rows) as the dW chunk
   rule of O3 so every T works and both runtimes agree on it.
2. State the dx rule: the two input-gradient partials (through W_gate and W_up) are summed in fp32
   and rounded once.
3. Give the VJP formula explicitly — `dg = (dh*u) * (s*(1+g*(1-s)))`, `du = dh * silu(g)` with
   `silu(g) = g/(1+exp(-g))` evaluated exactly as in the forward — because `(dh*u)*dsilu` and
   `dh*(u*dsilu)` differ in bits.
4. Drop the implicit expectation that the elementwise kernel matters for speed: it is 1% of the row
   and already faster than torch. Rephrase the row's performance target as a GEMM target (the
   contract's "≥ 80 TFLOP/s" is met by the repo kernel with bf16 operands at 91–109 TFLOP/s; cuBLAS
   is 256 TFLOP/s) and note that the in-register tree saves nothing measurable — tile/pipeline tuning
   is the lever.
5. R4 "no matmul shortcuts" should allow an fp64 cuBLAS golden for ops that contain a GEMM.
6. Record the measured accuracy budget for the row: bf16 strict vs fp64 at T=64 random weights
   max_abs 1.1e-2 / mean 1.7e-3; with real layer-0 weights max_abs 1.8e-2 / mean 3.4e-4 — tighter
   than HF-style bf16 cuBLAS (2.4e-2 / 4.9e-4), so no tolerance row beyond `reduction/bfloat16` is
   needed.
7. The performance note's "TritonDetGemmOp runs ~43 ms = 5 TFLOP/s" is optimistic for the FFN shapes:
   the bf16 tree path measured 561 ms (1.9 TFLOP/s) at 4096×6656×19968 and it rounds at every node;
   the usable fallback is its fp32-output kernel with bf16 operands, as done here.

## Time spent

About 2 hours: 25 min reading contract/prior art and probing the three GEMM paths, 60 min
implementation, 25 min tests/benchmark/debug, 15 min report.
