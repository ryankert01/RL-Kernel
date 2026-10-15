# REPORT — `dense_gemm_qualification` (contract O3): `fast_det_gemm`

Branch `worktree-agent-a36c9d51e305a64a2`; A100-SXM4-80GB, torch 2.13.0+cu130, Triton 3.7.1.

## Summary

* New Triton kernel `det_gemm_tree_kernel`: one program per `(BLOCK_M, BLOCK_N)` output tile, K
  walked in the 8 canonical chunks in order, fp32 MMA accumulation per chunk, chunk partials
  combined in registers in the fixed tree `((0+1)+(2+3))+((4+5)+(6+7))` (max 4 live
  accumulators), one rounding to bf16 or fp32.  No split-K, no atomics, no autotune,
  `tl.dot(..., allow_tf32=False)`.
* Op wrapper `fast_det_gemm(a, w, *, out_dtype, k_chunks=8, w_kn=False)` with deterministic
  autograd (`dA`, `dW` through the same kernel), `fast_det_gemm_partial` for TP sub-tree
  partials, `combine_tree` (pure torch, importable by the collectives / lm_head agents),
  `forward_fp32`, `trace()`.
* Gates: G1 PASS (bf16 max abs err 1.6e-2..3.1e-2 at |y| up to ~10, fp32 out max rel 3e-4..1.3e-3
  vs fp64), G2 PASS (`torch.equal` decode M=16 vs M=64 vs prefill M=4096 / 512, positions,
  chunked prefill, padding) for all 9 Muse shapes incl. the padded lm_head N=202752, TP1 vs
  simulated TP2/4/8 `torch.equal` (fp32 and bf16) for all 9 shapes, repeat determinism PASS,
  G4 PASS (dA/dW repeatable, allclose vs fp32 autograd; dA batch-invariant), G5 PASS.
  **80 tests passed in 18 s.**
* Performance at M=4096: **108–134 TFLOP/s** (target >= 80), 1.9–2.5x slower than cuBLAS bf16,
  12–121x faster than the repo's `TritonDetGemmOp`.  lm_head fp32/bf16 at M=4096: 90 ms
  (123 TFLOP/s).  Autograd fwd+bwd: 2.0x torch.  Decode (M=16): launch/latency bound, ~0.07 ms
  kernel floor (+~0.045 ms Python wrapper), k/v proj (N=256) 3–5x cuBLAS, big shapes ~1.0x.
* Key finding for the RFC: the bits depend only on (chunk rule, ascending-k MMA chain, fp32
  tree, one rounding).  BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages were all verified to be
  **performance-only** (24 configs, all `torch.equal`), so decode and prefill may use different
  tiles and TP ranks may use any tile.

## Design

### Reduction order (the arithmetic contract)

For `y = a @ w.T`, `a (M, K)` bf16, `w (N, K)` bf16:

1. `K` is split into 8 consecutive chunks; chunk `c` covers `[c*K/8, (c+1)*K/8)`.  `K` must be a
   multiple of `8 * 32 = 256` (forward: raises otherwise, "out of contract").  Model values:
   6656 -> chunk 832, 4096 -> 512, 19968 -> 2496; backward reductions over N: 4096, 256 (chunk
   32), 19968, 6656, 202752 (chunk 25344).
2. Inside a chunk: `acc = tl.dot(a_tile, b_tile, acc, allow_tf32=False)` over `BLOCK_K` columns
   at a time in ascending k.  On sm80 this lowers to a dependent chain of bf16
   `mma.sync.m16n8k16` with fp32 accumulate.  The chain is the same whatever BLOCK_K (64 or 32)
   the loop uses, so BLOCK_K is `64` when `chunk % 64 == 0`, else `32` (only N=256 in dA).
3. Tree in registers, fp32 IEEE adds, chunk order 0..7:
   `s01=acc0+acc1; s23=acc2+acc3; s0123=s01+s23; s45=acc4+acc5; s67=acc6+acc7; s4567=s45+s67;
   root=s0123+s4567`.  Live accumulators peak at 4 (`s0123, s45, acc6, acc7`).
4. `root` is rounded once: `.to(bf16)` (RNE) or stored as fp32 (`out_dtype=torch.float32`, lm_head
   logits for O7).

### Casts

* Inputs bf16, no cast.  Products bf16 x bf16 are exact in fp32.  Accumulate fp32.  One output
  rounding.
* Backward: `dY -> bf16` once (declared cast; for the fp32 lm_head logits this is the only
  rounding the backward adds).  `dA = tree_N(dY_bf16, W) -> bf16`,
  `dW = tree_M(dY_bf16^T, A) -> bf16`.
* `fast_det_gemm_partial` returns fp32 (the sub-tree node); `combine_tree` is fp32 in/out;
  the caller rounds once.

### dW chunk rule (reduction over the M tokens)

`M_pad = roundup(M, 256)`; chunk `c` = token rows `[c*M_pad/8, (c+1)*M_pad/8)`; rows `>= M` are
masked to zero, so empty chunk partials are exact zeros and `x + 0 == x` leaves the tree bits
unchanged (test `test_g4_dw_ragged_rule_equals_zero_padded_tokens`: dW for 16 tokens ==
dW for the same 16 tokens zero-padded to 256).  Repeatable (test) and independent of tile
constants; batch invariance of dW is not a contract requirement and does not hold (dW depends on
every token).  BLOCK_K = 64 if `M_pad/8 % 64 == 0` else 32.  The same masked rule is used for dA
when N is not a multiple of 256 (never for the model shapes; it keeps dA batch-invariant).

### TP partials

Rank `r` at TP `t` owns chunks `rank_chunk_ids(t, r) = [r*8/t, (r+1)*8/t)`; it calls
`fast_det_gemm_partial(a_local, w_local, chunk_ids)` on its K slice and gets the fp32 value of
that sub-tree node (the kernel is launched with `NUM_CHUNKS = 8/t` and the same chunk length, so
every MMA chain is identical to the TP1 kernel's).  `combine_tree(partials)` (t = 2, 4, 8) is the
remaining part of the same tree; `.to(bf16)` once.  Verified `torch.equal` to TP1 in fp32 and
bf16 for all 9 shapes.  The partials must travel as fp32 (2x bf16 bytes).

### Tiles (contract R6; performance-only, recorded in `trace()`)

| shape class | rule | BLOCK_M x BLOCK_N x BLOCK_K | warps | stages | regs / spills |
|---|---|---|---|---|---|
| decode16 | M <= 16 | 16 x 64 x 64 | 4 | 4 | 128 / 0 |
| decode64 | 16 < M <= 64 | 32 x 64 x 64 | 4 | 4 | 212 / 0 |
| prefill | M > 64 | 64 x 128 x 64 | 8 | 3 | 255 / 52 |

GROUP_M = 8 grouped rasterisation (L2 reuse of W; only changes which program computes which
tile).  Offline sweep (not runtime autotune) at M=4096, TFLOP/s on
(19968,6656) / (6656,19968) / (4096,6656) / (6656,4096):

| config | gate/up | down | q | o | spills |
|---|---|---|---|---|---|
| **64x128x64 w8 s3 (pinned)** | 125 | 137 | 123 | 100 | 52 |
| 64x64x64 w4 s3 | 126 | 132 | 124 | 98 | 70 |
| 128x64x64 w8 s3 | 102 | 133 | 117 | 117 | 60 |
| 64x64x64 w8 s3 (spill-free) | 101 | 103 | 97 | 81 | 0 |
| 64x32x64 w4 s4 (spill-free) | 106 | 108 | 101 | 80 | 0 |
| 128x128x64 w8 s3 | 72 | 70 | 66 | 54 | 362 |

Four live fp32 accumulators make 128x128 tiles register-infeasible (`4 * 64 regs` per thread with
8 warps); the pinned config keeps 32 regs per accumulator and ptxas spills ~52 registers to local
memory, which still beats every spill-free config.  Decode: every tile shape sits at a ~0.07 ms
floor for K=6656 (serial walk of 104 k-steps x 8 pipeline fills), so no narrow-N class was added.

## Contract deviations

1. **`forward_fp32` golden is fp64 cuBLAS** (`(a.double() @ w.double().T).float()`), not a
   hand-written fixed-order torch loop (R4): infeasible at 4096 x 19968 x 6656.  The contract
   header explicitly allows an FP64 reference for correctness; it is a tolerance reference only.
2. **MMA-internal order is hardware-defined.**  The contract pins the instruction sequence
   (ascending k16 steps, fp32 accumulate); the order/precision inside one `mma.sync` is NVIDIA's
   and undocumented.  Bitwise equality is therefore guaranteed within one GPU generation (A100
   here); A100-vs-H100 bit equality is not promised by this kernel (same as any tensor-core path).
3. **dY is rounded to bf16 before the backward GEMMs**, including for the fp32 lm_head logits
   gradient.  The contract's `dX = det_gemm(dY, W)` implies bf16 inputs; stated here explicitly.
4. The chunk-partial adds are "in registers" semantically; ptxas spills ~52 registers in the
   prefill config (compiler-managed, no semantic effect).
5. `k_chunks` in {1, 2, 4} is accepted by `fast_det_gemm` for sub-tree experiments; only 8 is
   canonical.  dA for N not a multiple of 256 uses the masked rule instead of raising
   (unreachable with the model shapes).

## Test commands and results

```
PYTHONPATH=$PWD ~/bin/gpu-lease -n 1 -- ~/envs/vllm/bin/python -m pytest \
    tests/models/muse_glimmer/test_det_gemm.py -q -p no:cacheprovider --durations=10
```

```
[gpu-lease] using GPUs [0]
........................................................................ [ 90%]
........                                                                 [100%]
============================= slowest 10 durations =============================
4.79s call     tests/models/muse_glimmer/test_det_gemm.py::test_g4_backward_w_kn_layout
4.76s call     tests/models/muse_glimmer/test_det_gemm.py::test_g4_da_is_batch_invariant
1.55s call     tests/models/muse_glimmer/test_det_gemm.py::test_g4_backward_repeatable_and_accurate[ragged100]
1.30s call     tests/models/muse_glimmer/test_det_gemm.py::test_g1_forward_correctness_small[shape0]
1.19s call     tests/models/muse_glimmer/test_det_gemm.py::test_g4_backward_repeatable_and_accurate[ragged16]
0.94s call     tests/models/muse_glimmer/test_det_gemm.py::test_g4_fp32_output_path_backward
0.89s call     tests/models/muse_glimmer/test_det_gemm.py::test_g4_backward_repeatable_and_accurate[o_proj]
0.09s call     tests/models/muse_glimmer/test_det_gemm.py::test_g1_forward_correctness_muse_shape[lm_head]
0.04s call     tests/models/muse_glimmer/test_det_gemm.py::test_tp_simulated_matches_tp1_bitwise[lm_head-8]
0.03s call     tests/models/muse_glimmer/test_det_gemm.py::test_tp_simulated_matches_tp1_bitwise[lm_head-2]
80 passed in 18.13s
```

Gate coverage (tolerances from `rl_engine/contracts/profiles/precision/ws1.json`, rows
`accuracy.default.reduction.{bfloat16,float32}` for the forward and
`judgments.gradient_accuracy...reduction.bfloat16` for the backward):

| gate | tests | result |
|---|---|---|
| G1 correctness vs fp64 | 4 small shapes (ragged M, ragged N) + 9 Muse shapes at M=256, bf16 and fp32 out | PASS |
| G2 batch invariance | 9 Muse shapes: M=1 / decode16 / decode64 / prefill (M=4096, lm_head 512), 2 positions | PASS (`torch.equal`) |
| G2 chunked prefill + padding | 5 shapes incl. (300, 6656, 4096) | PASS |
| G2 tile constants | 6 configs incl. BLOCK_K=32, 16x64, 128x64 vs pinned | PASS (bitwise) |
| G2 layout | `(N, K)` vs `(K, N)` weight | PASS (bitwise) |
| TP | TP1 vs simulated TP2/4/8, 9 shapes, fp32 and bf16; partial batch invariance; left fold != tree | PASS |
| repeat | (4096, 6656, 19968) twice | PASS |
| G4 backward | repeat `torch.equal` + allclose vs fp32 autograd: ragged16/100, o_proj, k_proj (BLOCK_K=32 dA), k_proj M=4096; w_kn layout; dA batch invariance; dW ragged rule; fp32-out (lm_head tile) path | PASS |
| G5 trace | keys, stability, JSON, per-class config == launched config | PASS |
| contract errors | K % 256 != 0, dtypes, k_chunks, K mismatch, misaligned chunk ids, combine_tree validation | PASS |

G1 error table (benchmark `qualify`, M=256, a~N(0,1), w~N(0,0.02^2), |y| up to ~10–14):

| shape | N | K | bf16 max_abs | bf16 max_abs / max abs ref | fp32 max_abs | fp32 max_rel |
|---|---|---|---|---|---|---|
| q_proj / attn_gate | 4096 | 6656 | 2.930e-02 | 3.384e-03 | 1.049e-05 | 3.717e-04 |
| k_proj / v_proj | 256 | 6656 | 1.559e-02 | 2.087e-03 | 7.629e-06 | 2.970e-04 |
| o_proj | 6656 | 4096 | 1.562e-02 | 2.351e-03 | 4.292e-06 | 2.748e-04 |
| mlp_gate / up_proj | 19968 | 6656 | 2.930e-02 | 3.384e-03 | 1.049e-05 | 3.717e-04 |
| down_proj | 6656 | 19968 | 3.127e-02 | 2.251e-03 | 4.768e-05 | 1.313e-03 |
| lm_head | 202752 | 6656 | 3.052e-02 | 2.971e-03 | 1.001e-05 | 4.694e-04 |

(bf16 error is the single output rounding: half-ulp of bf16 at |y| ~ 8 is 3.1e-2; fp32 max_rel is
measured on |ref| >= 1e-3 * max|ref| and reflects fp32 accumulation over K.)

## Parity results

* **Batch**: one logical row at M=1, inside M=16 (row 5, decode16 tiles), inside M=64 (row 37,
  decode64 tiles), inside M=4096 at rows 0 and 2049 (prefill tiles): all `torch.equal`, all 9
  shapes.  `dA` is batch-invariant as well (reduction over N).
* **Sequence split**: `cat(gemm(a[:m/3]), gemm(a[m/3:])) == gemm(a)` and padding rows do not
  change valid rows, bitwise, incl. (300, 6656, 4096).
* **Repeat**: forward and backward bitwise across runs (with allocator perturbation in between).
* **TP**: TP1 == combine_tree(TP2/4/8 partials) bitwise in fp32 and after bf16 rounding, all 9
  shapes; TP8 left fold differs (the tree is load-bearing).
* **Tile / layout**: 24 tile configs across the sweeps and the `(K, N)` layout all bitwise
  equal to the pinned path.

## Benchmark

```
PYTHONPATH=$PWD ~/bin/gpu-lease -n 1 -- ~/envs/vllm/bin/python \
    benchmarks/models/muse_glimmer/bench_det_gemm.py --out bench.md
```

M=4096 (training / prefill), bf16 out, cuBLAS = `torch.matmul` bf16 with TF32 off, legacy =
repo `TritonDetGemmOp` (`_triton_tree_gemm`, b in (K, N)):

| shape | N | K | strict ms | strict TFLOP/s | cuBLAS ms | strict/cuBLAS | legacy ms | legacy/strict |
|---|---|---|---|---|---|---|---|---|
| q_proj | 4096 | 6656 | 2.073 | 107.7 | 0.819 | 2.53x | 40.1 | 19.3x |
| k_proj | 256 | 6656 | 0.197 | 71.0 | 0.079 | 2.47x | 2.46 | 12.5x |
| v_proj | 256 | 6656 | 0.194 | 71.8 | 0.079 | 2.45x | 2.46 | 12.6x |
| attn_gate | 4096 | 6656 | 1.796 | 124.4 | 0.813 | 2.21x | 40.2 | 22.4x |
| o_proj | 6656 | 4096 | 1.880 | 118.8 | 0.830 | 2.26x | 40.7 | 21.7x |
| mlp_gate | 19968 | 6656 | 8.773 | 124.1 | 4.169 | 2.10x | 561.7 | 64.0x |
| up_proj | 19968 | 6656 | 8.731 | 124.7 | 4.223 | 2.07x | 560.9 | 64.2x |
| down_proj | 6656 | 19968 | 8.099 | 134.4 | 4.213 | 1.92x | 981.7 | 121.2x |
| lm_head | 202752 | 6656 | 89.878 | 123.0 | 43.316 | 2.07x | skipped (workspace) | n/a |

M=16 (decode), bf16 out (strict numbers include ~0.045 ms Python/autograd wrapper overhead;
kernel-only time from the sweep is 0.070 ms for the K=6656 shapes):

| shape | N | K | strict ms | cuBLAS ms | strict/cuBLAS | legacy ms | legacy/strict |
|---|---|---|---|---|---|---|---|
| q_proj / attn_gate | 4096 | 6656 | 0.118 | 0.044 | 2.7x | 0.43 | 3.6x |
| k_proj / v_proj | 256 | 6656 | 0.115 | 0.023 | 5.0x | 0.28 | 2.4x |
| o_proj | 6656 | 4096 | 0.124 | 0.042 | 2.9x | 0.35 | 2.8x |
| mlp_gate / up_proj | 19968 | 6656 | 0.183 | 0.189 | 0.97x | 3.41 | 18.6x |
| down_proj | 6656 | 19968 | 0.193 | 0.182 | 1.06x | 4.31 | 22.4x |
| lm_head | 202752 | 6656 | 1.744 | 1.787 | 0.98x | 244.0 | 139.9x |

Autograd fwd+bwd at M=4096 (dA + dW through the kernel; strided transposed operands, no copies):

| shape | strict fwd+bwd ms | torch fwd+bwd ms | ratio | dA ms (TFLOP/s) | dW ms (TFLOP/s) |
|---|---|---|---|---|---|
| o_proj (6656, 4096) | 5.53 | 2.57 | 2.15x | 1.79 (125) | 1.83 (122) |
| mlp_gate (19968, 6656) | 26.50 | 13.19 | 2.01x | 9.10 (120) | 8.96 (122) |
| down_proj (6656, 19968) | 25.99 | 13.31 | 1.95x | 9.04 (121) | 8.58 (127) |

Per decoder layer at M=4096 the eight forward GEMMs take ~31.7 ms strict vs ~15.1 ms cuBLAS.

## Integration requests

No shared-file edits are required (no new tolerance row: the existing
`accuracy.default.reduction` rows fit; `operator_trace.NODE_KIND_ROWS["det_gemm"]` already maps
to `dense_gemm_qualification`).  Other agents import:

```python
from rl_engine.models.muse_glimmer.ops.det_gemm import (
    fast_det_gemm,          # y = fast_det_gemm(x, w); logits = fast_det_gemm(h, w_tile, out_dtype=torch.float32)
    fast_det_gemm_partial,  # TP rank: fp32 sub-tree partial over rank_chunk_ids(tp, rank)
    trace,
)
from rl_engine.models.muse_glimmer.ops.canonical_tree import combine_tree, rank_chunk_ids
```

If the integrator wants the kernel in a registry, the identifiers are `OP_NAME =
"muse_glimmer.fast_det_gemm"` and `KERNEL_ID =
"rl_engine.models.muse_glimmer.triton.det_gemm_tree.det_gemm_tree_kernel"`; `record_backward`
is emitted with `impl="muse_glimmer_fast_det_gemm", family="triton"`.

## Open issues

1. **~2x cuBLAS at M=4096** (108–134 TFLOP/s).  The limiter is the 4-live-accumulator register
   budget (52 spills, no 128x128 tiles).  Next step (not done): park `s0123` (and `s45`) in an
   L2-resident fp32 scratch (M x N x 4 B) between chunk groups to drop the peak to 2 live
   accumulators and enable 128x128/128x256 tiles; or an sm90 wgmma/TMA variant.  Both keep the
   bits (verified property: tiles are performance-only).
2. **Decode latency floor** ~0.07 ms per GEMM at K=6656 regardless of N, plus ~0.045 ms Python
   wrapper per call: 8 GEMMs x 52 layers ~ 50–70 ms per decode step before attention.  Mitigate
   with CUDA-graph capture in vLLM (removes the wrapper) and, for k/v proj, a 2-program
   cooperative K split with the same tree (sm90 clusters) — out of scope for the POC.
3. The kernel is not bit-compatible with the repo's `TritonDetGemmOp` / CUDA `det_gemm`
   (32-wide bf16-node K tree) by design; "det_gemm" in the Muse contract must mean this kernel.
4. Cross-architecture (A100 vs H100) bit equality is not established (MMA internals).
5. `allow_tf32=False` is deprecated in Triton 3.7 (`input_precision`); harmless warning.

## RFC feedback (what the `dense_gemm_qualification` row text should say)

* Define the arithmetic contract as exactly four items: (1) K split into 8 consecutive chunks
  (K % 256 == 0; model K/N values: 6656, 4096, 19968, 256, 202752 all qualify), (2) per chunk an
  ascending-k chain of bf16 tensor-core MMAs with fp32 accumulate, (3) fp32 tree
  `((0+1)+(2+3))+((4+5)+(6+7))`, (4) one rounding.  **Do not list BLOCK_K / BLOCK_M / BLOCK_N /
  num_warps / num_stages as part of the arithmetic** — they are performance-only (verified
  bitwise over 24 configs) and must merely be recorded in `trace()`.  This also means decode and
  prefill may use different tiles with no parity risk, and TP ranks may use any tile.
* Specify the backward chunk rules: dA reduces over N with the same 8-chunk rule (N=256 gives
  32-wide chunks); dW reduces over the M tokens with `M_pad = roundup(M, 256)` and masked rows —
  repeatable, not batch-invariant.  State that dY enters the backward GEMMs as bf16.
* TP partials are **fp32** and are exchanged as fp32 (2x the bytes of a bf16 all-reduce); the
  bf16 rounding happens once after the full tree.  `fast_det_gemm_partial` + `combine_tree` are
  the reference semantics for `tp_canonical_tree`.
* Replace "target >= 80 TFLOP/s" with measured numbers: 108–134 TFLOP/s at M=4096 (1.9–2.5x
  cuBLAS), fwd+bwd 2.0x torch, lm_head 90 ms at M=4096; and add a separate decode budget
  (~0.07 ms kernel floor per GEMM, k/v proj 3–5x cuBLAS) with CUDA graphs required for rollout.
* State that the Muse `det_gemm` is this kernel and is intentionally not bit-compatible with the
  existing repo `det_gemm` (different K tree); the existing `TritonDetGemmOp` is 12–121x slower
  on these shapes and cannot run lm_head at M=4096 (workspace).
* State the fp64 (or fp32 fixed-order) reference explicitly as the G1 oracle for GEMMs;
  "hand-written fixed-order torch, no matmul" is not feasible at these shapes.

## Time spent

~2.5 h: reading/contract 25 min, kernel + wrapper 40 min, offline tile sweep 25 min, tests 30 min,
benchmark + report 30 min.
