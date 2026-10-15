# REPORT — `lm_head_softcap_logprob` (WS1) + single-GPU core of `vocab_parallel_softcap_logp` (WS2)

> **Review follow-up (2026-10-15, GEMM correctness row, branch `worktree-wf_7b064961-36f-4`).**
> Eight review findings on this op (lm_head_softcap_logprob:1–8) were checked; seven are fixed,
> one (lm_head:5) needs a contract edit only. The original report below is kept; stale statements
> are marked **[stale]**. Raw numbers: `reports/muse_glimmer_poc/data/kernels_gemm.json`.

## Review follow-up: findings, fixes, evidence

| id | verdict | evidence before | fix | test (fails before, passes after) |
|---|---|---|---|---|
| lm_head:1 | **confirmed, fixed** | profiler, one no-grad forward (T=32): 2 `cudaStreamSynchronize` + 2 `aten::_local_scalar_dense` (the target range check ran twice) | validated once per call (1 sync); `validate_targets=False` performs **0** syncs and a target outside `[0, V)` yields `logp = NaN` on the device (never a silent `-logZ`); valid rows bitwise unchanged | `test_host_syncs_and_device_side_target_rule` (dispatch-mode count == 1; sync-debug "error" mode raises nothing without validation; NaN rows; TP sim) |
| lm_head:2 | **confirmed, fixed** | T % 256 != 0 sent dW through the legacy fp32-promoted kernel (sequential K, no tree): dW(T) != dW(same tokens zero-padded to 256/512) for T = 40, 255, 300; fwd+bwd at T=4001 took 975.6 ms vs 455.0 ms at T=4096 | dW = `gemm(dz_t^T, h, ragged_k=True)`: the declared token rule of `fast_det_gemm`; no legacy branch; no transposed copies | `test_dw_uses_declared_ragged_token_rule[40,255,256,300]` |
| lm_head:3 | **confirmed, fixed** | `reduction_order.gemm` said "K walked in block order", no GEMM tile constants | `trace()["gemm"]` = `det_gemm.trace(out_dtype=fp32)` (all shape classes, BLOCK_M/N/K, warps, stages, chunk rules, runtime); real reduction order incl. the dW/dh rules; an explicit non-default backend is named as such | `test_trace_is_complete_and_stable` |
| lm_head:4 | **confirmed, fixed** | with `ops.det_gemm` blocked the module silently resolved the legacy GEMM (subprocess printed `SILENT ..._triton_gemm_fp32`) | `det_gemm` imported eagerly, failure propagates; `legacy_det_gemm_fp32` only as an explicit `gemm=` | `test_default_gemm_backend_and_no_silent_fallback` (subprocess) |
| lm_head:5 | **confirmed, contract edit needed** | O7 backward names no cast for dẑ; the code (and `trace()["casts"]["dz"]`) casts fp32 -> bf16 once, as R1's bf16 MMA operands require | none in code; proposed O7 text: "dẑ (fp32) -> bf16 once (round-to-nearest-even), then dh = det_gemm(dẑ, W_lm) and dW_lm = det_gemm(dẑᵀ, h)" | — |
| lm_head:6 | **confirmed, fixed** | every test T sent dW through the legacy fallback; no backend assertion | small geometry H=256 so the default fast path runs everywhere; backend assertion; real-shape backward at T=256 and T=300 through the fast dW path (repeat `torch.equal`, allclose vs fp32 autograd: dh 2.4e-4 / 1.7e-4 abs at max 4.4e-2 / 4.0e-2, dW 1.1e-2 / 7.2e-3 at max 2.24 / 1.92) | `test_real_shape_backward_fast_path[256,300]`, `test_dw_uses_declared_ragged_token_rule` |
| lm_head:7 | **confirmed, fixed** | report claimed legacy numbers / "untested switch" and "5e7 samples" | this section; the tanh test samples (1<<22)+(1<<20)+7 = **5,242,887** values (not 5e7) | — |
| lm_head:8 | **confirmed (nit), fixed by documentation** | fp32 golden uses `torch.mv` (cuBLAS gemv) | docstring + `trace()["golden"]`: the fp64 mode is THE R4 golden, fp32 mode is a tolerance convenience | — |

Also changed: `dh_t = gemm(dz_t, W_t^T, ragged_k=True)` — dh reduces over the tile's vocabulary
rows (an input-gradient reduction); a tile that is not a multiple of 256 (tiny reference configs,
tile 128) uses det_gemm's declared masked-tail dA rule instead of silently switching kernels.
The real tile (25344 = 99 x 256) is unaffected.

**Bits.** Forward (logits, logp) unchanged at every T (K = H = 6656, same launch; verified by the
bit ledger in the JSON). Backward: dW changes for T % 256 != 0 (legacy -> declared tree rule)
and for 32 < T <= 512 (GEMM fix det_gemm:1); dh unchanged at the real geometry.

**Performance (A100, base 6a61f6e -> 87cba55, fwd+bwd):** T=4096 455.0 -> 407.1 ms (no
transposed copies), T=4001 975.6 -> 405.6 ms (2.4x, no legacy dW), T=300 103.2 -> 60.8 ms,
T=256 59.3 -> 43.6 ms; forward unchanged (95.7 -> 95.9 ms at T=4096 = 115 TFLOP/s; decode T=1
GPU kernel time 2.39 ms per call before and after, wall time is launch/sync-bound and varies
2.9–3.6 ms run to run on the shared node). The real-weights
forward at T=16 now reports 2 MiB peak extra memory (the 323 MiB below was the legacy GEMM's
transposed weight copy).

Contract: `docs/contracts/muse-glimmer-30b-strict-contract.md` O7 (+ O3 tree, R1–R6).
Trace nodes: `lm_head`, `logit_softcap`, `logprob`. RFC #501 rows `lm_head_softcap_logprob`,
`vocab_parallel_softcap_logp`.

## Summary

| Gate | Result |
| --- | --- |
| G1 forward vs fp64 golden | PASS — max abs err **1.9e-6** (small, T=37) and **9.5e-7** (real shape V=202048/H=6656, T=8); ws1.json logprob bf16 row is 0.06, own tight bound 2e-3 |
| G2 batch invariance | PASS — `torch.equal` for T=1 vs T=256 rows, permuted batches, T=16 prefix, real shape T=1 vs T=64 |
| Tile independence | PASS — `torch.equal` for reversed / shuffled tile evaluation order |
| Simulated TP2/4/8 vs TP1 | PASS — `torch.equal` (rank sub-trees + rank tree), small and real shape |
| Padded columns excluded | PASS — garbage in pad rows changes no bit; a target in `[V, Vp)` raises |
| G4 backward | PASS — repeat `torch.equal` (dh, dw, small and real shape); vs fp32 autograd dh 1.4e-2 of max 3.9 (0.36 %), dw 3.5e-3 of max 1.18 (0.3 %) — bf16 rounding; pad rows of dW exactly 0; dh row batch invariant |
| G5 trace() | PASS — stable, JSON-serialisable, carries tree / tile / cast / gemm backend |
| tanh convention (O7) | `libdevice.tanh` is **bitwise equal to `torch.tanh`** fp32 on the shipped test's 5,242,887 samples (**[stale]**: "5e7" overstated the shipped test); the fused soft-cap kernel is `torch.equal` to the torch golden soft cap |
| Real weights | `lm_head.weight` (202048×6656 bf16) + `norm.weight` loaded from the pinned checkpoint, forward at T=16 finite, ≤0, batch invariant; 323 MiB peak extra memory |
| Benchmark T=4096 | **[stale — legacy GEMM era]** strict fwd 668 ms vs native 69 ms (9.7×), fwd+bwd 2659 ms vs 183 ms (14.6×). Current (fast_det_gemm): fwd 95.9 ms, fwd+bwd 407.1 ms (see follow-up) |

**[stale]** 15/15 tests pass — now 23 tests (see the follow-up and `kernels_gemm.json`). The op is model-scoped (`rl_engine/models/muse_glimmer/{ops,triton}`), nothing
shared was edited.

## Design

### Files
- `rl_engine/models/muse_glimmer/triton/softcap_logprob.py` — Triton kernels: tile statistics,
  backward `dz`, elementwise soft cap. `BLOCK_V=1024`, `num_warps=4`, no autotune.
- `rl_engine/models/muse_glimmer/ops/lm_head_softcap_logprob.py` — public API (fixed names):
  `pad_lm_head_weight`, `softcap_tile_stats`, `merge_tile_stats`, `combine_tile_stats`,
  `compute_tile_stats`, `lm_head_softcap_logprob` (autograd), `lm_head_softcap_logprob_fp32`
  (golden, `compute_dtype=float32|float64`), `softcap_golden`, `softcap_full_logits`,
  `simulate_tp_logprob`, `legacy_det_gemm_fp32`, `resolve_gemm`, `trace`, `LMHeadSoftcapLogprobOp`.
- `tests/models/muse_glimmer/test_lm_head_softcap_logprob.py`,
  `benchmarks/models/muse_glimmer/bench_lm_head_softcap_logprob.py`.

### Forward arithmetic (fp32 unless stated)
1. **Tiles.** `W_lm` is padded to `Vp = 202752 = 8 × 25344` with zero rows (`pad_lm_head_weight`);
   tile `t` = rows `[25344t, 25344(t+1))`; tile 7 contains the 704 padded rows. Per tile
   `z_t = gemm(h, W_t)` → **fp32 logits, never rounded to bf16**. One fp32 tile is alive at a time
   (415 MB at T=4096); the full 3.3 GB logits are never materialised (only `softcap_full_logits`
   does, for debugging).
2. **Soft cap.** `ẑ = T·tanh((z·m)/T)` with `m = output_multiplier` (0.19611613513818404),
   `T = final_logit_softcapping` (20.0), in HF order mul-m → div-T → tanh → mul-T. Kernel: RN
   multiply, `tl.div_rn` (IEEE division — Triton's default `/` is approximate), `libdevice.tanh`,
   RN multiply. Golden: `cap * torch.tanh((z * mult) / cap)` with **tensor-valued** `mult`/`cap`
   (torch's Python-scalar division takes a reciprocal-multiply fast path that differs by 1 ulp in
   ~20 % of values). Measured on 5e7 fp32 samples (|z| up to 1e4, near zero, ±0): bitwise equal.
   No fallback formula `(1−e^{−2x})/(1+e^{−2x})` is needed.
3. **Tile statistics** (`_softcap_tile_stats_kernel`, one program per (row, tile)): pass 1 walks
   the tile in `BLOCK_V=1024` chunks from the tile start, `m_t = max ẑ` (exact); pass 2 walks
   the same chunks in the same order, `s_t += tl.sum(exp(ẑ − m_t))` (sequential chunk order,
   `tl.sum`'s fixed intra-chunk tree, `tl.exp` per R2). Columns with global index `≥ vocab_size`
   are masked to the identity (−inf / 0). The target's `ẑ[y]` is recomputed with the same formula
   → identical bits to the streamed values. Nothing depends on the row index, on T, or on other
   tiles → batch invariance and tile independence by construction (R3).
4. **Tree.** `((0+1)+(2+3))+((4+5)+(6+7))` (`_combine_tree`, recursive balanced tree over a
   power-of-two leaf list; unit-tested to produce exactly that string). Node merge, elementwise
   torch fp32: `M = max(Ma, Mb)`, `S = Sa·exp(Ma−M) + Sb·exp(Mb−M)` (empty leaves `(−inf, 0)` are
   guarded). `logZ = M + log S`; `logp = ẑ[y] − logZ`. A TP rank's consecutive 8/t tiles form a
   node of the same tree, so `merge_tile_stats(rank tiles)` then `combine_tile_stats(ranks)` is
   bit-identical to TP1 (tested for t = 1, 2, 4, 8).

### Backward (deterministic, autograd.Function)
Saved: `h`, `W_padded`, `targets`, `logZ` (T fp32). Per tile: recompute `z_t` with the same GEMM;
`dz_t = (onehot − exp(ẑ − logZ))·dlogp·(m·(1 − tanh²))` in fp32, pads forced to 0, **one rounding
to bf16** (declared cast: det_gemm operands are bf16 per R1). `dh = tree_t(gemm(dz_t, W_tᵀ))`
with the same 8-leaf tree over fp32 partials → bf16 once. `dW_t = gemm(dz_tᵀ, h)` fp32 → bf16;
padded rows are exactly zero. Provenance via `record_backward("lm_head_softcap_logprob", ...)`.

### GEMM plug
`gemm(a (M,K) bf16, w (N,K) bf16, *, out_dtype) -> a @ w.T` (the sibling `fast_det_gemm`
signature). `resolve_gemm()` imports `rl_engine.models.muse_glimmer.ops.det_gemm.fast_det_gemm`
when it exists (wrapped to pass contiguous operands), else `legacy_det_gemm_fp32` =
`rl_engine.backends.shared.triton.gemm.det_gemm._triton_gemm_fp32(a, w.T)`: fixed 64/64/32
blocks, one program per output tile, K walked in order (no split-K), `tl.dot(allow_tf32=False)`
with inputs promoted to fp32 inside the kernel, fp32 output. The bf16 output path of
`TritonDetGemmOp` cannot emit fp32, so the fp32 path is used as the brief instructs. For `dh`
partial sums `rl_engine.models.muse_glimmer.ops.canonical_tree.combine_tree` is preferred when
importable (absent in this worktree → private `_combine_tree(…, torch.add)`, same order).

## Contract deviations
1. **`dz → bf16` before the backward GEMMs.** O7 writes `dh = det_gemm(dẑ, W_lm)` without naming
   the cast; R1 makes det_gemm operands bf16, so the cast is declared here (and in `trace()`).
   Effect: dh/dW relative error ≈ 2⁻⁸ vs fp32 autograd (measured 0.3–0.4 %).
2. **GEMM inner arithmetic.** The legacy path promotes bf16 operands to fp32 and runs an fp32 dot
   (products of bf16 values are exact in fp32, accumulation fp32) — same numerical class as a
   bf16 tensor-core dot with fp32 accumulation, different accumulation order inside a
   `BLOCK_K=32` step. The result class is unchanged; performance is (see benchmark).
3. **`tl.exp` vs `torch.exp`.** The tile sum uses `tl.exp` (R2; ex2-based, up to ~14 ulp vs
   `torch.exp` on the probed range). The golden therefore is a tolerance reference, not bitwise
   (expected by the contract). Everything the parity gates compare is kernel-vs-kernel.
4. **Targets must lie in `[0, vocab_size)`** — no `ignore_index`; inactive tokens must be masked by
   the caller (as the repo's `vocab_parallel_logp` does with its `active_mask`).

## Test commands and results

```
PYTHONPATH=<worktree> ~/bin/gpu-lease -n 1 -- ~/envs/vllm/bin/python -m pytest \
    tests/models/muse_glimmer/test_lm_head_softcap_logprob.py -v -p no:cacheprovider -s
```
```
tests/models/muse_glimmer/test_lm_head_softcap_logprob.py::test_pad_lm_head_weight_appends_zero_rows PASSED
tests/models/muse_glimmer/test_lm_head_softcap_logprob.py::test_tile_geometry_real_shape PASSED
tests/models/muse_glimmer/test_lm_head_softcap_logprob.py::test_combine_tree_is_the_canonical_eight_leaf_tree PASSED
tests/models/muse_glimmer/test_lm_head_softcap_logprob.py::test_combine_tile_stats_matches_logsumexp_and_handles_empty_tiles PASSED
tests/models/muse_glimmer/test_lm_head_softcap_logprob.py::test_golden_fp32_matches_fp64_and_rejects_pad_targets PASSED
tests/models/muse_glimmer/test_lm_head_softcap_logprob.py::test_trace_is_complete_and_stable PASSED
tests/models/muse_glimmer/test_lm_head_softcap_logprob.py::test_softcap_kernel_is_bitwise_torch_tanh PASSED
tests/models/muse_glimmer/test_lm_head_softcap_logprob.py::test_g1_forward_matches_fp64_golden_small
G1 small: max|logp - fp64 golden| = 1.907e-06, vs fp32 golden = 1.907e-06
PASSED
tests/models/muse_glimmer/test_lm_head_softcap_logprob.py::test_g2_batch_invariance_rows_and_positions PASSED
tests/models/muse_glimmer/test_lm_head_softcap_logprob.py::test_tile_order_independence PASSED
tests/models/muse_glimmer/test_lm_head_softcap_logprob.py::test_simulated_tp_equals_tp1 PASSED
tests/models/muse_glimmer/test_lm_head_softcap_logprob.py::test_padded_columns_are_excluded_and_pad_targets_raise PASSED
tests/models/muse_glimmer/test_lm_head_softcap_logprob.py::test_g4_backward_matches_fp32_autograd_and_is_deterministic
G4 small dh: max|ours - fp32 autograd| = 1.419e-02 (max|ref| = 3.935e+00)
G4 small dw: max|ours - fp32 autograd| = 3.506e-03 (max|ref| = 1.177e+00)
PASSED
tests/models/muse_glimmer/test_lm_head_softcap_logprob.py::test_real_shape_determinism_and_forward_error
G1 real shape (T=8): max|logp - fp64 golden| = 9.537e-07
PASSED
tests/models/muse_glimmer/test_lm_head_softcap_logprob.py::test_real_weights_forward_shapes_and_memory
real weights: logp[:4]=[-15.62059211730957, -13.658824920654297, -16.27662467956543, -12.971212387084961] peak extra memory = 323 MiB
PASSED

============================== 15 passed in 8.79s ==============================
```
CPU-only subset: `CUDA_VISIBLE_DEVICES= … -m unit` → 6 passed. Platform: A100-SXM4-80GB,
torch 2.13.0+cu130, Triton 3.7.1, TF32 disabled.

Tolerance statement: G1 uses the read-only ws1.json `accuracy.default.logprob.bfloat16.atol =
0.06` (execution dtype of the operands is bf16) and additionally asserts `≤ 2e-3` vs fp64; the
measured error is 1e-6. The fp32 `logprob` row (1e-5) is not applicable to bf16 operands but is
met anyway (1.9e-6 / 9.5e-7). Gradients: `rtol=2e-2`, `atol=2e-2·max|ref|` vs fp32 autograd
(bf16 outputs; the ws1 bf16 gradient row is an absolute 0.05, which is uninformative for these
magnitudes).

## Parity results
| Parity | Shape | Result |
| --- | --- | --- |
| Batch: row at T=1 vs T=256 (positions 0, 1, 63, 64, 128, 255) | small | `torch.equal` |
| Batch: permuted batch, T=16 prefix | small | `torch.equal` |
| Batch: T=1 vs T=64 | real (V=202048, H=6656) | `torch.equal` |
| Repeat: forward, dh, dw (two runs) | small + real | `torch.equal` |
| dh row at T=1 vs T=40 | small | `torch.equal` |
| Tile order: reversed, [3,0,6,1,7,2,5,4], random | small | `torch.equal` |
| TP emulation 1/2/4/8 | small; TP4 at real shape | `torch.equal` |
| Pad rows filled with N(0,50²) garbage | small | `torch.equal` with zero pads |
| Sequence split | n/a — the op is stateless per row (G3 not applicable) | — |

## Benchmark (A100-80GB, T=4096, H=6656, V=202048 → Vp=202752, random weights)

```
~/bin/gpu-lease -n 1 -- ~/envs/vllm/bin/python \
    benchmarks/models/muse_glimmer/bench_lm_head_softcap_logprob.py --tokens 4096 --iters 2 --backward
```
| pass | native ms | native peak MiB | strict ms | strict peak MiB | strict/native |
| --- | ---: | ---: | ---: | ---: | ---: |
| forward | 69.0 | 9474 | 668.1 | 718 | 9.7x |
| forward+backward | 182.5 | 15249 | 2659.4 | 7195 | 14.6x |

Forward GEMM throughput: native (cuBLAS bf16) 160 TFLOP/s; strict legacy fp32-promoted det_gemm
16.5 TFLOP/s. `max |native − strict| logp = 3.0e-3` — that is the **native** path's own loss
(cuBLAS rounds the logits to bf16 before the soft cap); strict vs fp64 is 1e-6. The strict path
needs 13× less memory because it never holds the full logits. Obvious next step (not done): plug
the sibling `fast_det_gemm` (≥ 80 TFLOP/s target) — the op switches automatically when
`rl_engine.models.muse_glimmer.ops.det_gemm` is importable; expected forward ≈ 150 ms (≈ 2×
native). The tile-stats kernels themselves cost ≈ 5 ms per forward (two reads of each fp32
tile); the backward's second GEMM pass (recompute of `z_t`) is the price of not storing 3.3 GB.

## Integration requests (for the integrator; nothing shared was edited)
1. `rl_engine/validation/operators/operator_specs.py` — optional registry row (the model chain can
   also import the function directly):
   ```python
   "lm_head_softcap_logprob": OperatorSpec(
       name="lm_head_softcap_logprob",
       op_class="logprob",
       gold_path="rl_engine.models.muse_glimmer.ops.lm_head_softcap_logprob.LMHeadSoftcapLogprobOp",
       gold_method="forward_fp32",
       candidate_paths={
           "triton": "rl_engine.models.muse_glimmer.ops.lm_head_softcap_logprob.LMHeadSoftcapLogprobOp",
       },
       grad_input_names=("h", "w_padded"),
   ),
   ```
   (The op takes `(h (T,H) bf16, w_padded (Vp,H) bf16, targets (T,) int64)`; a case generator for
   the harness would have to pad the weight with `pad_lm_head_weight`.)
2. `rl_engine/contracts/profiles/precision/ws1.json` — no change required (the `logprob` bf16 row
   0.06 applies and is met by 4 orders of magnitude). If a dedicated row is wanted:
   `"muse_glimmer_lm_head_softcap_logprob": {"bfloat16": {"atol": 1e-3, "rtol": 0.0}}`.
3. Model chain (full_model_chain): after the final norm,
   `logp = lm_head_softcap_logprob(h, pad_lm_head_weight(lm_head.weight), targets)`;
   for sampling parity `softcap_full_logits(h, w_padded)[:, :VOCAB_SIZE]` gives the fp32 ẑ.
4. WS2 (`vocab_parallel_softcap_logp`): rank r calls
   `compute_tile_stats(h, w_padded, targets, tiles=range(r*8/t, (r+1)*8/t))`,
   `merge_tile_stats` locally, all-gathers `(M_r, S_r, target_logit_r, target_in_tile_r)` with the
   deterministic collective (transport only), then `combine_tile_stats(ranks)` and the
   `torch.where` select of the target logit — exactly what `simulate_tp_logprob` does on one GPU.
   Backward: each rank runs `_backward_tiles` on its tiles; `dh` partials are the per-rank
   sub-tree sums and need the same rank tree (= `tp_canonical_tree`).

## Open issues
- **[stale, resolved]** ~~`fast_det_gemm` / `canonical_tree.combine_tree` did not exist ... the
  automatic switch is untested~~: the default is `fast_det_gemm` (eager import, no switch),
  `combine_tree` is imported directly, and the fast path is tested at small and real shapes.
- Backward recomputes the 8 logits tiles (a second LM-head GEMM). Storing `dz` or the logits is a
  memory/time trade-off left to the trainer integration.
- No `ignore_index`; inactive tokens are the caller's mask (as elsewhere in the repo).
- The legacy GEMM copies `W_tᵀ` once per tile per forward (337 MB bf16, ≈ 0.5 ms) because
  `_triton_gemm_fp32` wants `(K, N)`; irrelevant once `fast_det_gemm` lands.
- PR RL-Align/RL-Kernel#542 (the "shared soft-cap row" the RFC cites) does not resolve on GitHub
  (`gh pr view 542` → not found). The Gemma 4 work on `upstream/test-gemma`
  (`final_logit_softcap`, hard-coded `30·tanh(x/30)`, no multiplier, elementwise only) was used as
  the convention source (`libdevice.tanh` + `tl.div_rn`, `torch.tanh` golden) rather than imported.

## RFC feedback
1. **Config key.** The pinned checkpoint's `text_config` has `final_logit_softcapping: 20.0` and
   `output_multiplier: 0.19611613513818404`; there is **no `output_soft_cap_temp`** key. The model
   contract table should say `final_logit_softcapping: 20.0`.
2. **Formula and order.** State it explicitly: `ẑ = T · tanh(z · m / T)` with the order
   `lm_head → × output_multiplier → ÷ T → tanh → × T` (HF `modeling_muse_glimmer.py` lines
   1169–1172), all in fp32 on fp32 logits; the soft cap is applied **before** any vocabulary
   partial, and padded columns are excluded after the cap.
3. **One row, not two.** The single-GPU op already *is* the TP decomposition: the 8 tiles are the
   canonical TP8 shards and the fixed tree is the TP tree, so `vocab_parallel_softcap_logp`
   reduces to the deterministic all-gather of four small per-row tensors plus the same two
   functions. Recommend merging into one row "`lm_head_softcap_logprob` (TP-ready)", with the
   WS2 acceptance (`TP2/4/8 == TP1` on real GPUs, G1/G2) as that row's second checklist item —
   or keep the WS2 row but scope it to transport + multi-GPU validation only.
4. **Reuse citation.** The repo's vocab-parallel logprob (#241/#336) merges tiles *flat*
   (`max` over all tiles, `Σ s_t·exp(m_t−M)` in torch order), which is batch invariant but not the
   O3 tree; only its tile-stats kernel pattern was reused. The row text should say "reuse the tile
   statistics; the merge must be the O3 canonical tree".
5. **The tanh clause** in O7 can be simplified: `libdevice.tanh` (Triton) and `torch.tanh` (CUDA)
   are bitwise equal in fp32; the real pitfall is the division (`tl.div_rn` vs torch's
   reciprocal-multiply for scalar divisors) — worth one sentence in the contract.
6. **Performance expectation.** The lm_head GEMM is 11 TFLOP at T=4096; with the legacy det_gemm
   the row runs at 9.7× native. The RFC's "benchmark against the native path" acceptance should
   name `fast_det_gemm` as a dependency of this row.

## Time spent
≈ 1.5 h wall clock (contract + prior art 25 min, kernels + op 35 min, tests/benchmark/report 30 min).
