# REPORT — gated attention BACKWARD (rows `global_gated_attention`, `local_gated_attention_train`)

## Wave B update (review wave A2) — backward kernels unchanged, bits unchanged

Tests (final code, one A100 of pool 3,7): the three owned files 223 passed, 1 skipped
(test_gated_attention_bwd.py: 75 tests, all passed); log
`reports/muse_glimmer_poc/data/kernels_attention_raw/waveB/tests_owned.log.txt`. Distributed CP
tests NOT re-run in this wave (one GPU at a time).

- **Backward bits**: the 14 backward digests of `gated_attention_bench digest` (dq, dk, dv,
  dgate at T = 1..4096, global + local) are identical to f91e8b6
  (`reports/muse_glimmer_poc/data/kernels_attention_raw/waveB/digest_cmp.json`).
- **Autograd validates in the forward (gated_attention_fwd:R1/R4).** Without `seq_meta`,
  `GatedAttentionFn.forward` now calls `prepare_sequence_meta` when any input requires grad and
  hands the meta to the forward wrapper (no extra sync there) and to the backward (no sync
  there): a packing the backward rejects (non-contiguous `q_pos`, `cu_seqlens_k[-1] != Tk`, ...)
  now fails before the forward kernel runs, and a layer pays 2 D2H copies per forward+backward
  instead of 1 (forward) + 2 (backward). Test `test_autograd_validates_in_forward_and_reuses_meta`
  (backward under `set_sync_debug_mode("error")`; fails on f91e8b6). With a per-micro-batch meta
  the whole op stays sync-free (unchanged).
- **Megatron binding (kernels_attention:R5)**: since c88607f, `strict_kernels.resolve_ops` binds
  `GatedAttentionFn` (the sibling op) for the Megatron path, which passes no `seq_meta`, so every
  layer pays the 2 copies above. Request to the integrator: build one `SequenceMeta` per
  micro-batch with `prepare_sequence_meta` and pass it to `gated_attention(..., seq_meta)`.
- **CP evidence (kernels_attention:R4).** `test_one_shot_equals_cp1_path` is now labelled a
  plumbing check (at CP1 the runtime calls the same preprocess -> partials -> combine as the
  one-shot entry, so equality is by construction). New
  `test_cp_backward_plumbing_cp4_cp8`: the real `runtime/cp_attention.cp_gated_attention_backward`
  of every rank at CP4 and CP8 on one GPU with a replayed all_gather (record pass, then the
  rank-ordered concatenation NCCL would return); dq/dk/dv/dgate concatenated over ranks equal the
  one-shot kernel bitwise at T = 2048 (CP4, CP8) and 2100 (CP4), global + local.
- **TP tree assertions (kernels_attention:R8)**: `test_tp_head_shards_equal_tp1_slices` docstring
  now says the per-rank partial equality carries the evidence and the final tree assertions are
  documentation of the binding recipe (same adds on bitwise-equal operands).
- **SequenceMeta precondition (kernels_attention:R7)**: documented in the class docstring — the
  meta is bound by identity + `_version`, so a write that bypasses PyTorch's version counter
  (Triton/CUDA kernel or NCCL into a persistent buffer, `.data`) is not detected; the forward
  kernel's device-side checks still catch a stale meta's out-of-range packing in the forward.
- **Local SDPA backward ratio (kernels_attention:R3)**: the SDPA backward baseline is the
  full-causal flash backward, so local-layer ratios are not comparable. The JSON now stores
  `ratio: null` + `baseline_not_comparable` for every local backward row; the "2.08x (local)"
  figure below is withdrawn. No windowed backward baseline exists in the envs (vLLM's FA2 build is
  forward-only).

## v2 update (wave A) — supersedes the v1 sections below where they disagree

Kernel id `muse_glimmer.triton.gated_attention_bwd.v2`. Machine-readable results (raw mismatch
counts, timing series, finding verdicts): `reports/muse_glimmer_poc/data/kernels_attention.json`.
Measured on A100-SXM4-80GB, torch 2.13.0+cu130, Triton 3.7.1 (`~/envs/vllm`), GPUs from pool 4-5.

| Gate | Result (observed this session) |
|---|---|
| Owned tests, final code, 2 leased GPUs | fwd + bwd files 143 passed; CP files 18 passed, 5 skipped (CP4 / TP2xCP2 need 4 GPUs) |
| TP2/4/8 rank call on its head shard == TP1 slice (y, o, lse, dq, dgate, delta, dK/dV group partials) | 0 bit-pattern mismatches (T=2100, packed [1500,600,333], T=8192; global + local) |
| Replicated-KV dK/dV: rank nodes folded in the canonical TP tree == TP1 dk/dv | 0 at TP2/4/8 |
| One-shot kernel == CP path (cp_attention CP1, simulated CP2/4/8, real CP2 on 2 GPUs) | 0 for dq/dk/dv/dgate, T in {2048, 2100, 4096, 8192} |
| Forward bits vs the old forward (c88607f) | 66/66 digests identical |
| Backward bits vs v1 | dq, dgate changed (sigmoid fix); dk, dv changed (query-chunk rule + sigmoid) |
| Backward time vs v1 (7 alternating rounds, median) | +5..14 % at T >= 1024, +55 % at T = 512; 2.52x SDPA causal bwd at T = 4096 (global; the local SDPA ratio is not comparable, see wave B) |

What changed versus v1:
1. **dK/dV query-chunk rule (CP-invariant).** Leaves (query chunk c, Q-head group g):
   chunk `c = [floor(c*Lk/8), floor((c+1)*Lk/8))` on absolute positions, 64-query tiles aligned
   at the chunk start, heads of g ascending then tiles ascending; `P_g = ((c0+c1)+(c2+c3))+
   ((c4+c5)+(c6+c7))`; KV head h = `(P_g0+P_g1)+(P_g2+P_g3)` over the 4 groups reading h -> bf16
   once. A CP rank owns whole chunks, so its node is a sub-tree of the one-shot tree. v1 (one
   ascending fold over all tiles) differed from its own CP1 path in 1557 / 5540 / 20440 dK
   elements at T = 2048 / 4096 / 8192 (global).
2. **Gate sigmoid = the forward's** (`sigmoid_rn = div_rn(1, 1 + exp(-g))`, shared helper). Probe
   (`y = bf16(o*sg_fwd)` vs `do = bf16(o*sg_bwd)` on one-key sequences, 16.8M values): v2 0
   mismatches, v1 59.
3. **Head counts as parameters** (`num_q_heads`/`num_kv_heads` in (32,2), (16,1), (8,1), (4,1))
   in the backward entries; the fused `strict_gated_attention_bwd` raises for TP4/TP8 shards
   (partials only); `kv_partials_subtree` returns the rank's node of the per-KV-head tree.
4. Autograd binds the strict forward (no golden fallback); one forward golden (the bwd golden is
   a varlen adapter over `gated_attention_forward_fp32`); `dy` must be bf16;
   `cu_seqlens_k[-1] == Tk` and device checks; `Tk == 1` compiles (v1: CompilationError);
   `SequenceMeta` makes the direct backward and the autograd op (forward + backward) free of
   host syncs (tested under `torch.cuda.set_sync_debug_mode("error")`).

Bits changed versus v1 (bit-pattern mismatches / numel; "same do" feeds both kernels the same
`do`/`delta`, isolating the dK/dV rule):

| T | layer | dq | dk | dv | dgate | dq, same do | dk, same do | dv, same do |
|---|---|---|---|---|---|---|---|---|
| 2048 | global | 866/8.4M | 1830/524k | 1457/524k | 100/8.4M | 0 | 1560 | 1435 |
| 4096 | global | 1552/16.8M | 6032/1.05M | 5422/1.05M | 248/16.8M | 0 | 5543 | 5370 |
| 8192 | global | 3625/33.6M | 21546/2.1M | 20093/2.1M | 490/33.6M | 0 | 20438 | 19976 |
| 2048 | local | 866/8.4M | 1830/524k | 1457/524k | 100/8.4M | 0 | 1560 | 1435 |
| 4096 | local | 1645/16.8M | 4692/1.05M | 4022/1.05M | 232/16.8M | 0 | 4018 | 3973 |
| 8192 | local | 3614/33.6M | 9034/2.1M | 7748/2.1M | 510/33.6M | 0 | 7746 | 7611 |

dgate changes only through the sigmoid, dq only through `do`. Size of the changes over all 15
lengths (T = 2..8192): max |diff| dq 0.0039, dk 0.125, dv 0.0625, dgate 0.0039, i.e. at most
5.1e-3 of max |old| (dk); dgate and `do` differ by at most 1 bf16 ulp per element, but dq / dk /
dv also flip near-zero elements produced by cancellation (max ulp distance ~3e4 there), so the
change is not a last-bit change of every element. Forward outputs: unchanged.

Backward timing before/after (`gated_attention_bench bwd-ab`, ms, median of 7 alternating rounds
x 10 iterations; SDPA = causal GQA backward without window or gate; n/c = not comparable: the
local rows' SDPA baseline does the full-causal work, not the windowed one):

| T | layer | v1 | v2 | v2/v1 | SDPA | v2/SDPA |
|---|---|---|---|---|---|---|
| 512 | global | 0.634 | 0.981 | 1.55 | 0.174 | 5.64 |
| 1024 | global | 1.000 | 1.124 | 1.12 | 0.343 | 3.28 |
| 2048 | global | 2.013 | 2.124 | 1.05 | 0.813 | 2.61 |
| 4096 | global | 5.766 | 6.124 | 1.06 | 2.434 | 2.52 |
| 8192 | global | 19.817 | 20.973 | 1.06 | 8.376 | 2.50 |
| 512 | local | 0.622 | 0.983 | 1.58 | 0.202 (full causal) | n/c |
| 1024 | local | 0.891 | 1.009 | 1.13 | 0.316 (full causal) | n/c |
| 2048 | local | 2.050 | 2.156 | 1.05 | 0.814 (full causal) | n/c |
| 4096 | local | 4.440 | 5.048 | 1.14 | 2.431 (full causal) | n/c |
| 8192 | local | 9.547 | 10.818 | 1.13 | 8.407 (full causal) | n/c |

Transient backward memory 0.876 GiB at T = 8192 (v1 0.349 GiB): the fp32 chunk leaves
`(8, H_Q/4, Tk, 128)` dominate. A fused in-kernel chunk loop with a register-resident tree would
remove the extra programs, the leaf memory and the torch fold (not done).

Per-rank call (TP t, rank r; `H_Q = 32//t`, `H_KV = 1` for t >= 2, KV head `(2r)//t`):
`gated_attention_bwd_preprocess(dy_r, o_r, gate_r, num_q_heads=H_Q)` ->
`strict_gated_attention_bwd_partials(q_r, k_r, v_r, do, lse, delta, cu_q, cu_k, q_pos,
window=W, num_q_heads=H_Q, num_kv_heads=H_KV, seq_meta=meta)` -> `kv_partials_subtree(dkp)`;
TP1/TP2: that node -> bf16 is final; TP4/TP8: all_gather the nodes of the t/2 replicas of the
KV head in rank order, `combine_tree(nodes) -> bf16` (measured == TP1). Full text for the
integrator (O5 wording, signatures): `kernels_attention.json` `requests_to_integrator`.

Not run this session: CP4 / CP8 / TP2xCP2 (need more than the 2 leased GPUs); TP4/TP8 as
multi-process NCCL jobs (proven with per-rank calls on one GPU).

The sections below are the v1 report, kept for history; numbers there refer to v1.

Contract: `docs/contracts/muse-glimmer-30b-strict-contract.md` O5 ("Backward" paragraph) + O3 tree.
Scope: dQ, dK, dV and the gate gradient of the gated GQA attention, sliding-window and
full-causal, as deterministic Triton kernels; the autograd Function that binds them to the
sibling forward; the fp32 fixed-order golden forward + FP64 autograd reference used by the tests.

## Summary

| Gate | Result |
|---|---|
| G4 repeat determinism (`torch.equal`, dq/dk/dv/dgate + fp32 partials, global & local) | PASS |
| G4 accuracy vs FP64 autograd reference, L in {64, 2047, 2048, 2049, 4096} x {global, local} | PASS — normalized max error <= 5.7e-3 (dq), 5.5e-3 (dk), 3.7e-3 (dv), 4.9e-3 (dgate) |
| G2 batch invariance (sequence alone vs packed at batch slots 1 and 3, neighbours of different lengths) | PASS (`torch.equal` on dq/dgate rows, dk/dv rows and the 8 group partials) |
| TP simulation: tree over 8 group partials == per-rank sub-trees + tree over ranks for TP2/4/8 | PASS (`torch.equal`) |
| Real shape Tq = Tk = 4096, 32x2 heads | PASS (accuracy + determinism) |
| G5 `trace()` | PASS (stable, tiles pinned) |
| Benchmark, Tq=4096 bf16 A100 | strict 5.78 ms (global) / 4.47 ms (local) vs SDPA causal bwd 2.43 ms -> 2.38x / 1.84x |

24 tests, 32 s on one A100. No atomics, no split-K, no autotune, `allow_tf32=False`.

## Files

- `rl_engine/models/muse_glimmer/triton/gated_attention_bwd.py` — kernels + host wrappers
  (`gated_attention_bwd_preprocess`, `strict_gated_attention_bwd_partials`,
  `combine_kv_partials`, `strict_gated_attention_bwd`, `trace`).
- `rl_engine/models/muse_glimmer/ops/gated_attention_autograd.py` — `GatedAttentionFn`,
  `gated_attention(...)`, `gated_attention_backward(...)`, `trace()`; forward = sibling
  `strict_gated_attention_fwd_varlen` if importable, else the clearly marked golden fallback.
- `rl_engine/models/muse_glimmer/ops/attention_golden_bwd.py` — `golden_forward_fixed_order`
  (fp32, contract casts, no matmul), `reference_forward`/`reference_backward` (FP64 autograd).
- `tests/models/muse_glimmer/test_gated_attention_bwd.py`
- `benchmarks/models/muse_glimmer/bench_gated_attention_bwd.py`

## Design

### Pipeline
1. **Preprocess (row-local, one program per (token, head), fp32).**
   `sg = sigmoid(g)`; `dgate = dy*o*sg*(1-sg)` -> bf16; `do = dy*sg` -> bf16 (declared cast);
   `delta_i = sum_d do_bf16[d]*o_bf16[d]` (fp32, `tl.sum` over the fixed 128-lane block).
2. **dK/dV kernel — one program per (64-key block, sequence, canonical Q-head group g).**
   Grid `(max_k_blocks, B, 8)`. For `h` in the group's 4 heads ascending, for query tile `qb`
   ascending over the tiles that can see the key block:
   `sT = tl.dot(K_blk, Q_tile^T)*scale` (fp32) masked -> -inf; `pT = exp(sT - lse)`;
   `dV += tl.dot(pT->bf16, dO_tile)`; `dPT = tl.dot(V_blk, dO_tile^T)`; `dST = pT*(dPT - delta)`;
   `dK += tl.dot(dST->bf16, Q_tile)`; finally `dK *= scale`. Emitted as **fp32 partials**
   `dk_partials/dv_partials (8, Tk, 2, 128)`: slice `g` holds group g's contribution to KV head
   `(4g)//16`; the other KV head's slice of `g` is exactly 0.
3. **Tree combine.** `dK = ((p0+p1)+(p2+p3))+((p4+p5)+(p6+p7))` over the 8 group slices in fp32
   -> bf16, once (private `_combine_tree_fp32`; `canonical_tree.combine_tree` is imported instead
   when it exists — it did not exist in this worktree). Because the 4 slices of the other KV head
   are exact zeros this equals the per-KV-head 4-leaf tree `((g0+g1)+(g2+g3))` bit for bit.
4. **dQ kernel — one program per (64-query tile, sequence, head).** Grid `(max_q_blocks, B, 32)`.
   Key blocks ascending from `floor(max(0, p_first-W+1)/64)` (0 on global layers) to
   `floor(p_last/64)`: `s = tl.dot(Q, K_blk^T)*scale` masked; `p = exp(s - lse)`;
   `dP = tl.dot(dO, V_blk^T)`; `dS = p*(dP - delta)`; `dQ += tl.dot(dS->bf16, K_blk)`;
   `dQ *= scale` -> bf16.

### Reduction order and invariance
- Every reduction is a fixed-shape `tl.dot` (bf16 x bf16 -> fp32, `allow_tf32=False`) or a
  fixed-shape `tl.sum`; cross-block accumulation is the ascending block order above.
- Query tiles are aligned at the **sequence-local index 0**, key blocks at **absolute key
  position 0** (= index inside the sequence); hence the blocks of a sequence are identical
  whatever its packing offset -> batch invariance. Fully masked blocks/rows contribute exact
  zeros (`p = 0 => dS = 0`), so tiles that span positions never change a row's bits.
- Mask: global `k_pos <= q_pos`; local `q_pos - W < k_pos <= q_pos` (W = 2048), evaluated per
  element inside the kernels; the loop bounds only skip blocks that are provably fully masked.
- `lse = -inf` rows (no visible key) are guarded (`p = 0`), no NaN.

### Casts (all declared in `trace()["cast_points"]`)
`do` fp32->bf16; `p` fp32->bf16 before the dV dot; `dS` fp32->bf16 before the dQ/dK dots;
`dgate` fp32->bf16; dq/dk/dv fp32->bf16 once at the end; partials stay fp32.

### Tiles / launch constants (R6, in `trace()`)
`BLOCK_N = 64` keys (contract), `BLOCK_M = 64` queries, `HEAD_DIM = 128`, `num_warps = 8`,
`num_stages = 2`; preprocess `num_warps = 4`. No autotune.

### Golden / reference
- `golden_forward_fixed_order`: per sequence, 64-key blocks ascending from the contract's first
  visible block, online softmax with the exact O5 update order, `s = sum_d q_d k_d` and
  `acc += sum_j bf16(p_j) v_j` as sequential loops (no `torch.matmul`), `p->bf16` before PV,
  `out->bf16`, `y->bf16`, `lse = m + log l`. 2.6–2.8 s at L = 4096 (test/fallback use only).
- `reference_backward`: FP64 `torch.autograd` of the un-quantised function (matmul allowed: it
  is the FP64 reference the contract permits, not a strict path and not the golden).

## Contract deviations / clarifications
1. **Backward casts are not in the contract.** O5 says "FlashAttention-2 style ... fp32
   accumulation" but does not name the bf16 operands of the backward dots. This implementation
   declares `do`, `p`, `dS` -> bf16 before the tensor-core dots (standard FA2). Keeping `dS` or
   `do` in fp32 was measured (torch emulation, L = 512) to reduce the dK error only from
   7.5e-2 to 6.5e-2, i.e. they are not the dominant term (see 4).
2. **p recompute uses `lse = m + log l`** (`p = exp(s - lse)`), which is what the sibling
   forward returns; the contract wording "from the saved (m, l)" is read as this.
3. **delta uses the saved bf16 `o`**: `delta = rowsum(do_bf16 * o_bf16)` (FA2) instead of
   `sum_j p_ij dP_ij`; equal in exact arithmetic, differs by bf16 rounding of `o`.
4. **Accuracy tolerance applied at unit scale.** The ws1 `gradient_accuracy/attention/bf16` row
   (atol 5e-2, rtol 2e-2) is calibrated for O(1) tensors. Attention gradients scale with the
   arbitrary scale of `dy`; with unit-variance `dy`, `max|dK| ~ 26`, where the contract-mandated
   bf16 OUTPUT rounding alone is 0.0625 > atol (measured: an all-fp32 backward rounded to bf16
   misses the raw row by the same margin, 6.1e-2). The tests therefore apply the row to
   `tensor / max|reference|` and additionally assert `max|err| / max|ref| <= 1e-2`
   (observed <= 5.7e-3 across all 22 accuracy cases). Raw errors at L = 4096: dq 1.35e-2
   (max|ref| 2.47), dk 1.13e-1 (28.7), dv 5.2e-2 (14.4), dgate 1.2e-2 (2.47). Note the
   `gradient_accuracy/reduction/bf16` row (atol 0.1) would pass dk at L <= 2049 but not at 4096
   with unit dy — a scale-free criterion is needed for gradient rows (see RFC feedback).
5. **Query positions must be contiguous inside a sequence** (`q_pos[cu_q[b] + t] = q_pos0 + t`,
   training/prefill layout). The kernels bound their block loops from `q_pos0`; the host
   validates and raises otherwise. Arbitrary `q_pos0` (chunked-prefill style, keys 0..Lk-1) is
   supported and tested.
6. **The query tile of the dK/dV kernel (64) is not pinned by the contract**; it is part of the
   reduction order of dK/dV and is recorded in `trace()`.
7. **Forward fallback.** While the sibling forward is absent the autograd forward runs the fp32
   golden (slow, contract casts but not the kernel's bits); `trace()["forward_is_fallback"]`
   reports it. Not a strict forward — the integrator must land the sibling file.

## Test commands and results

```
cd <worktree>
PYTHONPATH=$PWD ~/bin/gpu-lease -n 1 -- ~/envs/vllm/bin/python -m pytest \
    tests/models/muse_glimmer/test_gated_attention_bwd.py -x -q -s
```
Tail (verbatim):
```
[accuracy window=None L=64] normalized max errors: {'dq': 0.005661594681441784, 'dk': 0.003743590787053108, 'dv': 0.0027982930187135935, 'dgate': 0.004937674384564161}
.
[accuracy window=2048 L=64] normalized max errors: {'dq': 0.005661594681441784, 'dk': 0.003743590787053108, 'dv': 0.0027982930187135935, 'dgate': 0.004937674384564161}
.
[accuracy window=None L=2047] normalized max errors: {'dq': 0.004875635262578726, 'dk': 0.003971153870224953, 'dv': 0.002971879905089736, 'dgate': 0.0035794172435998917}
.
[accuracy window=2048 L=2047] normalized max errors: {'dq': 0.004875635262578726, 'dk': 0.003971153870224953, 'dv': 0.002971879905089736, 'dgate': 0.0035794172435998917}
.
[accuracy window=None L=2048] normalized max errors: {'dq': 0.004883881658315659, 'dk': 0.005027051083743572, 'dv': 0.0031735401134938, 'dgate': 0.004393812734633684}
.
[accuracy window=2048 L=2048] normalized max errors: {'dq': 0.004883881658315659, 'dk': 0.005027051083743572, 'dv': 0.0031735401134938, 'dgate': 0.004393812734633684}
.
[accuracy window=None L=2049] normalized max errors: {'dq': 0.004153928253799677, 'dk': 0.003389958059415221, 'dv': 0.00308898463845253, 'dgate': 0.004226105287671089}
.
[accuracy window=2048 L=2049] normalized max errors: {'dq': 0.004153928253799677, 'dk': 0.003389958292245865, 'dv': 0.00308898463845253, 'dgate': 0.004226105287671089}
.
[accuracy window=None L=4096] normalized max errors: {'dq': 0.0037491030525416136, 'dk': 0.00546896131709218, 'dv': 0.003632467705756426, 'dgate': 0.004743609577417374}
.
[accuracy window=2048 L=4096] normalized max errors: {'dq': 0.004799604415893555, 'dk': 0.00425322912633419, 'dv': 0.003617619862779975, 'dgate': 0.004706304986029863}
...............
24 passed in 32.04s
```
Environment: A100-SXM4-80GB, torch 2.13.0+cu130, triton 3.7.1, TF32 off.

## Parity results
- **Repeat**: two runs (and the partials -> tree path vs the fused path) `torch.equal` for
  dq, dk, dv, dgate on lens [2049, 64, 300], global and local.
- **Batch**: sequence of 777 tokens alone vs packed as [130, A, 2049] (slot 1) and
  [2049, 64, 5, A] (slot 3): `torch.equal` on dq rows, dgate rows, dk/dv rows and all 8 fp32
  group partials; o/lse rows equal as a precondition.
- **Sequence-split (G3)**: not applicable to the training backward (no chunked backward); the
  forward fallback honours the contract's block rule so o/lse are chunk-independent.
- **TP2/4/8 simulation**: `tree(rank sub-trees) == tree(8 groups)` bit for bit; non-owned KV
  head slices of every group partial are exactly zero.

## Benchmark (Tq = Tk = 4096, 32x2 heads, D = 128, bf16, A100, 20 iters after warm-up)

```
PYTHONPATH=$PWD ~/bin/gpu-lease -n 1 -- ~/envs/vllm/bin/python \
    benchmarks/models/muse_glimmer/bench_gated_attention_bwd.py
```
| row | strict bwd total (ms) | strict core w/o tree (ms) | SDPA causal bwd (ms) | ratio | golden fwd fallback (s) |
|---|---|---|---|---|---|
| global_gated_attention | 5.78 | 5.66 | 2.43 | 2.38x | 2.83 |
| local_gated_attention_train | 4.47 | 4.32 | 2.43 | 1.84x | 2.62 |

The tree combine + partial allocation costs ~0.12 ms. The SDPA baseline is causal only (no
window, no gate), so the local row is compared against the same causal baseline.
Obvious next optimisation (not done): load Q/dO tiles pre-transposed and raise `BLOCK_M` of the
dQ kernel to 128 (K/V are currently re-read once per head, 16x per KV head); emit per-KV-head
partials `(4, Tk, 128)` instead of the half-zero `(8, Tk, 2, 128)` layout.

## Integration requests
1. Autograd op entry (for `rl_engine/validation/operators/operator_specs.py` /
   `rl_engine/runtime/registry.py`, exact format at the integrator's discretion):
   ```python
   # op "muse_glimmer.gated_attention" (rows global_gated_attention, local_gated_attention_train)
   from rl_engine.models.muse_glimmer.ops.gated_attention_autograd import gated_attention, trace
   # candidate "triton": gated_attention(q, k, v, gate_logits, cu_seqlens_q, cu_seqlens_k,
   #                                      q_pos, window)  -> y bf16 (Tq, 4096); backward attached
   ```
2. No new tolerance row: tests use `gradient_accuracy/attention/bfloat16` at unit scale plus
   the explicit normalized bound 1e-2 (see deviation 4). If the integrator wants a profile row,
   suggested text: `"attention_gradient": {"bfloat16": {"mode": "normalized_tolerance",
   "max_err_over_max_ref": 1.0e-2}}`.
3. Sibling hooks already wired: `rl_engine.models.muse_glimmer.triton.gated_attention_fwd
   .strict_gated_attention_fwd_varlen` (forward) and
   `rl_engine.models.muse_glimmer.ops.canonical_tree.combine_tree` (tree) are imported when
   present; the fallbacks are reported in `trace()`.
4. WS2 `kv_head_replication` consumes `strict_gated_attention_bwd_partials(...)` ->
   `(dq, dk_partials, dv_partials)`; at TP t a rank owning groups `[r*8/t, (r+1)*8/t)` runs the
   same kernel with its 4*8/t heads and reduces its slices with the recursive-halving tree, then
   the ranks' partials are gathered and reduced with the same tree (tested in simulation).

## Open issues
- Performance 1.8–2.4x SDPA backward (adequate for the POC; see benchmark note).
- The forward used by `GatedAttentionFn` is the slow golden until the sibling kernel lands; the
  autograd test switches to an `allclose` forward check automatically when it does.
- `_sequence_meta` syncs `cu_seqlens`/`q_pos` to the host once per backward (same as
  flash-attn's `max_seqlen` requirement); fine for training, would need caching for tight loops.
- The bitwise neutrality of fully-masked tiles inside `tl.dot` (exact-zero operands) is relied
  upon and verified by the batch-invariance tests on A100; it should be re-verified on each new
  MMA path (Hopper wgmma, ROCm MFMA) by the same tests.

## RFC feedback (open question 2 — per-group K/V gradient rule)
- **Workable: yes, and essentially free.** Emitting dK/dV per canonical Q-head group costs no
  extra arithmetic: a per-KV-head kernel would have to sum 16 heads inside one program (128
  programs at L = 4096 — 1.2 waves on 108 SMs), whereas 8 groups give 512 independent programs
  and the fixed tree over 4 groups is a ~0.1 ms elementwise op. Memory: fp32 partials
  8 x Tk x 2 x 128 x 4 B = 33.5 MB each for dK and dV at Tk = 4096 (half of it exact zeros in the
  `(8, Tk, 2, 128)` layout; `(4 groups, Tk, 128)` per KV head would halve it).
- **What the rows should say.** (a) "dK/dV are accumulated per canonical Q-head group (4
  consecutive heads = TP8 shard) in fp32, heads ascending then 64-query tiles ascending; the 4
  groups of a KV head are combined by the fixed tree ((g0+g1)+(g2+g3)) in fp32 -> bf16 once, at
  every TP including TP1." The current text "8 partial dK/dV per KV head" is misleading: a KV
  head has 4 contributing groups, the other 4 are zero. (b) State that at TP2 a rank owns a whole
  KV head (no cross-rank K/V-gradient reduction), at TP4 two ranks share one (reduce 2 leaves),
  at TP8 four ranks share one (reduce 4 leaves) — the tree is the same object in all cases.
  (c) Name the backward casts (do, p, dS -> bf16 before the dots, delta from bf16 o) in the
  contract so CUDA and Triton kernels agree bitwise. (d) Replace the absolute gradient tolerance
  with a scale-free one (error relative to max|reference|, or state the dy scale); the absolute
  ws1 rows cannot be met by any bf16-output gradient once |grad| > 16.
- #415/#434 can share the rule: it only needs the forward's lse/o and the group -> KV-head map.

## Time spent
~2.5 h (reading 25 min, kernels + golden 60 min, tests/bench/debug 45 min, report 20 min).
