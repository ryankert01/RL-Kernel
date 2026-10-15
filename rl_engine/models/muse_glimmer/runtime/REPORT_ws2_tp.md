# REPORT — WS2 tensor-parallel rows (RFC #501, Muse-Glimmer-30B strict POC)

Rows: `tp_canonical_tree` (row-parallel `o_proj` / `down_proj` on the canonical 8-leaf tree),
`vocab_parallel_softcap_logp` (soft-capped selected-token logprob on 8 canonical vocab tiles),
`kv_head_replication` forward half (declared replicated-KV ownership map + proof that
replicated-KV attention at TP t equals TP1).

## Summary

| Row (real shapes, random weights) | TP2==TP1 | TP4==TP1 | TP2==TP4 | TP8==TP1 | Repeat | Correctness |
|---|---|---|---|---|---|---|
| `tp_canonical_tree(o_proj)` M=256 K=4096 N=6656 | PASS bitwise | PASS bitwise | PASS | simulated PASS (NCCL run opt-in) | PASS | bf16 vs cuBLAS fp32 within ws1 reduction/bf16 tol |
| `tp_canonical_tree(down)` M=256 K=19968 N=6656 | PASS | PASS | PASS | simulated PASS | PASS | same |
| `vocab_parallel_softcap_logp` T=64 H=6656 V=202048/202752 | PASS | PASS | PASS | simulated PASS | PASS | max abs err vs fp64 merge 3.9e-6 (gate 1e-5) |
| `kv_head_replication` fwd, T=512 (2 seqs) and T=2176 window 2048 | PASS | PASS | PASS | PASS (single device, 8 ranks) | PASS | vs SDPA fp32 ref max 0.0097 (ws1 attention/bf16 tol 0.05) |

Negative control: NCCL `all_reduce(SUM)` of the same fp32 sub-tree roots then bf16 is equal to
TP1 at TP2 (two operands commute) and NOT equal at TP4 (31-50 mismatching bf16 elements of
1,703,936 per run) — the tree is needed. All rows were run twice: first against labelled
fallbacks (repo fp32 det_gemm kernel, pure-torch tile statistics), then, after the sibling ops
landed (`ops/det_gemm.fast_det_gemm_partial`, `ops/canonical_tree.combine_tree`,
`ops/lm_head_softcap_logprob.{softcap_tile_stats,combine_tile_stats}`), against the real
kernels. Same verdicts both times. `rl_engine._C` is not built on this node, so the CUDA-IPC
collective is replaced by `TorchDistributedDeterministicCollective` (NCCL all-gather as
transport only, same balanced tree evaluated locally). The strict attention forward kernel is
not landed, so `kv_replication` still runs on my pure-torch fixed-order fallback.

## Design (reduction order, casts, tiles)

### Layout (`runtime/tp_layout.py`)
`TPLayout(tp, canonical_tp=8)` derives per-rank Q heads, KV head(s), canonical K-chunks and
vocab tiles from `OFFICIAL_PARALLEL_LAYOUT` / `OFFICIAL_FINGERPRINT`;
`fail_closed_if_undeclared(declared, world_size=)` raises on non-replicated KV, another canonical
TP, other shard widths, other vocab padding or `world_size != tp`. Chunk/tile `c` -> rank
`c // (8/tp)`.

| chunk / tile | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| TP1 rank | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| TP2 rank | 0 | 0 | 0 | 0 | 1 | 1 | 1 | 1 |
| TP4 rank | 0 | 0 | 1 | 1 | 2 | 2 | 3 | 3 |
| TP8 rank | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| `o_proj` K cols (x512) | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| `down` K cols (x2496) | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| vocab rows (x25344) | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 (+704 pad rows) |

KV ownership map (declared, `TPLayout.kv_ownership_map()`; q head h reads kv head h//16; rank r
holds kv head floor(2r/tp)):

| TP | rank -> Q heads | rank -> KV head | replication |
|---|---|---|---|
| 1 | 0 -> [0,32) | 0 -> {0,1} | 1x |
| 2 | 0 -> [0,16), 1 -> [16,32) | 0 -> 0, 1 -> 1 | 1x (identity) |
| 4 | r -> [8r, 8r+8) | 0,1 -> 0 ; 2,3 -> 1 | 2x |
| 8 | r -> [4r, 4r+4) | 0-3 -> 0 ; 4-7 -> 1 | 4x |

`o_proj` chunk c (512 columns) is exactly the 4 Q heads of canonical rank c, so the row-parallel
K ownership of `o_proj` coincides with the column ownership of the attention output.

### `tp_canonical_tree` (`runtime/tp_canonical_tree.py`)
`row_parallel_tree_gemm(a_local (M,K_local) bf16, w_local (N,K_local) bf16, layout, collective)
-> bf16 (M,N)`: (1) fp32 sub-tree partial over the owned consecutive chunks
(`fast_det_gemm_partial`; fallback: repo `_det_gemm_fp32_kernel` per chunk — one program per
output tile, BLOCK_K=32 ascending, no split-K, `allow_tf32=False`, launched on the transposed
weight view and verified bit-identical to a contiguous-transpose launch — plus a private
pairwise fold); (2) rank-ordered deterministic `all_gather` of the fp32 roots (no arithmetic);
(3) `combine_tree` over the `tp` roots in fp32; (4) one bf16 rounding. TP1 folds all 8 chunks
locally. Because `8/tp` consecutive aligned chunks are always one node of
`((0+1)+(2+3))+((4+5)+(6+7))`, every TP evaluates the same fp32 expression. `trace()` records
dtypes, order, shard widths, backends and chunk->rank.

### `vocab_parallel_softcap_logp` (`runtime/vocab_parallel_softcap_logp.py`)
`W_lm (202048, 6656)` zero-padded to 202752 = 8 x 25344. Per owned tile: fp32 logits
(`fast_det_gemm(out_dtype=float32)`), soft cap `T*tanh((z*m)/T)` in fp32 in the HF order
(mul m, div T, tanh, mul T; tensor-valued constants => IEEE division), pad columns (>= 202048,
all in tile 7) excluded (-inf), `M_t = max`, `S_t = sum exp(zhat - M_t)` fixed order (sibling
Triton kernel; fallback: pairwise tree over 128 lanes then over 198 lane-blocks zero-padded to
256 — elementwise ops only, batch-invariant), target logit if the target is in the tile, and the
ownership flag. The four `(8/tp, T)` fp32 statistics travel by ONE `all_gather_many`; every rank
merges the 8 tiles in global tile order with `M=max(Ma,Mb); S=Sa*exp(Ma-M)+Sb*exp(Mb-M)` in the
same tree, `logZ = M + log S`, `logp = zhat[y] - logZ` (fp32). Targets are validated on every
rank BEFORE any collective; a padded (>= 202048) or negative id raises `ValueError` on all
ranks.

### `kv_head_replication` forward (`runtime/kv_replication.py`)
Rank r runs gated attention on its `32/tp` Q heads against its single replicated KV head
(`k.index_select(1, [floor(2r/tp)])`) and its gate channels. Proof: attention + gate are
separable per Q head (no cross-head reduction) and every replica holds byte-equal K/V (same
deterministic `k_proj`/`v_proj`), so `y_tp1[:, cols(r)] == y_r` and the rank-ordered concat
equals TP1. `prove_kv_replication()` checks TP1/2/4/8 on one device; the NCCL tests rebuild the
TP1 tensor through the deterministic all-gather. Fallback attention follows O5 (64-key blocks
aligned at absolute position 0, online softmax fp32, `p -> bf16` before PV, `acc/l -> bf16`,
gate `sigmoid(g.float())` fp32 -> bf16) with pairwise elementwise trees for the 128-lane and
64-key reductions, so a head's bits do not depend on how many heads share the launch. The
import chain prefers `triton.gated_attention_fwd.strict_gated_attention_fwd_varlen` when it
lands. Backward half not implemented; `trace()["backward_rule"]` states the 4-leaf rule and
points at `triton.gated_attention_bwd.strict_gated_attention_bwd_partials` (fp32
`(8, Tk, 2, 128)` group partials), which is exactly the input that rule consumes.

## Contract deviations
1. Collective: CUDA-IPC `DeterministicCollective` needs `rl_engine._C` (absent);
   `make_collective()` falls back to `TorchDistributedDeterministicCollective` (same API, same
   tree, NCCL all-gather as transport). Bits identical by construction; CUDA-IPC path untested.
2. Attention: the strict Triton forward is not landed; the KV proof ran on the pure-torch
   fallback (QK / PV as elementwise pairwise trees rather than `tl.dot`). The proof only needs
   per-head separability, which any O5 kernel has; re-run `test_kv_replication.py` when the
   kernel lands (no code change: import chain).
3. Fallback `S_t` order (lane tree + block tree) and the fallback per-chunk K order are my
   definitions; the sibling kernels now define the bits and all tests pass on them unchanged.
4. Soft-cap association `tanh((z*m)/T)*T` with tensor constants, identical to the sibling
   `softcap_golden`. Both runtimes must keep this exact form.
5. `kv_head_replication` backward (dK/dV 4-leaf sub-tree + all-gather) is out of scope here.
6. Sequence parallelism (reduce-scatter instead of all-reduce) is not covered; the row is
   defined with SP off.

## Test commands and results

Single-GPU gates (included in the 2-GPU run below; also run alone):
```
gpu-lease -n 1 -- pytest tests/models/muse_glimmer/distributed -q
48 passed, 18 skipped in 20.37s          # fallback kernels
```
Two GPUs, sibling kernels (`gpu-lease -n 2 --contiguous --pool 0-5`, full suite):
```
............sss.......................ssssss.s.................sss       [100%]
53 passed, 13 skipped in 46.77s          # skips: need 4 / 8 devices
```
Four GPUs, sibling kernels (`gpu-lease -n 4 --contiguous --pool 0-5`, `-k CrossTP`):
```
...s......ss.....s                                                       [100%]
14 passed, 4 skipped, 48 deselected in 114.04s (0:01:54)   # skips: TP8 opt-in
```
Fallback-kernel runs before the merge: 2 GPUs `5 passed, 13 skipped` (34.75s); 4 GPUs
`14 passed, 4 skipped` (138.02s). TP8 tests exist in every file and skip unless
`MUSEGLIMMER_TP8=1` with 8 visible devices.

`torchrun --standalone --nproc_per_node=4 -m rl_engine.models.muse_glimmer.runtime.check_tp_parity`
(sibling kernels):
```
# gemm=muse_glimmer.ops.det_gemm.fast_det_gemm_partial
# tree=muse_glimmer.ops.canonical_tree.combine_tree
# softcap=muse_glimmer.ops.lm_head_softcap_logprob
# attention=fallback:pure_torch_fp32_blockwise_online_softmax
## Equality matrix (TP4 vs TP1, per rank)
row                                shape                                        r0 r1 r2 r3  rerun
tp_canonical_tree(o_proj)          M=256 K=4096 N=6656                          == == == ==  ok
tp_canonical_tree(down)            M=256 K=19968 N=6656                         == == == ==  ok
vocab_parallel_softcap_logp        T=64 H=6656 V=202048/202752                  == == == ==  ok
kv_head_replication(forward)       T=512 seqs=[160, 352] window=None            == == == ==  ok
## Negative control: NCCL all_reduce(SUM) of the fp32 sub-tree roots -> bf16
tp_canonical_tree(o_proj)          equals TP1: False  mismatches: 46/1703936
tp_canonical_tree(down)            equals TP1: False  mismatches: 48/1703936
RESULT: PASS
```
`--nproc_per_node=2`: same matrix all `==`, NCCL control `equals TP1: True  mismatches: 0`
(commutativity), `RESULT: PASS`.

## Parity results
* Batch: `row_parallel_tree_gemm_tp1(a[5:6])[0] == full[5]`; tile statistics of one row alone
  == the row inside the T=64 batch; attention per-head bits independent of the head count in
  the launch (the TP proof itself). All `torch.equal`.
* Sequence split: not a stateful row here; the attention cases cover 2 packed sequences
  (160+352, global) and one 2176-token local sequence crossing the 2048 window.
* Repeat: every row twice in-process and twice across the collective — bitwise.
* Cross-TP: sha256 of the TP-t output equals sha256 of TP1 on every rank; all ranks identical.
* Negative controls: left fold of the 8 fp32 chunk partials vs the tree (single device):
  o_proj 80 (8 leaves) / 41 (4) / 0 (2) mismatching bf16 elements of 1,703,936; down 80 / 28 /
  0. NCCL all_reduce of the sub-tree roots: TP2 0 mismatches (both runs); TP4 31/50 (fallback
  run) and 46/48 (sibling run) for o_proj/down.
* Pad handling: poisoning the 704 padded LM-head rows with 64.0 leaves logp bitwise unchanged;
  targets 202048, 202751, 202752, -1 raise before any collective.

## Benchmark
Fallback per-chunk fp32 det_gemm (A100, Triton 3.7.1): o_proj chunk 256x512x6656 0.4 ms, down
chunk 256x2496x6656 0.7 ms, vocab tile 64x6656x25344 1.8 ms; cuBLAS fp32 (TF32 off) 0.27 /
0.98 / 2.50 ms; repo bf16 `TritonDetGemmOp` tree path 0.4 / 3.1 / 17.6 ms. Full TP1 rows on
the fallbacks: down 13 ms, vocab logp 190 ms (pure-torch statistics dominate), o_proj 571 ms on
the first (compiling) call. Sibling-kernel single-GPU timings are in `ops/REPORT_det_gemm.md`
and `ops/REPORT_lm_head_softcap_logprob.md`.

Rank reduction of the fp32 (256, 6656) root (6.8 MB), ms per call, rank 0, sibling run:

| | TP2 o_proj | TP2 down | TP4 o_proj | TP4 down |
|---|---|---|---|---|
| deterministic all_gather + tree (strict path) | 0.162 | 0.155 | 0.224 | 0.222 |
| NCCL all_reduce SUM (control, order not fixed) | 0.164 | 0.114 | 0.151 | 0.149 |
| collective.all_reduce (gather + balanced tree) | 0.181 | 0.205 | 0.230 | 0.227 |

The strict path costs 1.0-1.5x a plain NCCL all-reduce at this size (the fp32 root is 2x the
bytes of a bf16 all-reduce and the tree is a few elementwise adds). Obvious next step (not
done): CUDA-IPC `DeterministicCollective.all_reduce` once `_C` is built, and a fused
gather+tree kernel.

## Integration requests
* No shared file touched; no tolerance row needed (all WS2 gates are `torch.equal`).
* Registration is import-time: the three runtime modules import
  `rl_engine.models.muse_glimmer.ops.{det_gemm, canonical_tree, lm_head_softcap_logprob}` and
  `triton.gated_attention_fwd` (when present) and fall back on `ImportError`. Nothing to add to
  registries for the POC.
* CI lines: `gpu-lease -n 1 -- pytest tests/models/muse_glimmer/distributed` and
  `gpu-lease -n 4 --contiguous -- pytest tests/models/muse_glimmer/distributed -k CrossTP`;
  idle node: `MUSEGLIMMER_TP8=1 pytest ... -k tp8` and
  `torchrun --standalone --nproc_per_node=8 -m rl_engine.models.muse_glimmer.runtime.check_tp_parity`.
* Both runtimes (vLLM provider, Megatron Bridge provider) should call
  `official_layout(tp).fail_closed_if_undeclared(world_size=...)` before the first strict
  collective and emit `TPLayout.provenance()` into the run manifest.

## Open issues
1. CUDA-IPC collective untested (no `_C`); the torch.distributed transport produced every number.
2. Strict attention forward kernel not landed; KV proof on the fallback (re-run on landing).
3. `kv_head_replication` backward half: partials exist
   (`strict_gated_attention_bwd_partials`), the rank all-gather + 4-leaf tree is a follow-up.
4. Sequence parallelism (Megatron SP on => reduce-scatter) is not covered; the row text should
   say SP off, or add an SP row (`collective.reduce_scatter` already evaluates the same tree).
5. TP8 multi-process run pending on the idle node; 8-rank single-device simulations pass.

## RFC feedback
* `tp_canonical_tree` row: "o_proj and FFN-down are row-parallel over 8 canonical K-chunks
  (512 / 2496). Each chunk's fp32 det_gemm partial is folded in the fixed tree
  ((0+1)+(2+3))+((4+5)+(6+7)); at TP t a rank folds its 8/t consecutive chunks locally, the
  fp32 sub-tree roots are exchanged by a deterministic rank-ordered all-gather (no arithmetic
  in the collective) and every rank folds the roots with the same tree, then rounds to bf16
  once. TP2/4/8 == TP1 byte-for-byte. A NCCL/RCCL all-reduce of the same roots is NOT
  acceptable (order not fixed; it matches only at TP2 by commutativity). Defined with sequence
  parallelism OFF." Add that the o_proj chunk boundaries coincide with the canonical 4-head
  groups, and that partials travel as fp32 (2x bf16 bytes).
* `vocab_parallel_softcap_logp` row: state the padded layout (202048 -> 202752 = 8 x 25344;
  the 704 pad rows live in tile 7, are masked to -inf and are never selectable), that the soft
  cap is `T*tanh((z*m)/T)` in fp32 on fp32 logits BEFORE the per-tile (max, sumexp)
  statistics, that the 8 tile statistics are merged in the SAME canonical tree as the GEMM row
  (not a flat 8-way logsumexp), and that the payload is four `(8/tp, T)` fp32 vectors (max,
  sumexp, target logit, ownership flag) in one `all_gather_many`. Cite #542 only "if landed".
* `kv_head_replication` row: "Replicated KV ownership map rank r -> KV head floor(2r/TP)
  (identity at TP2, 2x at TP4, 4x at TP8), declared in `TPLayout.kv_ownership_map()` and
  enforced by `fail_closed_if_undeclared` on both runtimes (native in vLLM; replaces Megatron's
  QKV all-gather + slice on the training side). Forward needs no communication and is
  byte-equal to TP1 because attention + gate are per-head separable and every replica holds the
  same k_proj/v_proj bits. Backward: dK/dV and the k_proj/v_proj weight gradients are produced
  per canonical 4-head group and summed in the 4-leaf sub-tree (c0+c1)+(c2+c3) per KV head at
  every physical TP including TP1."
* Open question 2 (K/V gradient rule): the forward half supports the per-canonical-group rule.
  With canonical TP 8, KV head g is read by exactly canonical ranks 4g..4g+3, so its gradient is
  one 4-leaf sub-tree of the SAME tree the forward uses for o_proj (chunks 4g..4g+3 = heads
  16g..16g+15); the landed backward kernel already emits the 8 group partials. Recommend
  adopting it as "the o_proj chunk tree restricted to the KV head's half", parameterised only by
  (num_kv_heads, canonical_tp) so #415 / #434 can share it. What this POC cannot settle: whether
  the group partials are kept fp32 until the single bf16 rounding (as O3 does and as the kernel
  now emits) or rounded per group as Megatron's native bf16 reduce-scatter would — the forward
  evidence (TP4 NCCL control fails only through re-association) says keep fp32 and round once.

## Time spent
About 4 hours: 1 h reading the contract / collectives / prior art, 1.5 h implementation and
single-GPU gates, 1 h leased multi-GPU runs, 0.5 h sibling-op merge, re-runs and this report.
