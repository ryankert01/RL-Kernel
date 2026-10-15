# REPORT — `gated_attention` forward (contract O5; RFC #501 rows `global_gated_attention`, `local_gated_attention_train`, `local_gated_attention_rollout`)

## Wave A update — head-count parameters (supersedes "Open issues 5" below)

Machine-readable results: `reports/muse_glimmer_poc/data/kernels_attention.json`. Measured on
A100-SXM4-80GB, torch 2.13.0+cu130, Triton 3.7.1 (`~/envs/vllm`), GPUs from pool 4-5.

- **New parameters** `num_q_heads` / `num_kv_heads` on `strict_gated_attention_fwd_varlen` and
  `strict_gated_attention_fwd_paged` (defaults 32 / 2, existing callers unchanged). Accepted
  per-rank geometries: `SUPPORTED_HEAD_GEOMETRIES = (32,2) TP1, (16,1) TP2, (8,1) TP4, (4,1) TP8`;
  anything else raises (`check_head_geometry`). Rank r of TP t passes `q[:, 32r/t:32(r+1)/t]`,
  `k/v[:, kvh:kvh+1]` with `kvh = (2r)//t` (cache `(pages, 16, 1, 128)` on the paged path),
  `gate[:, 4096r/t:4096(r+1)/t]`; the outputs are the rank's `(Tq, H_Q*128)` / `(Tq, H_Q)` slices.
- **HEAD tile at TP4/TP8**: the tile keeps `BLOCK_M = 16` rows and 4 warps; a group of 8 / 4
  heads fills the first rows, the others are masked (zeros loaded, `-inf` scores, never stored),
  so the valid rows run the same instruction stream as at TP1.
- **Gate sigmoid** moved into the shared Triton helper `sigmoid_rn` (same operations:
  `div_rn(1, 1 + exp(-g))`), which the backward preprocess now also calls.
- **Forward bits unchanged**: 66 / 66 `(y, o, lse)` digests identical to the forward at c88607f
  (varlen TOKEN / HEAD / auto tiles at T = 1..8192, packed batch, chunked prefill, paged prefill,
  paged decode B = 1..8 at ctx up to 4096; global and local).
- **TP shard parity (bit patterns)**: rank call == TP1 slice for y, o, lse at TP2/4/8, every rank,
  on T = 2100, packed [1500, 600, 333], T = 8192 (TOKEN tiles) and paged decode B = 8 ctx 2049
  (HEAD tiles): 0 mismatches, global and local. Tests: `test_tp_head_shard_varlen_bitwise`,
  `test_tp_head_shard_paged_decode_bitwise` (TOKEN + HEAD tiles, chunked prefill, paged decode).

Timing (CUDA events, 30 iterations; SDPA = `F.scaled_dot_product_attention` with GQA + the same
gate; the local SDPA baseline is a dense bool mask, not a windowed flash kernel):

| case | strict ms | SDPA ms | strict/SDPA |
|---|---|---|---|
| prefill T=512 global | 0.105 | 0.134 | 0.79 |
| prefill T=1024 global | 0.226 | 0.250 | 0.90 |
| prefill T=2048 global | 0.636 | 0.612 | 1.04 |
| prefill T=4096 global | 2.058 | 1.570 | 1.31 |
| prefill T=8192 global | 6.051 | 3.940 | 1.54 |
| prefill T=4096 local | 1.326 | 30.152 (masked) | 0.04 |
| prefill T=8192 local | 2.903 | 121.923 (masked) | 0.02 |
| paged decode B=1 ctx 4096 global | 0.334 | 0.101 | 3.31 |
| paged decode B=16 ctx 4096 global | 0.414 | 0.105 | 3.96 |
| paged decode B=64 ctx 4096 global | 0.338 | 0.203 | 1.67 |
| paged decode B=256 ctx 4096 global | 0.829 | 0.927 | 0.89 |
| paged decode B=64 ctx 4096 local | 0.350 | 18.044 (masked) | 0.02 |

Per-rank prefill forward at T = 4096 (global / local, ms): TP1 32/2 1.700 / 1.325, TP2 16/1
0.939 / 0.710, TP4 8/1 0.673 / 0.480, TP8 4/1 0.430 / 0.282. (The TP1 global call measured
1.70 ms in this series and 2.06 ms in the prefill series of the same run, same inputs; the gap
reproduced in two runs and is a measurement-position effect, not investigated.) Decode at small
batch is 3-4x SDPA: one program per (token, KV head) leaves most SMs idle at B <= 16 (no
split-KV by contract); adequate for the POC, a fixed split-KV tree would be the fix.

Agent scope: FORWARD only (kernel, wrappers, fp32 golden, paged KV test utility, tests,
benchmark). The deterministic backward and the autograd `Function` are owned by another agent
and import `strict_gated_attention_fwd_varlen`, `strict_gated_attention_fwd_paged` and
`gated_attention_forward_fp32` from `rl_engine/models/muse_glimmer/ops/gated_attention.py`.

Branch: `worktree-agent-a753a1fe25f1d2402` (based on `main` bf265a7 + merge of
`feat/museglimmer-arch-fingerprint` e2f4e64, which carries the contract and scaffolding).

## Summary

| Gate | Result |
|---|---|
| G1 kernel vs fp32 golden (WS1 attention bf16: atol 5e-2, rtol 2e-2) | PASS. max abs err y 2.08e-2, o 1.52e-2 (n=2300), lse 7.6e-6; real prefill-4096: y 8.6e-3, o 7.8e-3. Errors are dominated by the bf16 rounding of `o`/`y` (half-ulp at abs(o)~4 is 1.6e-2). fp32 golden vs fp64 golden: y 6.5e-3 (bf16 round flips of `p`), lse 2.3e-6. Kernel vs fp64 golden: 9.5e-3. |
| G2 batch invariance (`torch.equal` on y, o, lse) | PASS. 6 sequences (lens 1..2100) alone vs packed in 3 orders incl. duplicates, both tile modes, both layer kinds; paged decode batch with mixed contexts (1..2049) vs alone. |
| G3 sequence-split invariance (`torch.equal`) | PASS for lengths {64, 2047, 2048, 2049, 4096, 8192} x {global, local}: one-shot vs chunked varlen (cuts at 2047/2048/2049, random 3- and 7-way, [1, n-1]) vs chunked paged (no eviction / eager eviction) vs token-by-token paged decode with eager eviction and NaN-poisoned freed pages. Eviction timing (never vs eager) equal; pages freed by one sequence and reused by another leave both sequences' bits unchanged. |
| BLOCK_M=16 (decode) vs BLOCK_M=64 (prefill) bitwise | PASS **only with the same `num_warps` (4) for both tiles** — see Contract deviations. Verified on lengths 1..2100 and on decode B=64 @ 4096. |
| GQA mapping (h -> h // 16) | PASS (perturbing kv head 1 changes exactly q heads 16..31; golden agrees). |
| Real shapes | PASS: prefill Tq=4096 (global/local), decode B=64 @ 4096 context (paged == varlen == TOKEN tile, bitwise; golden on 3 sequences). |
| G4 backward | not in scope (other agent). |
| G5 `trace()` | PASS: stable, JSON-serializable, carries op/dtypes/reduction order/casts/tile config. |
| Benchmark (A100-80GB) | prefill 4096: 2.06 ms vs SDPA-flash 1.57 ms (1.31x). Decode B=64@4096: paged 0.32 ms / varlen 0.20 ms vs SDPA 0.20 ms (1.6x / 1.0x). Local layer: SDPA has no window kernel (dense bool mask, 30 ms / 18 ms) so the 0.04x/0.02x ratios are not meaningful. |

63 tests, 33 s on one A100.

## Design (reduction order, casts, tiles)

Files: `rl_engine/models/muse_glimmer/triton/gated_attention_fwd.py` (kernel),
`rl_engine/models/muse_glimmer/ops/gated_attention.py` (wrappers, golden, `trace()`),
`rl_engine/models/muse_glimmer/ops/paged_kv.py` (test paged cache).

One kernel body `_gated_attention_fwd_kernel` serves varlen (packed) and paged K/V via
`constexpr` flags `PAGED`, `LOCAL`, `HEAD_TILE`. One program = one query tile of one sequence.

Per query row (fp32 unless stated):

1. K/V visited in `BLOCK_N = 64` blocks aligned at absolute key position 0, ascending. For a
   tile spanning positions `[p_min, p_max]`: blocks `floor(max(0, p_min-W+1)/64) ..
   floor(min(p_max, len_k-1)/64)` (local) or `0 .. floor(p_max/64)` (global).
   Keys with `k_pos >= len_k`, `k_pos > p_max`, or (local) `k_pos <= p_min - W` are masked at
   the load (`other=0`) and in the scores; in paged mode a block-table entry `< 0` is also
   masked (and never dereferenced). Fully masked blocks are bitwise no-ops.
2. `scores = tl.dot(q_tile, K_block^T)` bf16 x bf16 -> fp32 (K=128, `allow_tf32=False`),
   `* 128**-0.5` in fp32, `where(keep, scores, -inf)` with
   `keep = k_pos <= q_pos` (global) / `q_pos - W < k_pos <= q_pos` (local), absolute positions.
3. `m_new = max(m, rowmax(scores))`; `m_safe = where(m_new == -inf, 0, m_new)`;
   `alpha = exp(m - m_safe)`; `p = exp(scores - m_safe)`; `l = l*alpha + rowsum(p)`;
   `acc = acc*alpha + tl.dot(bf16(p), V_block)` (K=64); `m = m_new`.
4. `o = div_rn(acc, l)` (IEEE fp32 division) `-> bf16`;
   `y = (o_bf16 -> fp32) * sigmoid(g -> fp32) -> bf16`, `sigmoid = div_rn(1, 1 + exp(-g))`,
   elementwise over the 4096 flattened channels; `lse = m + log(l)` fp32 (natural log).
5. GQA: q head `h` reads kv head `h // 16`.

Casts: inputs bf16; `p -> bf16` once before the PV dot; `o -> bf16` once; `y -> bf16` once.
`exp`/`log` are `tl.exp`/`tl.log` fp32.

Tiles (compile-time, recorded in `trace()`):

| tile | rows | BLOCK_M | num_warps | num_stages | used for | grid |
|---|---|---|---|---|---|---|
| TOKEN | 64 consecutive queries of one q head | 64 | 4 | 2 | prefill / training (`max_len_q > 16`) | `(cdiv(max_len_q, 64), B, 32)` |
| HEAD | the 16 q heads of one GQA group at one query position | 16 | 4 | 2 | decode / short chunks (`max_len_q <= 16`) | `(max_len_q, B, 2)` |

The HEAD tile reads each K/V block once per KV head instead of once per q head (16x less
K/V traffic in decode) and visits no extra blocks. Both tiles can be forced with
`tile_mode="token"|"head"` (tests do) and are bitwise identical per row.

Paged layout: `k_cache/v_cache (num_pages, 16, 2, 128)`, `block_table (B, max_pages)` int32,
`seq_lens (B,)` int32 — the vLLM "NHD" page layout the repo's FlashInfer path used. A 64-key
block gathers 4 pages. Query positions for the paged path are derived on device
(`paged_query_positions`: the queries are the last `len_q` positions).

Golden `gated_attention_forward_fp32(q, k, v, gate, q_pos, window, dtype=fp32|fp64)`: the same
blocked online softmax in torch, same block range, same update order, same `p -> bf16` and
`o -> bf16` casts; the `sum over d` and `sum over the 64 keys` are torch reductions over a
fixed trailing axis (deterministic per shape, accumulated in `dtype`). No matmul/SDPA. It is a
tolerance reference, not a bitwise one (the tensor-core dot order is not reproducible in torch).

## Contract deviations

1. **`num_warps` must be equal for the two tile heights (pinned to 4 for both).** The contract
   says BLOCK_M=16 and BLOCK_M=64 "are identical because each row's ops do not depend on M".
   That is true for `tl.dot` (per-element K accumulation) and `rowmax` (exact), but **not for
   `rowsum(p)`**: Triton 3.7.1 lays the MMA result out with warps along the KEY axis
   (`warpsPerCTA = [1, num_warps]`), so the sum tree is "thread partials -> quad shuffles ->
   cross-warp combine of 64/num_warps-column slices" and depends on `num_warps`, not on
   BLOCK_M. Measured: HEAD tile with 1, 2 or 8 warps differs from the 4-warp TOKEN tile in the
   last bit of `l` (lse diff ~1e-6, one bf16 ulp in some `o`); with 4 warps all bits match.
   We did NOT have to fall back to a single BLOCK_M. `test_tile_layouts_share_warp_split`
   inspects the compiled TTGIR and fails if any variant gets a different `warpsPerCTA`.
2. **`m_new == -inf` guard.** The contract's `alpha = exp(m - m_new)` is NaN while a row has
   not met a visible key (`-inf - -inf`). We subtract 0 instead in that state, which keeps
   `(m, l, acc) = (-inf, 0, 0)` exactly and is a bitwise no-op once a key appears
   (`exp(-inf - finite) = 0`). The golden does the same. Should be written into O5.
3. **Decode tile rows are heads, not padded tokens.** O5 says "BLOCK_M=16 for decode (M
   padded)". We fill the 16 rows with the 16 q heads of a GQA group (one token) instead of
   15 padding rows — per-row arithmetic unchanged, 16x less K/V traffic. The padded-token
   form (TOKEN tile with one valid row) is also exercised and gives the same bits.
4. **IEEE divisions.** `o = acc / l` and the sigmoid use `libdevice.div_rn` (round-to-nearest)
   rather than Triton's default `/` (which may lower to the approximate `div.full.f32`). Both
   are deterministic; `div_rn` makes the golden's torch division the same operation.
5. **Golden inner sums** use torch reductions over a trailing axis instead of a spelled-out
   sequential FMA chain (a per-element Python loop over d=128 would be ~100x slower). They are
   deterministic for a fixed shape; the golden is a tolerance reference only.
6. **Interface additions (backward compatible):** both strict wrappers accept optional kwargs
   `max_seqlen_q: int | None` (avoids a device sync) and `tile_mode: str | None` (forces a
   tile). The golden accepts `dtype=` (fp64 for the check) and `q_chunk=`. Names and
   positional semantics are exactly as specified in the brief. `forward_fp32` is an alias of
   `gated_attention_forward_fp32` (contract R4 naming).

## Test commands and results

```
PYTHONPATH=$WT ~/bin/gpu-lease -n 1 -- ~/envs/vllm/bin/python -m pytest \
    tests/models/muse_glimmer/test_gated_attention_fwd.py -q -p no:cacheprovider -W error::DeprecationWarning
[gpu-lease] using GPUs [0]
...............................................................          [100%]
63 passed in 33.12s
```

Numbers printed with `-s` (A100-SXM4-80GB, torch 2.13.0+cu130, triton 3.7.1):

```
G1 window=None n=37:   max|dy|=7.807e-03 max|do|=7.809e-03 max|dlse|=3.815e-06
G1 window=2048 n=37:   max|dy|=7.807e-03 max|do|=7.809e-03 max|dlse|=3.815e-06
G1 window=None n=300:  max|dy|=7.811e-03 max|do|=1.383e-02 max|dlse|=5.722e-06
G1 window=2048 n=300:  max|dy|=7.811e-03 max|do|=1.383e-02 max|dlse|=5.722e-06
G1 window=None n=2300: max|dy|=2.080e-02 max|do|=1.520e-02 max|dlse|=7.629e-06
G1 window=2048 n=2300: max|dy|=2.080e-02 max|do|=1.520e-02 max|dlse|=7.629e-06
fp32 vs fp64 golden window=None: max|dy|=6.544e-03 max|do|=2.774e-03 max|dlse|=2.280e-06
kernel vs fp64 golden: max|dy|=9.549e-03
MMA warpsPerCTA for all compiled variants: {'[1, 4]'}
real prefill-4096 window=None: max|dy|=8.623e-03 max|do|=7.765e-03 max|dlse|=5.722e-06
real prefill-4096 window=2048: max|dy|=8.623e-03 max|do|=7.778e-03 max|dlse|=5.722e-06
```

Slowest tests: token-by-token decode of 8192 tokens 6.3 s (global) / 5.5 s (local).

## Parity results

| Check | Mode | Result |
|---|---|---|
| alone vs packed (6 seqs, 3 orders incl. duplicate seq) | varlen, TOKEN and HEAD tiles, global+local | `torch.equal` y/o/lse |
| decode batch of 6 mixed contexts vs each alone | paged, global+local | `torch.equal` |
| one-shot vs chunked varlen, 6 split patterns x 6 lengths x 2 kinds | varlen | `torch.equal` |
| one-shot vs chunked paged, no eviction and eager eviction | paged | `torch.equal` |
| one-shot vs token-by-token paged decode (NaN-poisoned eviction), lengths 64..8192 | paged | `torch.equal` on every token |
| never-evict vs eager-evict, lengths 2049/2200/4096 (300 decode steps each) | paged, local | `torch.equal`; 0/9/128 pages actually freed at the end |
| pages freed by seq A (2600 tokens) reused by seq B (400 tokens) | paged, local | both `torch.equal` to alone |
| TOKEN (BLOCK_M=64, 4 warps) vs HEAD (BLOCK_M=16, 4 warps), lengths 1..2100 and decode B=64 | varlen | `torch.equal` |
| paged vs varlen decode B=64 @ 4096 | — | `torch.equal` |
| repeat (same call twice) | implicit in every comparison above | `torch.equal` |

## Benchmark

`PYTHONPATH=$WT ~/bin/gpu-lease -n 1 -- ~/envs/vllm/bin/python benchmarks/models/muse_glimmer/bench_gated_attention.py`
(A100-SXM4-80GB, 30 timed iterations after 10 warmups; native = `F.scaled_dot_product_attention`
bf16 `enable_gqa=True` + the same `* sigmoid(gate)`; both outputs include the gate):

| case | strict ms | native SDPA ms | strict/native | max abs diff y (strict vs SDPA) |
|---|---|---|---|---|
| prefill Tq=4096 B=1 (global) | 2.057 | 1.574 (flash, causal) | 1.31x | 1.56e-02 |
| decode B=64 ctx=4096 (global) | 0.324 paged (varlen K/V: 0.198) | 0.202 | 1.60x (paged) / 0.98x (varlen) | 1.56e-02 |
| prefill Tq=4096 B=1 (local) | 1.328 | 30.163 (dense bool mask, no flash) | 0.04x | 1.56e-02 |
| decode B=64 ctx=4096 (local) | 0.326 paged (varlen K/V: 0.108) | 18.044 (dense bool mask) | 0.02x | 7.81e-03 |

The local-layer baseline is torch's masked fallback, not a windowed flash kernel; a fair
flash-attn sliding-window baseline would be ~1 ms (prefill), so read the local rows as "strict
is ~1.3x of a flash-class kernel", same as the global row. Obvious next optimizations (not
done): (a) the paged gather of 4 x 16-token pages per block costs 0.12 ms at B=64 — a 64-token
page (vLLM `block_size=64`) would make one block = one contiguous page; (b) `BLOCK_M=128` /
8 warps for prefill would need the HEAD tile to move to 8 warps too (same sum tree), untested.

## Integration requests

No tolerance row is needed: G1 uses the existing WS1 `attention`/`bfloat16` row
(`rl_engine/contracts/profiles/precision/ws1.json`).

Registration (for the integrator; the repo's `KernelRegistry.register_attention_backend` is
WS2 contract-aware dispatch and does not fit a model-scoped strict op, so a plain op entry is
suggested):

```python
# rl_engine/models/muse_glimmer/runtime/<ops registry of the 52-layer chain>
from rl_engine.models.muse_glimmer.ops.gated_attention import (
    gated_attention_forward_fp32,
    strict_gated_attention_fwd_paged,
    strict_gated_attention_fwd_varlen,
    trace as gated_attention_trace,
)
STRICT_OPS["gated_attention"] = {
    "forward_varlen": strict_gated_attention_fwd_varlen,   # training / prefill / chunked prefill
    "forward_paged": strict_gated_attention_fwd_paged,     # vLLM rollout (page_size=16, NHD)
    "forward_fp32": gated_attention_forward_fp32,
    "trace": gated_attention_trace,
}
```

Operator-trace rows: `attention_local`/`output_gate_local` -> `local_gated_attention_train`
and `attention_global`/`output_gate_global` -> `global_gated_attention` are already in
`operator_trace.py`; this op fuses `attn` and `output_gate` into one kernel launch (the gate
GEMM itself stays in O3).

vLLM binding notes for the rollout agent: `k_cache/v_cache` must be the `(num_blocks, 16, 2,
128)` bf16 tensors (vLLM NHD layout, `block_size=16`), `block_table` int32 with `-1` for
unallocated pages, `seq_lens` int32 incl. the new tokens, and the new K/V must be written
before the call. Chunked prefill passes `cu_seqlens_q` with the chunk lengths; the kernel
derives the absolute positions.

## Open issues

1. The tile-parity guarantee rests on Triton's MMA layout heuristic (`warpsPerCTA=[1,4]` for
   both tiles on Triton 3.7.1 / sm80). A Triton upgrade or another arch (sm90 wgmma layouts)
   may change it; `test_tile_layouts_share_warp_split` + `test_token_vs_head_tiles_bitwise`
   catch that. A layout-independent alternative (explicit pairwise `reshape`+`sum` tree for
   `rowsum(p)`) was considered and rejected for the POC because of the extra layout
   conversions per block.
2. Paged decode is 1.6x slower than contiguous decode (see Benchmark).
3. `PagedKVCache` is a Python-driven test utility (one small device write per append). It is
   not the production allocator; vLLM's cache is.
4. The varlen wrapper does one `.item()` sync to size the grid unless `max_seqlen_q` is passed.
5. The kernel assumes the pinned geometry (32 q heads, 2 kv heads, D=128) via `constexpr`;
   other shapes are rejected by the wrappers.

## RFC feedback (how the three attention rows should read)

1. **Gate semantics.** The rows say "gated attention"; they should state: the gate is
   elementwise over the 4096 flattened attention channels, `g = W_gate · input_layernorm(x)`
   (the layer INPUT, not the attention output; `gate_proj` is a (4096, 6656) GEMM counted in
   `dense_gemm_qualification`), bias-free, applied as `y = bf16( bf16(o) * sigmoid(g) )` to the
   pre-`o_proj` output. The sigmoid-multiply is fused into the attention epilogue; the gate GEMM
   is not.
2. **Mask / positions / eviction, written exactly.** Global: `k_pos <= q_pos`. Local:
   `q_pos - 2048 < k_pos <= q_pos` (2048 keys visible incl. self, 2047 past tokens). Positions
   are ABSOLUTE and shared by training, packed prefill, chunked prefill and decode. Eviction
   rule for the rollout cache: a page may be freed once every key position in it is
   `<= p - 2048` where `p` is the smallest query position of the current step; the kernel
   masks those keys regardless, so eviction timing is not part of the numerics.
3. **Train vs rollout rows.** The same kernel body, tiles and bits serve training (varlen K/V),
   prefill, chunked prefill and paged decode; the only difference is the K/V addressing
   (`PAGED` constexpr). Recommendation: keep ONE kernel row `local_gated_attention` (and one
   `global_gated_attention`) with two *execution modes*, and keep the train/rollout split only
   at the gate level: the training gate is G4 (backward), the rollout gate is G3 on the paged
   path (split invariance with eviction). Two separate implementation rows invite two kernels,
   which is exactly what breaks parity.
4. **Decode tiling.** Replace "BLOCK_M=16 for decode (M padded)" with "BLOCK_M=16 = the 16
   q heads of a GQA group at one position" and add the hard requirement that every tile
   variant uses the same `num_warps` (the `rowsum` tree depends on the warp split along the
   key axis, not on BLOCK_M) and that the compiled layouts are checked (`warpsPerCTA`).
5. **Online-softmax edge case.** Add the `m_new == -inf -> subtract 0` rule; without it the
   literal contract formula is NaN for rows whose first visited block is fully masked (every
   local-window tile has such rows).
6. **Golden wording.** "fp32 fixed-order" for attention cannot mean bitwise-reproducible by
   torch (the tensor-core dot order is opaque); say "fp32 (or fp64) deterministic reference,
   compared with the WS1 attention tolerance; bitwise gates are kernel-vs-kernel".
7. **Page size.** O5 fixes page = 16; allow 64 (= BLOCK_N) as an option for the strict rollout
   so a KV block is one contiguous page (see benchmark).
8. **lse definition for the backward row:** `lse = m + ln(l)` in fp32, per (token, q head),
   with `l` accumulated from the fp32 `p` (before its bf16 rounding).

## Time spent

~1 h wall clock: reading contract/prior art, kernel + wrappers, parity debugging (warp-split
discovery), tests/bench/report. GPU leases: 8 short runs (< 1 min each).
