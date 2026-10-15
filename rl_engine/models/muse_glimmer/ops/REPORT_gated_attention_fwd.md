# REPORT — `gated_attention` forward (contract O5; RFC #501 rows `global_gated_attention`, `local_gated_attention_train`, `local_gated_attention_rollout`)

## Wave B update (review wave A2) — supersedes the sections below where they disagree

Raw evidence: `reports/muse_glimmer_poc/data/kernels_attention_raw/waveB/`; summary and per-issue
verdicts in `reports/muse_glimmer_poc/data/kernels_attention.json` (`wave_b`). A100-SXM4-80GB,
torch 2.13.0+cu130, Triton 3.7.1 (`~/envs/vllm`), one GPU at a time from pool 3,7.

**Forward bits unchanged.** `gated_attention_bench digest` (driver from this branch run against
both trees) on the merged base f91e8b6 and on this branch: **97/97 forward `(y, o, lse)` cases
and 14/14 backward cases identical**. The 97 = the 66 wave A cases + 31 new ones that exercise
every path touched here: TP shards 16/1, 8/1, 4/1 (varlen TOKEN and HEAD tiles, paged decode),
paged chunked prefill with queries that are only the tail of the sequence (TOKEN / HEAD / auto,
interleaved page ids), and local decode with evicted (`-1`, NaN-filled) pages.

**Fail closed on bad metadata (R1-R4).** The kernel now checks, before it touches any data,
`cu_seqlens_q` (and `cu_seqlens_k` on the varlen path): start at 0, non-decreasing, end at
`Tq` (`Tk`); grid coverage (`len_q` <= rows of grid axis 0, so an under-reported
`max_seqlen_q` is caught); paged: `len_q <= seq_lens <= max_pages * 16`. Per row (varlen):
`0 <= q_pos < len_k`. Per visible key (paged): its block-table entry is in `[0, num_pages)`. A
violation writes 0 to a status word that the wrapper asserts on (`torch._assert_async`, message
`METADATA_ASSERT_MSG`), a bad page is never dereferenced, and a program with bad offsets returns
without reading or writing. Host-side validation with a `ValueError` is used whenever the
wrapper has the data on the host anyway:

| path | how it is validated | added cost |
|---|---|---|
| varlen, `seq_meta=` given (new kwarg; autograd with a per-step meta) | meta = validated packing (identity + version check) + in-kernel checks | 4-byte fill + 1 assert launch, no sync |
| varlen, `max_seqlen_q=None` | `validate_varlen_packing`: one D2H copy of cu_q, cu_k, q_pos (it already synced for `max`) -> `ValueError` | the q_pos copy (`4*Tq` bytes) + vectorised CPU checks |
| varlen / paged, `max_seqlen_q` given (vLLM, bench, hot paths) | in-kernel checks -> device-side assert at the next sync | 4-byte fill + 1 assert launch, no sync |
| paged, `max_seqlen_q=None` | `validate_paged_packing` (cu_q, seq_lens on host) + in-kernel page checks | as before (one sync) |
| autograd `GatedAttentionFn` without meta, grads needed | builds the `SequenceMeta` in the forward (backward reuses it) | 2 D2H per layer instead of 1 + 2 |

The block-table rule changed meaning: a `-1` (or out-of-range id) under a key some query of the
call can see is now an error, not "masked". Evicted pages are still never read: they hold only
keys `<= p_min - W`, which are never `key_ok` (the vLLM sliding-window manager frees blocks below
`num_computed_tokens - W + 1`, i.e. strictly below that bound). The device-side assert poisons
the CUDA context (fail closed, no recovery); every probe of the review (p1, p1b, p2-p8) is a test
(`test_host_validation_*`, `test_device_checks_fail_closed[...]` in subprocesses,
`test_seq_meta_path_validates_without_sync`); 26 new tests fail on f91e8b6 and pass here
(`kernels_attention_raw/waveB/before_tests.txt`). Outputs are still `torch.empty`: with the
coverage check no row can stay unwritten without the assert firing, so a NaN pre-fill (3 extra
memsets) is not needed.

**Paged positions in the kernel (R6).** `pos = seq_lens[b] - len_q + t` is computed in the
kernel; `paged_query_positions` (6 small torch ops, ~0.19 ms host time per call) is off the hot
path (kept as a helper). Paged decode, full wrapper (ms, CUDA events, 200 iterations,
`kernels_attention_raw/waveB/decode_overhead_new.json`; before = review probe at e13ea68):

| ctx 4096 | before | after | kernel launch alone (after) |
|---|---|---|---|
| B=1 global | 0.321 | 0.186 | 0.178 |
| B=1 local | 0.316 | 0.110 | 0.084 |
| B=64 global | 0.323 | 0.231 | 0.225 |
| B=64 local | 0.320 | 0.133 | 0.126 |

So the earlier decode analysis below ("the paged gather of 4 x 16-token pages costs 0.12 ms",
"next: 64-token pages") was wrong: the cost was host-side position arithmetic. The remaining
wrapper overhead is 0.007-0.026 ms.

**Timing (R3, R6).** The bench driver now times strict / SDPA / vLLM FlashAttention-2
(`vllm.vllm_flash_attn.flash_attn_varlen_func`; causal, `window_size=[2047, 0]` on local layers;
on the same paged cache for decode) in 5 alternating rounds x 30 iterations and stores the raw
per-round series; each baseline's output is checked against strict `y` (max |dy| 0.004-0.023,
bf16 rounding). Local-layer SDPA rows (dense bool mask) and local SDPA backward rows (full causal,
no window) are stored with `ratio: null` + `baseline_not_comparable`; FA2 is the comparable local
baseline. Paired runs, same GPU, one after the other (`bench_base_fwd.json` with the f91e8b6
wrappers, `bench_new_fwd.json` with this branch), median ms:

| case | f91e8b6 strict | this branch strict [range] | SDPA | FA2 (vLLM) | strict/SDPA | strict/FA2 |
|---|---|---|---|---|---|---|
| prefill T=512 global | 0.228 | 0.140 [0.138, 0.149] | 0.139 | 0.147 | 1.00 | 0.95 |
| prefill T=1024 global | 0.227 | 0.232 [0.231, 0.233] | 0.250 | 0.228 | 0.93 | 1.02 |
| prefill T=2048 global | 0.527 | 0.534 [0.532, 0.535] | 0.535 | 0.488 | 1.00 | 1.09 |
| prefill T=4096 global | 1.705 | 1.715 [1.714, 1.717] | 1.353 | 1.228 | 1.27 | 1.40 |
| prefill T=8192 global | 6.074 | 6.084 [6.082, 6.091] | 3.949 | 3.591 | 1.54 | 1.69 |
| prefill T=512 local | 0.104 | 0.141 [0.138, 0.155] | 0.712 (dense mask) | 0.146 | n/c | 0.97 |
| prefill T=1024 local | 0.193 | 0.315 [0.202, 0.381] | 2.157 (dense mask) | 0.196 | n/c | 1.61 |
| prefill T=2048 local | 0.542 | 0.559 [0.558, 0.563] | 7.576 (dense mask) | 0.542 | n/c | 1.03 |
| prefill T=4096 local | 1.327 | 1.369 [1.368, 1.371] | 30.111 (dense mask) | 1.162 | n/c | 1.18 |
| prefill T=8192 local | 2.910 | 2.999 [2.995, 3.002] | 121.820 (dense mask) | 2.392 | n/c | 1.25 |
| paged decode ctx 4096 B=1 global | 0.329 | 0.157 [0.156, 0.159] | 0.134 | 0.192 | 1.17 | 0.82 |
| paged decode ctx 4096 B=16 global | 0.321 | 0.188 [0.186, 0.347] | 0.103 | 0.180 | 1.82 | 1.05 |
| paged decode ctx 4096 B=64 global | 0.338 | 0.278 [0.246, 0.422] | 0.214 | 0.239 | 1.30 | 1.17 |
| paged decode ctx 4096 B=256 global | 0.783 | 0.676 [0.675, 0.833] | 0.788 | 0.786 | 0.86 | 0.86 |
| paged decode ctx 4096 B=1 local | 0.324 | 0.137 [0.136, 0.139] | 0.485 (dense mask) | 0.146 | n/c | 0.94 |
| paged decode ctx 4096 B=16 local | 0.316 | 0.139 [0.138, 0.240] | 4.570 (dense mask) | 0.251 | n/c | 0.55 |
| paged decode ctx 4096 B=64 local | 0.331 | 0.144 [0.139, 0.239] | 18.064 (dense mask) | 0.878 | n/c | 0.16 |
| paged decode ctx 4096 B=256 local | 0.430 | 0.359 [0.358, 0.379] | 71.706 (dense mask) | 5.039 | n/c | 0.07 |

n/c = not comparable. Readings: decode is 1.16-2.4x faster than before (positions in the
kernel); prefill at T >= 2048 moved by +0.2..+3.2 % (the in-kernel checks, local layers the
most); the T <= 1024 rows are launch-bound and noisy in this run (the FA2 column jumps in the
same rounds; the earlier full run of the same code, `bench_waveB.json`, gave 0.114 / 0.232 /
0.113 / 0.200 ms for 512 global / 1024 global / 512 local / 1024 local, within ~0.01 ms of
f91e8b6, which is about the cost of the two extra launches);
prefill at T = 4096 is 1.27x SDPA and 1.40x FA2 on global layers and 1.18x FA2 on local layers. The T = 4096 "2.06 ms vs 1.70 ms" puzzle of wave A
does not reproduce with alternating rounds (1.705 / 1.715 ms; TP1 row of the TP series 1.711
and 1.751 ms in two runs). FA2's windowed paged decode is slow at B >= 16 in this vLLM build
(0.88 ms at B=64 vs 0.24 ms global); not investigated, so the local decode FA2 ratios say little.
The decode SDPA baseline reads contiguous K/V (not the paged cache). Run-to-run variation on
this node is up to ~20 % for some rows (two full runs recorded in the JSON): the backward and
TP series are single series per run, and the backward wrapper syncs inside every call.

**Compiled arithmetic (R5).** PTX/TTGIR of the compiled variants (Triton 3.7.1, sm80,
`enable_fp_fusion` default True): `tl.exp` = `ex2.approx.f32(x * log2 e)` (the exp2 form),
`tl.log` = libdevice `__nv_logf` with FTZ, `libdevice.div_rn` = `div.rn.ftz.f32` (subnormals
flushed), `l*alpha + rowsum(p)` = one `fma.rn.f32`, and the PV `tl.dot` accumulates into
`acc*alpha` inside the MMA. Recorded in `trace()["compiled_arithmetic"]` and pinned by
`test_trace_matches_compiled_ptx`. Contract deviation 4 below ("div_rn makes the golden's torch
division the same operation") is withdrawn: the golden (torch.exp, IEEE division without FTZ,
separate mul/add) is a tolerance reference; all bitwise gates are kernel vs kernel, and every
variant compiles the same sequence. Not changed to `enable_fp_fusion=False` because that would
change the forward bits (the task pins them).

**Window values (R7): not pinned.** The wrappers keep accepting any positive window: the
tiny-model tests of other rows run the strict kernels with windows 64 / 100 / 40
(`test_megatron_strict_binding.py`, `reference/weights.py`, `test_ablation.py`). The 30B
bindings must pass exactly 2048 (`trace()["window_values"]`); request to the integrator: assert
`window == 2048` in the vLLM / Megatron bindings for the 30B config.

**Tests (R8, R9).** `test_tile_layouts_share_warp_split` now compiles all 32 launched variants
(4 head geometries x TOKEN/HEAD x varlen/paged x global/local) before scanning the TTGIR: one MMA
layout `warpsPerCTA = [1, 4]`. Final run: 223 passed, 1 skipped over the three owned test files
(101 / 75 / 48 tests; the skip needs two GPUs); details in "Test commands" below.

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
gate; the local SDPA baseline is a dense bool mask, not a windowed flash kernel, so local ratios
are n/c = not comparable; superseded by the wave B table above, which adds FA2):

| case | strict ms | SDPA ms | strict/SDPA |
|---|---|---|---|
| prefill T=512 global | 0.105 | 0.134 | 0.79 |
| prefill T=1024 global | 0.226 | 0.250 | 0.90 |
| prefill T=2048 global | 0.636 | 0.612 | 1.04 |
| prefill T=4096 global | 2.058 | 1.570 | 1.31 |
| prefill T=8192 global | 6.051 | 3.940 | 1.54 |
| prefill T=4096 local | 1.326 | 30.152 (masked) | n/c |
| prefill T=8192 local | 2.903 | 121.923 (masked) | n/c |
| paged decode B=1 ctx 4096 global | 0.334 | 0.101 | 3.31 |
| paged decode B=16 ctx 4096 global | 0.414 | 0.105 | 3.96 |
| paged decode B=64 ctx 4096 global | 0.338 | 0.203 | 1.67 |
| paged decode B=256 ctx 4096 global | 0.829 | 0.927 | 0.89 |
| paged decode B=64 ctx 4096 local | 0.350 | 18.044 (masked) | n/c |

Per-rank prefill forward at T = 4096 (global / local, ms): TP1 32/2 1.700 / 1.325, TP2 16/1
0.939 / 0.710, TP4 8/1 0.673 / 0.480, TP8 4/1 0.430 / 0.282. (The TP1 global call measured
1.70 ms in this series and 2.06 ms in the prefill series of the same run, same inputs; with
alternating rounds in wave B the prefill series gives 1.705-1.715 ms, so the 2.06 ms was the
measurement position.) Decode at small batch was 3-4x SDPA; wave B showed that most of it was
the host-side position computation, not SM occupancy (see the wave B update).

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
| Benchmark (A100-80GB) | (wave A numbers; wave B table at the top supersedes them) prefill 4096: 2.06 ms vs SDPA-flash 1.57 ms (1.31x). Decode B=64@4096: paged 0.32 ms / varlen 0.20 ms vs SDPA 0.20 ms (1.6x / 1.0x). Local layer: SDPA has no window kernel (dense bool mask, 30 ms / 18 ms), the ratios are not comparable. |

63 tests, 33 s on one A100 (wave A count; wave B: 101 tests in this file, see Test commands).

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
   the load (`other=0`) and in the scores. In paged mode every remaining (visible) key's
   block-table entry must lie in `[0, num_pages)`: since wave B a `-1` or out-of-range id there
   fails closed (status word + device-side assert, the page is never dereferenced); it is no
   longer treated as masked. Evicted pages only hold invisible keys and are never read. Fully
   masked blocks are bitwise no-ops.
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
block gathers 4 pages. Query positions for the paged path are derived in the kernel since wave B
(`seq_lens[b] - len_q + t`: the queries are the last `len_q` positions; before, the wrapper
called `paged_query_positions`, ~0.19 ms of host time per call).

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
4. **Round-to-nearest divisions.** `o = acc / l` and the sigmoid use `libdevice.div_rn`
   rather than Triton's default `/` (which may lower to the approximate `div.full.f32`). The
   compiled instruction is `div.rn.ftz.f32` (subnormals flushed), and `tl.exp` is the exp2 form,
   so the golden's torch division/exp are NOT the same operations (corrected in wave B; the
   golden is a tolerance reference, see `trace()["compiled_arithmetic"]`).
5. **Golden inner sums** use torch reductions over a trailing axis instead of a spelled-out
   sequential FMA chain (a per-element Python loop over d=128 would be ~100x slower). They are
   deterministic for a fixed shape; the golden is a tolerance reference only.
6. **Interface additions (backward compatible):** both strict wrappers accept optional kwargs
   `max_seqlen_q: int | None` (avoids a device sync) and `tile_mode: str | None` (forces a
   tile). The golden accepts `dtype=` (fp64 for the check) and `q_chunk=`. Names and
   positional semantics are exactly as specified in the brief. `forward_fp32` is an alias of
   `gated_attention_forward_fp32` (contract R4 naming).

## Test commands and results

Wave B (final code; log `reports/muse_glimmer_poc/data/kernels_attention_raw/waveB/tests_owned.log.txt`):

```
PYTHONPATH=$WT ~/bin/gpu-lease -n 1 --pool 3,7 -- ~/envs/vllm/bin/python -m pytest -q -p no:cacheprovider \
    tests/models/muse_glimmer/test_gated_attention_fwd.py tests/models/muse_glimmer/test_gated_attention_bwd.py \
    tests/models/muse_glimmer/test_norms.py -rs
SKIPPED [1] tests/models/muse_glimmer/test_norms.py:164: needs two visible GPUs
223 passed, 1 skipped, 1 warning in 107.87s (0:01:47)
```

(101 tests in test_gated_attention_fwd.py, 75 in test_gated_attention_bwd.py, 48 in test_norms.py.)
Whole single-GPU Muse-Glimmer suite in the vLLM env at the same code (`full_suite_vllm.log.txt`):
`pytest tests/models/muse_glimmer --ignore=tests/models/muse_glimmer/distributed` -> 675 passed,
8 skipped (2-GPU, checkpoint and E2E gates). Megatron env (`megatron_env_tests.log.txt`):
`test_megatron_strict_binding.py test_strict_model_tiny.py` -> 35 passed, 2 skipped. The
distributed tests were NOT re-run (one GPU at a time in this wave).

Wave A (historical):

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
done): (a) [withdrawn in wave B: the 0.12 ms paged-vs-varlen decode gap was the host-side
`paged_query_positions`, now gone, not the 4 x 16-token page gather]; (b) `BLOCK_M=128` /
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
unallocated pages (since wave B only where no query of the call can see the page: beyond
`seq_lens`, or evicted below the window; a `-1` under a visible key fails closed), `seq_lens`
int32 incl. the new tokens, and the new K/V must be written before the call. Chunked prefill passes `cu_seqlens_q` with the chunk lengths; the kernel
derives the absolute positions.

## Open issues

1. The tile-parity guarantee rests on Triton's MMA layout heuristic (`warpsPerCTA=[1,4]` for
   both tiles on Triton 3.7.1 / sm80). A Triton upgrade or another arch (sm90 wgmma layouts)
   may change it; `test_tile_layouts_share_warp_split` + `test_token_vs_head_tiles_bitwise`
   catch that. A layout-independent alternative (explicit pairwise `reshape`+`sum` tree for
   `rowsum(p)`) was considered and rejected for the POC because of the extra layout
   conversions per block.
2. Paged decode was 1.6x slower than contiguous decode (see Benchmark); fixed in wave B (the
   cost was the host-side position computation).
3. `PagedKVCache` is a Python-driven test utility (one small device write per append). It is
   not the production allocator; vLLM's cache is.
4. The varlen wrapper syncs once (host validation of the packing, wave B) unless `max_seqlen_q`
   or a `SequenceMeta` is passed; then the kernel validates on the device (no sync).
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
7. **Page size.** O5 fixes page = 16. (Wave A suggested allowing 64 so a KV block is one
   contiguous page; withdrawn in wave B: the measured paged-decode overhead was host-side
   position arithmetic, not the page gather. Page 16 is fine.)
9. **exp2 (wave B, review R5).** R2 says "no exp2 tricks unless golden and kernel use them
   identically", but Triton's fp32 `tl.exp` IS the exp2 form (`ex2.approx.f32(x * log2 e)`) and
   `libdevice.div_rn` compiles to `div.rn.ftz.f32`. R2 should say: kernels may use `tl.exp` /
   `div_rn` as compiled by the pinned Triton; bitwise gates are kernel vs kernel; goldens are
   tolerance references. If a bit-exact torch reference were ever required, the kernels would
   need `enable_fp_fusion=False`, an explicit IEEE exp, and non-FTZ division (the norms already
   use `enable_fp_fusion=False` and `div_rn`).
8. **lse definition for the backward row:** `lse = m + ln(l)` in fp32, per (token, q head),
   with `l` accumulated from the fp32 `p` (before its bf16 rounding).

## Time spent

~1 h wall clock: reading contract/prior art, kernel + wrappers, parity debugging (warp-split
discovery), tests/bench/report. GPU leases: 8 short runs (< 1 min each).
