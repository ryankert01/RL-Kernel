# REPORT — WS2 context parallel: `cp_global_attention`, `cp_local_attention` (RFC #501)

## Wave A update — the O5 CP defect is resolved in the kernel (supersedes deviation 1 below)

- **Backward redesigned.** The query-chunk rule now lives inside
  `triton/gated_attention_bwd.py` (kernel id v2): dK/dV leaves per (canonical query chunk
  `c = [floor(cT/8), floor((c+1)T/8))`, Q-head group), chunk tree, then per KV head the tree over
  its 4 groups. A CP rank makes ONE kernel call on its rows against the gathered K/V; the chunks
  it does not own are exact zero leaves, so each group partial is its sub-tree root. Rank roots
  `(G, T, 128)` travel by a rank-ordered all_gather and are folded in the same tree, then the
  per-KV-head group tree, one bf16 rounding. The old per-chunk launches, the head-shard launcher
  `_strict_fwd_head_shard` and the labelled torch fallback are gone (the WS1 wrappers take
  `num_q_heads` / `num_kv_heads`; strict ops fail closed).
- **One-shot == CP1 == CP2, bitwise.** Real CP2 on two GPUs (torchrun, NCCL, this session,
  `check_cp_parity.py --backward --lengths 2048 2100 4096 8192`), every rank, global + local:
  y / o / lse / dq / dk / dv / dgate mismatches vs CP1 = 0, vs the one-shot kernel = 0, and the
  CP1 path vs the one-shot kernel = 0 (bit-pattern counts). One GPU (`gated_attention_bench
  parity`): `cp_gated_attention_backward` at CP1 and simulated CP2/4/8 ranks vs the one-shot
  kernel = 0 at T in {8, 300, 2048, 2100, 4096, 8192}. In v1 the one-shot kernel differed from
  the CP1 path in 1557 / 5540 / 20440 dK elements at T = 2048 / 4096 / 8192.
- **Tests (this session).** 2 GPUs: `tests/models/muse_glimmer/distributed/test_cp_*.py` ->
  CP2 forward + backward, global + local, pass; the CP4 and TP2 x CP2 cases skip (need 4 GPUs).
  The harness and `check_cp_parity.py` now count mismatches on bit patterns (`-0.0 != +0.0`)
  and report "CP1 vs one-shot" explicitly.
- **CP2 cost (T = 8192, A100 NVLink, NCCL all_gather_into_tensor):** AG of K+V 0.280 ms (global)
  / 0.343 ms (local) vs a 5.1-6.3 ms (global) / 3.0-3.2 ms (local) rank forward; CP2 fwd+bwd
  27 ms (global) / 13 ms (local) per rank at T = 8192.
- **Not run this session:** CP4 (pytest or torchrun), CP8, TP2 x CP2 on 4 GPUs; the backward at
  TP2 x CP2 (the CP backward accepts 16/1 shards; TP4/TP8 shards raise because their dK/dV are
  partials that need the cross-TP replica tree first).

The sections below are the original report; its deviation 1 ("O5 backward order is not
CP-invariant") is fixed by the kernel change above, and its integration request for head
shards is done.

STATUS: complete for the briefed scope on the REAL sibling kernels (forward
`ops/gated_attention.strict_gated_attention_fwd_varlen`, backward
`triton/gated_attention_bwd` partials, `ops/canonical_tree.combine_tree`). Not run before the
session closed: the torchrun matrix at CP4 (the pytest CP4 forward+backward gates DID pass) and
anything at CP8. Resume with:

```
~/bin/gpu-lease -n 4 --contiguous --pool 0-5 -- ~/envs/vllm/bin/python -m pytest \
    tests/models/muse_glimmer/distributed -q            # 23 tests, ~2 min, 4 GPUs
~/bin/gpu-lease -n 4 --contiguous --pool 0-5 -- ~/envs/vllm/bin/python -m torch.distributed.run \
    --nproc_per_node=4 rl_engine/models/muse_glimmer/runtime/check_cp_parity.py --backward
```

## Summary

| Gate | Result |
|---|---|
| CP2 == CP1 bitwise forward (y, o, lse), global + local, T in {2048, 4096, 8192} | PASS (pytest + torchrun) |
| CP4 == CP1 bitwise forward, same matrix | PASS (pytest) |
| CP2/CP4 == CP1 bitwise backward (dq, dk, dv, dgate) via autograd, global + local, 3 lengths | PASS |
| Repeat determinism of every CP call (`torch.equal`) | PASS |
| TP2 x CP2 (4 GPUs) == single-GPU TP1 x CP1 restricted to the rank's heads/rows, global + local, T in {4096, 8192} | PASS |
| zigzag / striped / ring / permuted-position layouts raise `CPLayoutError` before any collective | PASS |
| Window on ABSOLUTE positions (relative numbering provably differs) | PASS |
| Teeth: CP-oblivious relative positions on rank>0 differ from CP1 | PASS |
| O5 row property (rows alone with full K/V == full-run bits, aligned and unaligned splits) | PASS |

All 23 tests pass (`tests/models/muse_glimmer/distributed`, ~2 min on 4 A100s, run via
`gpu-lease -n 4`). Equality matrix (mismatch counts, torchrun CP2, real kernels; pytest CP4 is
the same matrix with the same zeros):

| layer | T | CP | y!= | o!= | lse!= | dq!= | dk!= | dv!= | dgate!= | fwd ms (local rows) | fwd+bwd ms |
|---|---|---|---|---|---|---|---|---|---|---|---|
| global | 2048 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1.2 | 5 |
| global | 4096 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2.0 | 10 |
| global | 8192 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 5.1 | 25 |
| local  | 2048 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1.1 | 5 |
| local  | 4096 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1.5 | 7 |
| local  | 8192 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2.3 | 12 |
| global/local | 2048/4096/8192 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | (pytest) | (pytest) |

Communication cost (CP2, T=8192, NCCL `all_gather_into_tensor` as transport, A100 NVLink):
all_gather of K+V = 4 MB in / 8 MB out per rank in **0.23 ms** (global) / 0.22 ms (local);
positions gather ~0.05 ms. Versus the 8192-row forward at 5.1 ms (global) the AG is ~4 %.

## Design

* Layout: only the #235 `allgather` layout; rank i of c owns tokens `[iT/c, (i+1)T/c)`.
  `CPAttentionPlan` rejects any other layout name at construction, and the forward rejects
  positions that are not the rank's contiguous chunk, then checks the GATHERED positions equal
  `arange(T)` (fail closed on zigzag/striped shards even if the plan says allgather).
* Forward: deterministic rank-ordered AG of K, V (one packed collective) and positions — pure
  copies, no arithmetic — then the strict kernel on the local queries against the full K/V with
  the local queries' ABSOLUTE positions and the window. Bitwise equality to CP1 follows from O5:
  a row's arithmetic is a function of its q, its absolute position and the K/V bits only; extra
  fully-masked key blocks are no-ops. Verified on the real kernel at tile-aligned and
  non-aligned splits.
* Gate: applied inside the kernel epilogue on the owner rank's rows. There is no `(out, lse)`
  merge on this layout — every rank finishes the complete softmax of its own queries.
* Backward: dq/dgate row-local (kernel). dK/dV use one canonical two-level fp32 tree: inner =
  the kernel's 8 Q-head-group partials `((0+1)+(2+3))+((4+5)+(6+7))`; outer = 8 contiguous
  canonical query chunks of the FULL sequence combined with the same tree. Rank i of c owns
  chunks `[8i/c, 8(i+1)/c)`, combines them locally (its subtree root) and the rank roots are
  exchanged by rank-ordered all_gather and combined locally in the same tree; one bf16 rounding.
  No NCCL reduce-scatter anywhere. CP1 evaluates the same 8-chunk tree, so CP-c == CP-1 by the
  subtree property (O3's rule applied to the query axis).
* Collective: `make_cp_collective` uses the CUDA-IPC fixed-tree collective when `rl_engine._C`
  exists, else `TorchDistributedDeterministicCollective` (all_gather_into_tensor transport,
  balanced rank tree) — this POC environment has no `_C`.
* TP x CP: `TPCPTopology` (tp fast, cp slow; CP groups {0,2},{1,3}); TP rank t owns Q heads
  `[16t,16t+16)`, KV head t, gate channels `[2048t, 2048t+2048)`; attention needs no TP
  collective. The WS1 wrapper pins 32/2 heads, so shards launch the SAME kernel body
  (`_gated_attention_fwd_kernel`, token tiles, same constants) with `H_Q=16/H_KV=1`
  (`_strict_fwd_head_shard`); bitwise against the single-GPU run.
* Fallback (labelled): fixed-shape pure-torch fp32 kernel following the O5 block order, used
  only when the siblings are absent; `trace()` reports `forward_is_fallback`/`backward_is_fallback`.

## Contract deviations / RFC feedback

1. **O5 backward dK/dV order is not CP-invariant as written.** "dK/dV accumulated over query
   blocks in ascending order" (one sequential fold) gives, at CP>1, a different association than
   CP1. Measured with the real Triton partials at T=2048 local: one-shot sequential dK vs the
   8-chunk tree differ (rel 1.6e-5; with the torch fallback 1.6e-7). The row text should become:
   "dK/dV are reduced over the query axis in 8 canonical contiguous query chunks (each chunk
   accumulated over its query blocks ascending) combined in the O3 tree at every CP including
   CP1; CP rank i of c owns chunks [8i/c, 8(i+1)/c) and the rank roots are combined by
   all_gather + the same tree". Today the WS1 one-shot `strict_gated_attention_bwd` at TP1 and
   this CP path at CP1 therefore differ in the last bits of dK/dV; pick one (recommend the chunk
   rule in the kernel: one program per (KV block, group, query chunk) or a chunk loop).
2. **"Apply the output gate after the CP merge" (RFC cp rows):** there is no merge on the
   allgather layout; the sentence should read "the gate is applied by the kernel epilogue on the
   owner rank's rows (O5); layouts that need an (out, lse) merge (zigzag/ring) have a second
   reduction order and are rejected (fail closed)".
3. **Zigzag fail-closed rule:** should be stated for both the plan name AND the token positions
   (a zigzag shard presented as "allgather" is caught by `pos_full != arange(T)`).
4. `cp_local_attention` row: state explicitly that the window is evaluated on absolute positions
   against the GATHERED K/V, so chunk boundaries need no halo exchange; at CP4/T=2048 the chunk
   (512) is smaller than W and every rank reads neighbours' keys (tested).
5. Communication: the RFC's cost model should count AG(K,V) = 2·T·256·2 B / cp per rank out
   plus AG(positions); at T=8192 this is 0.23 ms vs a 5 ms forward. The backward's rank-root
   exchange moves 2·T·2·128·4 B (16 MB at T=8192) — an all_to_all would cut it by cp×
   (obvious next optimization, not done).

## Integration requests

* WS1 `ops/gated_attention.strict_gated_attention_fwd_varlen`: accept head shards
  `(Tq, 32/tp, 128)` / `(Tk, 2/tp, 128)` (the kernel already is generic in H_Q/H_KV) so
  `cp_attention._strict_fwd_head_shard` can be deleted; same for the backward wrapper
  (`_check_inputs` pins 32/2), which is why TP2 x CP2 is forward-only here.
* The packed-varlen case (several sequences per CP chunk) is not implemented on the CP path:
  `gated_attention_rows` builds `cu_seqlens = [0, T]`; the trainer integration must pass the
  packed `cu_seqlens_k` of the gathered buffer.
* No registry/tolerance rows needed: parity gates are bitwise; the correctness oracle uses
  ws1.json attention bf16 (atol 5e-2 / rtol 2e-2) unchanged.
* Environment note: `~/envs/vllm` has an editable `rl_kernel` install whose finder resolves to a
  different checkout; `check_cp_parity.py` pins its own repo root on `sys.path` for torchrun.

## Files

* `rl_engine/models/muse_glimmer/runtime/cp_attention.py` — plan, topology, forward, backward,
  autograd Function, collective factory, head-shard launcher, fallback kernel, `trace()`.
* `rl_engine/models/muse_glimmer/runtime/check_cp_parity.py` — torchrun matrix + AG timing.
* `tests/models/muse_glimmer/distributed/{__init__.py,_cp_harness.py,test_cp_global_attention.py,
  test_cp_local_attention.py,test_cp_tp_topology.py}`.

## Test commands and results (tail)

```
$ gpu-lease -n 4 --contiguous --pool 0-5 -- python -m pytest tests/models/muse_glimmer/distributed -q -k "not tp2_cp2"
22 passed, 1 deselected in 74.20s
$ gpu-lease -n 4 --contiguous --pool 0-5 -- python -m pytest tests/models/muse_glimmer/distributed -q -k "tp2_cp2 or rows_alone or window_is or naive"
5 passed, 18 deselected in 19.64s
$ gpu-lease -n 2 --contiguous --pool 0-5 -- python -m torch.distributed.run --nproc_per_node=2 \
      rl_engine/models/muse_glimmer/runtime/check_cp_parity.py --backward
all bitwise: True  (5.9s)      # table above
```

## Open issues

* CP8 and the CP4 torchrun run were not executed (time); CP4 pytest passed.
* Backward at TP>1 is untested (WS1 backward wrapper pins 32/2 heads).
* dK/dV rank-root exchange uses all_gather (cp× the minimal traffic); all_to_all would be exact
  and cheaper.

## Time spent

~4 h including the two sibling merges.
