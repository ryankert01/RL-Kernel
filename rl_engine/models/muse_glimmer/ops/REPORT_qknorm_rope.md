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

## Wave B review follow-ups (2026-10-15, issues kernels_rope:R1..R12)

Code at `39053b2` (the later commits of this row only add scripts, reports and data);
measurements at `615ee61`; A100-SXM4-80GB, Triton 3.7.1, torch 2.13.0 cu130 (vLLM env) and
cu129 (Megatron env). Machine-readable: `reports/muse_glimmer_poc/data/kernels_rope.json` →
`wave_b` (verdicts, the per-test before/after table with the reason each test failed before,
the behavioural probes on the old and the new tree) and `experiments.bench_graph`.

**In short:** the adversarial review was right that two wave A findings (qknorm_rope:1 and :4)
were reported as fixed while the failure still happened on every production call path. Both
are now enforced in the code this row owns and are integration requests where it does not; the
verdicts in the wave A table below are corrected. No output bit changes.

Behaviour before (tree of `f91e8b6`) and after, from the SAME probe script
(`scripts/rope/probe_wave_b.py`, public API only):

| id | verdict | change | before (f91e8b6) | after |
|---|---|---|---|---|
| R1 (qknorm_rope:4 still open) | fixed where owned; call sites = integration patch 2 | `local=None` is counted (`runtime_provenance()["layer_kind_calls"]`), warns once per process, and raises `ValueError` when `MUSEGLIMMER_O4_REQUIRE_LAYER_KIND=1` or `require_explicit_layer_kind(True)`; the reference model passes `local=layer.is_local` | `cos=None` on a local layer with the env var set: no error, 15,994,827 / 16,777,216 q̂ differ from the RoPE path | env var set: `ValueError`; unset: same silent NoPE (backward compatible) plus one `RuntimeWarning` |
| R2 (qknorm_rope:1 only fixed in one builder) | fixed for every strict builder path; fallback builders and runtime digest logging = integration patches 1/3 | `rope_table()` (the builder that the vLLM and Megatron bindings and the reference bind) now returns copies of the rows of the ONE cached canonical table per device instead of building a private table; `reference/golden_ops.rope_table` delegates to it | `golden_ops.rope_table(device="cpu")` accepted; `rope_table` does not touch the canonical cache | both raise on CPU; `rope_table` reads the canonical cache. The op itself still cannot tell where a table came from |
| R3 | fixed | build records the tensors' version counters; `canonical_rope_table`, `rope_rows`, `rope_table`, `rope_table_sha256` raise after an in-place change (no device sync); `verify=True` re-hashes (no memo); the digest memo is keyed by the version counters | in-place change not detected; after a `.data` write `verify=True` passed and the logged digest still equalled the pin (stale memo) | in-place change raises; `.data` write (bypasses the counters, documented) → `verify=True` raises and the logged digest is the actual one |
| R4 | fixed | the first build inside CUDA-graph capture raises and caches nothing (runtimes build at init with `canonical_rope_table(dev, verify=True)`); the table is built as a normal tensor even under `inference_mode`, so the counters exist | build recorded, not executed: the cached table is wrong until the first replay (an eager 64-position gather right after capture: all 16,384 cos/sin elements wrong); table built under `inference_mode` is an inference tensor | `RuntimeError` ("must be built before CUDA-graph capture"), nothing cached; normal tensor |
| R5 | fixed | all probe / benchmark / JSON-builder scripts committed under `reports/muse_glimmer_poc/scripts/rope/`, no hard-coded paths; `run_rope_measurements.sh <out> [<tree>]` reruns them | – | – |
| R6 | fixed | tests of the negative `verify` paths: wrong pinned digest, in-place change, `.data` write | – | – |
| R7 | fixed | `test_compiled_kernels_use_int64_row_offsets` checks the compiled TTIR (program id sign-extended to i64; i64 row offsets of the Q and K tiles; i64 table row base), no 20 GiB allocation; `tests_new.per_test[*].before_failure_kind` says why each wave A test failed before | 6a61f6e kernel: 4/4 cases fail (`%t not extended`) | pass |
| R8 | fixed | `strict_kernels._fb_rope_table` added to the builder-identity test (all 4 builders) | – | – |
| R9 | fixed | `tests_summary_lines` names exactly what ran; `findings[2].verdict = coverage_gap_test_added` | – | – |
| R10 | fixed | `bench_series.plot`: host-load-confounded `before_*` series removed, GBps only for T ≥ 8192; new `experiments.bench_graph` (CUDA-graph replay = device time, old and new kernel interleaved) | – | table below |
| R11 | fixed | `trace()` stays static and says so (`scope`); `rope_table.sha256` → `sha256_pinned`; `runtime_provenance()` reports the digest of the table this process built, whether it equals the pin, and the layer-kind call counts | – | – |
| R12 | fixed | Design usage line (below) shows the contract form | – | – |

Tests. 16 wave B test cases (13 new, 3 changed; `wave_b.tests.per_test`): on the old code 14
fail, 11 of them on behaviour (did not raise / did not warn / wrong result / TTIR shows int32)
and 3 because a new name or trace field did not exist yet; 2 pass before and after (the
`verify` mismatch path and the canonical-table identity, coverage only). The wave A table is
re-labelled the same way: of its 18 before-failures only 5 are behavioural (CPU build, three
dtype checks, the T = 524,352 crash); 13 failed because a new name or trace field was missing.
After: `test_qknorm_rope.py` + `test_strict_model_tiny.py` 113 passed (vLLM env);
`test_qknorm_rope.py` + `test_megatron_strict_binding.py` + `test_strict_model_tiny.py`
129 passed, 2 skipped (Megatron env; the lm_head TP test, renamed `test_lm_head_tiles_tp_equals_tp1_and_reference` by its owner, passes); full single-GPU
`tests/models/muse_glimmer` suite (vLLM env, `--ignore distributed`, at 3404cf9 = the code of
39053b2 minus a test-fixture tweak) 653 passed, 8 skipped, 0 failed.

Device time per launch (CUDA-graph replay, 20 launches per graph, median of 5 interleaved
rounds; `experiments.bench_graph`; local layers, µs). Old = 6a61f6e kernel (int32 offsets),
new = current. Bits equal at every T, both layer kinds, forward and backward.

| T | fwd old | fwd new | bwd old | bwd new | eager launcher, new fwd | fwd GB/s (T ≥ 8192) |
|---|---|---|---|---|---|---|
| 1 | 3.3 | 3.3 | 4.4 | 4.4 | 42.9 |  |
| 16 | 3.5 | 3.5 | 4.6 | 4.6 | 42.7 |  |
| 64 | 3.6 | 3.7 | 5.0 | 4.9 | 42.1 |  |
| 256 | 5.4 | 5.5 | 8.0 | 7.9 | 42.2 |  |
| 1024 | 10.9 | 11.0 | 16.8 | 16.7 | 43.6 |  |
| 2048 | 28.5 | 28.2 | 40.6 | 40.4 | 42.5 |  |
| 4096 | 52.1 | 51.7 | 73.3 | 73.1 | 54.3 |  |
| 8192 | 96.8 | 96.6 | 138.6 | 138.0 | 99.6 | 1563 |
| 16384 | 187.0 | 187.3 | 269.4 | 268.1 | 190.2 | 1612 |
| 32768 | 367.8 | 368.3 | 526.9 | 524.5 | 371.9 | 1640 |

Below T ≈ 2048 the eager number (~42 µs) is the host launch path, not the kernel (3-28 µs
on the device); at T = 4096 the device time (52 µs) happens to equal that floor, which is why
the wave A text called T = 4096 "GPU-bound" without evidence. The int64 offsets cost nothing
measurable.

Public API: backward compatible. New names are additive (`runtime_provenance`,
`require_explicit_layer_kind`, `layer_kind_required`, `LAYER_KIND_ENV`); `local=None` still works
by default. Behaviour changes for inputs no production caller uses: `rope_table` rejects
positions outside `[0, 131072)` (it used to evaluate the formula for any int; one test that fed
negative positions was moved to valid ones), and the first canonical-table build inside
CUDA-graph capture raises. `trace()["rope_table"]["sha256"]` was renamed `sha256_pinned` (no
reader in the repo).

How to reproduce (wave B):
```
git archive f91e8b6 | tar -x -C <old>        # the tree before the wave B fixes
~/bin/gpu-lease -n 1 -- env PYTHONPATH=<old> ~/envs/vllm/bin/python \
    reports/muse_glimmer_poc/scripts/rope/probe_wave_b.py <out>/probe_wave_b_before.json
~/bin/gpu-lease -n 1 -- env PYTHONPATH=$PWD ~/envs/vllm/bin/python \
    reports/muse_glimmer_poc/scripts/rope/probe_wave_b.py <out>/probe_wave_b_after.json
git show 6a61f6e:rl_engine/models/muse_glimmer/triton/qknorm_rope.py > <out>/old_kernel.py
~/bin/gpu-lease -n 1 -- env PYTHONPATH=$PWD ~/envs/vllm/bin/python \
    reports/muse_glimmer_poc/scripts/rope/bench_graph.py <out>/bench_graph.json --old-kernel <out>/old_kernel.py
cp tests/models/muse_glimmer/test_{qknorm_rope,strict_model_tiny}.py <old>/tests/models/muse_glimmer/
(cd <old> && ~/bin/gpu-lease -n 1 -- env PYTHONPATH=<old> ~/envs/vllm/bin/python -m pytest -p no:cacheprovider \
    tests/models/muse_glimmer/test_qknorm_rope.py tests/models/muse_glimmer/test_strict_model_tiny.py -q -rfE)
    # -> 13 failed, 100 passed (before); the same files at HEAD: 113 passed
~/bin/gpu-lease -n 1 -- env PYTHONPATH=$PWD ~/envs/megatron/bin/python -m pytest -p no:cacheprovider \
    tests/models/muse_glimmer/test_qknorm_rope.py tests/models/muse_glimmer/test_megatron_strict_binding.py \
    tests/models/muse_glimmer/test_strict_model_tiny.py -q -rA      # -> 129 passed, 2 skipped
python reports/muse_glimmer_poc/scripts/rope/update_json_wave_b.py --json reports/muse_glimmer_poc/data/kernels_rope.json ...
```
(the exact argument list is in the script's docstring; `run_rope_measurements.sh` reruns the
wave A probes and benchmarks as well).

### Wave B: integration requests (other owners' files; exact patches)

Line numbers are for f91e8b6; the bindings are being edited by other agents, so match by context.

1. **Fallback O4 functions accept `local=` and the fallback table builders stop building
   private tables** (`runtime/fallback_ops.py`, `runtime/strict_kernels.py`; kernels_rope:R1/R2).
   ```python
   # runtime/fallback_ops.py: replace the body of rope_table (line ~278)
   def rope_table(positions, *, theta, dim, device):
       from rl_engine.models.muse_glimmer.ops.qknorm_rope import rope_table as _canonical_rows
       return _canonical_rows(positions, theta=float(theta), dim=int(dim), device=device)

   # runtime/fallback_ops.py::qknorm_scaled_rope (line ~305) and
   # runtime/strict_kernels.py::_fb_qknorm_scaled_rope (line ~345): add the keyword and check
       *, eps: float, qk_scale: float, local: bool | None = None,
   ) -> tuple[torch.Tensor, torch.Tensor]:
       if local is not None and bool(local) != (cos is not None):
           raise ValueError(f"local={local} but cos/sin are {'None' if cos is None else 'given'}")

   # runtime/strict_kernels.py::_fb_rope_table (line ~331): same delegation as fallback_ops
   def _fb_rope_table(positions, *, theta, dim, device):
       from rl_engine.models.muse_glimmer.ops.qknorm_rope import rope_table as _canonical_rows
       return _canonical_rows(positions, theta=float(theta), dim=int(dim), device=device)
   ```
   Both builders are bitwise identical to the canonical table on CUDA today
   (`test_other_repo_table_builders_are_bitwise_identical_on_cuda`), so no bit changes; a CPU
   device then raises instead of building a table that differs in ~19 % of its elements.
2. **Every call site states the layer kind** (kernels_rope:R1), after item 1:
   - `runtime/vllm_strict.py` (attention forward, `ops.qknorm_scaled_rope(q_heads, k_heads, cos,
     sin, eps=..., qk_scale=...)`, line ~943): add `local=bool(self.use_rope),`
   - `runtime/megatron_strict.py::_layer` (line ~556/559): sliding branch `local=True`, global
     branch `local=False`
   - `runtime/strict_ops.py::probe_rope` (line ~236/238): `local=True` / `local=False`
   - `runtime/vllm_tp_sim.py` (line ~158): `local=True`
   - `ablation/ablation_strict.py` (line ~134): `local=layer.is_local`
   - then the E2E harness exports `MUSEGLIMMER_O4_REQUIRE_LAYER_KIND=1` to both runtimes
     (rollout workers and trainer ranks): any remaining implicit call raises instead of
     silently running NoPE on a sliding layer. `qknorm_rope.runtime_provenance()
     ["layer_kind_calls"]["implicit_local"/"implicit_global"]` must be 0 in a run log.
3. **Table identity at runtime** (kernels_rope:R2/R4):
   - `runtime/vllm_strict.py`: where `state.cfg["rope_table_positions"]` is set (worker init,
     line ~708, i.e. before warm-up and CUDA-graph capture), add
     ```python
     from rl_engine.models.muse_glimmer.ops import qknorm_rope as _o4
     if state.ops.is_strict("rope_table"):
         dev = torch.device("cuda", torch.cuda.current_device())
         _o4.canonical_rope_table(dev, verify=True)   # builds once, raises on a digest mismatch
         state.cfg["rope_table_sha256"] = _o4.rope_table_sha256(dev)
     ```
     `_rope_for` may keep `state.ops.rope_table(arange(max_pos))`: it now returns copies of the
     canonical rows (same bits).
   - `runtime/megatron_strict.py` (trainer init, before the first forward): the same two calls
     with `self.device`, and record `rope_table_sha256` in the provenance.
   - E2E orchestrator: compare the two runtimes' `rope_table_sha256` (or
     `runtime_provenance()["rope_table_sha256_built"]`) and fail the run if they differ.

## Review follow-ups (2026-10-15, findings qknorm_rope:1..5)

Branch `worktree-wf_7b064961-36f-6`, measured before at `6a61f6e` (POC head) and after at
`9b2317b` (code; later commits only add this report and the JSON); A100-SXM4-80GB, Triton 3.7.1, torch 2.13.0 cu130 (vLLM env) and cu129 (Megatron
env). Machine-readable results, including the per-bucket, per-shard, per-test and per-T series
for plotting: `reports/muse_glimmer_poc/data/kernels_rope.json`.

**In short:** all five findings were reproduced on the GPU. Four were real bugs; wave A made
qknorm_rope:2 and :5 fail closed, but qknorm_rope:1 and :4 only in this module's own API (the
wave B review above corrects those two verdicts). The fifth (TP shards) was only a missing
test: the kernel was already bit-exact, and the test is now added. No output bit changes for any shape. The
sha256 of the forward and backward outputs at T=4096 (local, global, long-context positions) is
identical before and after the fix, and identical between the two runtimes' envs.

| id | finding | verdict | what we observed before | fix | after |
|---|---|---|---|---|---|
| qknorm_rope:1 | O4 table identity | real; wave A fixed this module's builder only (wave B: every strict builder path reads the canonical table; fallback builders + runtime digest logging are integration requests) | `rope_table(device="cpu")` accepted; a CPU-built table differs from the CUDA one in 97,294 / 524,288 cos (18.6 %) and 99,316 sin elements at p < 4096, 23 % over all 131,072 positions (max abs 3.9e-3); `inv_freq` differs in 4/64 channels (6, 8, 56, 57); fed to the op it flips 20,007 / 16.8 M q̂ and 1,283 / 1.05 M k̂ bf16 outputs (positions 0..4095) and 331,488 q̂ at positions 126,976..131,071 | `rope_table` raises `RuntimeError` on non-CUDA devices; `canonical_rope_table(device)` = one cached object per CUDA device (131,072 rows, 64 MiB each), `verify=True` checks the pinned `CANONICAL_ROPE_TABLE_SHA256`; `rope_rows(positions)` gathers it; `rope_table_sha256()`; `trace()["rope_table"]` | CPU build raises. On CUDA, all 4 table builders in the repo (this module, `reference/golden_ops`, `runtime/fallback_ops`, `runtime/strict_kernels._fb_rope_table`) are bitwise identical over 16.8 M elements. Table sha256 `088dbb8c…56d9c` is identical in the cu130 and cu129 envs |
| qknorm_rope:2 | non-bf16 accepted | real, fixed | fp32 q/k → fp32 outputs and grads (no bf16 rounding: 262,139 / 262,144 elements differ from the bf16 path); fp16 → fp16 outputs | `_check_inputs` requires bf16 q and k (`TypeError`); backward launcher requires bf16 output grads | `TypeError` for fp32, fp16 and mixed inputs; bf16 bits unchanged |
| qknorm_rope:3 | no TP-shard test | coverage gap, test added (kernel was correct) | 84 shard runs (T ∈ {1, 7, 4096} × local/global × TP2/4/8 × every rank; 16/8/4 Q heads + 1 KV head; BLOCK_HQ 16/8/4, BLOCK_HK 1): 0 mismatching elements in q̂, k̂, dq, dk vs the full-call slices | `TestG2TensorParallelShards` (T ∈ {1, 7, 256, 4096}, fwd+bwd, local/global, plus strided views and trace tile configs) | passes before and after (as expected for a coverage gap) |
| qknorm_rope:4 | implicit local/global | real; wave A only PARTIALLY fixed it (opt-in API, every production caller still implicit); wave B adds counting, a warning and the fail-closed env switch; open until integration patch 2 | `cos=None` on a sliding layer runs NoPE silently: 15,994,827 / 16,777,216 q̂ differ, no error; no `local` kwarg | `qknorm_scaled_rope(..., local=True/False)` checked against cos/sin (`ValueError`); `qknorm_scaled_rope_at(q, k, pos, local=… or layer_idx=…)` (contract signature, layer kind mandatory, canonical table); `layer_is_local(i)` (checked against the manifest's `layer_types`); `trace()["layer_selection"]` | explicit mismatches raise. `local=None` (legacy implicit) is still accepted so that existing callers keep working; integrator patches below make every call site explicit |
| qknorm_rope:5 | int32 offsets | real, fixed | T = 524,352 (T·32·128 ≥ 2³¹ from row 524,288): default allocation → `CUDA error: an illegal memory access`; with an allocation layout where the wrapped address (X − 4 GiB) is mapped → **no error**, all 64 rows ≥ 524,288 wrong and 262,144 elements of an unrelated tensor overwritten | `t = tl.program_id(0).to(tl.int64)` in the fwd and bwd kernels (all offsets int64); `tile_config["offsets"] = "int64"` | 0 wrong rows, fwd+bwd, local/global, both layouts. No measurable speed change (interleaved A/B old vs new kernel: T=4096 local fwd 55.8 → 55.9 µs, bwd 76.1 → 75.7 µs; T=8192 fwd 99.7 → 100.0 µs; T=32768 fwd 370.7 → 372.2 µs), identical output bits at all 11 T × 2 layer kinds |

Tests. 40 new tests in `test_qknorm_rope.py` (79 in total). On the old code 18 of them fail,
but only 5 on behaviour (the CPU build, three dtype checks, and the overflow test, which dies
with an illegal memory access and was run in its own process); 13 fail only because a new name
or trace field did not exist yet (`tests_new.per_test[*].before_failure_kind`). The other 22 pass before and after: the TP shards, plus the
model-shape sweeps that guard against fail-closed checks breaking the model. After the fix all
79 pass in BOTH envs (vLLM env and Megatron env).

Shapes the model hits stay bit-exact. With kernel == fp32 golden → bf16, bitwise, these all
pass: T ∈ {1..17, 31, 63..65, 127, 255..257, 1000, 2047..2049, 4095..4097, 8191, 8192} at
random positions in [0, 131072), local and global; every decode batch 1..256 (batch-prefix
invariance); backward vs the hand-written VJP at T ∈ {1, 17, 2049, 8192}. No alignment or
padding rule is needed for this op: every head row is independent, BLOCK_H = next_pow2(H) is
masked, and all T ≥ 0 work.

Which bits changed: **none**. The int64 change touches only address arithmetic. The dtype,
device and layer-kind checks only reject inputs that used to run, and those rejected inputs
are never produced at Muse-Glimmer geometry by the current callers: all of them pass bf16
q/k and CUDA tables. The model's forward and backward bits are unchanged (sha256 rows in the
JSON, `experiments.bitwise_unchanged_by_fix`).

Public API: backward compatible. The new arguments are keyword-only and optional
(`local=None`), and the new functions are additive. The callers in `reference/`,
`runtime/vllm_*`, `runtime/megatron_*` and `runtime/strict_ops.py` are unchanged, and their tests
pass (`test_vllm_strict_binding.py` + `test_strict_model_tiny.py`: 39 passed, 1 skipped
(needs E2E env); `test_megatron_strict_binding.py`: 8/9, where the one failure,
`test_lm_head_tiles_tp2_equals_tp1_and_reference` (`dh2 != dh1`), is the pre-existing lm_head
failure listed in RESUME.md and does not involve this op).

(Wave B correction: "launch-bound up to T ≈ 2048" is right, but the device time was never
measured in wave A; see the CUDA-graph table in the wave B section.)

CPU overhead (T=64, wall clock, two runs; the second ran at host load average ~250 on 240
cores): forward through autograd 67 / 77 µs; fwd+bwd through autograd ≈ 365 / 335 µs, of which
~110 µs CPU is in each of the forward and backward autograd nodes (Triton launch path;
`record_backward` is not the cost: 364 vs 370 µs and 335 vs 322 µs without it). `rope_rows` costs
109 / 151 µs with its range check (one device→host sync) and 35 / 46 µs with
`check_range=False`, so call it once per forward, not once per layer. The kernels themselves
are launch-bound (~40–55 µs floor, depending on host load) up to T ≈ 2048.

### Follow-up benchmark (A100, median, µs; `experiments.bench_ab_old_vs_new_kernel` and `bench_series`)

| T | fwd old (int32) | fwd new (int64) | bwd old | bwd new | torch-native HF fwd (bf16) | native / strict fwd |
|---|---|---|---|---|---|---|
| 1 | 54.0 | 51.7 | 56.7 | 55.6 | 315 | 6× (launch-bound both) |
| 64 | 54.5 | 54.1 | 58.0 | 57.2 | 375 | 7× |
| 1024 | 54.1 | 53.8 | 57.9 | 58.7 | 298 | 5.5× |
| 4096 | 55.8 | 55.9 | 76.1 | 75.7 | 862 | 15× |
| 8192 | 99.7 | 100.0 | 141.6 | 141.0 | 1609 | 16× |
| 32768 | 370.7 | 372.2 | 533.2 | 530.9 | NOT RUN | – |

Local layers; old/new interleaved in one process, 7 rounds, same inputs; bits equal at every T.
Native-HF column from `bench_series` (final run, same lease, higher host load). The repo
benchmark `benchmarks/models/muse_glimmer/bench_qknorm_rope.py` (T=4096) at the final commit:
forward 0.086 ms strict vs 0.952 ms native (11.0×), forward+backward 0.431 vs 2.388 ms (5.5×).
At 6a61f6e under lower load it gave 0.071 / 0.954 ms (13.5×) and 0.327 / 2.461 ms (7.5×). The
difference is host-CPU launch overhead, not kernel time (wave B: device time from CUDA-graph
replay is in `experiments.bench_graph`; T = 4096 is at the launch floor, T ≥ 8192 is device-bound).

### Follow-up: how to reproduce

```
# from the worktree root; scripts are committed in reports/muse_glimmer_poc/scripts/rope/ (wave B, R5)
~/bin/gpu-lease -n 1 --pool 4-5 -- env PYTHONPATH=$PWD ~/envs/vllm/bin/python -m pytest \
    tests/models/muse_glimmer/test_qknorm_rope.py -q -rA -p no:cacheprovider        # 79 passed
~/bin/gpu-lease -n 1 --pool 4-5 -- env PYTHONPATH=$PWD ~/envs/megatron/bin/python -m pytest \
    tests/models/muse_glimmer/test_qknorm_rope.py -q -rA -p no:cacheprovider        # 79 passed
~/bin/gpu-lease -n 1 --pool 4-5 -- env PYTHONPATH=$PWD ~/envs/vllm/bin/python -m pytest \
    tests/models/muse_glimmer --ignore=tests/models/muse_glimmer/distributed -q      # 440 passed, 3 skipped, 1 failed (pre-existing lm_head TP2)
git archive 6a61f6e | tar -x -C <old>
~/bin/gpu-lease -n 1 -- reports/muse_glimmer_poc/scripts/rope/run_rope_measurements.sh <before> <old>
~/bin/gpu-lease -n 1 -- reports/muse_glimmer_poc/scripts/rope/run_rope_measurements.sh <after>
python3 reports/muse_glimmer_poc/scripts/rope/build_json.py <before> <after> <sha> reports/muse_glimmer_poc/data/kernels_rope.json
python3 reports/muse_glimmer_poc/scripts/rope/bench_ab.py <out.json> <old kernel file>   # eager A/B
```
Pre-fix evidence: the 40 new tests run against 6a61f6e gave 22 passed and 17 failed, and the
overflow test, run alone, failed with `torch.AcceleratorError: CUDA error: an illegal memory
access was encountered`. Per-test before/after outcomes are in `tests_new.per_test`.

### Follow-up: integration requests (wave A; superseded by the wave B list above)

1. Make every call site explicit about the layer kind (finding 4). All three non-Triton
   implementations must accept the keyword first. Add it to
   `reference/golden_ops.py::qknorm_scaled_rope`, `runtime/fallback_ops.py::qknorm_scaled_rope`
   and `runtime/strict_kernels.py::_fb_qknorm_scaled_rope`:
   ```python
       *, eps: float = ..., qk_scale: float = ..., local: bool | None = None,
   ) -> tuple[torch.Tensor, torch.Tensor]:
       if local is not None and bool(local) != (cos is not None):
           raise ValueError(f"local={local} but cos is {'None' if cos is None else 'given'}")
   ```
   Then at the call sites:
   - `reference/strict_model.py:138` → `ops.qknorm_scaled_rope(q, k, cos, sin, eps=cfg.rms_norm_eps, qk_scale=cfg.qk_scale_factor, local=layer.is_local)`
   - `runtime/vllm_strict.py:896` → add `local=bool(self.use_rope),`
   - `runtime/megatron_strict.py:511` → add `local=True`; `:514` → add `local=False`
   - `runtime/strict_ops.py:226/228` (probe) → add `local=True` / `local=False`
2. Table identity (finding 1). Each runtime should prove it builds the canonical table once and
   log the digest:
   - `runtime/vllm_strict.py::_rope_for`: when `state.ops.is_strict("rope_table")`, replace the
     `state.ops.rope_table(torch.arange(max_pos), ...)` build with
     `from rl_engine.models.muse_glimmer.ops.qknorm_rope import canonical_rope_table, rope_table_sha256`;
     `tables = canonical_rope_table(positions.device, verify=True)`, and record
     `state.cfg["rope_table_sha256"] = rope_table_sha256(positions.device)` in provenance.
     Rows: keep `cos[idx]`, which gives the same bits.
   - `runtime/megatron_strict.py::rope`: the same, gathering
     `rope_rows(positions, device=self.device)` once per forward (one host sync), and log
     `rope_table_sha256(self.device)` in provenance.
   - `reference/strict_model.py:132-136`: hoist the table out of the layer loop. Today it calls
     `ops.rope_table(positions)` for each of the 39 local layers; compute `cos, sin` once before
     `for layer in w.layers:`. The bits are unchanged.
   - The E2E orchestrator compares the two runtimes' `rope_table_sha256` and fails if they differ.
3. Add `<scratchpad>/rope_fu/bench_series.py` and `bench_ab.py` as
   `benchmarks/models/muse_glimmer/bench_qknorm_rope_series.py` if the series benchmark should
   live in the repo. Optional; I don't own `benchmarks/`.

### Follow-up: open issues

- `local=None` (implicit layer kind) is still accepted by default for backward compatibility
  (wave B: counted, warned once, rejected with `MUSEGLIMMER_O4_REQUIRE_LAYER_KIND=1`). Once
  integration patch 2 is merged, make `local` required (default of `layer_kind_required()`).
- The pinned table sha256 has been verified on A100 with torch 2.13 cu129 and cu130 only. Other
  GPUs, or a different torch/CUDA math library, may change `cos`/`sin`/`pow` bits;
  `test_canonical_table_fingerprint_is_pinned` and `canonical_rope_table(verify=True)` catch it.
- CPU launch overhead (~50 µs per kernel launch, ~350 µs for an autograd fwd+bwd at small T)
  dominates decode-size calls: 52 layers × ~65 µs ≈ 3.4 ms of host time per decode step for
  this op alone. CUDA graphs or fusing the op into the QKV GEMM epilogue are the fix. This
  wasn't tuned (performance only needs to be adequate).
- Wave B: the real-checkpoint 52-layer chain with the patched module RAN (reference row,
  `REPORT_full_model_chain.md`: train == prefill == chunked == decode byte-equal on both
  backends, batched decode gate, regenerated `xrt_reference_triton.json`). NOT RUN by this row:
  the vLLM/Megatron 30B engine runs with the wave B module (other owners).

Contract wording proposed for O4 (RFC feedback):
- "The RoPE table is `rope_table(arange(131072))` built ONCE per device by the fixed fp32
  torch-CUDA formula `inv_freq = 1/torch.pow(θ, arange(0,128,2)/128)`, `angle = p·inv_freq`,
  `cos/sin = cos/sin(cat(angle, angle))`. Rows are gathered by absolute position. Builds on any
  other device are forbidden. Each runtime logs the table's sha256 (`088dbb8c…56d9c` on
  torch 2.13 cu129/cu130), and train/rollout parity requires equal digests."
- "Inputs q/k MUST be bf16; the op rejects other dtypes."
- "The op takes the layer kind explicitly: `qknorm_scaled_rope(q, k, pos, local)`, where
  `local = (i % 4 != 3)`. Passing tables to a global layer, or omitting them on a local layer, is
  an error."
- "Head-sharded calls (TP t: 32/t Q heads, 1 KV head) are bitwise equal to the slice of the TP1
  call (parity axis TP1 vs TP2/4/8, G2)."
- "Index arithmetic is 64-bit (no limit on T·H·D)."

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
# added by the review follow-up (all additive / keyword-only):
qknorm_scaled_rope(..., local: bool | None = None)          # explicit layer kind, checked
qknorm_scaled_rope_at(q, k, positions | None, *, local=None, layer_idx=None, check_range=True)
canonical_rope_table(device=None, *, verify=False) -> (cos, sin) (131072, 128) fp32, cached object
rope_rows(positions, *, device=None, check_range=True) -> rows of the canonical table
rope_table_sha256(device=None) -> str ;  layer_is_local(i) -> bool
# added by the wave B review (additive):
runtime_provenance(*, compute_digest=True) -> dict   # built-table digest, layer-kind call counts
require_explicit_layer_kind(flag=True) ; layer_kind_required() ; LAYER_KIND_ENV
```
Usage (contract form): at init, `canonical_rope_table(dev, verify=True)` (before warm-up and
CUDA-graph capture; log `rope_table_sha256(dev)`); per forward, ONE `cos, sin = rope_rows(pos)`
(rows of the canonical table, gathered by absolute position), then per layer
`qknorm_scaled_rope(q, k, cos, sin, local=True)` on sliding layers and
`qknorm_scaled_rope(q, k, None, None, local=False)` on global layers (or
`qknorm_scaled_rope_at(q, k, pos, layer_idx=i)`). `rope_table(pos)` returns the same rows.

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
   generic (power-of-two padded); the TP head shards 16/8/4 Q + 1 KV are tested bitwise
   against the full-call slices (finding qknorm_rope:3).
3. ~~The table is "built once and cached" by the caller.~~ Superseded by the review follow-up
   (finding qknorm_rope:1): the module now owns the cache (`canonical_rope_table(device)`, one
   object per CUDA device, sha256 pinned) and `rope_table` raises on non-CUDA devices.
   `rope_table(pos)`, `rope_table(arange)[pos]` and `rope_rows(pos)` are bitwise equal; since
   wave B `rope_table` IS `rope_rows` (copies of the canonical rows; positions must lie in
   `[0, 131072)`).
4. ~~The op also accepts fp16/fp32 q/k.~~ Superseded (finding qknorm_rope:2): non-bf16 q/k and
   non-bf16 output gradients raise `TypeError`.
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
