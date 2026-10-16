import pathlib

p = pathlib.Path("rl_engine/models/muse_glimmer/ops/REPORT_qknorm_rope.md")
s = p.read_text()
INTEG = pathlib.Path(
    "/tmp/claude-1000/-home-ubuntu-RL-Kernel--claude-worktrees-swift-oak-hsv7/"
    "80655dc5-1eae-4992-a93b-f6f87e8ab1d8/scratchpad/wb/integration_requests.md"
).read_text()


def rep(old, new):
    global s
    assert s.count(old) == 1, old[:100]
    s = s.replace(old, new)


WAVE_B = '''## Wave B review follow-ups (2026-10-15, issues kernels_rope:R1..R12)

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
| R4 | fixed | the first build inside CUDA-graph capture raises and caches nothing (runtimes build at init with `canonical_rope_table(dev, verify=True)`); the table is built as a normal tensor even under `inference_mode`, so the counters exist | build recorded, not executed: cached table wrong until replay, 16,384 wrong cos/sin elements served to an eager caller; table built under `inference_mode` is an inference tensor | `RuntimeError` ("must be built before CUDA-graph capture"), nothing cached; normal tensor |
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
129 passed, 2 skipped (Megatron env; the lm_head TP2 test now passes there); full single-GPU
`tests/models/muse_glimmer` suite (vLLM env, `--ignore distributed`, at 3404cf9 = the code of
39053b2 minus a test-fixture tweak) 653 passed, 8 skipped, 0 failed.

Device time per launch (CUDA-graph replay, 20 launches per graph, median of 5 interleaved
rounds; `experiments.bench_graph`; local layers, µs). Old = 6a61f6e kernel (int32 offsets),
new = current. Bits equal at every T, both layer kinds, forward and backward.

| T | fwd old | fwd new | bwd old | bwd new | eager launcher, new fwd | fwd GB/s (T ≥ 8192) |
|---|---|---|---|---|---|---|
{GRAPH_ROWS}

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
~/bin/gpu-lease -n 1 -- env PYTHONPATH=<old> ~/envs/vllm/bin/python \\
    reports/muse_glimmer_poc/scripts/rope/probe_wave_b.py <out>/probe_wave_b_before.json
~/bin/gpu-lease -n 1 -- env PYTHONPATH=$PWD ~/envs/vllm/bin/python \\
    reports/muse_glimmer_poc/scripts/rope/probe_wave_b.py <out>/probe_wave_b_after.json
git show 6a61f6e:rl_engine/models/muse_glimmer/triton/qknorm_rope.py > <out>/old_kernel.py
~/bin/gpu-lease -n 1 -- env PYTHONPATH=$PWD ~/envs/vllm/bin/python \\
    reports/muse_glimmer_poc/scripts/rope/bench_graph.py <out>/bench_graph.json --old-kernel <out>/old_kernel.py
cp tests/models/muse_glimmer/test_{qknorm_rope,strict_model_tiny}.py <old>/tests/models/muse_glimmer/
(cd <old> && ~/bin/gpu-lease -n 1 -- env PYTHONPATH=<old> ~/envs/vllm/bin/python -m pytest -p no:cacheprovider \\
    tests/models/muse_glimmer/test_qknorm_rope.py tests/models/muse_glimmer/test_strict_model_tiny.py -q -rfE)
    # -> 13 failed, 100 passed (before); the same files at HEAD: 113 passed
~/bin/gpu-lease -n 1 -- env PYTHONPATH=$PWD ~/envs/megatron/bin/python -m pytest -p no:cacheprovider \\
    tests/models/muse_glimmer/test_qknorm_rope.py tests/models/muse_glimmer/test_megatron_strict_binding.py \\
    tests/models/muse_glimmer/test_strict_model_tiny.py -q -rA      # -> 129 passed, 2 skipped
python reports/muse_glimmer_poc/scripts/rope/update_json_wave_b.py --json reports/muse_glimmer_poc/data/kernels_rope.json ...
```
(the exact argument list is in the script's docstring; `run_rope_measurements.sh` reruns the
wave A probes and benchmarks as well).

{INTEGRATION}
'''

import json  # noqa: E402

g = json.loads(pathlib.Path("reports/muse_glimmer_poc/data/kernels_rope.json").read_text())
rows = []
for r in g["experiments"]["bench_graph"]["series"]:
    if not r["local"]:
        continue
    gbps = r.get("new_fwd_GBps")
    rows.append(
        f"| {r['T']} | {r['old_fwd_gpu_us']:.1f} | {r['new_fwd_gpu_us']:.1f} | "
        f"{r['old_bwd_gpu_us']:.1f} | {r['new_bwd_gpu_us']:.1f} | {r['new_fwd_eager_us']:.1f} | "
        f"{'' if gbps is None else f'{gbps:.0f}'} |"
    )
WAVE_B = WAVE_B.replace("{GRAPH_ROWS}", "\n".join(rows)).replace("{INTEGRATION}", INTEG.strip())

rep("## Review follow-ups (2026-10-15, findings qknorm_rope:1..5)\n",
    WAVE_B + "\n## Review follow-ups (2026-10-15, findings qknorm_rope:1..5)\n")

# wave A table verdicts (R1, R2, R9)
rep("| qknorm_rope:1 | O4 table identity | real, fixed |",
    "| qknorm_rope:1 | O4 table identity | real; wave A fixed this module's builder only (wave B: every strict builder path reads the canonical table; fallback builders + runtime digest logging are integration requests) |")
rep("| qknorm_rope:3 | no TP-shard test | coverage gap only (kernel correct), test added |",
    "| qknorm_rope:3 | no TP-shard test | coverage gap, test added (kernel was correct) |")
rep("| qknorm_rope:4 | implicit local/global | real, fixed |",
    "| qknorm_rope:4 | implicit local/global | real; wave A only PARTIALLY fixed it (opt-in API, every production caller still implicit); wave B adds counting, a warning and the fail-closed env switch; open until integration patch 2 |")

rep('''**In short:** all five findings were reproduced on the GPU. Four were real bugs and are
now fixed so that they fail closed. The fifth (TP shards) was only a missing test: the kernel
was already bit-exact, and the test is now added.''',
    '''**In short:** all five findings were reproduced on the GPU. Four were real bugs; wave A made
qknorm_rope:2 and :5 fail closed, but qknorm_rope:1 and :4 only in this module's own API (the
wave B review above corrects those two verdicts). The fifth (TP shards) was only a missing
test: the kernel was already bit-exact, and the test is now added.''')

rep('''Tests. 40 new tests in `test_qknorm_rope.py` (79 in total). On the old code 18 of them fail:
17 assertion/API failures and the overflow test, which dies with an illegal memory access and
was run in its own process.''',
    '''Tests. 40 new tests in `test_qknorm_rope.py` (79 in total). On the old code 18 of them fail,
but only 5 on behaviour (the CPU build, three dtype checks, and the overflow test, which dies
with an illegal memory access and was run in its own process); 13 fail only because a new name
or trace field did not exist yet (`tests_new.per_test[*].before_failure_kind`).''')

rep('''CPU overhead (T=64, wall clock, two runs;''',
    '''(Wave B correction: "launch-bound up to T ≈ 2048" is right, but the device time was never
measured in wave A; see the CUDA-graph table in the wave B section.)

CPU overhead (T=64, wall clock, two runs;''')

rep('''At 6a61f6e under lower load it gave 0.071 / 0.954 ms (13.5×) and 0.327 / 2.461 ms (7.5×). The
difference is host-CPU launch overhead, not kernel time: the GPU-bound T ≥ 4096 rows match.''',
    '''At 6a61f6e under lower load it gave 0.071 / 0.954 ms (13.5×) and 0.327 / 2.461 ms (7.5×). The
difference is host-CPU launch overhead, not kernel time (wave B: device time from CUDA-graph
replay is in `experiments.bench_graph`; T = 4096 is at the launch floor, T ≥ 8192 is device-bound).''')

rep('''# from the worktree root; scripts are in the session scratchpad (paths in kernels_rope.json "scripts")''',
    '''# from the worktree root; scripts are committed in reports/muse_glimmer_poc/scripts/rope/ (wave B, R5)''')
rep('''<scratchpad>/rope_fu/run_probes.sh <outdir> f1 f2 f3 f4 xenv xenv_megatron f5 f5c  # probes per finding
<scratchpad>/rope_fu/run_bench.sh <label> <outdir>                                # series + repo bench
<scratchpad>/rope_fu/run_script.sh vllm <scratchpad>/rope_fu/bench_ab.py <log> <json>  # old vs new A/B
python3 <scratchpad>/rope_fu/build_json.py <sha> reports/muse_glimmer_poc/data/kernels_rope.json''',
    '''git archive 6a61f6e | tar -x -C <old>
~/bin/gpu-lease -n 1 -- reports/muse_glimmer_poc/scripts/rope/run_rope_measurements.sh <before> <old>
~/bin/gpu-lease -n 1 -- reports/muse_glimmer_poc/scripts/rope/run_rope_measurements.sh <after>
python3 reports/muse_glimmer_poc/scripts/rope/build_json.py <before> <after> <sha> reports/muse_glimmer_poc/data/kernels_rope.json
python3 reports/muse_glimmer_poc/scripts/rope/bench_ab.py <out.json> <old kernel file>   # eager A/B''')

rep('''### Follow-up: integration requests (other owners' files; exact patches)
''', '''### Follow-up: integration requests (wave A; superseded by the wave B list above)
''')

rep('''- `local=None` (implicit layer kind) is still accepted for backward compatibility. Once the
  patches in item 1 are merged, make `local` required (a one-line change in
  `qknorm_scaled_rope`, plus the tests).''',
    '''- `local=None` (implicit layer kind) is still accepted by default for backward compatibility
  (wave B: counted, warned once, rejected with `MUSEGLIMMER_O4_REQUIRE_LAYER_KIND=1`). Once
  integration patch 2 is merged, make `local` required (default of `layer_kind_required()`).''')
rep('''- NOT RUN: the real-checkpoint 52-layer chain and the vLLM/Megatron 30B engine runs with the
  patched module. They are covered indirectly: forward and backward output bits are
  sha256-identical before and after, and all caller unit tests pass.''',
    '''- Wave B: the real-checkpoint 52-layer chain with the patched module RAN (reference row,
  `REPORT_full_model_chain.md`: train == prefill == chunked == decode byte-equal on both
  backends, batched decode gate, regenerated `xrt_reference_triton.json`). NOT RUN by this row:
  the vLLM/Megatron 30B engine runs with the wave B module (other owners).''')

rep('''Usage: `cos, sin = rope_table(torch.arange(max_pos, device=dev), device=dev)` once per device,
then `qknorm_scaled_rope(q, k, cos[pos], sin[pos])` (local) / `qknorm_scaled_rope(q, k, None, None)`
(global). Gathered rows are bitwise identical to `rope_table(pos)` (tested).''',
    '''Usage (contract form): at init, `canonical_rope_table(dev, verify=True)` (before warm-up and
CUDA-graph capture; log `rope_table_sha256(dev)`); per forward, ONE `cos, sin = rope_rows(pos)`
(rows of the canonical table, gathered by absolute position), then per layer
`qknorm_scaled_rope(q, k, cos, sin, local=True)` on sliding layers and
`qknorm_scaled_rope(q, k, None, None, local=False)` on global layers (or
`qknorm_scaled_rope_at(q, k, pos, layer_idx=i)`). `rope_table(pos)` returns the same rows.''')

rep('''canonical_rope_table(device=None, *, verify=False) -> (cos, sin) (131072, 128) fp32, cached object
rope_rows(positions, *, device=None, check_range=True) -> rows of the canonical table
rope_table_sha256(device=None) -> str ;  layer_is_local(i) -> bool
```''', '''canonical_rope_table(device=None, *, verify=False) -> (cos, sin) (131072, 128) fp32, cached object
rope_rows(positions, *, device=None, check_range=True) -> rows of the canonical table
rope_table_sha256(device=None) -> str ;  layer_is_local(i) -> bool
# added by the wave B review (additive):
runtime_provenance(*, compute_digest=True) -> dict   # built-table digest, layer-kind call counts
require_explicit_layer_kind(flag=True) ; layer_kind_required() ; LAYER_KIND_ENV
```''')

rep('''   object per CUDA device, sha256 pinned) and `rope_table` raises on non-CUDA devices.
   `rope_table(pos)`, `rope_table(arange)[pos]` and `rope_rows(pos)` are bitwise equal.''',
    '''   object per CUDA device, sha256 pinned) and `rope_table` raises on non-CUDA devices.
   `rope_table(pos)`, `rope_table(arange)[pos]` and `rope_rows(pos)` are bitwise equal; since
   wave B `rope_table` IS `rope_rows` (copies of the canonical rows; positions must lie in
   `[0, 131072)`).''')

p.write_text(s)
print("ok")
