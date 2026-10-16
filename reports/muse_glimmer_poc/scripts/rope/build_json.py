# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
# ruff: noqa: E501  (kept as run in wave A; lines are measurement plumbing)
# Wave A: assemble reports/muse_glimmer_poc/data/kernels_rope.json from the probe/bench/pytest
# outputs of run_rope_measurements.sh (a "before" dir measured at 6a61f6e, an "after" dir at the
# fixed commit). Wave B fields are added on top by update_json_wave_b.py.
# usage: python build_json.py <before_dir> <after_dir> <commit_after> <out.json>
import json
import re
import sys
from pathlib import Path

B, A = Path(sys.argv[1]), Path(sys.argv[2])
commit_after, out = sys.argv[3], sys.argv[4]
SP = "reports/muse_glimmer_poc/scripts/rope"


def j(p):
    return json.loads(Path(p).read_text())


def pytest_outcomes(log):
    res = {}
    for line in Path(log).read_text().splitlines():
        m = re.match(r"^(PASSED|FAILED|ERROR|SKIPPED)\s+(\S+)", line)
        if m:
            res[m.group(2).split("::", 1)[1]] = m.group(1)
    return res


def tail(log, n=1):
    lines = [x for x in Path(log).read_text().splitlines() if x.strip()]
    return lines[-n:]


def strip_env(d):
    d = dict(d)
    d.pop("env", None)
    d.pop("probe", None)
    return d


before_new = pytest_outcomes(B / "pytest_new_tests_before.log")
before_new[
    "TestF5LargeTokenCounts::test_rows_beyond_int32_offsets_match_small_calls"
] = "FAILED (torch.AcceleratorError: CUDA error: an illegal memory access was encountered)"
after_all = pytest_outcomes(A / "pytest_qknorm_rope_vllm.log")
after_meg = pytest_outcomes(A / "pytest_qknorm_rope_megatron.log")
new_tests = sorted(before_new)
test_table = [
    {
        "test": t,
        "before_6a61f6e": before_new[t],
        "after_vllm_env": after_all.get(t, "NOT RUN"),
        "after_megatron_env": after_meg.get(t, "NOT RUN"),
    }
    for t in new_tests
]

f1b, f1a = j(B / "f1.json"), j(A / "f1.json")
f2b, f2a = j(B / "f2.json"), j(A / "f2.json")
f3b, f3a = j(B / "f3.json"), j(A / "f3.json")
f4b, f4a = j(B / "f4.json"), j(A / "f4.json")
f5b_, f5a = j(B / "f5.json"), j(A / "f5.json")
f5cb, f5ca = j(B / "f5c.json"), j(A / "f5c.json")
f5bb = j(B / "f5b.json")
xb, xa = j(B / "xenv.json"), j(A / "xenv.json")
xmb, xma = j(B / "xenv_megatron.json"), j(A / "xenv_megatron.json")
bench_b, bench_a = j(B / "bench_series_before.json"), j(A / "bench_series_final.json")
overhead = j(A / "overhead.json")


def parse_repo_bench(log):
    txt = Path(log).read_text()
    d = {}
    for name in ("forward", "forward+backward"):
        m = re.search(rf"^{re.escape(name)}\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)x", txt, re.M)
        if m:
            d[name] = {
                "strict_ms": float(m.group(1)),
                "native_hf_ms": float(m.group(2)),
                "native_over_strict": float(m.group(3)),
            }
    m = re.search(r"bandwidth: (\d+) GB/s", txt)
    d["strict_fwd_GBps"] = int(m.group(1)) if m else None
    d["raw"] = txt.strip().splitlines()
    return d


def series(bench, local, key):
    return [r[key] for r in bench["series"] if r["local"] == local]


Ts = [r["T"] for r in bench_a["series"] if r["local"]]
sha_rows = []
for name, b, a in (("vllm env (cu130)", xb, xa), ("megatron env (cu129)", xmb, xma)):
    for k in ("table_sha256",):
        sha_rows.append({"env": name, "what": "canonical table (cos||sin, 131072 x 128 fp32)",
                         "before": b[k], "after": a[k], "equal": a[k] == b[k]})
    for kind in ("local", "global"):
        for k in ("fwd_sha256", "bwd_sha256"):
            sha_rows.append({"env": name, "what": f"T=4096 {kind} {k[:3]} (q_hat,k_hat | dq,dk)",
                             "before": b[kind][k], "after": a[kind][k],
                             "equal": a[kind][k] == b[kind][k]})
    k = "local_pos_126976_131072_fwd_sha256"
    sha_rows.append({"env": name, "what": "T=4096 local fwd at positions 126976..131071",
                     "before": b[k], "after": a[k], "equal": a[k] == b[k]})
cross_env_equal = all(
    xa[k] == xma[k] for k in ("table_sha256", "local", "global", "local_pos_126976_131072_fwd_sha256")
)

findings = [
    {
        "id": "qknorm_rope:1",
        "title": "O4 RoPE-table build/cache identity across runtimes",
        "verdict": "confirmed_fixed",
        "reproduced": True,
        "plain_language": (
            "The op accepted cos/sin tables built anywhere. A table built on the CPU has different "
            "last bits than one built on the GPU (about 1 element in 5), which flips ~0.1% of the "
            "bf16 outputs. All current callers happen to build on CUDA and all 4 table builders in "
            "the repo agree bit-for-bit on CUDA, and the vLLM (CUDA 13.0) and Megatron (CUDA 12.9) "
            "envs produce the same table, but nothing enforced it."),
        "evidence_before": {
            "cpu_build_accepted": f1b["cpu_build_accepted"],
            "cpu_vs_cuda_table": {k: v for k, v in f1b["cpu_vs_cuda_table"].items()
                                  if not k.endswith("per_bucket")},
            "cpu_vs_cuda_inv_freq_mismatch_channels": f1b["cpu_vs_cuda_inv_freq_mismatch_channels"],
            "op_flips_cpu_table_pos_0_4096": f1b["op_flips_cpu_table_pos_0_4096"],
            "op_flips_cpu_table_pos_126976_131072": f1b["op_flips_cpu_table_pos_126976_131072"],
            "cuda_cross_impl_vs_ops_rope_table": f1b["cuda_cross_impl_vs_ops_rope_table"],
            "canonical_table_sha256_vllm_env": xb["table_sha256"],
            "canonical_table_sha256_megatron_env": xmb["table_sha256"],
        },
        "evidence_after": {
            "cpu_build_accepted": f1a["cpu_build_accepted"],
            "cpu_build_error": f1a.get("cpu_build_error"),
            "cuda_cross_impl_vs_ops_rope_table": f1a["cuda_cross_impl_vs_ops_rope_table"],
            "canonical_table_sha256": f1a["canonical_table_sha256"],
        },
        "fix": (
            "rope_table raises RuntimeError on non-CUDA devices; canonical_rope_table(device) "
            "(one cached (cos, sin) object per CUDA device, 131072 rows, verify=True checks the "
            "pinned sha256); rope_rows(positions) = index_select of it; rope_table_sha256(); "
            "trace()['rope_table'] records builder, device_type, formula, sha256."),
        "tests": [t for t in new_tests if t.startswith("TestF1")],
        "bits_changed": "none (CUDA builds are unchanged; only non-CUDA builds now raise)",
    },
    {
        "id": "qknorm_rope:2",
        "title": "non-bf16 inputs silently accepted",
        "verdict": "confirmed_fixed",
        "reproduced": True,
        "plain_language": (
            "fp32/fp16 q,k ran and returned fp32/fp16 outputs, skipping the contract's single "
            "bf16 rounding. Now the strict kernel raises TypeError for anything but bf16."),
        "evidence_before": {k: v for k, v in f2b.items() if k.startswith("torch.")},
        "evidence_after": {k: v for k, v in f2a.items() if k.startswith("torch.")},
        "fix": "_check_inputs requires q.dtype == k.dtype == bf16 (TypeError); the backward "
               "launcher requires bf16 output gradients (TypeError).",
        "tests": [t for t in new_tests if t.startswith("TestF2")],
        "bits_changed": "none (bf16 path unchanged)",
    },
    {
        "id": "qknorm_rope:3",
        "title": "no TP-parity test for head shards (BLOCK_HQ 16/8/4 vs 32)",
        "verdict": "confirmed_fixed",
        "reproduced": True,
        "plain_language": (
            "The coverage gap was real; the kernel itself was already correct. Every TP2/4/8 "
            "rank shard (16/8/4 query heads + 1 KV head) gives exactly the bits of the matching "
            "slice of the full 32/2-head call, forward and backward, local and global layers: "
            "0 mismatching elements in 84 shard runs (T = 1, 7, 4096). Test added."),
        "evidence_before": {"cases": len(f3b["cases"]), "total_mismatch": f3b["total_mismatch"]},
        "evidence_after": {"cases": len(f3a["cases"]), "total_mismatch": f3a["total_mismatch"]},
        "fix": "test only (TestG2TensorParallelShards); no kernel change needed",
        "tests": [t for t in new_tests if t.startswith("TestG2TensorParallel")],
        "bits_changed": "none",
    },
    {
        "id": "qknorm_rope:4",
        "title": "implicit local/global selection",
        "verdict": "confirmed_fixed",
        "reproduced": True,
        "plain_language": (
            "Passing cos=None on a sliding-window layer silently ran the NoPE path (95% of the "
            "outputs differ, no error). Callers can now say local=True/False (checked against "
            "cos/sin) or use qknorm_scaled_rope_at(q, k, pos, local=|layer_idx=) where the layer "
            "kind is mandatory. local=None keeps the old behaviour so existing callers work."),
        "evidence_before": strip_env(f4b),
        "evidence_after": strip_env(f4a),
        "fix": "qknorm_scaled_rope(..., local=None|True|False); qknorm_scaled_rope_at(q, k, "
               "positions, *, local | layer_idx, check_range=True); layer_is_local(i); "
               "trace()['layer_selection'].",
        "tests": [t for t in new_tests if t.startswith("TestF4")],
        "bits_changed": "none",
        "residual": "local=None (legacy implicit) is still accepted for backward compatibility; "
                    "the call sites must pass local= (integrator patches) before it can be removed.",
    },
    {
        "id": "qknorm_rope:5",
        "title": "int32 offset overflow",
        "verdict": "confirmed_fixed",
        "reproduced": True,
        "plain_language": (
            "With T >= 524288 tokens (T*32*128 >= 2^31) the element offsets wrapped. Depending on "
            "memory layout the old kernel either crashed (illegal memory access) or silently "
            "wrote 64 output rows into an unrelated tensor (262,144 elements) and returned "
            "garbage rows. Offsets are now int64; no measurable speed change; output bits "
            "unchanged for every shape."),
        "evidence_before": {
            "default_layout": strip_env(f5b_),
            "pad_before_q_layout": strip_env(f5bb),
            "silent_layout": strip_env(f5cb),
        },
        "evidence_after": {"default_layout": strip_env(f5a), "silent_layout": strip_env(f5ca)},
        "fix": "t = tl.program_id(0).to(tl.int64) in the forward and backward kernels (all row, "
               "element and table offsets int64); tile_config['offsets'] = 'int64'.",
        "tests": [t for t in new_tests if t.startswith("TestF5")],
        "bits_changed": "none (address arithmetic only; sha256 of fwd/bwd outputs identical)",
        "realistic_exposure": "none at Muse-Glimmer geometry: training T<=8192, rollout batches "
                              "<=256 are 64x below the wrap point",
    },
]

res = {
    "row": "qknorm_scaled_rope (review follow-ups, findings qknorm_rope:1..5)",
    "status": "complete",
    "commit": commit_after,
    "commit_before": "6a61f6e",
    "branch": "worktree-wf_7b064961-36f-6",
    "env": {
        "gpu": xa["env"]["gpu"],
        "vllm_env": {"python": "~/envs/vllm/bin/python", "torch": xa["env"]["torch"],
                     "cuda": xa["env"]["cuda"], "triton": xa["env"]["triton"], "vllm": "0.31.0"},
        "megatron_env": {"python": "~/envs/megatron/bin/python", "torch": xma["env"]["torch"],
                         "cuda": xma["env"]["cuda"], "triton": xma["env"]["triton"],
                         "megatron_core": "0.20.0"},
    },
    "commands": [
        "# wave A (findings qknorm_rope:1..5): before = tree of 6a61f6e, after = the fixed commit",
        "git archive 6a61f6e | tar -x -C <old>",
        f"~/bin/gpu-lease -n 1 -- {SP}/run_rope_measurements.sh <before_dir> <old>",
        f"~/bin/gpu-lease -n 1 -- {SP}/run_rope_measurements.sh <after_dir>",
        "~/bin/gpu-lease -n 1 -- env PYTHONPATH=$PWD ~/envs/vllm/bin/python -m pytest "
        "tests/models/muse_glimmer --ignore=tests/models/muse_glimmer/distributed -q",
        f"python {SP}/build_json.py <before_dir> <after_dir> <commit> "
        "reports/muse_glimmer_poc/data/kernels_rope.json",
    ],
    "scripts": {
        "dir": SP,
        "probes": f"{SP}/probe_findings.py",
        "probes_wave_b": f"{SP}/probe_wave_b.py",
        "bench_series": f"{SP}/bench_series.py",
        "bench_ab": f"{SP}/bench_ab.py",
        "bench_graph": f"{SP}/bench_graph.py",
        "overhead": f"{SP}/overhead_probe.py",
        "runner": f"{SP}/run_rope_measurements.sh",
        "json_builder": [f"{SP}/build_json.py", f"{SP}/update_json_wave_b.py"],
    },
    "findings": findings,
    "tests_new": {
        "count": len(new_tests),
        "before_failed": sum(1 for t in test_table if t["before_6a61f6e"].startswith("FAILED")),
        "before_passed": sum(1 for t in test_table if t["before_6a61f6e"] == "PASSED"),
        "after_passed_vllm_env": sum(1 for t in test_table if t["after_vllm_env"] == "PASSED"),
        "after_passed_megatron_env": sum(
            1 for t in test_table if t["after_megatron_env"] == "PASSED"),
        "per_test": test_table,
    },
    "tests_summary_lines": {
        "test_qknorm_rope_vllm_env": tail(A / "pytest_qknorm_rope_vllm.log")[0],
        "test_qknorm_rope+megatron_binding_megatron_env": tail(
            A / "pytest_qknorm_rope_megatron.log")[0],
        "muse_glimmer_suite_vllm_env": tail(A / "pytest_suite_vllm.log")[0],
    },
    "experiments": {
        "table_cpu_vs_cuda_per_bucket": {
            "description": "fraction of cos/sin elements whose fp32 bits differ between a CPU-built "
                           "and a CUDA-built rope_table, per 2048-position bucket (0..131071)",
            "bucket_start": list(range(0, 131072, 2048)),
            "cos_mismatch_frac": f1b["cpu_vs_cuda_table"]["cos_mismatch_frac_per_bucket"],
            "sin_mismatch_frac": f1b["cpu_vs_cuda_table"]["sin_mismatch_frac_per_bucket"],
        },
        "tp_shards": {
            "description": "per (T, layer kind, TP degree, rank) mismatching elements of the shard "
                           "call vs the slice of the full 32/2-head call (after the fix)",
            "cases": f3a["cases"],
        },
        "bitwise_unchanged_by_fix": {
            "description": "sha256 of op outputs on fixed inputs, before (6a61f6e) vs after, in "
                           "both runtimes' envs",
            "rows": sha_rows,
            "all_equal": all(r["equal"] for r in sha_rows),
            "vllm_env_equals_megatron_env": cross_env_equal,
        },
        "bench_series": {
            "description": "median ms over 5 reps (CUDA events); strict_fwd_kernel = launcher incl. "
                           "checks; strict_fwd_api = autograd.Function forward; fwd_bwd_autograd = "
                           "fwd + torch.autograd.grad; native_hf_fwd = torch HF-style bf16 "
                           "composition (local layers only)",
            "T": Ts,
            "before": bench_b["series"],
            "after": bench_a["series"],
            "plot": {
                "after_local_fwd_kernel_ms": series(bench_a, True, "strict_fwd_kernel_ms"),
                "before_local_fwd_kernel_ms": series(bench_b, True, "strict_fwd_kernel_ms"),
                "after_local_bwd_kernel_ms": series(bench_a, True, "strict_bwd_kernel_ms"),
                "before_local_bwd_kernel_ms": series(bench_b, True, "strict_bwd_kernel_ms"),
                "after_local_native_hf_fwd_ms": series(bench_a, True, "native_hf_fwd_ms"),
                "after_global_fwd_kernel_ms": series(bench_a, False, "strict_fwd_kernel_ms"),
                "after_local_fwd_GBps": series(bench_a, True, "fwd_kernel_GBps"),
            },
        },
        "bench_ab_old_vs_new_kernel": {
            **{k: v for k, v in j(A / "bench_ab.json").items() if k != "series"},
            "T": sorted({r["T"] for r in j(A / "bench_ab.json")["series"]}),
            "series": j(A / "bench_ab.json")["series"],
            "all_bitwise_equal": all(
                r["bitwise_equal_fwd"] and r["bitwise_equal_bwd"]
                for r in j(A / "bench_ab.json")["series"]
            ),
            "plot": {
                key: [r[key] for r in j(A / "bench_ab.json")["series"] if r["local"]]
                for key in ("old_fwd_ms", "new_fwd_ms", "old_bwd_ms", "new_bwd_ms")
            },
            "note": "the fair before/after timing for the int64-offset change (same process, "
                    "same inputs, interleaved); local-layer series in 'plot'",
        },
        "host_load_note": (
            "the node's CPU load average was ~205-250 (240 cores) during the final runs; "
            "launch-bound timings (T <= 2048, ~40-55 us floors) and the autograd fwd+bwd numbers "
            "vary by ~25% between runs with host load, GPU-bound timings (T >= 4096) do not. "
            "bench_series 'before' ran at lower load than 'after'; use bench_ab_old_vs_new_kernel "
            "for the before/after comparison."),
        "bench_repo_T4096": {
            "before": parse_repo_bench(B / "bench_repo_before.log"),
            "after": parse_repo_bench(A / "bench_repo_final.log"),
        },
        "cpu_overhead_T64_us": overhead,
    },
    "suggested_figures": [
        {"id": "fig_table_cpu_vs_cuda", "kind": "line",
         "data": "experiments.table_cpu_vs_cuda_per_bucket",
         "caption": "A CPU-built RoPE table differs from the CUDA-built one in ~18-27% of its "
                    "elements at every position range; the fix makes non-CUDA builds raise."},
        {"id": "fig_findings_tests", "kind": "stacked bar (before/after)",
         "data": "tests_new",
         "caption": f"{len(new_tests)} new tests: "
                    f"{sum(1 for t in test_table if t['before_6a61f6e'].startswith('FAILED'))} fail "
                    "on the old code (one of them by an illegal memory access), all pass after; "
                    "the others (TP shards, model shapes) pass before and after."},
        {"id": "fig_tp_shards", "kind": "heatmap TP degree x rank (all zeros)",
         "data": "experiments.tp_shards", "caption": "Head shards are bitwise equal to the full "
                                                     "call slices at TP2/4/8."},
        {"id": "fig_overflow", "kind": "annotated timeline / bar",
         "data": "findings[4].evidence_before",
         "caption": "Row 524288 is where T*32*128 crosses 2^31: old kernel crash or silent "
                    "corruption, new kernel exact."},
        {"id": "fig_latency", "kind": "log-log line, ms vs T",
         "data": "experiments.bench_ab_old_vs_new_kernel.plot (old vs new) + "
                 "experiments.bench_series.plot (native HF)",
         "caption": "Strict fwd/bwd kernel latency vs T, before/after int64 offsets (identical) "
                    "and vs the torch-native HF composition; flat ~40 us floor = launch "
                    "overhead for T <= 2048."},
    ],
    "not_run": [
        "real-checkpoint full-model chain with the patched op (MUSEGLIMMER_CHECKPOINT run) - "
        "NOT RUN; covered indirectly by the bitwise-unchanged sha256 check",
        "vLLM engine / Megatron Bridge 30B runs - NOT RUN (out of scope of this row)",
    ],
}
Path(out).write_text(json.dumps(res, indent=1) + "\n")
print("wrote", out, len(json.dumps(res)))
