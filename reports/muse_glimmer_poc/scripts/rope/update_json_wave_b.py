# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Wave B: update reports/muse_glimmer_poc/data/kernels_rope.json (built for wave A by
build_json.py) with the wave B review verdicts (kernels_rope:R1..R12), the per-test before/after
outcomes with the reason each test failed before, the behavioural probes (probe_wave_b.py on
the old and the new tree) and the CUDA-graph benchmark (bench_graph.py).

usage: python update_json_wave_b.py --json <kernels_rope.json> --commit <sha> \\
    --wave-a-before-log <pytest log of the wave A tests at 6a61f6e (-rA)> \\
    --tests-before-log <pytest -rfE log of the wave B tests at f91e8b6> \\
    --ttir-before-log <pytest log of the TTIR test with the 6a61f6e kernel> \\
    --tests-after-vllm <pytest -rA log> --tests-after-megatron <pytest -rA log> \\
    --suite-vllm <log> --suite-commit <sha> --probe-before <json> --probe-after <json> \\
    --bench-graph <json>
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def outcomes(log: str) -> dict[str, str]:
    res = {}
    for line in Path(log).read_text().splitlines():
        m = re.match(r"^(PASSED|FAILED|ERROR|SKIPPED)\s+(\S+)", line)
        if m and "::" in m.group(2):
            res[m.group(2).split("::", 1)[1]] = m.group(1)
    return res


def first_errors(log: str) -> dict[str, str]:
    """test id (Class.test[param]) -> first 'E   ' line of its failure section."""
    out: dict[str, str] = {}
    name = None
    for line in Path(log).read_text().splitlines():
        m = re.match(r"^_+ (\S+) _+$", line)
        if m:
            name = m.group(1)
            continue
        if name and line.startswith("E   ") and name not in out:
            out[name] = line[1:].strip()[:200]
    return out


def classify(err: str | None) -> str:
    if err is None:
        return "unknown (no error line captured)"
    if "illegal memory access" in err:
        return "behaviour: crash (illegal memory access)"
    if re.search(r"AttributeError|KeyError|unexpected keyword argument|has no attribute", err):
        return "api_missing (new name/field; says nothing about behaviour)"
    return "behaviour (wrong result / did not raise / did not warn)"


def per_test(before: dict[str, str], errors: dict[str, str], after_v: dict[str, str],
             after_m: dict[str, str], tests: list[str]) -> list[dict]:
    rows = []
    for t in tests:
        key = t.replace("::", ".")
        err = errors.get(key)
        # the before logs are -rfE: a collected test that is not listed there passed
        b = before.get(t, "PASSED")
        ok = b == "PASSED"
        rows.append({
            "test": t,
            "before": b,
            "before_failure": None if ok else err,
            "before_failure_kind": None if ok else classify(err),
            "after_vllm_env": after_v.get(t, "NOT RUN"),
            "after_megatron_env": after_m.get(t, "NOT RUN"),
        })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    for name in ("json", "commit", "wave_a_before_log", "tests_before_log", "ttir_before_log",
                 "tests_after_vllm", "tests_after_megatron", "suite_vllm", "suite_commit",
                 "probe_before", "probe_after", "bench_graph"):
        ap.add_argument("--" + name.replace("_", "-"), required=True, dest=name)
    a = ap.parse_args()
    d = json.loads(Path(a.json).read_text())
    after_v, after_m = outcomes(a.tests_after_vllm), outcomes(a.tests_after_megatron)

    # ---- wave A per-test table: why each test failed before (R7)
    wa_before = {t["test"]: t["before_6a61f6e"] for t in d["tests_new"]["per_test"]}
    wa_err = first_errors(a.wave_a_before_log)
    for row in d["tests_new"]["per_test"]:
        b = row["before_6a61f6e"]
        if b == "PASSED":
            row["before_failure_kind"] = None
            continue
        err = "illegal memory access" if "illegal memory access" in b else wa_err.get(
            row["test"].replace("::", "."))
        row["before_failure"] = err
        row["before_failure_kind"] = classify(err)
        row["after_vllm_env_wave_b"] = after_v.get(row["test"], "NOT RUN")
        row["after_megatron_env_wave_b"] = after_m.get(row["test"], "NOT RUN")
    kinds = [r.get("before_failure_kind") or "" for r in d["tests_new"]["per_test"]]
    d["tests_new"]["before_failed_by_kind"] = {
        "behaviour": sum(k.startswith("behaviour") for k in kinds),
        "api_missing": sum(k.startswith("api_missing") for k in kinds),
        "unknown": sum(k.startswith("unknown") for k in kinds),
    }
    d["tests_new"]["note"] = (
        "wave A: of the 18 tests that fail on 6a61f6e, only the 'behaviour' ones detect a bug; "
        "'api_missing' ones fail because a new name or trace field did not exist yet. The "
        "string checks of trace() (offsets, layer selection, tile config) cannot detect a "
        "regression; the int64 guard is now test_compiled_kernels_use_int64_row_offsets (TTIR) "
        "plus the 20 GiB T=524352 test, which skips without the memory.")
    del wa_before

    # ---- wave B tests
    prefixes = ("TestWaveB", "TestF5LargeTokenCounts::test_compiled_kernels")
    wb_tests = sorted(t for t in after_v if t.startswith(prefixes))
    wb_tests += [
        "TestF1RopeTableIdentity::test_canonical_table_is_cached_identical_object",
        "TestF1RopeTableIdentity::test_other_repo_table_builders_are_bitwise_identical_on_cuda",
        "TestF1RopeTableIdentity::test_trace_records_table_provenance",
    ]
    before_b = outcomes(a.tests_before_log)
    errors_b = first_errors(a.tests_before_log)
    ttir_before = outcomes(a.ttir_before_log)
    ttir_err = first_errors(a.ttir_before_log)
    if not ttir_err:  # a grep-filtered log: no section headers, take its first 'E' line
        e_lines = [x[1:].strip() for x in Path(a.ttir_before_log).read_text().splitlines()
                   if x.startswith("E ")]
        ttir_err = {t.replace("::", "."): e_lines[0][:200] for t in ttir_before if e_lines}
    before_b.update(ttir_before)
    errors_b.update(ttir_err)
    rows = per_test(before_b, errors_b, after_v, after_m, wb_tests)
    for r in rows:
        if r["test"].startswith("TestF5LargeTokenCounts::test_compiled"):
            r["before_tree"] = "f91e8b6 tree with the 6a61f6e triton/qknorm_rope.py (int32)"
        else:
            r["before_tree"] = "f91e8b6"

    probe_b = json.loads(Path(a.probe_before).read_text())
    probe_a = json.loads(Path(a.probe_after).read_text())
    graph = json.loads(Path(a.bench_graph).read_text())
    loc = [r for r in graph["series"] if r["local"]]
    glo = [r for r in graph["series"] if not r["local"]]
    has_old = "old_fwd_gpu_us" in graph["series"][0]

    def col(rows, key):
        return [r.get(key) for r in rows]

    findings_b = [
        {"id": "kernels_rope:R1", "severity": "should_fix", "verdict": "fixed (partially; "
         "call sites need integration patch 2)",
         "what": "qknorm_rope:4 was reported 'fixed' while every production caller still used "
                 "local=None. Now: the legacy form is counted (runtime_provenance), warns once "
                 "per process, and raises when MUSEGLIMMER_O4_REQUIRE_LAYER_KIND=1 or "
                 "require_explicit_layer_kind(True); the reference model passes local= "
                 "explicitly. The vLLM/Megatron call sites are other owners' files: patch in "
                 "requests_to_integrator; the E2E harness should set the env var once they "
                 "land. findings[3].verdict corrected.",
         "evidence_before": probe_b.get("r1_layer_kind"),
         "evidence_after": probe_a.get("r1_layer_kind")},
        {"id": "kernels_rope:R2", "severity": "should_fix", "verdict": "fixed for every strict "
         "builder path; fallback builders + runtime digest logging are integration requests",
         "what": "rope_table (the builder the vLLM/Megatron bindings and the reference bind) "
                 "now returns rows of the one canonical cached table per device; the reference "
                 "golden builder delegates to it and raises on CPU. fallback_ops.rope_table and "
                 "strict_kernels._fb_rope_table (bound only when the strict module is missing) "
                 "and canonical_rope_table(verify=True) + digest logging in the runtimes are "
                 "integration patches. The op still cannot tell where a table came from. "
                 "findings[0].verdict corrected.",
         "evidence_before": probe_b.get("r2_builders"),
         "evidence_after": probe_a.get("r2_builders")},
        {"id": "kernels_rope:R3", "severity": "should_fix", "verdict": "fixed",
         "what": "version counters recorded at build; canonical_rope_table/rope_rows/"
                 "rope_table/rope_table_sha256 raise after an in-place change; verify=True "
                 "re-hashes (no memo) and the digest memo is invalidated by the version "
                 "counters. Writes through .data bypass the counters (documented); verify=True "
                 "catches them.",
         "evidence_before": probe_b.get("r3_in_place"),
         "evidence_after": probe_a.get("r3_in_place")},
        {"id": "kernels_rope:R4", "severity": "should_fix", "verdict": "fixed",
         "what": "the first build of a canonical table inside CUDA-graph capture raises "
                 "(nothing is cached); the table is also built as a normal tensor under "
                 "inference_mode so its version counters exist.",
         "evidence_before": {**probe_b.get("r4_capture", {}),
                             "inference_mode": probe_b.get("inference_mode_build")},
         "evidence_after": {**probe_a.get("r4_capture", {}),
                            "inference_mode": probe_a.get("inference_mode_build")}},
        {"id": "kernels_rope:R5", "severity": "should_fix", "verdict": "fixed",
         "what": "scripts committed under reports/muse_glimmer_poc/scripts/rope/ without "
                 "hard-coded worktree paths (run_rope_measurements.sh <out> [<tree>]); "
                 "'scripts' and 'commands' point at them."},
        {"id": "kernels_rope:R6", "severity": "nit", "verdict": "fixed",
         "what": "tests: verify=True with a wrong pinned digest raises; an in-place change "
                 "raises; a .data write is caught by verify=True."},
        {"id": "kernels_rope:R7", "severity": "nit", "verdict": "fixed",
         "what": "test_compiled_kernels_use_int64_row_offsets checks the TTIR (program id "
                 "sign-extended to i64, i64 row offsets for the Q and K tiles); it fails on "
                 "the 6a61f6e kernel ('%t not extended'). tests_new.per_test records the reason "
                 "each test failed before (before_failure_kind)."},
        {"id": "kernels_rope:R8", "severity": "nit", "verdict": "fixed",
         "what": "strict_kernels._fb_rope_table added to the builder-identity test."},
        {"id": "kernels_rope:R9", "severity": "nit", "verdict": "fixed",
         "what": "tests_summary_lines keys name exactly what ran; findings[2].verdict = "
                 "coverage_gap_test_added."},
        {"id": "kernels_rope:R10", "severity": "nit", "verdict": "fixed",
         "what": "bench_series.plot: before_* series removed (host-load confounded); GBps only "
                 "for T >= 8192; new experiments.bench_graph (CUDA-graph replay, device time "
                 "per launch, old vs new kernel interleaved)."},
        {"id": "kernels_rope:R11", "severity": "nit", "verdict": "fixed",
         "what": "trace() is static and says so (scope); rope_table.sha256 -> sha256_pinned; "
                 "runtime_provenance() reports the digest of the table this process built and "
                 "the explicit/implicit layer-kind call counts."},
        {"id": "kernels_rope:R12", "severity": "nit", "verdict": "fixed",
         "what": "REPORT Design usage line now shows rope_rows once per forward + local=."},
    ]

    d["status"] = "complete (wave B review fixes; see wave_b)"
    d["scripts"] = {
        "dir": "reports/muse_glimmer_poc/scripts/rope",
        "probes_wave_a": "reports/muse_glimmer_poc/scripts/rope/probe_findings.py",
        "probes_wave_b": "reports/muse_glimmer_poc/scripts/rope/probe_wave_b.py",
        "bench_series": "reports/muse_glimmer_poc/scripts/rope/bench_series.py",
        "bench_ab": "reports/muse_glimmer_poc/scripts/rope/bench_ab.py",
        "bench_graph": "reports/muse_glimmer_poc/scripts/rope/bench_graph.py",
        "overhead": "reports/muse_glimmer_poc/scripts/rope/overhead_probe.py",
        "runner": "reports/muse_glimmer_poc/scripts/rope/run_rope_measurements.sh",
        "json_builder": ["reports/muse_glimmer_poc/scripts/rope/build_json.py",
                         "reports/muse_glimmer_poc/scripts/rope/update_json_wave_b.py"],
        "note": "committed; the wave A numbers were produced by the same scripts from the session "
                "scratchpad (identical code, hard-coded paths removed)",
    }
    # findings verdicts (R1, R2, R9)
    f = {x["id"]: x for x in d["findings"]}
    f["qknorm_rope:1"]["verdict"] = (
        "fixed_in_module_builder; wave B: every strict builder path reads the canonical table; "
        "fallback builders and runtime digest logging = integration requests")
    f["qknorm_rope:3"]["verdict"] = "coverage_gap_test_added"
    f["qknorm_rope:4"]["verdict"] = (
        "partially_fixed: explicit API (wave A); wave B adds counting, a one-time warning and the "
        "opt-in fail-closed switch; open until the call sites pass local= (integration patch 2)")
    f["qknorm_rope:4"]["residual"] = (
        "local=None is still accepted by default (backward compatible with the vLLM/Megatron "
        "bindings being edited by other owners); MUSEGLIMMER_O4_REQUIRE_LAYER_KIND=1 rejects it.")

    # tests_summary_lines (R9)
    def last_line(log):
        lines = [x for x in Path(log).read_text().splitlines() if x.strip()]
        return lines[-1] if lines else None

    old = d.get("tests_summary_lines", {})
    d["tests_summary_lines"] = {
        "wave_a": {
            "test_qknorm_rope_vllm_env (9b2317b)": old.get("test_qknorm_rope_vllm_env"),
            "test_qknorm_rope_megatron_env (9b2317b; the log contains only "
            "test_qknorm_rope.py)": old.get("test_qknorm_rope+megatron_binding_megatron_env"),
            "muse_glimmer_suite_vllm_env (9b2317b; the failure is the pre-existing lm_head TP2 "
            "test)": old.get("muse_glimmer_suite_vllm_env"),
        },
        "wave_b": {
            "commit": a.commit,
            "test_qknorm_rope+test_strict_model_tiny_vllm_env": last_line(a.tests_after_vllm),
            "test_qknorm_rope+test_megatron_strict_binding+test_strict_model_tiny_megatron_env":
                last_line(a.tests_after_megatron),
            f"muse_glimmer_suite_vllm_env (--ignore distributed; at {a.suite_commit})":
                last_line(a.suite_vllm),
        },
    }

    # bench_series plot (R10)
    bs = d["experiments"]["bench_series"]
    plot = bs.get("plot", {})
    for k in [k for k in plot if k.startswith("before_")]:
        plot.pop(k)
    ts = bs["T"]
    if "after_local_fwd_GBps" in plot:
        plot["after_local_fwd_GBps_T_ge_8192"] = [
            g if t >= 8192 else None for t, g in zip(ts, plot.pop("after_local_fwd_GBps"))]
    bs["plot_note"] = (
        "eager launcher timings (CUDA events around back-to-back calls): launch-bound up to "
        "T~4096 (the GPU work is shorter than the host launch path), so they measure host "
        "overhead and depend on host load; the 'before' raw series ran at a different host load "
        "and is not plotted (not a code effect). Device time: experiments.bench_graph. GBps only "
        "for T >= 8192.")

    d["experiments"]["bench_graph"] = {
        **{k: v for k, v in graph.items() if k != "series"},
        "commit": a.commit,
        "series": graph["series"],
        "plot": {
            "T": [r["T"] for r in loc],
            "local_new_fwd_gpu_us": col(loc, "new_fwd_gpu_us"),
            "local_new_bwd_gpu_us": col(loc, "new_bwd_gpu_us"),
            "local_new_fwd_eager_us": col(loc, "new_fwd_eager_us"),
            "global_new_fwd_gpu_us": col(glo, "new_fwd_gpu_us"),
            "local_new_fwd_GBps_T_ge_8192": col(loc, "new_fwd_GBps"),
            **({"local_old_fwd_gpu_us": col(loc, "old_fwd_gpu_us"),
                "local_old_bwd_gpu_us": col(loc, "old_bwd_gpu_us")} if has_old else {}),
        },
        "all_bitwise_equal_old_vs_new": (all(r["bitwise_equal_fwd"] and r["bitwise_equal_bwd"]
                                             for r in graph["series"]) if has_old else None),
    }
    d["wave_b"] = {
        "commit": a.commit,
        "commit_before": "f91e8b6",
        "review": "adversarial review of wave A (issues kernels_rope:R1..R12)",
        "findings": findings_b,
        "tests": {
            "count": len(rows),
            "before_failed": sum(r["before"] in ("FAILED", "ERROR") for r in rows),
            "before_failed_behaviour": sum((r["before_failure_kind"] or "").startswith(
                "behaviour") for r in rows),
            "before_failed_api_missing": sum((r["before_failure_kind"] or "").startswith(
                "api_missing") for r in rows),
            "after_passed_vllm_env": sum(r["after_vllm_env"] == "PASSED" for r in rows),
            "after_passed_megatron_env": sum(r["after_megatron_env"] == "PASSED" for r in rows),
            "per_test": rows,
        },
        "probe_wave_b": {"before_f91e8b6": probe_b, "after": probe_a},
        "bits_changed": "none: rope_table rows are gathered from the canonical table, which is "
                        "the same formula (tests compare against the formula evaluated "
                        "directly); xrt_reference_triton.json was regenerated at this commit",
    }
    d["not_run"] = [
        "vLLM engine / Megatron Bridge 30B runs with the wave B module: NOT RUN by this row "
        "(other owners); the call-site patches are requests_to_integrator",
    ]
    d["suggested_figures"] = [x for x in d.get("suggested_figures", [])
                              if x.get("id") != "fig_latency"] + [
        {"id": "fig_latency_graph", "kind": "log-log line, us vs T",
         "data": "experiments.bench_graph.plot",
         "caption": "Device time per launch (CUDA-graph replay) of the strict fwd/bwd kernels, "
                    "old int32 vs new int64 offsets, and the eager launcher time (host launch "
                    "floor) for comparison."},
    ]
    Path(a.json).write_text(json.dumps(d, indent=1) + "\n")
    print("wrote", a.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
