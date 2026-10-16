"""Assemble reports/muse_glimmer_poc/data/kernels_gemm.json from the probe/bench outputs."""

import json
import re
import sys
from pathlib import Path

S = Path(__file__).resolve().parent
OUT = Path(sys.argv[1])
COMMIT = sys.argv[2]
BASE = "6a61f6e791b7b42a4b5746b2b974f2431f524994"


def load(name):
    p = S / name
    return json.loads(p.read_text()) if p.exists() else None


def pytest_summary(log_name):
    p = S / log_name
    if not p.exists():
        return None
    text = p.read_text()
    m = re.findall(r"=+ (.*?) in ([\d.]+)s", text)
    tail = [ln for ln in text.splitlines() if re.search(r"\d+ (passed|failed)", ln)]
    return {"log": str(p), "summary_line": tail[-1].strip() if tail else None,
            "timing": m[-1] if m else None}


def finding_records():
    return json.loads((S / "verdicts.json").read_text())


probe_before, probe_after = load("probe_before.json"), load("probe_after.json")


def bit_ledger():
    before, after = load("bits_before.json"), load("bits_after.json")
    rows = []
    for key, vb in before.items():
        va = after.get(key)
        if isinstance(vb, dict) and isinstance(va, dict):
            for sub in vb:
                rows.append({"path": f"{key}:{sub}", "before": vb[sub], "after": va.get(sub),
                             "changed": vb[sub] != va.get(sub)})
        elif isinstance(va, dict):  # before raised
            rows.append({"path": key, "before": vb, "after": va, "changed": True,
                         "note": "raised on the base commit"})
        else:
            rows.append({"path": key, "before": vb, "after": va, "changed": vb != va})
    fwd = [r for r in rows if r["path"].split("/")[0] in
           ("fwd_bf16", "fwd_fp32", "tp_partial", "ffn_fwd", "lm_head_fwd")]
    return {
        "description": (
            "SHA-256 (first 16 hex) of the output bytes for identical seeded inputs on the base "
            "commit and on the fixed code; 'changed' = bits differ"
        ),
        "summary": {
            "entries": len(rows),
            "changed": sum(r["changed"] for r in rows),
            "forward_entries": len(fwd),
            "forward_changed": sum(r["changed"] for r in fwd),
        },
        "rows": rows,
    }
bench_before, bench_after = load("bench_before.json"), load("bench_after.json")
if bench_after:
    bench_after["env"]["commit"] = "87cba55 (fixed code at benchmark time)"
if bench_before:
    bench_before["env"]["commit"] = BASE

BASE_TEST_REASONS = {
    "description": (
        "the new/updated tests run against the base-commit code (87 failed, 117 passed, "
        "1 skipped); failures grouped by reason (pytest --tb=line)"
    ),
    "bitwise_or_behaviour_failures": {
        "tile variants at K=256/512: ITERS=1 plan (assert chunk/BLOCK_K >= 2) or bits": 12,
        "dA over N=256 != declared tree (M64, M4096)": 2,
        "dW tokens != declared tree (T255, T256, T300, T512)": 4,
        "FFN default-gemm backward raised ValueError (T=1, 64, 255, 300 x2)": 5,
        "lm_head dW(T) != dW(zero-padded) (T=40, 255, 300)": 3,
        "lm_head host syncs: 2 instead of 1": 1,
        "silent fallbacks (swiglu adapter, lm_head legacy) in the import-block subprocess": 2,
        "adapter K=36 did not raise": 1,
        "trace content (swiglu gemm stub, lm_head 'block order', dw rule key)": 3,
        "plan_chunks expectations (BLOCK_K 16/32 for 32/64-wide chunks)": 2,
    },
    "new_api_failures": (
        "AttributeError/TypeError on APIs added by the fix (MIN_ITERS, column/row-parallel "
        "entries, ragged_k, fast_det_gemm.trace, _VERIFIED_IR): the remaining failures; "
        "the bit-level evidence for those shapes is in det_gemm_1_iters_probe"
    ),
}


def pair_series(section, keys):
    """Join before/after rows of a bench section on ``keys``."""
    rows = []
    if not bench_before or not bench_after:
        return rows
    after = {tuple(r.get(k) for k in keys): r for r in bench_after[section]}
    for b in bench_before[section]:
        key = tuple(b.get(k) for k in keys)
        a = after.get(key)
        row = {k: b.get(k) for k in keys}
        for f in ("strict_ms", "strict_tflops", "cublas_ms", "native_ms", "strict_peak_gib",
                  "native_peak_gib", "block_k", "chunk_len", "shape_class"):
            if f in b:
                row[f + "_before"] = b[f]
            if a is not None and f in a:
                row[f + "_after"] = a[f]
        if b.get("strict_ms_error"):
            row["error_before"] = b["strict_ms_error"]
        if a is not None and a.get("strict_ms_error"):
            row["error_after"] = a["strict_ms_error"]
        rows.append(row)
    return rows


rec = {
    "row": "gemm_correctness_followups (fast tree GEMM + FFN / LM-head adapters)",
    "status": "complete",
    "commit": COMMIT,
    "base_commit": BASE,
    "env": {
        "gpu": "NVIDIA A100-SXM4-80GB (sm80)",
        "torch": "2.13.0+cu130",
        "triton": "3.7.1",
        "python_env": "~/envs/vllm (vLLM 0.31.0 env; Megatron env not needed for this row)",
        "gpu_pool": "4-5 via ~/bin/gpu-lease -n 1",
    },
    "commands": json.loads((S / "commands.json").read_text()),
    "headline": [
        "det_gemm:1: with 32/64-wide K chunks Triton folded the tree adds into the MMA chain "
        "(3 of 7 adds left); 48-67 % of outputs differed from the declared tree for K=256/512, "
        "k/v-proj dA and dW at 32 < T <= 512. Fixed (>= 2 BLOCK_K iterations, static assert, "
        "TTGIR guard): 0 mismatches.",
        "det_gemm:2: rank-local autograd at TP2/4/8 differs from TP1 in 37-41 % of dA elements; "
        "the new column/row-parallel autograd shards (fp32 sub-tree partials + canonical tree) "
        "give 0 mismatches.",
        "No Muse-Glimmer forward output changed (bit ledger: 101/101 forward hashes equal); "
        "53/126 backward hashes changed, all in the shapes listed in findings[].bits_changed.",
        "FFN backward now works for any T (was ValueError for T % 256 != 0); lm_head fwd+bwd at "
        "T=4001: 975.6 -> 405.6 ms; FFN fwd+bwd T=4096: 87.9 -> 79.3 ms.",
    ],
    "suggested_figures": [
        {"title": "Kernel vs declared tree: mismatching outputs before/after",
         "data": "det_gemm_1_iters_probe.{before,after}.forward_sweep (x=K, y=mismatch/numel, "
                 "series=M) and .backward_paths"},
        {"title": "Compiled IR: tree adds present (TTIR/TTGIR addf, PTX add.f32) by ITERS",
         "data": "det_gemm_1_iters_probe.{before,after}.ir"},
        {"title": "TP dA: naive rank-local backward vs canonical-tree backward",
         "data": "det_gemm_2_tp_probe.column_parallel (x=tp, y=naive_mismatch/numel vs "
                 "canonical_mismatch_per_rank)"},
        {"title": "Which bits changed (ledger heat map)",
         "data": "bit_ledger.rows (path family x shape x T, changed true/false)"},
        {"title": "Before/after time per shape",
         "data": "benchmarks.gemm_forward / gemm_backward_kernels / ffn / lm_head "
                 "(strict_ms_before vs strict_ms_after)"},
        {"title": "lm_head host syncs per forward",
         "data": "lm_head_host_syncs.{before,after}"},
    ],
    "findings": finding_records(),
    "det_gemm_1_iters_probe": {
        "description": (
            "kernel (8 chunks in one launch) vs the declared tree evaluated outside the kernel "
            "(8 single-chunk launches + canonical combine_tree in torch fp32); mismatch = number "
            "of fp32 outputs that differ bitwise. ITERS = chunk_len / BLOCK_K."
        ),
        "before": probe_before,
        "after": probe_after,
    },
    "det_gemm_2_tp_probe": load("probe_tp.json"),
    "lm_head_host_syncs": {"before": load("syncs_before.json"), "after": load("syncs_after.json")},
    "bit_ledger": bit_ledger(),
    "decode_latency_repeats": {
        "description": (
            "median of 200 (GEMM) / 100 (FFN, lm_head) CUDA-event timed calls, ms; runs "
            "alternated before/after twice on a shared node: run-to-run noise of the SAME code "
            "is up to 25 % for lm_head T=1 (2.91 vs 3.64 ms on the base commit), so decode "
            "latency differences below that are not significant"
        ),
        "runs": {f"{tag}_{i}": load(f"small_{tag}_{i}.json")
                 for tag in ("before", "after") for i in (1, 2)},
        "lm_head_T1_gpu_kernel_time_ms_per_call": {
            "before": 47.717 / 20, "after": 47.792 / 20,
            "source": "torch.profiler CUDA self time over 20 calls (prof_lm_gpu.py)",
        },
    },
    "megatron_binding_lm_head_tp_test": {
        "test": "tests/models/muse_glimmer/test_megatron_strict_binding.py::"
                "test_lm_head_tiles_tp2_equals_tp1_and_reference",
        "status": "fails before and after this row's changes (pre-existing)",
        "root_cause": (
            "test helper RecordReplayComm: in the record pass all_gather returns the rank's own "
            "tensor for every peer, so the forward logZ of the record pass is wrong and the dh "
            "partials it records for the peers (used in the replay pass) were computed with that "
            "wrong logZ; a second record pass that replays the correct forward gathers fixes it"
        ),
        "probe": "probe_megatron_lmhead.py: original helper dh equal [False, False]; fixed helper "
                 "TP2/TP4/TP8 lp and dh equal on every rank; binding TP1 == sibling op True",
    },
    "benchmarks": {
        "description": (
            "median CUDA-event ms; strict = this row's kernels, cublas/native = torch bf16 with "
            "TF32 off; before = base commit, after = fixed commit; same GPU model, leased one "
            "GPU of pool 4-5 (other agents may share the node, so +-5 % noise)"
        ),
        "gemm_forward": pair_series("gemm_forward", ["shape", "M", "N", "K"]),
        "gemm_backward_kernels": pair_series("gemm_backward_kernels",
                                             ["kind", "shape", "M", "tokens"]),
        "gemm_autograd_fwd_bwd": pair_series("gemm_autograd", ["shape", "M"]),
        "ffn": pair_series("ffn", ["pass", "T"]),
        "lm_head": pair_series("lm_head", ["pass", "T"]),
        "raw_before": bench_before,
        "raw_after": bench_after,
    },
    "tests": {
        "after_fix_owned_files": pytest_summary("tests_after_owned.log"),
        "new_tests_on_base_commit": pytest_summary("tests_on_before_tbline.log"),
        "new_tests_on_base_commit_reasons": BASE_TEST_REASONS,
        "full_single_gpu_suite_after": pytest_summary("suite_after.log"),
    },
}
OUT.write_text(json.dumps(rec, indent=1) + "\n")
print("wrote", OUT, OUT.stat().st_size, "bytes")
