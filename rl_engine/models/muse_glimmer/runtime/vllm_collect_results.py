# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Collect the ``run_vllm_strict.py`` outputs into ``reports/muse_glimmer_poc/data/vllm_runtime.json``.

Usage::

    python -m rl_engine.models.muse_glimmer.runtime.vllm_collect_results \
        --runs <dir with *.json run files> --data reports/muse_glimmer_poc/data \
        --extra <dir>/tests.json --out reports/muse_glimmer_poc/data/vllm_runtime.json

Every number in the output is copied from a run file written by an actual engine run (or from the
``--extra`` JSON of unit-test / simulation outcomes); anything missing is reported as
``"NOT RUN"``.  Raw series (per token, per step) are kept so the report can plot them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path
from typing import Any

XRT_FILES = {
    ("strict", 1): "xrt_vllm_strict_tp1.json",
    ("strict", 2): "xrt_vllm_strict_tp2.json",
    ("native", 1): "xrt_vllm_native_tp1.json",
    ("native", 2): "xrt_vllm_native_tp2.json",
}
XRT_LORA_FILES = {
    ("strict", 1): "xrt_vllm_strict_lora_tp1.json",
    ("strict", 2): "xrt_vllm_strict_lora_tp2.json",
    ("native", 1): "xrt_vllm_native_lora_tp1.json",
}


def _f32(hex_str: str) -> float:
    return struct.unpack(">f", bytes.fromhex(hex_str))[0]


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def compare_xrt(a: dict, b: dict) -> dict[str, Any]:
    """Token-level bitwise comparison of two muse_glimmer_xrt_logprobs_v1 files."""

    by_id = {s["id"]: s["logprobs_hex"] for s in b["sequences"]}
    total = mismatched = 0
    max_abs = sum_abs = 0.0
    per_seq = []
    for seq in a["sequences"]:
        ha, hb = seq["logprobs_hex"], by_id[seq["id"]]
        if len(ha) != len(hb):
            raise ValueError(f"{seq['id']}: {len(ha)} vs {len(hb)} tokens")
        diffs = [abs(_f32(x) - _f32(y)) for x, y in zip(ha, hb)]
        count = sum(x != y for x, y in zip(ha, hb))
        per_seq.append(
            {
                "id": seq["id"],
                "tokens": len(ha),
                "mismatched": count,
                "max_abs": max(diffs or [0.0]),
                "abs_diff_per_token": diffs,
            }
        )
        total += len(ha)
        mismatched += count
        max_abs = max(max_abs, max(diffs or [0.0]))
        sum_abs += sum(diffs)
    return {
        "a": f"{a['mode']} tp{a['tp']}",
        "mean_abs_diff": sum_abs / total if total else 0.0,
        "b": f"{b['mode']} tp{b['tp']}",
        "tokens": total,
        "mismatched": mismatched,
        "bitwise_equal": mismatched == 0,
        "sequences_byte_identical": json.dumps(a["sequences"], sort_keys=True)
        == json.dumps(b["sequences"], sort_keys=True),
        "max_abs_diff": max_abs,
        "per_sequence": per_seq,
    }


def _xrt_summary(doc: dict) -> dict[str, Any]:
    blob = json.dumps(doc["sequences"], sort_keys=True).encode()
    return {
        "mode": doc["mode"],
        "tp": doc["tp"],
        "backend": doc.get("backend"),
        "commit": doc.get("commit"),
        "lora_sha256": doc.get("lora_sha256"),
        "sequences": len(doc["sequences"]),
        "tokens": sum(len(s["logprobs_hex"]) for s in doc["sequences"]),
        "sequences_sha256": hashlib.sha256(blob).hexdigest(),
        "score_ms": doc.get("score_ms"),
        "logprobs_per_sequence": {
            s["id"]: [_f32(h) for h in s["logprobs_hex"]] for s in doc["sequences"]
        },
    }


def _generate_records(run: dict) -> list[dict[str, Any]]:
    out = []
    kwargs = (run.get("meta") or {}).get("engine_kwargs") or {}
    engine_lora = str(kwargs.get("enable_lora", "False")) == "True"
    for gen in run.get("generate", []):
        check = gen.get("decode_vs_prefill") or {}
        out.append(
            {
                "run": run["_file"],
                "mode": run["mode"],
                "tp": run["tp"],
                "engine_lora_enabled": engine_lora,
                "lora": None,
                "temperature": gen["temperature"],
                "prompts": len(gen["token_ids"]),
                "max_tokens": gen["max_tokens"],
                "generated_tokens": gen["generated_tokens"],
                "generate_ms": gen["generate_ms"],
                "tokens_per_s": gen["tokens_per_s"],
                "ids": gen["ids"],
                "prompt_lens": gen["prompt_lens"],
                "token_ids": gen["token_ids"],
                "logprobs": gen["logprobs"],
                "finish_reasons": gen["finish_reasons"],
                "texts": gen["texts"],
                "decode_vs_prefill": {
                    k: check.get(k)
                    for k in (
                        "tokens",
                        "mismatched",
                        "bitwise_equal",
                        "max_abs_diff",
                        "first_mismatch",
                        "per_sequence",
                        "abs_diff_per_token",
                        "prefill_seconds",
                    )
                }
                if check
                else "NOT RUN",
            }
        )
    lora = run.get("lora_random") or {}
    for gen in lora.get("generate", []):
        check = gen.get("decode_vs_prefill") or {}
        out.append(
            {
                "run": run["_file"],
                "mode": run["mode"],
                "tp": run["tp"],
                "engine_lora_enabled": engine_lora,
                "lora": lora.get("sha256"),
                "temperature": gen["temperature"],
                "prompts": len(gen["token_ids"]),
                "max_tokens": gen["max_tokens"],
                "generated_tokens": gen["generated_tokens"],
                "generate_ms": gen["generate_ms"],
                "tokens_per_s": gen["tokens_per_s"],
                "token_ids": gen["token_ids"],
                "logprobs": gen["logprobs"],
                "texts": gen["texts"],
                "decode_vs_prefill": {
                    k: check.get(k)
                    for k in (
                        "tokens",
                        "mismatched",
                        "bitwise_equal",
                        "max_abs_diff",
                        "per_sequence",
                        "abs_diff_per_token",
                    )
                },
            }
        )
    return out


def _runs_in(directory: Path | None) -> list[dict]:
    runs = []
    if directory is None:
        return runs
    for path in sorted(directory.glob("*.json")):
        doc = _load(path)
        if doc.get("schema") == "muse_glimmer_vllm_run_v1":
            doc["_file"] = f"{directory.name}/{path.name}"
            runs.append(doc)
    return runs


def collect(
    runs_dir: Path,
    data_dir: Path,
    extra: dict[str, Any] | None,
    references: list[Path] | None = None,
    bench_dir: Path | None = None,
) -> dict[str, Any]:
    runs = _runs_in(runs_dir)
    bench_runs = _runs_in(bench_dir)
    perf_runs = bench_runs if bench_runs else runs
    result: dict[str, Any] = {
        "row": "vllm_runtime_adapter",
        "schema": "muse_glimmer_vllm_runtime_results_v1",
        "commits_measured": sorted({r.get("commit") for r in runs if r.get("commit")}),
        "env": next((r["env"] for r in runs if r.get("env")), {}),
        "commands": [r["command"] for r in runs + bench_runs],
        "runs": [
            {
                "file": r["_file"],
                "mode": r["mode"],
                "tp": r["tp"],
                "status": r.get("status"),
                "commit": r.get("commit"),
                "engine_load_s": r.get("engine_load_s"),
                "command": r["command"],
                "error": r.get("error"),
            }
            for r in runs + bench_runs
        ],
        "performance_source": "isolated bench runs" if bench_runs else "experiment runs",
    }
    result["generate"] = [g for r in runs for g in _generate_records(r)]
    result["lora_zero"] = [
        {"mode": r["mode"], "tp": r["tp"], **{k: v for k, v in r["lora_zero"].items()}}
        for r in runs
        if r.get("lora_zero")
    ] or "NOT RUN"
    lora_random = []
    for r in runs:
        rec = r.get("lora_random")
        if not rec:
            continue
        lora_random.append(
            {
                "mode": r["mode"],
                "tp": r["tp"],
                **{k: v for k, v in rec.items() if k != "generate"},
            }
        )
    result["lora_random"] = lora_random or "NOT RUN"
    xrt: dict[str, Any] = {}
    docs: dict[tuple[str, int], dict] = {}
    for key, name in XRT_FILES.items():
        path = data_dir / name
        if path.exists():
            docs[key] = _load(path)
            xrt[name] = _xrt_summary(docs[key])
        else:
            xrt[name] = "NOT RUN"
    lora_docs: dict[tuple[str, int], dict] = {}
    for key, name in XRT_LORA_FILES.items():
        path = data_dir / name
        if path.exists():
            lora_docs[key] = _load(path)
            xrt[name] = _xrt_summary(lora_docs[key])
        else:
            xrt[name] = "NOT RUN"
    comparisons = {}
    pairs = {
        "strict_tp2_vs_strict_tp1": (docs, ("strict", 2), ("strict", 1)),
        "native_tp2_vs_native_tp1": (docs, ("native", 2), ("native", 1)),
        "strict_tp1_vs_native_tp1": (docs, ("strict", 1), ("native", 1)),
        "strict_tp2_vs_native_tp2": (docs, ("strict", 2), ("native", 2)),
        "strict_lora_tp2_vs_strict_lora_tp1": (lora_docs, ("strict", 2), ("strict", 1)),
        "strict_lora_tp1_vs_native_lora_tp1": (lora_docs, ("strict", 1), ("native", 1)),
    }
    for name, (pool, ka, kb) in pairs.items():
        if ka in pool and kb in pool:
            comparisons[name] = compare_xrt(pool[ka], pool[kb])
        else:
            comparisons[name] = "NOT RUN"
    if ("strict", 1) in lora_docs and ("strict", 1) in docs:
        comparisons["strict_lora_tp1_vs_strict_tp1"] = compare_xrt(
            lora_docs[("strict", 1)], docs[("strict", 1)]
        )
    for ref_path in references or []:
        ref = _load(ref_path)
        label = "".join(
            c for c in ref.get("backend", ref_path.name).split("mode=")[-1] if c.isalnum() or c == "_"
        )
        xrt[f"reference_chain_{label}"] = {
            k: v for k, v in _xrt_summary(ref).items() if k != "logprobs_per_sequence"
        }
        xrt[f"reference_chain_{label}"]["file"] = str(ref_path)
        for key in (("strict", 1), ("strict", 2), ("native", 1), ("native", 2)):
            if key in docs:
                comparisons[f"{key[0]}_tp{key[1]}_vs_reference_chain_{label}"] = compare_xrt(
                    docs[key], ref
                )
    xrt["comparisons"] = comparisons
    result["xrt"] = xrt
    bench = []
    for r in perf_runs:
        if not r.get("bench"):
            continue
        b = dict(r["bench"])
        b["mode"], b["tp"] = r["mode"], r["tp"]
        bench.append(b)
    result["throughput"] = bench or "NOT RUN"
    profiles = []
    for r in perf_runs:
        if not r.get("profile"):
            continue
        entry: dict[str, Any] = {"mode": r["mode"], "tp": r["tp"]}
        for kind, prof in r["profile"].items():
            if not prof:
                entry[kind] = "NOT CAPTURED"
                continue
            prof = dict(prof)
            if "kernel_cuda_ms" not in prof:
                # Older captures: kernel rows are the ones without CPU time; top-25 only.
                prof["kernel_cuda_ms_top25"] = sum(
                    o["self_cuda_ms"] for o in prof["top_ops"] if o["self_cpu_ms"] == 0.0
                )
                prof.pop("total_self_cuda_ms", None)
            entry[kind] = prof
        profiles.append(entry)
    result["profile"] = profiles or "NOT RUN"
    provenance = []
    for r in runs:
        meta = r.get("meta") or {}
        for w in meta.get("workers", []):
            provenance.append(
                {
                    "run": r["_file"],
                    "tp_rank": w.get("tp_rank"),
                    "hooks": w.get("hooks"),
                    "op_impls": w.get("op_impls"),
                    "fallbacks": w.get("fallbacks"),
                    "demotions": w.get("demotions"),
                    "all_ops_strict": w.get("all_ops_strict"),
                    "backend_ids": w.get("backend_ids"),
                    "call_counts": w.get("call_counts"),
                    "kv_ownership": w.get("kv_ownership"),
                    "checks": w.get("checks"),
                    "notes": w.get("notes"),
                }
            )
    result["worker_provenance"] = provenance or "NOT RUN"
    if extra:
        result.update(extra)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", required=True)
    parser.add_argument("--bench-runs", default=None, help="isolated throughput/profile runs")
    parser.add_argument("--data", default="reports/muse_glimmer_poc/data")
    parser.add_argument("--extra", default=None, help="JSON merged at top level")
    parser.add_argument(
        "--reference", action="append", default=[], help="reference-chain xrt file(s)"
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    extra = _load(Path(args.extra)) if args.extra else None
    result = collect(
        Path(args.runs),
        Path(args.data),
        extra,
        [Path(p) for p in args.reference],
        Path(args.bench_runs) if args.bench_runs else None,
    )
    Path(args.out).write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps(result["xrt"].get("comparisons", {}), default=str)[:2000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
