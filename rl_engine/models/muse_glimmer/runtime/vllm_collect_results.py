# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Collect ``run_vllm_strict.py`` outputs into ``reports/muse_glimmer_poc/data/vllm_runtime.json``.

Usage::

    python -m rl_engine.models.muse_glimmer.runtime.vllm_collect_results \
        --runs <dir with *.json run files> --data reports/muse_glimmer_poc/data \
        --reference <ref>_train.json --reference <ref>_prefill.json \
        --extra <dir>/extra.json --out reports/muse_glimmer_poc/data/vllm_runtime.json

Every number in the output is copied from a run file written by an actual engine run (or from the
``--extra`` JSON of unit-test / simulation outcomes); anything missing is reported as
``"NOT RUN"``.  Raw series (per token, per step) are kept so the report can plot them.  The
reference-chain dumps passed with ``--reference`` are embedded whole (with their commit), so the
JSON carries the ground truth the vLLM dumps are compared with.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import struct
from pathlib import Path
from typing import Any

# (mode, tp, lora) -> file in the data dir
XRT_FILES: dict[tuple[str, int, bool], str] = {
    ("strict", 1, False): "xrt_vllm_strict_tp1.json",
    ("strict", 2, False): "xrt_vllm_strict_tp2.json",
    ("strict", 4, False): "xrt_vllm_strict_tp4.json",
    ("native", 1, False): "xrt_vllm_native_tp1.json",
    ("native", 2, False): "xrt_vllm_native_tp2.json",
    ("strict", 1, True): "xrt_vllm_strict_lora_tp1.json",
    ("strict", 2, True): "xrt_vllm_strict_lora_tp2.json",
    ("strict", 4, True): "xrt_vllm_strict_lora_tp4.json",
    ("native", 1, True): "xrt_vllm_native_lora_tp1.json",
}
# name -> (a key, b key)
XRT_PAIRS: dict[str, tuple[tuple[str, int, bool], tuple[str, int, bool]]] = {
    "strict_tp2_vs_strict_tp1": (("strict", 2, False), ("strict", 1, False)),
    "strict_tp4_vs_strict_tp1": (("strict", 4, False), ("strict", 1, False)),
    "native_tp2_vs_native_tp1": (("native", 2, False), ("native", 1, False)),
    "strict_tp1_vs_native_tp1": (("strict", 1, False), ("native", 1, False)),
    "strict_tp2_vs_native_tp2": (("strict", 2, False), ("native", 2, False)),
    "strict_lora_tp2_vs_strict_lora_tp1": (("strict", 2, True), ("strict", 1, True)),
    "strict_lora_tp4_vs_strict_lora_tp1": (("strict", 4, True), ("strict", 1, True)),
    "strict_lora_tp1_vs_native_lora_tp1": (("strict", 1, True), ("native", 1, True)),
    "strict_lora_tp1_vs_strict_tp1": (("strict", 1, True), ("strict", 1, False)),
    "native_lora_tp1_vs_native_tp1": (("native", 1, True), ("native", 1, False)),
}


def _f32(hex_str: str) -> float:
    return struct.unpack(">f", bytes.fromhex(hex_str))[0]


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def xrt_label(doc: dict) -> str:
    """Human label of an xrt dump: runtime/mode, LoRA marker, TP (reference chain: its mode)."""

    if str(doc.get("runtime", "")).startswith("reference"):
        backend = str(doc.get("backend", ""))
        mode = backend.split("mode=")[-1] if "mode=" in backend else str(doc.get("mode", ""))
        return f"reference chain {mode}".strip()
    lora = " lora" if doc.get("lora_sha256") else ""
    return f"{doc['mode']}{lora} tp{doc['tp']}"


def compare_xrt(
    a: dict, b: dict, *, a_label: str | None = None, b_label: str | None = None
) -> dict[str, Any]:
    """Token-level bitwise comparison of two muse_glimmer_xrt_logprobs_v1 files."""

    by_id = {s["id"]: s["logprobs_hex"] for s in b["sequences"]}
    total = mismatched = nonzero = 0
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
        nonzero += sum(1 for x in ha if _f32(x) != 0.0)
        max_abs = max(max_abs, max(diffs or [0.0]))
        sum_abs += sum(diffs)
    return {
        "a": a_label or xrt_label(a),
        "b": b_label or xrt_label(b),
        "a_commit": a.get("commit"),
        "b_commit": b.get("commit"),
        "tokens": total,
        "nonzero_tokens_a": nonzero,
        "mismatched": mismatched,
        "bitwise_equal": mismatched == 0,
        "sequences_byte_identical": json.dumps(a["sequences"], sort_keys=True)
        == json.dumps(b["sequences"], sort_keys=True),
        "max_abs_diff": max_abs,
        "mean_abs_diff": sum_abs / total if total else 0.0,
        "per_sequence": per_seq,
    }


def _xrt_summary(doc: dict, *, embed: bool = False) -> dict[str, Any]:
    blob = json.dumps(doc["sequences"], sort_keys=True).encode()
    out = {
        "label": xrt_label(doc),
        "mode": doc.get("mode"),
        "tp": doc.get("tp"),
        "runtime": doc.get("runtime"),
        "backend": doc.get("backend"),
        "commit": doc.get("commit"),
        "git_dirty": doc.get("git_dirty"),
        "git_status_porcelain": doc.get("git_status_porcelain"),
        "lora_sha256": doc.get("lora_sha256"),
        "sequences": len(doc["sequences"]),
        "tokens": sum(len(s["logprobs_hex"]) for s in doc["sequences"]),
        "positive_logprobs": sum(
            1 for s in doc["sequences"] for h in s["logprobs_hex"] if _f32(h) > 0.0
        ),
        "sequences_sha256": hashlib.sha256(blob).hexdigest(),
        "score_ms": doc.get("score_ms"),
        "logprobs_per_sequence": {
            s["id"]: [_f32(h) for h in s["logprobs_hex"]] for s in doc["sequences"]
        },
    }
    if embed:
        out["document"] = doc
    return out


_CHECK_KEYS = (
    "tokens",
    "mismatched",
    "bitwise_equal",
    "nonzero_tokens",
    "tokens_lp_below_1e-3",
    "tokens_lp_below_0.1",
    "positive_tokens",
    "max_abs_diff",
    "first_mismatch",
    "per_sequence",
    "abs_diff_per_token",
    "prefill_seconds",
    "prefill_logprobs",
)


def _check(check: dict | None) -> Any:
    if not check:
        return "NOT RUN"
    out = {k: check.get(k) for k in _CHECK_KEYS if k in check}
    if out.get("mismatched") == 0 and "prefill_logprobs" in out:
        # Bitwise equal to the decode logprobs stored next to this record: do not store twice.
        out["prefill_logprobs"] = "bitwise identical to the record's logprobs"
    return out


def _engine_flags(run: dict) -> dict[str, Any]:
    kwargs = run.get("engine_kwargs") or (run.get("meta") or {}).get("engine_kwargs") or {}
    return {
        "engine_lora_enabled": str(kwargs.get("enable_lora", "False")) == "True",
        "enforce_eager": str(kwargs.get("enforce_eager", "True")) == "True",
        "max_num_batched_tokens": kwargs.get("max_num_batched_tokens", "default (8192)"),
        "max_num_seqs": kwargs.get("max_num_seqs"),
    }


def _gen_record(run: dict, gen: dict, lora: str | None) -> dict[str, Any]:
    return {
        "run": run["_file"],
        "commit": run.get("commit"),
        "mode": run["mode"],
        "tp": run["tp"],
        **_engine_flags(run),
        "lora": lora,
        "temperature": gen["temperature"],
        "prompts": len(gen["token_ids"]),
        "max_tokens": gen["max_tokens"],
        "generated_tokens": gen["generated_tokens"],
        "generate_ms": gen["generate_ms"],
        "tokens_per_s": gen["tokens_per_s"],
        "ids": gen.get("ids"),
        "prompt_lens": gen.get("prompt_lens"),
        "token_ids": gen["token_ids"],
        "logprobs": gen["logprobs"],
        "logprobs_hex": gen.get("logprobs_hex"),
        "finish_reasons": gen.get("finish_reasons"),
        "texts": gen.get("texts"),
        "decode_vs_prefill": _check(gen.get("decode_vs_prefill")),
    }


def _generate_records(run: dict) -> list[dict[str, Any]]:
    out = [_gen_record(run, gen, None) for gen in run.get("generate", [])]
    lora = run.get("lora_random") or {}
    out += [_gen_record(run, gen, lora.get("sha256")) for gen in lora.get("generate", [])]
    return out


def _contract_records(run: dict) -> list[dict[str, Any]]:
    out = []
    for rec in run.get("contract_shapes", []):
        steps = dict(rec.get("engine_steps") or {})
        out.append(
            {
                "run": run["_file"],
                "commit": run.get("commit"),
                "mode": run["mode"],
                "tp": run["tp"],
                **_engine_flags(run),
                **{k: v for k, v in rec.items() if k not in ("decode_vs_prefill", "engine_steps")},
                "decode_vs_prefill": _check(rec.get("decode_vs_prefill")),
                "engine_steps": steps,
            }
        )
    return out


def _code_dirty(run: dict) -> Any:
    """True if a run's tree had modified files outside ``reports/`` (data-only edits are not
    code changes: the runs themselves rewrite the xrt dumps)."""

    if "git_dirty_code" in run:
        return run["git_dirty_code"]
    lines = run.get("git_status_porcelain")
    if lines is None:
        return run.get("git_dirty")
    return any(not line[3:].startswith("reports/") for line in lines)


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


def _same_bits(a: list[float], b: list[float]) -> bool:
    return [struct.pack(">f", x) for x in a] == [struct.pack(">f", x) for x in b]


def token_selection_invariance(
    generate: list[dict[str, Any]], contract: list[dict[str, Any]]
) -> dict[str, Any]:
    """Do two executions that must agree choose the same tokens (and logprob bits)?

    * ``across_tp``: the same generate config (mode, lora, temperature, prompts, seeds) at TP2/TP4
      vs TP1, and the same contract-shape batch at TP2/TP4 vs TP1;
    * ``across_batch``: request ``i`` of ``decode_batch_64`` vs request ``i`` of
      ``decode_batch_256`` (same prompt, same seed, different batch composition), same mode/TP.
    """

    def seq_compare(ta, la, tb, lb) -> dict[str, Any]:
        n = len(ta)
        same_tokens = sum(1 for x, y in zip(ta, tb) if x == y)
        same_bits = sum(
            1 for x, y, p, q in zip(ta, tb, la, lb) if x == y and _same_bits(p, q)
        )
        first = []
        for i, (x, y) in enumerate(zip(ta, tb)):
            if x != y:
                k = next((j for j, (u, v) in enumerate(zip(x, y)) if u != v), min(len(x), len(y)))
                first.append({"sequence": i, "first_divergent_token": k})
        return {
            "sequences": n,
            "identical_token_sequences": same_tokens,
            "identical_tokens_and_logprob_bits": same_bits,
            "divergences": first,
        }

    across_tp = []
    base = {}
    for g in generate:
        key = (g["mode"], g["lora"] is not None, g["temperature"], g["max_tokens"], g["prompts"])
        if g["tp"] == 1 and not g["engine_lora_enabled"]:
            base[key] = g
    for g in generate:
        key = (g["mode"], g["lora"] is not None, g["temperature"], g["max_tokens"], g["prompts"])
        ref = base.get(key)
        if g["tp"] == 1 or ref is None or g["engine_lora_enabled"]:
            continue
        across_tp.append(
            {
                "what": f"generate {g['mode']}{' lora' if g['lora'] else ''} "
                f"T={g['temperature']} tp{g['tp']} vs tp1",
                **seq_compare(g["token_ids"], g["logprobs"], ref["token_ids"], ref["logprobs"]),
            }
        )
    cbase = {}
    for c in contract:
        cbase.setdefault((c["mode"], c["label"]), {})[c["tp"]] = c
    for (mode, label), by_tp in sorted(cbase.items()):
        ref = by_tp.get(1)
        for tp, c in sorted(by_tp.items()):
            if tp == 1 or ref is None:
                continue
            across_tp.append(
                {
                    "what": f"{label} {mode} tp{tp} vs tp1",
                    **seq_compare(c["token_ids"], c["logprobs"], ref["token_ids"], ref["logprobs"]),
                }
            )
    across_batch = []
    for c64 in contract:
        if not c64["label"].startswith("decode_batch_64"):
            continue
        suffix = c64["label"][len("decode_batch_64") :]
        for c256 in contract:
            if (
                c256["label"] == f"decode_batch_256{suffix}"
                and c256["mode"] == c64["mode"]
                and c256["tp"] == c64["tp"]
            ):
                n = c64["requests"]
                if c256["seeds"][:n] != c64["seeds"] or c256["prompt_lens"][:n] != c64[
                    "prompt_lens"
                ]:
                    continue
                across_batch.append(
                    {
                        "what": f"{c64['mode']} tp{c64['tp']}{suffix}: request i of the 64-wide "
                        "batch vs request i of the 256-wide batch (same prompt and seed)",
                        **seq_compare(
                            c64["token_ids"],
                            c64["logprobs"],
                            c256["token_ids"][:n],
                            c256["logprobs"][:n],
                        ),
                    }
                )
    across_kv = []
    for tight in contract:
        if "_tightkv" not in tight["label"]:
            continue
        plain_label = tight["label"].replace("_tightkv", "")
        for plain in contract:
            if (
                plain["label"] != plain_label
                or plain["mode"] != tight["mode"]
                or plain["tp"] != tight["tp"]
            ):
                continue
            n = min(plain["requests"], tight["requests"])
            if plain["seeds"][:n] != tight["seeds"][:n] or plain["temperatures"][:n] != tight[
                "temperatures"
            ][:n]:
                continue
            across_kv.append(
                {
                    "what": f"{tight['mode']} tp{tight['tp']} {plain_label}: first {n} requests "
                    "in a tiny KV pool (page reuse, preemption/recompute: "
                    f"{tight.get('scheduler_events')}) vs the normal pool (same prompt and seed)",
                    **seq_compare(
                        tight["token_ids"][:n],
                        tight["logprobs"][:n],
                        plain["token_ids"][:n],
                        plain["logprobs"][:n],
                    ),
                }
            )
    return {
        "across_tp": across_tp or "NOT RUN",
        "across_batch": across_batch or "NOT RUN",
        "across_kv_pressure": across_kv or "NOT RUN",
    }


def _bench_entry(run: dict, bench: dict, *, lora: bool) -> dict[str, Any]:
    b = {k: v for k, v in bench.items()}
    flags = _engine_flags(run)
    b.update(
        {
            "run": run["_file"],
            "commit": run.get("commit"),
            "mode": run["mode"],
            "tp": run["tp"],
            "cuda_graphs": not flags["enforce_eager"],
            "lora_loaded": lora,
            "label": f"{run['mode']} tp{run['tp']}"
            + (" cuda-graphs" if not flags["enforce_eager"] else " eager")
            + (" +LoRA r16" if lora else ""),
        }
    )
    rates = b.get("tokens_per_s_repeats") or []
    if rates:
        b["tokens_per_s_median"] = statistics.median(rates)
        b["tokens_per_s_stdev"] = statistics.pstdev(rates)
    mem = b.get("memory_per_rank") or []
    b["peak_allocated_gib_max_rank"] = max((m["max_allocated_gib"] for m in mem), default=None)
    return b


def collect(
    runs_dir: Path,
    data_dir: Path,
    extra: dict[str, Any] | None,
    references: list[Path] | None = None,
) -> dict[str, Any]:
    runs = _runs_in(runs_dir)
    result: dict[str, Any] = {
        "row": "vllm_runtime_adapter",
        "schema": "muse_glimmer_vllm_runtime_results_v2",
        "commits_measured": sorted({r.get("commit") for r in runs if r.get("commit")}),
        "dirty_runs": [r["_file"] for r in runs if _code_dirty(r)],
        "dirty_runs_data_only": [
            {"run": r["_file"], "modified": r.get("git_status_porcelain")}
            for r in runs
            if r.get("git_dirty") and not _code_dirty(r)
        ],
        "env": next((r["env"] for r in runs if r.get("env")), {}),
        "commands": [r["command"] for r in runs],
        "runs": [
            {
                "file": r["_file"],
                "mode": r["mode"],
                "tp": r["tp"],
                "status": r.get("status"),
                "commit": r.get("commit"),
                "git_dirty": r.get("git_dirty"),
                "git_dirty_code": _code_dirty(r),
                "git_status_porcelain": r.get("git_status_porcelain"),
                "engine_load_s": r.get("engine_load_s"),
                "engine": _engine_flags(r),
                "command": r["command"],
                "error": r.get("error"),
            }
            for r in runs
        ],
    }
    generate = [g for r in runs for g in _generate_records(r)]
    contract = [c for r in runs for c in _contract_records(r)]
    result["generate"] = generate or "NOT RUN"
    result["contract_shapes"] = contract or "NOT RUN"
    result["token_selection_invariance"] = token_selection_invariance(generate, contract)
    result["lora_zero"] = [
        {"run": r["_file"], "mode": r["mode"], "tp": r["tp"], **r["lora_zero"]}
        for r in runs
        if r.get("lora_zero")
    ] or "NOT RUN"
    result["lora_random"] = [
        {
            "run": r["_file"],
            "mode": r["mode"],
            "tp": r["tp"],
            **{k: v for k, v in r["lora_random"].items() if k != "generate"},
        }
        for r in runs
        if r.get("lora_random")
    ] or "NOT RUN"

    xrt: dict[str, Any] = {}
    docs: dict[tuple[str, int, bool], dict] = {}
    for key, name in XRT_FILES.items():
        path = data_dir / name
        if path.exists():
            docs[key] = _load(path)
            xrt[name] = _xrt_summary(docs[key])
        else:
            xrt[name] = "NOT RUN"
    comparisons: dict[str, Any] = {}
    for name, (ka, kb) in XRT_PAIRS.items():
        if ka in docs and kb in docs:
            comparisons[name] = compare_xrt(docs[ka], docs[kb])
        else:
            comparisons[name] = "NOT RUN"
    for ref_path in references or []:
        ref = _load(ref_path)
        label = xrt_label(ref)
        slug = label.replace(" ", "_")
        xrt[slug] = _xrt_summary(ref, embed=True)
        xrt[slug]["file"] = str(ref_path)
        for key, doc in docs.items():
            if key[2]:
                continue  # the reference chain has no LoRA
            comparisons[f"{key[0]}_tp{key[1]}_vs_{slug}"] = compare_xrt(doc, ref)
    xrt["comparisons"] = comparisons
    result["xrt"] = xrt

    throughput = []
    for r in runs:
        if r.get("bench"):
            throughput.append(_bench_entry(r, r["bench"], lora=False))
        if r.get("bench_lora"):
            throughput.append(_bench_entry(r, r["bench_lora"], lora=True))
    result["throughput"] = throughput or "NOT RUN"
    profiles = []
    for r in runs:
        if not r.get("profile"):
            continue
        entry: dict[str, Any] = {"run": r["_file"], "mode": r["mode"], "tp": r["tp"]}
        for kind, prof in r["profile"].items():
            entry[kind] = prof if prof else "NOT CAPTURED"
        profiles.append(entry)
    result["profile"] = profiles or "NOT RUN"
    provenance = []
    for r in runs:
        meta = r.get("meta") or {}
        provenance.append(
            {
                "run": r["_file"],
                "engine": "StrictRolloutEngine",
                "lora_sha256": meta.get("lora_sha256"),
                "lora_sha256_source": meta.get("lora_sha256_source"),
                "rl_engine_file": meta.get("rl_engine_file"),
                "worker_rl_engine_files": meta.get("worker_rl_engine_files"),
                "native_lora_reports": meta.get("native_lora_reports"),
            }
        )
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
                    "sampling": w.get("sampling"),
                    "lora_sha256": w.get("lora_sha256"),
                    "lora_sha256_source": w.get("lora_sha256_source"),
                    "rl_engine_file": w.get("rl_engine_file"),
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
    parser.add_argument("--data", default="reports/muse_glimmer_poc/data")
    parser.add_argument("--extra", default=None, help="JSON merged at top level")
    parser.add_argument(
        "--reference", action="append", default=[], help="reference-chain xrt file(s)"
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    extra = _load(Path(args.extra)) if args.extra else None
    result = collect(
        Path(args.runs), Path(args.data), extra, [Path(p) for p in args.reference]
    )
    # Compact: the raw per-token series dominate the size; indent would put each float on a line.
    Path(args.out).write_text(json.dumps(result, separators=(",", ":")), encoding="utf-8")
    summary = {
        k: (v.get("mismatched"), v.get("tokens")) if isinstance(v, dict) else v
        for k, v in result["xrt"].get("comparisons", {}).items()
    }
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
