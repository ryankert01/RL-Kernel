#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Assemble the full_model_chain follow-up results into one machine-readable JSON object.

Reads the run outputs of ``run_chain.py`` (chain gate, HF anchor, ``--score-file`` scoring) and
``run_profile.py`` (strict / HF op-class profiles) from ``--runs`` and writes
``reports/muse_glimmer_poc/data/reference_profile.json`` (plot-ready series: per-token logp and
|Δlogp|, per-layer residual drift, per-config stacked op-class bars) and copies the two
``muse_glimmer_xrt_logprobs_v1`` files next to it. CPU only.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import sys
from importlib import metadata
from typing import Any

REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rl_engine.models.muse_glimmer.reference.profiling import CLASSES, GROUPS  # noqa: E402
from rl_engine.models.muse_glimmer.reference.xrt_scoring import (  # noqa: E402
    compare_xrt,
    source_commit,
)

DATA = REPO_ROOT / "reports" / "muse_glimmer_poc" / "data"


def _load(path: pathlib.Path) -> dict[str, Any] | None:
    return json.loads(path.read_text()) if path.is_file() else None


def chain_record(payload: dict[str, Any] | None) -> dict[str, Any]:
    if payload is None:
        return {"status": "NOT RUN"}
    chain = payload["chain"]
    eq = {e["rhs"]: e for e in chain["equalities"]}
    paths = []
    for r in chain["results"]:
        e = eq.get(r["spec"]["name"])
        paths.append({
            "name": r["spec"]["name"],
            "mode": r["spec"]["mode"],
            "chunk_sizes": r["spec"]["chunk_sizes"],
            "n_chunks": None if r["spec"]["chunk_sizes"] is None else len(r["spec"]["chunk_sizes"]),
            "seconds": r["seconds"],
            "peak_alloc_gib": r["peak_alloc_gib"],
            "equal_tokens": r["logp"].__len__() if e is None else e["n_equal"],
            "n_tokens": len(r["logp"]),
            "byte_equal_to_train": True if e is None else e["equal"],
            "max_abs_diff_vs_train": 0.0 if e is None else e["max_abs_diff"],
        })
    train = chain["results"][0]
    rec: dict[str, Any] = {
        "status": "PASS" if chain["all_equal"] else "FAIL",
        "backend": chain["backend"],
        "prompt_len": chain["prompt_len"],
        "completion_len": chain["completion_len"],
        "all_paths_byte_equal": chain["all_equal"],
        "paths": paths,
        "drifts": chain["drifts"],
        "train_logp_series": train["logp"],
        "device": payload.get("device"),
        "torch": payload.get("torch"),
    }
    hf = payload.get("hf_compare")
    if hf is not None:
        rec["hf_anchor"] = {
            "hf": hf.get("hf"),
            "positions": hf["positions"],
            "completion_start": hf["completion_start"],
            "logits_all_positions": hf["logits_all_positions"],
            "logits_completion_positions": hf["logits_completion_positions"],
            "logp_all": hf["logp_all"],
            "logp_completion": hf["logp_completion"],
            "argmax_agreement_all": hf["argmax_agreement_all"],
            "argmax_agreement_completion": hf["argmax_agreement_completion"],
            "per_layer": hf.get("per_layer"),
            "series": hf.get("series"),
            "note": "position i scores token i+1; completion positions are "
                    "[completion_start-1, positions-1)",
        }
    return rec


def xrt_record(golden: dict[str, Any] | None, triton: dict[str, Any] | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, x in (("golden", golden), ("triton", triton)):
        if x is None:
            out[name] = {"status": "NOT RUN"}
            continue
        out[name] = {
            "status": "PASS" if x["checks"]["all_byte_equal_to_train"] else "FAIL",
            "commit": x.get("commit"),
            "seconds": x.get("seconds"),
            "peak_allocated_gib": x.get("peak_allocated_gib"),
            "n_tokens": sum(s["n_tokens"] for s in x["sequences"]),
            "check_modes": x["checks"]["modes"],
            "all_byte_equal_to_train": x["checks"]["all_byte_equal_to_train"],
            "per_sequence": x["checks"]["sequences"],
            "logprobs": {s["id"]: s["logprobs"] for s in x["sequences"]},
        }
    if golden is not None and triton is not None:
        d = compare_xrt(golden, triton)
        out["golden_vs_triton"] = {
            "n_tokens": d["n_tokens"],
            "equal_tokens": d["equal_tokens"],
            "per_sequence": [
                {k: v for k, v in r.items()} for r in d["sequences"]
            ],
            "note": "two different strict op sets (fp32 golden fallbacks vs Triton kernels): "
                    "NOT expected to be byte-equal; tolerance comparison only",
        }
    return out


def profile_record(strict: dict[str, Any] | None, hf: dict[str, Any] | None) -> dict[str, Any]:
    out: dict[str, Any] = {"classes": list(CLASSES), "groups": list(GROUPS)}
    configs: list[str] = []

    def label(r: dict[str, Any]) -> str:
        if r["phase"] == "decode":
            return f"decode B={r['batch']} ctx={r['context']}"
        return f"{r['phase']} T={r['T']}"

    for name, prof in (("strict_triton", strict), ("hf_bf16_sdpa", hf)):
        if prof is None:
            out[name] = {"status": "NOT RUN"}
            continue
        runs = []
        for r in prof["runs"]:
            lab = label(r)
            if lab not in configs:
                configs.append(lab)
            runs.append({"config": lab, **r})
        out[name] = {k: v for k, v in prof.items() if k != "runs"}
        out[name]["runs"] = runs
    out["configs"] = configs
    # plot-ready stacked bars: ms per group per config, both targets
    bars: dict[str, Any] = {"configs": configs, "groups": list(GROUPS)}
    for name in ("strict_triton", "hf_bf16_sdpa"):
        prof = out.get(name, {})
        by_cfg = {r["config"]: r for r in prof.get("runs", [])}
        bars[name] = {
            "ms": {g: [by_cfg[c]["by_group_ms"][g] if c in by_cfg else None for c in configs]
                   for g in GROUPS},
            "frac": {g: [by_cfg[c]["by_group_frac"][g] if c in by_cfg else None for c in configs]
                     for g in GROUPS},
            "clean_total_ms": [by_cfg[c]["clean_total_ms_median"] if c in by_cfg else None
                               for c in configs],
            "instrumented_total_ms": [by_cfg[c]["total_ms_median"] if c in by_cfg else None
                                      for c in configs],
            "peak_allocated_gib": [by_cfg[c]["peak_allocated_gib"] if c in by_cfg else None
                                   for c in configs],
        }
    s_runs = {r["config"]: r for r in out.get("strict_triton", {}).get("runs", [])}
    h_runs = {r["config"]: r for r in out.get("hf_bf16_sdpa", {}).get("runs", [])}
    bars["strict_over_hf_clean"] = [
        (s_runs[c]["clean_total_ms_median"] / h_runs[c]["clean_total_ms_median"])
        if c in s_runs and c in h_runs else None
        for c in configs
    ]
    out["stacked_bars"] = bars
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", required=True, help="directory with the run JSON files")
    p.add_argument("--out", default=str(DATA / "reference_profile.json"))
    p.add_argument("--status", default="complete")
    p.add_argument("--commands", default=None, help="JSON list of the exact command lines")
    args = p.parse_args(argv)
    runs = pathlib.Path(args.runs)
    golden = _load(runs / "xrt_reference_golden.json")
    triton = _load(runs / "xrt_reference_triton.json")
    for name, payload in (("golden", golden), ("triton", triton)):
        if payload is not None:
            shutil.copyfile(runs / f"xrt_reference_{name}.json",
                            DATA / f"xrt_reference_{name}.json")
    strict = _load(runs / "profile_strict.json")
    hf = _load(runs / "profile_hf.json")
    env: dict[str, Any] = {}
    for prof in (strict, hf):
        if prof is not None:
            env.update(prof.get("env", {}))
    for pkg in ("vllm", "torch", "triton", "transformers"):
        try:
            env[f"{pkg}_installed"] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            env[f"{pkg}_installed"] = None
    result = {
        "row": "full_model_chain follow-up: reference scoring of xrt_tokens.json + single-GPU "
               "profiling",
        "status": args.status,
        "commit": source_commit(),
        "env": env,
        "commands": json.loads(pathlib.Path(args.commands).read_text()) if args.commands else [],
        "experiments": {
            "chain_triton_512_hf_pre_change": chain_record(
                _load(runs / "chain_triton_512_hf_base.json")),
            "chain_triton_512_hf": chain_record(_load(runs / "chain_triton_512_hf.json")),
            "chain_triton_2100_hf": chain_record(_load(runs / "chain_triton_2100_hf.json")),
            "chain_golden_2100": chain_record(_load(runs / "chain_golden_2100.json")),
            "xrt_scoring": xrt_record(golden, triton),
            "profile": profile_record(strict, hf),
            "decode_batch_invariance": (strict or {}).get("decode_batch_invariance",
                                                          {"status": "NOT RUN"}),
        },
    }
    pathlib.Path(args.out).write_text(json.dumps(result, indent=1))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
