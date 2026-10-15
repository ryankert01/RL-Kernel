#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""CLI for the Muse-Glimmer drift ablation (RFC #501 row ``museglimmer_ablation``).

Each GPU subcommand loads its model ONCE and writes a part file after every setting (a cut-off
run keeps everything measured so far); ``merge`` builds ``reports/muse_glimmer_poc/data/
ablation.json`` from the parts.  One GPU per process:

    T=reports/muse_glimmer_poc/data/xrt_tokens.json; R=rl_engine/models/muse_glimmer/ablation
    ~/bin/gpu-lease -n 1 --pool 6-7 -- env PYTHONPATH=$PWD ~/envs/vllm/bin/python \
        $R/run_ablation.py hf --tokens $T --out $S/part_hf.json
    ~/bin/gpu-lease -n 1 --pool <other GPU> -- env PYTHONPATH=$PWD ~/envs/vllm/bin/python \
        $R/run_ablation.py hf-baseline --tokens $T --out $S/part_hf_gpu2.json
    ~/bin/gpu-lease -n 1 --pool 6-7 -- env PYTHONPATH=$PWD ~/envs/vllm/bin/python \
        $R/run_ablation.py strict --tokens $T --out $S/part_strict.json
    ~/envs/vllm/bin/python $R/run_ablation.py merge --parts $S/part_*.json \
        --out reports/muse_glimmer_poc/data/ablation.json
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import pathlib
import subprocess
import sys
import time
from typing import Any

REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from rl_engine.models.muse_glimmer.ablation import ablation_hf as A  # noqa: E402

DEFAULT_TOKENS = "reports/muse_glimmer_poc/data/xrt_tokens.json"
DEFAULT_CKPT = "~/models/Muse-Glimmer-30B"


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                                       text=True).strip()
    except Exception:  # pragma: no cover - git missing
        return "unknown"


def _git_dirty() -> bool:
    try:
        out = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"],
                                      cwd=REPO_ROOT, text=True)
        return bool(out.strip())
    except Exception:  # pragma: no cover
        return True


def _write(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1))
    os.replace(tmp, path)


def _part_header(kind: str, args: argparse.Namespace, device: torch.device,
                 tokens_meta: dict[str, Any]) -> dict[str, Any]:
    return {
        "part": kind,
        "commit": _git_sha(),
        "worktree_dirty": _git_dirty(),
        "argv": [sys.executable, *sys.argv],
        "env": A.env_info(device),
        "tokens": tokens_meta,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "baseline": None,
        "settings": [],
        "records": [],
        "errors": [],
    }


def _setting_summary(engine: str, name: str, factor: str, factor_id: int, desc: str,
                     records: list[dict[str, Any]], stats: dict[str, Any]) -> dict[str, Any]:
    out = {"engine": engine, "setting": name, "factor": factor, "factor_id": factor_id,
           "description": desc}
    out.update(A.aggregate(records))
    out.update(stats)
    return out


# --------------------------------------------------------------------------------------------
# HF
# --------------------------------------------------------------------------------------------
def cmd_hf(args: argparse.Namespace, baseline_only: bool = False) -> int:
    from rl_engine.validation.operators.accelerator import disable_tf32

    device = torch.device(args.device)
    disable_tf32(device.type)  # contract policy for the whole process; factors flip it locally
    seqs, tmeta = A.load_xrt_tokens(args.tokens)
    out = pathlib.Path(args.out)
    part = _part_header("hf_baseline_only" if baseline_only else "hf", args, device, tmeta)
    t0 = time.perf_counter()
    model, lm_head, cfg = A.load_text_model(args.checkpoint, device, attn="eager")
    torch.cuda.synchronize(device)
    part["load_s"] = time.perf_counter() - t0
    part["weights_gib"] = (sum(p.numel() for p in model.parameters()) + lm_head.numel()) * 2 / 2**30
    part["rope_inv_freq_hex"] = A.fp32_hex(model.rotary_emb.inv_freq)
    _log(f"HF text model loaded in {part['load_s']:.1f}s ({part['weights_gib']:.1f} GiB) on "
         f"{part['env'].get('gpu')} lease={part['env'].get('gpu_lease')}")
    part["baseline"] = {
        "name": "hf_baseline",
        "description": "HF 5.17 eager bf16, TF32 off, each sequence alone (L tokens), one-shot,"
                       " 5.17 matmul RoPE, bf16 logits+softcap, fp32 log_softmax",
    }
    base, stats = A.score_all(model, lm_head, cfg, seqs, driver="alone")
    part["baseline"].update(stats)
    part["baseline"]["per_sequence"] = A.baseline_payload(seqs, base)
    _log(f"baseline: {stats['wall_ms']:.0f} ms, peak {stats['peak_mem_gib']:.1f} GiB")
    _write(out, part)
    settings = list(A.HF_SETTINGS)
    if baseline_only:
        settings = [s for s in settings if s.name == "repeat_immediate"]
    elif args.settings:
        settings = [s for s in settings if s.name in set(args.settings)]
    part["skipped"] = [] if baseline_only else list(A.SKIPPED_SETTINGS)
    for st in settings:
        _log(f"setting {st.name} ({st.factor}) ...")
        try:
            var, stats = A.run_setting(model, lm_head, cfg, seqs, st)
        except torch.cuda.OutOfMemoryError as exc:
            part["errors"].append({"setting": st.name, "error": f"OOM: {exc}"[:500]})
            _log(f"  OOM: {str(exc)[:200]}")
            gc.collect()
            torch.cuda.empty_cache()
            _write(out, part)
            continue
        recs = A.make_records("hf", st.name, st.factor, st.factor_id, "hf_baseline", seqs, base,
                              var)
        part["records"].extend(recs)
        summ = _setting_summary("hf", st.name, st.factor, st.factor_id, st.description, recs,
                                stats)
        part["settings"].append(summ)
        _log(f"  {summ['n_bits_differ']}/{summ['n_tokens']} tokens differ, max|d| "
             f"{summ['max_abs_dlogp']:.3e}, mean|d| {summ['mean_abs_dlogp']:.3e}, k3 "
             f"{summ['mean_k3']:.3e}, argmax flips {summ['argmax_flips']}, "
             f"{stats['wall_ms']:.0f} ms, peak {stats['peak_mem_gib']:.1f} GiB")
        _write(out, part)
    if not baseline_only:
        _log("repeat_after_all_factors (restore + repeat null control) ...")
        var, stats = A.score_all(model, lm_head, cfg, seqs, driver="alone")
        recs = A.make_records("hf", "repeat_after_all_factors", "repeat_null", 9, "hf_baseline",
                              seqs, base, var)
        part["records"].extend(recs)
        part["settings"].append(_setting_summary(
            "hf", "repeat_after_all_factors", "repeat_null", 9,
            "baseline again after every factor ran and was restored", recs, stats))
        _log(f"  {part['settings'][-1]['n_bits_differ']} tokens differ")
    part["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    part["status"] = "complete" if not part["errors"] else "partial"
    _write(out, part)
    _log(f"wrote {out}")
    return 0


# --------------------------------------------------------------------------------------------
# Strict reference
# --------------------------------------------------------------------------------------------
STRICT_SETTINGS = (
    # name, factor, factor_id, description, kind, arg
    ("repeat_immediate", "repeat_null", 9, "train-mode baseline again", "alone", ("train", None)),
    ("packed_batch9", "batch_composition", 1,
     "all 9 sequences in one varlen training pass (cu_seqlens)", "packed", "packed"),
    ("packed_batch9_reversed", "batch_composition", 1,
     "same, reversed batch order (every sequence at a new row offset)", "packed",
     "packed_reversed"),
    ("right_pad_batch9", "batch_composition", 1,
     "every segment right-padded to 2439 tokens (21951 rows)", "packed", "right_pad"),
    ("left_pad_batch9", "batch_composition", 1,
     "a pad segment in front of every shorter sequence (left padding, positions from 0)",
     "packed", "left_pad"),
    ("paged_prefill_batch9", "batch_composition", 1,
     "rollout form: one paged prefill of L-1 tokens for all 9 sequences", "paged_prefill", None),
    ("prefill_alone", "prefill_vs_decode", 2, "reference mode=prefill (paged, L-1 tokens)",
     "alone", ("prefill", None)),
    ("decode_alone", "prefill_vs_decode", 2,
     "reference mode=decode: prompt prefill + 1-token decode steps", "alone", ("decode", None)),
    ("batched_decode9", "prefill_vs_decode", 2,
     "continuous batching: 9 prompts prefilled together, decode steps pack all active sequences",
     "batched_decode", None),
    ("chunked_prefill_64", "chunked_prefill", 3, "reference mode=chunked, chunks of 64", "alone",
     ("chunked", 64)),
    ("chunked_prefill_256", "chunked_prefill", 3, "reference mode=chunked, chunks of 256",
     "alone", ("chunked", 256)),
    ("chunked_prefill_1024", "chunked_prefill", 3, "reference mode=chunked, chunks of 1024",
     "alone", ("chunked", 1024)),
)


def cmd_strict(args: argparse.Namespace) -> int:
    from rl_engine.models.muse_glimmer.ablation import ablation_strict as S
    from rl_engine.models.muse_glimmer.reference.strict_model import MuseGlimmerStrictModel
    from rl_engine.models.muse_glimmer.reference.weights import MuseGlimmerWeights
    from rl_engine.validation.operators.accelerator import disable_tf32

    device = torch.device(args.device)
    disable_tf32(device.type)
    seqs, tmeta = A.load_xrt_tokens(args.tokens)
    out = pathlib.Path(args.out)
    part = _part_header("strict", args, device, tmeta)
    t0 = time.perf_counter()
    weights = MuseGlimmerWeights.load_checkpoint(args.checkpoint, device=device, layout="nk",
                                                 progress=_log)
    model = MuseGlimmerStrictModel(weights, args.backend)
    torch.cuda.synchronize(device)
    part["load_s"] = time.perf_counter() - t0
    part["weights_gib"] = weights.memory_bytes() / 2**30
    part["ops"] = model.ops.describe()["provenance"]
    _log(f"strict model ({args.backend}) loaded in {part['load_s']:.1f}s on "
         f"{part['env'].get('gpu')} lease={part['env'].get('gpu_lease')}")
    torch.cuda.reset_peak_memory_stats(device)
    t1 = time.perf_counter()
    base = S.score_alone(model, seqs, "train")
    part["baseline"] = {
        "name": "strict_train_alone",
        "description": f"strict reference ({args.backend} backend) training forward, each "
                       "sequence alone (MuseGlimmerStrictModel.logprobs mode=train)",
        "wall_ms": (time.perf_counter() - t1) * 1e3,
        "peak_mem_gib": torch.cuda.max_memory_allocated(device) / 2**30,
        "per_sequence": A.baseline_payload(seqs, base),
    }
    _log(f"strict baseline {part['baseline']['wall_ms']:.0f} ms")
    _write(out, part)
    wanted = set(args.settings) if args.settings else None
    for name, factor, fid, desc, kind, arg in STRICT_SETTINGS:
        if wanted is not None and name not in wanted:
            continue
        _log(f"strict setting {name} ...")
        torch.cuda.reset_peak_memory_stats(device)
        t1 = time.perf_counter()
        extra: dict[str, Any] = {}
        try:
            if kind == "alone":
                var = S.score_alone(model, seqs, arg[0], arg[1])
            elif kind == "packed":
                var = S.score_packed_batch(model, seqs, arg)
            elif kind == "paged_prefill":
                var = S.score_paged_batch_prefill(model, seqs)
            elif kind == "batched_decode":
                var, steps = S.score_batched_decode(model, seqs)
                extra["decode_steps"] = steps
            else:  # pragma: no cover
                raise ValueError(kind)
        except torch.cuda.OutOfMemoryError as exc:
            part["errors"].append({"setting": name, "error": f"OOM: {exc}"[:500]})
            gc.collect()
            torch.cuda.empty_cache()
            _write(out, part)
            continue
        stats = {"wall_ms": (time.perf_counter() - t1) * 1e3,
                 "peak_mem_gib": torch.cuda.max_memory_allocated(device) / 2**30, **extra}
        recs = A.make_records("strict", name, factor, fid, "strict_train_alone", seqs, base, var)
        part["records"].extend(recs)
        summ = _setting_summary("strict", name, factor, fid, desc, recs, stats)
        part["settings"].append(summ)
        _log(f"  {summ['n_bits_differ']}/{summ['n_tokens']} tokens differ, max|d| "
             f"{summ['max_abs_dlogp']:.3e}, {stats['wall_ms']:.0f} ms, peak "
             f"{stats['peak_mem_gib']:.1f} GiB")
        _write(out, part)
    part["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    part["status"] = "complete" if not part["errors"] else "partial"
    _write(out, part)
    _log(f"wrote {out}")
    return 0


# --------------------------------------------------------------------------------------------
# Merge
# --------------------------------------------------------------------------------------------
def _scores_from_baseline(b: dict[str, Any]) -> list[A.SeqScore]:
    out = []
    for ps in b["per_sequence"]:
        am = ps.get("argmax")
        out.append(A.SeqScore(torch.tensor(ps["logp"], dtype=torch.float32),
                              None if am is None else torch.tensor(am), ps["wall_ms"]))
    return out


def cmd_merge(args: argparse.Namespace) -> int:
    parts = [json.loads(pathlib.Path(p).read_text()) for p in args.parts]
    by_kind: dict[str, list[dict[str, Any]]] = {}
    for p, path in zip(parts, args.parts):
        p["_path"] = str(path)
        by_kind.setdefault(p["part"], []).append(p)
    if "hf" not in by_kind:
        raise SystemExit("merge needs the main 'hf' part")
    hf = by_kind["hf"][0]
    seqs, _ = A.load_xrt_tokens(args.tokens)
    records = list(hf["records"])
    settings = list(hf["settings"])
    hf_base = _scores_from_baseline(hf["baseline"])
    # Factor 8: the same baseline on another GPU (separate process, separate lease).
    for other in by_kind.get("hf_baseline_only", []):
        gpu_a, gpu_b = hf["env"].get("gpu_lease"), other["env"].get("gpu_lease")
        name = (f"second_gpu_{gpu_b}_vs_{gpu_a}" if gpu_a != gpu_b
                else f"new_process_same_gpu_{gpu_b}")
        factor, fid = ("second_gpu_null", 8) if gpu_a != gpu_b else ("repeat_null", 9)
        var = _scores_from_baseline(other["baseline"])
        recs = A.make_records("hf", name, factor, fid, "hf_baseline", seqs, hf_base, var,
                              extra={"gpu_lease_variant": gpu_b, "gpu_lease_baseline": gpu_a})
        records.extend(recs)
        settings.append(_setting_summary(
            "hf", name, factor, fid,
            f"baseline in a separate process on GPU {gpu_b} (baseline ran on GPU {gpu_a})",
            recs, {"wall_ms": other["baseline"]["wall_ms"],
                   "peak_mem_gib": other["baseline"]["peak_mem_gib"]}))
        for r in other["records"]:  # its own immediate repeat
            r = dict(r, setting=f"{r['setting']}_gpu_{gpu_b}", baseline=f"hf_baseline_gpu_{gpu_b}")
            records.append(r)
        settings.extend(dict(s, setting=f"{s['setting']}_gpu_{gpu_b}")
                        for s in other["settings"])
    strict_parts = by_kind.get("strict", [])
    for st in strict_parts:
        records.extend(st["records"])
        settings.extend(st["settings"])
        # Context: how far the strict reference sits from the HF baseline (not a factor).
        sb = _scores_from_baseline(st["baseline"])
        recs = A.make_records("cross", "strict_vs_hf_baseline", "engine_context", 0,
                              "hf_baseline", seqs, hf_base, sb)
        records.extend(recs)
        settings.append(_setting_summary(
            "cross", "strict_vs_hf_baseline", "engine_context", 0,
            "strict train-mode logp minus HF baseline logp (different numerics by design)",
            recs, {}))
    commits = sorted({p["commit"] for p in parts})
    status = "complete"
    if any(p.get("status") != "complete" for p in parts) or not strict_parts \
            or not by_kind.get("hf_baseline_only"):
        status = "partial"
    payload = {
        "row": "museglimmer_ablation",
        "schema": "muse_glimmer_ablation_v1",
        "status": status,
        "commit": commits[0] if len(commits) == 1 else commits,
        "env": {p["part"] + ("" if i == 0 else f"_{i}"): p["env"]
                for i, p in enumerate(parts)},
        "commands": args.command or [" ".join(p["argv"]) for p in parts],
        "tokens": hf["tokens"],
        "metric_definitions": {
            "d": "logp_variant - logp_baseline of the gold token (fp32 values, fp64 difference)",
            "bits_differ": "fp32 bit patterns of the two logp values differ",
            "k3": "exp(d) - d - 1 (fp64); token-level k3 KL estimator, tokens are gold answers",
            "argmax_flip": "argmax over the vocab of variant logits != baseline (HF only)",
            "dlogp/k3 arrays": "rounded to 6/4 significant digits; logp_hex is exact",
        },
        "baselines": {
            "hf": hf["baseline"],
            **({"strict": strict_parts[0]["baseline"]} if strict_parts else {}),
            **{f"hf_gpu_{o['env'].get('gpu_lease')}": o["baseline"]
               for o in by_kind.get("hf_baseline_only", [])},
        },
        "parts": [{"part": p["part"], "path": p["_path"], "commit": p["commit"],
                   "worktree_dirty": p.get("worktree_dirty"), "load_s": p.get("load_s"),
                   "weights_gib": p.get("weights_gib"), "started": p.get("started"),
                   "finished": p.get("finished"), "status": p.get("status"),
                   "errors": p.get("errors"), "gpu_lease": p["env"].get("gpu_lease")}
                  for p in parts],
        "rope_inv_freq_hex": hf.get("rope_inv_freq_hex"),
        "strict_ops": strict_parts[0].get("ops") if strict_parts else None,
        "skipped": hf.get("skipped", []),
        "settings": settings,
        "records": records,
    }
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1))
    _log(f"wrote {out}: {len(records)} records, {len(settings)} settings, status {status}")
    print(markdown_table(settings))
    return 0


def markdown_table(settings: list[dict[str, Any]]) -> str:
    head = ("| engine | factor | setting | tokens differ | max abs dlogp | mean abs dlogp | "
            "mean k3 | argmax flips | wall ms |\n|---|---|---|---|---|---|---|---|---|")
    rows = [head]
    for s in settings:
        flips = "-" if s.get("argmax_flips") is None else str(s["argmax_flips"])
        wall = s.get("wall_ms")
        rows.append(
            f"| {s['engine']} | {s['factor_id']} {s['factor']} | {s['setting']} | "
            f"{s['n_bits_differ']}/{s['n_tokens']} | {s['max_abs_dlogp']:.3g} | "
            f"{s['mean_abs_dlogp']:.3g} | {s['mean_k3']:.3g} | {flips} | "
            f"{'-' if wall is None else f'{wall:.0f}'} |")
    return "\n".join(rows)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("hf", "hf-baseline", "strict"):
        q = sub.add_parser(name)
        q.add_argument("--checkpoint", default=DEFAULT_CKPT)
        q.add_argument("--tokens", default=str(REPO_ROOT / DEFAULT_TOKENS))
        q.add_argument("--out", required=True)
        q.add_argument("--device", default="cuda:0")
        q.add_argument("--settings", nargs="*", default=None)
        if name == "strict":
            q.add_argument("--backend", choices=("golden", "triton"), default="triton")
    m = sub.add_parser("merge")
    m.add_argument("--parts", nargs="+", required=True)
    m.add_argument("--tokens", default=str(REPO_ROOT / DEFAULT_TOKENS))
    m.add_argument("--out", default=str(REPO_ROOT / "reports/muse_glimmer_poc/data/ablation.json"))
    m.add_argument("--command", action="append", default=None,
                   help="exact command line that produced a part (repeatable)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.cmd in ("hf", "hf-baseline", "strict"):
        args.checkpoint = str(pathlib.Path(args.checkpoint).expanduser())
    if args.cmd == "hf":
        return cmd_hf(args)
    if args.cmd == "hf-baseline":
        return cmd_hf(args, baseline_only=True)
    if args.cmd == "strict":
        return cmd_strict(args)
    return cmd_merge(args)


if __name__ == "__main__":
    raise SystemExit(main())
