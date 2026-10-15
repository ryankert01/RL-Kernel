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
# HF per-layer drift + TF32 probe
# --------------------------------------------------------------------------------------------
LAYER_SETTINGS = (
    # name, factor, factor_id, driver, chunk, contexts
    ("repeat_immediate", "repeat_null", 9, "alone", None, ()),
    ("padded_batch9_right", "batch_composition", 1, "padded_right", None, ()),
    ("padded_batch9_left", "batch_composition", 1, "padded_left", None, ()),
    ("kv_cache_decode", "prefill_vs_decode", 2, "decode", None, ()),
    ("prefill_L_minus_1", "chunked_prefill", 3, "prefill_minus_last", None, ()),
    ("chunked_prefill_256", "chunked_prefill", 3, "chunked", 256, ()),
    ("sdpa", "attention_impl", 4, "alone", None, ("sdpa",)),
    ("tf32_on", "tf32", 5, "alone", None, ("tf32_on",)),
    ("bf16_reduced_reduction_off", "gemm_reduction", 10, "alone", None, ("bf16_rr_off",)),
)
LAYER_SEQUENCES = ("gsm8k_test_0", "gsm8k_long_window")


def cmd_hf_layers(args: argparse.Namespace) -> int:
    from rl_engine.validation.operators.accelerator import disable_tf32

    device = torch.device(args.device)
    disable_tf32(device.type)
    seqs, tmeta = A.load_xrt_tokens(args.tokens)
    out = pathlib.Path(args.out)
    part = _part_header("hf_layers", args, device, tmeta)
    part["tf32_probe"] = A.tf32_probe(device)
    _log(f"tf32 probe: {part['tf32_probe']}")
    t0 = time.perf_counter()
    model, _lm_head, _cfg = A.load_text_model(args.checkpoint, device, attn="eager")
    part["load_s"] = time.perf_counter() - t0
    part["layer_records"] = []
    _write(out, part)
    by_id = {s.id: i for i, s in enumerate(seqs)}
    for sid in LAYER_SEQUENCES:
        target = by_id[sid]
        base = A.layer_trace(model, seqs, target, "alone")
        for name, factor, fid, driver, chunk, ctx in LAYER_SETTINGS:
            if args.settings and name not in set(args.settings):
                continue
            torch.cuda.synchronize(device)
            t1 = time.perf_counter()
            with A.enter_contexts(model, ctx):
                var = A.layer_trace(model, seqs, target, driver, chunk)
            torch.cuda.synchronize(device)
            layers = A.layer_diff(base, var, seqs[target].completion_start)
            first = next((r["point"] for r in layers if r["frac_elems_differ"] > 0), None)
            part["layer_records"].append({
                "engine": "hf", "setting": name, "factor": factor, "factor_id": fid,
                "sequence": sid, "first_point_with_diff": first,
                "wall_ms": (time.perf_counter() - t1) * 1e3, "layers": layers,
            })
            fin = layers[-1]
            _log(f"{sid} {name}: first diff at {first}; final_norm frac "
                 f"{fin['frac_elems_differ']:.3f} mean|d| {fin['mean_abs']:.3e}")
            del var
            _write(out, part)
        del base
        gc.collect()
    part["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    part["status"] = "complete"
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
    hf_parts = sorted(by_kind["hf"], key=lambda p: -len(p["settings"]))
    hf = hf_parts[0]
    seqs, _ = A.load_xrt_tokens(args.tokens)
    records = list(hf["records"])
    settings = list(hf["settings"])
    hf_base = _scores_from_baseline(hf["baseline"])
    # Later HF runs (extra settings) count only if their own baseline is bitwise the main one.
    base_hex = [p["logp_hex"] for p in hf["baseline"]["per_sequence"]]
    extra_hf = []
    for other in hf_parts[1:]:
        same = [p["logp_hex"] for p in other["baseline"]["per_sequence"]] == base_hex
        extra_hf.append({"path": other["_path"], "gpu_lease": other["env"].get("gpu_lease"),
                         "baseline_bitwise_equal_to_main": same})
        if not same:
            continue
        have = {s["setting"] for s in settings}
        records.extend(r for r in other["records"] if r["setting"] not in have)
        settings.extend(s for s in other["settings"] if s["setting"] not in have)
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
    layer_parts = by_kind.get("hf_layers", [])
    # The main strict part is the one with the most settings; others are null controls.
    strict_parts = sorted(by_kind.get("strict", []), key=lambda p: -len(p["settings"]))
    if strict_parts:
        st = strict_parts[0]
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
        for other in strict_parts[1:]:
            gpu_a, gpu_b = st["env"].get("gpu_lease"), other["env"].get("gpu_lease")
            same = gpu_a == gpu_b
            name = f"new_process_same_gpu_{gpu_b}" if same else f"second_gpu_{gpu_b}_vs_{gpu_a}"
            factor, fid = ("repeat_null", 9) if same else ("second_gpu_null", 8)
            recs = A.make_records("strict", name, factor, fid, "strict_train_alone", seqs, sb,
                                  _scores_from_baseline(other["baseline"]),
                                  extra={"gpu_lease_variant": gpu_b, "gpu_lease_baseline": gpu_a})
            records.extend(recs)
            settings.append(_setting_summary(
                "strict", name, factor, fid,
                f"strict baseline in a separate process on GPU {gpu_b} (main ran on GPU {gpu_a})",
                recs, {"wall_ms": other["baseline"]["wall_ms"],
                       "peak_mem_gib": other["baseline"]["peak_mem_gib"]}))
    commits = sorted({p["commit"] for p in parts})
    status = "complete"
    if any(p.get("status") != "complete" for p in parts) or not strict_parts \
            or not by_kind.get("hf_baseline_only"):
        status = "partial"
    payload = {
        "row": "museglimmer_ablation",
        "schema": "muse_glimmer_ablation_v1",
        "status": status,
        "commit": hf["commit"],
        "commits_all_parts": commits,
        "env": {f"{p['part']}@gpu{p['env'].get('gpu_lease')}": p["env"] for p in parts},
        "commands": _commands(args, parts),
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
        "extra_hf_parts": extra_hf,
        "tf32_probe": layer_parts[0].get("tf32_probe") if layer_parts else None,
        "layer_records": [r for p in layer_parts for r in p.get("layer_records", [])],
        "settings": settings,
        "analysis": analysis(records, hf["baseline"]),
        "records": records,
    }
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1))
    _log(f"wrote {out}: {len(records)} records, {len(settings)} settings, status {status}")
    print(markdown_table(settings))
    print()
    print(markdown_matrix(records))
    print()
    print(markdown_matrix(records, "max_abs_dlogp"))
    return 0


PROB_BUCKETS = ((0.9, "p>0.9"), (0.5, "0.5<p<=0.9"), (0.1, "0.1<p<=0.5"), (-1.0, "p<=0.1"))


def analysis(records: list[dict[str, Any]], baseline: dict[str, Any]) -> dict[str, Any]:
    """Derived views for the report: which HF settings produce identical bits on the same
    token (pairwise), and |dlogp| by the baseline probability of the gold token."""
    import math

    hf_recs = [r for r in records if r["engine"] == "hf"]
    by: dict[str, dict[str, dict[str, Any]]] = {}
    for r in hf_recs:
        by.setdefault(r["setting"], {})[r["sequence"]] = r
    names = list(by)
    seq_ids = list(by[names[0]]) if names else []
    common = [s for s in seq_ids if all(s in by[n] for n in names)]
    matrix = []
    for a in names:
        row = []
        for b in names:
            row.append(sum(x == y for s in common
                           for x, y in zip(by[a][s]["logp_hex"], by[b][s]["logp_hex"])))
        matrix.append(row)
    base = {p["sequence"]: p for p in baseline["per_sequence"]}
    buckets: dict[str, dict[str, dict[str, float]]] = {}
    for r in records:
        if r["sequence"] not in base:
            continue
        key = f"{r['engine']}/{r['setting']}"
        acc = buckets.setdefault(key, {lab: {"n": 0, "sum": 0.0, "max": 0.0}
                                       for _, lab in PROB_BUCKETS})
        for lp, dd in zip(base[r["sequence"]]["logp"], r["dlogp"]):
            p = math.exp(lp)
            lab = next(lab for lo, lab in PROB_BUCKETS if p > lo)
            a = acc[lab]
            a["n"] += 1
            a["sum"] += abs(dd)
            a["max"] = max(a["max"], abs(dd))
    by_bucket = {k: {lab: {"n": v["n"], "mean_abs_dlogp": v["sum"] / v["n"] if v["n"] else None,
                           "max_abs_dlogp": v["max"]} for lab, v in acc.items()}
                 for k, acc in buckets.items()}
    return {
        "hf_pairwise_identical_tokens": {"settings": names, "sequences": common,
                                         "n_tokens": sum(len(base[s]["logp"]) for s in common),
                                         "matrix": matrix},
        "abs_dlogp_by_baseline_prob": by_bucket,
        "prob_buckets": [lab for _, lab in PROB_BUCKETS],
    }


def _commands(args: argparse.Namespace, parts: list[dict[str, Any]]) -> list[str]:
    """Exact command lines: from ``--commands-file`` / ``--command``, else each part's argv."""
    cmds = list(args.command or [])
    if args.commands_file:
        cmds += [ln.strip() for ln in pathlib.Path(args.commands_file).read_text().splitlines()
                 if ln.strip()]
    return cmds or [" ".join(p["argv"]) for p in parts]


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


def markdown_matrix(records: list[dict[str, Any]], key: str = "n_bits_differ") -> str:
    """(engine/setting) x sequence matrix of ``key`` (the heatmap the report draws)."""
    seq_ids: list[str] = []
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for r in records:
        if r["sequence"] not in seq_ids:
            seq_ids.append(r["sequence"])
        rows.setdefault((r["engine"], r["setting"]), {})[r["sequence"]] = r
    short = [s.replace("gsm8k_test_", "t").replace("gsm8k_long_window", "long") for s in seq_ids]
    out = ["| engine/setting | " + " | ".join(short) + " |",
           "|---|" + "---|" * len(seq_ids)]
    for (eng, name), per in rows.items():
        cells = []
        for s in seq_ids:
            r = per.get(s)
            cells.append("-" if r is None else
                         (f"{r[key]}/{r['n_tokens']}" if key == "n_bits_differ"
                          else f"{r[key]:.2g}"))
        out.append(f"| {eng}/{name} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("hf", "hf-baseline", "hf-layers", "strict"):
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
    m.add_argument("--commands-file", default=None,
                   help="text file, one exact command line per non-empty line")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.cmd in ("hf", "hf-baseline", "hf-layers", "strict"):
        args.checkpoint = str(pathlib.Path(args.checkpoint).expanduser())
    if args.cmd == "hf":
        return cmd_hf(args)
    if args.cmd == "hf-baseline":
        return cmd_hf(args, baseline_only=True)
    if args.cmd == "hf-layers":
        return cmd_hf_layers(args)
    if args.cmd == "strict":
        return cmd_strict(args)
    return cmd_merge(args)


if __name__ == "__main__":
    raise SystemExit(main())
