#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""CLI for the Muse-Glimmer drift ablation (RFC #501 row ``museglimmer_ablation``).

Each GPU subcommand loads its model ONCE and writes a part file after every setting (a cut-off
run keeps everything measured so far); ``merge`` builds ``reports/muse_glimmer_poc/data/
ablation.json`` from the parts.  One GPU per process (run from the worktree root):

    R=rl_engine/models/muse_glimmer/ablation; L="~/bin/gpu-lease -n 1 --pool <GPU> --"
    $L env PYTHONPATH=$PWD ~/envs/vllm/bin/python $R/run_ablation.py hf --out $S/part_hf.json
    $L env PYTHONPATH=$PWD ~/envs/vllm/bin/python $R/run_ablation.py hf-baseline \
        --out $S/part_hf_baseline_gpuX.json        # null controls: other GPU / same GPU
    $L env PYTHONPATH=$PWD ~/envs/vllm/bin/python $R/run_ablation.py hf-layers \
        --out $S/part_hf_layers.json
    $L env PYTHONPATH=$PWD ~/envs/vllm/bin/python $R/run_ablation.py strict \
        --out $S/part_strict.json
    $L env PYTHONPATH=$PWD ~/envs/vllm/bin/python $R/run_ablation.py strict \
        --settings repeat_immediate --out $S/part_strict_gpuX.json
    PYTHONPATH=$PWD ~/envs/vllm/bin/python $R/run_ablation.py merge --parts $S/part_*.json \
        --out reports/muse_glimmer_poc/data/ablation.json
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
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

NULL_FACTORS = ("repeat_null", "second_gpu_null")
DEFAULT_TOKENS = "reports/muse_glimmer_poc/data/xrt_tokens.json"
DEFAULT_CKPT = "~/models/Muse-Glimmer-30B"
SCHEMA = "muse_glimmer_ablation_v2"


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True)


def _git_sha() -> str:
    try:
        return _git("rev-parse", "HEAD").strip()
    except Exception:  # pragma: no cover - git missing
        return "unknown"


def _git_state() -> dict[str, Any]:
    """``worktree_dirty`` over tracked files and, when dirty, the sha256 of ``git diff HEAD``
    (so a dirty measurement still names the exact code it ran)."""
    try:
        dirty = bool(_git("status", "--porcelain", "--untracked-files=no").strip())
        diff = _git("diff", "HEAD") if dirty else ""
    except Exception:  # pragma: no cover
        return {"worktree_dirty": True, "worktree_diff_sha256": None}
    return {"worktree_dirty": dirty,
            "worktree_diff_sha256": hashlib.sha256(diff.encode()).hexdigest() if dirty else None}


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
        **_git_state(),
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
                     records: list[dict[str, Any]], stats: dict[str, Any],
                     meta: dict[str, Any] | None = None) -> dict[str, Any]:
    out = {"engine": engine, "setting": name, "factor": factor, "factor_id": factor_id,
           "description": desc, "n_factors": 1, "one_factor_ref": None}
    if meta:
        out.update({k: v for k, v in meta.items() if k not in ("setting", "description")})
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
    cuda_inv = A.rope_inv_freq_formula(model.config, device)
    part["rope_inv_freq_cuda_hex"] = A.fp32_hex(cuda_inv)
    part["rope_inv_freq_cpu_vs_cuda_entries_differ"] = int(
        (A.fp32_bits(cuda_inv) != A.fp32_bits(model.rotary_emb.inv_freq)).sum())
    _log(f"HF text model loaded in {part['load_s']:.1f}s ({part['weights_gib']:.1f} GiB) on "
         f"{part['env'].get('gpu')} lease={part['env'].get('gpu_lease')}; inv_freq CPU vs CUDA "
         f"pow: {part['rope_inv_freq_cpu_vs_cuda_entries_differ']}/64 entries differ")
    part["baseline"] = {
        "name": "hf_baseline",
        "description": "HF 5.17 eager bf16, TF32 off, each sequence alone (L tokens), one-shot,"
                       " 5.17 matmul RoPE with CPU-built inv_freq, bf16 logits+softcap, fp32 "
                       "log_softmax",
    }
    base, stats = A.score_all(model, lm_head, cfg, seqs, driver="alone", keep_full=True)
    ref_full = [b.full_logp for b in base]
    for b in base:
        b.full_logp = None
    part["baseline"].update(stats)
    part["baseline"]["per_sequence"] = A.baseline_payload(seqs, base)
    part["baseline"]["n_top1_ties"] = sum(int((b.top2_gap == 0).sum()) for b in base)
    _log(f"baseline: {stats['wall_ms']:.0f} ms, peak {stats['peak_mem_gib']:.1f} GiB, "
         f"{part['baseline']['n_top1_ties']} positions with a top-1 tie")
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
            var, stats = A.run_setting(model, lm_head, cfg, seqs, st, ref_full=ref_full)
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
                                stats, st.summary())
        part["settings"].append(summ)
        _log(f"  {summ['n_bits_differ']}/{summ['n_tokens']} tokens differ, max|d| "
             f"{summ['max_abs_dlogp']:.3e}, mean|d| {summ['mean_abs_dlogp']:.3e}, k3 "
             f"{summ['mean_k3']:.3e}, KL {summ['mean_kl_full']:.3e}, argmax flips "
             f"{summ['argmax_flips']} (unique {summ['argmax_flips_unique']}, at ties "
             f"{summ['argmax_flips_at_ties']}), {stats['wall_ms']:.0f} ms, peak "
             f"{stats['peak_mem_gib']:.1f} GiB")
        _write(out, part)
    if not baseline_only and not args.settings:
        _log("repeat_after_all_factors (restore + repeat null control) ...")
        var, stats = A.score_all(model, lm_head, cfg, seqs, driver="alone", ref_full=ref_full)
        recs = A.make_records("hf", "repeat_after_all_factors", "repeat_null", 9, "hf_baseline",
                              seqs, base, var)
        part["records"].extend(recs)
        part["settings"].append(_setting_summary(
            "hf", "repeat_after_all_factors", "repeat_null", 9,
            "baseline again after every factor ran and was restored", recs, stats,
            {"n_factors": 0}))
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
    ("prefill_L_minus_1", "prefill_length", 3, "prefill_minus_last", None, ()),
    ("chunked_prefill_256", "chunked_prefill", 3, "chunked", 256, ()),
    ("sdpa", "attention_impl", 4, "alone", None, ("sdpa",)),
    ("tf32_on", "tf32", 5, "alone", None, ("tf32_on",)),
    ("rope_inv_freq_cuda", "rope_table", 7, "alone", None, ("rope_inv_freq_cuda",)),
    ("bf16_reduced_reduction_off", "gemm_reduction", 10, "alone", None, ("bf16_rr_off",)),
)
LAYER_SEQUENCES = ("gsm8k_test_0", "gsm8k_long_window")
LAYER_SUBLAYERS = (0,)


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
    part["layer_points"] = {"rope": True, "sublayers": list(LAYER_SUBLAYERS),
                            "sublayer_points": list(A.SUBLAYER_POINTS)}
    _write(out, part)
    by_id = {s.id: i for i, s in enumerate(seqs)}
    kw = {"rope": True, "sublayers": LAYER_SUBLAYERS}
    for sid in LAYER_SEQUENCES:
        target = by_id[sid]
        base = A.layer_trace(model, seqs, target, "alone", **kw)
        for name, factor, fid, driver, chunk, ctx in LAYER_SETTINGS:
            if args.settings and name not in set(args.settings):
                continue
            torch.cuda.synchronize(device)
            t1 = time.perf_counter()
            with A.enter_contexts(model, ctx):
                var = A.layer_trace(model, seqs, target, driver, chunk, **kw)
            torch.cuda.synchronize(device)
            points = A.layer_diff(base, var, seqs[target].completion_start)
            first = next((r["point"] for r in points if r["frac_elems_differ"] > 0), None)
            layers = [r for r in points if A.is_residual_point(r["point"])]
            part["layer_records"].append({
                "engine": "hf", "setting": name, "factor": factor, "factor_id": fid,
                "sequence": sid, "first_point_with_diff": first,
                "first_residual_point_with_diff": next(
                    (r["point"] for r in layers if r["frac_elems_differ"] > 0), None),
                "wall_ms": (time.perf_counter() - t1) * 1e3,
                "layers": layers,  # embedding, layer_0..layer_51, final_norm
                "detail_points": [r for r in points if not A.is_residual_point(r["point"])],
                "execution_order": [r["point"] for r in points],
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
# Kinds that run through ablation_strict.packed_pass (no public multi-sequence entry point);
# they only run after selfcheck_packed_pass proved packed_pass byte-equal to the reference.
PACKED_KINDS = ("packed", "paged_prefill", "continuous_mixed")
STRICT_SETTINGS = (
    # name, factor, factor_id, description, kind, arg
    ("repeat_immediate", "repeat_null", 9, "train-mode baseline again", "alone", ("train", None)),
    ("packed_batch9", "batch_composition", 1,
     "all 9 sequences in one varlen training pass (cu_seqlens; packed_pass)", "packed", "packed"),
    ("packed_batch9_reversed", "batch_composition", 1,
     "same, reversed batch order (every sequence at a new row offset)", "packed",
     "packed_reversed"),
    ("right_pad_batch9", "batch_composition", 1,
     "every segment right-padded to 2439 tokens (21951 rows)", "packed", "right_pad"),
    ("left_pad_batch9", "batch_composition", 1,
     "a pad segment in front of every shorter sequence (left padding, positions from 0)",
     "packed", "left_pad"),
    ("paged_prefill_batch9", "batch_composition", 1,
     "rollout form: one paged prefill call holding L-1 tokens of all 9 sequences (packed_pass)",
     "paged_prefill", None),
    ("prefill_alone", "prefill_vs_decode", 2, "reference mode=prefill (paged, L-1 tokens)",
     "alone", ("prefill", None)),
    ("decode_alone", "prefill_vs_decode", 2,
     "reference mode=decode: prompt prefill + 1-token decode steps", "alone", ("decode", None)),
    ("batched_decode9", "prefill_vs_decode", 2,
     "continuous batching, public API only: each prompt through model.prefill, then ONE "
     "model.decode_batch call per step over every running sequence (up to 9 rows)",
     "continuous", (1, 0)),
    ("batched_decode72_staggered", "prefill_vs_decode", 2,
     "8 copies of every sequence (72 slots), copy c admitted at step 4c; public API "
     "(model.prefill + one model.decode_batch per step); decode batch grows to 72 rows",
     "continuous", (8, 4)),
    ("mixed_step72_staggered", "batch_composition", 1,
     "the same 72 slots, every engine step ONE packed paged call holding the admitted prompts "
     "AND one token of every running sequence (vLLM-style mixed prefill/decode; packed_pass)",
     "continuous_mixed", (8, 4)),
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
    todo = [s for s in STRICT_SETTINGS if wanted is None or s[0] in wanted]
    packed_ok = True
    if any(s[4] in PACKED_KINDS for s in todo):
        shortest = sorted(seqs, key=lambda s: s.length)[:2]
        try:
            part["packed_pass_selfcheck"] = S.selfcheck_packed_pass(model, shortest)
            _log(f"packed_pass selfcheck: {part['packed_pass_selfcheck']}")
        except RuntimeError as exc:
            packed_ok = False
            part["packed_pass_selfcheck"] = {"all_equal": False, "error": str(exc)[:2000]}
            part["errors"].append({"setting": "packed_pass_selfcheck", "error": str(exc)[:500]})
            _log(f"packed_pass selfcheck FAILED, packed settings not run: {exc}")
        _write(out, part)
    for name, factor, fid, desc, kind, arg in todo:
        if kind in PACKED_KINDS and not packed_ok:
            part["errors"].append({"setting": name,
                                   "error": "not run: packed_pass selfcheck failed"})
            continue
        _log(f"strict setting {name} ...")
        torch.cuda.reset_peak_memory_stats(device)
        t1 = time.perf_counter()
        extra: dict[str, Any] = {}
        run = None
        try:
            if kind == "alone":
                var = S.score_alone(model, seqs, arg[0], arg[1])
            elif kind == "packed":
                var = S.score_packed_batch(model, seqs, arg)
            elif kind == "paged_prefill":
                var = S.score_paged_batch_prefill(model, seqs)
            elif kind in ("continuous", "continuous_mixed"):
                copies, stagger = arg
                run = S.score_continuous_batching(model, seqs, copies=copies, stagger=stagger,
                                                  mixed=kind == "continuous_mixed")
                var = run.first_copy
                extra.update({
                    "copies": copies, "stagger_steps": stagger, "engine_steps": run.steps,
                    "decode_steps": sum(1 for r in run.decode_rows_per_step if r),
                    "max_decode_rows": run.max_decode_rows,
                    "decode_rows_per_step": run.decode_rows_per_step,
                    "prefill_seqs_per_step": run.prefill_seqs_per_step,
                    "api": "packed_pass (selfchecked)" if kind == "continuous_mixed"
                    else "public: model.prefill + model.decode_batch",
                })
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
        if run is not None:
            per = S.copies_bits_differ(base, run)
            for rec, row in zip(recs, per):
                rec["copies_n_bits_differ"] = row
            stats["n_tokens_all_copies"] = sum(s.n_completion for s in seqs) * len(run.per_copy)
            stats["n_bits_differ_all_copies"] = sum(sum(r) for r in per)
        part["records"].extend(recs)
        summ = _setting_summary("strict", name, factor, fid, desc, recs, stats)
        part["settings"].append(summ)
        _log(f"  {summ['n_bits_differ']}/{summ['n_tokens']} tokens differ"
             + (f" (all copies {stats['n_bits_differ_all_copies']}/"
                f"{stats['n_tokens_all_copies']}, max {extra['max_decode_rows']} decode rows)"
                if run is not None else "")
             + f", max|d| {summ['max_abs_dlogp']:.3e}, {stats['wall_ms']:.0f} ms, peak "
             f"{stats['peak_mem_gib']:.1f} GiB")
        del run
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
        am, gap = ps.get("argmax"), ps.get("top2_gap")
        out.append(A.SeqScore(A.hex_to_fp32(ps["logp_hex"]),
                              None if am is None else torch.tensor(am, dtype=torch.long),
                              ps["wall_ms"],
                              top2_gap=None if gap is None else torch.tensor(gap,
                                                                            dtype=torch.float32)))
    return out


def _part_key(i: int, p: dict[str, Any]) -> str:
    """Unique key of a part (index + kind + GPU + file stem), used for the env map."""
    return f"{i}:{p['part']}@gpu{p['env'].get('gpu_lease')}:{pathlib.Path(p['_path']).stem}"


def cmd_merge(args: argparse.Namespace) -> int:
    paths = list(args.parts)
    parts = [json.loads(pathlib.Path(p).read_text()) for p in paths]
    for p, path in zip(parts, paths):
        p["_path"] = str(path)
    payload = merge_parts(parts, args.tokens, commands=_commands(args, parts),
                          tokenizer=args.tokenizer)
    payload["previous_runs"] = [compare_previous(payload, json.loads(pathlib.Path(p).read_text()),
                                                 str(p)) for p in (args.previous or [])]
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1))
    _log(f"wrote {out}: {len(payload['records'])} records, {len(payload['settings'])} settings, "
         f"status {payload['status']}")
    print(markdown_table(payload["settings"]))
    print()
    print(markdown_matrix(payload["records"]))
    print()
    print(markdown_matrix(payload["records"], "max_abs_dlogp"))
    return 0


def merge_parts(parts: list[dict[str, Any]], tokens: str, *, commands: list[str] | None = None,
                tokenizer: str | None = None) -> dict[str, Any]:
    """Build the ``ablation.json`` payload from loaded part dicts (each with ``_path``)."""
    by_kind: dict[str, list[dict[str, Any]]] = {}
    for p in parts:
        by_kind.setdefault(p["part"], []).append(p)
    if "hf" not in by_kind:
        raise SystemExit("merge needs the main 'hf' part")
    hf_parts = sorted(by_kind["hf"], key=lambda p: -len(p["settings"]))
    hf = hf_parts[0]
    seqs, _ = A.load_xrt_tokens(tokens)
    records = list(hf["records"])
    settings = list(hf["settings"])
    hf_base = _scores_from_baseline(hf["baseline"])
    # Later HF runs (extra settings) count only if their own baseline is bitwise the main one.
    base_hex = [p["logp_hex"] for p in hf["baseline"]["per_sequence"]]
    extra_hf = []

    def null_control(other: dict[str, Any], engine: str, base_scores: list[A.SeqScore],
                     base_name: str, desc: str) -> tuple[str, list[dict[str, Any]]]:
        main = hf if engine == "hf" else strict_parts[0]
        gpu_a, gpu_b = main["env"].get("gpu_lease"), other["env"].get("gpu_lease")
        same = gpu_a == gpu_b
        name = f"new_process_same_gpu_{gpu_b}" if same else f"second_gpu_{gpu_b}_vs_{gpu_a}"
        have = {s["setting"] for s in settings if s["engine"] == engine}
        k = 2
        while name in have:
            name = f"{name.rsplit('#', 1)[0]}#{k}"
            k += 1
        factor, fid = ("repeat_null", 9) if same else ("second_gpu_null", 8)
        recs = A.make_records(engine, name, factor, fid, base_name, seqs, base_scores,
                              _scores_from_baseline(other["baseline"]),
                              extra={"gpu_lease_variant": gpu_b, "gpu_lease_baseline": gpu_a,
                                     "variant_part": other["_path"]})
        records.extend(recs)
        settings.append(_setting_summary(
            engine, name, factor, fid,
            f"{desc} in a separate process ({other.get('started')}) on GPU {gpu_b}; main ran on "
            f"GPU {gpu_a}", recs,
            {"wall_ms": other["baseline"]["wall_ms"],
             "peak_mem_gib": other["baseline"]["peak_mem_gib"]}, {"n_factors": 0}))
        return name, recs

    strict_parts = sorted(by_kind.get("strict", []), key=lambda p: -len(p["settings"]))
    for other in hf_parts[1:]:
        same = [p["logp_hex"] for p in other["baseline"]["per_sequence"]] == base_hex
        extra_hf.append({"path": other["_path"], "gpu_lease": other["env"].get("gpu_lease"),
                         "baseline_bitwise_equal_to_main": same})
        null_control(other, "hf", hf_base, "hf_baseline", "HF baseline re-measured")
        if not same:
            continue
        have = {s["setting"] for s in settings}
        records.extend(r for r in other["records"] if r["setting"] not in have)
        settings.extend(s for s in other["settings"] if s["setting"] not in have)
    # Factor 8 / 9: the same baseline in a separate process (other GPU or same GPU).
    for other in by_kind.get("hf_baseline_only", []):
        name, _ = null_control(other, "hf", hf_base, "hf_baseline", "HF baseline")
        for r in other["records"]:  # its own immediate repeat
            records.append(dict(r, setting=f"{r['setting']}@{name}",
                                baseline=f"hf_baseline@{name}"))
        settings.extend(dict(s, setting=f"{s['setting']}@{name}") for s in other["settings"])
    # Factor 3 etc. one factor at a time, derived from the stored exact per-token values.
    d_recs, d_sets = A.derive_one_factor_records(records, settings, "hf")
    records.extend(d_recs)
    settings.extend(d_sets)
    layer_parts = by_kind.get("hf_layers", [])
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
            recs, {}, {"n_factors": None}))
        for other in strict_parts[1:]:
            null_control(other, "strict", sb, "strict_train_alone", "strict baseline")
    for s in settings:  # null controls flip nothing, whatever the part recorded
        if s.get("factor") in NULL_FACTORS:
            s["n_factors"] = 0
    commits = sorted({p["commit"] for p in parts})
    have = {(s["engine"], s["factor"]) for s in settings}
    missing = [name for name, ok in (
        ("all parts complete", all(p.get("status") == "complete" for p in parts)),
        ("strict part", bool(strict_parts)),
        ("hf second-GPU null control", ("hf", "second_gpu_null") in have),
        ("strict second-GPU null control", ("strict", "second_gpu_null") in have),
        ("hf new-process null control", any(
            s["engine"] == "hf" and s["setting"].startswith("new_process_same_gpu")
            for s in settings)),
        ("hf_layers part", bool(layer_parts)),
    ) if not ok]
    status = "partial" if missing else "complete"
    head = _git_sha()
    hf_lookup = {s.name: s for s in A.HF_SETTINGS}
    payload = {
        "row": "museglimmer_ablation",
        "schema": SCHEMA,
        "status": status,
        "status_missing": missing,
        "commit": hf["commit"],
        "commits_all_parts": commits,
        "head_at_merge": head,
        **{f"{k}_at_merge": v for k, v in _git_state().items()},
        "all_parts_at_one_clean_commit": len(commits) == 1
        and not any(p.get("worktree_dirty") for p in parts),
        "scope": "HF transformers 5.17.0 (eager/SDPA, bf16) and the strict reference, one "
                 "A100-SXM4-80GB at a time; vLLM 0.31 and Megatron were NOT run in this row",
        "env": {_part_key(i, p): p["env"] for i, p in enumerate(parts)},
        "commands": commands or [" ".join(p["argv"]) for p in parts],
        "tokens": hf["tokens"],
        "metric_definitions": {
            "d": "logp_variant - logp_baseline of the gold token (fp32 values, fp64 difference)",
            "bits_differ": "fp32 bit patterns of the two logp values differ",
            "k3": "exp(d) - d - 1 (fp64) of the gold token; a mismatch size, NOT a KL (the "
                  "tokens are gold answers, not samples of either distribution)",
            "kl_full": "KL(baseline || variant) over the full vocabulary per position, fp64 sum "
                       "of the two fp32 log_softmax rows (HF settings measured in the main "
                       "part only; None for derived / cross / strict records)",
            "argmax_flip": "argmax id of the variant logits != the baseline's (HF only); "
                           "var_argmax holds the variant ids",
            "argmax_flips_unique / argmax_flips_at_ties": "flips where both top-1 logits are "
                                                         "unique / where either run has a top-1"
                                                         " tie (top2_gap == 0; the bf16 logits "
                                                         "tie exactly and argmax returns the "
                                                         "lowest tied id)",
            "derived_from_stored_values": "record computed in the merge from two measured "
                                          "settings' exact per-token values (logp_hex, argmax "
                                          "ids, top-2 gaps), no extra forward",
            "dlogp/k3/kl_full arrays": "rounded to 6/4/6 significant digits; logp_hex is exact",
            "n_factors": "factors the comparison against its baseline flips (0 = null control)",
        },
        "settings_table": {
            "hf": [s.summary() for s in A.HF_SETTINGS],
            "hf_skipped": list(A.SKIPPED_SETTINGS),
            "strict": [{"setting": n, "factor": f, "factor_id": i, "description": d,
                        "kind": k, "arg": a, "via_packed_pass": k in PACKED_KINDS}
                       for n, f, i, d, k, a in STRICT_SETTINGS],
            "hf_layer_settings": [{"setting": n, "factor": f, "factor_id": i, "driver": dr,
                                   "chunk": c, "contexts": list(cx)}
                                  for n, f, i, dr, c, cx in LAYER_SETTINGS],
        },
        "baselines": {
            "hf": hf["baseline"],
            **({"strict": strict_parts[0]["baseline"]} if strict_parts else {}),
            **{f"hf@{_part_key(i, p)}": p["baseline"] for i, p in enumerate(parts)
               if p["part"] == "hf_baseline_only"},
        },
        "parts": [{"part": p["part"], "path": p["_path"], "commit": p["commit"],
                   "worktree_dirty": p.get("worktree_dirty"),
                   "worktree_diff_sha256": p.get("worktree_diff_sha256"),
                   "load_s": p.get("load_s"), "weights_gib": p.get("weights_gib"),
                   "started": p.get("started"), "finished": p.get("finished"),
                   "status": p.get("status"), "errors": p.get("errors"),
                   "gpu_lease": p["env"].get("gpu_lease"), "argv": p.get("argv"),
                   "env": p["env"], "key": _part_key(i, p)}
                  for i, p in enumerate(parts)],
        "rope_inv_freq_hex": hf.get("rope_inv_freq_hex"),
        "rope_inv_freq_cuda_hex": hf.get("rope_inv_freq_cuda_hex"),
        "rope_inv_freq_cpu_vs_cuda_entries_differ": hf.get(
            "rope_inv_freq_cpu_vs_cuda_entries_differ"),
        "strict_ops": strict_parts[0].get("ops") if strict_parts else None,
        "packed_pass_selfcheck": strict_parts[0].get("packed_pass_selfcheck")
        if strict_parts else None,
        "skipped": hf.get("skipped", []),
        "extra_hf_parts": extra_hf,
        "tf32_probe": layer_parts[0].get("tf32_probe") if layer_parts else None,
        "layer_points": layer_parts[0].get("layer_points") if layer_parts else None,
        "layer_records": [r for p in layer_parts for r in p.get("layer_records", [])],
        "settings": settings,
        "analysis": analysis(records, settings, hf["baseline"], tokenizer),
        "records": records,
    }
    unknown = [s["setting"] for s in hf["settings"]
               if s["setting"] not in hf_lookup and s["setting"] != "repeat_after_all_factors"]
    if unknown:
        payload["unregistered_hf_settings"] = unknown
    payload["headline"] = headline(payload)
    return payload


def compare_previous(payload: dict[str, Any], prev: dict[str, Any], path: str) -> dict[str, Any]:
    """Cross-run repeat: tokens whose baseline logp bits differ between this merge and an
    earlier run (a merged ``ablation.json`` or a single part file)."""
    if "baselines" in prev:
        old = {k: prev["baselines"].get(k) for k in ("hf", "strict")}
        commit = prev.get("commit")
        commits = prev.get("commits_all_parts") or prev.get("commits")
    else:
        kind = "strict" if prev.get("part") == "strict" else "hf"
        old = {kind: prev.get("baseline")}
        commit, commits = prev.get("commit"), None
    out: dict[str, Any] = {"path": path, "commit": commit, "commits_all_parts": commits}
    for k, b in old.items():
        new = payload["baselines"].get(k)
        if not b or not new:
            out[k] = None
            continue
        ob = {p["sequence"]: p["logp_hex"] for p in b["per_sequence"]}
        per = {p["sequence"]: sum(x != y for x, y in zip(p["logp_hex"], ob.get(p["sequence"], [])))
               for p in new["per_sequence"]}
        out[k] = {"tokens_differ": sum(per.values()),
                  "n_tokens": sum(len(p["logp_hex"]) for p in new["per_sequence"]),
                  "per_sequence": per}
    return out


PROB_BUCKETS = ((0.9, "p>0.9"), (0.5, "0.5<p<=0.9"), (0.1, "0.1<p<=0.5"), (-1.0, "p<=0.1"))


def _bucket(lp: float) -> str:
    p = math.exp(lp)
    return next(lab for lo, lab in PROB_BUCKETS if p > lo)


def _low_p_view(baseline: dict[str, Any], tokenizer: str | None) -> dict[str, Any]:
    """Where the low-probability gold tokens are: inside GSM8K ``<<...>>`` annotations, the
    first completion token, or elsewhere (decoded with ``tokenizer.json`` as data)."""
    if not tokenizer:
        return {"status": "NOT RUN (no tokenizer given)"}
    path = pathlib.Path(tokenizer).expanduser()
    if not path.is_file():
        return {"status": f"NOT RUN ({path} missing)"}
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(path))
    dec = lambda ids: tok.decode(ids, skip_special_tokens=False)  # noqa: E731
    per_bucket = {lab: {"n": 0, "n_in_calc": 0, "n_first_completion_token": 0}
                  for _, lab in PROB_BUCKETS}
    low = []
    pieces = {}
    for ps in baseline["per_sequence"]:
        text, spans = A.token_char_spans(dec, ps["targets"])
        calc = A.calc_annotation_flags(text, spans)
        pieces[ps["sequence"]] = [text[a:b] for a, b in spans]
        for i, (lp, (a, b), c) in enumerate(zip(ps["logp"], spans, calc)):
            lab = _bucket(lp)
            acc = per_bucket[lab]
            acc["n"] += 1
            acc["n_in_calc"] += c
            acc["n_first_completion_token"] += int(i == 0)
            if lab == "p<=0.1":
                low.append({"sequence": ps["sequence"], "index": i, "logp": lp,
                            "text": text[a:b], "in_calc": bool(c)})
    return {"status": "ok", "tokenizer": str(path),
            "tokenizer_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "by_bucket": per_bucket, "low_p_tokens": low, "target_pieces": pieces}


def analysis(records: list[dict[str, Any]], settings: list[dict[str, Any]],
             baseline: dict[str, Any], tokenizer: str | None = None) -> dict[str, Any]:
    """Derived views for the report: one-factor table, which measured HF settings produce
    identical bits on the same token (pairwise), |dlogp| by the baseline probability of the
    gold token, the baseline's top-1 ties, and where the low-probability gold tokens sit."""
    hf_recs = [r for r in records
               if r["engine"] == "hf" and not r.get("derived_from_stored_values")]
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
            a = acc[_bucket(lp)]
            a["n"] += 1
            a["sum"] += abs(dd)
            a["max"] = max(a["max"], abs(dd))
    by_bucket = {k: {lab: {"n": v["n"], "mean_abs_dlogp": v["sum"] / v["n"] if v["n"] else None,
                           "max_abs_dlogp": v["max"]} for lab, v in acc.items()}
                 for k, acc in buckets.items()}
    ties = {p["sequence"]: [i for i, t in enumerate(p.get("top1_tie") or []) if t]
            for p in baseline["per_sequence"]}
    one_factor = [
        {k: s.get(k) for k in ("engine", "setting", "factor", "factor_id", "n_tokens",
                               "n_bits_differ", "frac_bits_differ", "mean_abs_dlogp",
                               "max_abs_dlogp", "mean_k3", "mean_kl_full", "argmax_flips",
                               "argmax_flips_unique", "argmax_flips_at_ties",
                               "derived_from_stored_values", "reference_setting")}
        for s in settings if s.get("n_factors") == 1
    ]
    return {
        "one_factor_settings": one_factor,
        "hf_pairwise_identical_tokens": {"settings": names, "sequences": common,
                                         "n_tokens": sum(len(base[s]["logp"]) for s in common),
                                         "matrix": matrix},
        "abs_dlogp_by_baseline_prob": by_bucket,
        "prob_buckets": [lab for _, lab in PROB_BUCKETS],
        "hf_baseline_top1_ties": {"n": sum(len(v) for v in ties.values()), "positions": ties},
        "low_p_gold_tokens": _low_p_view(baseline, tokenizer),
    }


def headline(payload: dict[str, Any]) -> str:
    """One data-derived sentence for the report header."""
    one = [s for s in payload["analysis"]["one_factor_settings"] if s["engine"] == "hf"]
    strict = [s for s in payload["settings"] if s["engine"] == "strict"]
    if not one:
        return ""
    moved = [s for s in one if s["n_bits_differ"]]
    n = one[0]["n_tokens"]
    tot_strict = sum(s.get("n_tokens_all_copies") or s["n_tokens"] for s in strict)
    bad_strict = sum(s.get("n_bits_differ_all_copies") or s["n_bits_differ"] for s in strict)
    rows = max((s.get("max_decode_rows") or 0 for s in strict), default=0)
    return (f"HF transformers 5.17 bf16 on one A100: {len(moved)} of {len(one)} one-factor "
            f"settings change logprob bits (up to {max(s['n_bits_differ'] for s in one)} of "
            f"{n} gold tokens); strict reference: {bad_strict} of {tot_strict} scored tokens "
            f"differ over {len(strict)} strict settings (batching, padding, paged prefill, "
            f"decode, continuous batching with up to {rows} decode rows, chunking, repeats).")


def _commands(args: argparse.Namespace, parts: list[dict[str, Any]]) -> list[str]:
    """Exact command lines: from ``--commands-file`` / ``--command``, else each part's argv."""
    cmds = list(args.command or [])
    if args.commands_file:
        cmds += [ln.strip() for ln in pathlib.Path(args.commands_file).read_text().splitlines()
                 if ln.strip()]
    return cmds or [" ".join(p["argv"]) for p in parts]


def _fmt(x: Any, spec: str = ".3g") -> str:
    return "-" if x is None else format(x, spec)


def markdown_table(settings: list[dict[str, Any]]) -> str:
    head = ("| engine | factor | setting | n factors | tokens differ | max abs dlogp | "
            "mean abs dlogp | mean k3 | mean KL | argmax flips (unique / at ties) | wall ms |\n"
            "|---|---|---|---|---|---|---|---|---|---|---|")
    rows = [head]
    for s in settings:
        flips = ("-" if s.get("argmax_flips") is None else
                 f"{s['argmax_flips']} ({_fmt(s.get('argmax_flips_unique'), 'd')} / "
                 f"{_fmt(s.get('argmax_flips_at_ties'), 'd')})")
        differ = f"{s['n_bits_differ']}/{s['n_tokens']}"
        if s.get("n_tokens_all_copies"):
            differ += f" (all copies {s['n_bits_differ_all_copies']}/{s['n_tokens_all_copies']})"
        rows.append(
            f"| {s['engine']} | {s['factor_id']} {s['factor']} | {s['setting']} | "
            f"{_fmt(s.get('n_factors'), 'd') if isinstance(s.get('n_factors'), int) else '-'} | "
            f"{differ} | {s['max_abs_dlogp']:.3g} | {s['mean_abs_dlogp']:.3g} | "
            f"{s['mean_k3']:.3g} | {_fmt(s.get('mean_kl_full'))} | {flips} | "
            f"{_fmt(s.get('wall_ms'), '.0f')} |")
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
    m.add_argument("--tokenizer", default=f"{DEFAULT_CKPT}/tokenizer.json",
                   help="tokenizer.json (read as data) for the low-probability token view; "
                        "'' to skip")
    m.add_argument("--previous", nargs="*", default=None,
                   help="earlier ablation.json / part files: record how many baseline tokens "
                        "differ from them (cross-run repeat)")
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
