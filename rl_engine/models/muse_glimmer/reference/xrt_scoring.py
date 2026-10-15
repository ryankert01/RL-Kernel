# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Cross-runtime scoring of the shared token file with the strict reference (RFC #501).

Input: ``muse_glimmer_xrt_v1`` (``reports/muse_glimmer_poc/data/xrt_tokens.json``): sequences of
token ids with a ``completion_start``. Output: ``muse_glimmer_xrt_logprobs_v1`` with
``logprobs_hex[k] = '%08x' % uint32(fp32 bits of log p(token[cs+k] | token[:cs+k]))`` from the
TRAIN forward (varlen attention, all positions at once), so the vLLM / Megatron dumps can be
byte-diffed against it. The same tokens are also run through the rollout forms (one-shot paged
prefill, prompt prefill + token-by-token decode, optional chunkings) and compared bitwise with
the train forward; the result is recorded under ``checks``.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import subprocess
import time
from typing import Any, Callable, Sequence

import torch

from rl_engine.models.muse_glimmer.reference.strict_model import MuseGlimmerStrictModel

SCHEMA_IN = "muse_glimmer_xrt_v1"
SCHEMA_OUT = "muse_glimmer_xrt_logprobs_v1"
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]


def source_commit() -> dict[str, Any]:
    """HEAD sha of the checkout this module runs from, and whether tracked files are dirty."""

    def run(*args: str) -> str:
        try:
            proc = subprocess.run(["git", "-C", str(_REPO_ROOT), *args], capture_output=True,
                                  text=True, check=False)
        except OSError:
            return ""
        return proc.stdout.strip()

    return {"sha": run("rev-parse", "HEAD"),
            "dirty": bool(run("status", "--porcelain", "--untracked-files=no"))}


def fp32_hex(values: torch.Tensor) -> list[str]:
    """'%08x' of the fp32 bit pattern (as uint32) of every value."""
    bits = values.detach().float().cpu().contiguous().view(torch.int32).tolist()
    return [f"{b & 0xFFFFFFFF:08x}" for b in bits]


def hex_to_fp32(hexes: Sequence[str]) -> torch.Tensor:
    ints = [int(h, 16) for h in hexes]
    ints = [i - (1 << 32) if i >= (1 << 31) else i for i in ints]
    return torch.tensor(ints, dtype=torch.int32).view(torch.float32)


def _parse_mode(spec: str, n_scored: int, prompt_len: int) -> tuple[str, list[int] | None]:
    """``prefill`` | ``decode`` | ``per_token`` | ``chunked:<name or n1/n2/...>`` with names
    ``prompt_boundary``, ``halves``, ``block64``, ``ragged``, ``window_boundary_<b>``.
    Raises ``LookupError`` when a chunking does not apply to this sequence length."""
    from rl_engine.models.muse_glimmer.reference.chain_harness import (
        _RAGGED_PATTERN,
        _cycle_chunks,
    )

    mode, _, arg = spec.partition(":")
    if mode != "chunked":
        return mode, None
    if arg == "prompt_boundary":
        sizes = [prompt_len, n_scored - prompt_len]
    elif arg == "halves":
        sizes = [n_scored // 2, n_scored - n_scored // 2]
    elif arg == "block64":
        sizes = list(_cycle_chunks(n_scored, (64,)))
    elif arg == "ragged":
        sizes = list(_cycle_chunks(n_scored, _RAGGED_PATTERN))
    elif arg.startswith("window_boundary_"):
        b = int(arg.rsplit("_", 1)[1])
        sizes = [b, n_scored - b]
    else:
        sizes = [int(x) for x in arg.split("/")]
    if sum(sizes) != n_scored or any(c <= 0 for c in sizes):
        raise LookupError(f"{spec} does not apply to {n_scored} scored tokens")
    return mode, sizes


def score_token_file(
    model: MuseGlimmerStrictModel,
    score_file: str | pathlib.Path,
    *,
    check_modes: Sequence[str] = ("prefill", "decode"),
    log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Score every sequence with the train forward and compare ``check_modes`` bitwise.

    ``check_modes`` entries: ``prefill``, ``decode``, ``per_token`` or
    ``chunked:<n1>/<n2>/...`` (chunk sizes over the scored ``len - 1`` tokens).
    """
    commit = source_commit()  # captured when the run starts (the code that is measured)
    raw = pathlib.Path(score_file).read_bytes()
    data = json.loads(raw)
    if data.get("schema") != SCHEMA_IN:
        raise ValueError(f"{score_file}: schema {data.get('schema')!r} != {SCHEMA_IN}")
    dev = model.device
    seqs_out: list[dict[str, Any]] = []
    checks: list[dict[str, Any]] = []
    for seq in data["sequences"]:
        ids = torch.tensor(seq["token_ids"], dtype=torch.long)
        cs = int(seq["completion_start"])
        prompt, completion = ids[:cs], ids[cs:]
        torch.cuda.synchronize(dev)
        t0 = time.perf_counter()
        lp = model.logprobs(prompt, completion, mode="train").float().cpu()
        torch.cuda.synchronize(dev)
        train_s = time.perf_counter() - t0
        seqs_out.append({
            "id": seq["id"],
            "logprobs_hex": fp32_hex(lp),
            "n_tokens": int(lp.numel()),
            "completion_start": cs,
            "length": int(ids.numel()),
            "logprobs": [float(x) for x in lp.tolist()],
        })
        row: dict[str, Any] = {"id": seq["id"], "length": int(ids.numel()),
                               "completion_start": cs, "n_tokens": int(lp.numel()),
                               "train_seconds": train_s, "modes": {}}
        for spec in check_modes:
            try:
                mode, sizes = _parse_mode(spec, int(ids.numel()) - 1, cs)
            except LookupError as exc:
                row["modes"][spec] = {"skipped": str(exc)}
                continue
            torch.cuda.synchronize(dev)
            t1 = time.perf_counter()
            other = model.logprobs(prompt, completion, mode=mode, chunk_sizes=sizes)
            other = other.float().cpu()
            torch.cuda.synchronize(dev)
            same = torch.eq(other, lp)
            bad = (~same).nonzero()
            row["modes"][spec] = {
                "seconds": time.perf_counter() - t1,
                "equal_tokens": int(same.sum()),
                "n_tokens": int(lp.numel()),
                "byte_equal": bool(same.all()),
                "first_mismatch": None if bad.numel() == 0 else int(bad[0, 0]),
                "max_abs_diff": float((other.double() - lp.double()).abs().max()),
            }
        checks.append(row)
        if log:
            log(f"[score] {seq['id']}: {lp.numel()} tokens, sum logp "
                f"{float(lp.double().sum()):.4f}, train {train_s:.2f}s; "
                + ", ".join(f"{m}: {v['equal_tokens']}/{v['n_tokens']} ({v['seconds']:.1f}s)"
                            for m, v in row["modes"].items() if "skipped" not in v))
    all_equal = all(v["byte_equal"] for r in checks for v in r["modes"].values()
                    if "skipped" not in v)
    return {
        "schema": SCHEMA_OUT,
        "runtime": "reference",
        "mode": "train",
        "tp": 1,
        "backend": model.ops.backend,
        "sequences": seqs_out,
        "source": {"path": str(score_file), "sha256": hashlib.sha256(raw).hexdigest(),
                   "checkpoint_revision": data.get("checkpoint_revision")},
        "checks": {"modes": list(check_modes), "all_byte_equal_to_train": all_equal,
                   "sequences": checks},
        "commit": commit,
        "ops_provenance": dict(model.ops.provenance),
        "device": torch.cuda.get_device_name(dev),
        "torch": torch.__version__,
    }


def compare_xrt(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """Byte-diff two muse_glimmer_xrt_logprobs_v1 payloads; per-sequence counts and |Δ|."""
    by_id = {s["id"]: s for s in b["sequences"]}
    rows = []
    for s in a["sequences"]:
        o = by_id.get(s["id"])
        if o is None:
            rows.append({"id": s["id"], "missing": True})
            continue
        x, y = hex_to_fp32(s["logprobs_hex"]), hex_to_fp32(o["logprobs_hex"])
        same = sum(1 for p, q in zip(s["logprobs_hex"], o["logprobs_hex"], strict=True) if p == q)
        d = (x.double() - y.double()).abs()
        rows.append({"id": s["id"], "n_tokens": len(s["logprobs_hex"]), "equal_tokens": same,
                     "max_abs_diff": float(d.max()), "mean_abs_diff": float(d.mean()),
                     "abs_diff": [float(v) for v in d.tolist()]})
    n = sum(r.get("n_tokens", 0) for r in rows)
    eq = sum(r.get("equal_tokens", 0) for r in rows)
    return {"n_tokens": n, "equal_tokens": eq, "sequences": rows}


@torch.no_grad()
def score_token_file_hf(
    hf_model: Any,
    score_file: str | pathlib.Path,
    *,
    log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Native anchor: the same tokens through transformers' bf16 model (one forward per
    sequence, its softcapped bf16 logits, fp32 log_softmax + gather). NOT a strict path; its
    logprobs are the "what a native runtime would say" reference for the drift plots."""
    commit = source_commit()
    raw = pathlib.Path(score_file).read_bytes()
    data = json.loads(raw)
    dev = next(hf_model.parameters()).device
    seqs_out = []
    for seq in data["sequences"]:
        ids = torch.tensor(seq["token_ids"], dtype=torch.long, device=dev)
        cs = int(seq["completion_start"])
        out = hf_model(input_ids=ids[None], use_cache=False)
        logits = out.logits[0, cs - 1 : -1].float()
        lp = torch.log_softmax(logits, dim=-1).gather(-1, ids[cs:, None])[:, 0].cpu()
        argmax_hit = (logits.argmax(-1) == ids[cs:]).cpu()
        seqs_out.append({
            "id": seq["id"],
            "logprobs_hex": fp32_hex(lp),
            "n_tokens": int(lp.numel()),
            "completion_start": cs,
            "length": int(ids.numel()),
            "logprobs": [float(x) for x in lp.tolist()],
            "argmax_is_token": [int(x) for x in argmax_hit.tolist()],
        })
        if log:
            log(f"[score:hf] {seq['id']}: {lp.numel()} tokens, sum logp "
                f"{float(lp.double().sum()):.4f}")
        del out, logits
    return {
        "schema": SCHEMA_OUT,
        "runtime": "hf",
        "mode": "native_bf16",
        "tp": 1,
        "backend": f"transformers_{hf_model.config.text_config._attn_implementation}",
        "sequences": seqs_out,
        "source": {"path": str(score_file), "sha256": hashlib.sha256(raw).hexdigest(),
                   "checkpoint_revision": data.get("checkpoint_revision")},
        "commit": commit,
        "device": torch.cuda.get_device_name(dev),
        "torch": torch.__version__,
        "note": "HF MuseGlimmerForConditionalGeneration bf16; logits are HF's bf16 softcapped "
                "logits; logp = fp32 log_softmax. Tolerance anchor only, not a strict path.",
    }


def _main(argv: Sequence[str] | None = None) -> int:
    """``python -m rl_engine.models.muse_glimmer.reference.xrt_scoring --hf ...``: the HF bf16
    anchor for the token file (the strict scoring CLI is ``run_chain.py --score-file``)."""
    import argparse

    p = argparse.ArgumentParser(description=_main.__doc__)
    p.add_argument("--checkpoint", default="~/models/Muse-Glimmer-30B")
    p.add_argument("--score-file", required=True)
    p.add_argument("--score-out", required=True)
    p.add_argument("--attn", default="sdpa")
    args = p.parse_args(argv)
    from transformers import MuseGlimmerForConditionalGeneration

    ckpt = str(pathlib.Path(args.checkpoint).expanduser())
    model = MuseGlimmerForConditionalGeneration.from_pretrained(
        ckpt, dtype=torch.bfloat16, attn_implementation=args.attn
    )
    model.eval().to("cuda")
    out = score_token_file_hf(model, args.score_file, log=print)
    pathlib.Path(args.score_out).write_text(json.dumps(out, indent=1))
    print(f"wrote {args.score_out}")
    return 0


__all__ = ["SCHEMA_IN", "SCHEMA_OUT", "compare_xrt", "fp32_hex", "hex_to_fp32",
           "score_token_file", "score_token_file_hf", "source_commit"]


if __name__ == "__main__":
    raise SystemExit(_main())
