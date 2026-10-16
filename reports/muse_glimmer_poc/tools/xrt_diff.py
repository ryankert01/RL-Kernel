#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Byte-diff every cross-runtime logprob dump against every other one.

Inputs: reports/muse_glimmer_poc/data/xrt_tokens.json (muse_glimmer_xrt_v1) and every
reports/muse_glimmer_poc/data/xrt_*.json in muse_glimmer_xrt_logprobs_v1 format.
Output: reports/muse_glimmer_poc/data/xrt_summary.json with
  * files: label -> {file, runtime, mode, tp, backend, n_tokens, sha256 of the concatenated hex}
  * matrix: mismatching-token counts for every ordered pair (symmetric)
  * per_sequence: for every pair with mismatches, counts per sequence and the first mismatch
  * max_abs: max |a - b| over tokens for every pair (fp32 values decoded from the hex)
  * native_base: the native dumps that use the base weights (no LoRA), HF bf16 included; pairwise
    ranges of native drift are taken over these only (a LoRA dump is a different model)
  * example: one token's hex in each file, for the blog's opening strip: the first token (window-crossing
    sequence first) where every base strict dump agrees and the base native dumps all disagree with each
    other, so the strip shows one strict result against several native ones.
Exit code 1 if any two STRICT dumps differ (the POC's pass criterion), else 0.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"


def f32(h: str) -> float:
    return struct.unpack(">f", bytes.fromhex(h))[0]


def label_of(d: dict, path: Path) -> str:
    """Stable, unique label from the file name: xrt_<runtime>_<mode>[_lora]_tp<k>.json -> '<runtime> <mode>[ lora] tp<k>'."""
    return path.stem.removeprefix("xrt_").replace("_", " ")


def group_of(d: dict, path: Path) -> str | None:
    """Which strict group a dump belongs to (None = not a strict dump). Strict dumps of one group must be byte-equal.

    The golden reference backend is a fixed-order fp32 *accuracy* reference, not the bitwise target; the strict target is
    the Triton strict path, which the reference chain runs as backend 'triton'."""
    stem = path.stem
    lora = d.get("lora_sha256") or "_lora_" in stem
    if d.get("runtime") == "reference":
        return "base" if d.get("backend") == "triton" else None
    if d.get("mode") == "strict":
        return f"lora:{d.get('lora_sha256')}" if lora else "base"
    return None


def is_native_base(d: dict, path: Path) -> bool:
    """A native (non-strict) dump on the base weights: vLLM / Megatron native or HF bf16, no LoRA loaded."""
    return str(d.get("mode", "")).startswith("native") and not d.get("lora_sha256") and "_lora_" not in path.stem


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DATA))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    data = Path(a.data)
    tokens = json.loads((data / "xrt_tokens.json").read_text())
    want = {s["id"]: len(s["token_ids"]) - s["completion_start"] for s in tokens["sequences"]}
    files: dict[str, dict] = {}
    maps: dict[str, dict[str, list[str]]] = {}
    problems: list[str] = []
    for p in sorted(data.glob("xrt_*.json")):
        if p.name in ("xrt_tokens.json", "xrt_summary.json"):
            continue
        d = json.loads(p.read_text())
        if d.get("schema") != "muse_glimmer_xrt_logprobs_v1":
            problems.append(f"{p.name}: schema {d.get('schema')!r}")
            continue
        lab = label_of(d, p)
        m = {s["id"]: [h.lower() for h in s["logprobs_hex"]] for s in d["sequences"]}
        for sid, n in want.items():
            if sid not in m:
                problems.append(f"{p.name}: missing sequence {sid}")
            elif len(m[sid]) != n:
                problems.append(f"{p.name}: {sid} has {len(m[sid])} logprobs, expected {n}")
        cat = "".join(h for sid in sorted(m) for h in m[sid])
        files[lab] = {"file": p.name, "runtime": d.get("runtime"), "mode": d.get("mode"), "tp": d.get("tp"),
                      "backend": d.get("backend"), "strict_group": group_of(d, p), "native_base": is_native_base(d, p),
                      "lora_sha256": d.get("lora_sha256"), "n_tokens": sum(len(v) for v in m.values()),
                      "sha256": hashlib.sha256(cat.encode()).hexdigest()}
        maps[lab] = m
    labels = list(files)
    matrix, per_seq, max_abs = {}, {}, {}
    for x in labels:
        matrix[x], max_abs[x] = {}, {}
        for y in labels:
            n, worst, detail = 0, 0.0, {}
            for sid in want:
                hx, hy = maps[x].get(sid, []), maps[y].get(sid, [])
                diff = [k for k, (u, v) in enumerate(zip(hx, hy)) if u != v] + list(range(min(len(hx), len(hy)), max(len(hx), len(hy))))
                if diff:
                    detail[sid] = {"mismatches": len(diff), "first": diff[0], "of": max(len(hx), len(hy))}
                n += len(diff)
                for u, v in zip(hx, hy):
                    if u != v:
                        worst = max(worst, abs(f32(u) - f32(v)))
            matrix[x][y] = n
            max_abs[x][y] = worst
            if detail and x < y:
                per_seq[f"{x} | {y}"] = detail
    groups: dict[str, list[str]] = {}
    for l in labels:
        if files[l]["strict_group"]:
            groups.setdefault(files[l]["strict_group"], []).append(l)
    strict = [l for g in groups.values() for l in g]
    strict_ok = all(matrix[x][y] == 0 for g in groups.values() for x in g for y in g)
    native_base = [l for l in labels if files[l]["native_base"]]
    # example token: every base strict dump agrees and the base native dumps all disagree with each other
    # (window-crossing sequence first); fall back to the token with the most distinct native values
    example = []
    if labels:
        seqs = ["gsm8k_long_window"] + [s for s in want if s != "gsm8k_long_window"]
        base = groups.get("base", [])
        pick, best = None, (-1, None)
        for sid in seqs:
            for k in range(want[sid]):
                sv = {maps[l][sid][k] for l in base if sid in maps[l] and k < len(maps[l][sid])}
                nv = [maps[l][sid][k] for l in native_base if sid in maps[l] and k < len(maps[l][sid])]
                if len(sv) != 1 or not nv or sv & set(nv):
                    continue
                if len(set(nv)) == len(nv):
                    pick = (sid, k)
                    break
                if len(set(nv)) > best[0]:
                    best = (len(set(nv)), (sid, k))
            if pick:
                break
        if pick is None:
            pick = best[1] or ("gsm8k_long_window", 0)
        sid, k = pick
        for l in labels:
            if sid in maps[l] and k < len(maps[l][sid]):
                h = maps[l][sid][k]
                example.append({"label": l, "hex": "0x" + h, "value": f32(h), "strict": l in strict, "native_base": files[l]["native_base"]})
        example = sorted(example, key=lambda r: (not r["strict"], r["label"]))
    out = {"schema": "muse_glimmer_xrt_summary_v1", "n_scored_tokens": sum(want.values()), "files": files,
           "strict_labels": strict, "strict_groups": groups, "strict_all_equal": strict_ok, "native_base": native_base,
           "matrix": matrix, "max_abs": max_abs,
           "per_sequence": per_seq, "example_token": {"sequence": pick[0], "index": pick[1]} if labels else None,
           "example": example, "problems": problems}
    (Path(a.out) if a.out else data / "xrt_summary.json").write_text(json.dumps(out, indent=1) + "\n")
    w = max([len(l) for l in labels] + [4])
    print(" " * w, " ".join(f"{i:>5}" for i in range(len(labels))))
    for i, x in enumerate(labels):
        print(f"{x:<{w}}", " ".join(f"{matrix[x][y]:>5}" for y in labels), f"  [{i}]")
    for g, ls in groups.items():
        print(f"strict group {g[:20]}: {len(ls)} dumps, byte-equal: {all(matrix[x][y] == 0 for x in ls for y in ls)}")
    print("strict dumps all byte-equal within their group:", strict_ok, "| problems:", problems or "none")
    return 0 if strict_ok and not problems else 1


if __name__ == "__main__":
    sys.exit(main())
