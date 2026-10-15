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
  * example: one token's hex in each file (the first token of the window-crossing sequence where
    the native dumps disagree, if any), for the blog's opening strip.
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
    rt = d.get("runtime", path.stem)
    mode = d.get("mode", "")
    tp = d.get("tp")
    be = d.get("backend")
    parts = [rt, mode]
    if tp:
        parts.append(f"tp{tp}")
    if be and be not in ("", None) and rt == "reference":
        parts.append(be)
    return " ".join(p for p in parts if p)


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
        if lab in files:
            lab = f"{lab} ({p.stem})"
        m = {s["id"]: [h.lower() for h in s["logprobs_hex"]] for s in d["sequences"]}
        for sid, n in want.items():
            if sid not in m:
                problems.append(f"{p.name}: missing sequence {sid}")
            elif len(m[sid]) != n:
                problems.append(f"{p.name}: {sid} has {len(m[sid])} logprobs, expected {n}")
        cat = "".join(h for sid in sorted(m) for h in m[sid])
        files[lab] = {"file": p.name, "runtime": d.get("runtime"), "mode": d.get("mode"), "tp": d.get("tp"),
                      "backend": d.get("backend"), "n_tokens": sum(len(v) for v in m.values()),
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
    strict = [l for l in labels if (files[l]["mode"] or "").startswith("strict") or files[l]["runtime"] == "reference"]
    strict_ok = all(matrix[x][y] == 0 for x in strict for y in strict)
    # example token: first token where any two dumps differ, inside the long sequence if possible
    example = []
    if labels:
        seqs = ["gsm8k_long_window"] + [s for s in want if s != "gsm8k_long_window"]
        pick = None
        for sid in seqs:
            n = want[sid]
            for k in range(n):
                vals = {maps[l][sid][k] for l in labels if sid in maps[l] and k < len(maps[l][sid])}
                if len(vals) > 1:
                    pick = (sid, k)
                    break
            if pick:
                break
        if pick is None:
            pick = ("gsm8k_long_window", 0)
        sid, k = pick
        for l in labels:
            if sid in maps[l] and k < len(maps[l][sid]):
                h = maps[l][sid][k]
                example.append({"label": l, "hex": "0x" + h, "value": f32(h), "strict": l in strict})
        example = sorted(example, key=lambda r: (not r["strict"], r["label"]))
    out = {"schema": "muse_glimmer_xrt_summary_v1", "n_scored_tokens": sum(want.values()), "files": files,
           "strict_labels": strict, "strict_all_equal": strict_ok, "matrix": matrix, "max_abs": max_abs,
           "per_sequence": per_seq, "example_token": {"sequence": pick[0], "index": pick[1]} if labels else None,
           "example": example, "problems": problems}
    (Path(a.out) if a.out else data / "xrt_summary.json").write_text(json.dumps(out, indent=1) + "\n")
    w = max([len(l) for l in labels] + [4])
    print(" " * w, " ".join(f"{i:>5}" for i in range(len(labels))))
    for i, x in enumerate(labels):
        print(f"{x:<{w}}", " ".join(f"{matrix[x][y]:>5}" for y in labels), f"  [{i}]")
    print("strict dumps all byte-equal:", strict_ok, "| problems:", problems or "none")
    return 0 if strict_ok and not problems else 1


if __name__ == "__main__":
    sys.exit(main())
