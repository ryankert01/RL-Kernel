# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Assemble ``reports/muse_glimmer_poc/data/megatron_runtime.json`` from the raw run outputs.

Inputs are the per-run JSON files written by ``run_megatron_strict`` (one per TP size, plus an
optional separate identity run), the TP-gathered fp32 LoRA gradient dumps (``--grads-out``) and
the ``xrt_megatron_*`` log-probability files. Cross-TP comparisons are computed here from raw
bits (no tolerance): per-sequence log-prob mismatch counts and per-tensor gradient mismatch
counts. Run with the megatron (or vllm) env, CPU only::

    python -m rl_engine.models.muse_glimmer.runtime.megatron_results \\
        --run 1=out/run_tp1.json --run 2=out/run_tp2.json --identity 1=out/identity_tp1.json \\
        --grads 1=out/grads_tp1.pt --grads 2=out/grads_tp2.pt --out reports/.../megatron_runtime.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

_REPO = Path(__file__).resolve().parents[4]
_DATA = _REPO / "reports" / "muse_glimmer_poc" / "data"


def _kv(items: list[str]) -> dict[int, Path]:
    out = {}
    for item in items:
        k, v = item.split("=", 1)
        out[int(k)] = Path(v)
    return out


def _hex_mismatch(a: list[str], b: list[str]) -> int:
    if len(a) != len(b):
        return max(len(a), len(b))
    return sum(1 for x, y in zip(a, b, strict=True) if x != y)


def compare_xrt(path_a: Path, path_b: Path) -> dict[str, Any]:
    a = json.loads(path_a.read_text())
    b = json.loads(path_b.read_text())
    per = []
    for sa, sb in zip(a["sequences"], b["sequences"], strict=True):
        if sa["id"] != sb["id"]:
            raise ValueError(f"sequence order differs: {sa['id']} vs {sb['id']}")
        mism = _hex_mismatch(sa["logprobs_hex"], sb["logprobs_hex"])
        la = torch.tensor(sa.get("logprobs", []), dtype=torch.float64)
        lb = torch.tensor(sb.get("logprobs", []), dtype=torch.float64)
        max_abs = float((la - lb).abs().max()) if la.numel() and la.shape == lb.shape else None
        per.append({"id": sa["id"], "n": len(sa["logprobs_hex"]), "bit_mismatches": mism,
                    "max_abs": max_abs})
    return {
        "a": str(path_a.name), "b": str(path_b.name),
        "tokens": sum(p["n"] for p in per),
        "bit_mismatches": sum(p["bit_mismatches"] for p in per),
        "per_sequence": per,
    }


def compare_grads(path_a: Path, path_b: Path) -> dict[str, Any]:
    a = torch.load(path_a, map_location="cpu", weights_only=True)
    b = torch.load(path_b, map_location="cpu", weights_only=True)
    ga, gb = a["grads"], b["grads"]
    if sorted(ga) != sorted(gb):
        raise ValueError("gradient key sets differ")
    per = []
    for key in sorted(ga):
        x, y = ga[key].float().contiguous(), gb[key].float().contiguous()
        if x.shape != y.shape:
            raise ValueError(f"{key}: shape {tuple(x.shape)} vs {tuple(y.shape)}")
        mism = int((x.view(torch.int32) != y.view(torch.int32)).sum())
        per.append({"key": key, "numel": int(x.numel()), "bit_mismatches": mism,
                    "max_abs": float((x.double() - y.double()).abs().max()),
                    "abs_max": float(x.abs().max())})
    lp_mism = [
        int((p.float().view(torch.int32) != q.float().view(torch.int32)).sum())
        for p, q in zip(a["lps"], b["lps"], strict=True)
    ]
    return {
        "tensors": len(per),
        "elements": int(sum(p["numel"] for p in per)),
        "bit_mismatching_elements": int(sum(p["bit_mismatches"] for p in per)),
        "bit_mismatching_tensors": int(sum(1 for p in per if p["bit_mismatches"])),
        "lora_logprob_bit_mismatches_per_sequence": lp_mism,
        "per_tensor": per,
    }


def _slim_identity(rep: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in rep.items()}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", default=[], help="TP=path of a run_megatron_strict JSON")
    ap.add_argument("--identity", action="append", default=[], help="TP=path of a run with identity")
    ap.add_argument("--grads", action="append", default=[], help="TP=path of a --grads-out dump")
    ap.add_argument("--extra", action="append", default=[], help="name=path of extra JSON to embed")
    ap.add_argument("--data-dir", default=str(_DATA))
    ap.add_argument("--out", default=str(_DATA / "megatron_runtime.json"))
    ap.add_argument("--status", default="partial")
    args = ap.parse_args(argv)

    runs = {tp: json.loads(p.read_text()) for tp, p in _kv(args.run).items()}
    idents = {tp: json.loads(p.read_text()) for tp, p in _kv(args.identity).items()}
    grads = _kv(args.grads)
    data_dir = Path(args.data_dir)
    out: dict[str, Any] = {
        "row": "megatron_runtime (Megatron Bridge strict binding + LoRA + weight_sync_identity, StrictTrainer)",
        "status": args.status,
        "commit": None,
        "env": None,
        "commands": [],
        "runs": {},
        "cross_tp": {},
        "xrt_files": {},
    }
    for tp in sorted(runs):
        r = runs[tp]
        out["commit"] = out["commit"] or r.get("commit")
        out["env"] = out["env"] or r.get("env")
        out["commands"].append(r.get("command"))
        rec = {k: v for k, v in r.items() if k not in ("env",)}
        if tp in idents and "identity" in idents[tp].get("checks", {}):
            rec.setdefault("checks", {})["identity"] = _slim_identity(idents[tp]["checks"]["identity"])
            rec["identity_run"] = {"command": idents[tp].get("command"), "commit": idents[tp].get("commit"),
                                   "load": idents[tp].get("load")}
            out["commands"].append(idents[tp].get("command"))
        out["runs"][f"tp{tp}"] = rec
    for mode in ("strict", "native"):
        files = sorted(data_dir.glob(f"xrt_megatron_{mode}_tp*.json"))
        out["xrt_files"][mode] = [f.name for f in files]
        base = data_dir / f"xrt_megatron_{mode}_tp1.json"
        for f in files:
            if f != base and base.exists():
                out["cross_tp"][f"xrt_{mode}_{f.stem.rsplit('_', 1)[-1]}_vs_tp1"] = compare_xrt(base, f)
    s1, n1 = data_dir / "xrt_megatron_strict_tp1.json", data_dir / "xrt_megatron_native_tp1.json"
    if s1.exists() and n1.exists():
        out["cross_tp"]["xrt_native_tp1_vs_strict_tp1"] = compare_xrt(s1, n1)
    if 1 in grads:
        for tp, path in sorted(grads.items()):
            if tp != 1:
                out["cross_tp"][f"lora_grads_tp{tp}_vs_tp1"] = compare_grads(grads[1], path)
    if 1 in runs:
        r1 = runs[1].get("checks", {})
        for tp, r in runs.items():
            if tp == 1:
                continue
            c = r.get("checks", {})
            entry: dict[str, Any] = {}
            if "step" in r1 and "step" in c:
                entry["lora_sha256_after_step_equal"] = (
                    r1["step"]["lora_sha256_after"] == c["step"]["lora_sha256_after"]
                )
                entry["post_step_logprob_bit_mismatches"] = sum(
                    _hex_mismatch(a, b)
                    for a, b in zip(r1["step"]["post_step_logprobs_hex"], c["step"]["post_step_logprobs_hex"], strict=True)
                )
                entry["grad_norm_tp1"] = r1["step"]["info"]["grad_norm"]
                entry[f"grad_norm_tp{tp}"] = c["step"]["info"]["grad_norm"]
                entry["grad_norm_equal"] = entry["grad_norm_tp1"] == entry[f"grad_norm_tp{tp}"]
            if "grad_repeat" in r1 and "grad_repeat" in c:
                entry["lora_logprob_bit_mismatches"] = sum(
                    _hex_mismatch(a, b)
                    for a, b in zip(r1["grad_repeat"]["lora_logprobs_hex"], c["grad_repeat"]["lora_logprobs_hex"], strict=True)
                )
                entry["grad_aggregate_sha256_equal"] = (
                    r1["grad_repeat"]["aggregate_sha256"] == c["grad_repeat"]["aggregate_sha256"]
                )
            out["cross_tp"][f"step_tp{tp}_vs_tp1"] = entry
    for item in args.extra:
        name, path = item.split("=", 1)
        out[name] = json.loads(Path(path).read_text())
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")
    for k, v in out["cross_tp"].items():
        brief = {kk: vv for kk, vv in v.items() if not isinstance(vv, list)}
        print(k, brief)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
