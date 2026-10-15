# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""torchrun parity matrix for the Muse-Glimmer CP attention rows (RFC #501 WS2).

    torchrun --nproc_per_node=<CP> rl_engine/models/muse_glimmer/runtime/check_cp_parity.py \\
        --lengths 2048 4096 8192 --layers global local [--backward] [--json out.json]

Every rank runs the CP1 reference over the full logical sequence and the CP-<world> path on its
contiguous chunk, then reports bitwise mismatch counts for y/o/lse (and dq/dk/dv/dgate with
``--backward``), the forward time, and the cost of the K/V + positions all_gather at the largest
length. Rank 0 prints one markdown table (and optionally writes JSON). ``--nproc_per_node=1``
runs the CP1-vs-CP1 repeat check only.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

import torch
import torch.distributed as dist

# torchrun puts the script directory, not the repo root, on sys.path, and the shared POC
# environment may carry an editable install of ANOTHER checkout: pin this checkout first.
_REPO_ROOT = str(Path(__file__).resolve().parents[4])
if sys.path[:1] != [_REPO_ROOT]:
    sys.path.insert(0, _REPO_ROOT)

from rl_engine.models.muse_glimmer.runtime import cp_attention as cpa  # noqa: E402
from rl_engine.models.muse_glimmer.triton import gated_attention_bwd as _bwd  # noqa: E402

_HQ, _HKV, _D = 32, 2, 128


def _inputs(total: int, seed: int, device: torch.device) -> dict[str, torch.Tensor]:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    to = {"device": device, "dtype": torch.bfloat16}
    return {
        "q": torch.randn(total, _HQ, _D, generator=gen).to(**to),
        "k": torch.randn(total, _HKV, _D, generator=gen).to(**to),
        "v": torch.randn(total, _HKV, _D, generator=gen).to(**to),
        "gate": torch.randn(total, _HQ * _D, generator=gen).to(**to),
        "dy": torch.randn(total, _HQ * _D, generator=gen).to(**to),
        "positions": torch.arange(total, dtype=torch.int32, device=device),
    }


def _mismatch(got: torch.Tensor, want: torch.Tensor) -> tuple[int, float]:
    return int((got != want).sum().item()), float((got.float() - want.float()).abs().max())


def _time_ms(fn, *, warmup: int = 3, iters: int = 10) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def _run(args: argparse.Namespace) -> list[dict[str, Any]]:
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if world > 1:
        dist.init_process_group(backend="nccl")
    collective = cpa.make_cp_collective(None, device) if world > 1 else None
    plan = cpa.CPAttentionPlan(world, rank)
    records: list[dict[str, Any]] = []
    largest = max(args.lengths)
    for layer in args.layers:
        window: Optional[int] = None if layer == "global" else cpa.SLIDING_WINDOW
        for total in args.lengths:
            inp = _inputs(total, seed=total, device=device)
            rows = slice(*plan.local_token_range(total))
            ref_y, ref_o, ref_lse = cpa.cp_gated_attention_forward(
                inp["q"], inp["k"], inp["v"], inp["gate"], inp["positions"],
                cpa.CPAttentionPlan(1, 0), None, window=window, return_aux=True,
            )
            local = {n: inp[n][rows] for n in ("q", "k", "v", "gate", "positions")}

            def forward():
                return cpa.cp_gated_attention_forward(
                    local["q"], local["k"], local["v"], local["gate"], local["positions"],
                    plan, collective, window=window, return_aux=True,
                )

            y, o, lse = forward()
            y2, o2, lse2 = forward()
            rec: dict[str, Any] = {
                "layer": layer,
                "window": window,
                "T": total,
                "rank": rank,
                "cp": world,
                "y_mismatch": _mismatch(y, ref_y[rows])[0],
                "o_mismatch": _mismatch(o, ref_o[rows])[0],
                "lse_mismatch": _mismatch(lse, ref_lse[rows])[0],
                "max_abs_diff_y": _mismatch(y, ref_y[rows])[1],
                "repeat_equal": bool(
                    torch.equal(y, y2) and torch.equal(o, o2) and torch.equal(lse, lse2)
                ),
                "cp_forward_ms": _time_ms(forward, warmup=1, iters=3),
            }
            if total == largest:
                if collective is not None:
                    k_local, v_local = local["k"].contiguous(), local["v"].contiguous()
                    pos_local = local["positions"].contiguous()
                    rec["ag_kv_ms"] = _time_ms(
                        lambda: collective.all_gather_many((k_local, v_local))
                    )
                    rec["ag_positions_ms"] = _time_ms(lambda: collective.all_gather(pos_local))
                    rec["ag_kv_bytes_per_rank"] = 2 * k_local.numel() * k_local.element_size()
                rec["kernel_rows_ms"] = _time_ms(
                    lambda: cpa.gated_attention_rows(
                        local["q"], inp["k"], inp["v"], local["gate"], local["positions"],
                        window=window,
                    ),
                    warmup=1,
                    iters=3,
                )
            if args.backward:
                leaves = {n: inp[n].clone().requires_grad_(True) for n in ("q", "k", "v", "gate")}
                cpa.cp_gated_attention(
                    leaves["q"], leaves["k"], leaves["v"], leaves["gate"], inp["positions"],
                    cpa.CPAttentionPlan(1, 0), None, window=window,
                ).backward(inp["dy"])
                loc = {
                    n: inp[n][rows].clone().requires_grad_(True) for n in ("q", "k", "v", "gate")
                }

                def fwd_bwd():
                    for leaf in loc.values():
                        leaf.grad = None
                    cpa.cp_gated_attention(
                        loc["q"], loc["k"], loc["v"], loc["gate"], inp["positions"][rows],
                        plan, collective, window=window,
                    ).backward(inp["dy"][rows])

                rec["cp_fwd_bwd_ms"] = _time_ms(fwd_bwd, warmup=1, iters=2)
                # the one-shot WS1 kernel on the full sequence (query-chunk rule, no CP plumbing)
                cu = torch.tensor([0, total], dtype=torch.int32, device=device)
                one_shot = _bwd.strict_gated_attention_bwd(
                    inp["q"], inp["k"], inp["v"], inp["gate"], ref_o, ref_lse, inp["dy"], cu, cu,
                    inp["positions"], window=window,
                )
                for name, kernel_grad in zip(("q", "k", "v", "gate"), one_shot):
                    rec[f"d{name}_mismatch"] = _mismatch(loc[name].grad, leaves[name].grad[rows])[0]
                    rec[f"d{name}_vs_one_shot_mismatch"] = _mismatch(
                        loc[name].grad, kernel_grad[rows]
                    )[0]
            records.append(rec)
    if world > 1:
        gathered: list[Any] = [None] * world
        dist.all_gather_object(gathered, records)
        records = [r for part in gathered for r in part]
        dist.barrier()
        dist.destroy_process_group()
    return records if rank == 0 else []


def _table(records: list[dict[str, Any]], backward: bool) -> str:
    head = ["layer", "T", "cp", "rank", "y!=", "o!=", "lse!=", "repeat", "fwd ms", "AG K/V ms"]
    if backward:
        head += ["dq!=", "dk!=", "dv!=", "dgate!=", "vs one-shot dk!=/dv!=", "fwd+bwd ms"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in sorted(records, key=lambda r: (r["layer"], r["T"], r["rank"])):
        row = [
            r["layer"], str(r["T"]), str(r["cp"]), str(r["rank"]), str(r["y_mismatch"]),
            str(r["o_mismatch"]), str(r["lse_mismatch"]), str(r["repeat_equal"]),
            f"{r['cp_forward_ms']:.1f}",
            f"{r['ag_kv_ms']:.3f}" if "ag_kv_ms" in r else "-",
        ]
        if backward:
            row += [
                str(r["dq_mismatch"]), str(r["dk_mismatch"]), str(r["dv_mismatch"]),
                str(r["dgate_mismatch"]),
                f"{r['dk_vs_one_shot_mismatch']}/{r['dv_vs_one_shot_mismatch']}",
                f"{r['cp_fwd_bwd_ms']:.0f}",
            ]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--lengths", type=int, nargs="+", default=[2048, 4096, 8192])
    parser.add_argument(
        "--layers", nargs="+", choices=["global", "local"], default=["global", "local"]
    )
    parser.add_argument("--backward", action="store_true", help="also compare dq/dk/dv/dgate")
    parser.add_argument("--json", type=str, default=None, help="write the records to this file")
    args = parser.parse_args()
    started = time.perf_counter()
    records = _run(args)
    if records:
        print(_table(records, args.backward))
        all_ok = all(
            r["y_mismatch"] == 0
            and r["o_mismatch"] == 0
            and r["lse_mismatch"] == 0
            and r["repeat_equal"]
            for r in records
        )
        if args.backward:
            all_ok = all_ok and all(
                r[f"d{n}_mismatch"] == 0 and r[f"d{n}_vs_one_shot_mismatch"] == 0
                for r in records
                for n in ("q", "k", "v", "gate")
            )
        print(f"\ntrace: {json.dumps(cpa.trace(), sort_keys=True)}")
        print(f"all bitwise: {all_ok}  ({time.perf_counter() - started:.1f}s)")
        if args.json:
            payload = {"records": records, "trace": cpa.trace()}
            Path(args.json).write_text(json.dumps(payload, indent=1))
        return 0 if all_ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
