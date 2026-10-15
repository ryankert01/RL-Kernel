# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Shared process-spawn harness for the Muse-Glimmer CP and TP x CP parity tests.

Pattern (as in ``tests/distributed/cp``): ``torch.multiprocessing.spawn`` one process per GPU,
NCCL initialised through a file store in the test's temporary directory, every rank writes a
JSON result record that the parent asserts on. Tests skip when fewer GPUs are visible than the
topology needs or when an external launcher already set ``WORLD_SIZE``.

Acceptance everywhere is bitwise (``torch.equal``) against a CP1 run of the same strict core on
the same logical sequence: CP performs no arithmetic of its own on the allgather layout, so any
difference means a second reduction order crept in.
"""

from __future__ import annotations

import json
import os
import time
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from rl_engine.models.muse_glimmer.runtime import cp_attention as cpa

LENGTHS = (2048, 4096, 8192)
#: backward lengths add T % 8 != 0 (2100: canonical chunks of 262/263 rows, CP2/CP4 chunks of
#: 1050/525) to exercise the query-chunk tail rule.
BACKWARD_LENGTHS = (2048, 2100, 4096, 8192)
WINDOW = 2048
NUM_Q_HEADS = 32
NUM_KV_HEADS = 2
HEAD_DIM = 128
EXTERNAL_WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))


def visible_gpus() -> int:
    return torch.cuda.device_count() if torch.cuda.is_available() else 0


def disable_tf32() -> None:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def make_global_inputs(
    total: int,
    seed: int,
    device: torch.device,
    *,
    q_heads: int = NUM_Q_HEADS,
    kv_heads: int = NUM_KV_HEADS,
) -> dict[str, torch.Tensor]:
    """Identical full-sequence Q/K/V/gate/dy on every rank: generated on CPU from fixed seeds."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.randn(total, q_heads, HEAD_DIM, generator=gen, dtype=torch.float32)
    k = torch.randn(total, kv_heads, HEAD_DIM, generator=gen, dtype=torch.float32)
    v = torch.randn(total, kv_heads, HEAD_DIM, generator=gen, dtype=torch.float32)
    gate = torch.randn(total, q_heads * HEAD_DIM, generator=gen, dtype=torch.float32)
    dy = torch.randn(total, q_heads * HEAD_DIM, generator=gen, dtype=torch.float32)
    to = {"device": device, "dtype": torch.bfloat16}
    return {
        "q": q.to(**to),
        "k": k.to(**to),
        "v": v.to(**to),
        "gate": gate.to(**to),
        "dy": dy.to(**to),
        "positions": torch.arange(total, dtype=torch.int32, device=device),
    }


def naive_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    q_pos: torch.Tensor,
    *,
    window: Optional[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Plain fp32 softmax attention with the O5 mask on absolute positions (tolerance oracle)."""
    tq, hq, d = q.shape
    group = hq // k.shape[1]
    kf = k.float().repeat_interleave(group, dim=1)
    vf = v.float().repeat_interleave(group, dim=1)
    scores = torch.einsum("qhd,khd->hqk", q.float(), kf) * d**-0.5
    k_pos = torch.arange(k.shape[0], device=q.device)
    keep = k_pos[None, :] <= q_pos[:, None]
    if window is not None:
        keep &= k_pos[None, :] > q_pos[:, None] - window
    scores = scores.masked_fill(~keep[None], float("-inf"))
    p = torch.softmax(scores, dim=-1)
    o = torch.einsum("hqk,khd->qhd", p, vf)
    y = o.to(torch.bfloat16).float() * torch.sigmoid(gate.float().view(tq, hq, d))
    return y.reshape(tq, hq * d), o.reshape(tq, hq * d), torch.logsumexp(scores, -1).t()


def _bits(t: torch.Tensor) -> torch.Tensor:
    """Integer view of a float tensor: compares bit patterns (``-0.0 != +0.0``, NaN == NaN)."""
    if not t.is_floating_point():
        return t
    return t.contiguous().view({2: torch.int16, 4: torch.int32, 8: torch.int64}[t.element_size()])


def compare(name: str, got: torch.Tensor, want: torch.Tensor) -> dict[str, Any]:
    """Bitwise comparison record (bit-pattern mismatch count, max abs diff) for one tensor."""
    if got.shape != want.shape or got.dtype != want.dtype:
        return {
            "name": name,
            "equal": False,
            "mismatches": -1,
            "max_abs_diff": float("nan"),
            "error": (
                f"shape/dtype {tuple(got.shape)}/{got.dtype} vs "
                f"{tuple(want.shape)}/{want.dtype}"
            ),
        }
    diff = (got.float() - want.float()).abs()
    mismatches = int((_bits(got) != _bits(want)).sum().item())
    return {
        "name": name,
        "equal": mismatches == 0,
        "mismatches": mismatches,
        "max_abs_diff": float(diff.max().item()) if diff.numel() else 0.0,
    }


def init_rank(rank: int, world: int, init_file: str) -> torch.device:
    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="nccl",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world,
        timeout=timedelta(minutes=10),
    )
    disable_tf32()
    return torch.device("cuda", rank)


def finalize_rank() -> None:
    dist.barrier()
    dist.destroy_process_group()


def write_result(result_dir: str, rank: int, record: dict[str, Any]) -> None:
    Path(result_dir, f"rank{rank}.json").write_text(json.dumps(record, indent=1))


def spawn_and_collect(
    world: int,
    worker: Callable[..., None],
    tmp_path: Path,
    *args: Any,
) -> list[dict[str, Any]]:
    """Run ``worker(rank, world, init_file, result_dir, *args)`` on ``world`` GPUs."""
    result_dir = tmp_path / "results"
    result_dir.mkdir(parents=True, exist_ok=True)
    init_file = tmp_path / "nccl_init"
    mp.spawn(
        worker,
        args=(world, str(init_file), str(result_dir), *args),
        nprocs=world,
        join=True,
    )
    records = []
    for rank in range(world):
        path = result_dir / f"rank{rank}.json"
        assert path.exists(), f"rank {rank} wrote no result record"
        records.append(json.loads(path.read_text()))
    return records


def assert_all_equal(records: list[dict[str, Any]]) -> None:
    """Every comparison record of every rank must be bitwise equal."""
    failures = []
    for record in records:
        for check in record["checks"]:
            if not check["equal"]:
                failures.append(f"rank {record['rank']}: {check}")
    assert not failures, "bitwise mismatches:\n" + "\n".join(failures)


def _sync_ms(fn: Callable[[], Any]) -> float:
    torch.cuda.synchronize()
    start = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1e3


# --------------------------------------------------------------------------------------------
# Workers (module level so ``mp.spawn`` can pickle them)
# --------------------------------------------------------------------------------------------
def forward_parity_worker(
    rank: int,
    world: int,
    init_file: str,
    result_dir: str,
    lengths: Sequence[int],
    window: Optional[int],
) -> None:
    """CP-``world`` forward vs CP1 on each length; bitwise y/o/lse, repeat, and a teeth check."""
    device = init_rank(rank, world, init_file)
    try:
        collective = cpa.make_cp_collective(None, device)
        plan = cpa.CPAttentionPlan(world, rank)
        checks: list[dict[str, Any]] = []
        timings: dict[str, float] = {}
        teeth: dict[str, bool] = {}
        for total in lengths:
            inp = make_global_inputs(total, seed=total, device=device)
            ref_y, ref_o, ref_lse = cpa.cp_gated_attention_forward(
                inp["q"], inp["k"], inp["v"], inp["gate"], inp["positions"],
                cpa.CPAttentionPlan(1, 0), None, window=window, return_aux=True,
            )
            rows = slice(*plan.local_token_range(total))
            local = {name: inp[name][rows] for name in ("q", "k", "v", "gate", "positions")}
            run = lambda: cpa.cp_gated_attention_forward(  # noqa: E731
                local["q"], local["k"], local["v"], local["gate"], local["positions"],
                plan, collective, window=window, return_aux=True,
            )
            y, o, lse = run()
            timings[f"T={total} cp_forward_ms"] = _sync_ms(run)
            y2, o2, lse2 = run()
            checks += [
                compare(f"T={total} y", y, ref_y[rows]),
                compare(f"T={total} o", o, ref_o[rows]),
                compare(f"T={total} lse", lse, ref_lse[rows]),
                compare(f"T={total} repeat y", y2, y),
                compare(f"T={total} repeat o", o2, o),
                compare(f"T={total} repeat lse", lse2, lse),
            ]
            if rank > 0:
                # Teeth: a CP-oblivious kernel call with RELATIVE positions must NOT match.
                relative = local["positions"] - int(plan.local_token_range(total)[0])
                y_rel, _, _ = cpa.gated_attention_rows(
                    local["q"], inp["k"], inp["v"], local["gate"], relative, window=window
                )
                teeth[f"T={total}"] = not torch.equal(y_rel, ref_y[rows])
        write_result(
            result_dir,
            rank,
            {
                "rank": rank,
                "world": world,
                "window": window,
                "checks": checks,
                "timings_ms": timings,
                "teeth_relative_positions_differ": teeth,
                "trace": cpa.trace(plan),
            },
        )
    finally:
        finalize_rank()


def backward_parity_worker(
    rank: int,
    world: int,
    init_file: str,
    result_dir: str,
    lengths: Sequence[int],
    window: Optional[int],
) -> None:
    """CP-``world`` autograd backward vs CP1 AND vs the one-shot WS1 kernel
    (``triton.gated_attention_bwd.strict_gated_attention_bwd``): bitwise dq/dk/dv/dgate rows."""
    from rl_engine.models.muse_glimmer.triton import gated_attention_bwd as bwd

    device = init_rank(rank, world, init_file)
    try:
        collective = cpa.make_cp_collective(None, device)
        plan = cpa.CPAttentionPlan(world, rank)
        checks: list[dict[str, Any]] = []
        timings: dict[str, float] = {}
        for total in lengths:
            inp = make_global_inputs(total, seed=total + 7, device=device)
            leaves = {n: inp[n].clone().requires_grad_(True) for n in ("q", "k", "v", "gate")}
            y_ref = cpa.cp_gated_attention(
                leaves["q"], leaves["k"], leaves["v"], leaves["gate"], inp["positions"],
                cpa.CPAttentionPlan(1, 0), None, window=window,
            )
            y_ref.backward(inp["dy"])
            rows = slice(*plan.local_token_range(total))
            # one-shot kernel on the full sequence (single launch, TP1, no CP plumbing)
            cu = torch.tensor([0, total], dtype=torch.int32, device=device)
            _y, o_full, lse_full = cpa.gated_attention_rows(
                inp["q"], inp["k"], inp["v"], inp["gate"], inp["positions"], window=window
            )
            one_shot = bwd.strict_gated_attention_bwd(
                inp["q"], inp["k"], inp["v"], inp["gate"], o_full, lse_full, inp["dy"], cu, cu,
                inp["positions"], window=window,
            )
            for name, kernel_grad in zip(("q", "k", "v", "gate"), one_shot):
                checks.append(
                    compare(f"T={total} CP1 d{name} vs one-shot kernel", leaves[name].grad,
                            kernel_grad)
                )
            local = {n: inp[n][rows].clone().requires_grad_(True) for n in ("q", "k", "v", "gate")}

            def run() -> None:
                for leaf in local.values():
                    leaf.grad = None
                y = cpa.cp_gated_attention(
                    local["q"], local["k"], local["v"], local["gate"], inp["positions"][rows],
                    plan, collective, window=window,
                )
                y.backward(inp["dy"][rows])

            timings[f"T={total} cp_fwd_bwd_ms"] = _sync_ms(run)
            first = {n: t.grad.clone() for n, t in local.items()}
            run()
            for name, kernel_grad in zip(("q", "k", "v", "gate"), one_shot):
                checks.append(compare(f"T={total} d{name}", first[name], leaves[name].grad[rows]))
                checks.append(compare(f"T={total} repeat d{name}", local[name].grad, first[name]))
                checks.append(
                    compare(f"T={total} CP{world} d{name} vs one-shot kernel", first[name],
                            kernel_grad[rows])
                )
        write_result(
            result_dir,
            rank,
            {
                "rank": rank,
                "world": world,
                "window": window,
                "checks": checks,
                "timings_ms": timings,
                "trace": cpa.trace(plan),
            },
        )
    finally:
        finalize_rank()


def tpcp_parity_worker(
    rank: int,
    world: int,
    init_file: str,
    result_dir: str,
    lengths: Sequence[int],
    windows: Sequence[Optional[int]],
    tp_world: int,
    cp_world: int,
) -> None:
    """TP x CP forward on ``world = tp*cp`` GPUs vs the single-GPU TP1 x CP1 result."""
    device = init_rank(rank, world, init_file)
    try:
        topology = cpa.TPCPTopology.from_global_rank(rank, tp_world=tp_world, cp_world=cp_world)
        groups = {}
        for tp_rank in range(tp_world):  # every rank creates every CP group, same order
            ranks = cpa.TPCPTopology.cp_group_ranks_for(
                tp_rank, tp_world=tp_world, cp_world=cp_world
            )
            groups[tp_rank] = dist.new_group(ranks=list(ranks))
        collective = cpa.make_cp_collective(groups[topology.tp_rank], device)
        checks: list[dict[str, Any]] = []
        timings: dict[str, float] = {}
        q_heads = slice(*topology.q_head_range())
        kv_heads = slice(*topology.kv_head_range())
        channels = slice(*topology.gate_channel_range())
        for total in lengths:
            rows = slice(*topology.plan.local_token_range(total))
            for window in windows:
                inp = make_global_inputs(total, seed=total + 11, device=device)
                ref_y, ref_o, ref_lse = cpa.cp_gated_attention_forward(
                    inp["q"], inp["k"], inp["v"], inp["gate"], inp["positions"],
                    cpa.CPAttentionPlan(1, 0), None, window=window, return_aux=True,
                )
                run = lambda: cpa.cp_tp_gated_attention_forward(  # noqa: E731
                    inp["q"][rows, q_heads], inp["k"][rows, kv_heads], inp["v"][rows, kv_heads],
                    inp["gate"][rows, channels], inp["positions"][rows],
                    topology, collective, window=window, return_aux=True,
                )
                y, o, lse = run()
                tag = f"T={total} window={window}"
                timings[f"{tag} tpcp_forward_ms"] = _sync_ms(run)
                checks += [
                    compare(f"{tag} y", y, ref_y[rows, channels]),
                    compare(f"{tag} o", o, ref_o[rows, channels]),
                    compare(f"{tag} lse", lse, ref_lse[rows, q_heads]),
                ]
        write_result(
            result_dir,
            rank,
            {
                "rank": rank,
                "world": world,
                "topology": topology.provenance(),
                "checks": checks,
                "timings_ms": timings,
            },
        )
    finally:
        finalize_rank()
