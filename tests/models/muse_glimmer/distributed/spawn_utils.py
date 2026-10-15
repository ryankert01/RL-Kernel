# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Shared multi-GPU test plumbing for the Muse-Glimmer WS2 rows.

Pattern copied from ``tests/distributed/tp/test_vocab_parallel_logp.py``: one spawned process
per rank, NCCL initialised from a file in a temporary directory, results handed back through a
queue. TP8 scenarios are present but skipped unless ``MUSEGLIMMER_TP8=1`` (the integrator runs
them on the idle node).
"""

from __future__ import annotations

import hashlib
import os
import queue
import tempfile
import traceback
from pathlib import Path
from typing import Any, Callable

import pytest
import torch
import torch.multiprocessing as mp

SPAWN_TIMEOUT_S = 900
TP8_ENV = "MUSEGLIMMER_TP8"


def cuda_device_count() -> int:
    return torch.cuda.device_count() if torch.cuda.is_available() else 0


def requires_gpus(count: int):
    marks = [
        pytest.mark.skipif(
            cuda_device_count() < count,
            reason=f"needs {count} CUDA devices (one rank per device)",
        )
    ]
    if count >= 8:
        marks.append(
            pytest.mark.skipif(
                os.environ.get(TP8_ENV) != "1",
                reason=f"TP8 run is opt-in: set {TP8_ENV}=1 on the idle node",
            )
        )

    def apply(fn):
        for mark in marks:
            fn = mark(fn)
        return fn

    return apply


def disable_tf32() -> None:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def sha256_of(tensor: torch.Tensor) -> str:
    raw = tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _rank_main(
    rank: int,
    world_size: int,
    init_method: str,
    result_queue: Any,
    worker: Callable[..., dict[str, Any]],
    args: tuple[Any, ...],
) -> None:
    import torch.distributed as dist

    try:
        disable_tf32()
        torch.cuda.set_device(rank)
        dist.init_process_group(
            backend="nccl", init_method=init_method, rank=rank, world_size=world_size
        )
        result = worker(rank, world_size, *args)
        result = dict(result)
        result.update({"ok": True, "rank": rank})
        result_queue.put(result)
    except Exception:  # pragma: no cover - forwarded to the parent process
        result_queue.put({"ok": False, "rank": rank, "traceback": traceback.format_exc()})
        raise
    finally:
        import torch.distributed as dist

        if dist.is_initialized():
            dist.destroy_process_group()


def run_spawned(
    world_size: int, worker: Callable[..., dict[str, Any]], *args: Any
) -> list[dict[str, Any]]:
    """Run ``worker(rank, world_size, *args) -> dict`` on ``world_size`` NCCL ranks."""

    ctx = mp.get_context("spawn")
    with tempfile.TemporaryDirectory() as tmpdir:
        init_method = (Path(tmpdir) / "nccl_init").as_uri()
        result_queue = ctx.Queue()
        processes = [
            ctx.Process(
                target=_rank_main,
                args=(rank, world_size, init_method, result_queue, worker, args),
            )
            for rank in range(world_size)
        ]
        results: list[dict[str, Any]] = []
        try:
            for process in processes:
                process.start()
            for _ in range(world_size):
                try:
                    results.append(result_queue.get(timeout=SPAWN_TIMEOUT_S))
                except queue.Empty:
                    for process in processes:
                        if process.is_alive():
                            process.terminate()
                    pytest.fail(f"timed out waiting for {world_size} NCCL workers")
        finally:
            for process in processes:
                process.join(timeout=60)
                if process.is_alive():
                    process.terminate()
    results.sort(key=lambda item: item["rank"])
    for result in results:
        assert result["ok"], result.get("traceback")
    for process in processes:
        assert process.exitcode == 0
    return results


__all__ = [
    "SPAWN_TIMEOUT_S",
    "TP8_ENV",
    "cuda_device_count",
    "disable_tf32",
    "requires_gpus",
    "run_spawned",
    "sha256_of",
]
