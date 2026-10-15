# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Worker-side measurement helpers for the Muse Glimmer vLLM 0.31 runs (``LLM.collective_rpc``).

Every function takes the vLLM ``Worker`` as its first argument and returns plain Python data
(dicts/lists of numbers and strings), so it travels back through vLLM's msgpack utility channel.
They work for both the strict binding and plain native vLLM (nothing here touches the strict
state).  Used by ``run_vllm_strict.py`` for the throughput / memory / profiler records.

* :func:`worker_memory` — peak / current allocated and reserved CUDA memory, KV-cache blocks and
  bytes as allocated by the model runner.
* :func:`worker_step_timer` — wraps ``Worker.execute_model`` / ``Worker.sample_tokens`` on the
  instance and records one entry per engine step: scheduled tokens, requests, and the wall time
  from the start of ``execute_model`` to the end of the matching ``sample_tokens`` after a
  ``torch.cuda.synchronize()`` (so the number is GPU-complete).  The synchronize serializes the
  step with the scheduler; it is a measurement mode, not the production configuration.
* :func:`worker_profile` — arms ``torch.profiler`` for exactly one future engine step (selected
  by its index after arming) and returns the top ops by CUDA time.
"""

from __future__ import annotations

import time
from typing import Any

import torch

_TIMER_ATTR = "_rlk_museglimmer_step_timer"
_PROFILE_ATTR = "_rlk_museglimmer_profile"


def _kv_cache_bytes(worker: Any) -> tuple[int, int | None]:
    runner = getattr(worker, "model_runner", None)
    total = 0
    seen: set[int] = set()
    caches = getattr(runner, "kv_caches", None)
    if isinstance(caches, dict):
        caches = list(caches.values())
    for cache in caches or []:
        tensors = cache if isinstance(cache, (list, tuple)) else [cache]
        for tensor in tensors:
            if not isinstance(tensor, torch.Tensor):
                continue
            storage = tensor.untyped_storage()
            key = storage.data_ptr()
            if key in seen:
                continue
            seen.add(key)
            total += storage.nbytes()
    blocks = getattr(getattr(worker, "cache_config", None), "num_gpu_blocks", None)
    return total, (int(blocks) if blocks is not None else None)


def worker_memory(worker: Any, reset_peak: bool = False) -> dict[str, Any]:
    """CUDA memory of this worker's device (GiB) plus the KV-cache allocation."""

    torch.cuda.synchronize()
    gib = float(1 << 30)
    kv_bytes, blocks = _kv_cache_bytes(worker)
    out = {
        "device": str(torch.cuda.current_device()),
        "max_allocated_gib": torch.cuda.max_memory_allocated() / gib,
        "max_reserved_gib": torch.cuda.max_memory_reserved() / gib,
        "allocated_gib": torch.cuda.memory_allocated() / gib,
        "reserved_gib": torch.cuda.memory_reserved() / gib,
        "kv_cache_gib": kv_bytes / gib,
        "num_gpu_blocks": blocks,
        "block_size_tokens": getattr(getattr(worker, "cache_config", None), "block_size", None),
        "device_total_gib": torch.cuda.get_device_properties(0).total_memory / gib,
    }
    if reset_peak:
        torch.cuda.reset_peak_memory_stats()
    return out


def _scheduled(scheduler_output: Any) -> tuple[int, int]:
    tokens = int(getattr(scheduler_output, "total_num_scheduled_tokens", 0) or 0)
    per_req = getattr(scheduler_output, "num_scheduled_tokens", None) or {}
    return tokens, len(per_req)


def worker_step_timer(worker: Any, action: str = "start") -> list[dict[str, Any]] | None:
    """``action='start'`` installs the per-step timer; ``'stop'`` removes it and returns steps."""

    if action == "start":
        if getattr(worker, _TIMER_ATTR, None) is not None:
            worker_step_timer(worker, "stop")
        record: dict[str, Any] = {"steps": [], "open": None}
        original_execute = worker.execute_model
        original_sample = worker.sample_tokens

        def execute_model(scheduler_output: Any, *args: Any, **kwargs: Any) -> Any:
            tokens, reqs = _scheduled(scheduler_output)
            if tokens == 0:
                return original_execute(scheduler_output, *args, **kwargs)
            torch.cuda.synchronize()
            start = time.perf_counter()
            out = original_execute(scheduler_output, *args, **kwargs)
            if out is not None:
                torch.cuda.synchronize()
                record["steps"].append(
                    {
                        "tokens": tokens,
                        "reqs": reqs,
                        "ms": (time.perf_counter() - start) * 1e3,
                        "sampled_in_execute": True,
                    }
                )
            else:
                record["open"] = (start, tokens, reqs)
            return out

        def sample_tokens(*args: Any, **kwargs: Any) -> Any:
            out = original_sample(*args, **kwargs)
            pending = record["open"]
            if pending is not None:
                torch.cuda.synchronize()
                start, tokens, reqs = pending
                record["steps"].append(
                    {"tokens": tokens, "reqs": reqs, "ms": (time.perf_counter() - start) * 1e3}
                )
                record["open"] = None
            return out

        record["originals"] = (original_execute, original_sample)
        worker.execute_model = execute_model
        worker.sample_tokens = sample_tokens
        setattr(worker, _TIMER_ATTR, record)
        return None
    record = getattr(worker, _TIMER_ATTR, None)
    if record is None:
        return []
    for name in ("execute_model", "sample_tokens"):
        if name in worker.__dict__:
            del worker.__dict__[name]
    setattr(worker, _TIMER_ATTR, None)
    return list(record["steps"])


def _top_ops(prof: Any, top: int) -> list[dict[str, Any]]:
    rows = []
    for evt in prof.key_averages():
        cuda_us = float(
            getattr(evt, "self_device_time_total", None)
            or getattr(evt, "self_cuda_time_total", 0.0)
            or 0.0
        )
        total_us = float(
            getattr(evt, "device_time_total", None) or getattr(evt, "cuda_time_total", 0.0) or 0.0
        )
        rows.append(
            {
                "name": str(evt.key)[:160],
                "calls": int(evt.count),
                "self_cuda_ms": cuda_us / 1e3,
                "cuda_total_ms": total_us / 1e3,
                "self_cpu_ms": float(evt.self_cpu_time_total) / 1e3,
            }
        )
    rows.sort(key=lambda r: r["self_cuda_ms"], reverse=True)
    return rows[:top]


def worker_profile(
    worker: Any,
    action: str = "arm",
    kind: str = "any",
    min_tokens: int = 0,
    reqs: int = 0,
    top: int = 25,
) -> dict[str, Any] | None:
    """Profile exactly one engine step.

    ``action='arm'``: the first non-empty ``execute_model`` after arming that matches ``kind``
    runs under ``torch.profiler`` until its ``sample_tokens`` returns.  ``kind='decode'`` matches a
    pure decode step (scheduled tokens == requests) with at least ``reqs`` requests;
    ``kind='prefill'`` matches a step with at least ``min_tokens`` scheduled tokens and more
    tokens than requests; ``kind='any'`` matches the next step.  (Requests reach the engine core
    asynchronously, so step indices are not a reliable selector.)  ``action='result'`` returns
    ``{"step": {...}, "top_ops": [...], "total_self_cuda_ms": x}`` and disarms.
    """

    if action == "arm":
        state: dict[str, Any] = {"seen": 0, "prof": None, "result": None}
        original_execute = worker.execute_model
        original_sample = worker.sample_tokens

        def _finish() -> None:
            prof = state["prof"]
            torch.cuda.synchronize()
            prof.__exit__(None, None, None)
            ops = _top_ops(prof, int(top))
            total = sum(r["self_cuda_ms"] for r in _top_ops(prof, 10**6))
            state["result"] = {
                "step": dict(state["step"], wall_ms=(time.perf_counter() - state["t0"]) * 1e3),
                "top_ops": ops,
                "total_self_cuda_ms": total,
            }
            state["prof"] = None

        def execute_model(scheduler_output: Any, *args: Any, **kwargs: Any) -> Any:
            tokens, reqs_now = _scheduled(scheduler_output)
            if tokens == 0 or state["result"] is not None or state["prof"] is not None:
                return original_execute(scheduler_output, *args, **kwargs)
            index = state["seen"]
            state["seen"] += 1
            if kind == "decode":
                match = tokens == reqs_now and reqs_now >= int(reqs)
            elif kind == "prefill":
                match = tokens > reqs_now and tokens >= int(min_tokens)
            else:
                match = True
            if not match:
                return original_execute(scheduler_output, *args, **kwargs)
            torch.cuda.synchronize()
            prof = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ]
            )
            prof.__enter__()
            state["prof"] = prof
            state["step"] = {"tokens": tokens, "reqs": reqs_now, "index": index, "kind": kind}
            state["t0"] = time.perf_counter()
            out = original_execute(scheduler_output, *args, **kwargs)
            if out is not None:
                _finish()
            return out

        def sample_tokens(*args: Any, **kwargs: Any) -> Any:
            out = original_sample(*args, **kwargs)
            if state["prof"] is not None:
                _finish()
            return out

        worker.execute_model = execute_model
        worker.sample_tokens = sample_tokens
        setattr(worker, _PROFILE_ATTR, state)
        return None
    state = getattr(worker, _PROFILE_ATTR, None)
    for name in ("execute_model", "sample_tokens"):
        if name in worker.__dict__:
            del worker.__dict__[name]
    setattr(worker, _PROFILE_ATTR, None)
    if state is None:
        return None
    return state["result"]


__all__ = ["worker_memory", "worker_profile", "worker_step_timer"]
