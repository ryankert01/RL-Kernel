# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Per-step timeline and profiling helpers for the Muse-Glimmer-30B POC GRPO loop.

Imported by the orchestrator (vLLM env) and by the trainer server (Megatron env); depends on
``torch`` and the standard library only.

* :class:`PhaseClock` measures wall time (``time.perf_counter``) of the named phases of one step.
  :data:`PHASE_ORDER` is the canonical order the report plots them in.
* :func:`profiled` wraps one call in ``torch.profiler`` and writes a top-ops table as JSON
  (:func:`top_ops_table`); CUDA activity is recorded only when CUDA is already initialized in the
  calling process, so profiling never creates a CUDA context on its own.
* :func:`summarize_timeline` and :func:`summarize_bridge_calls` turn ``steps.jsonl`` records into
  the plottable series stored in ``summary.json["timeline"]``.
"""

from __future__ import annotations

import contextlib
import json
import time
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import torch

TIMELINE_SCHEMA = "muse_glimmer_e2e_timeline_v1"
TOP_OPS_SCHEMA = "muse_glimmer_top_ops_v1"

# Canonical order of the phases of one GRPO step (also the stacking order of the report figure).
PHASE_ORDER = (
    "rollout_generate",  # rollout.generate (vLLM): sampling + strict selected-token logprobs
    "reward",  # flatten groups + verifiable reward per sample
    "trainer_logprobs",  # trainer.token_logprobs on the same tokens (bridge RPC included)
    "drift_metrics",  # compare_logprobs(train, rollout)
    "rollout_recheck",  # rollout.logprobs prefill re-score + compare (decode-vs-prefill check)
    "sidecar",  # mismatch sidecar torch.save
    "grpo_loss",  # advantages + clipped token loss
    "backward_step",  # trainer.step(loss): local backward + gradient hand-off + optimizer step
    "lora_export",  # trainer.export_lora + canonical hash verification (bridge RPC included)
    "lora_transfer_load",  # optional file hop + rollout.load_lora + weight_sync_identity check
)


class PhaseClock:
    """Accumulates wall seconds per phase name (re-entering a phase adds to it)."""

    def __init__(self) -> None:
        self._seconds: dict[str, float] = {}
        self._start = time.perf_counter()

    @contextlib.contextmanager
    def phase(self, name: str) -> Iterator[None]:
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self._seconds[name] = self._seconds.get(name, 0.0) + (time.perf_counter() - t0)

    def add(self, name: str, seconds: float) -> None:
        self._seconds[name] = self._seconds.get(name, 0.0) + float(seconds)

    def get(self, name: str) -> float:
        return self._seconds.get(name, 0.0)

    def elapsed(self) -> float:
        return time.perf_counter() - self._start

    def as_dict(self) -> dict[str, float]:
        ordered = {name: self._seconds.get(name, 0.0) for name in PHASE_ORDER}
        for name, value in self._seconds.items():
            if name not in ordered:
                ordered[name] = value
        return ordered


# --------------------------------------------------------------------------------------------
# torch.profiler capture -> top-ops JSON
# --------------------------------------------------------------------------------------------


def _event_attr(event: Any, *names: str) -> float:
    for name in names:
        value = getattr(event, name, None)
        if value is not None:
            return float(value)
    return 0.0


def top_ops_table(
    prof: Any, *, label: str, limit: int = 40, device_activity: bool
) -> dict[str, Any]:
    """Top operators of a finished ``torch.profiler.profile`` as a JSON-able table.

    Sorted by self device time when device activity was recorded, else by self CPU time. Times
    are microseconds as reported by the profiler (``FunctionEventAvg``).
    """

    rows = []
    for event in prof.key_averages():
        rows.append(
            {
                "name": str(event.key),
                "count": int(event.count),
                "self_cpu_us": _event_attr(event, "self_cpu_time_total"),
                "cpu_total_us": _event_attr(event, "cpu_time_total"),
                "self_device_us": _event_attr(
                    event, "self_device_time_total", "self_cuda_time_total"
                ),
                "device_total_us": _event_attr(event, "device_time_total", "cuda_time_total"),
            }
        )
    sort_key = "self_device_us" if device_activity else "self_cpu_us"
    rows.sort(key=lambda row: row[sort_key], reverse=True)
    total_self_cpu = sum(row["self_cpu_us"] for row in rows)
    total_self_device = sum(row["self_device_us"] for row in rows)
    return {
        "schema": TOP_OPS_SCHEMA,
        "label": label,
        "sort_by": sort_key,
        "device_activity": device_activity,
        "num_distinct_ops": len(rows),
        "total_self_cpu_us": total_self_cpu,
        "total_self_device_us": total_self_device,
        "rows": rows[:limit],
    }


@contextlib.contextmanager
def profiled(
    enabled: bool, *, out_path: Path | None, label: str, limit: int = 40
) -> Iterator[dict[str, Any]]:
    """Profile the ``with`` body when ``enabled``; write the top-ops table to ``out_path``.

    Yields a dict that is filled after the body with ``{"path", "wall_s", "device_activity",
    "num_distinct_ops"}`` (empty when disabled).
    """

    info: dict[str, Any] = {}
    if not enabled:
        yield info
        return
    from torch.profiler import ProfilerActivity, profile

    device_activity = bool(torch.cuda.is_available() and torch.cuda.is_initialized())
    activities = [ProfilerActivity.CPU]
    if device_activity:
        activities.append(ProfilerActivity.CUDA)
    t0 = time.perf_counter()
    with profile(activities=activities, record_shapes=False) as prof:
        yield info
        if device_activity:
            torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    table = top_ops_table(prof, label=label, limit=limit, device_activity=device_activity)
    table["wall_s"] = wall
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(table, indent=2) + "\n", encoding="utf-8")
    info.update(
        {
            "path": str(out_path) if out_path is not None else None,
            "label": label,
            "wall_s": wall,
            "device_activity": device_activity,
            "num_distinct_ops": table["num_distinct_ops"],
        }
    )


# --------------------------------------------------------------------------------------------
# Summaries for summary.json
# --------------------------------------------------------------------------------------------


def _mean(values: Sequence[float]) -> float | None:
    return (sum(values) / len(values)) if values else None


def summarize_bridge_calls(calls: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate RemoteTrainer call stats per op: RTT vs server compute vs transport overhead."""

    per_op: dict[str, dict[str, Any]] = {}
    for call in calls:
        op = str(call.get("op"))
        server = call.get("server") or {}
        entry = per_op.setdefault(
            op,
            {
                "count": 0,
                "rtt_s": [],
                "server_compute_s": [],
                "overhead_s": [],
                "request_bytes": [],
                "response_bytes": [],
            },
        )
        rtt = float(call.get("rtt_s", 0.0))
        compute = float(server.get("compute_s", 0.0))
        entry["count"] += 1
        entry["rtt_s"].append(rtt)
        entry["server_compute_s"].append(compute)
        entry["overhead_s"].append(rtt - compute)
        entry["request_bytes"].append(int(call.get("request_bytes", 0)))
        entry["response_bytes"].append(int(call.get("response_bytes", 0)))
    out = {}
    for op, entry in per_op.items():
        out[op] = {
            "count": entry["count"],
            "rtt_s_mean": _mean(entry["rtt_s"]),
            "server_compute_s_mean": _mean(entry["server_compute_s"]),
            "overhead_s_mean": _mean(entry["overhead_s"]),
            "request_bytes_total": sum(entry["request_bytes"]),
            "response_bytes_total": sum(entry["response_bytes"]),
            "rtt_s": entry["rtt_s"],
            "server_compute_s": entry["server_compute_s"],
        }
    return out


def summarize_timeline(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Per-phase series / means / totals / fractions of step time, plus per-step scalars."""

    phases: list[str] = list(PHASE_ORDER)
    for record in records:
        for name in record.get("timeline", {}).get("phases_s", {}):
            if name not in phases:
                phases.append(name)
    per_step = {
        name: [float(r.get("timeline", {}).get("phases_s", {}).get(name, 0.0)) for r in records]
        for name in phases
    }
    step_s = [float(r["timings"]["step_s"]) for r in records]
    accounted = [sum(per_step[name][i] for name in phases) for i in range(len(records))]
    per_step["other"] = [max(0.0, s - a) for s, a in zip(step_s, accounted)]
    total_step = sum(step_s)
    calls: list[Mapping[str, Any]] = []
    for record in records:
        calls.extend(record.get("bridge_calls") or [])
    return {
        "schema": TIMELINE_SCHEMA,
        "phase_order": phases + ["other"],
        "steps": [int(r["step"]) for r in records],
        "step_s": step_s,
        "phases_s": per_step,
        "mean_s": {name: _mean(series) for name, series in per_step.items()},
        "total_s": {name: sum(series) for name, series in per_step.items()},
        "fraction_of_step": {
            name: (sum(series) / total_step) if total_step > 0 else None
            for name, series in per_step.items()
        },
        "tokens_generated": [int(r["num_completion_tokens"]) for r in records],
        "rollout_tokens_per_s": [
            float(r.get("timeline", {}).get("rollout_tokens_per_s") or 0.0) for r in records
        ],
        "reward_mean": [float(r["reward_mean"]) for r in records],
        "bitwise_mismatch_count": [int(r["drift"]["bitwise_mismatch_count"]) for r in records],
        "token_count": [int(r["drift"]["token_count"]) for r in records],
        "max_abs_diff": [float(r["drift"]["max_abs_diff"]) for r in records],
        "mismatch_k3_kl": [float(r["drift"]["mismatch_k3_kl"]) for r in records],
        "bridge": summarize_bridge_calls(calls) if calls else None,
    }
