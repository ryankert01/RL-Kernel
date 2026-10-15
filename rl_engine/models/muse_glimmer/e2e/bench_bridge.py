# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Measurements of the cross-environment trainer bridge for the POC report.

Writes (merges into) one JSON object, default ``reports/muse_glimmer_poc/data/e2e_bridge.json``:

* ``surrogate_exactness``: the gradient of ``sum_i (lp_i * g_i).sum()`` w.r.t. ``lp_i`` vs ``g_i``,
  byte compare over random + special fp32 values (CPU; CUDA with ``--gpu``).
* ``bridge_equivalence``: RemoteTrainer (server in the Megatron env / in this env; process
  launcher, and a 2-rank torchrun control-plane check where every rank holds a full replica)
  vs the same trainer in process: per step logprob bit mismatches, grad norms, LoRA hash and
  byte mismatches.
* ``transport``: echo round trips of 1 KiB .. 256 MiB uint8 payloads and of a LoRA state dict
  with the REAL Muse-Glimmer-30B rank-16 key schema and shapes (832 bf16 tensors, 200 MiB),
  with TCP_NODELAY on (production) and off (Nagle, both ends), every repeat as a raw row: RTT,
  client encode/decode, server time, throughput, bit-exactness, lora_sha256 cost.
* ``startup_overlap`` (``--startup-overlap``): a mock trainer whose build takes
  ``build_delay_s`` is launched with ``wait=False``; the caller works ``caller_work_s`` (vLLM
  start-up stand-in) and then calls ``wait_ready``: blocked time and launch-to-ready per row.
* ``failure_handling``: time from the failing call to the client-side exception for a server
  crash, a hang (timeout), a remote exception in a stateful op, startup failures, a killed
  client, a crashing torchrun rank, divergent torchrun ranks and a forward-only trainer.
* ``mock_e2e``: the GRPO CLI on the CPU mock engines with the trainer behind the bridge (all four
  modes) and in process (strict/strict): per-step phase timeline, drift, RPC breakdown.
* ``gpu_smoke`` (``--gpu``, run under a 1-GPU lease): placement (the server sees exactly the
  leased GPU, matched by UUID), CUDA surrogate exactness, a CUDA mock trainer through the bridge
  vs in process, server peak memory.
* ``real_e2e_smoke`` (``--import-e2e-run DIR ...``): key numbers and raw per-step series of real
  ``run_poc_grpo`` run directories (30B engines), read from their summary/steps/provenance files
  and the trainer-server log.

    PYTHONPATH=$PWD ~/envs/vllm/bin/python -m rl_engine.models.muse_glimmer.e2e.bench_bridge \\
        --cpu --hash-check --startup-overlap
    ~/bin/gpu-lease -n 1 --pool 3,7 -- env PYTHONPATH=$PWD ~/envs/vllm/bin/python \\
        -m rl_engine.models.muse_glimmer.e2e.bench_bridge --gpu
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
from importlib import metadata
from pathlib import Path
from typing import Any, Callable

import torch

from rl_engine.models.muse_glimmer.e2e import bridge, grpo, mocks, run_poc_grpo
from rl_engine.models.muse_glimmer.e2e.sync import LORA_PROJECTIONS, lora_sha256

ROW = "final_e2e_lora (cross-environment trainer bridge + timeline instrumentation)"
DEFAULT_OUT = bridge.REPO_ROOT / "reports" / "muse_glimmer_poc" / "data" / "e2e_bridge.json"
MEGATRON_PY = str(Path("~/envs/megatron/bin/python").expanduser())
SEQS = [[256, 3, 5, 7, 9, 11, 13], [256, 4, 6, 8, 10], [256, 1, 2, 3, 4, 5, 6, 7], [256, 9, 9]]
STARTS = [3, 2, 4, 1]
ADVANTAGES = torch.tensor([1.0, -1.0, 0.5, -0.5])

# Muse-Glimmer-30B (manifest): hidden 6656, Q/O width 4096, K/V width 256, FFN 19968, 52 layers.
REAL_LORA_SHAPES = {  # proj -> (in K, out N)
    "q_proj": (6656, 4096),
    "k_proj": (6656, 256),
    "v_proj": (6656, 256),
    "gate_proj": (6656, 4096),
    "o_proj": (4096, 6656),
    "mlp.gate_proj": (6656, 19968),
    "mlp.up_proj": (6656, 19968),
    "mlp.down_proj": (19968, 6656),
}
REAL_LAYERS = 52


def _bits(t: torch.Tensor) -> bytes:
    return bridge.tensor_bytes(t)


def _byte_mismatches(a: torch.Tensor, b: torch.Tensor) -> int:
    """Number of ELEMENTS whose bytes differ (shapes must match)."""

    if a.shape != b.shape or a.dtype != b.dtype:
        return max(a.numel(), b.numel())
    width = a.element_size()
    ua = a.detach().cpu().contiguous().reshape(-1).view(torch.uint8).view(-1, width)
    ub = b.detach().cpu().contiguous().reshape(-1).view(torch.uint8).view(-1, width)
    return int((ua != ub).any(dim=1).sum().item())


def _git_head() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=bridge.REPO_ROOT, capture_output=True, text=True,
            check=True,
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _client_env() -> dict[str, Any]:
    out: dict[str, Any] = {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_lease": os.environ.get("GPU_LEASE"),
        "cpu_count": os.cpu_count(),
    }
    for name in ("triton", "vllm", "transformers"):
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def _remote(work: Path, *, python: str = MEGATRON_PY, launcher: str = "process", tp: int = 1,
            factory: str = bridge.MOCK_TRAINER_FACTORY, factory_kwargs=None, gpu_ids=(),
            call_timeout_s: float = 300.0, startup_timeout_s: float = 300.0,
            threads: int = 2, **kwargs: Any) -> bridge.RemoteTrainer:
    return bridge.RemoteTrainer(
        "mock", tensor_parallel_size=tp, gpu_ids=list(gpu_ids), lora_rank=4, lr=1e-3,
        strict=True, seed=7, python=python, launcher=launcher, factory=factory,
        factory_kwargs=factory_kwargs or {}, work_dir=work, call_timeout_s=call_timeout_s,
        startup_timeout_s=startup_timeout_s, extra_env={"OMP_NUM_THREADS": str(threads)},
        **kwargs,
    )


def _local(device: str = "cpu") -> mocks.MockTrainer:
    return mocks.MockTrainer(
        "mock", tensor_parallel_size=1, gpu_ids=[], lora_rank=4, lr=1e-3, strict=True, seed=7,
        device=device,
    )


def _steps(trainer: Any, steps: int) -> list[dict[str, Any]]:
    out = []
    for _ in range(steps):
        lps = trainer.token_logprobs(SEQS, STARTS)
        rollout = [t.detach().clone() for t in lps]
        loss = grpo.grpo_token_loss(lps, rollout, ADVANTAGES).loss
        info = trainer.step(loss)
        state_dict, sha = trainer.export_lora()
        out.append(
            {
                "logprobs": [t.detach().cpu() for t in lps],
                "grad_norm": float(info["grad_norm"]),
                "sha256": sha,
                "state_dict": {k: v.detach().cpu() for k, v in state_dict.items()},
            }
        )
    return out


def _compare_runs(remote: list[dict], local: list[dict]) -> list[dict[str, Any]]:
    rows = []
    for step, (r, l) in enumerate(zip(remote, local)):
        lp_tokens = sum(t.numel() for t in l["logprobs"])
        lp_mis = sum(_byte_mismatches(a, b) for a, b in zip(r["logprobs"], l["logprobs"]))
        lora_bytes = sum(t.numel() * t.element_size() for t in l["state_dict"].values())
        lora_mis = sum(
            _byte_mismatches(r["state_dict"][k], l["state_dict"][k]) for k in l["state_dict"]
        )
        rows.append(
            {
                "step": step,
                "logprob_tokens": lp_tokens,
                "logprob_bit_mismatches": lp_mis,
                "grad_norm_remote": r["grad_norm"],
                "grad_norm_local": l["grad_norm"],
                "grad_norm_equal": r["grad_norm"] == l["grad_norm"],
                "lora_sha256_remote": r["sha256"],
                "lora_sha256_local": l["sha256"],
                "lora_sha256_equal": r["sha256"] == l["sha256"],
                "lora_bytes": lora_bytes,
                "lora_element_mismatches": lora_mis,
            }
        )
    return rows


# --------------------------------------------------------------------------------------------
# Experiments
# --------------------------------------------------------------------------------------------


def surrogate_exactness(device: str) -> dict[str, Any]:
    gen = torch.Generator().manual_seed(1234)
    lengths = [1, 7, 64, 256, 1000, 4096]
    grads = []
    for n in lengths:
        g = torch.randn(n, generator=gen) * torch.logspace(-40, 38, n, dtype=torch.float32)
        grads.append(g)
    special = torch.tensor(
        [0.0, -0.0, 1e-45, -1e-45, 2.0**-149, 1.1754942e-38, 3.4e38, -3.4e38, 1.0, -1.0],
        dtype=torch.float32,
    )
    grads[-1][: special.numel()] = special
    subnormals = sum(int(((g != 0) & (g.abs() < 1.1754944e-38)).sum()) for g in grads)
    src = torch.randn(sum(lengths), device=device, requires_grad=True)
    packed = src * 1.0
    logps = list(torch.split(packed, lengths))
    dev_grads = [g.to(device) for g in grads]
    surrogate = bridge.surrogate_loss(logps, dev_grads)
    got = torch.autograd.grad(surrogate, logps, retain_graph=True)
    per_tensor = [_byte_mismatches(a, b) for a, b in zip(got, grads)]
    (g_packed,) = torch.autograd.grad(surrogate, [packed])
    packed_mis = _byte_mismatches(g_packed, torch.cat(grads))
    return {
        "device": device,
        "device_name": torch.cuda.get_device_name() if device.startswith("cuda") else "cpu",
        "num_tensors": len(lengths),
        "lengths": lengths,
        "num_elements": sum(lengths),
        "subnormal_elements": subnormals,
        "signed_zero_elements": 2,
        "element_mismatches_per_tensor": per_tensor,
        "element_mismatches_total": sum(per_tensor),
        "packed_split_element_mismatches": packed_mis,
        "bit_exact": sum(per_tensor) == 0 and packed_mis == 0,
    }


def bridge_equivalence(work: Path, steps: int = 4) -> list[dict[str, Any]]:
    local = _steps(_local(), steps)
    out = []
    notes = {
        "megatron_env_torchrun_2rank_replicated_control_plane": (
            "NOT tensor parallel: each of the 2 torchrun ranks builds a full-replica CPU "
            "MockTrainer (gpu_ids=[]) and rank 0 answers. Validates only the control plane: "
            "every op broadcast to every rank in order, per-rank status gather, rank LoRA-hash "
            "and logprob-digest agreement. No TP numerics are exercised."
        ),
    }
    for name, python, launcher, tp in (
        ("megatron_env_process", MEGATRON_PY, "process", 1),
        ("same_env_process", sys.executable, "process", 1),
        ("megatron_env_torchrun_2rank_replicated_control_plane", MEGATRON_PY, "torchrun", 2),
    ):
        t0 = time.perf_counter()
        remote = _remote(work / name, python=python, launcher=launcher, tp=tp)
        startup = time.perf_counter() - t0
        try:
            hello = remote.wait_ready()
            got = _steps(remote, steps)
            rank_sha = remote.last_export.get("rank_sha256") if remote.last_export else None
            calls = [
                {"op": c["op"], "rtt_ms": c["rtt_s"] * 1e3,
                 "server_ms": (c["server"].get("compute_s") or 0.0) * 1e3,
                 "request_bytes": c["request_bytes"], "response_bytes": c["response_bytes"]}
                for c in remote.calls
            ]
        finally:
            remote.close()
        rows = _compare_runs(got, local)
        out.append(
            {
                "name": name,
                "server_python": python,
                "server_torch": hello["versions"]["torch"],
                "launcher": launcher,
                "tp": tp,
                "startup_s": startup,
                "steps": rows,
                "all_bit_exact": all(
                    r["logprob_bit_mismatches"] == 0 and r["lora_element_mismatches"] == 0
                    and r["grad_norm_equal"] and r["lora_sha256_equal"] for r in rows
                ),
                "last_export_rank_sha256": rank_sha,
                "note": notes.get(name),
                "calls": calls,
            }
        )
    return out


def _real_shape_lora(seed: int = 0) -> dict[str, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    out: dict[str, torch.Tensor] = {}
    r = 16
    for layer in range(REAL_LAYERS):
        for proj in LORA_PROJECTIONS:
            k, n = REAL_LORA_SHAPES[proj]
            out[f"layers.{layer}.{proj}.lora_A"] = (
                torch.randn(r, k, generator=gen) * 0.01
            ).to(torch.bfloat16)
            out[f"layers.{layer}.{proj}.lora_B"] = (
                torch.randn(n, r, generator=gen) * 0.01
            ).to(torch.bfloat16)
    return out


def _ping_row(nodelay: bool, rep: int, order: int, size: int, stats: dict, ok: bool) -> dict:
    return {
        "tcp_nodelay": nodelay,
        "payload_bytes": size,
        "repeat": rep,
        "order_in_pair": order,
        "rtt_ms": stats["rtt_s"] * 1e3,
        "client_encode_ms": stats["encode_s"] * 1e3,
        "client_decode_ms": stats["decode_s"] * 1e3,
        "server_ms": (stats["server"].get("compute_s") or 0.0) * 1e3,
        "request_bytes": stats["request_bytes"],
        "response_bytes": stats["response_bytes"],
        "round_trip_MBps": 2 * size / stats["rtt_s"] / 1e6,
        "bit_exact": ok,
    }


def transport(work: Path, repeats: int = 5) -> dict[str, Any]:
    """Echo RTTs with TCP_NODELAY on and off. Both servers stay up and every repeat pings both,
    alternating which goes first, so a busy shared node perturbs both variants alike."""

    sizes = [1 << 10, 16 << 10, 256 << 10, 4 << 20, 32 << 20, 128 << 20, 256 << 20]
    remotes = {nd: _remote(work / f"transport_nodelay_{int(nd)}", tcp_nodelay=nd)
               for nd in (True, False)}
    rows: list[dict[str, Any]] = []
    lora_rows: list[dict[str, Any]] = []
    load_before = os.getloadavg()
    try:
        for nd, remote in remotes.items():
            assert remote.wait_ready()["tcp_nodelay"] is nd
        gen = torch.Generator().manual_seed(0)
        for size in sizes:
            payload = torch.randint(0, 256, (size,), dtype=torch.uint8, generator=gen)
            for rep in range(repeats):
                order = (True, False) if rep % 2 == 0 else (False, True)
                for k, nd in enumerate(order):
                    echoed, stats = remotes[nd].ping(payload)
                    rows.append(_ping_row(nd, rep, k, size, stats,
                                          bool(torch.equal(echoed, payload))))
        lora = _real_shape_lora()
        lora_bytes = sum(t.numel() * t.element_size() for t in lora.values())
        t0 = time.perf_counter()
        sha_before = lora_sha256(lora)
        sha_ms = (time.perf_counter() - t0) * 1e3
        for rep in range(repeats):
            order = (True, False) if rep % 2 == 0 else (False, True)
            for k, nd in enumerate(order):
                echoed, stats = remotes[nd].ping(lora)  # type: ignore[arg-type]
                t0 = time.perf_counter()
                sha_after = lora_sha256(echoed)
                sha_after_s = time.perf_counter() - t0
                lora_rows.append(
                    {
                        "tcp_nodelay": nd,
                        "repeat": rep,
                        "order_in_pair": k,
                        "num_tensors": len(lora),
                        "lora_bytes": lora_bytes,
                        "rtt_ms": stats["rtt_s"] * 1e3,
                        "client_encode_ms": stats["encode_s"] * 1e3,
                        "client_decode_ms": stats["decode_s"] * 1e3,
                        "server_ms": (stats["server"].get("compute_s") or 0.0) * 1e3,
                        "one_way_MBps": lora_bytes / (stats["rtt_s"] / 2) / 1e6,
                        "sha256_equal": sha_after == sha_before,
                        "lora_sha256_ms": sha_after_s * 1e3,
                    }
                )
    finally:
        for remote in remotes.values():
            remote.close()
    return {
        "note": "echo through the trainer server (megatron env, process launcher, CPU); one "
        "round trip = request + reply, each a torch.save blob over a localhost socket. "
        "tcp_nodelay=True is the production setting (bridge.set_nodelay on both ends); "
        "tcp_nodelay=False keeps Nagle on both ends (RemoteTrainer(tcp_nodelay=False)). Both "
        "servers run at once and every repeat pings both (order alternates per repeat). The "
        "node is shared with other agents: see load_average_1_5_15.",
        "repeats": repeats,
        "load_average_1_5_15": {"before": list(load_before), "after": list(os.getloadavg())},
        "payloads": rows,
        "real_lora_rank16": {
            "shape_source": "museglimmer_manifest dims, 52 layers x 8 projections x (A, B), bf16",
            "lora_sha256_ms_before_send": sha_ms,
            "rows": lora_rows,
        },
    }


def startup_overlap(work: Path, *, build_delay_s: float = 6.0,
                    caller_work_s: tuple[float, ...] = (0.0, 3.0, 6.0, 9.0, 12.0)
                    ) -> dict[str, Any]:
    """Does the trainer build overlap the caller's work between launch and ``wait_ready``?"""

    rows = []
    for caller in caller_work_s:
        t0 = time.monotonic()
        remote = _remote(work / f"overlap_{caller:g}", wait=False,
                         factory_kwargs={"build_delay_s": build_delay_s})
        try:
            time.sleep(caller)  # stands in for the orchestrator building vLLM
            t1 = time.monotonic()
            hello = remote.wait_ready()
            t2 = time.monotonic()
        finally:
            remote.close()
        rows.append(
            {
                "caller_work_s": caller,
                "build_delay_s": build_delay_s,
                "server_build_s": hello["build_s"],
                "wait_ready_blocked_s": t2 - t1,
                "launch_to_ready_s": t2 - t0,
                "startup": hello.get("startup"),
            }
        )
    return {
        "note": "process launcher, megatron env, CPU MockTrainer(build_delay_s) = a trainer whose "
        "build takes build_delay_s (the 30B StrictTrainer build took 100-164 s in the real "
        "smokes). Serial behaviour: launch_to_ready ~ caller_work + interpreter start + build; "
        "overlapped: ~ max(caller_work, interpreter start + build).",
        "build_delay_s": build_delay_s,
        "rows": rows,
    }


def _timed(fn: Callable[[], Any]) -> tuple[float, str]:
    t0 = time.monotonic()
    try:
        fn()
    except Exception as exc:  # noqa: BLE001 - we measure the failure
        lines = [line.strip() for line in str(exc).splitlines() if line.strip()]
        text = f"{type(exc).__name__}: {lines[0] if lines else ''}"
        if len(lines) > 1 and not lines[-1].startswith("[trainer_server"):
            text += f" ... {lines[-1]}"
        return (time.monotonic() - t0) * 1e3, text
    return (time.monotonic() - t0) * 1e3, "NO EXCEPTION"


def failure_handling(work: Path) -> list[dict[str, Any]]:
    faulty = "rl_engine.models.muse_glimmer.e2e.mocks:FaultyMockTrainer"
    rows: list[dict[str, Any]] = []

    def scored(remote: bridge.RemoteTrainer) -> torch.Tensor:
        lps = remote.token_logprobs(SEQS, STARTS)
        return grpo.grpo_token_loss(lps, [t.detach() for t in lps], ADVANTAGES).loss

    for name, launcher, tp, kwargs, op, timeout in (
        ("server_exit_during_step", "process", 1, {"fault": "exit", "fault_op": "step"}, "step",
         120.0),
        ("server_hang_during_scoring", "process", 1,
         {"fault": "hang", "fault_op": "token_logprobs"}, "token_logprobs", 5.0),
        ("remote_exception_in_step", "process", 1, {"fault": "raise", "fault_op": "step"},
         "step", 120.0),
        ("torchrun_rank1_exit_during_step", "torchrun", 2,
         {"fault": "exit", "fault_op": "step", "fault_rank": 1}, "step", 120.0),
        ("torchrun_rank1_divergent_lora_export", "torchrun", 2,
         {"fault": "diverge_lora", "fault_op": "export_lora", "fault_rank": 1}, "export_lora",
         120.0),
        ("torchrun_rank1_divergent_logprobs", "torchrun", 2,
         {"fault": "diverge_logprobs", "fault_op": "token_logprobs", "fault_rank": 1},
         "token_logprobs", 120.0),
        ("forward_only_trainer_scoring", "process", 1,
         {"fault": "forward_only", "fault_op": "token_logprobs"}, "score_check", 120.0),
    ):
        remote = _remote(work / name, launcher=launcher, tp=tp, factory=faulty,
                         factory_kwargs=kwargs, call_timeout_s=timeout)
        pid = remote.server_pid
        try:
            if op == "step":
                loss = scored(remote)
                ms, err = _timed(lambda: remote.step(loss))
            elif op == "export_lora":
                ms, err = _timed(remote.export_lora)
            elif op == "score_check":
                from rl_engine.models.muse_glimmer.e2e.loop import _check_train_logps

                ms, err = _timed(lambda: _check_train_logps(
                    remote.token_logprobs(SEQS, STARTS), SEQS, STARTS))
            else:
                ms, err = _timed(lambda: remote.token_logprobs(SEQS, STARTS))
            alive_after = remote.alive()
            survives = None
            if alive_after:
                try:
                    survives = remote.provenance().get("engine") == "MockTrainer"
                except bridge.RemoteTrainerError:
                    survives = False
        finally:
            remote.close()
        rows.append({"scenario": name, "launcher": launcher, "call_timeout_s": timeout,
                     "detect_ms": ms, "error": err, "server_alive_after_error": alive_after,
                     "server_usable_after_error": survives, "server_pid": pid,
                     "server_gone_after_close": not _pid_alive(pid)})

    for name, factory, kwargs in (
        ("startup_bad_factory", "rl_engine.models.muse_glimmer.e2e.mocks:NoSuchTrainer", {}),
        ("startup_constructor_raises", faulty, {"fault": "build"}),
    ):
        ms, err = _timed(lambda: _remote(work / name, factory=factory, factory_kwargs=kwargs))
        rows.append({"scenario": name, "launcher": "process", "detect_ms": ms, "error": err})

    # Client killed: the server must notice and take its process group down.
    script = (
        "import sys, time\nfrom pathlib import Path\n"
        "from rl_engine.models.muse_glimmer.e2e import bridge\n"
        "r = bridge.RemoteTrainer('mock', tensor_parallel_size=1, gpu_ids=[], lora_rank=4, "
        "lr=1e-3, strict=True, seed=7, python=sys.argv[1], factory=bridge.MOCK_TRAINER_FACTORY,"
        " work_dir=Path(sys.argv[2]))\nprint(r.server_pid, flush=True)\ntime.sleep(600)\n"
    )
    env = dict(os.environ, PYTHONPATH=str(bridge.REPO_ROOT))
    client = subprocess.Popen([sys.executable, "-c", script, MEGATRON_PY,
                               str(work / "client_killed")], stdout=subprocess.PIPE, text=True,
                              env=env)
    server_pid = int(client.stdout.readline())  # type: ignore[union-attr]
    t0 = time.monotonic()
    client.send_signal(signal.SIGKILL)
    client.wait(timeout=10)
    while _pid_alive(server_pid) and time.monotonic() - t0 < 30.0:
        time.sleep(0.05)
    rows.append({"scenario": "client_killed_server_self_terminates", "launcher": "process",
                 "detect_ms": (time.monotonic() - t0) * 1e3,
                 "server_gone": not _pid_alive(server_pid)})
    return rows


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    try:
        with open(f"/proc/{pid}/stat", encoding="utf-8") as handle:
            return handle.read().split()[2] != "Z"
    except FileNotFoundError:
        return False


def mock_e2e(work: Path, *, steps: int, prompts: int, samples: int, max_tokens: int) -> dict:
    runs = {}
    base = ["--mock", "--steps", str(steps), "--prompts-per-step", str(prompts),
            "--samples-per-prompt", str(samples), "--max-tokens", str(max_tokens),
            "--lora-rank", "4", "--lr", "1e-3", "--threads", "4", "--trainer-python",
            MEGATRON_PY]
    plan = [(f"bridge_{m.replace('/', '-')}", m, ["--trainer-bridge", "on"])
            for m in ("strict/strict", "native/native", "strict/native", "native/strict")]
    plan.append(("inprocess_strict-strict", "strict/strict", ["--trainer-bridge", "off"]))
    plan.append(("bridge_strict-strict_profiled", "strict/strict",
                 ["--trainer-bridge", "on", "--profile-step", "1"]))
    commands = []
    for name, mode, extra in plan:
        run_dir = work / name
        argv = base + ["--mode", mode, "--run-dir", str(run_dir)] + extra
        commands.append(
            "python -m rl_engine.models.muse_glimmer.e2e.run_poc_grpo " + " ".join(argv)
        )
        code = run_poc_grpo.main(argv)
        summary = json.loads((run_dir / "summary.json").read_text())
        timeline = summary["timeline"]
        entry = {
            "mode": mode,
            "trainer_bridge": summary["config"]["trainer_bridge"],
            "exit_code": code,
            "status": summary["status"],
            "drift_totals": summary["drift_totals"],
            "first_drift": summary["first_drift"],
            "weight_sync": {k: summary["weight_sync"][k] for k in ("checks", "passed")},
            "final_lora_sha256": summary["weight_sync"]["final_sha256"],
            "reward_trajectory": summary["reward"]["trajectory"],
            "timeline": timeline,
        }
        if summary.get("profiles"):
            prof = summary["profiles"][0]
            tables = {}
            paths = {"orchestrator_rollout_generate": prof["orchestrator_rollout_generate"]}
            for item in prof["trainer_server"]:
                paths[f"trainer_server_{item['op']}"] = item
            for key, info in paths.items():
                if info and info.get("path"):
                    table = json.loads(Path(info["path"]).read_text())
                    tables[key] = {"wall_s": table.get("wall_s"), "sort_by": table["sort_by"],
                                   "total_self_cpu_us": table["total_self_cpu_us"],
                                   "rows": table["rows"][:15]}
            entry["profile_step"] = prof["step"]
            entry["top_ops"] = tables
        runs[name] = entry
    return {
        "note": "CPU mock engines (tiny byte LM, ~0.1 M params); trainer server in the megatron "
        "env. Timings show the orchestration + bridge overhead, NOT the 30B engines.",
        "config": {"steps": steps, "prompts_per_step": prompts, "samples_per_prompt": samples,
                   "max_tokens": max_tokens, "lora_rank": 4, "lr": 1e-3, "seed": 1234},
        "commands": commands,
        "runs": runs,
    }


def gpu_smoke(work: Path, steps: int = 3) -> dict[str, Any]:
    lease = bridge.parse_visible_devices(os.environ.get("CUDA_VISIBLE_DEVICES"))
    if not lease or not torch.cuda.is_available():
        return {"status": "NOT RUN", "reason": "needs a GPU lease (CUDA_VISIBLE_DEVICES)"}
    gpu = lease[:1]
    parent_uuid = str(torch.cuda.get_device_properties(0).uuid)
    out: dict[str, Any] = {
        "status": "ran",
        "lease": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_ids_physical": gpu,
        "parent_device_uuid": parent_uuid,
        "parent_device_name": torch.cuda.get_device_name(0),
        "surrogate_exactness_cuda": surrogate_exactness("cuda"),
    }
    local = _steps(_local("cuda"), steps)
    variants = []
    for name, python in (("megatron_env", MEGATRON_PY), ("same_env", sys.executable)):
        remote = _remote(work / f"gpu_{name}", python=python, gpu_ids=gpu,
                         factory_kwargs={"device": "cuda"})
        try:
            hello = remote.wait_ready()
            got = _steps(remote, steps)
            peak = [
                (c["server"].get("memory") or {}).get("peak_allocated_gib")
                for c in remote.calls if c["op"] in ("token_logprobs", "step")
            ]
            prov = remote.provenance()
        finally:
            remote.close()
        devices = hello["devices"]
        rows = _compare_runs(got, local)
        variants.append(
            {
                "name": name,
                "server_torch": hello["versions"]["torch"],
                "server_cuda_visible_devices": devices["cuda_visible_devices"],
                "server_device_count": devices["device_count"],
                "server_device_uuid": devices["devices"][0]["uuid"] if devices["devices"] else None,
                "placement_ok": devices["device_count"] == 1
                and devices["devices"][0]["uuid"] == parent_uuid,
                "trainer_device": prov.get("device"),
                "server_peak_allocated_gib_per_call": peak,
                "steps": rows,
                "all_bit_exact_vs_inprocess_cuda": all(
                    r["logprob_bit_mismatches"] == 0 and r["lora_element_mismatches"] == 0
                    and r["grad_norm_equal"] for r in rows
                ),
            }
        )
    out["cuda_mock_trainer_through_bridge"] = variants
    out["gated_by"] = (
        "tests/models/muse_glimmer/test_e2e_bridge_gpu.py::"
        "test_cuda_trainer_through_bridge_sees_exactly_the_leased_gpu asserts placement and bit "
        "equality with the in-process CUDA mock for BOTH variants (megatron_env and same_env)"
    )
    return out


def hash_schema_check() -> dict[str, Any]:
    """Does the Megatron binding's LoRA hash equal the loop's muse_glimmer_lora_sha256_v1?"""

    from rl_engine.models.muse_glimmer.runtime import megatron_lora

    gen = torch.Generator().manual_seed(5)
    cases = {
        "known_answer": {"layers.0.q_proj.lora_A": torch.zeros(1, 2, dtype=torch.bfloat16)},
        "random_rank16_layer0": {
            f"layers.0.{proj}.lora_{m}": (
                torch.randn(*((16, k) if m == "A" else (n, 16)), generator=gen) * 0.01
            ).to(torch.bfloat16)
            for proj, (k, n) in (("q_proj", (64, 32)), ("mlp.down_proj", (96, 64)))
            for m in ("A", "B")
        },
    }
    rows = []
    for name, state in cases.items():
        canonical = lora_sha256(state)
        megatron = megatron_lora.lora_sha256(state)
        rows.append({"case": name, "num_tensors": len(state),
                     "e2e_sync_muse_glimmer_lora_sha256_v1": canonical,
                     "runtime_megatron_lora_sha256": megatron, "equal": canonical == megatron})
    return {"rows": rows, "all_equal": all(r["equal"] for r in rows)}


XRT_TOKENS = bridge.REPO_ROOT / "reports" / "muse_glimmer_poc" / "data" / "xrt_tokens.json"


def real_trainer_smoke(work: Path, *, tp: int, launcher: str, model_path: str,
                       num_sequences: int) -> dict[str, Any]:
    """The REAL StrictTrainer (30B) behind the bridge: build, score, one step, export, close.

    GPUs come from the lease (``CUDA_VISIBLE_DEVICES``, first ``tp`` ids). Scores the first
    ``num_sequences`` sequences of ``xrt_tokens.json``; the loss is a GRPO loss with
    advantages +1/-1 alternating and rollout logprobs = the trainer's own (ratio 1).
    """

    lease = bridge.parse_visible_devices(os.environ.get("CUDA_VISIBLE_DEVICES"))
    if not lease or len(lease) < tp:
        return {"status": "NOT RUN", "reason": f"needs a lease of {tp} GPU(s)"}
    tokens = json.loads(XRT_TOKENS.read_text())
    seqs = tokens["sequences"][:num_sequences]
    ids = [list(s["token_ids"]) for s in seqs]
    starts = [int(s["completion_start"]) for s in seqs]
    out: dict[str, Any] = {"status": "ran", "tp": tp, "launcher": launcher, "gpus": lease[:tp],
                           "sequences": [s["id"] for s in seqs],
                           "scored_tokens": sum(len(i) - s for i, s in zip(ids, starts))}
    t0 = time.perf_counter()
    remote = bridge.RemoteTrainer(
        model_path, tensor_parallel_size=tp, gpu_ids=lease[:tp], lora_rank=16, lr=1e-5,
        strict=True, seed=1234, launcher=launcher, work_dir=work / f"real_tp{tp}",
        startup_timeout_s=3600.0, call_timeout_s=3600.0,
    )
    try:
        hello = remote.wait_ready()
        out["startup_s"] = time.perf_counter() - t0
        out["server_build_s"] = hello["build_s"]
        out["server_versions"] = hello["versions"]
        out["server_devices"] = hello["devices"]
        out["server_memory_after_build"] = hello["memory"]
        t1 = time.perf_counter()
        lps = remote.token_logprobs(ids, starts)
        out["token_logprobs_s"] = time.perf_counter() - t1
        out["logprobs_finite"] = all(bool(torch.isfinite(t).all()) for t in lps)
        out["logprobs_first8_hex"] = [
            ["%08x" % (v & 0xFFFFFFFF) for v in t.detach().view(torch.int32)[:8].tolist()]
            for t in lps
        ]
        adv = torch.tensor([1.0 if i % 2 == 0 else -1.0 for i in range(len(lps))])
        loss = grpo.grpo_token_loss(lps, [t.detach().clone() for t in lps], adv).loss
        t1 = time.perf_counter()
        info = remote.step(loss)
        out["step_s"] = time.perf_counter() - t1
        out["step_info"] = bridge.to_wire(info)
        t1 = time.perf_counter()
        try:
            state_dict, sha = remote.export_lora()
            out["export"] = {"ok": True, "sha256": sha, "num_tensors": len(state_dict),
                             "num_bytes": remote.last_export["num_bytes"]}
        except Exception as exc:  # noqa: BLE001 - recorded (e.g. hash-schema mismatch)
            out["export"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        out["export_s"] = time.perf_counter() - t1
        out["calls"] = [
            {"op": c["op"], "rtt_s": c["rtt_s"], "server": c["server"],
             "request_bytes": c["request_bytes"], "response_bytes": c["response_bytes"]}
            for c in remote.calls
        ]
    except Exception as exc:  # noqa: BLE001 - recorded for the report
        out["status"] = "failed"
        out["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        remote.close()
    return out


_LOG_LINE = re.compile(r"^\[trainer_server rank(\d+) (\d\d):(\d\d):(\d\d)\] (.*)$")


def _server_log_events(path: Path) -> list[dict[str, Any]]:
    """``[trainer_server rank0 HH:MM:SS] message`` lines (rank, clock, seconds of day, text)."""

    events = []
    if not path.exists():
        return events
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = _LOG_LINE.match(line.strip())
        if match:
            rank, hh, mm, ss, text = match.groups()
            events.append({"rank": int(rank), "clock": f"{hh}:{mm}:{ss}",
                           "sec_of_day": int(hh) * 3600 + int(mm) * 60 + int(ss),
                           "message": text})
    return events


def import_e2e_run(run_dir: Path, *, commit: str | None) -> dict[str, Any]:
    """Key numbers and raw per-step series of one real ``run_poc_grpo`` run directory."""

    summary = json.loads((run_dir / "summary.json").read_text())
    steps = [json.loads(line) for line in (run_dir / "steps.jsonl").read_text().splitlines()
             if line.strip()]
    prov = json.loads((run_dir / "provenance.json").read_text())
    cfg = summary["config"]
    code = summary.get("code") or {}
    rollout0, trainer0 = prov.get("rollout_start", {}), prov.get("trainer_start", {})
    hello = (trainer0.get("bridge") or {}).get("hello") or {}
    events = _server_log_events(run_dir / "trainer_server" / "trainer_server.log")

    def first(prefix: str) -> dict[str, Any] | None:
        return next((e for e in events if e["message"].startswith(prefix)), None)

    start, connected, built = first("start "), first("client connected"), first("trainer ")
    close = first("close ")
    rows = []
    for rec in steps:
        trainer_step = rec.get("trainer_step") or {}
        rows.append({
            "step": rec["step"],
            "num_samples": rec["num_samples"],
            "num_completion_tokens": rec["num_completion_tokens"],
            "rewards": rec["rewards"],
            "advantages": rec["advantages"],
            "drift": rec["drift"],
            "rollout_decode_vs_prefill": rec.get("rollout_decode_vs_prefill"),
            "grpo": {k: rec["grpo"].get(k) for k in (
                "loss", "ratio_mean", "ratio_min", "ratio_max", "ratio_exactly_one_frac",
                "clip_frac")},
            "trainer_step": {k: trainer_step.get(k) for k in (
                "grad_norm", "recompute_logp_mismatches", "recompute_logp_checked",
                "micro_batches", "fwd_bwd_s", "optimizer_s", "num_lora_params_local")},
            "sync": rec["sync"],
            "phases_s": (rec.get("timeline") or {}).get("phases_s"),
            "step_s": rec["timings"]["step_s"],
            "bridge_calls": [
                {"op": c["op"], "rtt_s": c["rtt_s"],
                 "server_compute_s": (c.get("server") or {}).get("compute_s"),
                 "request_bytes": c["request_bytes"], "response_bytes": c["response_bytes"]}
                for c in rec.get("bridge_calls") or []
            ],
        })
    return {
        "run_dir": str(run_dir),
        "commit": code.get("git_commit") or commit,
        "commit_source": "summary.code" if code.get("git_commit") else (
            "--e2e-run-commit (the run directory records no commit)" if commit else None),
        "mode": summary["mode"],
        "status": summary["status"],
        "error": summary["error"],
        "config": {k: cfg.get(k) for k in (
            "steps", "prompts_per_step", "samples_per_prompt", "max_tokens", "temperature",
            "lora_rank", "lr", "seed", "rollout_tp", "rollout_gpus", "train_tp", "train_gpus",
            "trainer_launcher", "task", "split")},
        "rollout": {k: rollout0.get(k) for k in (
            "engine", "runtime", "strict", "lora_mode", "logprob_path", "tensor_parallel_size",
            "gpu_ids")} | {"load_seconds": (rollout0.get("timing") or {}).get("load_seconds")},
        "trainer": {k: trainer0.get(k) for k in (
            "engine", "runtime", "strict", "tp", "binding", "load_seconds", "init_seconds")}
        | {"build_s": hello.get("build_s"), "startup_s": hello.get("startup_s"),
           "versions": hello.get("versions")},
        "trainer_server_log": {
            "start": start and start["clock"],
            "client_connected": connected and connected["clock"],
            "trainer_built": built and built["clock"],
            "close": close and close["message"],
            "start_to_client_connected_s": (connected["sec_of_day"] - start["sec_of_day"])
            if start and connected else None,
            "start_to_trainer_built_s": (built["sec_of_day"] - start["sec_of_day"])
            if start and built else None,
            "events": [{"clock": e["clock"], "message": e["message"][:200]} for e in events
                       if not e["message"].startswith("trainer construction failed")][:40],
        },
        "drift_totals": summary["drift_totals"],
        "first_drift": summary["first_drift"],
        "weight_sync": summary["weight_sync"],
        "lora_sha256": {
            "rollout_start": rollout0.get("lora_sha256"),
            "trainer_start": trainer0.get("lora_sha256"),
            "rollout_end": (prov.get("rollout_end") or {}).get("lora_sha256"),
            "trainer_end": (prov.get("trainer_end") or {}).get("lora_sha256"),
        },
        "reward_trajectory": summary["reward"]["trajectory"],
        "timings_mean_s": summary["timings_mean_s"],
        "steps": rows,
    }


# Integration findings of this row against the sibling engines, with their current status.
INTEGRATION_FINDINGS = [
    {
        "id": "F1",
        "what": "runtime/megatron_lora.lora_sha256 (key\\n + shape\\n + bytes) was not "
        "muse_glimmer_lora_sha256_v1; StrictTrainer.export_lora reported the former and the "
        "bridge failed closed at the initial LoRA sync (reviewer's TP1 smoke: WeightSyncError).",
        "status": "fixed in the merged code: megatron_lora.lora_sha256 delegates to "
        "e2e.sync.lora_sha256",
        "evidence": "hash_schema_check all_equal (this JSON); real_e2e_smoke: trainer_start, "
        "rollout_start and every sync report the same canonical hash in both modes",
        "follow_up": "e2e_bridge:R11 - runtime imports e2e (dependency inversion): move the "
        "hash into a shared module (request to the integrator)",
    },
    {
        "id": "F2",
        "what": "StrictTrainer(strict=False) had no LoRA (step/export_lora raised).",
        "status": "fixed in the merged code (native trainer LoRA hooks)",
        "evidence": "real_e2e_smoke native/native: step ok, export_lora ok, weight sync 2/2",
    },
    {
        "id": "F3",
        "what": "StrictRolloutEngine(strict=False).load_lora raised.",
        "status": "fixed in the merged code (vLLM LoRARequest from a PEFT folder)",
        "evidence": "real_e2e_smoke native/native rollout lora_mode 'vLLM native punica LoRA "
        "(PEFT folder)', weight sync 2/2",
    },
    {
        "id": "F4",
        "what": "StrictRolloutEngine.provenance() had no top-level lora_sha256 and the workers "
        "stored the declared hash.",
        "status": "fixed in the merged code (top-level lora_sha256; None when worker hashes "
        "differ); e2e/sync no longer accepts per-worker values at all",
        "evidence": "real_e2e_smoke: rollout_sha256_source 'provenance.lora_sha256'; "
        "test_rollout_hash_only_from_the_engine_level_provenance_key",
    },
    {
        "id": "F5",
        "what": "StrictTrainer TP worker NCCL group used the default 10 min timeout.",
        "status": "fixed in the merged code (30 min, strict_trainer._init_distributed)",
        "evidence": "code reading",
    },
    {
        "id": "F6",
        "what": "StrictTrainer spawns its own TP workers, so it runs under --trainer-launcher "
        "process; under torchrun it fails closed on len(gpu_ids) == tp.",
        "status": "open (by design for the POC)",
        "evidence": "real_e2e_smoke uses the process launcher with train TP2",
    },
    {
        "id": "F7",
        "what": "StrictTrainer.close() took ~127 s at TP2 in both real smokes (trainer-server "
        "log 'close ... compute=127295 ms' / '127799 ms'): the TP worker join (timeout 120 s) "
        "expires and the worker is terminated.",
        "status": "open (runtime/strict_trainer.py, not this row)",
        "evidence": "real_e2e_smoke[*].trainer_server_log.close",
    },
    {
        "id": "F8",
        "what": "Before e2e_bridge:R3 the trainer server accepted the client before building, "
        "so the 30B trainer build ran after vLLM start-up instead of during it.",
        "status": "fixed in this row (build before accept); NOT re-measured on the 30B",
        "evidence": "real_e2e_smoke strict: server start 20:48:49, client connected 20:49:33, "
        "trainer built 20:52:17 (start-up 209.6 s for a 163.7 s build); startup_overlap rows",
    },
]

STATUS = (
    "bridge + instrumentation complete; review wave B fixes in (rank agreement, start-up "
    "overlap, stateful-failure retire, top-level rollout hash only, lease-relative placement). "
    "Real 30B: two one-step E2E smokes through the bridge by the integrator (real_e2e_smoke); "
    "the four-mode 8-step runs are NOT RUN (integrator)."
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--cpu", action="store_true", help="CPU experiments")
    parser.add_argument("--gpu", action="store_true", help="GPU smoke (run under a lease)")
    parser.add_argument(
        "--real-trainer-smoke",
        action="store_true",
        help="REAL StrictTrainer (30B) behind the bridge on the leased GPUs (score, step, export)",
    )
    parser.add_argument(
        "--hash-check", action="store_true", help="Megatron LoRA hash vs the loop's schema"
    )
    parser.add_argument("--train-tp", type=int, default=1)
    parser.add_argument("--trainer-launcher", choices=bridge.LAUNCHERS, default="process")
    parser.add_argument("--model-path", default="~/models/Muse-Glimmer-30B")
    parser.add_argument("--num-sequences", type=int, default=2)
    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--command", default=None, help="command line to record in the JSON")
    parser.add_argument("--transport-only", action="store_true",
                        help="re-measure only the transport section (CPU)")
    parser.add_argument("--startup-overlap", action="store_true",
                        help="trainer build vs caller work before wait_ready (CPU mock)")
    parser.add_argument("--overlap-label", default="current",
                        help="key under startup_overlap for this measurement")
    parser.add_argument("--import-e2e-run", nargs="*", default=None, metavar="RUN_DIR",
                        help="record real run_poc_grpo run directories under real_e2e_smoke")
    parser.add_argument("--e2e-run-commit", default=None,
                        help="commit the imported runs used (when their summary has none)")
    args = parser.parse_args(argv)
    torch.set_num_threads(4)
    out_path = Path(args.out)
    data: dict[str, Any] = {}
    if out_path.exists():
        data = json.loads(out_path.read_text())
    work = Path(args.work_dir or tempfile.mkdtemp(prefix="mg-bridge-bench-"))
    work.mkdir(parents=True, exist_ok=True)
    data.update({"row": ROW, "commit": _git_head()})
    if args.cpu:
        data["cpu_sections_commit"] = data["commit"]
    mode = "gpu" if (args.gpu or args.real_trainer_smoke) else "cpu"
    data.setdefault("env", {})[f"client_{mode}_run"] = _client_env()
    data["env"][f"client_{mode}_run"]["load_average_1_5_15_at_start"] = list(os.getloadavg())
    data.setdefault("commands", [])
    command = args.command or (
        f"PYTHONPATH=<worktree> {sys.executable} -m rl_engine.models.muse_glimmer.e2e.bench_bridge "
        + " ".join(sys.argv[1:] if argv is None else argv)
    )
    if command not in data["commands"]:
        data["commands"].append(command)
    if args.transport_only:
        data["transport"] = transport(work)
        data["transport"]["commit"] = data["commit"]
    if args.cpu:
        data["surrogate_exactness"] = {"cpu": surrogate_exactness("cpu")}
        data["bridge_equivalence"] = bridge_equivalence(work / "equiv")
        data["transport"] = transport(work)
        data["transport"]["commit"] = data["commit"]
        data["failure_handling"] = failure_handling(work / "faults")
        data["mock_e2e"] = mock_e2e(work / "mock_e2e", steps=4, prompts=4, samples=4,
                                    max_tokens=32)
        megatron = data["bridge_equivalence"][0]
        data["env"]["trainer_server"] = {"python": MEGATRON_PY, "torch": megatron["server_torch"]}
    if args.gpu:
        data["gpu_smoke"] = gpu_smoke(work / "gpu")
        data["gpu_smoke"]["commit"] = data["commit"]
        if data["gpu_smoke"].get("status") == "ran":
            data["env"]["gpu"] = data["gpu_smoke"]["parent_device_name"]
    if args.hash_check:
        data["hash_schema_check"] = hash_schema_check()
    if args.real_trainer_smoke:
        key = f"real_trainer_smoke_tp{args.train_tp}"
        data[key] = real_trainer_smoke(
            work / "real", tp=args.train_tp, launcher=args.trainer_launcher,
            model_path=args.model_path, num_sequences=args.num_sequences,
        )
        data[key]["commit"] = data["commit"]
    if args.startup_overlap:
        entry = startup_overlap(work / "overlap")
        entry["commit"] = data["commit"]
        data.setdefault("startup_overlap", {})[args.overlap_label] = entry
    if args.import_e2e_run:
        data["real_e2e_smoke"] = [
            import_e2e_run(Path(d), commit=args.e2e_run_commit) for d in args.import_e2e_run
        ]
    # Superseded hand-written entries (replaced by measured sections above).
    data.pop("transport_before_tcp_nodelay", None)
    for tp in (1, 2):
        stub = data.get(f"real_trainer_smoke_tp{tp}")
        if isinstance(stub, dict) and stub.get("status") == "NOT RUN":
            data.pop(f"real_trainer_smoke_tp{tp}")
    data["integration_findings"] = INTEGRATION_FINDINGS
    data["status"] = STATUS
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, indent=1, default=str) + "\n", encoding="utf-8")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
