# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Cross-environment trainer bridge of the Muse-Glimmer-30B strict POC (``final_e2e_lora``).

Why: the rollout runs vLLM 0.31 (``~/envs/vllm``, torch cu130) and the trainer runs Megatron
Bridge + TransformerEngine (``~/envs/megatron``, torch cu129). The two torch builds cannot share a
process, so the GRPO orchestrator (vLLM env) drives the trainer in a child process started with
the Megatron env's Python. :class:`RemoteTrainer` is a ``TrainerLike`` (``engines.py``) client;
``trainer_server.py`` is the other end.

Transport (dependency-free): ``multiprocessing.connection`` over ``127.0.0.1`` with a random
32-byte authkey (HMAC challenge both ways; the key travels in the child's environment, never on
the command line). Every message is ONE ``torch.save`` blob, loaded with ``weights_only=True``
(tensors and plain containers only). Tensors are moved to the CPU and serialized as raw bytes,
so fp32 logprobs, fp32 gradients and bf16 LoRA tensors cross the boundary bit-exactly.

Gradient hand-off (the only numerics the bridge touches): ``token_logprobs`` returns fresh CPU
leaf tensors ``lp_i`` (``requires_grad=True``) holding the trainer's fp32 values;
``step(loss)`` runs ``loss.backward()`` in the orchestrator, ships ``g_i = lp_i.grad`` to the
server, and the server calls ``trainer.step(S)`` with the surrogate

    S = sum_i (lp_i_server * g_i).sum()          (:func:`surrogate_loss`)

whose gradient w.r.t. ``lp_i_server`` is ``1.0 * g_i == g_i`` bit for bit (sum backward
broadcasts the exact 1.0; mul backward multiplies it by ``g_i``; IEEE multiplication by 1.0 is
exact for every finite value, signed zeros and subnormals included). The value of ``S`` is not
the GRPO loss and must not be logged as such; ``step`` reports the orchestrator's loss.

Start-up: the server builds the trainer as soon as it starts and accepts the client's
connection in a background thread, so with ``wait=False`` the trainer load (minutes for the 30B)
overlaps whatever the caller does before :meth:`RemoteTrainer.wait_ready` (vLLM start-up in
``run_poc_grpo``). ``hello["startup"]`` records launch -> build start/end -> connect -> ready.

Failure handling: every call has a deadline (``call_timeout_s``; startup has its own). While
waiting, the client polls the socket and the server process; a dead server raises
:class:`RemoteTrainerError` with its exit status and the tail of its log, a deadline raises
:class:`RemoteTrainerTimeout` after killing the server's whole process group (it runs in its own
session, so torchrun ranks and spawned TP workers die with it). The server kills its own group
when the client process disappears. After a transport failure, or after a remote failure of a
stateful op (:data:`STATEFUL_OPS`: scoring, step, LoRA export, which may leave the TP ranks in
different states), the client kills the server group and refuses further calls; a failed
stateless op (``ping``, ``provenance``, an unknown op) leaves the bridge usable.

Rank agreement (torchrun): every rank recomputes the canonical LoRA hash of what it exported and
a sha256 of the logprob bytes it scored; rank 0 gathers them and the call fails closed if any
rank differs (:class:`RemoteWeightSyncError` for the LoRA). The client re-checks the gathered
LoRA hashes against its own recomputation (:func:`check_rank_hashes`).

GPU placement: ``gpu_ids`` are physical GPU ids; the client sets ``CUDA_VISIBLE_DEVICES`` for the
server process from them (``""`` when empty: a CPU-only server cannot touch any GPU) and the
trainer sees local devices ``0..tp-1``. :func:`default_gpu_ids` derives the ids from the lease
when the placement flags are omitted; :func:`plan_gpu_placement` validates them.
"""

from __future__ import annotations

import atexit
import io
import json
import os
import secrets
import signal
import socket
import subprocess
import tempfile
import threading
import time
from multiprocessing.connection import Client, Connection
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from rl_engine.models.muse_glimmer.e2e.sync import WeightSyncError, lora_sha256

PROTOCOL = "muse_glimmer_trainer_bridge_v1"
SERVER_MODULE = "rl_engine.models.muse_glimmer.e2e.trainer_server"
DEFAULT_TRAINER_PYTHON = "~/envs/megatron/bin/python"
DEFAULT_TRAINER_FACTORY = "rl_engine.models.muse_glimmer.runtime.strict_trainer:StrictTrainer"
MOCK_TRAINER_FACTORY = "rl_engine.models.muse_glimmer.e2e.mocks:MockTrainer"
LAUNCHERS = ("process", "torchrun")
# Ops whose remote failure may leave the trainer (or its TP ranks) in an undefined state.
STATEFUL_OPS = frozenset({"token_logprobs", "step", "export_lora"})

AUTHKEY_ENV = "MUSE_GLIMMER_BRIDGE_AUTHKEY"
CLIENT_PID_ENV = "MUSE_GLIMMER_BRIDGE_CLIENT_PID"
CLIENT_PGID_ENV = "MUSE_GLIMMER_BRIDGE_CLIENT_PGID"
# Environment the vLLM side sets for its own workers; never leak it into the trainer process.
_ROLLOUT_ONLY_ENV = ("RL_KERNEL_MUSEGLIMMER_STRICT", "RL_KERNEL_READBACK_DIR")

REPO_ROOT = Path(__file__).resolve().parents[4]


class BridgeError(RuntimeError):
    """Base class of trainer-bridge failures."""


class RemoteTrainerError(BridgeError):
    """The remote trainer raised, died, or the transport broke."""


class RemoteTrainerTimeout(RemoteTrainerError, TimeoutError):
    """A bridge call (or the server startup) exceeded its deadline; the server was killed."""


class RemoteWeightSyncError(RemoteTrainerError, WeightSyncError):
    """The LoRA crossing the bridge failed a hash check (trainer, server, client or a TP rank)."""


def check_rank_hashes(rank_sha256: Any, expected: str, *, world: int) -> None:
    """Every trainer rank must report ``expected`` (the client's own recomputation)."""

    if not isinstance(rank_sha256, list) or len(rank_sha256) != int(world):
        raise RemoteWeightSyncError(
            f"trainer ranks did not report one LoRA hash each (world {world}): {rank_sha256!r}"
        )
    bad = {i: h for i, h in enumerate(rank_sha256) if h != expected}
    if bad:
        raise RemoteWeightSyncError(
            f"trainer ranks hold different LoRA weights: expected {expected} on every rank, "
            f"ranks {sorted(bad)} report {[bad[i] for i in sorted(bad)]}"
        )


# --------------------------------------------------------------------------------------------
# Wire format
# --------------------------------------------------------------------------------------------


def wire_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Detached CPU tensor with its own compact storage (bit-exact copy when a copy is needed)."""

    t = tensor.detach()
    if t.device.type != "cpu":
        t = t.to("cpu")
    compact = (
        t.is_contiguous()
        and t.storage_offset() == 0
        and t.untyped_storage().nbytes() == t.numel() * t.element_size()
    )
    if not compact:
        t = t.clone(memory_format=torch.contiguous_format)
    return t


def to_wire(obj: Any) -> Any:
    """Convert ``obj`` to what ``torch.load(weights_only=True)`` accepts (lossless for tensors)."""

    if obj is None or type(obj) in (bool, int, float, str):
        return obj
    # Subclasses (torch.__version__ is a TorchVersion, enums, numpy floats) are not on the
    # weights_only allow-list: reduce them to the exact builtin type.
    if isinstance(obj, bool):
        return bool(obj)
    if isinstance(obj, int):
        return int(obj)
    if isinstance(obj, float):
        return float(obj)
    if isinstance(obj, str):
        return obj.encode("utf-8", "surrogatepass").decode("utf-8", "surrogatepass")
    if isinstance(obj, torch.Tensor):
        return wire_tensor(obj)
    if isinstance(obj, Mapping):
        return {str(key): to_wire(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_wire(value) for value in obj]
    if isinstance(obj, Path):
        return str(obj)
    item = getattr(obj, "item", None)
    if callable(item):
        try:
            value = item()
        except (TypeError, ValueError, RuntimeError):
            value = None
        if isinstance(value, (bool, int, float)):
            return value
    return str(obj)


def encode_message(obj: Any) -> bytes:
    buffer = io.BytesIO()
    torch.save(obj, buffer)
    return buffer.getvalue()


def decode_message(data: bytes) -> Any:
    return torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)


def tensor_bytes(tensor: torch.Tensor) -> bytes:
    flat = tensor.detach().to("cpu").contiguous().reshape(-1)
    return flat.view(torch.uint8).numpy().tobytes() if flat.numel() else b""


# --------------------------------------------------------------------------------------------
# Gradient hand-off
# --------------------------------------------------------------------------------------------


def surrogate_loss(logprobs: Sequence[torch.Tensor], grads: Sequence[torch.Tensor]) -> torch.Tensor:
    """``sum_i (logprobs[i] * grads[i]).sum()``; d/d ``logprobs[i]`` is exactly ``grads[i]``.

    ``grads[i]`` is moved to ``logprobs[i].device`` (a bit-exact copy). Shapes and dtypes must
    match exactly (no silent cast).
    """

    if len(logprobs) != len(grads):
        raise ValueError(f"{len(grads)} gradients for {len(logprobs)} logprob tensors")
    if not logprobs:
        raise ValueError("no logprob tensors")
    total: torch.Tensor | None = None
    for i, (lp, g) in enumerate(zip(logprobs, grads)):
        if tuple(g.shape) != tuple(lp.shape):
            raise ValueError(f"sample {i}: gradient shape {tuple(g.shape)} != {tuple(lp.shape)}")
        if g.dtype != lp.dtype:
            raise TypeError(f"sample {i}: gradient dtype {g.dtype} != logprob dtype {lp.dtype}")
        if not lp.requires_grad:
            raise RuntimeError(
                f"sample {i}: the trainer's logprobs carry no autograd history; this trainer "
                "cannot take a gradient step (native forward-only trainer?)"
            )
        term = (lp * g.to(device=lp.device)).sum()
        total = term if total is None else total + term
    assert total is not None
    return total


# --------------------------------------------------------------------------------------------
# GPU placement
# --------------------------------------------------------------------------------------------


def parse_visible_devices(value: str | None) -> list[int] | None:
    """``CUDA_VISIBLE_DEVICES`` as ints; ``None`` if unset or not a plain integer list."""

    if value is None:
        return None
    parts = [part.strip() for part in value.split(",") if part.strip()]
    try:
        return [int(part) for part in parts]
    except ValueError:
        return None


def default_gpu_ids(
    *,
    rollout_tp: int,
    train_tp: int,
    parent_visible: str | None,
    rollout_gpus: Sequence[int] | None = None,
    train_gpus: Sequence[int] | None = None,
) -> tuple[list[int], list[int]]:
    """Fill omitted placement flags from the lease: rollout = the first ``rollout_tp`` leased ids,
    trainer = the next ``train_tp`` ids not used by the rollout (lease order).

    ``gpu-lease`` picks physical ids at run time, so ``gpu-lease -n 3 -- run_poc_grpo
    --rollout-tp 1 --train-tp 2`` needs no ids. Given flags are returned unchanged (physical ids;
    :func:`plan_gpu_placement` validates them against the lease).
    """

    rollout = None if rollout_gpus is None else [int(g) for g in rollout_gpus]
    train = None if train_gpus is None else [int(g) for g in train_gpus]
    if rollout is not None and train is not None:
        return rollout, train
    lease = parse_visible_devices(parent_visible)
    if lease is None:
        raise ValueError(
            "no --rollout-gpus/--train-gpus and no GPU lease (CUDA_VISIBLE_DEVICES): run under "
            "~/bin/gpu-lease or pass the physical ids"
        )
    taken = set(rollout or []) | set(train or [])
    free = [g for g in lease if g not in taken]
    need = (rollout_tp if rollout is None else 0) + (train_tp if train is None else 0)
    if len(free) < need:
        raise ValueError(
            f"placement needs {need} GPU(s) from the lease {parent_visible!r} but only "
            f"{len(free)} are free (rollout TP{rollout_tp} + train TP{train_tp})"
        )
    if rollout is None:
        rollout, free = free[:rollout_tp], free[rollout_tp:]
    if train is None:
        train = free[:train_tp]
    return rollout, train


def plan_gpu_placement(
    *,
    rollout_gpus: Sequence[int],
    rollout_tp: int,
    train_gpus: Sequence[int],
    train_tp: int,
    parent_visible: str | None,
) -> dict[str, Any]:
    """Validate the explicit placement flags and return the per-child ``CUDA_VISIBLE_DEVICES``.

    ``rollout_gpus`` / ``train_gpus`` are PHYSICAL GPU ids. They must be one per TP rank,
    disjoint, and inside the parent's ``CUDA_VISIBLE_DEVICES`` (the lease) when that is set.
    """

    rollout = [int(g) for g in rollout_gpus]
    train = [int(g) for g in train_gpus]
    if len(rollout) != rollout_tp:
        raise ValueError(f"--rollout-gpus lists {len(rollout)} GPUs for rollout TP{rollout_tp}")
    if len(train) != train_tp:
        raise ValueError(f"--train-gpus lists {len(train)} GPUs for train TP{train_tp}")
    for label, ids in (("rollout", rollout), ("train", train)):
        if len(set(ids)) != len(ids):
            raise ValueError(f"{label} GPUs repeat an id: {ids}")
    overlap = sorted(set(rollout) & set(train))
    if overlap:
        raise ValueError(f"rollout and train GPUs overlap on {overlap}")
    lease = parse_visible_devices(parent_visible)
    if lease is not None:
        outside = sorted((set(rollout) | set(train)) - set(lease))
        if outside:
            raise ValueError(
                f"GPUs {outside} are outside this process's CUDA_VISIBLE_DEVICES={parent_visible!r}"
                " (lease the whole set for the orchestrator)"
            )
    return {
        "parent_cuda_visible_devices": parent_visible,
        "gpu_lease": os.environ.get("GPU_LEASE"),
        "rollout_gpus": rollout,
        "train_gpus": train,
        "rollout_cuda_visible_devices": ",".join(str(g) for g in rollout),
        "train_cuda_visible_devices": ",".join(str(g) for g in train),
    }


def child_env(gpu_ids: Sequence[int], extra: Mapping[str, str] | None = None) -> dict[str, str]:
    """Environment of a trainer server: this checkout first on PYTHONPATH, GPUs from the flags."""

    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("VLLM_") and key not in _ROLLOUT_ONLY_ENV
    }
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(int(g)) for g in gpu_ids)
    paths = [str(REPO_ROOT)] + [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p]
    env["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(paths))
    env["PYTHONUNBUFFERED"] = "1"
    env.pop(AUTHKEY_ENV, None)
    if extra:
        env.update({str(k): str(v) for k, v in extra.items()})
    return env


def set_nodelay(conn: Connection) -> None:
    """Disable Nagle on a TCP ``Connection`` (both ends call it).

    ``Connection.send_bytes`` writes header and payload separately above 16 KiB; with Nagle +
    delayed ACK on loopback that stalled 16 KiB .. 1 MiB messages by ~40-90 ms per round trip.
    """

    dup = socket.socket(fileno=os.dup(conn.fileno()))
    try:
        dup.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    finally:
        dup.close()


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def server_command(
    python: str, launcher: str, tensor_parallel_size: int, spec_path: Path
) -> list[str]:
    """Command line of the trainer server for ``launcher`` (``process`` or ``torchrun``)."""

    if launcher == "process":
        return [python, "-m", SERVER_MODULE, "--spec", str(spec_path)]
    if launcher == "torchrun":
        return [
            python,
            "-m",
            "torch.distributed.run",
            "--nnodes=1",
            f"--nproc-per-node={int(tensor_parallel_size)}",
            "--master-addr=127.0.0.1",
            f"--master-port={_free_port()}",
            "--monitor-interval=1",
            "--module",
            SERVER_MODULE,
            "--spec",
            str(spec_path),
        ]
    raise ValueError(f"launcher must be one of {LAUNCHERS}, got {launcher!r}")


def _signal_name(code: int | None) -> str:
    if code is None:
        return "running"
    if code < 0:
        try:
            return f"killed by {signal.Signals(-code).name}"
        except ValueError:
            return f"killed by signal {-code}"
    return f"exit code {code}"


# --------------------------------------------------------------------------------------------
# Client
# --------------------------------------------------------------------------------------------


class RemoteTrainer:
    """``TrainerLike`` client of a trainer running in another Python environment.

    Constructor arguments mirror ``StrictTrainer`` (``model_path``, ``tensor_parallel_size``,
    ``gpu_ids``, ``lora_rank``, ``lr``, ``strict``, ``seed``) plus the bridge settings.
    ``gpu_ids`` are PHYSICAL ids (the server's ``CUDA_VISIBLE_DEVICES``); ``[]`` = CPU-only.
    ``wait=False`` returns right after launching so the caller can build the rollout engine
    while the trainer loads; call :meth:`wait_ready` (any trainer call does it implicitly).
    """

    def __init__(
        self,
        model_path: str,
        *,
        tensor_parallel_size: int,
        gpu_ids: Sequence[int],
        lora_rank: int,
        lr: float,
        strict: bool,
        seed: int,
        python: str = DEFAULT_TRAINER_PYTHON,
        launcher: str = "process",
        factory: str = DEFAULT_TRAINER_FACTORY,
        factory_kwargs: Mapping[str, Any] | None = None,
        work_dir: str | Path | None = None,
        startup_timeout_s: float = 3600.0,
        call_timeout_s: float = 3600.0,
        close_timeout_s: float = 300.0,
        dist_init: str = "after_build",
        extra_env: Mapping[str, str] | None = None,
        wait: bool = True,
        tcp_nodelay: bool = True,
    ) -> None:
        if launcher not in LAUNCHERS:
            raise ValueError(f"launcher must be one of {LAUNCHERS}, got {launcher!r}")
        if dist_init not in ("after_build", "before_build"):
            raise ValueError("dist_init must be 'after_build' or 'before_build'")
        self.model_path = os.path.expanduser(str(model_path))
        self.tensor_parallel_size = int(tensor_parallel_size)
        self.gpu_ids = [int(g) for g in gpu_ids]
        if self.gpu_ids and len(self.gpu_ids) != self.tensor_parallel_size:
            raise ValueError(
                f"gpu_ids {self.gpu_ids} must list one GPU per TP rank "
                f"(TP{self.tensor_parallel_size})"
            )
        self.lora_rank = int(lora_rank)
        self.lr = float(lr)
        self.strict = bool(strict)
        self.seed = int(seed)
        self.tcp_nodelay = bool(tcp_nodelay)
        self.python = os.path.expanduser(str(python))
        if not Path(self.python).exists():
            raise FileNotFoundError(f"trainer python {self.python} does not exist")
        self.launcher = launcher
        self.factory = factory
        self.factory_kwargs = dict(factory_kwargs or {})
        self.startup_timeout_s = float(startup_timeout_s)
        self.call_timeout_s = float(call_timeout_s)
        self.close_timeout_s = float(close_timeout_s)
        self.work_dir = (
            Path(work_dir) if work_dir is not None else Path(tempfile.mkdtemp(prefix="mg-bridge-"))
        )
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.work_dir / "trainer_server.log"
        self.spec_path = self.work_dir / "spec.json"
        self.handshake_path = self.work_dir / "handshake.json"
        self.handshake_path.unlink(missing_ok=True)

        self.spec = {
            "protocol": PROTOCOL,
            "factory": factory,
            "factory_kwargs": self.factory_kwargs,
            "model_path": self.model_path,
            "tensor_parallel_size": self.tensor_parallel_size,
            "num_visible_gpus": len(self.gpu_ids),
            "lora_rank": self.lora_rank,
            "lr": self.lr,
            "strict": self.strict,
            "seed": self.seed,
            "launcher": launcher,
            "dist_init": dist_init,
            "handshake_path": str(self.handshake_path),
            "tcp_nodelay": self.tcp_nodelay,
        }
        self.spec_path.write_text(json.dumps(self.spec, indent=2) + "\n", encoding="utf-8")

        self._authkey = secrets.token_bytes(32)
        env = child_env(
            self.gpu_ids,
            {
                AUTHKEY_ENV: self._authkey.hex(),
                CLIENT_PID_ENV: str(os.getpid()),
                CLIENT_PGID_ENV: str(os.getpgrp()),
                **dict(extra_env or {}),
            },
        )
        self.command = server_command(
            self.python, launcher, self.tensor_parallel_size, self.spec_path
        )
        self._log_file = open(self.log_path, "ab")  # noqa: SIM115 - owned until close()
        self._proc = subprocess.Popen(
            self.command,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=self._log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        self._launch_time = time.monotonic()
        self._launch_unix = time.time()
        self._conn: Connection | None = None
        self._hello: dict[str, Any] | None = None
        self._broken: str | None = None
        self._closed = False
        self._call_id = 0
        self._lock = threading.Lock()
        self._last_lps: list[torch.Tensor] | None = None
        self.calls: list[dict[str, Any]] = []
        self._pending_stats: list[dict[str, Any]] = []
        self.last_export: dict[str, Any] | None = None
        atexit.register(self._kill_server)
        if wait:
            self.wait_ready()

    # ---------------------------------------------------------------------------- lifecycle
    @property
    def server_pid(self) -> int:
        return int(self._proc.pid)

    def alive(self) -> bool:
        return self._proc.poll() is None

    def wait_ready(self) -> dict[str, Any]:
        """Block until the server has built the trainer; return its hello (env, versions)."""

        if self._hello is not None:
            return self._hello
        if self._broken:
            raise RemoteTrainerError(f"trainer bridge unusable: {self._broken}")
        called = time.monotonic()
        deadline = self._launch_time + self.startup_timeout_s
        port = self._wait_handshake(deadline)
        try:
            self._conn = Client(("127.0.0.1", port), authkey=self._authkey)
            if self.tcp_nodelay:
                set_nodelay(self._conn)
        except (OSError, EOFError) as exc:
            raise self._dead("connect", exc) from exc
        data = self._recv(deadline, "startup", self.startup_timeout_s)
        reply = decode_message(data)
        if not reply.get("ok"):
            error = reply.get("error") or "unknown error"
            self._broken = "trainer construction failed"
            self._kill_server()
            raise RemoteTrainerError(
                f"trainer server failed to build the trainer ({self.factory}):\n{error}"
            )
        hello = dict(reply["result"])
        ready = time.monotonic()
        hello["startup_s"] = ready - self._launch_time
        stamps = hello.get("server_unix") or {}

        def after_launch(key: str) -> float | None:
            value = stamps.get(key)
            return None if value is None else float(value) - self._launch_unix

        hello["startup"] = {
            "launch_unix": self._launch_unix,
            "server_start_after_launch_s": after_launch("start"),
            "server_build_start_after_launch_s": after_launch("build_start"),
            "server_build_end_after_launch_s": after_launch("build_end"),
            "client_connected_after_launch_s": after_launch("client_connected"),
            "wait_ready_called_after_launch_s": called - self._launch_time,
            "ready_after_launch_s": hello["startup_s"],
            "wait_ready_blocked_s": ready - called,
        }
        if hello.get("protocol") != PROTOCOL:
            self._broken = "protocol mismatch"
            self._kill_server()
            raise RemoteTrainerError(f"server speaks {hello.get('protocol')!r}, not {PROTOCOL}")
        self._hello = hello
        return hello

    def _wait_handshake(self, deadline: float) -> int:
        while True:
            if self.handshake_path.exists():
                try:
                    info = json.loads(self.handshake_path.read_text(encoding="utf-8"))
                    return int(info["port"])
                except (ValueError, KeyError, OSError):
                    pass  # partially visible on an exotic filesystem; retry
            if self._proc.poll() is not None:
                raise self._dead("startup", None)
            if time.monotonic() > deadline:
                raise self._timeout("startup", self.startup_timeout_s)
            time.sleep(0.05)

    def _log_tail(self, lines: int = 40) -> str:
        try:
            self._log_file.flush()
            text = self.log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return "(no server log)"
        return "\n".join(text.splitlines()[-lines:])

    def _kill_server(self) -> None:
        """Terminate the server's whole process group (idempotent; also the atexit hook)."""

        proc = self._proc
        for sig, wait_s in ((signal.SIGTERM, 10.0), (signal.SIGKILL, 10.0)):
            if proc.poll() is None:
                try:
                    os.killpg(proc.pid, sig)
                except (ProcessLookupError, PermissionError):
                    pass
                try:
                    proc.wait(timeout=wait_s)
                except subprocess.TimeoutExpired:
                    continue
        # Stragglers in the server's group (torchrun ranks, spawned TP workers) die with it.
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass

    def _dead(self, op: str, exc: BaseException | None) -> RemoteTrainerError:
        try:
            self._proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            pass
        status = _signal_name(self._proc.poll())
        self._broken = f"server died during {op} ({status})"
        self._kill_server()
        cause = f" ({type(exc).__name__}: {exc})" if exc is not None else ""
        return RemoteTrainerError(
            f"trainer server (pid {self._proc.pid}, {self.launcher}) died during {op!r}: "
            f"{status}{cause}. Log {self.log_path} tail:\n{self._log_tail()}"
        )

    def _timeout(self, op: str, timeout_s: float) -> RemoteTrainerTimeout:
        self._broken = f"{op} timed out after {timeout_s:.1f}s"
        self._kill_server()
        return RemoteTrainerTimeout(
            f"trainer server (pid {self._proc.pid}) did not answer {op!r} within {timeout_s:.1f}s;"
            f" killed its process group. Log {self.log_path} tail:\n{self._log_tail()}"
        )

    def _recv(self, deadline: float, op: str, timeout_s: float) -> bytes:
        assert self._conn is not None
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise self._timeout(op, timeout_s)
            try:
                ready = self._conn.poll(min(0.25, remaining))
            except (OSError, EOFError) as exc:
                raise self._dead(op, exc) from exc
            if ready:
                try:
                    return self._conn.recv_bytes()
                except (OSError, EOFError) as exc:
                    raise self._dead(op, exc) from exc
            if self._proc.poll() is not None:
                raise self._dead(op, None)

    def _call(
        self, op: str, args: Mapping[str, Any], *, timeout_s: float | None = None
    ) -> tuple[Any, dict[str, Any]]:
        if self._closed:
            raise RemoteTrainerError("RemoteTrainer is closed")
        if self._broken:
            raise RemoteTrainerError(
                f"trainer bridge unusable after an earlier failure: {self._broken}"
            )
        self.wait_ready()
        assert self._conn is not None
        timeout = float(timeout_s if timeout_s is not None else self.call_timeout_s)
        with self._lock:
            self._call_id += 1
            call_id = self._call_id
            t0 = time.perf_counter()
            payload = encode_message({"op": op, "id": call_id, "args": to_wire(dict(args))})
            t_enc = time.perf_counter()
            deadline = time.monotonic() + timeout
            # A server that stops reading can block send_bytes on a full socket buffer; the
            # watchdog kills it at the deadline, which turns the blocked send into an error.
            watchdog = threading.Timer(timeout + 1.0, self._kill_server)
            watchdog.daemon = True
            watchdog.start()
            try:
                try:
                    self._conn.send_bytes(payload)
                except (OSError, ValueError) as exc:
                    if time.monotonic() >= deadline:
                        raise self._timeout(op, timeout) from exc
                    raise self._dead(op, exc) from exc
                t_sent = time.perf_counter()
                data = self._recv(deadline, op, timeout)
            finally:
                watchdog.cancel()
            t_recv = time.perf_counter()
            reply = decode_message(data)
            t_dec = time.perf_counter()
        if reply.get("id") != call_id:
            self._broken = "protocol desync"
            self._kill_server()
            raise RemoteTrainerError(f"reply id {reply.get('id')} != request id {call_id} ({op})")
        stats = {
            "op": op,
            "id": call_id,
            "request_bytes": len(payload),
            "response_bytes": len(data),
            "encode_s": t_enc - t0,
            "send_s": t_sent - t_enc,
            "wait_s": t_recv - t_sent,
            "decode_s": t_dec - t_recv,
            "rtt_s": t_dec - t0,
            "server": reply.get("server") or {},
        }
        self.calls.append(stats)
        self._pending_stats.append(stats)
        if not reply.get("ok"):
            error_cls = (
                RemoteWeightSyncError
                if reply.get("error_type") == "WeightSyncError"
                else RemoteTrainerError
            )
            message = f"remote {op} raised:\n{reply.get('error')}"
            if op in STATEFUL_OPS:
                # The trainer (or one TP rank) may have half-applied the op: retire the bridge.
                self._retire(f"remote {op} failed; trainer state is undefined")
                message += "\n(the trainer server was stopped; the bridge refuses further calls)"
            raise error_cls(message)
        return reply.get("result"), stats

    def _retire(self, reason: str) -> None:
        """Mark the bridge unusable and take the server's process group down."""

        self._broken = reason
        self._kill_server()

    def drain_call_stats(self) -> list[dict[str, Any]]:
        """Call stats since the previous drain (the loop stores them per step)."""

        out, self._pending_stats = self._pending_stats, []
        return out

    # ---------------------------------------------------------------------------- TrainerLike
    def token_logprobs(
        self, token_ids: list[list[int]], completion_start: list[int]
    ) -> list[torch.Tensor]:
        args = {
            "token_ids": [[int(t) for t in row] for row in token_ids],
            "completion_start": [int(c) for c in completion_start],
        }
        result, _ = self._call("token_logprobs", args)
        tensors = list(result["logprobs"])
        flags = result.get("requires_grad")
        if not isinstance(flags, list) or len(flags) != len(tensors):
            self._retire("token_logprobs reply without per-tensor requires_grad flags")
            raise RemoteTrainerError(
                f"token_logprobs reply carries requires_grad={flags!r} for {len(tensors)} tensors"
            )
        # Mirror the trainer: a forward-only trainer's logprobs carry no autograd history, so
        # the loop's guard rejects them at scoring time instead of failing later in step().
        out = [t.requires_grad_(bool(flag)) for t, flag in zip(tensors, flags)]
        self._last_lps = out
        return out

    def step(self, loss: torch.Tensor) -> dict[str, Any]:
        lps = self._last_lps
        if lps is None:
            raise RuntimeError("call token_logprobs before step (one step per scoring call)")
        # A step attempt consumes the scored logprobs on both sides, whether or not it succeeds.
        self._last_lps = None
        if not all(lp.requires_grad for lp in lps):
            raise RuntimeError(
                "the remote trainer's logprobs carry no autograd history (forward-only trainer); "
                "it cannot take a gradient step"
            )
        for lp in lps:
            lp.grad = None
        t0 = time.perf_counter()
        loss.backward()
        backward_s = time.perf_counter() - t0
        grads = [
            lp.grad.detach() if lp.grad is not None else torch.zeros_like(lp.detach())
            for lp in lps
        ]
        loss_value = float(loss.detach().item())
        result, stats = self._call("step", {"grads": grads, "loss": loss_value})
        info = dict(result.get("info") or {})
        if "loss" in info:
            info["server_reported_loss"] = info["loss"]  # the surrogate's value, not the loss
        info["loss"] = loss_value
        info["bridge"] = {
            "client_backward_s": backward_s,
            "rtt_s": stats["rtt_s"],
            "server_compute_s": stats["server"].get("compute_s"),
            "surrogate_value": result.get("surrogate_value"),
            "grad_elements": int(sum(g.numel() for g in grads)),
        }
        return info

    def export_lora(self) -> tuple[dict[str, torch.Tensor], str]:
        result, stats = self._call("export_lora", {})
        state_dict = dict(result["state_dict"])
        reported = str(result["sha256"])
        server_sha = str(result["server_sha256"])
        client_sha = lora_sha256(state_dict)
        world = int((self._hello or {}).get("world", 1))
        try:
            if not (reported == server_sha == client_sha):
                raise RemoteWeightSyncError(
                    "LoRA hash mismatch across the trainer bridge: trainer reported "
                    f"{reported}, server recomputed {server_sha}, client recomputed {client_sha}"
                )
            check_rank_hashes(result.get("rank_sha256"), client_sha, world=world)
        except RemoteWeightSyncError:
            self._retire("LoRA export failed hash verification")
            raise
        self.last_export = {
            "sha256": reported,
            "num_tensors": len(state_dict),
            "num_bytes": int(sum(t.numel() * t.element_size() for t in state_dict.values())),
            "rank_sha256": result.get("rank_sha256"),
            "rtt_s": stats["rtt_s"],
        }
        return state_dict, reported

    def provenance(self) -> dict[str, Any]:
        result, _ = self._call("provenance", {})
        out = dict(result or {})
        out["bridge"] = self.bridge_info()
        return out

    def bridge_info(self) -> dict[str, Any]:
        return {
            "protocol": PROTOCOL,
            "client": "RemoteTrainer",
            "launcher": self.launcher,
            "python": self.python,
            "factory": self.factory,
            "factory_kwargs": to_wire(self.factory_kwargs),
            "command": self.command,
            "server_pid": self.server_pid,
            "gpu_ids_physical": list(self.gpu_ids),
            "server_cuda_visible_devices": ",".join(str(g) for g in self.gpu_ids),
            "tcp_nodelay": self.tcp_nodelay,
            "hello": self._hello,
            "calls": len(self.calls),
            "log": str(self.log_path),
        }

    def close(self) -> None:
        if self._closed:
            return
        try:
            if self._conn is not None and not self._broken and self._proc.poll() is None:
                try:
                    self._call("close", {}, timeout_s=self.close_timeout_s)
                except RemoteTrainerError:
                    pass
        finally:
            self._closed = True
            try:
                self._proc.wait(timeout=30.0)
            except subprocess.TimeoutExpired:
                pass
            self._kill_server()
            if self._conn is not None:
                try:
                    self._conn.close()
                except OSError:
                    pass
            try:
                self._log_file.close()
            except OSError:
                pass
            atexit.unregister(self._kill_server)

    def __enter__(self) -> "RemoteTrainer":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ---------------------------------------------------------------------------- extras
    def arm_profile(self, out_dir: str | Path, tag: str, ops: Sequence[str] = ()) -> None:
        """Profile the next ``token_logprobs`` and ``step`` (default) on the server."""

        ops = list(ops) or ["token_logprobs", "step"]
        self._call("profile_arm", {"out_dir": str(out_dir), "tag": tag, "ops": ops})

    def ping(self, payload: torch.Tensor | None = None) -> tuple[torch.Tensor | None, dict]:
        """Echo ``payload`` through the server (transport check / bandwidth benchmark)."""

        result, stats = self._call("ping", {"payload": payload})
        return result.get("payload"), stats
