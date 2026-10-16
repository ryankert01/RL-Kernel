# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Trainer server of the cross-environment bridge (runs with the Megatron env's Python).

Started by :class:`rl_engine.models.muse_glimmer.e2e.bridge.RemoteTrainer`; never run by hand
except for debugging. ``torch`` is imported first (TransformerEngine must come after it).

Launchers:

* ``process`` (one process): the trainer manages its own TP ranks, as today's ``StrictTrainer``
  does (rank 0 in this process, ranks 1..tp-1 spawned by the trainer). The trainer gets
  ``gpu_ids = [0 .. tp-1]`` (local ids under the ``CUDA_VISIBLE_DEVICES`` the client set).
* ``torchrun`` (``nproc = tp``): every rank builds its own trainer (``gpu_ids = [LOCAL_RANK]``,
  ``tensor_parallel_size = tp``); rank 0 is the RPC endpoint and broadcasts every request on a
  gloo control group (``broadcast_object_list``), so all ranks execute the same trainer method
  in the same order; replies come from rank 0, per-rank status (ok / error / peak memory / op
  counters) is gathered to rank 0 and returned with the reply.

Protocol: the client writes ``spec.json`` and passes the authkey in the environment; rank 0
listens on an ephemeral 127.0.0.1 port, publishes it in the handshake file (atomic rename) and
accepts the client in a background thread while EVERY rank builds its trainer right away (the
build overlaps the orchestrator's vLLM start-up; the client only connects in ``wait_ready``).
Once the trainer is built and the client is connected, rank 0 sends a hello (or the traceback if
building failed), then serves requests ``{"op", "id", "args"}`` until ``close`` or EOF. Ops:
``token_logprobs``, ``step`` (surrogate gradient hand-off, see ``bridge.surrogate_loss``),
``export_lora``, ``provenance``, ``profile_arm``, ``ping``, ``close``.

Rank agreement (``torchrun``): ``export_lora`` reports every rank's canonical LoRA hash
(``rank_sha256``) and ``token_logprobs`` every rank's sha256 of its logprob bytes; if any rank
differs from rank 0 the reply is an error (``error_type`` ``WeightSyncError`` /
``RankDivergenceError``), never a silent rank-0 answer.
"""

from __future__ import annotations

import torch  # noqa: I001 - must precede anything that imports transformer_engine

import argparse
import hashlib
import importlib
import json
import os
import signal
import sys
import threading
import time
import traceback
from datetime import timedelta
from importlib import metadata
from multiprocessing.connection import Connection, Listener
from pathlib import Path
from typing import Any, Mapping

import torch.distributed as dist

from rl_engine.models.muse_glimmer.e2e.bridge import (
    AUTHKEY_ENV,
    CLIENT_PGID_ENV,
    CLIENT_PID_ENV,
    PROTOCOL,
    decode_message,
    encode_message,
    set_nodelay,
    surrogate_loss,
    to_wire,
    wire_tensor,
)
from rl_engine.models.muse_glimmer.e2e.sync import lora_sha256
from rl_engine.models.muse_glimmer.e2e.timeline import profiled

_LONG = timedelta(days=1)


def build_trainer(spec: Mapping[str, Any], *, rank: int, world: int, local_rank: int) -> Any:
    """THE place the trainer is constructed. Adapt here when the trainer's constructor changes.

    Uses only the ``StrictTrainer`` constructor surface: ``(model_path, *, tensor_parallel_size,
    gpu_ids, lora_rank, lr, strict, seed, **factory_kwargs)``. Under ``process`` the trainer owns
    all TP ranks (``gpu_ids = 0..n-1``); under ``torchrun`` each rank passes only its own GPU
    (``[LOCAL_RANK]``), so a trainer that still spawns its own TP workers fails closed on its
    ``len(gpu_ids) == tensor_parallel_size`` check instead of double-spawning.
    """

    module_name, _, attr = str(spec["factory"]).partition(":")
    trainer_cls = getattr(importlib.import_module(module_name), attr)
    visible = int(spec.get("num_visible_gpus", 0))
    if world > 1:
        gpu_ids = [local_rank] if visible else []
    else:
        gpu_ids = list(range(visible))
    kwargs: dict[str, Any] = {
        "tensor_parallel_size": int(spec["tensor_parallel_size"]),
        "gpu_ids": gpu_ids,
        "lora_rank": int(spec["lora_rank"]),
        "lr": float(spec["lr"]),
        "strict": bool(spec["strict"]),
        "seed": int(spec["seed"]),
    }
    kwargs.update(spec.get("factory_kwargs") or {})
    return trainer_cls(spec["model_path"], **kwargs)


# --------------------------------------------------------------------------------------------
# Process plumbing
# --------------------------------------------------------------------------------------------


def _log(rank: int, message: str) -> None:
    print(f"[trainer_server rank{rank} {time.strftime('%H:%M:%S')}] {message}", flush=True)


def _start_client_watchdog(rank: int) -> None:
    """Kill this server's process group when the client process disappears (no orphans)."""

    pid = int(os.environ.get(CLIENT_PID_ENV, "0") or 0)
    client_pgid = int(os.environ.get(CLIENT_PGID_ENV, "0") or 0)
    if pid <= 0:
        return

    def watch() -> None:
        while True:
            time.sleep(1.0)
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                _log(rank, f"client pid {pid} is gone; terminating the server process group")
                own = os.getpgrp()
                if own != client_pgid and own == os.getsid(0):
                    os.killpg(own, signal.SIGKILL)
                os._exit(3)
            except PermissionError:
                pass

    threading.Thread(target=watch, name="bridge-client-watchdog", daemon=True).start()


def _versions() -> dict[str, Any]:
    out: dict[str, Any] = {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
    }
    for dist_name in ("triton", "megatron-core", "megatron-bridge", "transformer_engine",
                      "transformers", "vllm"):
        try:
            out[dist_name] = metadata.version(dist_name)
        except metadata.PackageNotFoundError:
            out[dist_name] = None
    return out


def _device_info() -> dict[str, Any]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    info: dict[str, Any] = {"cuda_visible_devices": visible, "devices": []}
    if visible == "":
        info["device_count"] = 0
        return info
    count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    info["device_count"] = count
    for i in range(count):
        props = torch.cuda.get_device_properties(i)
        info["devices"].append(
            {
                "index": i,
                "name": props.name,
                "uuid": str(getattr(props, "uuid", "")),
                "total_memory_gib": props.total_memory / 2**30,
                "pci_bus_id": getattr(props, "pci_bus_id", None),
            }
        )
    return info


def _cuda_active() -> bool:
    return bool(torch.cuda.is_available() and torch.cuda.is_initialized())


def _logprob_digest(tensors: list[torch.Tensor]) -> str:
    """sha256 over (dtype, shape, raw bytes) of each logprob tensor, in order."""

    digest = hashlib.sha256()
    for tensor in tensors:
        cpu = wire_tensor(tensor)
        shape = ",".join(str(int(d)) for d in cpu.shape)
        digest.update(f"{cpu.dtype}|{shape}\n".encode("utf-8"))
        flat = cpu.contiguous().reshape(-1)
        if flat.numel():
            digest.update(flat.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _rank_disagreement(statuses: list[dict[str, Any]], key: str) -> str | None:
    """A description of the ranks whose ``key`` differs from rank 0's (None if all agree)."""

    values = [s.get(key) for s in statuses]
    if values and all(v is not None and v == values[0] for v in values):
        return None
    return ", ".join(f"rank {s.get('rank')}: {s.get(key)}" for s in statuses)


def _memory_gib() -> dict[str, float] | None:
    if not _cuda_active():
        return None
    return {
        "allocated_gib": torch.cuda.memory_allocated() / 2**30,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "reserved_gib": torch.cuda.memory_reserved() / 2**30,
    }


# --------------------------------------------------------------------------------------------
# Server
# --------------------------------------------------------------------------------------------


class TrainerServer:
    """Executes bridge requests on one rank; rank 0 also talks to the client."""

    def __init__(
        self,
        trainer: Any,
        *,
        rank: int,
        world: int,
        conn: Connection | None,
        ctrl: Any,
        spec: Mapping[str, Any],
    ) -> None:
        self.trainer = trainer
        self.rank = rank
        self.world = world
        self.conn = conn
        self.ctrl = ctrl
        self.spec = dict(spec)
        self.last_logprobs: list[torch.Tensor] | None = None
        self.op_counts: dict[str, int] = {}
        self.profile_armed: dict[str, Any] | None = None
        self._rank_extra: dict[str, Any] = {}

    # ------------------------------------------------------------------ ops (every rank)
    def op_token_logprobs(self, args: Mapping[str, Any]) -> Any:
        token_ids = [[int(t) for t in row] for row in args["token_ids"]]
        starts = [int(c) for c in args["completion_start"]]
        logprobs = list(self.trainer.token_logprobs(token_ids, starts))
        if len(logprobs) != len(token_ids):
            raise ValueError(f"trainer returned {len(logprobs)} tensors for {len(token_ids)} rows")
        self.last_logprobs = logprobs
        self._rank_extra = {"logprobs_sha256": _logprob_digest(logprobs)}
        if self.rank != 0:
            return None
        return {
            "logprobs": [wire_tensor(t) for t in logprobs],
            "requires_grad": [bool(t.requires_grad) for t in logprobs],
            "device": str(logprobs[0].device) if logprobs else None,
        }

    def op_step(self, args: Mapping[str, Any]) -> Any:
        logprobs = self.last_logprobs
        if logprobs is None:
            raise RuntimeError("step without a preceding token_logprobs on this server")
        self.last_logprobs = None  # consumed by this attempt, success or not
        surrogate = surrogate_loss(logprobs, list(args["grads"]))
        info = self.trainer.step(surrogate)
        if self.rank != 0:
            return None
        return {"info": to_wire(info or {}), "surrogate_value": float(surrogate.detach().item())}

    def op_export_lora(self, args: Mapping[str, Any]) -> Any:
        state_dict, sha = self.trainer.export_lora()
        cpu = {str(name): wire_tensor(t) for name, t in state_dict.items()}
        server_sha = lora_sha256(cpu)
        self._rank_extra = {"sha256": server_sha}
        if self.rank != 0:
            return None
        return {
            "state_dict": cpu,
            "sha256": str(sha),
            "server_sha256": server_sha,
            "num_tensors": len(cpu),
            "num_bytes": int(sum(t.numel() * t.element_size() for t in cpu.values())),
        }

    def op_provenance(self, args: Mapping[str, Any]) -> Any:
        prov = self.trainer.provenance()
        return to_wire(prov) if self.rank == 0 else None

    def op_profile_arm(self, args: Mapping[str, Any]) -> Any:
        self.profile_armed = {
            "out_dir": str(args["out_dir"]),
            "tag": str(args["tag"]),
            "ops": [str(op) for op in args.get("ops") or ["token_logprobs", "step"]],
        }
        return {"armed": self.profile_armed} if self.rank == 0 else None

    def op_ping(self, args: Mapping[str, Any]) -> Any:
        return {"payload": args.get("payload")} if self.rank == 0 else None

    def op_close(self, args: Mapping[str, Any]) -> Any:
        self.trainer.close()
        return {"closed": True} if self.rank == 0 else None

    OPS = {
        "token_logprobs": op_token_logprobs,
        "step": op_step,
        "export_lora": op_export_lora,
        "provenance": op_provenance,
        "profile_arm": op_profile_arm,
        "ping": op_ping,
        "close": op_close,
    }

    # ------------------------------------------------------------------ dispatch
    def dispatch(self, op: str, args: Mapping[str, Any]) -> tuple[bool, Any, str | None, dict]:
        self._rank_extra = {}
        self.op_counts[op] = self.op_counts.get(op, 0) + 1
        handler = self.OPS.get(op)
        status: dict[str, Any] = {"rank": self.rank, "compute_s": 0.0, "memory": None}
        if handler is None:
            status.update(ok=False, error=f"unknown op {op!r}", error_type="KeyError")
            return False, None, status["error"], status
        armed = self.profile_armed
        profile_this = armed is not None and op in armed["ops"]
        out_path = None
        if profile_this:
            out_path = Path(armed["out_dir"]) / (
                f"trainer_server_{armed['tag']}_{op}_rank{self.rank}.json"
            )
        if _cuda_active():
            torch.cuda.reset_peak_memory_stats()
        prof: dict[str, Any] = {}
        error_type = None
        t0 = time.perf_counter()
        try:
            with profiled(profile_this, out_path=out_path, label=f"trainer_server.{op}") as prof:
                result = handler(self, args)
            ok, error = True, None
        except Exception as exc:  # noqa: BLE001 - reported to the client verbatim
            result, ok, error = None, False, traceback.format_exc()
            error_type = type(exc).__name__
        if _cuda_active():
            torch.cuda.synchronize()
        status["compute_s"] = time.perf_counter() - t0
        status["memory"] = _memory_gib()
        status["op_counts"] = dict(self.op_counts)
        status.update(self._rank_extra)
        if profile_this:
            status["profile"] = dict(prof)
            assert armed is not None
            armed["ops"] = [o for o in armed["ops"] if o != op]
            if not armed["ops"]:
                self.profile_armed = None
        status["ok"] = ok
        status["error"] = error
        status["error_type"] = error_type
        return ok, result, error, status

    def _broadcast(self, message: Any) -> Any:
        box = [message]
        dist.broadcast_object_list(box, src=0, group=self.ctrl)
        return box[0]

    def _gather(self, status: dict[str, Any]) -> list[dict[str, Any]] | None:
        out: list[Any] | None = [None] * self.world if self.rank == 0 else None
        dist.gather_object(status, out, dst=0, group=self.ctrl)
        return out

    def serve(self) -> None:
        while True:
            eof = False
            message: Any = None
            if self.rank == 0:
                assert self.conn is not None
                try:
                    message = decode_message(self.conn.recv_bytes())
                except (EOFError, OSError):
                    eof = True
                    message = {"op": "close", "id": -1, "args": {}}
                    _log(self.rank, "client connection closed; shutting down")
            if self.world > 1:
                message = self._broadcast(message)
            op = str(message["op"])
            ok, result, error, status = self.dispatch(op, message.get("args") or {})
            statuses = self._gather(status) if self.world > 1 else [status]
            if self.rank == 0:
                assert statuses is not None
                errors = [
                    f"rank {s['rank']}: {s.get('error')}" for s in statuses if not s.get("ok")
                ]
                error_types = [s.get("error_type") for s in statuses if not s.get("ok")]
                if not errors and op == "export_lora":
                    differ = _rank_disagreement(statuses, "sha256")
                    if differ is not None:
                        errors.append(f"trainer ranks hold different LoRA weights ({differ})")
                        error_types.append("WeightSyncError")
                if not errors and op == "token_logprobs":
                    differ = _rank_disagreement(statuses, "logprobs_sha256")
                    if differ is not None:
                        errors.append(f"trainer ranks returned different logprobs ({differ})")
                        error_types.append("RankDivergenceError")
                all_ok = ok and not errors
                server = {
                    "compute_s": status["compute_s"],
                    "ranks": statuses,
                    "world": self.world,
                    "memory": status["memory"],
                    "profile": status.get("profile"),
                }
                if op == "export_lora" and all_ok:
                    result["rank_sha256"] = [s.get("sha256") for s in statuses]
                if op == "token_logprobs" and all_ok:
                    result["rank_logprobs_sha256"] = [s.get("logprobs_sha256") for s in statuses]
                reply = {
                    "id": message.get("id"),
                    "ok": all_ok,
                    "result": result if all_ok else None,
                    "error": "\n".join(errors) if errors else None,
                    "error_type": next((t for t in error_types if t), None) if errors else None,
                    "server": to_wire(server),
                }
                if not eof:
                    try:
                        self.conn.send_bytes(encode_message(reply))  # type: ignore[union-attr]
                    except OSError:
                        _log(self.rank, "client vanished while replying; shutting down")
                        if op != "close":
                            self._shutdown_after_client_loss()
                        return
                _log(
                    self.rank,
                    f"{op} id={message.get('id')} ok={all_ok} "
                    f"compute={status['compute_s'] * 1e3:.1f} ms",
                )
            if op == "close":
                return

    def _shutdown_after_client_loss(self) -> None:
        try:
            self.trainer.close()
        except Exception:  # noqa: BLE001
            pass


def _init_default_group(world: int, local_rank: int) -> None:
    if dist.is_initialized() or world <= 1:
        return
    use_cuda = os.environ.get("CUDA_VISIBLE_DEVICES", "") != "" and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl" if use_cuda else "gloo", timeout=_LONG)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, help="spec.json written by RemoteTrainer")
    args = parser.parse_args(argv)
    spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    authkey_hex = os.environ.pop(AUTHKEY_ENV, "")
    stamps: dict[str, float] = {"start": time.time()}
    _start_client_watchdog(rank)
    nodelay = bool(spec.get("tcp_nodelay", True))
    _log(rank, f"start pid={os.getpid()} world={world} launcher={spec.get('launcher')} "
               f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r} "
               f"tcp_nodelay={nodelay}")

    accepted: dict[str, Any] = {}
    accept_thread: threading.Thread | None = None
    if rank == 0:
        if not authkey_hex:
            _log(rank, f"missing {AUTHKEY_ENV}; refusing to listen without authentication")
            return 2
        listener = Listener(("127.0.0.1", 0), authkey=bytes.fromhex(authkey_hex))
        handshake = Path(spec["handshake_path"])
        tmp = handshake.with_suffix(".tmp")
        tmp.write_text(
            json.dumps({"port": listener.address[1], "pid": os.getpid(), "protocol": PROTOCOL}),
            encoding="utf-8",
        )
        tmp.replace(handshake)

        def accept() -> None:
            # The client connects only in wait_ready(); the trainer builds meanwhile.
            try:
                client = listener.accept()
                if nodelay:
                    set_nodelay(client)
                accepted["conn"] = client
                stamps["client_connected"] = time.time()
                _log(rank, "client connected")
            except BaseException as exc:  # noqa: BLE001 - reported after the build
                accepted["error"] = exc
            finally:
                listener.close()

        accept_thread = threading.Thread(target=accept, name="bridge-accept", daemon=True)
        accept_thread.start()

    t0 = time.perf_counter()
    stamps["build_start"] = time.time()
    build_error: str | None = None
    trainer: Any = None
    ctrl = None
    try:
        if world > 1 and spec.get("dist_init") == "before_build":
            _init_default_group(world, local_rank)
        elif world > 1 and os.environ.get("CUDA_VISIBLE_DEVICES", "") != "":
            if torch.cuda.is_available():
                torch.cuda.set_device(local_rank)
        trainer = build_trainer(spec, rank=rank, world=world, local_rank=local_rank)
        ctrl = None
        if world > 1:
            if not dist.is_initialized():
                dist.init_process_group(backend="gloo", timeout=_LONG)
            ctrl = dist.new_group(backend="gloo", timeout=_LONG)
    except BaseException:  # noqa: BLE001 - shipped to the client, then exit non-zero
        build_error = traceback.format_exc()
        _log(rank, f"trainer construction failed:\n{build_error}")
    build_s = time.perf_counter() - t0
    stamps["build_end"] = time.time()

    conn: Connection | None = None
    if accept_thread is not None:
        accept_thread.join()  # until the client calls wait_ready (or dies: the watchdog exits)
        if "error" in accepted:
            exc = accepted["error"]
            _log(rank, f"accepting the client failed: {type(exc).__name__}: {exc}")
            return 2
        conn = accepted["conn"]
    if build_error is not None:
        if conn is not None:
            try:
                conn.send_bytes(encode_message({"id": 0, "ok": False, "error": build_error}))
            except OSError:
                pass
        return 1
    _log(rank, f"trainer {type(trainer).__name__} built in {build_s:.1f} s")
    if conn is not None:
        hello = {
            "protocol": PROTOCOL,
            "pid": os.getpid(),
            "world": world,
            "launcher": spec.get("launcher"),
            "trainer_class": f"{type(trainer).__module__}.{type(trainer).__qualname__}",
            "build_s": build_s,
            "server_unix": dict(stamps),
            "tcp_nodelay": nodelay,
            "versions": _versions(),
            "devices": _device_info(),
            "memory": _memory_gib(),
        }
        conn.send_bytes(encode_message({"id": 0, "ok": True, "result": to_wire(hello)}))
    server = TrainerServer(trainer, rank=rank, world=world, conn=conn, ctrl=ctrl, spec=spec)
    server.serve()
    if world > 1 and dist.is_initialized():
        try:
            dist.barrier(group=ctrl)
            dist.destroy_process_group()
        except Exception:  # noqa: BLE001 - best effort at shutdown
            pass
    if conn is not None:
        conn.close()
    _log(rank, "bye")
    return 0


if __name__ == "__main__":
    sys.exit(main())
