import sys
from pathlib import Path

p = Path(sys.argv[1]) / "rl_engine/models/muse_glimmer/e2e/trainer_server.py"
s = p.read_text()


def rep(old, new, count=1):
    global s
    assert s.count(old) == count, (s.count(old), old[:80])
    s = s.replace(old, new)


rep('''Protocol: the client writes ``spec.json`` and passes the authkey in the environment; rank 0
listens on an ephemeral 127.0.0.1 port, publishes it in the handshake file (atomic rename), sends
a hello once the trainer is built (or the traceback if building failed), then serves requests
``{"op", "id", "args"}`` until ``close`` or EOF. Ops: ``token_logprobs``, ``step`` (surrogate
gradient hand-off, see ``bridge.surrogate_loss``), ``export_lora``, ``provenance``,
``profile_arm``, ``ping``, ``close``.
"""''', '''Protocol: the client writes ``spec.json`` and passes the authkey in the environment; rank 0
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
"""''')

rep('''import argparse
import importlib
import json''', '''import argparse
import hashlib
import importlib
import json''')

rep('''def _memory_gib() -> dict[str, float] | None:''', '''def _logprob_digest(tensors: list[torch.Tensor]) -> str:
    """sha256 over (dtype, shape, raw bytes) of each logprob tensor, in order."""

    digest = hashlib.sha256()
    for tensor in tensors:
        cpu = wire_tensor(tensor)
        shape = ",".join(str(int(d)) for d in cpu.shape)
        digest.update(f"{cpu.dtype}|{shape}\\n".encode("utf-8"))
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


def _memory_gib() -> dict[str, float] | None:''')

rep('''        self.last_logprobs = logprobs
        if self.rank != 0:
            return None
        return {''', '''        self.last_logprobs = logprobs
        self._rank_extra = {"logprobs_sha256": _logprob_digest(logprobs)}
        if self.rank != 0:
            return None
        return {''')

rep('''    def dispatch(self, op: str, args: Mapping[str, Any]) -> tuple[bool, Any, str | None, dict]:
        self._rank_extra = {}
        self.op_counts[op] = self.op_counts.get(op, 0) + 1
        handler = self.OPS.get(op)
        status: dict[str, Any] = {"rank": self.rank, "compute_s": 0.0, "memory": None}
        if handler is None:
            status.update(ok=False, error=f"unknown op {op!r}")
            return False, None, status["error"], status''', '''    def dispatch(self, op: str, args: Mapping[str, Any]) -> tuple[bool, Any, str | None, dict]:
        self._rank_extra = {}
        self.op_counts[op] = self.op_counts.get(op, 0) + 1
        handler = self.OPS.get(op)
        status: dict[str, Any] = {"rank": self.rank, "compute_s": 0.0, "memory": None}
        if handler is None:
            status.update(ok=False, error=f"unknown op {op!r}", error_type="KeyError")
            return False, None, status["error"], status''')

rep('''        prof: dict[str, Any] = {}
        t0 = time.perf_counter()
        try:
            with profiled(profile_this, out_path=out_path, label=f"trainer_server.{op}") as prof:
                result = handler(self, args)
            ok, error = True, None
        except Exception:  # noqa: BLE001 - reported to the client verbatim
            result, ok, error = None, False, traceback.format_exc()''', '''        prof: dict[str, Any] = {}
        error_type = None
        t0 = time.perf_counter()
        try:
            with profiled(profile_this, out_path=out_path, label=f"trainer_server.{op}") as prof:
                result = handler(self, args)
            ok, error = True, None
        except Exception as exc:  # noqa: BLE001 - reported to the client verbatim
            result, ok, error = None, False, traceback.format_exc()
            error_type = type(exc).__name__''')

rep('''        status["ok"] = ok
        status["error"] = error
        return ok, result, error, status''', '''        status["ok"] = ok
        status["error"] = error
        status["error_type"] = error_type
        return ok, result, error, status''')

rep('''                errors = [
                    f"rank {s['rank']}: {s.get('error')}" for s in statuses if not s.get("ok")
                ]
                all_ok = ok and not errors
                server = {''', '''                errors = [
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
                server = {''')

rep('''                if op == "export_lora" and self.world > 1 and all_ok:
                    result["rank_sha256"] = [s.get("sha256") for s in statuses]
                reply = {
                    "id": message.get("id"),
                    "ok": all_ok,
                    "result": result if all_ok else None,
                    "error": "\\n".join(errors) if errors else None,
                    "server": to_wire(server),
                }''', '''                if op == "export_lora" and all_ok:
                    result["rank_sha256"] = [s.get("sha256") for s in statuses]
                if op == "token_logprobs" and all_ok:
                    result["rank_logprobs_sha256"] = [s.get("logprobs_sha256") for s in statuses]
                reply = {
                    "id": message.get("id"),
                    "ok": all_ok,
                    "result": result if all_ok else None,
                    "error": "\\n".join(errors) if errors else None,
                    "error_type": next((t for t in error_types if t), None) if errors else None,
                    "server": to_wire(server),
                }''')

rep('''    authkey_hex = os.environ.pop(AUTHKEY_ENV, "")
    _start_client_watchdog(rank)
    _log(rank, f"start pid={os.getpid()} world={world} launcher={spec.get('launcher')} "
               f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r}")

    conn: Connection | None = None
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
        conn = listener.accept()
        set_nodelay(conn)
        listener.close()
        _log(rank, "client connected")

    t0 = time.perf_counter()
    try:''', '''    authkey_hex = os.environ.pop(AUTHKEY_ENV, "")
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
    try:''')

rep('''            ctrl = dist.new_group(backend="gloo", timeout=_LONG)
    except BaseException:  # noqa: BLE001 - shipped to the client, then exit non-zero
        tb = traceback.format_exc()
        _log(rank, f"trainer construction failed:\\n{tb}")
        if conn is not None:
            try:
                conn.send_bytes(encode_message({"id": 0, "ok": False, "error": tb}))
            except OSError:
                pass
        return 1
    build_s = time.perf_counter() - t0
    _log(rank, f"trainer {type(trainer).__name__} built in {build_s:.1f} s")
    if conn is not None:
        hello = {''', '''            ctrl = dist.new_group(backend="gloo", timeout=_LONG)
    except BaseException:  # noqa: BLE001 - shipped to the client, then exit non-zero
        build_error = traceback.format_exc()
        _log(rank, f"trainer construction failed:\\n{build_error}")
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
        hello = {''')

rep('''            "build_s": build_s,
            "versions": _versions(),''', '''            "build_s": build_s,
            "server_unix": dict(stamps),
            "tcp_nodelay": nodelay,
            "versions": _versions(),''')
p.write_text(s)
print("patched", p)
