import sys
from pathlib import Path

p = Path(sys.argv[1]) / "rl_engine/models/muse_glimmer/e2e/bridge.py"
s = p.read_text()


def rep(old, new, count=1):
    global s
    assert s.count(old) == count, (s.count(old), old[:80])
    s = s.replace(old, new)


rep('''Failure handling: every call has a deadline (``call_timeout_s``; startup has its own). While
waiting, the client polls the socket and the server process; a dead server raises
:class:`RemoteTrainerError` with its exit status and the tail of its log, a deadline raises
:class:`RemoteTrainerTimeout` after killing the server's whole process group (it runs in its own
session, so torchrun ranks and spawned TP workers die with it). The server kills its own group
when the client process disappears. After a transport failure the client refuses further calls.

GPU placement: ``gpu_ids`` are physical GPU ids; the client sets ``CUDA_VISIBLE_DEVICES`` for the
server process from them (``""`` when empty: a CPU-only server cannot touch any GPU) and the
trainer sees local devices ``0..tp-1``. :func:`plan_gpu_placement` validates the orchestrator's
``--rollout-gpus`` / ``--train-gpus`` against the lease.
"""''', '''Start-up: the server builds the trainer as soon as it starts and accepts the client's
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
"""''')

rep('''LAUNCHERS = ("process", "torchrun")
''', '''LAUNCHERS = ("process", "torchrun")
# Ops whose remote failure may leave the trainer (or its TP ranks) in an undefined state.
STATEFUL_OPS = frozenset({"token_logprobs", "step", "export_lora"})
''')

rep('''class RemoteTrainerTimeout(RemoteTrainerError, TimeoutError):
    """A bridge call (or the server startup) exceeded its deadline; the server was killed."""
''', '''class RemoteTrainerTimeout(RemoteTrainerError, TimeoutError):
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
''')

rep('''def plan_gpu_placement(''', '''def default_gpu_ids(
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


def plan_gpu_placement(''')

rep('''        dist_init: str = "after_build",
        extra_env: Mapping[str, str] | None = None,
        wait: bool = True,
    ) -> None:''', '''        dist_init: str = "after_build",
        extra_env: Mapping[str, str] | None = None,
        wait: bool = True,
        tcp_nodelay: bool = True,
    ) -> None:''')

rep('''        self.python = os.path.expanduser(str(python))''', '''        self.tcp_nodelay = bool(tcp_nodelay)
        self.python = os.path.expanduser(str(python))''')

rep('''            "dist_init": dist_init,
            "handshake_path": str(self.handshake_path),
        }''', '''            "dist_init": dist_init,
            "handshake_path": str(self.handshake_path),
            "tcp_nodelay": self.tcp_nodelay,
        }''')

rep('''        self._launch_time = time.monotonic()''', '''        self._launch_time = time.monotonic()
        self._launch_unix = time.time()''')

rep('''        if self._broken:
            raise RemoteTrainerError(f"trainer bridge unusable: {self._broken}")
        deadline = self._launch_time + self.startup_timeout_s
        port = self._wait_handshake(deadline)
        try:
            self._conn = Client(("127.0.0.1", port), authkey=self._authkey)
            set_nodelay(self._conn)
        except (OSError, EOFError) as exc:
            raise self._dead("connect", exc) from exc''', '''        if self._broken:
            raise RemoteTrainerError(f"trainer bridge unusable: {self._broken}")
        called = time.monotonic()
        deadline = self._launch_time + self.startup_timeout_s
        port = self._wait_handshake(deadline)
        try:
            self._conn = Client(("127.0.0.1", port), authkey=self._authkey)
            if self.tcp_nodelay:
                set_nodelay(self._conn)
        except (OSError, EOFError) as exc:
            raise self._dead("connect", exc) from exc''')

rep('''        hello = dict(reply["result"])
        hello["startup_s"] = time.monotonic() - self._launch_time''', '''        hello = dict(reply["result"])
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
        }''')

rep('''        self.calls.append(stats)
        self._pending_stats.append(stats)
        if not reply.get("ok"):
            raise RemoteTrainerError(f"remote {op} raised:\\n{reply.get('error')}")
        return reply.get("result"), stats''', '''        self.calls.append(stats)
        self._pending_stats.append(stats)
        if not reply.get("ok"):
            error_cls = (
                RemoteWeightSyncError
                if reply.get("error_type") == "WeightSyncError"
                else RemoteTrainerError
            )
            message = f"remote {op} raised:\\n{reply.get('error')}"
            if op in STATEFUL_OPS:
                # The trainer (or one TP rank) may have half-applied the op: retire the bridge.
                self._retire(f"remote {op} failed; trainer state is undefined")
                message += "\\n(the trainer server was stopped; the bridge refuses further calls)"
            raise error_cls(message)
        return reply.get("result"), stats

    def _retire(self, reason: str) -> None:
        """Mark the bridge unusable and take the server's process group down."""

        self._broken = reason
        self._kill_server()''')

rep('''        result, _ = self._call("token_logprobs", args)
        tensors = list(result["logprobs"])
        out = [t.requires_grad_(True) for t in tensors]
        self._last_lps = out
        return out''', '''        result, _ = self._call("token_logprobs", args)
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
        return out''')

rep('''        # A step attempt consumes the scored logprobs on both sides, whether or not it succeeds.
        self._last_lps = None
        for lp in lps:''', '''        # A step attempt consumes the scored logprobs on both sides, whether or not it succeeds.
        self._last_lps = None
        if not all(lp.requires_grad for lp in lps):
            raise RuntimeError(
                "the remote trainer's logprobs carry no autograd history (forward-only trainer); "
                "it cannot take a gradient step"
            )
        for lp in lps:''')

rep('''        client_sha = lora_sha256(state_dict)
        if not (reported == server_sha == client_sha):
            raise WeightSyncError(
                "LoRA hash mismatch across the trainer bridge: trainer reported "
                f"{reported}, server recomputed {server_sha}, client recomputed {client_sha}"
            )
        self.last_export = {''', '''        client_sha = lora_sha256(state_dict)
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
        self.last_export = {''')

rep('''            "server_cuda_visible_devices": ",".join(str(g) for g in self.gpu_ids),
            "hello": self._hello,''', '''            "server_cuda_visible_devices": ",".join(str(g) for g in self.gpu_ids),
            "tcp_nodelay": self.tcp_nodelay,
            "hello": self._hello,''')
p.write_text(s)
print("patched", p)
