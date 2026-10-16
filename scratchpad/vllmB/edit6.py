from pathlib import Path

p = Path("/home/ubuntu/RL-Kernel/.claude/worktrees/wf_65f5032c-579-2/rl_engine/models/muse_glimmer/runtime/strict_rollout_engine.py")
s = p.read_text()

old = '''* ``load_lora(state_dict, sha256)`` -> verifies ``sha256`` against
  ``e2e.sync.lora_sha256`` (schema ``muse_glimmer_lora_sha256_v1``) and the key schema
  (``validate_lora_state_dict``), then installs the adapter (strict: unmerged strict slots on
  every TP rank, which re-hash what they received; native: vLLM's own LoRA from a PEFT folder).
* ``provenance()`` -> dict with ``lora_sha256`` (``None`` before the first load) plus engine,
  worker and timing records.
* ``close()``.

``strict=True`` turns on the RFC #501 binding (``RL_KERNEL_MUSEGLIMMER_STRICT=1`` is inherited by
the vLLM processes through the ``vllm.general_plugins`` entry point) and selects the strict
attention backend; ``strict=False`` is plain native vLLM.  Both run Model Runner V2 (the 0.31 CUDA
default) with ``enforce_eager=True`` and prefix caching off.
"""'''
new = '''* ``load_lora(state_dict, sha256)`` -> verifies ``sha256`` against
  ``e2e.sync.lora_sha256`` (schema ``muse_glimmer_lora_sha256_v1``) and the key schema
  (``validate_lora_state_dict``), then installs the adapter (strict: unmerged strict slots on
  every TP rank; native: vLLM's own LoRA from a PEFT folder, force-loaded on every rank).
* ``provenance()`` -> dict whose ``lora_sha256`` is the hash RECOMPUTED by every TP worker on the
  tensors it actually loaded (strict: the full tensors the rank read and sharded; native: the
  tensors vLLM's LoRA manager holds, PEFT names mapped back); ``None`` before the first load.
  The per-worker values are in ``lora_sha256_per_worker``; if the workers disagree with each
  other or with the driver's own hash of what it shipped, ``provenance()`` raises
  ``WeightSyncError`` (never an echo of the declared string).
* ``close()``.

``strict=True`` turns on the RFC #501 binding (``RL_KERNEL_MUSEGLIMMER_STRICT=1`` is inherited by
the vLLM processes through the ``vllm.general_plugins`` entry point) and selects the strict
attention backend; ``strict=False`` is plain native vLLM.  Both run Model Runner V2 (the 0.31 CUDA
default) with prefix caching off; strict requires ``enforce_eager=True`` (the default; native may
pass ``enforce_eager=False`` for a CUDA-graph baseline).  Every worker must import ``rl_engine``
from the same file as the driver (a stale editable install fails the constructor).
"""'''
assert old in s
s = s.replace(old, new, 1)

old = '''def _fp32_float(value: float) -> float:'''
new = '''def aggregate_worker_lora_hashes(
    reports: list[dict[str, Any]], *, key: str, driver_sha256: str | None
) -> tuple[str | None, list[Any]]:
    """The one LoRA hash every TP worker recomputed, or raise ``WeightSyncError``.

    ``reports`` are per-worker dicts carrying the worker-side recomputed hash under ``key``.
    Fails if a worker errored, if the workers disagree, or if their common value is not the
    driver's own hash of what it shipped (``driver_sha256``; ``None`` = no adapter loaded).
    """

    if not reports:
        raise WeightSyncError("no TP worker reported a LoRA hash (fail closed)")
    errors = [r for r in reports if "error" in r]
    if errors:
        raise WeightSyncError(f"LoRA hash RPC failed on {len(errors)} worker(s): {errors}")
    values = [r.get(key) for r in reports]
    if len(set(values)) != 1:
        raise WeightSyncError(f"TP workers recomputed different LoRA hashes: {values}")
    value = values[0]
    if value != driver_sha256:
        raise WeightSyncError(
            f"TP workers hold LoRA {value!r} but the driver loaded {driver_sha256!r}"
        )
    return value, values


def check_worker_paths(driver_file: str | None, worker_files: list[Any]) -> None:
    """Fail unless every worker imported ``rl_engine`` from the driver's file."""

    bad = [f for f in worker_files if f != driver_file]
    if driver_file is None or bad:
        raise RuntimeError(
            f"vLLM workers import rl_engine from {worker_files}, the driver from {driver_file!r} "
            "(stale editable install or PYTHONPATH not inherited; fail closed)"
        )


def _driver_rl_engine_file() -> str | None:
    try:
        import rl_engine

        return str(Path(rl_engine.__file__).resolve())
    except Exception:  # noqa: BLE001
        return None


def _fp32_float(value: float) -> float:'''
assert old in s
s = s.replace(old, new, 1)

old = '''        max_lora_rank: int = 16,
        enable_native_lora: bool = True,
    ) -> None:'''
new = '''        max_lora_rank: int = 16,
        enable_native_lora: bool = True,
        enforce_eager: bool = True,
        max_num_batched_tokens: int | None = None,
    ) -> None:'''
assert old in s
s = s.replace(old, new, 1)
old = '''        self._native_lora: Any = None
        self._native_lora_dir: str | None = None'''
new = '''        self._native_lora: Any = None
        self._native_lora_dir: str | None = None
        self._native_lora_reports: list[dict[str, Any]] = []
        if self.strict and not enforce_eager:
            raise ValueError("the strict binding requires enforce_eager=True")'''
assert old in s
s = s.replace(old, new, 1)
old = '''            enforce_eager=True,
            gpu_memory_utilization=float(gpu_memory_utilization),'''
new = '''            enforce_eager=bool(enforce_eager),
            gpu_memory_utilization=float(gpu_memory_utilization),'''
assert old in s
s = s.replace(old, new, 1)
old = '''        if self.strict:
            kwargs["attention_config"] = {"backend": "FLASH_ATTN"}
        if self.enable_native_lora:'''
new = '''        if max_num_batched_tokens is not None:
            kwargs["max_num_batched_tokens"] = int(max_num_batched_tokens)
        if self.strict:
            kwargs["attention_config"] = {"backend": "FLASH_ATTN"}
        if self.enable_native_lora:'''
assert old in s
s = s.replace(old, new, 1)
old = '''        self._worker_provenance: list[dict[str, Any]] = []
        if self.strict:
            self._assert_strict_bound()
'''
new = '''        self._worker_provenance: list[dict[str, Any]] = []
        self.rl_engine_file = _driver_rl_engine_file()
        self._check_worker_paths()
        if self.strict:
            self._assert_strict_bound()

    def _check_worker_paths(self) -> None:
        from rl_engine.models.muse_glimmer.runtime.vllm_worker_tools import worker_rl_engine_file

        files = list(self.llm.collective_rpc(worker_rl_engine_file))
        if len(files) != self.tensor_parallel_size:
            raise RuntimeError(f"{len(files)} workers answered, expected {self.tensor_parallel_size}")
        check_worker_paths(self.rl_engine_file, files)
        self.worker_rl_engine_files = files
'''
assert old in s
s = s.replace(old, new, 1)

# strict load_lora replies
old = '''            if len(replies) != self.tensor_parallel_size:
                raise RuntimeError("load_lora: not every TP rank answered")
            for reply in replies:
                if reply.get("sha256") != actual:
                    raise WeightSyncError(f"load_lora: rank reply {reply} does not hold {actual}")
            self._lora_replies = list(replies)'''
new = '''            if len(replies) != self.tensor_parallel_size:
                raise RuntimeError("load_lora: not every TP rank answered")
            aggregate_worker_lora_hashes(
                list(replies), key="recomputed_sha256", driver_sha256=actual
            )
            self._lora_replies = list(replies)'''
assert old in s
s = s.replace(old, new, 1)

# native load: force-load on workers and recompute there
old = '''        root = Path(tempfile.mkdtemp(prefix="museglimmer-native-lora-"))
        folder = write_peft_adapter(state_dict, root / sha256[:16])
        self._native_lora_dir = str(folder)
        self._lora_int_id = getattr(self, "_lora_int_id", 0) + 1
        self._native_lora = LoRARequest(
            f"museglimmer-{sha256[:12]}", self._lora_int_id, str(folder)
        )
'''
new = '''        from rl_engine.models.muse_glimmer.runtime.vllm_worker_tools import (
            worker_native_lora_report,
        )

        root = Path(tempfile.mkdtemp(prefix="museglimmer-native-lora-"))
        folder = write_peft_adapter(state_dict, root / sha256[:16])
        self._lora_int_id = getattr(self, "_lora_int_id", 0) + 1
        request = LoRARequest(f"museglimmer-{sha256[:12]}", self._lora_int_id, str(folder))
        # Load it on every rank now (vLLM would otherwise load lazily at the first request) and
        # recompute the canonical hash from what vLLM's LoRA manager holds.
        if not self.llm.llm_engine.add_lora(request):
            raise WeightSyncError(f"vLLM refused to add LoRA {request.lora_name}")
        reports = list(self.llm.collective_rpc(worker_native_lora_report, args=(request.lora_int_id,)))
        aggregate_worker_lora_hashes(reports, key="recomputed_sha256", driver_sha256=sha256)
        self._native_lora_reports = reports
        self._native_lora_dir = str(folder)
        self._native_lora = request
'''
assert old in s
s = s.replace(old, new, 1)

old = '''    def clear_lora(self) -> None:
        """Drop the adapter (strict: empty slots; native: no LoRA request)."""

        if self.strict:
            from rl_engine.models.muse_glimmer.runtime.vllm_strict import worker_load_lora_file

            self.llm.collective_rpc(worker_load_lora_file, args=("", ""))
        self._native_lora = None
        self._lora_sha256 = None'''
new = '''    def clear_lora(self) -> None:
        """Drop the adapter (strict: empty slots; native: no LoRA request)."""

        if self.strict:
            from rl_engine.models.muse_glimmer.runtime.vllm_strict import worker_load_lora_file

            self.llm.collective_rpc(worker_load_lora_file, args=("", ""))
        self._native_lora = None
        self._native_lora_reports = []
        self._lora_replies = []
        self._lora_sha256 = None'''
assert old in s
s = s.replace(old, new, 1)

old = '''    def provenance(self) -> dict[str, Any]:
        workers: list[dict[str, Any]] = []
        if self.strict and getattr(self, "llm", None) is not None:
            from rl_engine.models.muse_glimmer.runtime.vllm_strict import worker_provenance

            try:
                workers = list(self.llm.collective_rpc(worker_provenance))
            except Exception as exc:  # noqa: BLE001
                workers = [{"error": f"{type(exc).__name__}: {exc}"}]
        worker_hashes = sorted({str(w.get("lora_sha256")) for w in workers if "error" not in w})
        if self.strict and workers and worker_hashes != [str(self._lora_sha256)]:
            lora_hash: str | None = None  # ranks disagree with the driver: report no identity
        else:
            lora_hash = self._lora_sha256
'''
new = '''    def provenance(self) -> dict[str, Any]:
        """Engine provenance; ``lora_sha256`` is the TP workers' recomputed hash (see module doc).

        Raises ``WeightSyncError`` when the workers' recomputed hashes disagree (with each other or
        with what the driver shipped) and ``RuntimeError`` when the engine is closed.
        """

        if getattr(self, "llm", None) is None:
            raise RuntimeError("provenance() on a closed StrictRolloutEngine")
        workers: list[dict[str, Any]] = []
        if self.strict:
            from rl_engine.models.muse_glimmer.runtime.vllm_strict import worker_provenance

            workers = list(self.llm.collective_rpc(worker_provenance))
            lora_hash, per_worker = aggregate_worker_lora_hashes(
                workers, key="lora_sha256", driver_sha256=self._lora_sha256
            )
            source = (
                f"recomputed by each of the {len(workers)} TP workers over the tensors it read "
                "and installed (strict unmerged slots); all agree"
            )
        elif self._native_lora is not None:
            from rl_engine.models.muse_glimmer.runtime.vllm_worker_tools import (
                worker_native_lora_report,
            )

            reports = list(
                self.llm.collective_rpc(
                    worker_native_lora_report, args=(self._native_lora.lora_int_id,)
                )
            )
            lora_hash, per_worker = aggregate_worker_lora_hashes(
                reports, key="recomputed_sha256", driver_sha256=self._lora_sha256
            )
            self._native_lora_reports = reports
            source = (
                f"recomputed by each of the {len(reports)} TP workers from the LoRA tensors "
                "vLLM's LoRA manager holds (PEFT names mapped back); all agree"
            )
        else:
            lora_hash, per_worker = None, []
            source = "no adapter loaded"
'''
assert old in s
s = s.replace(old, new, 1)
old = '''            "lora_sha256": lora_hash,
            "lora_hash_schema": LORA_SHA256_SCHEMA,'''
new = '''            "lora_sha256": lora_hash,
            "lora_sha256_source": source,
            "lora_sha256_per_worker": per_worker,
            "lora_sha256_driver": self._lora_sha256,
            "lora_hash_schema": LORA_SHA256_SCHEMA,'''
assert old in s
s = s.replace(old, new, 1)
old = '''            "native_lora_dir": self._native_lora_dir,
            "lora_replies": list(self._lora_replies),'''
new = '''            "native_lora_dir": self._native_lora_dir,
            "native_lora_reports": list(self._native_lora_reports),
            "lora_replies": list(self._lora_replies),
            "rl_engine_file": self.rl_engine_file,
            "worker_rl_engine_files": list(getattr(self, "worker_rl_engine_files", [])),'''
assert old in s
s = s.replace(old, new, 1)
s = s.replace('''__all__ = ["RolloutSample", "StrictRolloutEngine"]''', '''__all__ = [
    "RolloutSample",
    "StrictRolloutEngine",
    "aggregate_worker_lora_hashes",
    "check_worker_paths",
]''', 1)
p.write_text(s)
print("ok")
