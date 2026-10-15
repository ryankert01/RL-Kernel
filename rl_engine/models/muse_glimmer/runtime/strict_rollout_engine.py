# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""``StrictRolloutEngine``: the E2E orchestrator's handle on a vLLM 0.31 Muse Glimmer rollout.

It implements ``rl_engine.models.muse_glimmer.e2e.engines.RolloutEngineLike`` exactly:

* ``generate(prompt_token_ids, *, n, max_tokens, temperature)`` -> one list of ``n``
  :class:`~rl_engine.models.muse_glimmer.e2e.engines.RolloutSample` per prompt.  ``logprobs`` are
  the fp32 selected-token log-probabilities of the completion tokens at temperature 1 from the raw
  (soft-capped, never temperature/top-k processed) logits, as Python floats exactly representable
  in fp32.  ``strict=True``: the strict O7 value (vocab tiles + fixed tree); ``strict=False``:
  vLLM's native ``raw_logprobs``.
* ``logprobs(token_ids, completion_start)`` -> one prefill per sequence (``prompt_logprobs=0``),
  returning ``len(ids) - completion_start`` fp32 values per sequence.
* ``load_lora(state_dict, sha256)`` -> verifies ``sha256`` against
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
"""

from __future__ import annotations

import gc
import json
import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from rl_engine.models.muse_glimmer.e2e.engines import RolloutSample
from rl_engine.models.muse_glimmer.e2e.sync import (
    LORA_SHA256_SCHEMA,
    WeightSyncError,
    lora_sha256,
    validate_lora_state_dict,
)
from rl_engine.models.muse_glimmer.loader import NATIVE_MODEL_CLASSES

STRICT_ENV = "RL_KERNEL_MUSEGLIMMER_STRICT"
READBACK_ENV = "RL_KERNEL_READBACK_DIR"
_SEED_STRIDE = 1_000_003


@dataclass
class _Timing:
    generate_seconds: float = 0.0
    generated_tokens: int = 0
    prompt_tokens: int = 0
    logprobs_seconds: float = 0.0
    logprobs_tokens: int = 0
    load_seconds: float = 0.0
    lora_seconds: float = 0.0
    history: list[dict[str, Any]] = field(default_factory=list)


def aggregate_worker_lora_hashes(
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


def _fp32_float(value: float) -> float:
    """Round a Python float to the nearest fp32 value (idempotent on fp32 values)."""

    return float(torch.tensor(float(value), dtype=torch.float32).item())


class StrictRolloutEngine:
    """vLLM 0.31 rollout engine for Muse-Glimmer-30B (strict binding or native)."""

    def __init__(
        self,
        model_path: str,
        *,
        tensor_parallel_size: int,
        gpu_ids: list[int],
        strict: bool,
        max_model_len: int,
        seed: int,
        max_num_seqs: int | None = None,
        gpu_memory_utilization: float | None = None,
        max_lora_rank: int = 16,
        enable_native_lora: bool = True,
        enforce_eager: bool = True,
        max_num_batched_tokens: int | None = None,
    ) -> None:
        self.model_path = str(model_path)
        self.tensor_parallel_size = int(tensor_parallel_size)
        self.gpu_ids = [int(g) for g in gpu_ids]
        self.strict = bool(strict)
        self.max_model_len = int(max_model_len)
        self.seed = int(seed)
        self.max_lora_rank = int(max_lora_rank)
        self.timing = _Timing()
        self._generate_calls = 0
        self._lora_sha256: str | None = None
        self._lora_loads = 0
        self._lora_replies: list[dict[str, Any]] = []
        self._native_lora: Any = None
        self._native_lora_dir: str | None = None
        self._native_lora_reports: list[dict[str, Any]] = []
        if self.strict and not enforce_eager:
            raise ValueError("the strict binding requires enforce_eager=True")
        self._readback_dir = tempfile.mkdtemp(prefix="museglimmer-vllm-readback-")
        os.environ[READBACK_ENV] = self._readback_dir
        # collective_rpc ships module-level callables (cloudpickle by reference) to the workers.
        os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
        if self.strict:
            os.environ[STRICT_ENV] = "1"
        else:
            os.environ.pop(STRICT_ENV, None)
        if self.gpu_ids and "CUDA_VISIBLE_DEVICES" not in os.environ:
            # A gpu-lease already pins CUDA_VISIBLE_DEVICES; honour it and only fill the gap.
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in self.gpu_ids)
        if len(self.gpu_ids) < self.tensor_parallel_size:
            raise ValueError("gpu_ids must provide at least tensor_parallel_size devices")

        from vllm import LLM

        if max_num_seqs is None:
            max_num_seqs = int(os.getenv("RL_KERNEL_MUSEGLIMMER_MAX_NUM_SEQS", "64"))
        if gpu_memory_utilization is None:
            gpu_memory_utilization = float(os.getenv("RL_KERNEL_MUSEGLIMMER_GPU_UTIL", "0.90"))
        self.enable_native_lora = bool(enable_native_lora) and not self.strict
        kwargs: dict[str, Any] = dict(
            model=self.model_path,
            tensor_parallel_size=self.tensor_parallel_size,
            dtype="bfloat16",
            max_model_len=self.max_model_len,
            seed=self.seed,
            enforce_eager=bool(enforce_eager),
            gpu_memory_utilization=float(gpu_memory_utilization),
            enable_prefix_caching=False,
            limit_mm_per_prompt={"image": 0, "video": 0},
            disable_log_stats=True,
            max_num_seqs=int(max_num_seqs),
            trust_remote_code=False,
        )
        if max_num_batched_tokens is not None:
            kwargs["max_num_batched_tokens"] = int(max_num_batched_tokens)
        if self.strict:
            kwargs["attention_config"] = {"backend": "FLASH_ATTN"}
        if self.enable_native_lora:
            kwargs.update(enable_lora=True, max_lora_rank=self.max_lora_rank, max_loras=1)
        self.engine_kwargs = {k: v for k, v in kwargs.items() if k != "model"}
        start = time.time()
        self.llm = LLM(**kwargs)
        self.timing.load_seconds = time.time() - start
        self._worker_provenance: list[dict[str, Any]] = []
        self.rl_engine_file = _driver_rl_engine_file()
        self._check_worker_paths()
        if self.strict:
            self._assert_strict_bound()

    def _check_worker_paths(self) -> None:
        from rl_engine.models.muse_glimmer.runtime.vllm_worker_tools import worker_rl_engine_file

        files = list(self.llm.collective_rpc(worker_rl_engine_file))
        if len(files) != self.tensor_parallel_size:
            raise RuntimeError(
                f"{len(files)} workers answered, expected {self.tensor_parallel_size}"
            )
        check_worker_paths(self.rl_engine_file, files)
        self.worker_rl_engine_files = files

    # ------------------------------------------------------------------------------------------
    def _assert_strict_bound(self) -> None:
        from rl_engine.models.muse_glimmer.runtime.vllm_strict import worker_provenance

        provenances = self.llm.collective_rpc(worker_provenance)
        if not provenances or len(provenances) != self.tensor_parallel_size:
            raise RuntimeError(
                f"strict rollout: {len(provenances or [])} workers answered the provenance RPC, "
                f"expected {self.tensor_parallel_size}"
            )
        for prov in provenances:
            if prov.get("model_class") not in NATIVE_MODEL_CLASSES:
                raise RuntimeError(
                    f"strict rollout: worker bound model class {prov.get('model_class')!r}, "
                    "not a native Muse Glimmer class (fail closed)"
                )
            if not prov.get("hooks"):
                raise RuntimeError("strict rollout: worker reports no installed hooks")
            if prov.get("checks", {}).get("check_runtime_binding") != "passed":
                raise RuntimeError("strict rollout: check_runtime_binding did not pass")
            if not prov.get("all_ops_strict") and os.getenv(
                "RL_KERNEL_MUSEGLIMMER_ALLOW_FALLBACK", ""
            ).strip() not in ("1", "true", "yes"):
                raise RuntimeError(
                    f"strict rollout: rank {prov.get('tp_rank')} runs fallbacks "
                    f"{prov.get('fallbacks')} (fail closed)"
                )
        self._worker_provenance = list(provenances)

    def _lora_request(self) -> Any:
        return self._native_lora if (self.enable_native_lora and self._native_lora) else None

    # ------------------------------------------------------------------------------------------
    def generate(
        self,
        prompt_token_ids: list[list[int]],
        *,
        n: int,
        max_tokens: int,
        temperature: float,
    ) -> list[list[RolloutSample]]:
        """Sample ``n`` completions per prompt; logprobs are fp32 at temperature 1 (raw logits)."""

        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt

        if int(n) <= 0 or int(max_tokens) <= 0:
            raise ValueError("n and max_tokens must be positive")
        call = self._generate_calls
        self._generate_calls += 1
        params = [
            SamplingParams(
                n=int(n),
                max_tokens=int(max_tokens),
                temperature=float(temperature),
                logprobs=0,
                # Reproducible and distinct per (engine seed, call, prompt); vLLM adds the sample
                # index for n > 1.
                seed=(self.seed * _SEED_STRIDE + call * 65_537 + i * int(n)) % (2**63 - 1),
                detokenize=False,
                skip_special_tokens=False,
            )
            for i in range(len(prompt_token_ids))
        ]
        prompts = [TokensPrompt(prompt_token_ids=[int(t) for t in ids]) for ids in prompt_token_ids]
        start = time.time()
        outputs = self.llm.generate(
            prompts, params, use_tqdm=False, lora_request=self._lora_request()
        )
        elapsed = time.time() - start
        results: list[list[RolloutSample]] = []
        generated = 0
        for request in outputs:
            samples = []
            completions = sorted(request.outputs, key=lambda c: c.index)
            for completion in completions:
                token_ids = [int(t) for t in completion.token_ids]
                logprob_dicts = completion.logprobs or []
                if len(logprob_dicts) != len(token_ids):
                    raise RuntimeError(
                        f"vLLM returned {len(logprob_dicts)} logprob entries for "
                        f"{len(token_ids)} tokens"
                    )
                logprobs = [
                    _fp32_float(entry[token].logprob)
                    for token, entry in zip(token_ids, logprob_dicts, strict=True)
                ]
                generated += len(token_ids)
                samples.append(
                    RolloutSample(
                        token_ids=token_ids,
                        logprobs=logprobs,
                        finish_reason=str(completion.finish_reason),
                    )
                )
            if len(samples) != int(n):
                raise RuntimeError(f"vLLM returned {len(samples)} samples, expected {n}")
            results.append(samples)
        if len(results) != len(prompt_token_ids):
            raise RuntimeError("vLLM returned a different number of requests than prompts")
        prompt_tokens = sum(len(p) for p in prompt_token_ids)
        self.timing.generate_seconds += elapsed
        self.timing.generated_tokens += generated
        self.timing.prompt_tokens += prompt_tokens
        self.timing.history.append(
            {
                "op": "generate",
                "seconds": elapsed,
                "prompts": len(prompt_token_ids),
                "n": int(n),
                "max_tokens": int(max_tokens),
                "temperature": float(temperature),
                "generated_tokens": generated,
                "prompt_tokens": prompt_tokens,
                "tokens_per_s": (generated / elapsed) if elapsed > 0 else None,
                "lora_sha256": self._lora_sha256,
            }
        )
        return results

    # ------------------------------------------------------------------------------------------
    def logprobs(
        self, token_ids: list[list[int]], completion_start: list[int]
    ) -> list[list[float]]:
        """fp32 log p(token_i | tokens_<i) for i in [completion_start, len) via one prefill."""

        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt

        if len(token_ids) != len(completion_start):
            raise ValueError("token_ids and completion_start must have the same length")
        for ids, begin in zip(token_ids, completion_start, strict=True):
            if not 1 <= int(begin) <= len(ids):
                raise ValueError(f"completion_start {begin} outside [1, {len(ids)}]")
            if len(ids) + 1 > self.max_model_len:
                raise ValueError(f"sequence of {len(ids)} tokens exceeds max_model_len")
        params = SamplingParams(
            max_tokens=1,
            temperature=0.0,
            prompt_logprobs=0,
            logprobs=0,
            detokenize=False,
            skip_special_tokens=False,
        )
        prompts = [TokensPrompt(prompt_token_ids=[int(t) for t in ids]) for ids in token_ids]
        start = time.time()
        outputs = self.llm.generate(
            prompts, params, use_tqdm=False, lora_request=self._lora_request()
        )
        elapsed = time.time() - start
        results: list[list[float]] = []
        scored = 0
        for ids, begin, request in zip(token_ids, completion_start, outputs, strict=True):
            prompt_logprobs = request.prompt_logprobs
            if prompt_logprobs is None or len(prompt_logprobs) != len(ids):
                raise RuntimeError("vLLM returned no/short prompt_logprobs")
            values = []
            for position in range(int(begin), len(ids)):
                entry = prompt_logprobs[position]
                if entry is None:
                    raise RuntimeError(f"no prompt logprob at position {position}")
                values.append(_fp32_float(entry[int(ids[position])].logprob))
            scored += len(values)
            results.append(values)
        self.timing.logprobs_seconds += elapsed
        self.timing.logprobs_tokens += scored
        self.timing.history.append(
            {
                "op": "logprobs",
                "seconds": elapsed,
                "sequences": len(token_ids),
                "scored_tokens": scored,
                "sequence_tokens": sum(len(ids) for ids in token_ids),
                "tokens_per_s": (sum(len(ids) for ids in token_ids) / elapsed) if elapsed else None,
                "lora_sha256": self._lora_sha256,
            }
        )
        return results

    # ------------------------------------------------------------------------------------------
    def load_lora(self, state_dict: dict[str, torch.Tensor], sha256: str) -> None:
        """Install a LoRA adapter after verifying its canonical hash (fail closed)."""

        start = time.time()
        cpu_state = {
            k: v.detach().to("cpu").contiguous() for k, v in dict(state_dict).items()
        }
        actual = lora_sha256(cpu_state)
        if actual != str(sha256):
            raise WeightSyncError(
                f"load_lora: declared sha256 {sha256} != {LORA_SHA256_SCHEMA} hash {actual}"
            )
        summary = validate_lora_state_dict(cpu_state)
        if max(summary["ranks"]) > self.max_lora_rank and not self.strict:
            raise ValueError(f"LoRA rank {summary['ranks']} exceeds max_lora_rank")
        if self.strict:
            from rl_engine.models.muse_glimmer.runtime.vllm_strict import worker_load_lora_file

            # Tensors nested in collective_rpc arguments do not survive vLLM's msgpack channel;
            # ship them through a node-local file (every rank re-hashes what it read).
            path = Path(self._readback_dir) / f"lora-{actual[:16]}.pt"
            torch.save(cpu_state, path)
            try:
                replies = self.llm.collective_rpc(worker_load_lora_file, args=(str(path), actual))
            finally:
                path.unlink(missing_ok=True)
            if len(replies) != self.tensor_parallel_size:
                raise RuntimeError("load_lora: not every TP rank answered")
            aggregate_worker_lora_hashes(
                list(replies), key="recomputed_sha256", driver_sha256=actual
            )
            self._lora_replies = list(replies)
        else:
            self._load_native_lora(cpu_state, actual)
        self._lora_sha256 = actual
        self._lora_loads += 1
        self.timing.lora_seconds += time.time() - start
        self.timing.history.append(
            {"op": "load_lora", "seconds": time.time() - start, "sha256": actual, **summary}
        )

    def clear_lora(self) -> None:
        """Drop the adapter (strict: empty slots; native: no LoRA request)."""

        if self.strict:
            from rl_engine.models.muse_glimmer.runtime.vllm_strict import worker_load_lora_file

            self.llm.collective_rpc(worker_load_lora_file, args=("", ""))
        self._native_lora = None
        self._native_lora_reports = []
        self._lora_replies = []
        self._lora_sha256 = None

    def _load_native_lora(self, state_dict: dict[str, torch.Tensor], sha256: str) -> None:
        """Native mode: vLLM's own (punica) LoRA from a PEFT folder written from the state dict."""

        if not self.enable_native_lora:
            raise RuntimeError("native engine was built with enable_native_lora=False")
        from vllm.lora.request import LoRARequest

        from rl_engine.models.muse_glimmer.runtime.vllm_native_lora import write_peft_adapter

        from rl_engine.models.muse_glimmer.runtime.vllm_worker_tools import (
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
        reports = list(
            self.llm.collective_rpc(worker_native_lora_report, args=(request.lora_int_id,))
        )
        aggregate_worker_lora_hashes(reports, key="recomputed_sha256", driver_sha256=sha256)
        self._native_lora_reports = reports
        self._native_lora_dir = str(folder)
        self._native_lora = request

    # ------------------------------------------------------------------------------------------
    def provenance(self) -> dict[str, Any]:
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
        files = []
        for path in sorted(Path(self._readback_dir).glob("*.json")):
            try:
                files.append(json.loads(path.read_text(encoding="utf-8")))
            except Exception:  # noqa: BLE001
                continue
        try:
            import vllm

            vllm_version = str(vllm.__version__)
        except Exception:  # noqa: BLE001
            vllm_version = "unknown"
        return {
            "engine": "StrictRolloutEngine",
            "runtime": f"vllm {vllm_version}",
            "strict": self.strict,
            "lora_sha256": lora_hash,
            "lora_sha256_source": source,
            "lora_sha256_per_worker": per_worker,
            "lora_sha256_driver": self._lora_sha256,
            "lora_hash_schema": LORA_SHA256_SCHEMA,
            "lora_loads": self._lora_loads,
            "lora_mode": (
                "strict unmerged: y = bf16(acc32(W x) + sum_k B[:,k] bf16(acc32(A x))[k])"
                if self.strict
                else "vLLM native punica LoRA (PEFT folder)"
            ),
            "logprob_temperature": 1.0,
            "logprob_path": (
                "strict O7: fp32 vocab-tile logits, softcap, 8-tile fixed tree"
                if self.strict
                else "vLLM raw_logprobs (native bf16 logits)"
            ),
            "tensor_parallel_size": self.tensor_parallel_size,
            "gpu_ids": list(self.gpu_ids),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "max_model_len": self.max_model_len,
            "seed": self.seed,
            "generate_calls": self._generate_calls,
            "vllm_version": vllm_version,
            "model_runner": "v2" if os.getenv("VLLM_USE_V2_MODEL_RUNNER", "1") != "0" else "v1",
            "engine_kwargs": {k: str(v) for k, v in self.engine_kwargs.items()},
            "attention_backend_requested": (
                "FLASH_ATTN(strict override)" if self.strict else "auto"
            ),
            "native_lora_dir": self._native_lora_dir,
            "native_lora_reports": list(self._native_lora_reports),
            "lora_replies": list(self._lora_replies),
            "rl_engine_file": self.rl_engine_file,
            "worker_rl_engine_files": list(getattr(self, "worker_rl_engine_files", [])),
            "readback_dir": self._readback_dir,
            "timing": {
                "load_seconds": self.timing.load_seconds,
                "generate_seconds": self.timing.generate_seconds,
                "generated_tokens": self.timing.generated_tokens,
                "prompt_tokens": self.timing.prompt_tokens,
                "logprobs_seconds": self.timing.logprobs_seconds,
                "logprobs_tokens": self.timing.logprobs_tokens,
                "lora_seconds": self.timing.lora_seconds,
                "history": list(self.timing.history),
            },
            "workers": workers,
            "readback_files": files,
        }

    # ------------------------------------------------------------------------------------------
    def close(self) -> None:
        llm = self.__dict__.pop("llm", None)
        if llm is not None:
            shutdown = getattr(getattr(llm, "llm_engine", None), "shutdown", None)
            if callable(shutdown):
                try:
                    shutdown()
                except Exception:  # noqa: BLE001
                    pass
            del llm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


__all__ = [
    "RolloutSample",
    "StrictRolloutEngine",
    "aggregate_worker_lora_hashes",
    "check_worker_paths",
]
