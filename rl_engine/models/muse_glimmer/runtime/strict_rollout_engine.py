# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""``StrictRolloutEngine``: the E2E orchestrator's handle on a vLLM 0.31 Muse Glimmer rollout.

``strict=True`` turns on the RFC #501 binding (``RL_KERNEL_MUSEGLIMMER_STRICT=1`` is inherited by
the vLLM worker processes through the ``vllm.general_plugins`` entry point) and selects the strict
attention backend; ``strict=False`` is plain native vLLM.  Both run Model Runner V2 (the 0.31 CUDA
default) with ``enforce_eager=True``.

``generate`` returns, per prompt, ``n`` samples with the fp32 selected-token log-probabilities of
the completion tokens (strict when ``strict=True``).  ``logprobs`` scores given sequences with one
prefill (``prompt_logprobs=0``) and returns the strict log-probability of every completion token.
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

from rl_engine.models.muse_glimmer.loader import NATIVE_MODEL_CLASSES

STRICT_ENV = "RL_KERNEL_MUSEGLIMMER_STRICT"
READBACK_ENV = "RL_KERNEL_READBACK_DIR"


@dataclass
class RolloutSample:
    token_ids: list[int]
    logprobs: list[float]
    finish_reason: str


@dataclass
class _Timing:
    generate_seconds: float = 0.0
    generated_tokens: int = 0
    prompt_tokens: int = 0
    logprobs_seconds: float = 0.0
    logprobs_tokens: int = 0
    load_seconds: float = 0.0
    history: list[dict[str, Any]] = field(default_factory=list)


class StrictRolloutEngine:
    def __init__(
        self,
        model_path: str,
        *,
        tensor_parallel_size: int,
        gpu_ids: list[int],
        strict: bool,
        max_model_len: int,
        seed: int,
    ) -> None:
        self.model_path = str(model_path)
        self.tensor_parallel_size = int(tensor_parallel_size)
        self.gpu_ids = [int(g) for g in gpu_ids]
        self.strict = bool(strict)
        self.max_model_len = int(max_model_len)
        self.seed = int(seed)
        self.timing = _Timing()
        self._readback_dir = tempfile.mkdtemp(prefix="museglimmer-vllm-readback-")
        os.environ[READBACK_ENV] = self._readback_dir
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

        kwargs: dict[str, Any] = dict(
            model=self.model_path,
            tensor_parallel_size=self.tensor_parallel_size,
            dtype="bfloat16",
            max_model_len=self.max_model_len,
            seed=self.seed,
            enforce_eager=True,
            gpu_memory_utilization=float(os.getenv("RL_KERNEL_MUSEGLIMMER_GPU_UTIL", "0.92")),
            enable_prefix_caching=False,
            limit_mm_per_prompt={"image": 0, "video": 0},
            disable_log_stats=True,
            max_num_seqs=int(os.getenv("RL_KERNEL_MUSEGLIMMER_MAX_NUM_SEQS", "16")),
        )
        if self.strict:
            kwargs["attention_config"] = {"backend": "FLASH_ATTN"}
        start = time.time()
        self.llm = LLM(**kwargs)
        self.timing.load_seconds = time.time() - start
        self._worker_provenance: list[dict[str, Any]] = []
        if self.strict:
            self._assert_strict_bound()

    # ------------------------------------------------------------------------------------------
    def _assert_strict_bound(self) -> None:
        from rl_engine.models.muse_glimmer.runtime.vllm_strict import worker_provenance

        provenances = self.llm.collective_rpc(worker_provenance)
        if not provenances:
            raise RuntimeError("strict rollout: no vLLM workers answered the provenance RPC")
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
        self._worker_provenance = list(provenances)

    # ------------------------------------------------------------------------------------------
    def generate(
        self,
        prompt_token_ids: list[list[int]],
        *,
        n: int,
        max_tokens: int,
        temperature: float,
    ) -> list[list[RolloutSample]]:
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt

        params = SamplingParams(
            n=int(n),
            max_tokens=int(max_tokens),
            temperature=float(temperature),
            logprobs=0,
            seed=self.seed,
            detokenize=False,
            skip_special_tokens=False,
        )
        prompts = [TokensPrompt(prompt_token_ids=[int(t) for t in ids]) for ids in prompt_token_ids]
        start = time.time()
        outputs = self.llm.generate(prompts, params, use_tqdm=False)
        elapsed = time.time() - start
        results: list[list[RolloutSample]] = []
        generated = 0
        for request in outputs:
            samples = []
            for completion in request.outputs:
                token_ids = [int(t) for t in completion.token_ids]
                logprob_dicts = completion.logprobs or []
                if len(logprob_dicts) != len(token_ids):
                    raise RuntimeError(
                        f"vLLM returned {len(logprob_dicts)} logprob entries for "
                        f"{len(token_ids)} tokens"
                    )
                # Logprob objects carry .logprob; the dict maps token -> Logprob.
                logprobs = [
                    float(torch.tensor(entry[token].logprob, dtype=torch.float32).item())
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
            results.append(samples)
        self.timing.generate_seconds += elapsed
        self.timing.generated_tokens += generated
        self.timing.prompt_tokens += sum(len(p) for p in prompt_token_ids)
        self.timing.history.append(
            {
                "op": "generate",
                "seconds": elapsed,
                "generated_tokens": generated,
                "prompt_tokens": sum(len(p) for p in prompt_token_ids),
                "tokens_per_s": (generated / elapsed) if elapsed > 0 else None,
            }
        )
        return results

    # ------------------------------------------------------------------------------------------
    def logprobs(self, token_ids: list[list[int]], completion_start: list[int]) -> list[list[float]]:
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt

        if len(token_ids) != len(completion_start):
            raise ValueError("token_ids and completion_start must have the same length")
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
        outputs = self.llm.generate(prompts, params, use_tqdm=False)
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
                values.append(
                    float(torch.tensor(entry[int(ids[position])].logprob, dtype=torch.float32).item())
                )
            scored += len(values)
            results.append(values)
        self.timing.logprobs_seconds += elapsed
        self.timing.logprobs_tokens += scored
        self.timing.history.append(
            {
                "op": "logprobs",
                "seconds": elapsed,
                "scored_tokens": scored,
                "sequence_tokens": sum(len(ids) for ids in token_ids),
                "tokens_per_s": (sum(len(ids) for ids in token_ids) / elapsed) if elapsed else None,
            }
        )
        return results

    # ------------------------------------------------------------------------------------------
    def load_lora(self, state_dict: dict[str, torch.Tensor], sha256: str) -> None:
        if not self.strict:
            raise RuntimeError("load_lora is only available on a strict engine")
        from rl_engine.models.muse_glimmer.runtime.vllm_strict import worker_load_lora

        cpu_state = {k: v.detach().to("cpu", torch.bfloat16).contiguous() for k, v in state_dict.items()}
        replies = self.llm.collective_rpc(worker_load_lora, args=(cpu_state, str(sha256)))
        if not replies:
            raise RuntimeError("load_lora: no workers answered")
        self._lora_replies = list(replies)

    # ------------------------------------------------------------------------------------------
    def provenance(self) -> dict[str, Any]:
        workers: list[dict[str, Any]] = []
        if self.strict and getattr(self, "llm", None) is not None:
            from rl_engine.models.muse_glimmer.runtime.vllm_strict import worker_provenance

            try:
                workers = list(self.llm.collective_rpc(worker_provenance))
            except Exception as exc:  # noqa: BLE001
                workers = [{"error": f"{type(exc).__name__}: {exc}"}]
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
            "engine": {
                "model_path": self.model_path,
                "strict": self.strict,
                "tensor_parallel_size": self.tensor_parallel_size,
                "gpu_ids": self.gpu_ids,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "max_model_len": self.max_model_len,
                "seed": self.seed,
                "vllm_version": vllm_version,
                "model_runner": "v2" if os.getenv("VLLM_USE_V2_MODEL_RUNNER", "1") != "0" else "v1",
                "enforce_eager": True,
                "attention_backend_requested": "FLASH_ATTN(strict override)" if self.strict else "auto",
                "readback_dir": self._readback_dir,
                "timing": {
                    "load_seconds": self.timing.load_seconds,
                    "generate_seconds": self.timing.generate_seconds,
                    "generated_tokens": self.timing.generated_tokens,
                    "prompt_tokens": self.timing.prompt_tokens,
                    "logprobs_seconds": self.timing.logprobs_seconds,
                    "logprobs_tokens": self.timing.logprobs_tokens,
                    "history": list(self.timing.history),
                },
                "lora_replies": list(getattr(self, "_lora_replies", [])),
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


__all__ = ["RolloutSample", "StrictRolloutEngine"]
