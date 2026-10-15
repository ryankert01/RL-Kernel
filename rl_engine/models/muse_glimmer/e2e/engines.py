# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Fixed engine interfaces of the Muse-Glimmer-30B strict POC end-to-end loop.

The orchestrator (``loop.py``) only depends on the two structural protocols below. The real
engines live in ``rl_engine.models.muse_glimmer.runtime.{strict_rollout_engine,strict_trainer}``
(implemented by sibling rows); ``mocks.py`` provides CPU stand-ins with the same surface.

Interface contract (frozen for the POC):

* ``RolloutSample.logprobs`` are the fp32 selected-token log-probabilities of the completion
  tokens at sampling temperature 1 (raw logits, never temperature- or top-k/p-processed),
  one per completion token, as Python floats that are exactly representable in fp32 (produce
  them with ``tensor.float().tolist()``). The loop converts them back with
  ``torch.tensor(values, dtype=torch.float32)``; the round trip is lossless for fp32 values.
* ``StrictTrainer.token_logprobs`` returns one fp32 tensor per sequence with exactly
  ``len(token_ids) - completion_start`` entries, carrying autograd history.
* ``export_lora`` returns bf16 tensors keyed ``layers.{i}.{proj}.lora_{A,B}`` and the canonical
  hash defined by :func:`rl_engine.models.muse_glimmer.e2e.sync.lora_sha256`.
* ``provenance()`` must report ``lora_sha256`` (the hash of the LoRA currently loaded; ``None``
  before the first sync) so the loop can verify ``weight_sync_identity`` after every sync.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

import torch


@dataclass
class RolloutSample:
    """One sampled completion: token ids, strict fp32 logprobs and the finish reason."""

    token_ids: list[int]
    logprobs: list[float]
    finish_reason: str


class RolloutEngineLike(Protocol):
    """Structural type of ``StrictRolloutEngine``."""

    def generate(
        self,
        prompt_token_ids: list[list[int]],
        *,
        n: int,
        max_tokens: int,
        temperature: float,
    ) -> list[list[Any]]: ...

    def logprobs(self, token_ids: list[list[int]], completion_start: list[int]) -> list[list[float]]: ...

    def load_lora(self, state_dict: dict[str, torch.Tensor], sha256: str) -> None: ...

    def provenance(self) -> dict[str, Any]: ...

    def close(self) -> None: ...


class TrainerLike(Protocol):
    """Structural type of ``StrictTrainer``."""

    def token_logprobs(
        self, token_ids: list[list[int]], completion_start: list[int]
    ) -> list[torch.Tensor]: ...

    def step(self, loss: torch.Tensor) -> dict[str, Any]: ...

    def export_lora(self) -> tuple[dict[str, torch.Tensor], str]: ...

    def provenance(self) -> dict[str, Any]: ...

    def close(self) -> None: ...


class EngineUnavailable(ImportError):
    """Raised when the real strict engines are not importable in this checkout."""


_ROLLOUT_MODULE = "rl_engine.models.muse_glimmer.runtime.strict_rollout_engine"
_TRAINER_MODULE = "rl_engine.models.muse_glimmer.runtime.strict_trainer"


def load_real_engine_classes() -> tuple[type, type]:
    """Import ``StrictRolloutEngine`` and ``StrictTrainer`` lazily.

    The sibling rows own those modules; this checkout may not contain them. Nothing CUDA- or
    vLLM-related is imported until this function is called.
    """

    try:
        rollout_module = importlib.import_module(_ROLLOUT_MODULE)
        trainer_module = importlib.import_module(_TRAINER_MODULE)
    except ImportError as exc:  # pragma: no cover - depends on sibling rows landing
        raise EngineUnavailable(
            f"real strict engines not importable ({exc}); run with --mock or merge the "
            "vllm_runtime_adapter / megatron_vime_provider rows first"
        ) from exc
    return rollout_module.StrictRolloutEngine, trainer_module.StrictTrainer


def coerce_samples(groups: Sequence[Sequence[Any]]) -> list[list[RolloutSample]]:
    """Normalize engine output (any object with token_ids/logprobs/finish_reason) to samples."""

    out: list[list[RolloutSample]] = []
    for group in groups:
        rows: list[RolloutSample] = []
        for sample in group:
            if isinstance(sample, RolloutSample):
                rows.append(sample)
                continue
            if isinstance(sample, Mapping):
                token_ids = sample["token_ids"]
                logprobs = sample["logprobs"]
                finish_reason = sample.get("finish_reason", "unknown")
            else:
                token_ids = sample.token_ids
                logprobs = sample.logprobs
                finish_reason = getattr(sample, "finish_reason", "unknown")
            token_ids = [int(t) for t in token_ids]
            logprobs = [float(v) for v in logprobs]
            if len(token_ids) != len(logprobs):
                raise ValueError(
                    "rollout sample has %d tokens but %d logprobs" % (len(token_ids), len(logprobs))
                )
            rows.append(RolloutSample(token_ids, logprobs, str(finish_reason)))
        out.append(rows)
    return out
