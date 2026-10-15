# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Train-vs-rollout selected-token logprob drift metrics and per-step sidecars.

Definition (mirrors the Qwen3 E2E ``compare_train_rollout_logps`` in
``rl_engine/integrations/orchestrators/vime/experiments/rocm_attention/validate_artifacts.py``,
without the VIME/Ray dependency):

* Inputs are, per sample, the fp32 logprobs of the SAME completion tokens from the trainer
  (teacher-forced) and from the rollout (sampling time). Optional boolean masks select the
  active tokens (default: all completion tokens).
* ``bitwise_mismatch_count``: number of active tokens whose 4 fp32 bytes differ
  (``view(torch.uint8)`` comparison; distinguishes ``-0.0`` from ``0.0`` and NaN payloads).
* ``value_mismatch_count``: ``torch.ne`` count (IEEE inequality).
* ``max_abs_diff`` / ``mean_abs_diff``: on the fp64 difference ``train - rollout``.
* ``mismatch_kl``: mean of ``rollout - train`` (the k1 estimator of KL(rollout || train)) and
  ``mismatch_k3_kl``: mean of ``exp(d) - d - 1`` with ``d = train - rollout``.
* ``first_mismatch``: ``(sample_index, token_index)`` of the first bitwise mismatch in batch
  order, where ``token_index`` is the position within the completion (0 = first generated
  token). This is the operational "first drift" of a step; the loop also records the first
  STEP with any mismatch.
* ``torch_equal`` is ``value_mismatch_count == 0 and max_abs_diff == 0``;
  ``bitwise_equal`` additionally requires ``bitwise_mismatch_count == 0``. The RFC parity
  criterion for strict/strict is ``bitwise_equal``.

Sidecars (``mismatch-sidecars/step{n:04d}.pt``) persist both logprob lists, the masks, the
token ids and the completion starts so the metrics can be recomputed offline
(:func:`metrics_from_sidecar`).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

SIDECAR_SCHEMA = "muse_glimmer_e2e_mismatch_sidecar_v1"


@dataclass
class DriftMetrics:
    sample_count: int
    token_count: int
    bitwise_mismatch_count: int
    value_mismatch_count: int
    max_abs_diff: float
    mean_abs_diff: float
    mismatch_kl: float
    mismatch_k3_kl: float
    first_mismatch: tuple[int, int] | None
    first_mismatch_abs_diff: float | None
    torch_equal: bool
    bitwise_equal: bool

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        if self.first_mismatch is not None:
            out["first_mismatch"] = list(self.first_mismatch)
        return out


def as_fp32_logprobs(values: Any, *, label: str) -> torch.Tensor:
    """Coerce a list of floats or an fp32 tensor to a detached contiguous fp32 CPU tensor."""

    if isinstance(values, torch.Tensor):
        if values.dtype != torch.float32:
            raise TypeError(f"{label} logprobs must be fp32, got {values.dtype}")
        tensor = values.detach().to("cpu")
    else:
        tensor = torch.tensor([float(v) for v in values], dtype=torch.float32)
    if tensor.dim() != 1:
        raise ValueError(
            f"{label} logprobs must be 1-D per sample, got shape {tuple(tensor.shape)}"
        )
    return tensor.contiguous()


def compare_logprobs(
    train: Sequence[Any],
    rollout: Sequence[Any],
    masks: Sequence[Any] | None = None,
) -> DriftMetrics:
    """Compute :class:`DriftMetrics` over all samples (see module docstring)."""

    if len(train) != len(rollout):
        raise ValueError(f"{len(train)} train samples vs {len(rollout)} rollout samples")
    if masks is not None and len(masks) != len(train):
        raise ValueError("masks must have one entry per sample")

    token_count = 0
    bitwise = 0
    value = 0
    sum_abs = 0.0
    sum_kl = 0.0
    sum_k3 = 0.0
    max_abs = 0.0
    first: tuple[int, int] | None = None
    first_abs: float | None = None

    for index, (t_raw, r_raw) in enumerate(zip(train, rollout)):
        t = as_fp32_logprobs(t_raw, label="train")
        r = as_fp32_logprobs(r_raw, label="rollout")
        if t.shape != r.shape:
            raise ValueError(
                f"sample {index}: train has {t.numel()} tokens, rollout has {r.numel()}"
            )
        if masks is None:
            positions = torch.arange(t.numel())
        else:
            mask = torch.as_tensor(masks[index]).to(torch.bool).reshape(-1)
            if mask.numel() != t.numel():
                raise ValueError(f"sample {index}: mask length {mask.numel()} != {t.numel()}")
            positions = mask.nonzero(as_tuple=False).reshape(-1)
            t = t[mask]
            r = r[mask]
        n = int(t.numel())
        if n == 0:
            continue
        if not bool(torch.isfinite(t).all()) or not bool(torch.isfinite(r).all()):
            raise ValueError(f"sample {index}: non-finite logprobs")
        t_bytes = t.view(torch.uint8).reshape(n, t.element_size())
        r_bytes = r.view(torch.uint8).reshape(n, r.element_size())
        per_token_bitwise = (t_bytes != r_bytes).any(dim=1)
        bitwise_here = int(per_token_bitwise.sum().item())
        bitwise += bitwise_here
        value += int(torch.ne(t, r).sum().item())
        delta = t.to(torch.float64) - r.to(torch.float64)
        absolute = delta.abs()
        if first is None and bitwise_here:
            j = int(per_token_bitwise.nonzero(as_tuple=False)[0].item())
            first = (index, int(positions[j].item()))
            first_abs = float(absolute[j].item())
        token_count += n
        sum_abs += float(absolute.sum().item())
        sum_kl += float((-delta).sum().item())
        k3 = torch.exp(delta) - delta - 1.0
        sum_k3 += float(k3.sum().item())
        max_abs = max(max_abs, float(absolute.max().item()))

    if token_count == 0:
        raise ValueError("no active completion tokens to compare")
    torch_equal = value == 0 and max_abs == 0.0
    return DriftMetrics(
        sample_count=len(train),
        token_count=token_count,
        bitwise_mismatch_count=bitwise,
        value_mismatch_count=value,
        max_abs_diff=max_abs,
        mean_abs_diff=sum_abs / token_count,
        mismatch_kl=sum_kl / token_count,
        mismatch_k3_kl=sum_k3 / token_count,
        first_mismatch=first,
        first_mismatch_abs_diff=first_abs,
        torch_equal=torch_equal,
        bitwise_equal=torch_equal and bitwise == 0,
    )


def write_sidecar(
    path: Path,
    *,
    step: int,
    token_ids: Sequence[Sequence[int]],
    completion_start: Sequence[int],
    train_logprobs: Sequence[Any],
    rollout_logprobs: Sequence[Any],
    metrics: DriftMetrics,
    provenance: Mapping[str, Any] | None = None,
    masks: Sequence[Any] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    """Persist one step's comparison inputs and metrics (``torch.save``)."""

    train = [as_fp32_logprobs(v, label="train") for v in train_logprobs]
    rollout = [as_fp32_logprobs(v, label="rollout") for v in rollout_logprobs]
    if masks is None:
        loss_masks = [torch.ones(t.numel(), dtype=torch.bool) for t in train]
    else:
        loss_masks = [torch.as_tensor(m).to(torch.bool).reshape(-1) for m in masks]
    payload: dict[str, Any] = {
        "schema_version": SIDECAR_SCHEMA,
        "step": int(step),
        "token_ids": [[int(t) for t in seq] for seq in token_ids],
        "completion_start": [int(c) for c in completion_start],
        "total_lengths": [len(seq) for seq in token_ids],
        "response_lengths": [len(seq) - int(c) for seq, c in zip(token_ids, completion_start)],
        "train_log_probs": train,
        "rollout_log_probs": rollout,
        "loss_masks": loss_masks,
        "metrics": metrics.to_dict(),
        "provenance": dict(provenance or {}),
    }
    if extra:
        payload["extra"] = dict(extra)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return path


def load_sidecar(path: Path) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("schema_version") != SIDECAR_SCHEMA:
        raise ValueError(f"{path}: not a {SIDECAR_SCHEMA} sidecar")
    return payload


def metrics_from_sidecar(payload: Mapping[str, Any]) -> DriftMetrics:
    """Recompute the metrics from a sidecar payload (offline validation)."""

    return compare_logprobs(
        payload["train_log_probs"], payload["rollout_log_probs"], payload["loss_masks"]
    )


def finite_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    return value if math.isfinite(value) else None
