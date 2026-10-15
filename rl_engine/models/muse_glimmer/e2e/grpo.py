# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""GRPO loss on selected-token logprobs (fp32, token-level, PPO-style clipping).

Everything here is plain fp32 torch on whatever device the trainer returned its tensors on.
No logits are needed: the trainer gives per-completion-token logprobs with grad, the rollout
gives the sampling-time logprobs of the same tokens.

* Advantages: per prompt group of ``samples_per_prompt`` rewards,
  ``A = (r - mean_g) / max(std_g, eps)`` with the population std (``unbiased=False``). A group
  with identical rewards gets exactly zero advantage (the numerator is exactly zero).
  Semantics match ``rl_engine.reference.loss.grpo_loss.NativeGRPOLossOp.group_advantages``.
* Ratio: ``rho = exp(logp_train - logp_rollout)`` with the rollout side detached.
* Surrogate: ``-min(rho * A, clip(rho, 1 - eps, 1 + eps) * A)`` per token.
* Optional KL (off by default): k3 estimator ``exp(ref - train) - (ref - train) - 1`` against
  reference logprobs, weighted by ``kl_coef``.
* Loss masking: only completion tokens (and an optional extra mask) contribute; the loss is
  the mean over all active tokens in the batch (token-level aggregation).

In strict/strict mode ``logp_train == logp_rollout`` bitwise, so ``rho == 1`` exactly, the
clip never activates and the gradient equals the group-normalized REINFORCE gradient.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import torch


@dataclass
class GRPOLossOutput:
    loss: torch.Tensor
    stats: dict[str, float] = field(default_factory=dict)


def group_normalized_advantages(
    rewards: Sequence[float] | torch.Tensor,
    samples_per_prompt: int,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Group-normalized advantages, fp32, one per sample, in the input order."""

    if samples_per_prompt <= 0:
        raise ValueError("samples_per_prompt must be positive")
    flat = torch.as_tensor(rewards, dtype=torch.float32).reshape(-1)
    if flat.numel() == 0 or flat.numel() % samples_per_prompt != 0:
        raise ValueError(
            f"{flat.numel()} rewards are not a multiple of samples_per_prompt={samples_per_prompt}"
        )
    groups = flat.view(-1, samples_per_prompt)
    centered = groups - groups.mean(dim=1, keepdim=True)
    std = centered.pow(2).mean(dim=1, keepdim=True).sqrt()
    return (centered / std.clamp_min(eps)).reshape(-1)


def pad_logprobs(
    sequences: Sequence[torch.Tensor | Sequence[float]], *, device: torch.device | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-pad 1-D fp32 logprob tensors into ``(B, T)`` and return the active mask."""

    tensors = []
    for seq in sequences:
        t = seq if isinstance(seq, torch.Tensor) else torch.tensor(seq, dtype=torch.float32)
        if t.dim() != 1:
            raise ValueError(f"expected 1-D logprobs, got shape {tuple(t.shape)}")
        tensors.append(t.to(torch.float32))
    if not tensors:
        raise ValueError("no sequences")
    device = device or tensors[0].device
    width = max(int(t.numel()) for t in tensors)
    rows = []
    mask = torch.zeros(len(tensors), width, dtype=torch.bool, device=device)
    for i, t in enumerate(tensors):
        n = int(t.numel())
        rows.append(torch.nn.functional.pad(t.to(device), (0, width - n)))
        mask[i, :n] = True
    return torch.stack(rows, dim=0), mask


def grpo_token_loss(
    logp_train: Sequence[torch.Tensor],
    logp_rollout: Sequence[torch.Tensor | Sequence[float]],
    advantages: torch.Tensor | Sequence[float],
    *,
    clip_eps: float = 0.2,
    kl_coef: float = 0.0,
    logp_ref: Sequence[torch.Tensor | Sequence[float]] | None = None,
    masks: Sequence[torch.Tensor | Sequence[bool]] | None = None,
) -> GRPOLossOutput:
    """Token-level clipped GRPO loss; see module docstring for the exact formula."""

    if len(logp_train) != len(logp_rollout):
        raise ValueError("logp_train and logp_rollout must have the same number of samples")
    train, mask = pad_logprobs(logp_train)
    rollout, mask_r = pad_logprobs(logp_rollout, device=train.device)
    if not torch.equal(mask, mask_r):
        raise ValueError("train and rollout completion lengths differ")
    if masks is not None:
        extra, _ = pad_logprobs(
            [torch.as_tensor(m).to(torch.float32) for m in masks], device=train.device
        )
        mask = mask & (extra > 0.5)
    adv = torch.as_tensor(advantages, dtype=torch.float32, device=train.device).reshape(-1)
    if adv.numel() != train.shape[0]:
        raise ValueError(f"{adv.numel()} advantages for {train.shape[0]} samples")
    adv = adv.view(-1, 1).expand_as(train)

    rollout = rollout.detach()
    log_ratio = train - rollout
    ratio = torch.exp(log_ratio)
    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
    surrogate = torch.minimum(unclipped, clipped)
    per_token = -surrogate

    kl_value = torch.zeros((), dtype=torch.float32, device=train.device)
    maskf = mask.to(torch.float32)
    active = maskf.sum().clamp_min(1.0)
    if kl_coef > 0.0:
        if logp_ref is None:
            raise ValueError("kl_coef > 0 requires logp_ref")
        ref, mask_ref = pad_logprobs(logp_ref, device=train.device)
        if not torch.equal(mask_ref, mask_r):
            raise ValueError("reference completion lengths differ")
        ref = ref.detach()
        diff = ref - train
        kl_tokens = torch.exp(diff) - diff - 1.0
        kl_value = (kl_tokens * maskf).sum() / active
        per_token = per_token + kl_coef * kl_tokens

    loss = (per_token * maskf).sum() / active
    with torch.no_grad():
        policy_loss = (-surrogate * maskf).sum() / active
        outside = ((ratio < 1.0 - clip_eps) | (ratio > 1.0 + clip_eps)) & mask
        clipped_selected = (clipped < unclipped) & mask
        ratio_active = ratio[mask]
        stats = {
            "loss": float(loss.item()),
            "policy_loss": float(policy_loss.item()),
            "kl": float(kl_value.item()),
            "kl_coef": float(kl_coef),
            "clip_eps": float(clip_eps),
            "ratio_mean": float(ratio_active.mean().item()),
            "ratio_min": float(ratio_active.min().item()),
            "ratio_max": float(ratio_active.max().item()),
            "ratio_exactly_one_frac": float((ratio_active == 1.0).to(torch.float32).mean().item()),
            "clip_frac": float(outside.sum().item() / active.item()),
            "clipped_selected_frac": float(clipped_selected.sum().item() / active.item()),
            "approx_kl_k3": float(
                (((ratio - 1.0) - log_ratio) * maskf).sum().item() / active.item()
            ),
            "mean_log_ratio": float((log_ratio * maskf).sum().item() / active.item()),
            "active_tokens": int(active.item()),
            "advantage_abs_mean": float(adv[:, 0].abs().mean().item()),
        }
    return GRPOLossOutput(loss=loss, stats=stats)


def reward_stats(rewards: Sequence[float], samples_per_prompt: int) -> dict[str, Any]:
    r = torch.as_tensor(rewards, dtype=torch.float32).reshape(-1, samples_per_prompt)
    group_solved = (r.max(dim=1).values > 0).to(torch.float32)
    group_uniform = (r.max(dim=1).values == r.min(dim=1).values).to(torch.float32)
    return {
        "reward_mean": float(r.mean().item()),
        "reward_std": float(r.std(unbiased=False).item()) if r.numel() > 1 else 0.0,
        "reward_max": float(r.max().item()),
        "groups_with_any_success_frac": float(group_solved.mean().item()),
        "groups_with_uniform_reward_frac": float(group_uniform.mean().item()),
    }
