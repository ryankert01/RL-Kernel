# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Operator resolution for the Muse Glimmer strict reference.

``backend="golden"`` binds every node to the pure-torch fixed-order fallbacks in
:mod:`golden_ops`. ``backend="triton"`` binds each node to the sibling-owned strict op under
``rl_engine.models.muse_glimmer.ops.<module>`` with the FIXED interfaces of RFC #501; a missing
sibling module is an error unless ``allow_partial=True``, in which case that node falls back to
golden and the substitution is recorded in ``OpSet.provenance`` (never silently).
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any, Callable

import torch

from rl_engine.models.muse_glimmer.reference import golden_ops
from rl_engine.models.muse_glimmer.reference.golden_ops import GoldenGemm, Linear, LmHead

BACKENDS = ("golden", "triton")

# node -> (sibling module, attribute)
_SIBLING_BINDINGS: dict[str, tuple[str, str]] = {
    "normalized_embedding": ("norms", "normalized_embedding"),
    "centered_rmsnorm": ("norms", "centered_rmsnorm"),
    "weighted_rmsnorm": ("norms", "weighted_rmsnorm"),
    "fused_post_norm_residual_pre_norm": ("norms", "fused_post_norm_residual_pre_norm"),
    "gemm": ("det_gemm", "fast_det_gemm"),
    "rope_table": ("qknorm_rope", "rope_table"),
    "qknorm_scaled_rope": ("qknorm_rope", "qknorm_scaled_rope"),
    "attention_varlen": ("gated_attention", "strict_gated_attention_fwd_varlen"),
    "attention_paged": ("gated_attention", "strict_gated_attention_fwd_paged"),
    "swiglu_ffn": ("swiglu_ffn", "swiglu_ffn"),
    "pad_lm_head_weight": ("lm_head_softcap_logprob", "pad_lm_head_weight"),
    "lm_head_softcap_logprob": ("lm_head_softcap_logprob", "lm_head_softcap_logprob"),
}
_SIBLING_PACKAGE = "rl_engine.models.muse_glimmer.ops"


class SiblingOpsMissing(ImportError):
    """A requested sibling strict op is not present in this worktree."""


@dataclass
class OpSet:
    """Resolved callables for every node kind plus their provenance."""

    backend: str
    normalized_embedding: Callable[..., torch.Tensor]
    centered_rmsnorm: Callable[..., torch.Tensor]
    weighted_rmsnorm: Callable[..., torch.Tensor]
    fused_post_norm_residual_pre_norm: Callable[..., tuple[torch.Tensor, torch.Tensor]]
    gemm: Any
    rope_table: Callable[..., tuple[torch.Tensor, torch.Tensor]]
    qknorm_scaled_rope: Callable[..., tuple[torch.Tensor, torch.Tensor]]
    attention_varlen: Callable[..., tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    attention_paged: Callable[..., tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    swiglu_ffn: Callable[..., torch.Tensor]
    pad_lm_head_weight: Callable[..., torch.Tensor]
    lm_head_softcap_logprob: Callable[..., torch.Tensor]
    provenance: dict[str, str] = field(default_factory=dict)

    @property
    def weights_as_objects(self) -> bool:
        """Golden ops consume :class:`Linear`/:class:`LmHead` holders (cached transposes);
        sibling ops consume raw natural-layout tensors."""
        return self.backend == "golden"

    def weight_arg(self, lin: Linear) -> Any:
        return lin if self.weights_as_objects else lin.nk()

    def lm_head_arg(self, head: LmHead) -> Any:
        return head if self.weights_as_objects else head.nk_padded()

    def preferred_layout(self) -> str:
        """Which single copy a forward-only weight set should keep for this backend."""
        return "kn" if self.backend == "golden" else "nk"

    def sibling_nodes(self) -> list[str]:
        return sorted(k for k, v in self.provenance.items() if not v.startswith("golden"))

    def golden_nodes(self) -> list[str]:
        return sorted(k for k, v in self.provenance.items() if v.startswith("golden"))

    def describe(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "provenance": dict(self.provenance),
            "golden_trace": golden_ops.trace(),
            "gemm_trace": self.gemm.trace() if hasattr(self.gemm, "trace") else str(self.gemm),
        }


def _golden_table(gemm_impl: str) -> dict[str, Any]:
    return {
        "normalized_embedding": golden_ops.normalized_embedding,
        "centered_rmsnorm": golden_ops.centered_rmsnorm,
        "weighted_rmsnorm": golden_ops.weighted_rmsnorm,
        "fused_post_norm_residual_pre_norm": golden_ops.fused_post_norm_residual_pre_norm,
        "gemm": GoldenGemm(gemm_impl),
        "rope_table": golden_ops.rope_table,
        "qknorm_scaled_rope": golden_ops.qknorm_scaled_rope,
        "attention_varlen": golden_ops.strict_gated_attention_fwd_varlen,
        "attention_paged": golden_ops.strict_gated_attention_fwd_paged,
        "swiglu_ffn": golden_ops.swiglu_ffn,
        "pad_lm_head_weight": golden_ops.pad_lm_head_weight,
        "lm_head_softcap_logprob": golden_ops.lm_head_softcap_logprob,
    }


def resolve_ops(
    backend: str = "golden", *, allow_partial: bool = False, gemm_impl: str = "triton_fp32"
) -> OpSet:
    """Bind every node kind for ``backend``. See the module docstring for the rules."""
    if backend not in BACKENDS:
        raise ValueError(f"unknown backend {backend!r}; expected one of {BACKENDS}")
    table = _golden_table(gemm_impl)
    provenance = {k: f"golden:{golden_ops.__name__}" for k in table}
    if backend == "golden":
        provenance["gemm"] = f"golden:{golden_ops.__name__}.GoldenGemm[{gemm_impl}]"
        return OpSet(backend=backend, provenance=provenance, **table)

    missing: list[str] = []
    for node, (module_name, attr) in _SIBLING_BINDINGS.items():
        path = f"{_SIBLING_PACKAGE}.{module_name}"
        try:
            module = importlib.import_module(path)
            fn = getattr(module, attr)
        except (ImportError, AttributeError) as exc:
            missing.append(f"{path}.{attr} ({type(exc).__name__}: {exc})")
            continue
        table[node] = fn
        provenance[node] = f"{path}.{attr}"
    if missing and not allow_partial:
        raise SiblingOpsMissing(
            "backend='triton' needs sibling ops that are not in this worktree:\n  "
            + "\n  ".join(missing)
            + "\nUse backend='golden' or allow_partial=True (records golden substitutes)."
        )
    opset = OpSet(backend=backend, provenance=provenance, **table)
    if opset.weights_as_objects is False and opset.golden_nodes():
        # Mixed set: golden fallbacks must accept raw tensors too (they do; see _kn/_nk).
        pass
    return opset


__all__ = ["BACKENDS", "OpSet", "SiblingOpsMissing", "resolve_ops"]
