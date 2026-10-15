# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Resolve the Muse Glimmer strict operator set, falling back per op with an honest label.

The vLLM binding codes against the FIXED sibling interfaces of RFC #501
(``rl_engine.models.muse_glimmer.ops.{det_gemm,canonical_tree,norms,qknorm_rope,gated_attention,
swiglu_ffn,lm_head_softcap_logprob}``).  Each op is resolved independently: when the sibling module
imports, its function is bound under the label ``strict:<module>.<name>``; otherwise the labelled
fallback from :mod:`fallback_ops` is bound under ``fallback:<...>``.  :func:`probe_ops` additionally
smoke-tests every strict binding on tiny device tensors so a sibling whose runtime signature drifted
is demoted to the fallback (with the reason recorded) instead of failing in the middle of a rollout.

Nothing here is ever called "strict" in provenance unless it is the sibling kernel.
"""

from __future__ import annotations

import importlib
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch

from rl_engine.models.muse_glimmer.runtime import fallback_ops as fb

_OPS_PACKAGE = "rl_engine.models.muse_glimmer.ops"
FORCE_FALLBACK_ENV = "RL_KERNEL_MUSEGLIMMER_FORCE_FALLBACK"

OP_NAMES = (
    "det_gemm",
    "det_gemm_partial",
    "combine_tree",
    "normalized_embedding",
    "centered_rmsnorm",
    "weighted_rmsnorm",
    "fused_post_norm_residual_pre_norm",
    "rope_table",
    "qknorm_scaled_rope",
    "gated_attention_paged",
    "gated_attention_varlen",
    "swiglu",
    "softcap_tile_stats",
    "combine_tile_stats",
)

# op name -> (sibling module, sibling attribute)
_SIBLING: dict[str, tuple[str, str]] = {
    "det_gemm": ("det_gemm", "fast_det_gemm"),
    "det_gemm_partial": ("det_gemm", "fast_det_gemm_partial"),
    "combine_tree": ("canonical_tree", "combine_tree"),
    "normalized_embedding": ("norms", "normalized_embedding"),
    "centered_rmsnorm": ("norms", "centered_rmsnorm"),
    "weighted_rmsnorm": ("norms", "weighted_rmsnorm"),
    "fused_post_norm_residual_pre_norm": ("norms", "fused_post_norm_residual_pre_norm"),
    "rope_table": ("qknorm_rope", "rope_table"),
    "qknorm_scaled_rope": ("qknorm_rope", "qknorm_scaled_rope"),
    "gated_attention_paged": ("gated_attention", "strict_gated_attention_fwd_paged"),
    "gated_attention_varlen": ("gated_attention", "strict_gated_attention_fwd_varlen"),
    "swiglu": ("swiglu_ffn", "swiglu"),
    "softcap_tile_stats": ("lm_head_softcap_logprob", "softcap_tile_stats"),
    "combine_tile_stats": ("lm_head_softcap_logprob", "combine_tile_stats"),
}

_FALLBACK: dict[str, tuple[Callable[..., Any], str]] = {
    "det_gemm": (fb.det_gemm, fb.GEMM_LABEL),
    "det_gemm_partial": (fb.det_gemm_partial, fb.GEMM_LABEL),
    "combine_tree": (fb.combine_tree, "fallback:torch_fp32_fixed_tree"),
    "normalized_embedding": (fb.normalized_embedding, "fallback:torch_fp32_fixed_order"),
    "centered_rmsnorm": (fb.centered_rmsnorm, "fallback:torch_fp32_fixed_order"),
    "weighted_rmsnorm": (fb.weighted_rmsnorm, "fallback:torch_fp32_fixed_order"),
    "fused_post_norm_residual_pre_norm": (
        fb.fused_post_norm_residual_pre_norm,
        "fallback:torch_fp32_fixed_order",
    ),
    "rope_table": (fb.rope_table, "fallback:torch_fp32"),
    "qknorm_scaled_rope": (fb.qknorm_scaled_rope, "fallback:torch_fp32_fixed_order"),
    "gated_attention_paged": (
        fb.gated_attention_fwd_paged,
        "fallback:torch_online_softmax_64key_blocks+" + fb.GEMM_LABEL,
    ),
    "gated_attention_varlen": (
        fb.gated_attention_fwd_varlen,
        "fallback:torch_online_softmax_64key_blocks+" + fb.GEMM_LABEL,
    ),
    "swiglu": (fb.swiglu, "fallback:torch_fp32_elementwise"),
    "softcap_tile_stats": (fb.softcap_tile_stats, "fallback:torch_fp32_fixed_order"),
    "combine_tile_stats": (fb.combine_tile_stats, "fallback:torch_fp32_fixed_tree"),
}


@dataclass
class StrictOps:
    """The resolved operator table plus its per-op provenance labels."""

    fns: dict[str, Callable[..., Any]]
    labels: dict[str, str]
    demotions: list[dict[str, str]] = field(default_factory=list)

    def __getattr__(self, name: str) -> Callable[..., Any]:
        fns = self.__dict__.get("fns")
        if fns is not None and name in fns:
            return fns[name]
        raise AttributeError(name)

    def is_strict(self, name: str) -> bool:
        return self.labels[name].startswith("strict:")

    @property
    def all_strict(self) -> bool:
        return all(self.is_strict(name) for name in OP_NAMES)

    @property
    def fallback_names(self) -> list[str]:
        return [name for name in OP_NAMES if not self.is_strict(name)]

    def demote(self, name: str, reason: str) -> None:
        fn, label = _FALLBACK[name]
        self.demotions.append({"op": name, "from": self.labels[name], "reason": reason})
        self.fns[name] = fn
        self.labels[name] = label

    def provenance(self) -> dict[str, Any]:
        return {
            "op_impls": dict(self.labels),
            "fallbacks": list(self.fallback_names),
            "demotions": list(self.demotions),
            "all_strict": self.all_strict,
        }


def _force_fallback_from_env() -> bool:
    return os.getenv(FORCE_FALLBACK_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


def _adapt(name: str, fn: Callable[..., Any]) -> Callable[..., Any]:
    """Normalize the sibling call signature to the one the binding uses."""

    if name == "det_gemm":

        def det_gemm(a, w, *, out_dtype=torch.bfloat16, k_chunks=8, w_kn=False):
            return fn(a, w, out_dtype=out_dtype, k_chunks=k_chunks, w_kn=w_kn)

        return det_gemm
    if name == "det_gemm_partial":

        def det_gemm_partial(a_local, w_local, chunk_ids, *, w_kn=False):
            return fn(a_local, w_local, chunk_ids, w_kn=w_kn)

        return det_gemm_partial
    return fn


def resolve_ops(force_fallback: bool | None = None) -> StrictOps:
    """Bind every op to the sibling strict implementation when importable, else the fallback."""

    force = _force_fallback_from_env() if force_fallback is None else bool(force_fallback)
    fns: dict[str, Callable[..., Any]] = {}
    labels: dict[str, str] = {}
    for name in OP_NAMES:
        module_name, attr = _SIBLING[name]
        strict_fn = None
        if not force:
            try:
                module = importlib.import_module(f"{_OPS_PACKAGE}.{module_name}")
                strict_fn = getattr(module, attr, None)
            except Exception:  # noqa: BLE001 - absent or broken sibling -> fallback
                strict_fn = None
        if strict_fn is not None:
            fns[name] = _adapt(name, strict_fn)
            labels[name] = f"strict:{_OPS_PACKAGE}.{module_name}.{attr}"
        else:
            fns[name], labels[name] = _FALLBACK[name]
    return StrictOps(fns=fns, labels=labels)


def _probe_cases(device: torch.device) -> dict[str, Callable[[StrictOps], None]]:
    """Tiny-shape smoke tests; each must run without raising and return sane shapes/dtypes."""

    g = torch.Generator(device="cpu").manual_seed(0)
    bf = torch.bfloat16

    def rnd(*shape):
        return torch.randn(*shape, generator=g).to(device=device, dtype=bf)

    def expect(cond: bool, message: str) -> None:
        if not cond:
            raise RuntimeError(message)

    def probe_gemm(ops: StrictOps) -> None:
        a = rnd(4, 512)
        w = rnd(256, 512)
        y = ops.det_gemm(a, w, out_dtype=bf)
        expect(y.shape == (4, 256) and y.dtype == bf, "det_gemm shape/dtype")
        y32 = ops.det_gemm(a, w, out_dtype=torch.float32)
        expect(y32.dtype == torch.float32, "det_gemm fp32 out")

    def probe_partial(ops: StrictOps) -> None:
        a = rnd(4, 512)
        w = rnd(256, 512)
        p = ops.det_gemm_partial(a[:, :256], w[:, :256], (0, 1, 2, 3))
        expect(p.shape == (4, 256) and p.dtype == torch.float32, "det_gemm_partial")

    def probe_tree(ops: StrictOps) -> None:
        parts = [torch.randn(3, generator=g).to(device) for _ in range(8)]
        expect(ops.combine_tree(parts).shape == (3,), "combine_tree")

    def probe_norms(ops: StrictOps) -> None:
        x = rnd(4, 512)
        w = rnd(512)
        expect(ops.centered_rmsnorm(x, w, eps=1e-5).dtype == bf, "centered_rmsnorm")
        expect(ops.weighted_rmsnorm(x, w, eps=1e-5).dtype == bf, "weighted_rmsnorm")
        ids = torch.tensor([1, 3, 2, 0], device=device)
        expect(ops.normalized_embedding(x, ids, eps=1e-5).shape == (4, 512), "normalized_embedding")
        r, pre = ops.fused_post_norm_residual_pre_norm(
            x, x, w, 1e-8, w, 1e-5, pre_form="centered"
        )
        expect(r.dtype == bf and pre.shape == x.shape, "fused_post_norm_residual_pre_norm")

    def probe_rope(ops: StrictOps) -> None:
        pos = torch.arange(4, device=device)
        cos, sin = ops.rope_table(pos, theta=500000.0, dim=128, device=device)
        expect(cos.dtype == torch.float32 and cos.shape[0] == 4, "rope_table")
        q = rnd(4, 32, 128)
        k = rnd(4, 2, 128)
        qh, kh = ops.qknorm_scaled_rope(q, k, cos, sin, eps=1e-5, qk_scale=3.87)
        expect(qh.shape == q.shape and kh.dtype == bf, "qknorm_scaled_rope")
        qh, kh = ops.qknorm_scaled_rope(q, k, None, None, eps=1e-5, qk_scale=3.87)
        expect(qh.shape == q.shape, "qknorm_scaled_rope (no rope)")

    def probe_attention(ops: StrictOps) -> None:
        t = 4
        q = rnd(t, 32, 128)
        gate = rnd(t, 4096)
        k_cache = rnd(8, 16, 2, 128)
        v_cache = rnd(8, 16, 2, 128)
        block_table = torch.tensor([[1, 2, 3, 4]], dtype=torch.int32, device=device)
        seq_lens = torch.tensor([t], dtype=torch.int32, device=device)
        cu_q = torch.tensor([0, t], dtype=torch.int32, device=device)
        y, o, lse = ops.gated_attention_paged(
            q, gate, k_cache, v_cache, block_table, seq_lens, cu_q, window=2048
        )
        expect(y.shape == (t, 4096) and y.dtype == bf, "gated_attention_paged")
        kv = rnd(t, 2, 128)
        pos = torch.arange(t, device=device)
        y2, _, _ = ops.gated_attention_varlen(q, kv, kv, gate, cu_q, cu_q, pos, window=None)
        expect(y2.shape == (t, 4096), "gated_attention_varlen")

    def probe_swiglu(ops: StrictOps) -> None:
        expect(ops.swiglu(rnd(4, 256), rnd(4, 256)).dtype == bf, "swiglu")

    def probe_logp(ops: StrictOps) -> None:
        z = torch.randn(4, 25344, generator=g).to(device)
        targets = torch.tensor([5, 25345, 202047, 7], device=device)
        stats = [
            ops.softcap_tile_stats(
                z, tile, targets, multiplier=0.19611613513818404, softcap=20.0, vocab_size=202048
            )
            for tile in range(8)
        ]
        expect(len(stats[0]) == 4 and stats[0][0].shape == (4,), "softcap_tile_stats")
        m, log_z = ops.combine_tile_stats(stats)
        expect(m.shape == (4,) and log_z.shape == (4,), "combine_tile_stats")

    return {
        "det_gemm": probe_gemm,
        "det_gemm_partial": probe_partial,
        "combine_tree": probe_tree,
        "normalized_embedding": probe_norms,
        "centered_rmsnorm": probe_norms,
        "weighted_rmsnorm": probe_norms,
        "fused_post_norm_residual_pre_norm": probe_norms,
        "rope_table": probe_rope,
        "qknorm_scaled_rope": probe_rope,
        "gated_attention_paged": probe_attention,
        "gated_attention_varlen": probe_attention,
        "swiglu": probe_swiglu,
        "softcap_tile_stats": probe_logp,
        "combine_tile_stats": probe_logp,
    }


def probe_ops(ops: StrictOps, device: torch.device) -> StrictOps:
    """Smoke-test every strict binding; demote the ones that fail (reason recorded)."""

    cases = _probe_cases(device)
    for _round in range(2):  # a demotion can change the inputs of a sibling probe; re-run once
        changed = False
        for name in OP_NAMES:
            if not ops.is_strict(name):
                continue
            try:
                cases[name](ops)
            except Exception as exc:  # noqa: BLE001 - demote and record
                ops.demote(name, f"{type(exc).__name__}: {exc}")
                changed = True
        if not changed:
            break
    return ops


__all__ = ["FORCE_FALLBACK_ENV", "OP_NAMES", "StrictOps", "probe_ops", "resolve_ops"]
