# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Autograd binding of the strict Muse-Glimmer gated GQA attention (contract O5).

``GatedAttentionFn`` glues the strict Triton forward
(:func:`rl_engine.models.muse_glimmer.ops.gated_attention.strict_gated_attention_fwd_varlen`)
to the deterministic Triton backward in
:mod:`rl_engine.models.muse_glimmer.triton.gated_attention_bwd`. There is no fallback: if the
strict forward cannot be imported, importing this module fails (strict ops fail closed).

Saved for backward: q, k, v, gate_logits, o (pre-gate bf16), lse (fp32), cu_seqlens, q_pos.
Gradients returned: dq (Tq,H_Q,128) bf16, dk/dv (Tk,H_KV,128) bf16 (canonical tree over the
Q-head-group partials, each folded over the 8 canonical query chunks), dgate (Tq,H_Q*128) bf16;
None for the index tensors, ``window`` and ``seq_meta``.

Head counts are taken from ``q.shape[1]`` / ``k.shape[1]`` and validated against
``SUPPORTED_HEAD_GEOMETRIES``. The autograd op is complete for TP1 (32/2) and TP2 (16/1) ranks;
TP4/TP8 ranks hold only part of their KV head's groups, so their backward raises and the binding
must use ``strict_gated_attention_bwd_partials`` with the cross-rank canonical tree.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch.autograd.function import once_differentiable

from rl_engine.models.muse_glimmer.ops import gated_attention as _fwd
from rl_engine.models.muse_glimmer.triton import gated_attention_bwd as _bwd

try:
    from rl_engine.ops.autograd.backward_runtime import record_backward
except ImportError:  # pragma: no cover - provenance hook only, not arithmetic

    def record_backward(*_args, **_kwargs) -> None:
        return None


FORWARD_IMPL = "rl_engine.models.muse_glimmer.ops.gated_attention.strict_gated_attention_fwd_varlen"


def strict_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate_logits: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    q_pos: torch.Tensor,
    *,
    window: Optional[int],
    max_seqlen_q: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(y bf16 (Tq,H_Q*128), o bf16 (Tq,H_Q*128), lse fp32 (Tq,H_Q))`` - the strict kernel."""
    return _fwd.strict_gated_attention_fwd_varlen(
        q, k, v, gate_logits, cu_seqlens_q, cu_seqlens_k, q_pos, window=window,
        max_seqlen_q=max_seqlen_q, num_q_heads=q.shape[1], num_kv_heads=k.shape[1],
    )


class GatedAttentionFn(torch.autograd.Function):
    """y = gated_attention(q, k, v, gate_logits); deterministic Triton backward."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        gate_logits: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        q_pos: torch.Tensor,
        window: Optional[int],
        seq_meta: Optional[_bwd.SequenceMeta] = None,
    ) -> torch.Tensor:
        q_c = q.contiguous()
        k_c = k.contiguous()
        v_c = v.contiguous()
        g_c = gate_logits.contiguous()
        max_seqlen_q = None
        if seq_meta is not None:
            # fail closed on a meta built for other tensors; no host sync (identity + versions)
            _bwd._resolve_meta(seq_meta, cu_seqlens_q, cu_seqlens_k, q_pos, k_c.shape[0])
            max_seqlen_q = seq_meta.max_seqlen_q
        y, o, lse = strict_forward(
            q_c, k_c, v_c, g_c, cu_seqlens_q, cu_seqlens_k, q_pos, window=window,
            max_seqlen_q=max_seqlen_q,
        )
        ctx.save_for_backward(q_c, k_c, v_c, g_c, o, lse, cu_seqlens_q, cu_seqlens_k, q_pos)
        ctx.window = None if window is None else int(window)
        ctx.seq_meta = seq_meta
        return y

    @staticmethod
    @once_differentiable
    def backward(ctx, dy: torch.Tensor):  # type: ignore[override]
        q, k, v, g, o, lse, cu_q, cu_k, q_pos = ctx.saved_tensors
        meta = ctx.seq_meta
        if meta is not None and not (
            meta.cu_seqlens_q is cu_q and meta.cu_seqlens_k is cu_k and meta.q_pos is q_pos
        ):
            meta = None  # not provably the validated tensors: re-validate (one host sync)
        dq, dk, dv, dgate = _bwd.strict_gated_attention_bwd(
            q, k, v, g, o, lse, dy.contiguous(), cu_q, cu_k, q_pos, window=ctx.window,
            num_q_heads=q.shape[1], num_kv_heads=k.shape[1], seq_meta=meta,
        )
        record_backward(
            "muse_glimmer.gated_attention",
            kernel_id=_bwd.KERNEL_ID,
            impl="triton",
            family="triton",
        )
        return dq, dk, dv, dgate, None, None, None, None, None


def gated_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate_logits: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    q_pos: torch.Tensor,
    window: Optional[int],
    seq_meta: Optional[_bwd.SequenceMeta] = None,
) -> torch.Tensor:
    """Convenience wrapper: ``y bf16 (Tq, H_Q*128)`` with the strict backward attached.

    ``seq_meta`` (optional, from ``triton.gated_attention_bwd.prepare_sequence_meta`` on the same
    ``cu_seqlens``/``q_pos`` tensors) removes every host synchronisation of the op: the forward
    takes its grid size (``max_seqlen_q``) from it and the backward skips re-validation. A meta
    built for other (or since modified) tensors raises.
    """
    return GatedAttentionFn.apply(
        q, k, v, gate_logits, cu_seqlens_q, cu_seqlens_k, q_pos, window, seq_meta
    )


def gated_attention_backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate_logits: torch.Tensor,
    o: torch.Tensor,
    lse: torch.Tensor,
    dy: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    q_pos: torch.Tensor,
    *,
    window: Optional[int],
    seq_meta: Optional[_bwd.SequenceMeta] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Direct (non-autograd) entry: ``(dq, dk, dv, dgate)`` bf16 from saved forward tensors."""
    return _bwd.strict_gated_attention_bwd(
        q, k, v, gate_logits, o, lse, dy, cu_seqlens_q, cu_seqlens_k, q_pos, window=window,
        num_q_heads=q.shape[1], num_kv_heads=k.shape[1], seq_meta=seq_meta,
    )


def trace() -> dict:
    """Provenance of the autograd op (contract R4): forward + backward kernel traces."""
    rec = dict(_bwd.trace())
    rec["op"] = "muse_glimmer.gated_attention"
    rec["forward_impl"] = FORWARD_IMPL
    rec["forward_is_fallback"] = False
    rec["forward"] = _fwd.trace()
    rec["backward_impl"] = "rl_engine.models.muse_glimmer.triton.gated_attention_bwd"
    return rec


__all__ = [
    "FORWARD_IMPL",
    "GatedAttentionFn",
    "gated_attention",
    "gated_attention_backward",
    "strict_forward",
    "trace",
]
