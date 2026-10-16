# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Autograd binding of the strict Muse-Glimmer gated GQA attention (contract O5).

``GatedAttentionFn`` glues the sibling strict forward
(``rl_engine.models.muse_glimmer.triton.gated_attention_fwd.strict_gated_attention_fwd_varlen``)
to the deterministic Triton backward in
``rl_engine.models.muse_glimmer.triton.gated_attention_bwd``. While the sibling forward is not
importable, the forward falls back to the fixed-order fp32 golden
(:func:`rl_engine.models.muse_glimmer.ops.attention_golden_bwd.golden_forward_fixed_order`);
the fallback is reported through :func:`trace` / :data:`FORWARD_IMPL` and is NOT the strict
forward (it is slow and only guarantees the contract's casts, not the kernel's bits).

Saved for backward: q, k, v, gate_logits, o (pre-gate bf16), lse (fp32), cu_seqlens, q_pos.
Gradients returned: dq (Tq,32,128) bf16, dk/dv (Tk,2,128) bf16 (canonical tree over the 8
Q-head-group partials), dgate (Tq,4096) bf16; None for the index tensors and ``window``.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch.autograd.function import once_differentiable

from rl_engine.models.muse_glimmer.ops.attention_golden_bwd import golden_forward_fixed_order
from rl_engine.models.muse_glimmer.triton import gated_attention_bwd as _bwd

try:
    from rl_engine.ops.autograd.backward_runtime import record_backward
except ImportError:  # pragma: no cover

    def record_backward(*_args, **_kwargs) -> None:
        return None


try:  # sibling agent's forward; may not exist in this worktree yet
    from rl_engine.models.muse_glimmer.ops.gated_attention import (
        strict_gated_attention_fwd_varlen as _strict_fwd,
    )

    FORWARD_IMPL = "rl_engine.models.muse_glimmer.triton.gated_attention_fwd"
except ImportError:  # pragma: no cover - depends on the sibling file
    _strict_fwd = None
    FORWARD_IMPL = "FALLBACK:attention_golden_bwd.golden_forward_fixed_order (fp32 fixed-order)"


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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(y bf16 (Tq,4096), o bf16 (Tq,4096), lse fp32 (Tq,32))`` - sibling kernel or fallback."""
    if _strict_fwd is not None:
        return _strict_fwd(q, k, v, gate_logits, cu_seqlens_q, cu_seqlens_k, q_pos, window=window)
    return golden_forward_fixed_order(
        q, k, v, gate_logits, cu_seqlens_q, cu_seqlens_k, q_pos, window=window
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
    ) -> torch.Tensor:
        q_c = q.contiguous()
        k_c = k.contiguous()
        v_c = v.contiguous()
        g_c = gate_logits.contiguous()
        y, o, lse = strict_forward(
            q_c, k_c, v_c, g_c, cu_seqlens_q, cu_seqlens_k, q_pos, window=window
        )
        ctx.save_for_backward(q_c, k_c, v_c, g_c, o, lse, cu_seqlens_q, cu_seqlens_k, q_pos)
        ctx.window = None if window is None else int(window)
        return y

    @staticmethod
    @once_differentiable
    def backward(ctx, dy: torch.Tensor):  # type: ignore[override]
        q, k, v, g, o, lse, cu_q, cu_k, q_pos = ctx.saved_tensors
        dq, dk, dv, dgate = _bwd.strict_gated_attention_bwd(
            q, k, v, g, o, lse, dy, cu_q, cu_k, q_pos, window=ctx.window
        )
        record_backward(
            "muse_glimmer.gated_attention",
            kernel_id=_bwd.KERNEL_ID,
            impl="triton",
            family="triton",
        )
        return dq, dk, dv, dgate, None, None, None, None


def gated_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate_logits: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    q_pos: torch.Tensor,
    window: Optional[int],
) -> torch.Tensor:
    """Convenience wrapper: ``y bf16 (Tq, 4096)`` with the strict backward attached."""
    return GatedAttentionFn.apply(q, k, v, gate_logits, cu_seqlens_q, cu_seqlens_k, q_pos, window)


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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Direct (non-autograd) entry: ``(dq, dk, dv, dgate)`` bf16 from saved forward tensors."""
    return _bwd.strict_gated_attention_bwd(
        q, k, v, gate_logits, o, lse, dy, cu_seqlens_q, cu_seqlens_k, q_pos, window=window
    )


def trace() -> dict:
    """Provenance of the autograd op (contract R4): forward impl + backward kernel trace."""
    rec = dict(_bwd.trace())
    rec["op"] = "muse_glimmer.gated_attention"
    rec["forward_impl"] = FORWARD_IMPL
    rec["forward_is_fallback"] = _strict_fwd is None
    return rec


__all__ = [
    "FORWARD_IMPL",
    "GatedAttentionFn",
    "gated_attention",
    "gated_attention_backward",
    "strict_forward",
    "trace",
]
