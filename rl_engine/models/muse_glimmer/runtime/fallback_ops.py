# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Labelled NON-STRICT fallbacks for the Muse Glimmer strict operator set (RFC #501).

These exist so that the vLLM runtime adapter (row ``vllm_runtime_adapter``) runs end to end before
the sibling strict Triton ops (``rl_engine.models.muse_glimmer.ops.*``) are merged.  Every function
here is fp32 fixed-order, repeatable and batch-invariant, but it is NOT the strict contract kernel:
its bits differ from the strict ops and provenance must label it ``fallback:*`` (never "strict").

Reduction order / cast conventions shared by every fallback
-----------------------------------------------------------
* GEMM: :func:`gemm_fp32` launches the repo's existing fixed-order Triton kernel
  (``rl_engine.backends.shared.triton.gemm.det_gemm._det_gemm_fp32_kernel``: one program per
  output tile, ``BLOCK_K = 32`` walked in ascending k, fp32 FMA accumulation, no split-K, no
  autotune), so the per-element reduction order never depends on M (batch invariant).  The
  canonical 8-chunk K split and the fixed tree ``((0+1)+(2+3))+((4+5)+(6+7))`` of contract O3 are
  applied on top of it by :func:`det_gemm_partial` / :func:`det_gemm`.  Without Triton the GEMM
  degrades to fp32 cuBLAS (deterministic per shape, NOT batch-invariant) and says so in its label.
* Row reductions (norm sums, softmax row sums, vocab-tile sums) use :func:`fixed_sum_last`: a
  pairwise halving tree over the largest power-of-two factor of the row length followed by a
  sequential sum over the remaining odd factor.  Every add is an elementwise fp32 add, so the order
  never depends on the batch size or the batch position.
* Casts to bf16 happen exactly where the contract says "-> bf16" and nowhere else.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch
import torch.nn.functional as F

try:  # the sibling tree module is tiny and pure torch; prefer it when merged
    from rl_engine.models.muse_glimmer.ops.canonical_tree import (
        CANONICAL_K_CHUNKS,
        combine_tree,
        rank_chunk_ids,
    )
except ImportError:  # pragma: no cover - exercised only before the sibling lands
    CANONICAL_K_CHUNKS = 8

    def combine_tree(partials: Sequence[torch.Tensor]) -> torch.Tensor:
        """``((0+1)+(2+3))+((4+5)+(6+7))`` over 1, 2, 4 or 8 fp32 partials (exact IEEE adds)."""

        parts = list(partials)
        if len(parts) not in (1, 2, 4, 8):
            raise ValueError(f"combine_tree expects 1, 2, 4 or 8 partials, got {len(parts)}")
        for part in parts:
            if part.dtype != torch.float32:
                raise TypeError("combine_tree partials must be float32")
        if len(parts) == 1:
            return parts[0]
        if len(parts) == 2:
            return parts[0] + parts[1]
        if len(parts) == 4:
            return (parts[0] + parts[1]) + (parts[2] + parts[3])
        lower = (parts[0] + parts[1]) + (parts[2] + parts[3])
        upper = (parts[4] + parts[5]) + (parts[6] + parts[7])
        return lower + upper

    def rank_chunk_ids(tp_size: int, rank: int) -> tuple[int, ...]:
        """Canonical chunk ids owned by ``rank`` at TP ``tp_size`` (1, 2, 4, 8)."""

        if tp_size not in (1, 2, 4, 8) or not 0 <= rank < tp_size:
            raise ValueError(f"invalid tp_size/rank {tp_size}/{rank}")
        per_rank = CANONICAL_K_CHUNKS // tp_size
        return tuple(range(rank * per_rank, (rank + 1) * per_rank))


try:
    import triton

    from rl_engine.backends.shared.triton.gemm import det_gemm as _repo_det_gemm

    _FP32_KERNEL: Any = getattr(_repo_det_gemm, "_det_gemm_fp32_kernel", None)
except Exception:  # pragma: no cover - Triton missing
    triton = None  # type: ignore[assignment]
    _FP32_KERNEL = None

_BLOCK_M, _BLOCK_N, _BLOCK_K = 64, 64, 32
BLOCK_N_KEYS = 64  # contract O5 key block
NEG_INF = float("-inf")

GEMM_LABEL = (
    "fallback:repo_triton_fixed_order_fp32_gemm(BLOCK 64x64x32, chunked K tree)"
    if _FP32_KERNEL is not None
    else "fallback:torch_cublas_fp32_chunked_k_tree (NOT batch-invariant)"
)


def fixed_sum_last(x: torch.Tensor) -> torch.Tensor:
    """Fixed-order fp32 sum over the last dim: pairwise halving tree, then a sequential tail.

    ``n = m * 2**k`` with ``m`` odd: the row is viewed as ``(m, 2**k)``, the power-of-two axis is
    halved ``k`` times with elementwise adds, and the ``m`` partials are added left to right.
    """

    if x.dtype != torch.float32:
        raise TypeError("fixed_sum_last expects float32")
    n = x.shape[-1]
    if n == 0:
        return torch.zeros(x.shape[:-1], dtype=torch.float32, device=x.device)
    k = n & -n
    m = n // k
    y = x.reshape(*x.shape[:-1], m, k)
    while y.shape[-1] > 1:
        y = y[..., 0::2] + y[..., 1::2]
    y = y[..., 0]
    out = y[..., 0]
    for index in range(1, m):
        out = out + y[..., index]
    return out


# ----------------------------------------------------------------------------------------------
# GEMM
# ----------------------------------------------------------------------------------------------


def gemm_fp32(a: torch.Tensor, b_kn: torch.Tensor) -> torch.Tensor:
    """Fixed-order fp32 ``a @ b_kn`` for ``a (M, K)`` and ``b_kn (K, N)`` with arbitrary strides.

    bf16 operands are promoted to fp32 inside the kernel (exact); accumulation is fp32 FMA in
    ascending k with no split-K, so the result is batch invariant and repeatable.
    """

    if a.dim() != 2 or b_kn.dim() != 2 or a.shape[1] != b_kn.shape[0]:
        raise ValueError(f"gemm_fp32 shape mismatch: {tuple(a.shape)} @ {tuple(b_kn.shape)}")
    m, k = a.shape
    n = b_kn.shape[1]
    if _FP32_KERNEL is None or not a.is_cuda:
        return torch.matmul(a.float(), b_kn.float())
    c = torch.empty((m, n), device=a.device, dtype=torch.float32)
    if m == 0 or n == 0:
        return c
    if k == 0:
        return c.zero_()
    if a.stride(1) != 1:
        a = a.contiguous()
    if b_kn.stride(0) != 1 and b_kn.stride(1) != 1:
        b_kn = b_kn.contiguous()
    grid = (triton.cdiv(m, _BLOCK_M), triton.cdiv(n, _BLOCK_N))
    _FP32_KERNEL[grid](
        a,
        b_kn,
        c,
        m,
        n,
        k,
        a.stride(0),
        a.stride(1),
        b_kn.stride(0),
        b_kn.stride(1),
        c.stride(0),
        c.stride(1),
        BLOCK_M=_BLOCK_M,
        BLOCK_N=_BLOCK_N,
        BLOCK_K=_BLOCK_K,
        PROMOTE_INPUTS=True,
    )
    return c


def det_gemm_partial(
    a_local: torch.Tensor,
    w_local: torch.Tensor,
    chunk_ids: Sequence[int],
    *,
    w_kn: bool = False,
) -> torch.Tensor:
    """fp32 sub-tree partial over the consecutive canonical chunks ``chunk_ids`` (contract O3).

    ``a_local (..., K_local)`` bf16, ``w_local (N, K_local)`` bf16 (or ``(K_local, N)`` when
    ``w_kn``).  Each chunk is one fixed-order fp32 GEMM; the chunk partials are combined with
    :func:`combine_tree`.  ``combine_tree`` over the ranks' partials equals the TP1 result bitwise.
    """

    ids = tuple(int(c) for c in chunk_ids)
    count = len(ids)
    if count not in (1, 2, 4, 8):
        raise ValueError(f"chunk_ids must hold 1, 2, 4 or 8 ids, got {count}")
    k_local = a_local.shape[-1]
    if k_local % count:
        raise ValueError(f"local K {k_local} is not divisible into {count} chunks")
    width = k_local // count
    a2 = a_local.reshape(-1, k_local)
    b_kn = w_local if w_kn else w_local.t()
    if b_kn.shape[0] != k_local:
        raise ValueError(f"weight K {b_kn.shape[0]} != activation K {k_local}")
    parts = [
        gemm_fp32(a2[:, i * width : (i + 1) * width], b_kn[i * width : (i + 1) * width])
        for i in range(count)
    ]
    return combine_tree(parts).reshape(*a_local.shape[:-1], b_kn.shape[1])


def det_gemm(
    a: torch.Tensor,
    w: torch.Tensor,
    *,
    out_dtype: torch.dtype = torch.bfloat16,
    k_chunks: int = CANONICAL_K_CHUNKS,
    w_kn: bool = False,
) -> torch.Tensor:
    """``a @ w.T`` (``w (N, K)``) through the chunked fixed-order fp32 GEMM, one cast at the end."""

    if k_chunks not in (1, 2, 4, 8):
        raise ValueError(f"k_chunks must be 1, 2, 4 or 8, got {k_chunks}")
    y = det_gemm_partial(a, w, tuple(range(k_chunks)), w_kn=w_kn)
    return y if out_dtype == torch.float32 else y.to(out_dtype)


# ----------------------------------------------------------------------------------------------
# Norms (contract O1 / O2)
# ----------------------------------------------------------------------------------------------


def _mean_square(xf: torch.Tensor) -> torch.Tensor:
    return fixed_sum_last(xf * xf) / float(xf.shape[-1])


def normalized_embedding(weight: torch.Tensor, ids: torch.Tensor, *, eps: float) -> torch.Tensor:
    """O1: ``E[token] -> fp32``, ``y = x * (mean(x^2) + eps)^(-1/2)`` -> bf16 (weightless)."""

    rows = weight[ids.to(torch.long)]
    xf = rows.float()
    inv = torch.pow(_mean_square(xf) + eps, -0.5)
    return (xf * inv.unsqueeze(-1)).to(torch.bfloat16)


def centered_rmsnorm(x: torch.Tensor, w: torch.Tensor, *, eps: float) -> torch.Tensor:
    """O2: ``x_hat = x * rsqrt(mean(x^2) + eps)``, ``y = x_hat * (1 + w)`` in fp32 -> bf16."""

    xf = x.float()
    x_hat = xf * torch.rsqrt(_mean_square(xf) + eps).unsqueeze(-1)
    return (x_hat * (w.float() + 1.0)).to(torch.bfloat16)


def weighted_rmsnorm(x: torch.Tensor, w: torch.Tensor, *, eps: float) -> torch.Tensor:
    """Final norm: ``y = x * (mean(x^2) + eps)^(-1/2) * w`` in fp32 -> bf16 (NOT 1 + w)."""

    xf = x.float()
    x_hat = xf * torch.pow(_mean_square(xf) + eps, -0.5).unsqueeze(-1)
    return (x_hat * w.float()).to(torch.bfloat16)


def fused_post_norm_residual_pre_norm(
    branch: torch.Tensor,
    resid: torch.Tensor,
    w_post: torch.Tensor,
    eps_post: float,
    w_pre: torch.Tensor,
    eps_pre: float,
    *,
    pre_form: str = "centered",
) -> tuple[torch.Tensor, torch.Tensor]:
    """[F1]: ``r' = bf16(fp32(r) + fp32(post_norm(branch)))`` and ``pre_norm(r')``."""

    post = centered_rmsnorm(branch, w_post, eps=eps_post)
    new_resid = (resid.float() + post.float()).to(torch.bfloat16)
    if pre_form == "centered":
        pre = centered_rmsnorm(new_resid, w_pre, eps=eps_pre)
    elif pre_form == "weighted":
        pre = weighted_rmsnorm(new_resid, w_pre, eps=eps_pre)
    else:
        raise ValueError(f"unknown pre_form {pre_form!r}")
    return new_resid, pre


# ----------------------------------------------------------------------------------------------
# QK-norm + scale + RoPE (contract O4)
# ----------------------------------------------------------------------------------------------


def rope_table(
    positions: torch.Tensor, *, theta: float, dim: int, device: torch.device | str
) -> tuple[torch.Tensor, torch.Tensor]:
    """fp32 ``cos(p * theta^(-2i/dim))`` / ``sin`` tables of shape ``(T, dim)`` (angle repeated
    for both halves, the sibling ``rope_table`` layout)."""

    # One table builder for every runtime (contract O4): the canonical CUDA build of ops.qknorm_rope.
    from rl_engine.models.muse_glimmer.ops.qknorm_rope import rope_table as _canonical_rows

    return _canonical_rows(positions, theta=float(theta), dim=int(dim), device=device)


def _qk_norm(xf: torch.Tensor, eps: float) -> torch.Tensor:
    return xf * torch.pow(_mean_square(xf) + eps, -0.5).unsqueeze(-1)


def _rotate(xf: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    half = xf.shape[-1] // 2
    x1, x2 = xf[..., :half], xf[..., half:]
    # Tables are (T, dim) with identical halves; rotate_half only needs the first half.
    c = cos[:, :half].unsqueeze(1)
    s = sin[:, :half].unsqueeze(1)
    return torch.cat((x1 * c - x2 * s, x2 * c + x1 * s), dim=-1)


def qknorm_scaled_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor | None,
    sin: torch.Tensor | None,
    *,
    eps: float,
    qk_scale: float,
    local: bool | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """O4: per-head weightless RMSNorm (fp32), ``q *= qk_scale`` (fp32), rotate_half RoPE on
    local layers (``cos``/``sin`` given), one cast to bf16 at the end."""
    if local is not None and bool(local) != (cos is not None):
        raise ValueError(f"local={local} but cos/sin are {'None' if cos is None else 'given'}")

    qf = _qk_norm(q.float(), eps) * float(qk_scale)
    kf = _qk_norm(k.float(), eps)
    if cos is not None:
        assert sin is not None
        qf = _rotate(qf, cos, sin)
        kf = _rotate(kf, cos, sin)
    return qf.to(torch.bfloat16), kf.to(torch.bfloat16)


# ----------------------------------------------------------------------------------------------
# Gated attention (contract O5)
# ----------------------------------------------------------------------------------------------


def _attend_rows(
    q: torch.Tensor,
    q_pos: torch.Tensor,
    keys: torch.Tensor,
    vals: torch.Tensor,
    window: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Online softmax over 64-key blocks for one sequence; ``keys[i]`` sits at absolute position i.

    ``q (Lq, nh, D)`` bf16, ``q_pos (Lq,)`` int64, ``keys``/``vals (Lk, nkv, D)`` bf16.  Returns
    ``o (Lq, nh, D)`` bf16 and ``lse (Lq, nh)`` fp32.  The per-row arithmetic follows contract O5
    exactly (block order, ``m/alpha/p/l/acc`` update order, ``p -> bf16`` before PV, ``o = acc / l``).
    """

    lq, nh, d = q.shape
    lk, nkv, _ = keys.shape
    group = nh // nkv
    device = q.device
    scale = float(d) ** -0.5
    nblk = -(-lk // BLOCK_N_KEYS) if lk else 0
    pad = nblk * BLOCK_N_KEYS - lk
    if pad:
        keys = F.pad(keys, (0, 0, 0, 0, 0, pad))
        vals = F.pad(vals, (0, 0, 0, 0, 0, pad))
    p_min = int(q_pos.min().item())
    p_max = int(q_pos.max().item())
    j0 = 0 if window is None else max(0, p_min - int(window) + 1) // BLOCK_N_KEYS
    j1 = min(p_max // BLOCK_N_KEYS, nblk - 1)
    m = torch.full((lq, nh), NEG_INF, dtype=torch.float32, device=device)
    lsum = torch.zeros((lq, nh), dtype=torch.float32, device=device)
    acc = torch.zeros((lq, nh, d), dtype=torch.float32, device=device)
    pos_rows = q_pos.repeat_interleave(group)
    zero = torch.zeros((), dtype=torch.float32, device=device)
    for j in range(j0, j1 + 1):
        kpos = torch.arange(j * BLOCK_N_KEYS, (j + 1) * BLOCK_N_KEYS, device=device)
        visible = kpos[None, :] <= pos_rows[:, None]
        if window is not None:
            visible &= kpos[None, :] > (pos_rows[:, None] - int(window))
        for g in range(nkv):
            heads = slice(g * group, (g + 1) * group)
            qg = q[:, heads, :].reshape(lq * group, d)
            k_blk = keys[j * BLOCK_N_KEYS : (j + 1) * BLOCK_N_KEYS, g, :]
            v_blk = vals[j * BLOCK_N_KEYS : (j + 1) * BLOCK_N_KEYS, g, :]
            scores = gemm_fp32(qg, k_blk.t()) * scale
            scores = torch.where(visible, scores, NEG_INF)
            m_old = m[:, heads].reshape(-1)
            m_new = torch.maximum(m_old, scores.max(dim=-1).values)
            m_use = torch.where(m_new == NEG_INF, zero, m_new)
            alpha = torch.exp(m_old - m_use)
            p = torch.exp(scores - m_use[:, None])
            l_new = lsum[:, heads].reshape(-1) * alpha + fixed_sum_last(p)
            pv = gemm_fp32(p.to(torch.bfloat16), v_blk)
            acc_new = acc[:, heads, :].reshape(lq * group, d) * alpha[:, None] + pv
            m[:, heads] = m_new.view(lq, group)
            lsum[:, heads] = l_new.view(lq, group)
            acc[:, heads, :] = acc_new.view(lq, group, d)
    o = (acc / lsum.unsqueeze(-1)).to(torch.bfloat16)
    lse = m + torch.log(lsum)
    return o, lse


def _check_heads(nh: int, nkv: int, num_q_heads: int | None, num_kv_heads: int | None) -> None:
    if num_q_heads is not None and int(num_q_heads) != int(nh):
        raise ValueError(f"num_q_heads={num_q_heads} but q has {nh} heads")
    if num_kv_heads is not None and int(num_kv_heads) != int(nkv):
        raise ValueError(f"num_kv_heads={num_kv_heads} but the K/V tensors have {nkv} heads")
    if nh % nkv:
        raise ValueError(f"{nh} query heads are not a multiple of {nkv} KV heads")


def _apply_gate(o: torch.Tensor, gate_logits: torch.Tensor) -> torch.Tensor:
    t = o.shape[0]
    o_flat = o.reshape(t, -1)
    y = o_flat.float() * torch.sigmoid(gate_logits.reshape(t, -1).float())
    return y.to(torch.bfloat16)


def gated_attention_fwd_paged(
    q: torch.Tensor,
    gate_logits: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    *,
    window: int | None,
    max_seqlen_q: int | None = None,
    num_q_heads: int | None = None,
    num_kv_heads: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Paged rollout attention: ``q (T, nh, D)``, caches ``(P, page, nkv, D)`` (any strides),
    ``block_table (B, maxP)``, ``seq_lens (B,)``, ``cu_seqlens_q (B+1,)``.  Query ``i`` of request
    ``b`` sits at absolute position ``seq_lens[b] - len_q[b] + i``.  Returns ``(y, o, lse)`` with
    ``y = o * sigmoid(gate)`` (bf16, ``(T, nh*D)``).  ``num_q_heads``/``num_kv_heads`` (the strict
    op's per-rank head counts) are checked against the shapes when given."""

    t, nh, d = q.shape
    nkv = k_cache.shape[2]
    _check_heads(nh, nkv, num_q_heads, num_kv_heads)
    page = k_cache.shape[1]
    cu = cu_seqlens_q.tolist()
    lens = seq_lens.tolist()
    o = torch.empty((t, nh, d), dtype=torch.bfloat16, device=q.device)
    lse = torch.empty((t, nh), dtype=torch.float32, device=q.device)
    for b in range(len(lens)):
        qs, qe = int(cu[b]), int(cu[b + 1])
        lq = qe - qs
        if lq == 0:
            continue
        length = int(lens[b])
        npages = -(-length // page)
        # ``-1`` (strict convention) / ``0`` (vLLM null block) mark absent or evicted pages; both
        # read the all-zero null page, whose keys are masked anyway.
        ids = block_table[b, :npages].to(torch.long).clamp(min=0)
        keys = k_cache[ids].reshape(npages * page, nkv, d)[:length]
        vals = v_cache[ids].reshape(npages * page, nkv, d)[:length]
        q_pos = torch.arange(length - lq, length, device=q.device, dtype=torch.int64)
        o_b, lse_b = _attend_rows(q[qs:qe], q_pos, keys, vals, window)
        o[qs:qe] = o_b
        lse[qs:qe] = lse_b
    return _apply_gate(o, gate_logits), o.reshape(t, nh * d), lse


def gated_attention_fwd_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate_logits: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    q_pos: torch.Tensor,
    *,
    window: int | None,
    num_q_heads: int | None = None,
    num_kv_heads: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Packed (training/prefill) attention; ``k``/``v`` rows of sequence ``b`` are its keys at
    absolute positions ``0..len_k[b]-1`` and ``q_pos`` holds the absolute query positions."""

    t, nh, d = q.shape
    _check_heads(nh, k.shape[1], num_q_heads, num_kv_heads)
    cu_q = cu_seqlens_q.tolist()
    cu_k = cu_seqlens_k.tolist()
    o = torch.empty((t, nh, d), dtype=torch.bfloat16, device=q.device)
    lse = torch.empty((t, nh), dtype=torch.float32, device=q.device)
    for b in range(len(cu_q) - 1):
        qs, qe = int(cu_q[b]), int(cu_q[b + 1])
        if qe == qs:
            continue
        ks, ke = int(cu_k[b]), int(cu_k[b + 1])
        o_b, lse_b = _attend_rows(
            q[qs:qe], q_pos[qs:qe].to(torch.int64), k[ks:ke], v[ks:ke], window
        )
        o[qs:qe] = o_b
        lse[qs:qe] = lse_b
    return _apply_gate(o, gate_logits), o.reshape(t, nh * d), lse


# ----------------------------------------------------------------------------------------------
# SwiGLU (contract O6)
# ----------------------------------------------------------------------------------------------


def swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """``h = silu(g) * u`` in fp32 with ``silu(x) = x / (1 + exp(-x))`` -> bf16."""

    gf = gate.float()
    h = (gf / (1.0 + torch.exp(-gf))) * up.float()
    return h.to(torch.bfloat16)


# ----------------------------------------------------------------------------------------------
# LM head softcap log-prob (contract O7)
# ----------------------------------------------------------------------------------------------

VOCAB_TILE_ROWS = 25344


def softcap_tile_stats(
    logits_tile: torch.Tensor,
    tile_id: int,
    targets: torch.Tensor,
    *,
    multiplier: float,
    softcap: float,
    vocab_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-tile ``(M_t, S_t, target_logit, target_in_tile)`` for fp32 ``logits_tile (T, V_t)``.

    ``z_hat = T * tanh((z * m) / T)`` in fp32 (order: mul multiplier, div T, tanh, mul T); padded
    rows (global index >= vocab_size) are excluded; ``S_t`` is a fixed-order sum of ``exp(z_hat -
    M_t)``.
    """

    if logits_tile.dtype != torch.float32:
        raise TypeError("softcap_tile_stats expects fp32 logits")
    t, width = logits_tile.shape
    base = int(tile_id) * width
    z_hat = float(softcap) * torch.tanh((logits_tile * float(multiplier)) / float(softcap))
    global_index = torch.arange(base, base + width, device=logits_tile.device)
    valid = global_index < int(vocab_size)
    z_hat = torch.where(valid[None, :], z_hat, NEG_INF)
    m_t = z_hat.max(dim=-1).values
    e = torch.exp(z_hat - m_t[:, None])
    e = torch.where(valid[None, :], e, torch.zeros((), dtype=torch.float32, device=e.device))
    s_t = fixed_sum_last(e)
    local = targets.to(torch.long) - base
    in_tile = (local >= 0) & (local < width) & (targets.to(torch.long) < int(vocab_size))
    gathered = z_hat.gather(1, local.clamp(0, width - 1).unsqueeze(1)).squeeze(1)
    target_logit = torch.where(in_tile, gathered, torch.zeros_like(gathered))
    return m_t, s_t, target_logit, in_tile


def softcap_logits(z: torch.Tensor, *, multiplier: float, softcap: float) -> torch.Tensor:
    """``z_hat = T * tanh((z * m) / T)`` elementwise in fp32 (the order of
    :func:`softcap_tile_stats`), tensor-valued constants so the division is IEEE."""

    if z.dtype != torch.float32:
        raise TypeError("softcap_logits expects fp32 logits")
    mult = torch.tensor(float(multiplier), dtype=torch.float32, device=z.device)
    cap = torch.tensor(float(softcap), dtype=torch.float32, device=z.device)
    return cap * torch.tanh((z * mult) / cap)


def _merge_stats(
    a: tuple[torch.Tensor, torch.Tensor], b: tuple[torch.Tensor, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor]:
    m_a, s_a = a
    m_b, s_b = b
    m = torch.maximum(m_a, m_b)
    return m, s_a * torch.exp(m_a - m) + s_b * torch.exp(m_b - m)


def combine_tile_stats(
    stats: Sequence[Sequence[torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Combine 8 (or 1/2/4) tile ``(M_t, S_t, ...)`` records in the fixed tree -> ``(M, logZ)``."""

    pairs = [(s[0], s[1]) for s in stats]
    if len(pairs) not in (1, 2, 4, 8):
        raise ValueError(f"combine_tile_stats expects 1, 2, 4 or 8 tiles, got {len(pairs)}")
    while len(pairs) > 1:
        pairs = [_merge_stats(pairs[i], pairs[i + 1]) for i in range(0, len(pairs), 2)]
    m, s = pairs[0]
    return m, m + torch.log(s)


def target_logit_from_stats(stats: Sequence[Sequence[torch.Tensor]]) -> torch.Tensor:
    """Fixed-tree sum of ``target_logit`` over tiles (exactly one tile is non-zero per row)."""

    parts = [torch.where(s[3].bool(), s[2], torch.zeros_like(s[2])) for s in stats]
    while len(parts) > 1:
        parts = [parts[i] + parts[i + 1] for i in range(0, len(parts), 2)]
    return parts[0]


def pad_lm_head_weight(w: torch.Tensor, padded_rows: int = 8 * VOCAB_TILE_ROWS) -> torch.Tensor:
    """Zero-pad ``(V, H)`` to ``(padded_rows, H)``."""

    if w.shape[0] == padded_rows:
        return w
    out = torch.zeros((padded_rows, w.shape[1]), dtype=w.dtype, device=w.device)
    out[: w.shape[0]] = w
    return out


def lm_head_softcap_logprob(
    h: torch.Tensor,
    w_padded: torch.Tensor,
    targets: torch.Tensor,
    *,
    multiplier: float,
    softcap: float,
    vocab_size: int,
    gemm: Any | None = None,
) -> torch.Tensor:
    """TP1 reference path: 8 vocab tiles, fp32 logits, softcap, tree-combined logsumexp."""

    rows = w_padded.shape[0]
    if rows % 8:
        raise ValueError("padded lm_head rows must split into 8 tiles")
    width = rows // 8
    stats = []
    for tile in range(8):
        w_tile = w_padded[tile * width : (tile + 1) * width]
        z = gemm(h, w_tile) if gemm is not None else det_gemm(h, w_tile, out_dtype=torch.float32)
        stats.append(
            softcap_tile_stats(
                z, tile, targets, multiplier=multiplier, softcap=softcap, vocab_size=vocab_size
            )
        )
    _, log_z = combine_tile_stats(stats)
    return target_logit_from_stats(stats) - log_z


def sqrt_head_dim_scale(head_dim: int) -> float:
    return math.sqrt(float(head_dim)) ** -1.0


__all__ = [
    "BLOCK_N_KEYS",
    "CANONICAL_K_CHUNKS",
    "GEMM_LABEL",
    "VOCAB_TILE_ROWS",
    "centered_rmsnorm",
    "combine_tile_stats",
    "combine_tree",
    "det_gemm",
    "det_gemm_partial",
    "fixed_sum_last",
    "fused_post_norm_residual_pre_norm",
    "gated_attention_fwd_paged",
    "gated_attention_fwd_varlen",
    "gemm_fp32",
    "lm_head_softcap_logprob",
    "normalized_embedding",
    "pad_lm_head_weight",
    "qknorm_scaled_rope",
    "rank_chunk_ids",
    "rope_table",
    "softcap_logits",
    "softcap_tile_stats",
    "swiglu",
    "target_logit_from_stats",
    "weighted_rmsnorm",
]
