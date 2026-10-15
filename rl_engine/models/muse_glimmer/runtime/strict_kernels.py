# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Strict operator resolution for the Muse Glimmer Megatron Bridge binding.

The binding is written against the FIXED interfaces of the sibling strict ops
(``rl_engine.models.muse_glimmer.ops.*``). Every op is resolved at import time; when a
sibling op is missing, a clearly labelled pure-torch fallback is used instead. Fallbacks are
reported as ``fallback:<detail>`` in :func:`provenance` and are NEVER labelled strict.

Numerical conventions shared by every path here (contract O3/O7):

* GEMMs take ``a`` as ``(M, K)`` bf16 and ``w`` as ``(N, K)`` bf16 (native linear layout),
  accumulate in fp32 and return fp32; the caller decides where the single bf16 rounding is.
* A reduction dimension is split into 8 canonical equal chunks; chunk partials are combined
  with the fixed binary tree ``((0+1)+(2+3))+((4+5)+(6+7))``. At TP ``t`` a rank owns ``8/t``
  consecutive chunks, reduces them with the sub-tree, all-gathers the fp32 partials in rank
  order (data movement only; never a reducing NCCL collective) and applies the same tree.
"""

from __future__ import annotations

import importlib
import math
import os
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import torch
import torch.distributed as dist

CANONICAL_CHUNKS = 8
_TILE_PAD = 64
_NEG_INF = float("-inf")


@dataclass(frozen=True)
class OpSource:
    """Where an operator came from; ``strict`` is True only for sibling strict kernels."""

    name: str
    strict: bool
    detail: str

    @property
    def label(self) -> str:
        return ("strict:" if self.strict else "fallback:") + self.detail


def _optional_module(path: str) -> Any | None:
    try:
        return importlib.import_module(path)
    except Exception:  # noqa: BLE001 - sibling ops may be absent or half-landed
        return None


def _optional_attr(module: Any | None, *names: str) -> Callable[..., Any] | None:
    if module is None:
        return None
    for name in names:
        fn = getattr(module, name, None)
        if callable(fn):
            return fn
    return None


# --------------------------------------------------------------------------------------
# Canonical tree and chunked GEMM
# --------------------------------------------------------------------------------------


def combine_tree_torch(partials: Sequence[torch.Tensor]) -> torch.Tensor:
    """Fixed binary tree over a power-of-two list of fp32 partials, lower index first."""
    items = list(partials)
    if not items:
        raise ValueError("combine_tree requires at least one partial")
    if len(items) & (len(items) - 1):
        raise ValueError(f"combine_tree requires a power-of-two count, got {len(items)}")
    while len(items) > 1:
        items = [items[i] + items[i + 1] for i in range(0, len(items), 2)]
    return items[0]


def chunk_slices(k_total: int, chunk_ids: Sequence[int]) -> list[slice]:
    if k_total % CANONICAL_CHUNKS:
        raise ValueError(f"K={k_total} is not divisible into {CANONICAL_CHUNKS} canonical chunks")
    width = k_total // CANONICAL_CHUNKS
    return [slice(c * width, (c + 1) * width) for c in chunk_ids]


def local_chunk_ids(rank: int, world: int) -> list[int]:
    """Canonical chunk ids owned by ``rank`` at TP ``world`` (contiguous, rank ordered)."""
    if world not in (1, 2, 4, 8) or CANONICAL_CHUNKS % world:
        raise ValueError(f"TP world {world} is not a divisor of {CANONICAL_CHUNKS}")
    per = CANONICAL_CHUNKS // world
    return list(range(rank * per, (rank + 1) * per))


class _GemmBackend:
    """``partial(a, w, chunk_ids) -> fp32`` over ``len(chunk_ids)`` equal K-chunks of ``a``/``w``."""

    def __init__(self) -> None:
        requested = os.getenv("MUSE_STRICT_GEMM", "auto").strip().lower()
        self.source: OpSource
        self._partial: Callable[..., torch.Tensor]
        self._combine: Callable[[Sequence[torch.Tensor]], torch.Tensor] = combine_tree_torch
        sibling = _optional_module("rl_engine.models.muse_glimmer.ops.det_gemm")
        tree_mod = _optional_module("rl_engine.models.muse_glimmer.ops.canonical_tree")
        sibling_partial = _optional_attr(sibling, "fast_det_gemm_partial")
        sibling_tree = _optional_attr(tree_mod, "combine_tree")
        if requested in ("auto", "sibling") and sibling_partial is not None:
            self._partial = self._sibling_partial
            self._sibling_fn = sibling_partial
            if sibling_tree is not None:
                self._combine = sibling_tree
            self.source = OpSource("gemm", True, "ops.det_gemm.fast_det_gemm_partial")
            return
        if requested == "sibling":
            raise RuntimeError("MUSE_STRICT_GEMM=sibling but ops.det_gemm.fast_det_gemm_partial is absent")
        repo = _optional_module("rl_engine.backends.shared.triton.gemm.det_gemm")
        repo_cls = getattr(repo, "TritonDetGemmOp", None) if repo is not None else None
        if requested in ("auto", "triton_repo") and repo_cls is not None:
            try:
                self._repo_op = repo_cls()
            except Exception as exc:  # noqa: BLE001
                if requested == "triton_repo":
                    raise
                self._repo_op = None
                detail = f"repo TritonDetGemmOp unavailable ({exc})"
            else:
                self._partial = self._repo_partial
                self.source = OpSource(
                    "gemm", False, "repo_triton_det_gemm_per_chunk_tree (batch-invariant, slow)"
                )
                return
        else:
            detail = "repo TritonDetGemmOp not selected"
        if requested == "triton_repo":
            raise RuntimeError(f"MUSE_STRICT_GEMM=triton_repo but {detail}")
        self._partial = self._cublas_partial
        self.source = OpSource("gemm", False, "cublas_fp32_per_chunk_tree (not batch-invariant)")

    # -- backends -------------------------------------------------------------------------
    def _sibling_partial(self, a: torch.Tensor, w: torch.Tensor, chunk_ids: Sequence[int]):
        return self._sibling_fn(a, w, list(chunk_ids))

    def _repo_partial(self, a: torch.Tensor, w: torch.Tensor, chunk_ids: Sequence[int]):
        parts = []
        for sl in chunk_slices(a.shape[1] * CANONICAL_CHUNKS // len(chunk_ids), range(len(chunk_ids))):
            a_c = a[:, sl].contiguous()
            b_c = w[:, sl].t().contiguous()
            parts.append(self._repo_op.forward_fp32(a_c, b_c))
        return self._combine(parts)

    @staticmethod
    def _cublas_partial(a: torch.Tensor, w: torch.Tensor, chunk_ids: Sequence[int]):
        parts = []
        for sl in chunk_slices(a.shape[1] * CANONICAL_CHUNKS // len(chunk_ids), range(len(chunk_ids))):
            parts.append(torch.mm(a[:, sl].float(), w[:, sl].float().t()))
        return combine_tree_torch(parts)

    # -- public ---------------------------------------------------------------------------
    def partial(self, a: torch.Tensor, w: torch.Tensor, chunk_ids: Sequence[int]) -> torch.Tensor:
        """fp32 ``(M, N)`` sub-tree partial of ``a @ w.T`` over the listed canonical chunks.

        ``a`` is ``(M, K_local)`` and ``w`` is ``(N, K_local)`` with
        ``K_local = len(chunk_ids) * chunk_width``. Rows (M) and output columns (N) are padded
        with zeros to multiples of 64 and sliced back; zero rows/columns cannot change the
        bits of other outputs. K is never padded (it would move the chunk boundaries).
        """
        if a.dtype != torch.bfloat16 or w.dtype != torch.bfloat16:
            raise TypeError("strict GEMM operands must be bf16")
        if a.ndim != 2 or w.ndim != 2 or a.shape[1] != w.shape[1]:
            raise ValueError(f"GEMM shape mismatch a={tuple(a.shape)} w={tuple(w.shape)}")
        n_chunks = len(chunk_ids)
        if n_chunks not in (1, 2, 4, 8) or a.shape[1] % n_chunks:
            raise ValueError(f"chunk_ids={list(chunk_ids)} does not divide K_local={a.shape[1]}")
        m, n = a.shape[0], w.shape[0]
        m_pad = -(-m // _TILE_PAD) * _TILE_PAD
        n_pad = -(-n // _TILE_PAD) * _TILE_PAD
        a_in = a if m_pad == m else torch.nn.functional.pad(a, (0, 0, 0, m_pad - m))
        w_in = w if n_pad == n else torch.nn.functional.pad(w, (0, 0, 0, n_pad - n))
        out = self._partial(a_in.contiguous(), w_in.contiguous(), chunk_ids)
        if out.dtype != torch.float32:
            raise TypeError("GEMM partial backend must return fp32")
        if m_pad != m or n_pad != n:
            out = out[:m, :n]
        return out

    def combine(self, partials: Sequence[torch.Tensor]) -> torch.Tensor:
        return self._combine(list(partials))


_GEMM: _GemmBackend | None = None


def gemm_backend() -> _GemmBackend:
    global _GEMM
    if _GEMM is None:
        _GEMM = _GemmBackend()
    return _GEMM


def gemm_fp32(a: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Full 8-chunk canonical GEMM ``a @ w.T`` → fp32 (no TP split)."""
    return gemm_backend().partial(a, w, list(range(CANONICAL_CHUNKS)))


def gemm_single_chunk_fp32(a: torch.Tensor, w: torch.Tensor, chunk_id: int) -> torch.Tensor:
    """One canonical chunk as a whole (sequential accumulation inside the kernel)."""
    return gemm_backend().partial(a, w, [chunk_id])


def small_k_fp32(z: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``z @ b.T`` for tiny K (the LoRA rank): fixed-order fp32 loop over K.

    ``z`` is ``(M, r)`` bf16 and ``b`` is ``(N, r)`` bf16. Every bf16×bf16 product is exact in
    fp32, so the only roundings are the ``r`` sequential fp32 additions in index order; the
    result is deterministic and batch invariant by construction (elementwise kernels only).
    """
    if z.dtype != torch.bfloat16 or b.dtype != torch.bfloat16:
        raise TypeError("small_k_fp32 operands must be bf16")
    r = z.shape[1]
    if b.shape[1] != r:
        raise ValueError("small_k_fp32 rank mismatch")
    acc = torch.zeros(z.shape[0], b.shape[0], dtype=torch.float32, device=z.device)
    zf = z.float()
    bf = b.float()
    for k in range(r):
        prod = zf[:, k : k + 1] * bf[:, k].unsqueeze(0)
        acc = acc + prod
    return acc


# --------------------------------------------------------------------------------------
# TP all-gather (data movement only)
# --------------------------------------------------------------------------------------


class TPComm:
    """Rank-ordered all-gather of fp32 partials; never a reducing collective.

    Uses the repo deterministic collective when the CUDA extension is present, otherwise
    ``torch.distributed.all_gather`` into rank-ordered slots (bitwise data movement).
    """

    def __init__(self, group: dist.ProcessGroup | None) -> None:
        self.group = group
        self.world = 1
        self.rank = 0
        self.backend = "none"
        self._collective: Any | None = None
        if group is not None and dist.is_available() and dist.is_initialized():
            self.world = dist.get_world_size(group=group)
            self.rank = dist.get_rank(group=group)
        if self.world == 1:
            return
        try:
            from rl_engine.distributed.algorithms.collectives import create_deterministic_collective

            self._collective = create_deterministic_collective(group, torch.cuda.current_device())
            self.backend = str(getattr(self._collective, "backend_id", "deterministic_collective"))
        except Exception:  # noqa: BLE001 - extension absent: fall back to NCCL all_gather
            self._collective = None
            self.backend = "nccl_all_gather_data_movement"

    def all_gather(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Return ``[x_rank0, x_rank1, ...]`` with identical bit patterns on every rank."""
        if self.world == 1:
            return [x]
        x = x.contiguous()
        if self._collective is not None:
            out = self._collective.all_gather(x)
            return list(out.split(x.shape[0], dim=0))
        slots = [torch.empty_like(x) for _ in range(self.world)]
        dist.all_gather(slots, x, group=self.group)
        return slots

    def all_gather_tree(self, local_partial: torch.Tensor) -> torch.Tensor:
        """Sub-tree partial in, full-tree total out (identical on every rank)."""
        if self.world == 1:
            return local_partial
        return gemm_backend().combine(self.all_gather(local_partial))


# --------------------------------------------------------------------------------------
# Row-local ops (norms, embedding, qk-norm + RoPE, swiglu, attention)
# --------------------------------------------------------------------------------------


def _fb_normalized_embedding(weight: torch.Tensor, ids: torch.Tensor, *, eps: float) -> torch.Tensor:
    x = weight[ids].float()
    ms = x.pow(2).mean(-1, keepdim=True)
    return (x * torch.pow(ms + eps, -0.5)).to(torch.bfloat16)


def _fb_centered_rmsnorm(x: torch.Tensor, w: torch.Tensor, *, eps: float) -> torch.Tensor:
    xf = x.float()
    xhat = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (xhat * (1.0 + w.float())).to(torch.bfloat16)


def _fb_weighted_rmsnorm(x: torch.Tensor, w: torch.Tensor, *, eps: float) -> torch.Tensor:
    xf = x.float()
    xhat = xf * torch.pow(xf.pow(2).mean(-1, keepdim=True) + eps, -0.5)
    return (xhat * w.float()).to(torch.bfloat16)


def _fb_fused_post_norm_residual_pre_norm(
    branch: torch.Tensor,
    resid: torch.Tensor,
    w_post: torch.Tensor,
    eps_post: float,
    w_pre: torch.Tensor,
    eps_pre: float,
    *,
    pre_form: str = "centered",
) -> tuple[torch.Tensor, torch.Tensor]:
    post = _fb_centered_rmsnorm(branch, w_post, eps=eps_post)
    new_resid = (resid.float() + post.float()).to(torch.bfloat16)
    if pre_form == "centered":
        pre = _fb_centered_rmsnorm(new_resid, w_pre, eps=eps_pre)
    elif pre_form == "weighted":
        pre = _fb_weighted_rmsnorm(new_resid, w_pre, eps=eps_pre)
    else:
        raise ValueError(f"unknown pre_form {pre_form!r}")
    return new_resid, pre


def _fb_rope_table(
    positions: torch.Tensor, *, theta: float, dim: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
    freqs = positions.to(device=device, dtype=torch.float32)[:, None] * inv_freq[None, :]
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _fb_qknorm_scaled_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor | None,
    sin: torch.Tensor | None,
    *,
    eps: float,
    qk_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    qf = q.float()
    kf = k.float()
    qf = qf * torch.pow(qf.pow(2).mean(-1, keepdim=True) + eps, -0.5) * qk_scale
    kf = kf * torch.pow(kf.pow(2).mean(-1, keepdim=True) + eps, -0.5)
    if cos is not None:
        c = cos[:, None, :]
        s = sin[:, None, :]
        qf = qf * c + _rotate_half(qf) * s
        kf = kf * c + _rotate_half(kf) * s
    return qf.to(torch.bfloat16), kf.to(torch.bfloat16)


def _fb_silu_mul(g: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    gf = g.float()
    return ((gf / (1.0 + torch.exp(-gf))) * u.float()).to(torch.bfloat16)


class _FallbackAttention(torch.autograd.Function):
    """Per-sequence fp32 attention with the contract's casts (p→bf16 before PV, out→bf16).

    Uses cuBLAS fp32 matmuls (TF32 off): repeatable for fixed shapes, not batch invariant.
    """

    @staticmethod
    def forward(ctx, q, k, v, gate, cu_q, cu_k, window):
        return _fb_attention_impl(q, k, v, gate, cu_q, cu_k, window, ctx)

    @staticmethod
    def backward(ctx, dy):
        q, k, v, gate = ctx.saved_tensors
        with torch.enable_grad():
            q_ = q.detach().requires_grad_(True)
            k_ = k.detach().requires_grad_(True)
            v_ = v.detach().requires_grad_(True)
            g_ = gate.detach().requires_grad_(True)
            y = _fb_attention_impl(q_, k_, v_, g_, ctx.cu_q, ctx.cu_k, ctx.window, None)
            grads = torch.autograd.grad(y, (q_, k_, v_, g_), dy)
        return (*grads, None, None, None)


def _fb_attention_impl(q, k, v, gate, cu_q, cu_k, window, ctx):
    if ctx is not None:
        ctx.save_for_backward(q, k, v, gate)
        ctx.cu_q, ctx.cu_k, ctx.window = cu_q, cu_k, window
    hq, d = q.shape[1], q.shape[2]
    hkv = k.shape[1]
    group = hq // hkv
    scale = d ** -0.5
    outs = []
    bounds = cu_q.tolist()
    for start, end in zip(bounds[:-1], bounds[1:], strict=True):
        qs = q[start:end].float().transpose(0, 1)  # (hq, s, d)
        ks = k[start:end].float().transpose(0, 1).repeat_interleave(group, dim=0)
        vs = v[start:end].float().transpose(0, 1).repeat_interleave(group, dim=0)
        s = end - start
        scores = torch.matmul(qs, ks.transpose(1, 2)) * scale
        pos = torch.arange(s, device=q.device)
        rel = pos[:, None] - pos[None, :]
        allowed = rel >= 0
        if window is not None:
            allowed &= rel < window
        scores = scores.masked_fill(~allowed[None], _NEG_INF)
        m = scores.amax(-1, keepdim=True)
        p = torch.exp(scores - m)
        l = p.sum(-1, keepdim=True)
        o = torch.matmul(p.to(torch.bfloat16).float(), vs) / l
        outs.append(o.transpose(0, 1).reshape(s, hq * d).to(torch.bfloat16))
    out = torch.cat(outs, dim=0)
    return (out.float() * torch.sigmoid(gate.float())).to(torch.bfloat16)


def _fb_gated_attention(q, k, v, gate_logits, cu_seqlens_q, cu_seqlens_k, q_pos, *, window):
    del q_pos  # positions are implied by cu_seqlens in the fallback (one prefill per sequence)
    return _FallbackAttention.apply(q, k, v, gate_logits, cu_seqlens_q, cu_seqlens_k, window)


# --------------------------------------------------------------------------------------
# LM head tile statistics (fallback Triton kernel: one program per logical row, R3)
# --------------------------------------------------------------------------------------

try:  # pragma: no cover - exercised on GPU only
    import triton
    import triton.language as tl

    try:
        from triton.language.extra import libdevice as _libdevice

        _HAVE_LIBDEVICE = True
    except Exception:  # noqa: BLE001
        _HAVE_LIBDEVICE = False

    @triton.jit
    def _tile_stats_kernel(
        z_ptr,
        m_ptr,
        s_ptr,
        valid_cols,
        stride_zm,
        mult,
        softcap,
        BLOCK: tl.constexpr,
        USE_LIBDEVICE: tl.constexpr,
    ):
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK)
        running_max = tl.full((), float("-inf"), tl.float32)
        for start in range(0, valid_cols, BLOCK):
            cols = start + offs
            mask = cols < valid_cols
            z = tl.load(z_ptr + row * stride_zm + cols, mask=mask, other=0.0).to(tl.float32)
            if USE_LIBDEVICE:
                t = _libdevice.tanh(z * mult / softcap)
            else:
                e = tl.exp(-2.0 * (z * mult / softcap))
                t = (1.0 - e) / (1.0 + e)
            zh = softcap * t
            zh = tl.where(mask, zh, float("-inf"))
            running_max = tl.maximum(running_max, tl.max(zh, 0))
        total = tl.zeros((), dtype=tl.float32)
        for start in range(0, valid_cols, BLOCK):
            cols = start + offs
            mask = cols < valid_cols
            z = tl.load(z_ptr + row * stride_zm + cols, mask=mask, other=0.0).to(tl.float32)
            if USE_LIBDEVICE:
                t = _libdevice.tanh(z * mult / softcap)
            else:
                e = tl.exp(-2.0 * (z * mult / softcap))
                t = (1.0 - e) / (1.0 + e)
            zh = softcap * t
            p = tl.where(mask, tl.exp(zh - running_max), 0.0)
            total += tl.sum(p, 0)
        tl.store(m_ptr + row, running_max)
        tl.store(s_ptr + row, total)

    _HAVE_TRITON = True
except Exception:  # noqa: BLE001
    _HAVE_TRITON = False
    _HAVE_LIBDEVICE = False

_TILE_BLOCK = 1024


def tile_stats_fp32(
    z: torch.Tensor, valid_cols: int, *, multiplier: float, softcap: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-row ``(max, sum exp)`` of ``softcap*tanh(z*multiplier/softcap)`` over ``valid_cols``.

    One Triton program per row walks the columns in fixed 1024-wide blocks (``tl.max``/
    ``tl.sum`` inside a block, sequential across blocks), so the bits do not depend on the
    number of rows (batch invariance by construction). Padded columns are excluded.
    """
    if not _HAVE_TRITON or not z.is_cuda:
        zh = softcap * torch.tanh(z[:, :valid_cols] * multiplier / softcap)
        m = zh.amax(-1)
        return m, torch.exp(zh - m[:, None]).sum(-1)
    z = z.contiguous()
    m = torch.empty(z.shape[0], dtype=torch.float32, device=z.device)
    s = torch.empty_like(m)
    _tile_stats_kernel[(z.shape[0],)](
        z,
        m,
        s,
        valid_cols,
        z.stride(0),
        float(multiplier),
        float(softcap),
        BLOCK=_TILE_BLOCK,
        USE_LIBDEVICE=_HAVE_LIBDEVICE,
        num_warps=8,
    )
    return m, s


def softcap_fp32(z: torch.Tensor, *, multiplier: float, softcap: float) -> torch.Tensor:
    return softcap * torch.tanh(z * multiplier / softcap)


def combine_tile_stats_tree(
    stats: Sequence[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fixed tree over ``(M_t, S_t)`` pairs: ``M=max(M1,M2)``, ``S=S1*e^(M1-M)+S2*e^(M2-M)``."""
    items = list(stats)
    if len(items) & (len(items) - 1):
        raise ValueError("combine_tile_stats_tree needs a power-of-two count")
    while len(items) > 1:
        merged = []
        for i in range(0, len(items), 2):
            m1, s1 = items[i]
            m2, s2 = items[i + 1]
            m = torch.maximum(m1, m2)
            merged.append((m, s1 * torch.exp(m1 - m) + s2 * torch.exp(m2 - m)))
        items = merged
    return items[0]


# --------------------------------------------------------------------------------------
# Resolution
# --------------------------------------------------------------------------------------


@dataclass
class StrictOps:
    """Resolved operator set with per-op provenance."""

    normalized_embedding: Callable[..., torch.Tensor]
    centered_rmsnorm: Callable[..., torch.Tensor]
    weighted_rmsnorm: Callable[..., torch.Tensor]
    fused_post_norm_residual_pre_norm: Callable[..., tuple[torch.Tensor, torch.Tensor]]
    rope_table: Callable[..., tuple[torch.Tensor, torch.Tensor]]
    qknorm_scaled_rope: Callable[..., tuple[torch.Tensor, torch.Tensor]]
    gated_attention: Callable[..., torch.Tensor]
    silu_mul: Callable[..., torch.Tensor]
    sources: dict[str, OpSource]

    def provenance(self) -> dict[str, Any]:
        gemm = gemm_backend().source
        out = {name: src.label for name, src in self.sources.items()}
        out["gemm"] = gemm.label
        out["lm_head_tile_stats"] = self.sources["lm_head_tile_stats"].label
        out["all_strict"] = all(s.strict for s in self.sources.values()) and gemm.strict
        return out


_OPS: StrictOps | None = None


def resolve_ops() -> StrictOps:
    global _OPS
    if _OPS is not None:
        return _OPS
    sources: dict[str, OpSource] = {}

    def pick(name: str, module_path: str, attr: str, fallback: Callable[..., Any], detail: str):
        fn = _optional_attr(_optional_module(module_path), attr)
        if fn is not None:
            sources[name] = OpSource(name, True, f"{module_path.rsplit('.', 1)[-1]}.{attr}")
            return fn
        sources[name] = OpSource(name, False, detail)
        return fallback

    norms = "rl_engine.models.muse_glimmer.ops.norms"
    qk = "rl_engine.models.muse_glimmer.ops.qknorm_rope"
    att = "rl_engine.models.muse_glimmer.ops.gated_attention_autograd"
    swi = "rl_engine.models.muse_glimmer.ops.swiglu_ffn"
    lm = "rl_engine.models.muse_glimmer.ops.lm_head_softcap_logprob"
    ops = StrictOps(
        normalized_embedding=pick(
            "normalized_embedding", norms, "normalized_embedding", _fb_normalized_embedding,
            "torch_fp32_rowwise",
        ),
        centered_rmsnorm=pick(
            "centered_rmsnorm", norms, "centered_rmsnorm", _fb_centered_rmsnorm, "torch_fp32_rowwise"
        ),
        weighted_rmsnorm=pick(
            "weighted_rmsnorm", norms, "weighted_rmsnorm", _fb_weighted_rmsnorm, "torch_fp32_rowwise"
        ),
        fused_post_norm_residual_pre_norm=pick(
            "fused_post_norm_residual_pre_norm", norms, "fused_post_norm_residual_pre_norm",
            _fb_fused_post_norm_residual_pre_norm, "torch_fp32_rowwise",
        ),
        rope_table=pick("rope_table", qk, "rope_table", _fb_rope_table, "torch_fp32_table"),
        qknorm_scaled_rope=pick(
            "qknorm_scaled_rope", qk, "qknorm_scaled_rope", _fb_qknorm_scaled_rope, "torch_fp32_rowwise"
        ),
        gated_attention=pick(
            "gated_attention", att, "gated_attention", _fb_gated_attention,
            "torch_fp32_cublas_per_sequence (not batch-invariant)",
        ),
        silu_mul=pick("silu_mul", swi, "silu_mul", _fb_silu_mul, "torch_fp32_elementwise"),
        sources=sources,
    )
    lm_mod = _optional_module(lm)
    if _optional_attr(lm_mod, "softcap_tile_stats") is not None:
        # The sibling tile kernel exists but its exact signature is not pinned by the brief;
        # the binding keeps its own composition so TP1 and TP>1 share one implementation.
        sources["lm_head_tile_stats"] = OpSource(
            "lm_head_tile_stats", False,
            "binding_triton_row_stats (sibling softcap_tile_stats present but not bound)",
        )
    else:
        sources["lm_head_tile_stats"] = OpSource(
            "lm_head_tile_stats", False,
            "binding_triton_row_stats" if _HAVE_TRITON else "torch_fp32_rowwise",
        )
    _OPS = ops
    return ops


def set_deterministic_torch() -> dict[str, Any]:
    """TF32 off, cuDNN deterministic, cuBLAS workspace pinned; returns the flags for provenance."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    return {
        "allow_tf32": False,
        "cudnn_deterministic": True,
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "fp32_matmul_precision": torch.get_float32_matmul_precision(),
    }


def ceil_to(value: int, multiple: int) -> int:
    return -(-value // multiple) * multiple


def tanh_softcap_grad(z: torch.Tensor, *, multiplier: float, softcap: float) -> torch.Tensor:
    """d softcap(z)/dz = multiplier * (1 - tanh²(z*multiplier/softcap)) in fp32."""
    t = torch.tanh(z * multiplier / softcap)
    return multiplier * (1.0 - t * t)


__all__ = [
    "CANONICAL_CHUNKS",
    "OpSource",
    "StrictOps",
    "TPComm",
    "ceil_to",
    "chunk_slices",
    "combine_tile_stats_tree",
    "combine_tree_torch",
    "gemm_backend",
    "gemm_fp32",
    "gemm_single_chunk_fp32",
    "local_chunk_ids",
    "resolve_ops",
    "set_deterministic_torch",
    "small_k_fp32",
    "softcap_fp32",
    "tanh_softcap_grad",
    "tile_stats_fp32",
]
