# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Strict operator resolution for the Muse Glimmer Megatron Bridge binding.

The binding is written against the FIXED interfaces of the sibling strict ops
(``rl_engine.models.muse_glimmer.ops.*`` and the attention / GEMM Triton modules). Every op is
resolved once per process by :func:`resolve_ops`. Strict mode FAILS CLOSED: unless the caller
passes ``allow_fallback=True`` (unit tests only), a missing or broken sibling op raises
:class:`StrictOpsUnavailable` naming every non-strict op and the import error behind it. The
labelled pure-torch fallbacks (``fallback:<detail>`` in :meth:`StrictOps.provenance`, never
labelled strict) exist only for those tests. Unknown ``MUSE_STRICT_ATTN`` / ``MUSE_STRICT_GEMM``
values raise.

Numerical conventions shared by every path here (contract O3/O7):

* GEMMs take ``a`` as ``(M, K)`` bf16 and ``w`` as ``(N, K)`` bf16 (native linear layout),
  accumulate in fp32 and return fp32; the caller decides where the single bf16 rounding is.
* A reduction dimension is split into 8 canonical equal chunks; chunk partials are combined
  with the fixed binary tree ``((0+1)+(2+3))+((4+5)+(6+7))``. At TP ``t`` a rank owns ``8/t``
  consecutive chunks, reduces them with the sub-tree, all-gathers the fp32 partials in rank
  order (data movement only; never a reducing NCCL collective) and applies the same tree.
* Reductions over TOKENS (LoRA weight gradients) use the det_gemm dW rule
  ``T_pad = roundup(T, 256)``, 8 chunks of ``T_pad/8`` rows, padded rows masked to zero
  (``ops.det_gemm.DW_CHUNK_RULE``, via ``fast_det_gemm(..., ragged_k=True)``).
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
TOKEN_PAD = 256  # det_gemm dW rule: T_pad = roundup(T, 256), 8 chunks of T_pad / 8 rows
ATTENTION_MODES = ("auto", "strict", "fallback")
GEMM_MODES = ("auto", "strict", "sibling", "triton_repo", "cublas")


class StrictOpsUnavailable(RuntimeError):
    """A strict op is missing or broken and the caller did not allow fallbacks (fail closed)."""


# import path -> "ExcType: message" for every sibling module that failed to import
_IMPORT_ERRORS: dict[str, str] = {}


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
    """Import a sibling module; on failure record the error (reported by the fail-closed check)."""
    try:
        return importlib.import_module(path)
    except Exception as exc:  # noqa: BLE001 - reported, never silently used in strict mode
        _IMPORT_ERRORS[path] = f"{type(exc).__name__}: {exc}"[:500]
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


class TreeAccumulator:
    """Canonical fixed tree built while the leaves arrive (left to right).

    ``push`` leaves 0..n-1 in order (n a power of two); equal-level neighbours are merged as
    soon as both exist (``left + right``), so 8 leaves give exactly
    ``((0+1)+(2+3))+((4+5)+(6+7))`` -- the bits of :func:`combine_tree_torch` and of the
    sibling ``ops.canonical_tree.combine_tree`` -- while at most ``log2(n) + 1`` fp32 partials
    are alive (the list form keeps all ``n``).
    """

    def __init__(self) -> None:
        self._stack: list[tuple[int, torch.Tensor]] = []
        self.count = 0

    def push(self, leaf: torch.Tensor) -> None:
        level, node = 0, leaf
        while self._stack and self._stack[-1][0] == level:
            _, left = self._stack.pop()
            node = left + node
            level += 1
        self._stack.append((level, node))
        self.count += 1

    def result(self) -> torch.Tensor:
        if self.count == 0 or self.count & (self.count - 1) or len(self._stack) != 1:
            raise ValueError(f"TreeAccumulator needs a power-of-two leaf count, got {self.count}")
        return self._stack[0][1]


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
        if requested not in GEMM_MODES:
            raise ValueError(f"MUSE_STRICT_GEMM={requested!r} is not one of {GEMM_MODES}")
        self.source: OpSource
        self._partial: Callable[..., torch.Tensor]
        self._combine: Callable[[Sequence[torch.Tensor]], torch.Tensor] = combine_tree_torch
        sibling = _optional_module("rl_engine.models.muse_glimmer.ops.det_gemm")
        tree_mod = _optional_module("rl_engine.models.muse_glimmer.ops.canonical_tree")
        sibling_partial = _optional_attr(sibling, "fast_det_gemm_partial")
        sibling_full = _optional_attr(sibling, "fast_det_gemm")
        sibling_tree = _optional_attr(tree_mod, "combine_tree")
        # Token-axis reductions (LoRA dA/dB): the det_gemm dW rule (ragged_k, 256-token unit).
        if sibling_full is not None and requested in ("auto", "strict", "sibling"):
            self._token_fn = sibling_full
            self._token_grad = self._sibling_token_grad
            self.token_source = OpSource(
                "gemm_token_grad", True,
                "ops.det_gemm.fast_det_gemm(ragged_k=True): T_pad=roundup(T,256), 8 chunks",
            )
        else:
            self._token_grad = self._torch_token_grad
            self.token_source = OpSource(
                "gemm_token_grad", False, "torch_fp32_per_chunk_tree (256-token rule, cuBLAS)"
            )
        if requested in ("auto", "strict", "sibling") and sibling_partial is not None:
            self._partial = self._sibling_partial
            self._sibling_fn = sibling_partial
            if sibling_tree is not None:
                self._combine = sibling_tree
            self.source = OpSource("gemm", True, "ops.det_gemm.fast_det_gemm_partial")
            return
        if requested in ("strict", "sibling"):
            raise StrictOpsUnavailable(
                f"MUSE_STRICT_GEMM={requested} but ops.det_gemm.fast_det_gemm_partial is absent "
                f"({_IMPORT_ERRORS.get('rl_engine.models.muse_glimmer.ops.det_gemm', 'no error')})"
            )
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

    # -- token-axis reduction (weight gradients) ------------------------------------------
    def _sibling_token_grad(self, lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self._token_fn(lhs.t(), rhs.t(), out_dtype=torch.float32, ragged_k=True)

    @staticmethod
    def _torch_token_grad(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
        t = lhs.shape[0]
        t_pad = ceil_to(max(t, 1), TOKEN_PAD)
        a = torch.nn.functional.pad(lhs.t().float(), (0, t_pad - t))
        b = torch.nn.functional.pad(rhs.t().float(), (0, t_pad - t))
        return _GemmBackend._cublas_partial_fp32(a, b)

    @staticmethod
    def _cublas_partial_fp32(a: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        parts = []
        for sl in chunk_slices(a.shape[1], range(CANONICAL_CHUNKS)):
            parts.append(torch.mm(a[:, sl], w[:, sl].t()))
        return combine_tree_torch(parts)

    def token_grad(self, lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
        """fp32 ``lhs.T @ rhs`` reduced over the token axis (dim 0 of both, ``T`` rows).

        ``lhs (T, P)``, ``rhs (T, Q)`` bf16 -> ``(P, Q)`` fp32 with the det_gemm dW rule:
        ``T_pad = roundup(T, 256)``, 8 chunks of ``T_pad/8`` rows (rows >= T masked to zero),
        fp32 MMA accumulation inside a chunk, chunk partials in the canonical tree.
        """
        if lhs.dtype != torch.bfloat16 or rhs.dtype != torch.bfloat16:
            raise TypeError("token_grad operands must be bf16")
        if lhs.ndim != 2 or rhs.ndim != 2 or lhs.shape[0] != rhs.shape[0]:
            raise ValueError(f"token_grad shape mismatch {tuple(lhs.shape)} / {tuple(rhs.shape)}")
        if lhs.shape[0] == 0:
            return torch.zeros(lhs.shape[1], rhs.shape[1], dtype=torch.float32, device=lhs.device)
        out = self._token_grad(lhs, rhs)
        if out.dtype != torch.float32:
            raise TypeError("token_grad backend must return fp32")
        return out

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
        # The sibling kernel addresses ``w`` through strides, so transposed views cost nothing.
        out = self._partial(a_in.contiguous(), w_in, chunk_ids)
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
    prod = torch.empty_like(acc)
    zf = z.float()
    bf = b.float()
    for k in range(r):
        # exact product, then one fp32 add per k in index order (same bits as acc = acc + prod;
        # in place so the (M, N) fp32 working set is two buffers, not three)
        torch.mul(zf[:, k : k + 1], bf[:, k].unsqueeze(0), out=prod)
        acc.add_(prod)
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
    """Per-(sequence, head) fp32 attention with the contract's casts (p→bf16 before PV, out→bf16).

    Uses cuBLAS fp32 matmuls (TF32 off) of identical ``(s,128)x(128,s)`` shape for every head,
    so the bits do not depend on how many heads a TP rank owns; repeatable for fixed shapes,
    NOT batch invariant (a sequence's length selects the cuBLAS kernel).
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
        s = end - start
        pos = torch.arange(s, device=q.device)
        rel = pos[:, None] - pos[None, :]
        allowed = rel >= 0
        if window is not None:
            allowed &= rel < window
        heads = []
        for h in range(hq):
            qs = q[start:end, h].float()  # (s, d)
            ks = k[start:end, h // group].float()
            vs = v[start:end, h // group].float()
            scores = torch.mm(qs, ks.t()) * scale
            scores = scores.masked_fill(~allowed, _NEG_INF)
            m = scores.amax(-1, keepdim=True)
            p = torch.exp(scores - m)
            l = p.sum(-1, keepdim=True)
            o = torch.mm(p.to(torch.bfloat16).float(), vs) / l
            heads.append(o.to(torch.bfloat16))
        outs.append(torch.stack(heads, dim=1).reshape(s, hq * d))
    out = torch.cat(outs, dim=0)
    return (out.float() * torch.sigmoid(gate.float())).to(torch.bfloat16)


@dataclass(frozen=True)
class AttentionKernels:
    """The per-rank attention entry points (contract O5, head-count parameters).

    Strict: ``fwd`` = ``ops.gated_attention.strict_gated_attention_fwd_varlen``, and the
    backward pieces of ``triton.gated_attention_bwd`` (``preprocess``, ``partials``,
    ``subtree``) with ``combine`` = ``ops.canonical_tree.combine_tree`` for the cross-rank fold
    of replicated KV heads. ``runtime.megatron_attention.rank_gated_attention`` drives them.
    Fallback (tests only, TP <= 2): ``fallback`` is the per-head fp32 torch attention.
    """

    source: OpSource
    fwd: Callable[..., Any] | None = None
    preprocess: Callable[..., Any] | None = None
    partials: Callable[..., Any] | None = None
    subtree: Callable[..., Any] | None = None
    combine: Callable[..., Any] | None = None
    fallback: Callable[..., Any] | None = None

    @property
    def strict(self) -> bool:
        return self.source.strict


def _resolve_attention(mode: str) -> AttentionKernels:
    if mode not in ATTENTION_MODES:
        raise ValueError(f"MUSE_STRICT_ATTN={mode!r} is not one of {ATTENTION_MODES}")
    fallback = AttentionKernels(
        source=OpSource(
            "gated_attention", False,
            "binding_torch_fp32_cublas_per_head (TP<=2 only, not batch invariant)",
        ),
        fallback=_FallbackAttention.apply,
    )
    if mode == "fallback":
        return fallback
    fwd_mod = _optional_module("rl_engine.models.muse_glimmer.ops.gated_attention")
    bwd_mod = _optional_module("rl_engine.models.muse_glimmer.triton.gated_attention_bwd")
    tree_mod = _optional_module("rl_engine.models.muse_glimmer.ops.canonical_tree")
    parts = {
        "fwd": _optional_attr(fwd_mod, "strict_gated_attention_fwd_varlen"),
        "preprocess": _optional_attr(bwd_mod, "gated_attention_bwd_preprocess"),
        "partials": _optional_attr(bwd_mod, "strict_gated_attention_bwd_partials"),
        "subtree": _optional_attr(bwd_mod, "kv_partials_subtree"),
        "combine": _optional_attr(tree_mod, "combine_tree"),
    }
    missing = [k for k, v in parts.items() if v is None]
    if missing:
        if mode == "strict":
            raise StrictOpsUnavailable(f"MUSE_STRICT_ATTN=strict but {missing} are absent")
        return AttentionKernels(
            source=OpSource("gated_attention", False, fallback.source.detail + f"; missing {missing}"),
            fallback=_FallbackAttention.apply,
        )
    return AttentionKernels(
        source=OpSource(
            "gated_attention", True,
            "gated_attention.strict_gated_attention_fwd_varlen(num_q_heads,num_kv_heads) + "
            "gated_attention_bwd.{gated_attention_bwd_preprocess,strict_gated_attention_bwd_"
            "partials,kv_partials_subtree} + canonical_tree.combine_tree (per-rank heads)",
        ),
        **parts,
    )


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


def _fb_tile_stats(
    z: torch.Tensor,
    tile_id: int,
    targets: torch.Tensor,
    *,
    multiplier: float,
    softcap: float,
    vocab_size: int,
    tile_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fallback ``(m_t, s_t, zhat[target], target_in_tile)`` for one fp32 logits tile."""
    col_begin = tile_id * tile_size
    valid = min(max(vocab_size - col_begin, 0), tile_size)
    m, s = tile_stats_fp32(z, valid, multiplier=multiplier, softcap=softcap)
    local = targets - col_begin
    hit = (local >= 0) & (local < valid)
    picked = z.gather(1, local.clamp(0, tile_size - 1).unsqueeze(1)).squeeze(1)
    zy = torch.where(hit, softcap_fp32(picked, multiplier=multiplier, softcap=softcap), torch.zeros_like(picked))
    return m, s, zy, hit


def _fb_tile_dz(
    z: torch.Tensor,
    logz: torch.Tensor,
    dlogp32: torch.Tensor,
    targets: torch.Tensor,
    *,
    tile_size: int,
    real_cols: int,
    col_begin: int,
    multiplier: float,
    softcap: float,
) -> torch.Tensor:
    """Fallback ``dz = dlogp * (onehot - p) * d softcap/dz`` for one tile, bf16 (declared cast)."""
    zh = softcap_fp32(z, multiplier=multiplier, softcap=softcap)
    p = torch.exp(zh - logz[:, None])
    dzh = -p
    local = targets - col_begin
    hit = (local >= 0) & (local < real_cols)
    if bool(hit.any()):
        idx = hit.nonzero(as_tuple=True)[0]
        dzh[idx, local[idx]] += 1.0
    dz = dzh * dlogp32[:, None] * tanh_softcap_grad(z, multiplier=multiplier, softcap=softcap)
    if real_cols < tile_size:
        dz[:, real_cols:] = 0.0
    return dz.to(torch.bfloat16)


class LMHeadTileOps:
    """Tile-level LM-head primitives: sibling kernels when present, labelled fallbacks otherwise."""

    def __init__(self) -> None:
        lm = _optional_module("rl_engine.models.muse_glimmer.ops.lm_head_softcap_logprob")
        kern = _optional_module("rl_engine.models.muse_glimmer.triton.softcap_logprob")
        stats = _optional_attr(lm, "softcap_tile_stats")
        dz = _optional_attr(kern, "launch_dz")
        merge = _optional_attr(lm, "merge_tile_stats")
        if stats is not None and dz is not None and merge is not None:
            self.tile_stats = stats
            self.tile_dz = dz
            self.merge = merge
            self.source = OpSource(
                "lm_head_tile_stats", True,
                "lm_head_softcap_logprob.softcap_tile_stats+launch_dz+merge_tile_stats",
            )
        else:
            self.tile_stats = _fb_tile_stats
            self.tile_dz = _fb_tile_dz
            self.merge = combine_tile_stats_tree
            self.source = OpSource(
                "lm_head_tile_stats", False,
                "binding_triton_row_stats" if _HAVE_TRITON else "torch_fp32_rowwise",
            )


_LM_TILE_OPS: LMHeadTileOps | None = None


def lm_head_tile_ops() -> LMHeadTileOps:
    global _LM_TILE_OPS
    if _LM_TILE_OPS is None:
        _LM_TILE_OPS = LMHeadTileOps()
    return _LM_TILE_OPS


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
    rope_rows: Callable[..., tuple[torch.Tensor, torch.Tensor]]
    qknorm_scaled_rope: Callable[..., tuple[torch.Tensor, torch.Tensor]]
    attention: AttentionKernels
    silu_mul: Callable[..., torch.Tensor]
    sources: dict[str, OpSource]
    rope_table_sha256: Callable[..., str] | None = None

    def all_sources(self) -> dict[str, OpSource]:
        out = dict(self.sources)
        out["gated_attention"] = self.attention.source
        backend = gemm_backend()
        out["gemm"] = backend.source
        out["gemm_token_grad"] = backend.token_source
        out["lm_head_tile_stats"] = lm_head_tile_ops().source
        return out

    def non_strict(self) -> list[str]:
        return [f"{name} -> {src.label}" for name, src in self.all_sources().items() if not src.strict]

    def provenance(self) -> dict[str, Any]:
        out: dict[str, Any] = {name: src.label for name, src in self.all_sources().items()}
        out["all_strict"] = not self.non_strict()
        return out


_OPS: StrictOps | None = None


def _rope_rows_fallback(
    positions: torch.Tensor, *, device: torch.device | str | int | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    dev = positions.device if device is None else torch.device(device)
    return _fb_rope_table(positions, theta=500000.0, dim=128, device=dev)


def reset_resolution() -> None:
    """Forget the resolved ops (tests that simulate a missing sibling module)."""
    global _OPS, _GEMM, _LM_TILE_OPS
    _OPS = None
    _GEMM = None
    _LM_TILE_OPS = None
    _IMPORT_ERRORS.clear()


def resolve_ops(*, allow_fallback: bool = False) -> StrictOps:
    """Resolve every op of the strict path; FAIL CLOSED unless ``allow_fallback=True``.

    Raises :class:`StrictOpsUnavailable` listing each non-strict op (with the sibling import
    errors) when a strict kernel is missing or broken. ``allow_fallback=True`` returns the
    labelled fallbacks instead (unit tests only; their labels start with ``fallback:``).
    """
    global _OPS
    if _OPS is None:
        _OPS = _build_ops()
    if not allow_fallback:
        bad = _OPS.non_strict()
        if bad:
            errors = "; ".join(f"{k}: {v}" for k, v in sorted(_IMPORT_ERRORS.items()))
            raise StrictOpsUnavailable(
                "strict mode requires every op to be a sibling strict kernel; non-strict: "
                + ", ".join(bad) + (f" (import errors: {errors})" if errors else "")
            )
    return _OPS


def _build_ops() -> StrictOps:
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
    swi = "rl_engine.models.muse_glimmer.ops.swiglu_ffn"
    attention = _resolve_attention(os.getenv("MUSE_STRICT_ATTN", "auto").strip().lower())
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
        rope_rows=pick("rope_rows", qk, "rope_rows", _rope_rows_fallback, "torch_fp32_table"),
        qknorm_scaled_rope=pick(
            "qknorm_scaled_rope", qk, "qknorm_scaled_rope", _fb_qknorm_scaled_rope, "torch_fp32_rowwise"
        ),
        attention=attention,
        silu_mul=pick("silu_mul", swi, "swiglu", _fb_silu_mul, "torch_fp32_elementwise"),
        sources=sources,
        rope_table_sha256=_optional_attr(_optional_module(qk), "rope_table_sha256"),
    )
    gemm_backend()
    lm_head_tile_ops()
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
    "ATTENTION_MODES",
    "AttentionKernels",
    "CANONICAL_CHUNKS",
    "GEMM_MODES",
    "LMHeadTileOps",
    "OpSource",
    "StrictOps",
    "StrictOpsUnavailable",
    "TOKEN_PAD",
    "TPComm",
    "TreeAccumulator",
    "ceil_to",
    "chunk_slices",
    "combine_tile_stats_tree",
    "combine_tree_torch",
    "gemm_backend",
    "gemm_fp32",
    "gemm_single_chunk_fp32",
    "lm_head_tile_ops",
    "local_chunk_ids",
    "reset_resolution",
    "resolve_ops",
    "set_deterministic_torch",
    "small_k_fp32",
    "softcap_fp32",
    "tanh_softcap_grad",
    "tile_stats_fp32",
]
