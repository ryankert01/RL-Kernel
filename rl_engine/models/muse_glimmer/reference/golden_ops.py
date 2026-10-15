# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Pure-torch fp32 fixed-order fallbacks for every Muse Glimmer strict operator.

These functions carry the SAME names and signatures as the sibling-owned ops under
``rl_engine.models.muse_glimmer.ops.*`` so the full-model chain can run before those land
(``backend="golden"``). Every function follows the casts and reduction orders of
``docs/contracts/muse-glimmer-30b-strict-contract.md``:

* all arithmetic in fp32, bf16 only at the declared cast points;
* every row reduction uses :func:`tree_sum_last` (adjacent-pair halving while the length is
  even, then left-to-right sequential) so the bits of one logical row never depend on how many
  rows are processed together (batch / chunk / decode invariance by construction);
* attention is an explicit online softmax over 64-key blocks aligned at absolute key position 0
  (contract O5), with ``p`` rounded to bf16 before the PV product, so one-shot prefill, chunked
  prefill and token-by-token decode produce identical bits;
* GEMMs go through :class:`GoldenGemm`: the repo ``TritonDetGemmOp.forward_fp32`` kernel (one
  program per output tile, K walked in BLOCK_K=32 order, ``tl.dot(allow_tf32=False)``, fp32
  accumulation, no split-K) with ONE rounding to the declared output dtype. Row-parallel GEMMs
  (o_proj, down_proj) use the contract's 8-chunk K split and the fixed partial tree
  ``((0+1)+(2+3))+((4+5)+(6+7))`` → bf16 once. ``impl="torch_fp32_debug"`` is a cuBLAS fp32
  matmul for debugging only and is NOT strict.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import torch

BLOCK_N = 64  # attention key block (contract O5)
KV_PAGE = 16  # paged KV page size (contract O5, rollout)
ROW_PARALLEL_CHUNKS = 8  # contract O3 canonical K split
VOCAB_TILES = 8  # contract O7 canonical vocab tiles

_NEG_INF = float("-inf")


# --------------------------------------------------------------------------------------------
# Fixed-order reductions
# --------------------------------------------------------------------------------------------
def tree_sum_last(x: torch.Tensor) -> torch.Tensor:
    """Fixed-order fp32 sum over the last dim.

    Order: adjacent-pair halving ``x[0::2] + x[1::2]`` while the length is even, then a
    left-to-right sequential sum of the odd remainder. Only elementwise kernels are used, so the
    result for one row is independent of every other dim (batch invariant by construction).
    """
    n = x.shape[-1]
    while n > 1 and n % 2 == 0:
        x = x[..., 0::2] + x[..., 1::2]
        n //= 2
    if n == 1:
        return x[..., 0]
    acc = x[..., 0]
    for i in range(1, n):
        acc = acc + x[..., i]
    return acc


def tree_mean_last(x: torch.Tensor) -> torch.Tensor:
    """``tree_sum_last(x) / H`` in fp32 (fixed-order mean over the last dim)."""
    return tree_sum_last(x) / x.shape[-1]


def fixed_tree_partials(parts: Sequence[torch.Tensor]) -> torch.Tensor:
    """Sum ``len(parts)`` fp32 partials in the contract's fixed binary tree.

    For 8 partials this is ``((0+1)+(2+3))+((4+5)+(6+7))``; any power of two generalises the
    same way, and an odd tail is added left to right.
    """
    level = list(parts)
    while len(level) > 1:
        nxt = [level[i] + level[i + 1] for i in range(0, len(level) - 1, 2)]
        if len(level) % 2:
            nxt.append(level[-1])
        level = nxt
    return level[0]


# --------------------------------------------------------------------------------------------
# Weight holders (shared with weights.py; duck-typed by the golden GEMM)
# --------------------------------------------------------------------------------------------
class Linear:
    """One projection weight in natural ``(N, K)`` layout with an optional ``(K, N)`` copy.

    ``w`` is what the HF checkpoint stores and what sibling ops take; ``w_t`` is what the golden
    Triton fp32 GEMM consumes. Forward-only models keep exactly one copy (``layout``).
    """

    def __init__(self, name: str, w: torch.Tensor | None, w_t: torch.Tensor | None = None):
        if w is None and w_t is None:
            raise ValueError(f"{name}: need w or w_t")
        self.name = name
        self.w = w
        self.w_t = w_t
        n, k = (w.shape if w is not None else (w_t.shape[1], w_t.shape[0]))
        self.n, self.k = int(n), int(k)

    def kn(self) -> torch.Tensor:
        """``(K, N)`` view for ``a @ w.T``; transposes in-graph when only ``w`` is kept."""
        if self.w_t is not None:
            return self.w_t
        return self.w.t().contiguous()

    def nk(self) -> torch.Tensor:
        if self.w is not None:
            return self.w
        return self.w_t.t().contiguous()

    @property
    def device(self) -> torch.device:
        return (self.w if self.w is not None else self.w_t).device

    def parameters(self) -> list[torch.Tensor]:
        return [t for t in (self.w, self.w_t) if t is not None]


class LmHead:
    """Padded lm_head weight: natural ``(Vp, H)`` and/or 8 transposed tiles ``(H, Vp/8)``."""

    def __init__(
        self,
        w_padded: torch.Tensor | None,
        tiles_t: list[torch.Tensor] | None,
        *,
        vocab_size: int,
        padded_vocab_size: int,
    ):
        self.w_padded = w_padded
        self.tiles_t = tiles_t
        self.vocab_size = int(vocab_size)
        self.padded_vocab_size = int(padded_vocab_size)
        self.tile = self.padded_vocab_size // VOCAB_TILES
        if self.tile * VOCAB_TILES != self.padded_vocab_size:
            raise ValueError("padded vocab must split into 8 equal tiles")

    def tile_kn(self, t: int) -> torch.Tensor:
        if self.tiles_t is not None:
            return self.tiles_t[t]
        return self.w_padded[t * self.tile : (t + 1) * self.tile].t().contiguous()

    def tile_nk(self, t: int) -> torch.Tensor:
        if self.w_padded is not None:
            return self.w_padded[t * self.tile : (t + 1) * self.tile]
        return self.tiles_t[t].t().contiguous()

    def nk_padded(self) -> torch.Tensor:
        if self.w_padded is not None:
            return self.w_padded
        return torch.cat([t.t() for t in self.tiles_t], dim=0).contiguous()

    @property
    def device(self) -> torch.device:
        return (self.w_padded if self.w_padded is not None else self.tiles_t[0]).device


def _kn(w: Any) -> torch.Tensor:
    """``(K, N)`` operand from a Linear or a raw ``(N, K)`` tensor."""
    if isinstance(w, Linear):
        return w.kn()
    return w.t().contiguous()


def _nk(w: Any) -> torch.Tensor:
    if isinstance(w, Linear):
        return w.nk()
    return w


# --------------------------------------------------------------------------------------------
# GEMM
# --------------------------------------------------------------------------------------------
class GoldenGemm:
    """Strict fallback GEMM ``a (M,K) bf16 @ w (N,K) bf16 .T`` with fp32 accumulation.

    ``impl="triton_fp32"`` (strict): ``TritonDetGemmOp.forward_fp32`` — one Triton program per
    64x64 output tile walks K in BLOCK_K=32 order with ``tl.dot(allow_tf32=False)`` and fp32
    accumulation, no split-K, so the bits of an output element depend only on its row of ``a``
    and column of ``w`` (batch invariant); then ONE rounding to ``out_dtype``.
    ``impl="torch_fp32_debug"``: cuBLAS fp32 matmul; convenient, NOT strict, not invariant.
    """

    IMPLS = ("triton_fp32", "torch_fp32_debug")

    def __init__(self, impl: str = "triton_fp32"):
        if impl not in self.IMPLS:
            raise ValueError(f"unknown GoldenGemm impl {impl!r}; expected one of {self.IMPLS}")
        self.impl = impl
        self._op = None
        if impl == "triton_fp32":
            from rl_engine.backends.shared.triton.gemm.det_gemm import TritonDetGemmOp

            self._op = TritonDetGemmOp()

    def strict(self) -> bool:
        return self.impl == "triton_fp32"

    def _fp32(self, a: torch.Tensor, b_kn: torch.Tensor) -> torch.Tensor:
        if a.dtype != torch.bfloat16 or b_kn.dtype != torch.bfloat16:
            raise TypeError("GoldenGemm expects bf16 operands")
        if self._op is not None:
            return self._op.forward_fp32(a.contiguous(), b_kn)
        return a.float() @ b_kn.float()

    def __call__(
        self, a: torch.Tensor, w: Any, *, out_dtype: torch.dtype = torch.bfloat16
    ) -> torch.Tensor:
        out = self._fp32(a, _kn(w))
        return out if out_dtype == torch.float32 else out.to(out_dtype)

    def matmul_kn(
        self, a: torch.Tensor, b_kn: torch.Tensor, *, out_dtype: torch.dtype = torch.bfloat16
    ) -> torch.Tensor:
        """Same GEMM with an already-transposed ``(K, N)`` operand (lm_head tiles)."""
        out = self._fp32(a, b_kn)
        return out if out_dtype == torch.float32 else out.to(out_dtype)

    def row_parallel(
        self, a: torch.Tensor, w: Any, *, chunks: int = ROW_PARALLEL_CHUNKS
    ) -> torch.Tensor:
        """Contract O3 row-parallel form: 8 fp32 K-chunk partials, fixed tree, → bf16 once."""
        b = _kn(w)
        k = a.shape[-1]
        if k % chunks:
            raise ValueError(f"row-parallel K={k} is not a multiple of {chunks} chunks")
        step = k // chunks
        parts = [
            self._fp32(a[:, c * step : (c + 1) * step], b[c * step : (c + 1) * step])
            for c in range(chunks)
        ]
        return fixed_tree_partials(parts).to(torch.bfloat16)

    def trace(self) -> dict[str, Any]:
        return {
            "op": "golden_gemm",
            "impl": self.impl,
            "strict": self.strict(),
            "dtype_in": "bf16",
            "dtype_acc": "fp32",
            "dtype_out": "declared (bf16 once, fp32 for logits)",
            "reduction_order": (
                "TritonDetGemmOp fp32 kernel: K ascending in BLOCK_K=32 tl.dot steps, "
                "one program per 64x64 tile, no split-K; row-parallel: 8 K-chunks then "
                "((0+1)+(2+3))+((4+5)+(6+7))"
                if self.strict()
                else "cuBLAS fp32 (debug, unspecified)"
            ),
            "tile_config": {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32},
        }


def row_parallel_gemm(gemm: Any, a: torch.Tensor, w: Any) -> torch.Tensor:
    """Row-parallel GEMM (o_proj, down_proj) through any gemm callable.

    :class:`GoldenGemm` exposes ``.row_parallel`` (8 fp32 K-chunk partials + fixed tree). A
    contract-conformant strict GEMM such as the sibling ``fast_det_gemm`` applies the same
    8-chunk K tree inside every call, so the row-parallel form is the plain call (slicing K
    here would nest a second tree and change the bits).
    """
    if hasattr(gemm, "row_parallel"):
        return gemm.row_parallel(a, w)
    return gemm(a, _nk(w))


# --------------------------------------------------------------------------------------------
# O1 / O2 norms
# --------------------------------------------------------------------------------------------
class _DetEmbeddingLookup(torch.autograd.Function):
    """``weight[ids]`` with a deterministic sorted-segment backward (contract O1).

    dW[tok] = sequential ascending-position fp32 sum of the grad rows of ``tok``; no atomics.
    """

    @staticmethod
    def forward(ctx, weight: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(ids)
        ctx.weight_shape = tuple(weight.shape)
        ctx.weight_dtype = weight.dtype
        return weight.index_select(0, ids)

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        (ids,) = ctx.saved_tensors
        grad = grad.float()
        dw = torch.zeros(ctx.weight_shape, dtype=torch.float32, device=grad.device)
        order = torch.argsort(ids, stable=True)
        sorted_ids = ids[order].tolist()
        start = 0
        while start < len(sorted_ids):
            stop = start
            while stop < len(sorted_ids) and sorted_ids[stop] == sorted_ids[start]:
                stop += 1
            acc = grad[order[start]]
            for j in range(start + 1, stop):
                acc = acc + grad[order[j]]
            dw[sorted_ids[start]] = acc
            start = stop
        return dw.to(ctx.weight_dtype), None


def normalized_embedding(weight: torch.Tensor, token_ids: torch.Tensor, *, eps: float):
    """O1: x = E[token] → fp32; y = x * (mean(x²) + eps)^(-1/2) → bf16. Weightless, no √H."""
    x = _DetEmbeddingLookup.apply(weight, token_ids.to(torch.long)).float()
    ms = tree_mean_last(x * x)
    return (x * torch.pow(ms + eps, -0.5)[..., None]).to(torch.bfloat16)


def centered_rmsnorm(x: torch.Tensor, w: torch.Tensor, *, eps: float) -> torch.Tensor:
    """O2: y = x̂ * (1 + w), x̂ = x * rsqrt(mean(x²) + eps); fp32 compute, → bf16."""
    xf = x.float()
    xhat = xf * torch.rsqrt(tree_mean_last(xf * xf) + eps)[..., None]
    return (xhat * (1.0 + w.float())).to(torch.bfloat16)


def weighted_rmsnorm(x: torch.Tensor, w: torch.Tensor, *, eps: float) -> torch.Tensor:
    """O2 final norm: y = x̂ * w, x̂ = x * (mean(x²) + eps)^(-1/2); fp32 compute, → bf16."""
    xf = x.float()
    xhat = xf * torch.pow(tree_mean_last(xf * xf) + eps, -0.5)[..., None]
    return (xhat * w.float()).to(torch.bfloat16)


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
    """O2 [F1]: r' = bf16(fp32(r) + fp32(post_norm(branch))); returns (r', pre_norm(r'))."""
    n = centered_rmsnorm(branch, w_post, eps=eps_post)
    r_new = (resid.float() + n.float()).to(torch.bfloat16)
    if pre_form == "centered":
        pre = centered_rmsnorm(r_new, w_pre, eps=eps_pre)
    elif pre_form == "weighted":
        pre = weighted_rmsnorm(r_new, w_pre, eps=eps_pre)
    else:
        raise ValueError(f"unknown pre_form {pre_form!r}")
    return r_new, pre


# --------------------------------------------------------------------------------------------
# O4 qk-norm + scaled RoPE
# --------------------------------------------------------------------------------------------
def rope_table(
    positions: torch.Tensor, *, theta: float, dim: int, device: torch.device | str
) -> tuple[torch.Tensor, torch.Tensor]:
    """O4 table: cos/sin fp32 ``(T, dim)`` of ``p * theta^(-2i/dim)`` for absolute positions.

    Delegates to THE builder, :func:`rl_engine.models.muse_glimmer.ops.qknorm_rope.rope_table`:
    rows of the canonical table built once per (CUDA device, theta, dim) by the fixed fp32
    torch-CUDA function, so the golden and the Triton backends (and the runtimes) read the same
    cached object. Non-CUDA devices and positions outside ``[0, 131072)`` raise (fail closed:
    a CPU build differs in ~19 % of the elements).
    """
    from rl_engine.models.muse_glimmer.ops.qknorm_rope import rope_table as _canonical_rows

    return _canonical_rows(positions, theta=float(theta), dim=int(dim), device=device)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def qknorm_scaled_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor | None,
    sin: torch.Tensor | None,
    *,
    eps: float = 1e-5,
    qk_scale: float = 3.87,
    local: bool | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """O4: per-head weightless rmsnorm (pow -0.5) in fp32; q *= qk_scale; rotate_half RoPE in
    fp32 over all channels when cos/sin are given (local layers); → bf16 once.

    ``local=True/False`` states the layer kind explicitly and must agree with ``cos``/``sin``
    (a local layer needs the table, a global layer must not get one; mismatch → ValueError).
    ``local=None`` keeps the legacy implicit form (``cos is None`` → NoPE)."""
    if (cos is None) != (sin is None):
        raise ValueError("cos and sin must both be given (local layer) or both None (global)")
    if local is not None and bool(local) != (cos is not None):
        raise ValueError(
            f"local={local} but cos/sin are {'None' if cos is None else 'given'}: a local "
            "(RoPE) layer needs the table rows, a global (NoPE) layer must not get them"
        )
    qf = q.float()
    kf = k.float()
    qh = qf * torch.pow(tree_mean_last(qf * qf) + eps, -0.5)[..., None]
    kh = kf * torch.pow(tree_mean_last(kf * kf) + eps, -0.5)[..., None]
    qh = qh * qk_scale
    if cos is not None:
        c = cos[:, None, :]
        s = sin[:, None, :]
        qh = qh * c + _rotate_half(qh) * s
        kh = kh * c + _rotate_half(kh) * s
    return qh.to(torch.bfloat16), kh.to(torch.bfloat16)


# --------------------------------------------------------------------------------------------
# O5 gated attention (explicit fixed-order online softmax over 64-key blocks)
# --------------------------------------------------------------------------------------------
def attention_core(
    q: torch.Tensor,
    k_rows: torch.Tensor,
    v_rows: torch.Tensor,
    q_pos: torch.Tensor,
    *,
    window: int | None,
    k_pos0: int = 0,
    k_len: int | None = None,
    query_tile: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Contract O5 arithmetic for one sequence.

    ``q`` (nq, Hq, D) bf16 at absolute positions ``q_pos`` (ascending); ``k_rows``/``v_rows``
    (n, Hkv, D) bf16 hold keys at absolute positions ``k_pos0 .. k_pos0+n-1``; keys at absolute
    position ≥ ``k_len`` (default ``k_pos0 + n``) do not exist yet and are masked.

    Per query tile the KV is visited in 64-key blocks aligned at absolute position 0, ascending,
    from ``floor(max(0, p_min−W+1)/64)`` (0 on global layers) to ``floor(p_max/64)``. Every
    block is padded to exactly 64 keys (zeros, masked) so the fixed trees below never depend on
    how many keys exist. Per block, in this order: scores = tree_sum_D(q·k) * D^-1/2 (fp32);
    mask → -inf; m_new = max(m, rowmax); α = exp(m − m_new); p = exp(s − m_new);
    l = l·α + tree_sum_64(p); acc = acc·α + tree_sum_64(bf16(p)·v). Rows that have not seen a
    key yet (m_new = -inf) use α = 1, p = 0 (bitwise no-op guard). out = acc / l → bf16;
    lse = m + log(l) fp32. GQA head h reads kv head h // (Hq/Hkv).
    """
    nq, hq, d = q.shape
    n_rows, hkv, _ = k_rows.shape
    group = hq // hkv
    scale = float(d) ** -0.5
    if k_len is None:
        k_len = k_pos0 + n_rows
    device = q.device
    qf = q.float()
    outs: list[torch.Tensor] = []
    lses: list[torch.Tensor] = []
    q_pos = q_pos.to(device=device, dtype=torch.long)
    zero_k = torch.zeros((BLOCK_N, hkv, d), dtype=torch.float32, device=device)
    for t0 in range(0, nq, query_tile):
        t1 = min(t0 + query_tile, nq)
        qt = qf[t0:t1]
        pos = q_pos[t0:t1]
        p_min = int(pos.min().item())
        p_max = int(pos.max().item())
        b_first = 0 if window is None else max(0, p_min - window + 1) // BLOCK_N
        b_last = p_max // BLOCK_N
        n = t1 - t0
        m = torch.full((n, hq), _NEG_INF, dtype=torch.float32, device=device)
        l = torch.zeros((n, hq), dtype=torch.float32, device=device)
        acc = torch.zeros((n, hq, d), dtype=torch.float32, device=device)
        for b in range(b_first, b_last + 1):
            k0 = b * BLOCK_N
            kpos = torch.arange(k0, k0 + BLOCK_N, device=device)
            r0 = k0 - k_pos0
            r1 = r0 + BLOCK_N
            if r0 >= 0 and r1 <= n_rows:
                kb = k_rows[r0:r1].float()
                vb = v_rows[r0:r1].float()
            else:
                kb = zero_k.clone()
                vb = zero_k.clone()
                lo, hi = max(r0, 0), min(r1, n_rows)
                if hi > lo:
                    kb[lo - r0 : hi - r0] = k_rows[lo:hi].float()
                    vb[lo - r0 : hi - r0] = v_rows[lo:hi].float()
            kb = kb.repeat_interleave(group, dim=1).permute(1, 0, 2)  # (Hq, 64, D)
            vb = vb.repeat_interleave(group, dim=1).permute(1, 0, 2)
            s = tree_sum_last(qt[:, :, None, :] * kb[None]) * scale  # (n, Hq, 64)
            visible = (kpos[None, :] <= pos[:, None]) & (kpos[None, :] < k_len)
            if window is not None:
                visible = visible & (kpos[None, :] > pos[:, None] - window)
            s = torch.where(visible[:, None, :], s, _NEG_INF)
            m_new = torch.maximum(m, s.amax(dim=-1))
            alive = m_new > _NEG_INF
            alpha = torch.where(alive, torch.exp(m - m_new), 1.0)
            p = torch.where(alive[..., None], torch.exp(s - m_new[..., None]), 0.0)
            l = l * alpha + tree_sum_last(p)
            pb = p.to(torch.bfloat16).float()
            pv = pb[..., None] * vb[None]  # (n, Hq, 64, D)
            acc = acc * alpha[..., None] + tree_sum_last(pv.permute(0, 1, 3, 2))
            m = m_new
        outs.append((acc / l[..., None]).to(torch.bfloat16))
        lses.append(m + torch.log(l))
    return torch.cat(outs, dim=0), torch.cat(lses, dim=0)


def _gate(o: torch.Tensor, gate_logits: torch.Tensor) -> torch.Tensor:
    """y = bf16( fp32(o) * sigmoid(fp32(g)) ), sigmoid(x) = 1 / (1 + exp(-x)) in fp32."""
    flat = o.reshape(o.shape[0], -1).float()
    sig = 1.0 / (1.0 + torch.exp(-gate_logits.float()))
    return (flat * sig).to(torch.bfloat16)


def strict_gated_attention_fwd_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate_logits: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    q_pos: torch.Tensor,
    *,
    window: int | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """O5 training/prefill form. Keys of sequence b sit at absolute positions 0..len_k−1.

    Returns (y gated bf16 (Tq, Hq*D), o ungated bf16 (Tq, Hq, D), lse fp32 (Tq, Hq)).
    """
    cq = [int(x) for x in cu_seqlens_q.tolist()]
    ck = [int(x) for x in cu_seqlens_k.tolist()]
    outs, lses = [], []
    for b in range(len(cq) - 1):
        o_b, lse_b = attention_core(
            q[cq[b] : cq[b + 1]],
            k[ck[b] : ck[b + 1]],
            v[ck[b] : ck[b + 1]],
            q_pos[cq[b] : cq[b + 1]],
            window=window,
        )
        outs.append(o_b)
        lses.append(lse_b)
    o = torch.cat(outs, dim=0)
    return _gate(o, gate_logits), o, torch.cat(lses, dim=0)


def gather_kv_blocks(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table_row: torch.Tensor,
    b_first: int,
    b_last: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather key blocks ``b_first..b_last`` (4 pages each) from a paged cache.

    ``k_cache`` is ``(P, 16, Hkv, D)`` with page 0 reserved as an all-zero page; block-table
    entries < 0 (never allocated or evicted) read that zero page. Copies are exact.
    """
    pages_per_block = BLOCK_N // KV_PAGE
    p0, p1 = b_first * pages_per_block, (b_last + 1) * pages_per_block
    max_pages = block_table_row.shape[0]
    idx = torch.zeros((p1 - p0,), dtype=torch.long, device=k_cache.device)
    hi = min(p1, max_pages)
    if hi > p0:
        bt = block_table_row[p0:hi].to(torch.long)
        idx[: hi - p0] = torch.where(bt < 0, torch.zeros_like(bt), bt)
    kg = k_cache.index_select(0, idx).reshape(-1, k_cache.shape[2], k_cache.shape[3])
    vg = v_cache.index_select(0, idx).reshape(-1, v_cache.shape[2], v_cache.shape[3])
    return kg, vg


def strict_gated_attention_fwd_paged(
    q: torch.Tensor,
    gate_logits: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    *,
    window: int | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """O5 rollout form (chunked prefill / decode). The queries of sequence b are the LAST
    ``n_q`` positions of ``seq_lens[b]`` (their K/V were written before this call)."""
    cq = [int(x) for x in cu_seqlens_q.tolist()]
    lens = [int(x) for x in seq_lens.tolist()]
    outs, lses = [], []
    for b in range(len(cq) - 1):
        n_q = cq[b + 1] - cq[b]
        seq_len = lens[b]
        q_pos = torch.arange(seq_len - n_q, seq_len, device=q.device)
        p_min, p_max = seq_len - n_q, seq_len - 1
        b_first = 0 if window is None else max(0, p_min - window + 1) // BLOCK_N
        b_last = p_max // BLOCK_N
        kg, vg = gather_kv_blocks(k_cache, v_cache, block_table[b], b_first, b_last)
        o_b, lse_b = attention_core(
            q[cq[b] : cq[b + 1]],
            kg,
            vg,
            q_pos,
            window=window,
            k_pos0=b_first * BLOCK_N,
            k_len=seq_len,
        )
        outs.append(o_b)
        lses.append(lse_b)
    o = torch.cat(outs, dim=0)
    return _gate(o, gate_logits), o, torch.cat(lses, dim=0)


# --------------------------------------------------------------------------------------------
# O6 SwiGLU FFN
# --------------------------------------------------------------------------------------------
def swiglu_ffn(
    x: torch.Tensor, w_gate: Any, w_up: Any, w_down: Any, *, gemm: Any = None
) -> torch.Tensor:
    """O6: g,u = gemm(x) bf16; h = bf16( silu(fp32 g) * fp32 u ), silu(x) = x / (1 + exp(-x));
    y = row-parallel gemm(h, w_down) with the 8-chunk K tree → bf16."""
    if gemm is None:
        gemm = GoldenGemm()
    g = gemm(x, w_gate).float()
    u = gemm(x, w_up).float()
    h = ((g / (1.0 + torch.exp(-g))) * u).to(torch.bfloat16)
    return row_parallel_gemm(gemm, h, w_down)


# --------------------------------------------------------------------------------------------
# O7 lm_head + softcap + selected logprob
# --------------------------------------------------------------------------------------------
def pad_lm_head_weight(w: torch.Tensor, padded_vocab_size: int | None = None) -> torch.Tensor:
    """Zero-pad ``(V, H)`` to ``(Vp, H)`` (Vp = 8 x 128-aligned, 202752 for Muse Glimmer)."""
    vocab, hidden = w.shape
    if padded_vocab_size is None:
        padded_vocab_size = -(-vocab // 1024) * 1024
    if padded_vocab_size < vocab:
        raise ValueError("padded vocab smaller than vocab")
    out = torch.zeros((padded_vocab_size, hidden), dtype=w.dtype, device=w.device)
    out[:vocab] = w
    return out


def _softcap(z: torch.Tensor, multiplier: float, softcap: float) -> torch.Tensor:
    """ẑ = T · tanh((z · m) / T) in fp32, in the manifest's op order."""
    return torch.tanh((z * multiplier) / softcap) * softcap


def _lm_head_tiles(
    h: torch.Tensor, w_padded: Any, *, multiplier: float, softcap: float, vocab_size: int, gemm
) -> list[torch.Tensor]:
    """Per tile: fp32 logits (NOT rounded), softcapped, padded columns set to -inf."""
    tile_as_kn = isinstance(w_padded, LmHead) and hasattr(gemm, "matmul_kn")
    if isinstance(w_padded, LmHead):
        tile = w_padded.tile
        get = w_padded.tile_kn if tile_as_kn else w_padded.tile_nk
        tile_args = [get(t) for t in range(VOCAB_TILES)]
    else:
        tile = w_padded.shape[0] // VOCAB_TILES
        tile_args = [w_padded[t * tile : (t + 1) * tile] for t in range(VOCAB_TILES)]
    outs = []
    for t, w_t in enumerate(tile_args):
        if tile_as_kn:
            z = gemm.matmul_kn(h, w_t, out_dtype=torch.float32)
        else:
            z = gemm(h, w_t, out_dtype=torch.float32)
        zc = _softcap(z, multiplier, softcap)
        lo = t * tile
        if lo + tile > vocab_size:
            col = torch.arange(lo, lo + tile, device=h.device)
            zc = torch.where((col < vocab_size)[None, :], zc, _NEG_INF)
        outs.append(zc)
    return outs


def _logsumexp_tiles(tiles: list[torch.Tensor]) -> torch.Tensor:
    """Per tile (M_t, S_t) with S_t a fixed tree over the tile; (M, S) combined in the O3 tree."""
    pairs = []
    for zc in tiles:
        m_t = zc.amax(dim=-1)
        s_t = tree_sum_last(torch.exp(zc - m_t[..., None]))
        pairs.append((m_t, s_t))
    level = pairs
    while len(level) > 1:
        nxt = []
        for i in range(0, len(level) - 1, 2):
            (ma, sa), (mb, sb) = level[i], level[i + 1]
            m = torch.maximum(ma, mb)
            nxt.append((m, sa * torch.exp(ma - m) + sb * torch.exp(mb - m)))
        if len(level) % 2:
            nxt.append(level[-1])
        level = nxt
    m, s = level[0]
    return m + torch.log(s)


def lm_head_softcap_logits(
    h: torch.Tensor,
    w_padded: Any,
    *,
    multiplier: float,
    softcap: float,
    vocab_size: int,
    gemm: Any = None,
) -> torch.Tensor:
    """Full softcapped fp32 logits ``(T, Vp)`` (padded columns = -inf). Diagnostics / HF anchor."""
    if gemm is None:
        gemm = GoldenGemm()
    tiles = _lm_head_tiles(
        h, w_padded, multiplier=multiplier, softcap=softcap, vocab_size=vocab_size, gemm=gemm
    )
    return torch.cat(tiles, dim=-1)


def lm_head_softcap_logprob(
    h: torch.Tensor,
    w_padded: Any,
    targets: torch.Tensor,
    *,
    multiplier: float,
    softcap: float,
    vocab_size: int,
    gemm: Any = None,
) -> torch.Tensor:
    """O7: logp(y) = ẑ[y] − Z, fp32 per selected token; ``targets < 0`` give NaN."""
    if gemm is None:
        gemm = GoldenGemm()
    tiles = _lm_head_tiles(
        h, w_padded, multiplier=multiplier, softcap=softcap, vocab_size=vocab_size, gemm=gemm
    )
    tile = tiles[0].shape[-1]
    tgt = targets.to(device=h.device, dtype=torch.long)
    valid = tgt >= 0
    safe = torch.where(valid, tgt, torch.zeros_like(tgt))
    selected = torch.zeros((h.shape[0],), dtype=torch.float32, device=h.device)
    for t, zc in enumerate(tiles):
        in_tile = (safe >= t * tile) & (safe < (t + 1) * tile)
        local = torch.where(in_tile, safe - t * tile, torch.zeros_like(safe))
        picked = zc.gather(1, local[:, None])[:, 0]
        selected = torch.where(in_tile, picked, selected)
    logp = selected - _logsumexp_tiles(tiles)
    return torch.where(valid, logp, torch.full_like(logp, math.nan))


def trace() -> dict[str, Any]:
    """Provenance of the golden fallbacks (contract R4)."""
    return {
        "op": "muse_glimmer_golden_ops",
        "dtype_in": "bf16",
        "dtype_acc": "fp32",
        "dtype_out": "bf16 at contract cast points; fp32 logits/logp/lse",
        "reduction_order": (
            "rows: adjacent-pair halving then sequential (tree_sum_last); attention: 64-key "
            "blocks ascending from floor(max(0,p-W+1)/64), fixed trees over D=128 and 64 keys, "
            "p→bf16 before PV; row-parallel GEMM: 8 K-chunks ((0+1)+(2+3))+((4+5)+(6+7)); "
            "lm_head: 8 vocab tiles, same tree over (M,S)"
        ),
        "tile_config": {"BLOCK_N": BLOCK_N, "KV_PAGE": KV_PAGE, "query_tile": 64},
    }


__all__ = [
    "BLOCK_N",
    "KV_PAGE",
    "GoldenGemm",
    "Linear",
    "LmHead",
    "attention_core",
    "centered_rmsnorm",
    "fixed_tree_partials",
    "fused_post_norm_residual_pre_norm",
    "gather_kv_blocks",
    "lm_head_softcap_logits",
    "lm_head_softcap_logprob",
    "normalized_embedding",
    "pad_lm_head_weight",
    "qknorm_scaled_rope",
    "rope_table",
    "row_parallel_gemm",
    "strict_gated_attention_fwd_paged",
    "strict_gated_attention_fwd_varlen",
    "swiglu_ffn",
    "trace",
    "tree_mean_last",
    "tree_sum_last",
    "weighted_rmsnorm",
]
