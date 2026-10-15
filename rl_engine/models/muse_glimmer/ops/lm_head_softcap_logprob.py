# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Muse-Glimmer-30B strict ``lm_head_softcap_logprob`` (contract O7).

RFC #501 rows: ``lm_head_softcap_logprob`` (WS1) and the single-GPU core of
``vocab_parallel_softcap_logp`` (WS2).  Trace nodes: ``lm_head``, ``logit_softcap``, ``logprob``.

Forward (per selected token, fp32 output)::

    for tile t in 0..7:                       # 8 canonical tiles of Vp/8 rows of W_lm (padded)
        z_t    = gemm(h, W_lm[tile t])        # det_gemm, bf16 x bf16 -> fp32 (NOT rounded to bf16)
        zhat_t = T * tanh((z_t * m) / T)      # fp32; HF order mul m -> div T -> tanh -> mul T
        (M_t, S_t) = (max zhat_t, sum exp(zhat_t - M_t))   over real columns only (pads excluded)
    (M, S) = ((0+1)+(2+3))+((4+5)+(6+7))      # fixed binary tree; node: M=max(Ma,Mb),
                                              #   S = Sa*exp(Ma-M) + Sb*exp(Mb-M)
    logZ   = M + log(S)
    logp   = zhat[target] - logZ

Casts: ``h``/``W_lm`` are bf16 and are consumed by the GEMM with fp32 accumulation; logits,
soft cap, statistics, tree and ``logp`` are fp32 end to end.  ``m``/``T`` are rounded to fp32
once.  No cuBLAS / ``torch.matmul`` anywhere on the strict path.

Backward (deterministic, tile by tile, same tree)::

    dz_t  = (onehot_t - exp(zhat_t - logZ)) * dlogp * (m * (1 - tanh**2))   fp32 -> bf16 once
    dh    = tree_t( gemm(dz_t, W_lm[tile t]^T, ragged_k=True) )  fp32 partials -> bf16 once
                    (K = tile, 8 chunks; a tile that is not a multiple of 256 uses the masked tail)
    dW_lm[tile t] = gemm(dz_t^T, h, ragged_k=True) fp32 -> bf16 ; padded rows are exactly 0
                    (K = T tokens: T_pad = roundup(T, 256), 8 chunks of T_pad/8, rows >= T masked)

The GEMM is pluggable (``gemm(a (M,K) bf16, w (N,K) bf16, *, out_dtype, ragged_k=False) ->
a @ w.T``).  The default is ``fast_det_gemm`` from ``rl_engine.models.muse_glimmer.ops.det_gemm``
(imported eagerly; an import failure propagates, there is no silent fallback).  Operands are
passed as strided views (no transposed copies).  The repo ``det_gemm`` fp32 path
(:func:`legacy_det_gemm_fp32`: one program per output tile, whole K walked in BLOCK_K=32 order,
no K tree, inputs promoted to fp32) is available only as an explicit ``gemm=`` argument
(benchmarks); its bits differ from the default.

Targets: ``validate_targets=True`` (default) checks ``0 <= target < vocab_size`` once on the host
(one device sync per call).  Bindings that must not sync (CUDA-graph capture) pass
``validate_targets=False``: a target outside the real vocabulary then yields ``logp = NaN`` on the
device (never a silent ``-logZ``).

Golden: ``lm_head_softcap_logprob_fp32`` with ``compute_dtype=torch.float64`` is the contract R4
golden (per-row ``torch.mv`` per tile, same chunked sum, same tree).  The fp32 mode is a
convenience: its ``torch.mv`` runs cuBLAS gemv in fp32, whose order is cuBLAS's, so it is a
tolerance reference only.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import torch

from rl_engine.models.muse_glimmer.ops import det_gemm as _det_gemm
from rl_engine.models.muse_glimmer.ops.canonical_tree import combine_tree as _canonical_combine
from rl_engine.models.muse_glimmer.workload import (
    CANONICAL_TP,
    OFFICIAL_FINGERPRINT,
    OFFICIAL_PARALLEL_LAYOUT,
)

OP_NAME = "lm_head_softcap_logprob"
NUM_VOCAB_TILES: int = int(CANONICAL_TP)  # 8 canonical tiles (= canonical TP)
VOCAB_SIZE: int = int(OFFICIAL_FINGERPRINT["vocab_size"])  # 202048
PADDED_VOCAB_SIZE: int = int(OFFICIAL_PARALLEL_LAYOUT["padded_vocab_size"])  # 202752
TILE_SIZE: int = PADDED_VOCAB_SIZE // NUM_VOCAB_TILES  # 25344
OUTPUT_MULTIPLIER: float = float(OFFICIAL_FINGERPRINT["output_multiplier"])
FINAL_LOGIT_SOFTCAP: float = float(OFFICIAL_FINGERPRINT["final_logit_softcapping"])
HIDDEN_SIZE: int = int(OFFICIAL_FINGERPRINT["hidden_size"])

GemmFn = Callable[..., torch.Tensor]
Stats = tuple[torch.Tensor, torch.Tensor]


# --------------------------------------------------------------------------------------
# Geometry and validation
# --------------------------------------------------------------------------------------


def pad_lm_head_weight(w: torch.Tensor, *, padded_vocab: int = PADDED_VOCAB_SIZE) -> torch.Tensor:
    """Append zero rows to ``w (V, H)`` so that it has ``padded_vocab`` rows (bf16 kept)."""

    if w.dim() != 2:
        raise ValueError(f"lm_head weight must be 2D (V, H), got {tuple(w.shape)}")
    vocab, hidden = w.shape
    if padded_vocab < vocab:
        raise ValueError(f"padded_vocab {padded_vocab} < vocab {vocab}")
    if padded_vocab % NUM_VOCAB_TILES != 0:
        raise ValueError(f"padded_vocab {padded_vocab} is not a multiple of {NUM_VOCAB_TILES}")
    if padded_vocab == vocab:
        return w
    pad = torch.zeros(padded_vocab - vocab, hidden, dtype=w.dtype, device=w.device)
    return torch.cat([w, pad], dim=0)


def tile_geometry(padded_vocab: int, vocab_size: int) -> tuple[int, list[tuple[int, int]]]:
    """``(tile_size, [(col_begin, real_cols) per tile])`` for the 8 canonical tiles."""

    if padded_vocab % NUM_VOCAB_TILES != 0:
        raise ValueError(f"padded vocab {padded_vocab} must be a multiple of {NUM_VOCAB_TILES}")
    if not 0 < vocab_size <= padded_vocab:
        raise ValueError(f"vocab_size {vocab_size} must lie in (0, {padded_vocab}]")
    tile = padded_vocab // NUM_VOCAB_TILES
    tiles = []
    for t in range(NUM_VOCAB_TILES):
        begin = t * tile
        tiles.append((begin, min(max(vocab_size - begin, 0), tile)))
    return tile, tiles


def _check_inputs(
    h: torch.Tensor,
    w_padded: torch.Tensor,
    targets: torch.Tensor,
    vocab_size: int,
    *,
    check_targets: bool = True,
) -> None:
    """Shape/dtype/device checks; ``check_targets`` adds the target range check (one sync)."""

    if h.dim() != 2 or w_padded.dim() != 2:
        raise ValueError("h must be (T, H) and w_padded (Vp, H)")
    if h.shape[1] != w_padded.shape[1]:
        raise ValueError(f"hidden mismatch: h {tuple(h.shape)} vs w {tuple(w_padded.shape)}")
    if h.dtype != torch.bfloat16 or w_padded.dtype != torch.bfloat16:
        raise TypeError("h and w_padded must be bfloat16 (contract O7)")
    if targets.dim() != 1 or targets.shape[0] != h.shape[0]:
        raise ValueError(f"targets must be (T,), got {tuple(targets.shape)} for T={h.shape[0]}")
    if targets.dtype != torch.int64:
        raise TypeError("targets must be int64")
    if not (h.device == w_padded.device == targets.device):
        raise ValueError("h, w_padded and targets must share a device")
    tile_geometry(w_padded.shape[0], vocab_size)
    if check_targets and targets.numel() and bool(((targets < 0) | (targets >= vocab_size)).any()):
        raise ValueError(
            f"targets must lie in [0, vocab_size={vocab_size}); a target in the padded "
            f"region [{vocab_size}, {w_padded.shape[0]}) has no probability mass"
        )


# --------------------------------------------------------------------------------------
# GEMM adapters
# --------------------------------------------------------------------------------------


def legacy_det_gemm_fp32(
    a: torch.Tensor,
    w: torch.Tensor,
    *,
    out_dtype: torch.dtype = torch.float32,
    ragged_k: bool = False,
) -> torch.Tensor:
    """``a @ w.T`` through the repo ``det_gemm`` fp32 path (``_triton_gemm_fp32(a, w.T)``).

    Explicit opt-in only (``gemm=legacy_det_gemm_fp32``; never selected implicitly).  Fixed
    BLOCK_M/N/K = 64/64/32, one program per output tile, the whole K walked in order (no K tree,
    no split-K), ``tl.dot(allow_tf32=False)`` with inputs promoted to fp32 inside the kernel, fp32
    output.  Any K is accepted (``ragged_k`` is accepted for interface parity and ignored).
    Slow (CUDA cores) and NOT bitwise equal to the default ``fast_det_gemm``.
    """

    del ragged_k
    if out_dtype != torch.float32:
        raise ValueError("legacy_det_gemm_fp32 only emits float32 (contract O7 logits)")
    det_gemm = importlib.import_module("rl_engine.backends.shared.triton.gemm.det_gemm")
    return det_gemm._triton_gemm_fp32(a, w.t())


legacy_det_gemm_fp32.name = (  # type: ignore[attr-defined]
    "rl_engine.backends.shared.triton.gemm.det_gemm._triton_gemm_fp32"
)


class _FastDetGemmAdapter:
    """The default GEMM: ``fast_det_gemm`` on strided views (no transposed copies), fail closed.

    A reduction length that is not a multiple of 256 raises inside ``fast_det_gemm`` unless the
    call site opts into the declared ragged token rule (``ragged_k=True``, used only for dW).
    """

    name = "fast_det_gemm"

    def __init__(self, fn: GemmFn) -> None:
        self._fn = fn

    def __call__(
        self,
        a: torch.Tensor,
        w: torch.Tensor,
        *,
        out_dtype: torch.dtype = torch.float32,
        ragged_k: bool = False,
    ) -> torch.Tensor:
        return self._fn(a, w, out_dtype=out_dtype, ragged_k=ragged_k)

    def trace(self) -> dict[str, Any]:
        return _det_gemm.trace(out_dtype=torch.float32)


_RESOLVED_GEMM: GemmFn = _FastDetGemmAdapter(_det_gemm.fast_det_gemm)


def resolve_gemm(gemm: GemmFn | None = None) -> GemmFn:
    """``gemm`` if given, else the default ``fast_det_gemm`` adapter (no implicit fallback)."""

    return _RESOLVED_GEMM if gemm is None else gemm


def gemm_backend_name(gemm: GemmFn | None = None) -> str:
    fn = resolve_gemm(gemm)
    return str(getattr(fn, "name", None) or getattr(fn, "__qualname__", repr(fn)))


def _gemm_provenance(gemm: GemmFn | None) -> dict[str, Any]:
    fn = resolve_gemm(gemm)
    tracer = getattr(fn, "trace", None)
    if callable(tracer):
        return tracer()
    return {"op": "det_gemm", "impl": gemm_backend_name(gemm)}


def _combine_tree_sum(partials: Sequence[torch.Tensor]) -> torch.Tensor:
    """Fixed-tree fp32 sum of GEMM partials with the canonical ``combine_tree``."""

    return _canonical_combine(list(partials))


# --------------------------------------------------------------------------------------
# Soft cap, tile statistics and the canonical tree
# --------------------------------------------------------------------------------------


def softcap_golden(z: torch.Tensor, *, multiplier: float, softcap: float) -> torch.Tensor:
    """``T * tanh((z * m) / T)`` in ``z.dtype`` with tensor-valued constants.

    Tensor divisors make torch use true IEEE division (a Python-scalar divisor takes a
    reciprocal-multiply fast path), so fp32 results are bitwise equal to the kernel's
    ``tl.div_rn`` + ``libdevice.tanh`` chain.
    """

    mult = torch.tensor(multiplier, dtype=z.dtype, device=z.device)
    cap = torch.tensor(softcap, dtype=z.dtype, device=z.device)
    return cap * torch.tanh((z * mult) / cap)


def softcap_tile_stats(
    logits_tile: torch.Tensor,
    tile_id: int,
    targets: torch.Tensor,
    *,
    multiplier: float,
    softcap: float,
    vocab_size: int,
    tile_size: int = TILE_SIZE,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Strict per-row ``(m_t, s_t, zhat[target], target_in_tile)`` of one fp32 logits tile.

    Columns with global index ``>= vocab_size`` are excluded; see the Triton module docstring
    for the fixed chunk order.
    """

    if not 0 <= tile_id < NUM_VOCAB_TILES:
        raise ValueError(f"tile_id must be in [0, {NUM_VOCAB_TILES}), got {tile_id}")
    _, tiles = tile_geometry(tile_size * NUM_VOCAB_TILES, vocab_size)
    col_begin, real_cols = tiles[tile_id]
    kernels = importlib.import_module("rl_engine.models.muse_glimmer.triton.softcap_logprob")
    return kernels.launch_tile_stats(
        logits_tile,
        targets,
        tile_size=tile_size,
        real_cols=real_cols,
        col_begin=col_begin,
        multiplier=multiplier,
        softcap=softcap,
    )


def _merge_pair(a: Stats, b: Stats) -> Stats:
    """One tree node: ``M = max(Ma, Mb)``; ``S = Sa*exp(Ma-M) + Sb*exp(Mb-M)`` (elementwise)."""

    m_a, s_a = a
    m_b, s_b = b
    m = torch.maximum(m_a, m_b)
    m_safe = torch.where(m == float("-inf"), torch.zeros_like(m), m)
    s = s_a * torch.exp(m_a - m_safe) + s_b * torch.exp(m_b - m_safe)
    return m, s


def _combine_tree(leaves: Sequence[Any], merge: Callable[[Any, Any], Any]) -> Any:
    """Balanced binary tree over a power-of-two leaf list; 8 leaves: ((0+1)+(2+3))+((4+5)+(6+7))."""

    n = len(leaves)
    if n == 0 or n & (n - 1):
        raise ValueError(f"the canonical tree needs a power-of-two leaf count, got {n}")
    if n == 1:
        return leaves[0]
    half = n // 2
    return merge(_combine_tree(leaves[:half], merge), _combine_tree(leaves[half:], merge))


def merge_tile_stats(stats: Sequence[Stats]) -> Stats:
    """Canonical-tree merge of ``(m_t, s_t)`` leaves into one ``(M, S)`` (no final log).

    With the 8 tile leaves this is the full tree; with a rank's consecutive 8/t tiles it is
    that rank's sub-tree, and merging the ranks' ``(M_r, S_r)`` with this function again gives
    bits identical to the TP1 tree (the sub-trees are nodes of the same balanced tree).
    """

    return _combine_tree([(m, s) for m, s in stats], _merge_pair)


def combine_tile_stats(stats: Sequence[Stats]) -> tuple[torch.Tensor, torch.Tensor]:
    """``(M, logZ)`` with ``logZ = M + log(S)`` after the canonical tree merge."""

    m, s = merge_tile_stats(stats)
    return m, m + torch.log(s)


# --------------------------------------------------------------------------------------
# Tile loop (forward)
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class TileStats:
    tile: int
    m: torch.Tensor
    s: torch.Tensor
    target_logit: torch.Tensor
    target_in_tile: torch.Tensor


def compute_tile_stats(
    h: torch.Tensor,
    w_padded: torch.Tensor,
    targets: torch.Tensor,
    *,
    tiles: Sequence[int] | None = None,
    multiplier: float = OUTPUT_MULTIPLIER,
    softcap: float = FINAL_LOGIT_SOFTCAP,
    vocab_size: int = VOCAB_SIZE,
    gemm: GemmFn | None = None,
    validate_targets: bool = True,
) -> list[TileStats]:
    """GEMM + soft cap + statistics for the requested tiles (default: all 8, in order).

    One fp32 logits tile is alive at a time.  A TP rank calls this with the tiles it owns.
    ``validate_targets`` runs the host-side target range check (one device sync).
    """

    _check_inputs(h, w_padded, targets, vocab_size, check_targets=validate_targets)
    return _compute_tile_stats(
        h, w_padded, targets, tiles=tiles, multiplier=multiplier, softcap=softcap,
        vocab_size=vocab_size, gemm=gemm,
    )


def _compute_tile_stats(
    h: torch.Tensor,
    w_padded: torch.Tensor,
    targets: torch.Tensor,
    *,
    tiles: Sequence[int] | None,
    multiplier: float,
    softcap: float,
    vocab_size: int,
    gemm: GemmFn | None,
) -> list[TileStats]:
    """:func:`compute_tile_stats` without input validation (callers validated once)."""

    gemm_fn = resolve_gemm(gemm)
    tile, geometry = tile_geometry(w_padded.shape[0], vocab_size)
    order = list(range(NUM_VOCAB_TILES)) if tiles is None else [int(t) for t in tiles]
    out: list[TileStats] = []
    for t in order:
        if not 0 <= t < NUM_VOCAB_TILES:
            raise ValueError(f"tile {t} out of range")
        w_t = w_padded[t * tile : (t + 1) * tile]
        z_t = gemm_fn(h, w_t, out_dtype=torch.float32)
        if z_t.dtype != torch.float32 or z_t.shape != (h.shape[0], tile):
            raise RuntimeError("gemm must return fp32 logits of shape (T, tile_size)")
        m_t, s_t, zt, in_t = softcap_tile_stats(
            z_t, t, targets, multiplier=multiplier, softcap=softcap, vocab_size=vocab_size,
            tile_size=tile,
        )
        del z_t
        out.append(TileStats(t, m_t, s_t, zt, in_t))
    return out


def _select_target(stats: Sequence[TileStats], like: torch.Tensor) -> torch.Tensor:
    """zhat[target] from the tile that holds it; NaN where no real column holds the target.

    Device-side fail-loud rule for targets outside ``[0, vocab_size)`` when the host check is
    skipped (``validate_targets=False``): the row's logp becomes NaN instead of ``-logZ``.
    Valid rows are untouched (``torch.where`` selects the same value).
    """

    target_logit = torch.zeros_like(like)
    for st in stats:
        target_logit = torch.where(st.target_in_tile, st.target_logit, target_logit)
    found = torch.stack([st.target_in_tile for st in stats]).any(dim=0)
    return torch.where(found, target_logit, float("nan"))


def _finish_logp(stats: Sequence[TileStats]) -> tuple[torch.Tensor, torch.Tensor]:
    ordered = sorted(stats, key=lambda st: st.tile)
    if [st.tile for st in ordered] != list(range(NUM_VOCAB_TILES)):
        raise ValueError("all 8 tiles are required to finish the log-probability")
    _, logz = combine_tile_stats([(st.m, st.s) for st in ordered])
    return _select_target(ordered, logz) - logz, logz


def _forward_tiles(
    h: torch.Tensor,
    w_padded: torch.Tensor,
    targets: torch.Tensor,
    *,
    multiplier: float,
    softcap: float,
    vocab_size: int,
    gemm: GemmFn | None,
    tile_order: Sequence[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward over the 8 tiles; inputs must already be validated by the caller."""

    stats = _compute_tile_stats(
        h, w_padded, targets, tiles=tile_order, multiplier=multiplier, softcap=softcap,
        vocab_size=vocab_size, gemm=gemm,
    )
    return _finish_logp(stats)


# --------------------------------------------------------------------------------------
# Backward
# --------------------------------------------------------------------------------------


def _backward_tiles(
    h: torch.Tensor,
    w_padded: torch.Tensor,
    targets: torch.Tensor,
    logz: torch.Tensor,
    dlogp: torch.Tensor,
    *,
    multiplier: float,
    softcap: float,
    vocab_size: int,
    gemm: GemmFn | None,
    need_dh: bool,
    need_dw: bool,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    gemm_fn = resolve_gemm(gemm)
    kernels = importlib.import_module("rl_engine.models.muse_glimmer.triton.softcap_logprob")
    tile, geometry = tile_geometry(w_padded.shape[0], vocab_size)
    dlogp32 = dlogp.to(torch.float32).contiguous()
    dh_partials: list[torch.Tensor] = []
    dw = torch.empty_like(w_padded) if need_dw else None
    for t in range(NUM_VOCAB_TILES):
        col_begin, real_cols = geometry[t]
        w_t = w_padded[t * tile : (t + 1) * tile]
        z_t = gemm_fn(h, w_t, out_dtype=torch.float32)  # recompute, one tile alive
        dz_t = kernels.launch_dz(
            z_t, logz, dlogp32, targets, tile_size=tile, real_cols=real_cols,
            col_begin=col_begin, multiplier=multiplier, softcap=softcap,
        )
        del z_t
        if need_dh:
            # dh_t = dz_t @ W_t  ->  gemm(a=dz_t (T,Vt), w=W_t^T (H,Vt)) = dz_t @ W_t.  K = Vt is
            # an input-gradient reduction over output features: det_gemm's dA rule (8 chunks;
            # Vt % 256 != 0, e.g. toy vocabularies, uses the masked tail; 25344 needs none).
            dh_partials.append(gemm_fn(dz_t, w_t.t(), out_dtype=torch.float32, ragged_k=True))
        if dw is not None:
            # dW_t = dz_t^T @ h  ->  gemm(a=dz_t^T (Vt,T), w=h^T (H,T)) = dz_t^T @ h.  K = T is the
            # token axis: declared ragged rule (T_pad = roundup(T, 256), masked rows), any T.
            dw[t * tile : (t + 1) * tile] = gemm_fn(
                dz_t.t(), h.t(), out_dtype=torch.float32, ragged_k=True
            ).to(torch.bfloat16)
        del dz_t
    dh = _combine_tree_sum(dh_partials).to(torch.bfloat16) if need_dh else None
    return dh, dw


class _LMHeadSoftcapLogprobFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h, w_padded, targets, multiplier, softcap, vocab_size, gemm):
        logp, logz = _forward_tiles(
            h, w_padded, targets, multiplier=multiplier, softcap=softcap,
            vocab_size=vocab_size, gemm=gemm,
        )
        ctx.save_for_backward(h, w_padded, targets, logz)
        ctx.multiplier = multiplier
        ctx.softcap = softcap
        ctx.vocab_size = vocab_size
        ctx.gemm = gemm
        return logp

    @staticmethod
    def backward(ctx, dlogp):
        h, w_padded, targets, logz = ctx.saved_tensors
        need_dh, need_dw = ctx.needs_input_grad[0], ctx.needs_input_grad[1]
        dh, dw = _backward_tiles(
            h, w_padded, targets, logz, dlogp, multiplier=ctx.multiplier, softcap=ctx.softcap,
            vocab_size=ctx.vocab_size, gemm=ctx.gemm, need_dh=need_dh, need_dw=need_dw,
        )
        try:
            from rl_engine.ops.autograd.backward_runtime import record_backward

            record_backward(
                OP_NAME,
                kernel_id=f"{gemm_backend_name(ctx.gemm)}+"
                "rl_engine.models.muse_glimmer.triton.softcap_logprob",
                impl="muse_glimmer_lm_head_softcap_logprob",
                family="triton",
            )
        except ImportError:  # provenance is optional
            pass
        return dh, dw, None, None, None, None, None


def lm_head_softcap_logprob(
    h: torch.Tensor,
    w_padded: torch.Tensor,
    targets: torch.Tensor,
    *,
    multiplier: float = OUTPUT_MULTIPLIER,
    softcap: float = FINAL_LOGIT_SOFTCAP,
    vocab_size: int = VOCAB_SIZE,
    gemm: GemmFn | None = None,
    validate_targets: bool = True,
) -> torch.Tensor:
    """Strict selected-token log-probability ``logp (T,) fp32`` with deterministic autograd.

    ``validate_targets=True`` raises ``ValueError`` for a target outside ``[0, vocab_size)``
    (one host sync per call); with ``False`` there is no sync and such a row's logp is NaN.
    """

    _check_inputs(h, w_padded, targets, vocab_size, check_targets=validate_targets)
    return _LMHeadSoftcapLogprobFn.apply(
        h, w_padded, targets, float(multiplier), float(softcap), int(vocab_size), gemm
    )


forward = lm_head_softcap_logprob


def simulate_tp_logprob(
    h: torch.Tensor,
    w_padded: torch.Tensor,
    targets: torch.Tensor,
    *,
    tp: int,
    multiplier: float = OUTPUT_MULTIPLIER,
    softcap: float = FINAL_LOGIT_SOFTCAP,
    vocab_size: int = VOCAB_SIZE,
    gemm: GemmFn | None = None,
) -> torch.Tensor:
    """Single-GPU emulation of ``vocab_parallel_softcap_logp`` at TP ``tp`` in {1,2,4,8}.

    Rank r owns the consecutive tiles ``[r*8/tp, (r+1)*8/tp)``, reduces them with its local
    sub-tree, and the ranks' ``(M_r, S_r)`` are merged with the same tree (the all-gather is
    transport only).  Must be ``torch.equal`` to :func:`lm_head_softcap_logprob`.
    """

    if tp not in (1, 2, 4, 8):
        raise ValueError("tp must be 1, 2, 4 or 8")
    _check_inputs(h, w_padded, targets, vocab_size)
    per_rank = NUM_VOCAB_TILES // tp
    rank_stats: list[Stats] = []
    all_stats: list[TileStats] = []
    for r in range(tp):
        stats = _compute_tile_stats(
            h, w_padded, targets, tiles=range(r * per_rank, (r + 1) * per_rank),
            multiplier=multiplier, softcap=softcap, vocab_size=vocab_size, gemm=gemm,
        )
        rank_stats.append(merge_tile_stats([(st.m, st.s) for st in stats]))
        all_stats.extend(stats)
    _, logz = combine_tile_stats(rank_stats)
    return _select_target(all_stats, logz) - logz


# --------------------------------------------------------------------------------------
# Goldens and debugging helpers
# --------------------------------------------------------------------------------------


def lm_head_softcap_logprob_fp32(
    h: torch.Tensor,
    w_padded: torch.Tensor,
    targets: torch.Tensor,
    *,
    multiplier: float = OUTPUT_MULTIPLIER,
    softcap: float = FINAL_LOGIT_SOFTCAP,
    vocab_size: int = VOCAB_SIZE,
    compute_dtype: torch.dtype = torch.float32,
    block_v: int = 1024,
) -> torch.Tensor:
    """Torch golden (R4 ``forward_fp32``); ``compute_dtype=torch.float64`` is THE contract golden.

    Order: tile-major, row-minor.  Per (tile, row): ``z = torch.mv(W_t, h_i)`` (one gemv of a
    fixed shape, so a row's bits never depend on ``T``), soft cap as :func:`softcap_golden`,
    ``m = max`` over the real columns, ``s = sum_c sum(exp(zhat_c - m))`` over ``block_v``-wide
    chunks in order, then the same canonical tree as the strict path.  Output fp32.
    ``torch.mv`` is cuBLAS gemv, whose internal order is cuBLAS's: in fp64 that noise is far below
    every tolerance (the header of the contract allows an FP64 reference); the fp32 mode is a
    convenience tolerance reference, never a bitwise target (finding lm_head_softcap_logprob:8).
    """

    _check_inputs(h, w_padded, targets, vocab_size)
    rows = h.shape[0]
    tile, geometry = tile_geometry(w_padded.shape[0], vocab_size)
    dev = h.device
    neg_inf = torch.tensor(float("-inf"), dtype=compute_dtype, device=dev)
    m_all = torch.full((NUM_VOCAB_TILES, rows), float("-inf"), dtype=compute_dtype, device=dev)
    s_all = torch.zeros((NUM_VOCAB_TILES, rows), dtype=compute_dtype, device=dev)
    target_logit = torch.zeros(rows, dtype=compute_dtype, device=dev)
    target_list = targets.tolist()
    for t in range(NUM_VOCAB_TILES):
        col_begin, real = geometry[t]
        w_t = w_padded[t * tile : (t + 1) * tile].to(compute_dtype)
        for i in range(rows):
            z = torch.mv(w_t, h[i].to(compute_dtype))
            zhat = softcap_golden(z[:real], multiplier=multiplier, softcap=softcap)
            if real == 0:
                m = neg_inf
                s = torch.zeros((), dtype=compute_dtype, device=dev)
            else:
                m = zhat.max()
                s = torch.zeros((), dtype=compute_dtype, device=dev)
                for c in range(0, real, block_v):
                    s = s + torch.exp(zhat[c : c + block_v] - m).sum()
            m_all[t, i] = m
            s_all[t, i] = s
            y = target_list[i]
            if col_begin <= y < col_begin + real:
                target_logit[i] = zhat[y - col_begin]
    _, logz = combine_tile_stats([(m_all[t], s_all[t]) for t in range(NUM_VOCAB_TILES)])
    return (target_logit - logz).to(torch.float32)


forward_fp32 = lm_head_softcap_logprob_fp32


def softcap_full_logits(
    h: torch.Tensor,
    w_padded: torch.Tensor,
    *,
    multiplier: float = OUTPUT_MULTIPLIER,
    softcap: float = FINAL_LOGIT_SOFTCAP,
    gemm: GemmFn | None = None,
) -> torch.Tensor:
    """``zhat (T, Vp) fp32`` with the strict GEMM and the kernel soft cap (debug / sampling).

    Materialises every tile: 3.3 GB at T=4096 for the real vocabulary -- debugging only.
    """

    gemm_fn = resolve_gemm(gemm)
    kernels = importlib.import_module("rl_engine.models.muse_glimmer.triton.softcap_logprob")
    padded = w_padded.shape[0]
    if padded % NUM_VOCAB_TILES:
        raise ValueError("w_padded rows must be a multiple of 8")
    tile = padded // NUM_VOCAB_TILES
    out = torch.empty((h.shape[0], padded), dtype=torch.float32, device=h.device)
    for t in range(NUM_VOCAB_TILES):
        z_t = gemm_fn(h, w_padded[t * tile : (t + 1) * tile], out_dtype=torch.float32)
        out[:, t * tile : (t + 1) * tile] = kernels.launch_softcap(
            z_t, multiplier=multiplier, softcap=softcap
        )
    return out


class LMHeadSoftcapLogprobOp:
    """Registry-style wrapper (R4): ``forward_fp32`` golden, strict ``forward``, ``trace``."""

    op_class = "logprob"
    is_batch_invariant = True
    backend_id = "triton-muse-glimmer-lm-head-softcap-logprob"

    def __init__(self, gemm: GemmFn | None = None) -> None:
        self.gemm = gemm

    def forward_fp32(
        self,
        h: torch.Tensor,
        w_padded: torch.Tensor,
        targets: torch.Tensor,
        *,
        multiplier: float = OUTPUT_MULTIPLIER,
        softcap: float = FINAL_LOGIT_SOFTCAP,
        vocab_size: int = VOCAB_SIZE,
    ) -> torch.Tensor:
        return lm_head_softcap_logprob_fp32(
            h, w_padded, targets, multiplier=multiplier, softcap=softcap, vocab_size=vocab_size
        )

    def forward(
        self,
        h: torch.Tensor,
        w_padded: torch.Tensor,
        targets: torch.Tensor,
        *,
        multiplier: float = OUTPUT_MULTIPLIER,
        softcap: float = FINAL_LOGIT_SOFTCAP,
        vocab_size: int = VOCAB_SIZE,
    ) -> torch.Tensor:
        return lm_head_softcap_logprob(
            h, w_padded, targets, multiplier=multiplier, softcap=softcap, vocab_size=vocab_size,
            gemm=self.gemm,
        )

    __call__ = forward
    apply = forward

    def trace(self) -> dict[str, Any]:
        return trace(gemm=self.gemm)


# --------------------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------------------


def trace(*, gemm: GemmFn | None = None) -> dict[str, Any]:
    """Contract R4 provenance: constants and reduction orders of this op (stable)."""

    try:
        kernels = importlib.import_module("rl_engine.models.muse_glimmer.triton.softcap_logprob")
        block_v, num_warps = int(kernels.BLOCK_V), int(kernels.NUM_WARPS)
    except ImportError:  # CPU-only host; constants are still the contract values
        block_v, num_warps = 1024, 4
    backend = gemm_backend_name(gemm)
    if backend == "fast_det_gemm":
        gemm_order = (
            "logits z_t = gemm(h, W_t) fp32 out, K = H in 8 canonical chunks (mma.sync chain per "
            "chunk, fixed fp32 tree ((0+1)+(2+3))+((4+5)+(6+7)), no split-K; H % 256 == 0 "
            "required); dh_t = gemm(dz_t, W_t^T, ragged_k=True) K = tile in 8 chunks (masked "
            "tail if tile % 256 != 0; 25344 needs none); dW_t = gemm(dz_t^T, h, ragged_k=True) "
            "K = T: T_pad = roundup(T, 256), 8 chunks, rows >= T masked; backend fast_det_gemm"
        )
    else:
        gemm_order = f"explicit non-default backend {backend} (see gemm.reduction_order)"
    return {
        "op": OP_NAME,
        "rows": ["lm_head_softcap_logprob", "vocab_parallel_softcap_logp"],
        "trace_nodes": ["lm_head", "logit_softcap", "logprob"],
        "dtype_in": "bfloat16",
        "dtype_acc": "float32",
        "dtype_out": "float32",
        "formula": "logp = zhat[y] - logZ ; zhat = T * tanh((z * m) / T) ; z = h @ W_lm[tile]^T",
        "constants": {
            "output_multiplier": OUTPUT_MULTIPLIER,
            "final_logit_softcapping": FINAL_LOGIT_SOFTCAP,
            "vocab_size": VOCAB_SIZE,
            "padded_vocab_size": PADDED_VOCAB_SIZE,
        },
        "reduction_order": {
            "gemm": gemm_order,
            "softcap": "fp32: RN mul by m, IEEE RN div by T (tl.div_rn), libdevice.tanh "
            "(== torch.tanh bitwise), RN mul by T; HF order mul_m -> div_T -> tanh -> mul_T",
            "tile_stats": f"one program per (row, tile); columns walked in BLOCK_V={block_v} "
            "chunks from the tile start; pass 1 m=max; pass 2 s += tl.sum(exp(zhat-m)) per chunk "
            "in order; columns >= vocab_size contribute the identity",
            "tile_tree": "((0+1)+(2+3))+((4+5)+(6+7)); node M=max(Ma,Mb), "
            "S=Sa*exp(Ma-M)+Sb*exp(Mb-M); logZ = M + log(S); TP sub-trees are nodes of this tree",
            "backward": "dz=(onehot-exp(zhat-logZ))*dlogp*(m*(1-tanh^2)) fp32 -> bf16 once; "
            "dh = same 8-leaf tree over fp32 gemm(dz_t, W_t^T) partials -> bf16 once; "
            "dW_t = gemm(dz_t^T, h) fp32 -> bf16; padded rows 0",
        },
        "casts": {
            "logits": "float32 (never rounded to bf16)",
            "dz": "float32 -> bfloat16 before the backward GEMMs",
            "dh": "float32 tree -> bfloat16 once",
            "dw": "float32 -> bfloat16 per tile",
        },
        "targets": (
            "validate_targets=True (default): host range check, ValueError outside "
            "[0, vocab_size); validate_targets=False: no sync, logp = NaN for such rows"
        ),
        "golden": "lm_head_softcap_logprob_fp32(compute_dtype=float64) (fp32 mode: tolerance only)",
        "gemm": _gemm_provenance(gemm),
        "tile_config": {
            "num_tiles": NUM_VOCAB_TILES,
            "tile_size": TILE_SIZE,
            "padded_vocab_size": PADDED_VOCAB_SIZE,
            "padded_columns_in_last_tile": PADDED_VOCAB_SIZE - VOCAB_SIZE,
            "BLOCK_V": block_v,
            "num_warps": num_warps,
            "autotune": False,
        },
    }


__all__ = [
    "FINAL_LOGIT_SOFTCAP",
    "LMHeadSoftcapLogprobOp",
    "NUM_VOCAB_TILES",
    "OP_NAME",
    "OUTPUT_MULTIPLIER",
    "PADDED_VOCAB_SIZE",
    "TILE_SIZE",
    "TileStats",
    "VOCAB_SIZE",
    "combine_tile_stats",
    "compute_tile_stats",
    "forward",
    "forward_fp32",
    "gemm_backend_name",
    "legacy_det_gemm_fp32",
    "lm_head_softcap_logprob",
    "lm_head_softcap_logprob_fp32",
    "merge_tile_stats",
    "pad_lm_head_weight",
    "resolve_gemm",
    "simulate_tp_logprob",
    "softcap_full_logits",
    "softcap_golden",
    "softcap_tile_stats",
    "tile_geometry",
    "trace",
]
