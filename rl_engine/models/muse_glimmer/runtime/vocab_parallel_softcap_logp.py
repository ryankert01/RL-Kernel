# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""``vocab_parallel_softcap_logp`` (RFC #501, WS2): soft-capped selected-token logprob under TP.

Contract (O7): the LM head ``W_lm (202048, 6656)`` is zero-padded to ``202752 = 8 x 25344``
rows. For canonical tile ``t`` the logits ``z_t = det_gemm(h, W_lm[tile t])`` are fp32 (never
rounded to bf16). The soft cap ``zhat = T * tanh((z * m) / T)`` is applied in fp32 BEFORE any
partial (multiplier first, then ``/ T``, ``tanh``, ``* T``, the HF order). Padded columns
(global id >= 202048, all in tile 7) are excluded (``-inf``). Per tile: ``M_t = max zhat``,
``S_t = sum exp(zhat - M_t)`` fixed-order. The 8 ``(M_t, S_t)`` pairs are merged in the fixed
tree ``((0+1)+(2+3))+((4+5)+(6+7))`` with ``M = max(Ma, Mb); S = Sa*exp(Ma-M) + Sb*exp(Mb-M)``
to ``logZ = M + log S``. ``logp(y) = zhat[y] - logZ`` in fp32.

TP t: rank r owns tiles ``[8r/t, 8(r+1)/t)``; the per-tile statistics (``M_t``, ``S_t``, the
target logit if the target falls in the tile, and that ownership flag) travel by a
deterministic rank-ordered all-gather (``all_gather_many``), so every rank holds all 8 tiles'
partials in global tile order and evaluates the same merge tree. TP t is byte-equal to TP1.

Sibling-op dependency (fixed interface, may be absent):
``rl_engine.models.muse_glimmer.ops.lm_head_softcap_logprob.{pad_lm_head_weight,
softcap_tile_stats, combine_tile_stats}``. Labelled fallbacks here: pad with zero rows; a pure
torch fp32 tile statistic whose ``S_t`` is a fixed elementwise tree (lanes 128 -> 1 pairwise,
then 198 lane-blocks zero-padded to 256 -> 1 pairwise; batch-invariant by construction); and the
private merge tree above. The fp32 tile GEMM reuses the row-parallel fallback kernel.
"""

from __future__ import annotations

from typing import Any, Sequence

import torch

from rl_engine.models.muse_glimmer.runtime.tp_canonical_tree import (
    _check_collective,
    _fallback_chunk_partial_fp32,
)
from rl_engine.models.muse_glimmer.runtime.tp_layout import CANONICAL_TREE_ORDER, TPLayout
from rl_engine.models.muse_glimmer.workload import OFFICIAL_FINGERPRINT

Tensor = torch.Tensor

OUTPUT_MULTIPLIER = float(OFFICIAL_FINGERPRINT["output_multiplier"])
FINAL_LOGIT_SOFTCAP = float(OFFICIAL_FINGERPRINT["final_logit_softcapping"])
VOCAB_LANE = 128  # fallback S_t reduction: lanes of 128 then lane-blocks

try:  # fixed interface owned by the lm_head_softcap_logprob agent
    from rl_engine.models.muse_glimmer.ops import (  # type: ignore
        lm_head_softcap_logprob as _sibling,
    )

    SOFTCAP_BACKEND = "muse_glimmer.ops.lm_head_softcap_logprob"
except ImportError:  # pragma: no cover - depends on sibling branch
    _sibling = None
    SOFTCAP_BACKEND = "fallback:pure_torch_fp32_fixed_tree"

try:  # fp32 tile logits: sibling det_gemm with fp32 output, else the repo fp32 kernel
    from rl_engine.models.muse_glimmer.ops.det_gemm import (  # type: ignore
        fast_det_gemm as _sibling_gemm,
    )

    TILE_GEMM_BACKEND = "muse_glimmer.ops.det_gemm.fast_det_gemm(out_dtype=float32)"
except ImportError:  # pragma: no cover - depends on sibling branch
    _sibling_gemm = None
    TILE_GEMM_BACKEND = (
        "fallback:rl_engine.backends.shared.triton.gemm.det_gemm._det_gemm_fp32_kernel"
    )


# --------------------------------------------------------------------------- #
# Weight padding and tile logits
# --------------------------------------------------------------------------- #


def pad_lm_head_weight(w: Tensor, padded_vocab_size: int | None = None) -> Tensor:
    """Zero-pad ``W_lm (V, H)`` to ``(Vp, H)`` (sibling op when present)."""

    if _sibling is not None and hasattr(_sibling, "pad_lm_head_weight"):
        return _sibling.pad_lm_head_weight(w)
    target = (
        int(padded_vocab_size) if padded_vocab_size is not None else TPLayout(1).padded_vocab_size
    )
    if w.dim() != 2 or w.size(0) > target:
        raise ValueError(f"W_lm {tuple(w.shape)} cannot be padded to {target} rows")
    if w.size(0) == target:
        return w
    pad = torch.zeros((target - w.size(0), w.size(1)), dtype=w.dtype, device=w.device)
    return torch.cat([w, pad], dim=0)


def tile_logits_fp32(h: Tensor, w_tile: Tensor) -> Tensor:
    """fp32 ``h @ w_tile^T`` for one tile (T, tile_rows); never rounded to bf16."""

    if _sibling_gemm is not None:
        return _sibling_gemm(h, w_tile, out_dtype=torch.float32)
    return _fallback_chunk_partial_fp32(h, w_tile)


# --------------------------------------------------------------------------- #
# Soft cap and per-tile statistics
# --------------------------------------------------------------------------- #


def softcap_fp32(z: Tensor, *, multiplier: float, softcap: float) -> Tensor:
    """``T * tanh((z * m) / T)`` in fp32, in the HF operation order (mul m, div T, tanh, mul T).

    ``m`` and ``T`` are fp32 tensors (not Python scalars) so torch performs a true IEEE
    division, matching the sibling op's ``softcap_golden`` / ``tl.div_rn`` chain bit for bit.
    """

    if z.dtype != torch.float32:
        raise TypeError("soft cap operates on fp32 logits")
    mult = torch.tensor(float(multiplier), dtype=z.dtype, device=z.device)
    cap = torch.tensor(float(softcap), dtype=z.dtype, device=z.device)
    return cap * torch.tanh((z * mult) / cap)


def _fixed_tree_rowsum(x: Tensor, lane: int = VOCAB_LANE) -> Tensor:
    """Fixed-order row sum over the last dim: pairwise tree over ``lane`` lanes, then a pairwise
    tree over the lane-blocks (zero-padded to a power of two). Elementwise ops only, so the
    result does not depend on the row count (batch invariance by construction)."""

    rows, cols = x.shape
    if cols % lane:
        raise ValueError(f"fallback row sum needs cols {cols} divisible by lane {lane}")
    blocks = cols // lane
    v = x.reshape(rows, blocks, lane)
    while v.size(-1) > 1:
        half = v.size(-1) // 2
        v = v[..., :half] + v[..., half:]
    v = v[..., 0]  # (rows, blocks)
    padded = 1 << (blocks - 1).bit_length()
    if padded != blocks:
        v = torch.cat([v, torch.zeros((rows, padded - blocks), dtype=v.dtype, device=v.device)], 1)
    while v.size(-1) > 1:
        half = v.size(-1) // 2
        v = v[:, :half] + v[:, half:]
    return v[:, 0]


def _fallback_softcap_tile_stats(
    logits_tile: Tensor,
    tile_id: int,
    targets: Tensor,
    *,
    multiplier: float,
    softcap: float,
    vocab_size: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    rows, tile_rows = logits_tile.shape
    start = int(tile_id) * tile_rows
    zhat = softcap_fp32(logits_tile, multiplier=multiplier, softcap=softcap)
    col_ids = torch.arange(start, start + tile_rows, device=zhat.device)
    real = col_ids < int(vocab_size)
    if not bool(real.all()):
        zhat = torch.where(real.unsqueeze(0), zhat, torch.full_like(zhat, float("-inf")))
    m_t = zhat.max(dim=-1).values  # max is order-independent
    finite = m_t > float("-inf")
    m_safe = torch.where(finite, m_t, torch.zeros_like(m_t))
    s_t = _fixed_tree_rowsum(torch.exp(zhat - m_safe.unsqueeze(-1)))
    s_t = torch.where(finite, s_t, torch.zeros_like(s_t))
    in_tile = (targets >= start) & (targets < start + tile_rows)
    local_idx = (targets - start).clamp(0, tile_rows - 1)
    picked = zhat[torch.arange(rows, device=zhat.device), local_idx]
    target_logit = torch.where(in_tile, picked, torch.zeros_like(picked))
    return m_t, s_t, target_logit, in_tile.to(torch.float32)


def softcap_tile_stats(
    logits_tile: Tensor,
    tile_id: int,
    targets: Tensor,
    *,
    multiplier: float,
    softcap: float,
    vocab_size: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """``(M_t, S_t, target_logit, target_in_tile)`` for one fp32 tile ``(T, 25344)``.

    ``target_logit`` is the soft-capped target logit for rows whose target lies in this tile
    and exactly ``0`` elsewhere; ``target_in_tile`` is ``1.0`` / ``0.0`` fp32 so the four
    statistics share one dtype for the grouped all-gather.
    """

    if logits_tile.dtype != torch.float32 or logits_tile.dim() != 2:
        raise TypeError("softcap_tile_stats expects fp32 (T, tile_rows) logits")
    if _sibling is not None and hasattr(_sibling, "softcap_tile_stats"):
        m_t, s_t, target_logit, in_tile = _sibling.softcap_tile_stats(
            logits_tile,
            int(tile_id),
            targets,
            multiplier=multiplier,
            softcap=softcap,
            vocab_size=vocab_size,
        )
        return m_t, s_t, target_logit, in_tile.to(torch.float32)
    return _fallback_softcap_tile_stats(
        logits_tile,
        tile_id,
        targets,
        multiplier=multiplier,
        softcap=softcap,
        vocab_size=vocab_size,
    )


def _merge_pair(ma: Tensor, sa: Tensor, mb: Tensor, sb: Tensor) -> tuple[Tensor, Tensor]:
    m = torch.maximum(ma, mb)
    finite = m > float("-inf")
    m_safe = torch.where(finite, m, torch.zeros_like(m))
    s = sa * torch.exp(ma - m_safe) + sb * torch.exp(mb - m_safe)
    return m, torch.where(finite, s, torch.zeros_like(s))


def _fallback_combine_tile_stats(m_all: Tensor, s_all: Tensor) -> tuple[Tensor, Tensor]:
    n = m_all.size(0)
    if n & (n - 1):
        raise ValueError(f"combine_tile_stats requires a power-of-two tile count, got {n}")
    m, s = m_all, s_all
    while m.size(0) > 1:
        m, s = _merge_pair(m[0::2], s[0::2], m[1::2], s[1::2])
    m0, s0 = m[0], s[0]
    return m0, m0 + torch.log(s0)


def combine_tile_stats(
    stats: Sequence[tuple[Tensor, Tensor]] | tuple[Tensor, Tensor],
) -> tuple[Tensor, Tensor]:
    """Merge the 8 per-tile ``(M_t, S_t)`` in the canonical tree -> ``(M, logZ)`` fp32.

    ``stats`` is either a sequence of ``(M_t, S_t)`` pairs in global tile order or a pair of
    stacked tensors ``(m_all (8, T), s_all (8, T))``.
    """

    if isinstance(stats, tuple) and len(stats) == 2 and isinstance(stats[0], Tensor):
        m_all, s_all = stats
    else:
        pairs = list(stats)
        m_all = torch.stack([m for m, _ in pairs], 0)
        s_all = torch.stack([s for _, s in pairs], 0)
    if _sibling is not None and hasattr(_sibling, "combine_tile_stats"):
        return _sibling.combine_tile_stats([(m, s) for m, s in zip(m_all, s_all, strict=True)])
    return _fallback_combine_tile_stats(m_all, s_all)


# --------------------------------------------------------------------------- #
# Public op
# --------------------------------------------------------------------------- #


def _validate_targets(targets: Tensor, rows: int, vocab_size: int) -> Tensor:
    if targets.dim() != 1 or targets.size(0) != rows:
        raise ValueError(f"targets must be (T,) with T={rows}; got {tuple(targets.shape)}")
    t = targets.to(torch.long)
    bad = (t < 0) | (t >= int(vocab_size))
    if bool(bad.any()):
        values = t[bad]
        raise ValueError(
            f"targets must lie in the real vocabulary [0, {vocab_size}); got values in "
            f"[{int(values.min())}, {int(values.max())}] (padded ids are not selectable)"
        )
    return t


def vocab_parallel_softcap_logp(
    h: Tensor,
    w_local_tiles: Tensor,
    targets: Tensor,
    layout: TPLayout,
    collective: Any = None,
    *,
    multiplier: float = OUTPUT_MULTIPLIER,
    softcap: float = FINAL_LOGIT_SOFTCAP,
    vocab_size: int | None = None,
    rank: int | None = None,
) -> Tensor:
    """fp32 ``logp (T,)`` of ``targets`` under the soft-capped LM head, TP-invariant.

    ``h (T, H)`` bf16 is replicated on every rank; ``w_local_tiles (8/tp * 25344, H)`` bf16 holds
    this rank's consecutive canonical tiles of the zero-padded LM head (the full padded weight at
    TP1); ``targets (T,)`` are replicated real-vocabulary ids (a padded or negative id raises on
    every rank before any collective).

    Reduction order: det_gemm per tile (fp32) -> soft cap (fp32) -> per-tile (M_t, S_t) with a
    fixed-order S_t -> rank-ordered all-gather of the tile statistics -> merge tree
    ``((0+1)+(2+3))+((4+5)+(6+7))`` -> ``logp = zhat[y] - (M + log S)``.
    """

    if h.dtype != torch.bfloat16 or w_local_tiles.dtype != torch.bfloat16:
        raise TypeError("vocab_parallel_softcap_logp requires bf16 hidden states and weights")
    if h.dim() != 2 or w_local_tiles.dim() != 2 or h.size(1) != w_local_tiles.size(1):
        raise ValueError(
            f"expected h (T, H) and w_local_tiles (rows, H); got {tuple(h.shape)} and "
            f"{tuple(w_local_tiles.shape)}"
        )
    rank = _check_collective(collective, layout, rank)
    vocab = layout.vocab_size if vocab_size is None else int(vocab_size)
    if vocab > layout.padded_vocab_size:
        raise ValueError(f"vocab_size {vocab} exceeds padded_vocab_size {layout.padded_vocab_size}")
    tile_rows = layout.vocab_tile_rows
    tiles = layout.vocab_tiles(rank)
    if w_local_tiles.size(0) != len(tiles) * tile_rows:
        raise ValueError(
            f"rank {rank} at tp={layout.tp} owns {len(tiles)} tiles x {tile_rows} rows = "
            f"{len(tiles) * tile_rows}; got {w_local_tiles.size(0)} weight rows"
        )
    targets_long = _validate_targets(targets.to(h.device), h.size(0), vocab)

    m_parts, s_parts, tl_parts, own_parts = [], [], [], []
    for local_index, tile_id in enumerate(tiles):
        w_tile = w_local_tiles.narrow(0, local_index * tile_rows, tile_rows)
        z_tile = tile_logits_fp32(h, w_tile)
        m_t, s_t, target_logit, in_tile = softcap_tile_stats(
            z_tile,
            tile_id,
            targets_long,
            multiplier=multiplier,
            softcap=softcap,
            vocab_size=vocab,
        )
        m_parts.append(m_t)
        s_parts.append(s_t)
        tl_parts.append(target_logit)
        own_parts.append(in_tile)
    local = tuple(torch.stack(p, 0).contiguous() for p in (m_parts, s_parts, tl_parts, own_parts))

    if layout.tp == 1:
        m_all, s_all, tl_all, own_all = local
    else:
        m_all, s_all, tl_all, own_all = collective.all_gather_many(local)  # (8, T) each
    _, log_z = combine_tile_stats((m_all, s_all))
    owner = torch.div(targets_long, tile_rows, rounding_mode="floor")
    rows = torch.arange(h.size(0), device=h.device)
    if not bool((own_all[owner, rows] == 1.0).all()):
        raise RuntimeError("target ownership flags disagree with the tile layout")
    target_logit = tl_all[owner, rows]
    return target_logit - log_z


def vocab_softcap_logp_reference_fp64(
    tile_logits: Sequence[Tensor],
    targets: Tensor,
    *,
    multiplier: float = OUTPUT_MULTIPLIER,
    softcap: float = FINAL_LOGIT_SOFTCAP,
    vocab_size: int,
) -> Tensor:
    """fp64 ``logp`` from the SAME fp32 tile logits (concatenated in tile order): soft cap and
    logsumexp in fp64 over the real vocabulary. Checks the fp32 merge, not the GEMM."""

    z = torch.cat([t.to(torch.float64) for t in tile_logits], dim=1)[:, : int(vocab_size)]
    zhat = torch.tanh((z * float(multiplier)) / float(softcap)) * float(softcap)
    log_z = torch.logsumexp(zhat, dim=-1)
    rows = torch.arange(z.size(0), device=z.device)
    return zhat[rows, targets.to(torch.long)] - log_z


def trace(layout: TPLayout, collective: Any = None) -> dict[str, Any]:
    return {
        "op": "vocab_parallel_softcap_logp",
        "dtype_in": "bfloat16",
        "dtype_acc": "float32",
        "dtype_logits": "float32",
        "dtype_out": "float32",
        "softcap": {
            "form": "T * tanh((z * m) / T)",
            "T": FINAL_LOGIT_SOFTCAP,
            "m": OUTPUT_MULTIPLIER,
        },
        "reduction_order": (
            "det_gemm per tile -> soft cap -> per-tile (max, sumexp) fixed order -> "
            f"rank-ordered all-gather -> merge tree {CANONICAL_TREE_ORDER} -> logp"
        ),
        "tile_config": {
            "vocab_size": layout.vocab_size,
            "padded_vocab_size": layout.padded_vocab_size,
            "tile_rows": layout.vocab_tile_rows,
            "tiles": layout.canonical_tp,
            "tiles_per_rank": layout.chunks_per_rank,
            "padded_logits": "excluded_from_logsumexp",
            "fallback_sumexp_order": (
                f"lanes {VOCAB_LANE} pairwise, then lane-blocks padded to 256 pairwise"
            ),
        },
        "softcap_backend": SOFTCAP_BACKEND,
        "tile_gemm_backend": TILE_GEMM_BACKEND,
        "collective_backend": (
            None if collective is None else getattr(collective, "backend_id", None)
        ),
        "tile_to_rank": {t: layout.chunk_owner(t) for t in range(layout.canonical_tp)},
    }


__all__ = [
    "FINAL_LOGIT_SOFTCAP",
    "OUTPUT_MULTIPLIER",
    "SOFTCAP_BACKEND",
    "TILE_GEMM_BACKEND",
    "combine_tile_stats",
    "pad_lm_head_weight",
    "softcap_fp32",
    "softcap_tile_stats",
    "tile_logits_fp32",
    "trace",
    "vocab_parallel_softcap_logp",
    "vocab_softcap_logp_reference_fp64",
]
