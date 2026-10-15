# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Context-parallel strict gated attention for Muse-Glimmer-30B.

RFC #501 WS2 rows ``cp_global_attention`` (full-causal NoPE layers) and ``cp_local_attention``
(sliding-window layers, window on ABSOLUTE positions), both on top of contract O5
(``docs/contracts/muse-glimmer-30b-strict-contract.md``).

Layout
------
Only the #235 ``allgather`` layout is supported: CP rank ``i`` of ``c`` owns the CONTIGUOUS
token chunk ``[i*T/c, (i+1)*T/c)`` of one logical sequence of ``T`` tokens, for queries, keys,
values and the gate pre-activation alike. Any other layout name (``zigzag``, ``striped``,
``ring``, ...) raises :class:`CPLayoutError` at plan construction, and tensors whose positions
are not that contiguous chunk raise the same error at call time. The rule is fail closed: a
layout that needs an ``(out, lse)`` merge has a second reduction order and cannot be bitwise
equal to CP1 under this contract.

Forward (one arithmetic graph at every CP degree)::

    K_full, V_full, pos_full = AG(K_local, V_local, pos_local)   # rank-ordered copy, no arithmetic
    require pos_full == arange(T)                                 # allgather-layout invariant
    y, o, lse = strict_gated_attention(q_local, K_full, V_full, g_local,
                                       q_pos = pos_local (ABSOLUTE), window)

Why CP-c == CP-1 bitwise (contract O5): the kernel's per-query arithmetic is a pure function of
the query row, its absolute position and the K/V bits of the keys it may attend to. It depends
neither on which other queries share the launch nor on extra fully-masked key blocks (those are
bitwise no-ops). A CP rank therefore reproduces exactly the bits CP1 produces for the same rows.
The sliding window ``q_pos - W < k_pos <= q_pos`` is evaluated on absolute positions against the
gathered full K/V, so a window crossing a chunk boundary reads keys owned by another rank with
no special casing.

Gate
----
There is NO ``(out, lse)`` merge on this layout: each rank finishes the whole softmax of its own
queries, so ``(out, lse)`` are final where they are produced and the kernel epilogue applies
``y = (o -> bf16 -> fp32) * sigmoid(g -> fp32) -> bf16`` on the owner rank's rows (O5). The RFC
sentence "apply the output gate after the CP merge" has no object here; see REPORT_ws2_cp.md.

Backward (training)
-------------------
``dq`` and ``dgate`` are row-local to the owner rank (same O5 argument; the full K/V is present).
``dK``/``dV`` need every rank's queries. So that CP-c bits equal CP-1 bits, the query axis is
reduced in ONE canonical two-level fp32 tree (the O3 K-chunk rule applied to the query axis):

* inner level: the kernel's 8 Q-head-group partials, ``((0+1)+(2+3))+((4+5)+(6+7))``;
* outer level: :data:`CP_CANONICAL_QUERY_CHUNKS` = 8 contiguous query chunks of the FULL
  sequence; each chunk is accumulated by the kernel over its own query blocks in ascending
  order and the chunk partials are combined in the same tree. CP rank ``i`` of ``c`` owns
  chunks ``[i*8/c, (i+1)*8/c)``, combines them locally (its sub-tree root), and the ranks'
  roots are combined by a deterministic rank-ordered all_gather followed by the same tree
  evaluated locally. No NCCL reduce-scatter is ever used.

Cast points: K/V/positions move as bits; the fp32 tree root of dK/dV is rounded to bf16 once;
``y``, ``o``, ``lse``, ``dq`` and ``dgate`` carry the kernel's own casts (O5).

Sibling ops
-----------
The strict kernel is imported from the WS1 siblings
(``rl_engine.models.muse_glimmer.ops.gated_attention.strict_gated_attention_fwd_varlen`` and
``...triton.gated_attention_bwd.{gated_attention_bwd_preprocess,
strict_gated_attention_bwd_partials}``). While they are absent
this module runs :func:`fallback_gated_attention_fwd` /
:func:`fallback_gated_attention_bwd_partials`: a pure-torch fp32 stand-in with FIXED tensor
shapes (one GQA group x 8 query tiles of 64 rows per GEMM) that follows the O5 block order and
casts, so that its per-row bits are independent of the launch composition and the CP tests are
meaningful today. It is reported as a fallback by :func:`trace` and is NOT the strict kernel.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import torch

__all__ = [
    "CP_CANONICAL_QUERY_CHUNKS",
    "CPAttentionPlan",
    "CPGatedAttentionFn",
    "CPLayoutError",
    "FORWARD_IMPL",
    "BACKWARD_IMPL",
    "SUPPORTED_CP_WORLDS",
    "SUPPORTED_LAYOUTS",
    "TPCPTopology",
    "all_gather_sequence",
    "combine_tree",
    "cp_gated_attention",
    "cp_gated_attention_backward",
    "cp_gated_attention_forward",
    "cp_tp_gated_attention_forward",
    "fallback_gated_attention_bwd_partials",
    "fallback_gated_attention_fwd",
    "gated_attention_rows",
    "make_cp_collective",
    "trace",
]

# --------------------------------------------------------------------------------------------
# Pinned Muse-Glimmer-30B attention geometry (museglimmer_manifest.json) and contract constants
# --------------------------------------------------------------------------------------------
NUM_Q_HEADS = 32
NUM_KV_HEADS = 2
HEAD_DIM = 128
GQA_GROUP = NUM_Q_HEADS // NUM_KV_HEADS  # 16
HEADS_PER_CANONICAL_GROUP = 4  # O5: dK/dV partials per 4-head group (TP8 shard)
SLIDING_WINDOW = 2048

SUPPORTED_CP_WORLDS = (1, 2, 4, 8)
SUPPORTED_LAYOUTS = ("allgather",)
CP_CANONICAL_QUERY_CHUNKS = 8
REJECTED_LAYOUTS = ("zigzag", "striped", "ring", "interleaved")

# O5 tile constants (compile-time; never autotuned). The fallback mirrors them.
BLOCK_N = 64
BLOCK_M = 64
# Fixed number of query tiles per fallback GEMM so that every GEMM has one shape
# (GQA_GROUP * FALLBACK_TILE_GROUP rows of BLOCK_M) regardless of T, CP degree or TP shard.
FALLBACK_TILE_GROUP = 8

_NEG_INF = float("-inf")


class CPLayoutError(ValueError):
    """Raised when a CP layout other than the contiguous ``allgather`` layout is requested."""


# --------------------------------------------------------------------------------------------
# Sibling ops (fixed interfaces) with clearly-labelled fallbacks
# --------------------------------------------------------------------------------------------
try:  # canonical fp32 tree (O3); tiny pure-torch sibling
    from rl_engine.models.muse_glimmer.ops.canonical_tree import combine_tree as _sibling_tree

    def combine_tree(partials: Sequence[torch.Tensor]) -> torch.Tensor:
        """Canonical fixed fp32 tree ``((0+1)+(2+3))+((4+5)+(6+7))`` (sibling implementation)."""
        return _sibling_tree(partials)

    TREE_IMPL = "rl_engine.models.muse_glimmer.ops.canonical_tree.combine_tree"
except ImportError:  # pragma: no cover - depends on the sibling worktree

    def combine_tree(partials: Sequence[torch.Tensor]) -> torch.Tensor:
        """Canonical fixed fp32 tree ``((0+1)+(2+3))+((4+5)+(6+7))`` (local copy of O3).

        1, 2, 4 or 8 fp32 partials in chunk/rank order; every ``+`` is one exact IEEE fp32
        add, so the result is bitwise identical to the same adds performed inside a kernel.
        """
        parts = list(partials)
        count = len(parts)
        if count not in (1, 2, 4, 8):
            raise ValueError(f"combine_tree expects 1, 2, 4 or 8 partials, got {count}")
        for index, part in enumerate(parts):
            if part.dtype != torch.float32:
                raise TypeError(f"combine_tree partial {index} must be float32, got {part.dtype}")
            if part.shape != parts[0].shape:
                raise ValueError("combine_tree partials must share one shape")
        if count == 1:
            return parts[0]
        if count == 2:
            return parts[0] + parts[1]
        if count == 4:
            return (parts[0] + parts[1]) + (parts[2] + parts[3])
        lower = (parts[0] + parts[1]) + (parts[2] + parts[3])
        upper = (parts[4] + parts[5]) + (parts[6] + parts[7])
        return lower + upper

    TREE_IMPL = "FALLBACK:cp_attention.combine_tree (local copy of the O3 tree)"


_strict_fwd = None
FORWARD_IMPL = (
    "FALLBACK:cp_attention.fallback_gated_attention_fwd (pure torch fp32, O5 block order)"
)
for _module_name in (
    "rl_engine.models.muse_glimmer.ops.gated_attention",
    "rl_engine.models.muse_glimmer.triton.gated_attention_fwd",
):
    try:
        _module = __import__(_module_name, fromlist=["strict_gated_attention_fwd_varlen"])
        _strict_fwd = getattr(_module, "strict_gated_attention_fwd_varlen")
        FORWARD_IMPL = f"{_module_name}.strict_gated_attention_fwd_varlen"
        break
    except (ImportError, AttributeError):  # pragma: no cover - depends on the sibling worktree
        continue

_strict_bwd_partials = None
BACKWARD_IMPL = (
    "FALLBACK:cp_attention.fallback_gated_attention_bwd_partials (pure torch fp32, FA2 order)"
)
try:  # sibling Triton backward: row-local preprocess + fp32 dK/dV partials per 4-head group
    from rl_engine.models.muse_glimmer.triton import gated_attention_bwd as _bwd_module

    def _strict_bwd_partials(  # type: ignore[misc]
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
        """Sibling kernels -> ``(dq, dk_partials (8,Tk,2,128) fp32, dv_partials, dgate)``."""
        dgate, do, delta = _bwd_module.gated_attention_bwd_preprocess(dy, o, gate_logits)
        dq, dk_partials, dv_partials = _bwd_module.strict_gated_attention_bwd_partials(
            q, k, v, do, lse, delta, cu_seqlens_q, cu_seqlens_k, q_pos, window=window
        )
        return dq, dk_partials, dv_partials, dgate

    BACKWARD_IMPL = (
        "rl_engine.models.muse_glimmer.triton.gated_attention_bwd."
        "gated_attention_bwd_preprocess + strict_gated_attention_bwd_partials"
    )
except ImportError:  # pragma: no cover - depends on the sibling worktree
    try:  # interface location named by the RFC brief
        from rl_engine.models.muse_glimmer.ops.gated_attention_autograd import (
            strict_gated_attention_bwd_partials as _strict_bwd_partials,
        )

        BACKWARD_IMPL = (
            "rl_engine.models.muse_glimmer.ops.gated_attention_autograd."
            "strict_gated_attention_bwd_partials"
        )
    except ImportError:
        pass


# --------------------------------------------------------------------------------------------
# Plan / topology
# --------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class CPAttentionPlan:
    """Contiguous-chunk (``allgather``) context-parallel plan for one logical sequence.

    ``cp_rank`` owns tokens ``[cp_rank*T/cp_world, (cp_rank+1)*T/cp_world)``. ``T`` must be a
    multiple of ``cp_world`` and, for the backward, of :data:`CP_CANONICAL_QUERY_CHUNKS`.
    """

    cp_world: int
    cp_rank: int
    layout: str = "allgather"

    def __post_init__(self) -> None:
        if isinstance(self.cp_world, bool) or not isinstance(self.cp_world, int):
            raise ValueError(f"cp_world must be an int, got {self.cp_world!r}")
        if self.cp_world not in SUPPORTED_CP_WORLDS:
            raise ValueError(f"cp_world must be one of {SUPPORTED_CP_WORLDS}, got {self.cp_world}")
        if isinstance(self.cp_rank, bool) or not isinstance(self.cp_rank, int):
            raise ValueError(f"cp_rank must be an int, got {self.cp_rank!r}")
        if not 0 <= self.cp_rank < self.cp_world:
            raise ValueError(f"cp_rank {self.cp_rank} out of range for cp_world {self.cp_world}")
        if not isinstance(self.layout, str):
            raise CPLayoutError(f"layout must be a string, got {self.layout!r}")
        if self.layout not in SUPPORTED_LAYOUTS:
            hint = (
                " (needs an (out, lse) merge whose reduction order differs from CP1; "
                "the strict contract fails closed)"
                if self.layout in REJECTED_LAYOUTS
                else ""
            )
            raise CPLayoutError(
                f"CP layout {self.layout!r} is not supported; the strict Muse-Glimmer CP path "
                f"accepts only {SUPPORTED_LAYOUTS}{hint}"
            )

    def local_token_range(self, total_tokens: int) -> tuple[int, int]:
        """``[start, end)`` of this rank's contiguous chunk of a ``total_tokens`` sequence."""
        total_tokens = int(total_tokens)
        if total_tokens <= 0 or total_tokens % self.cp_world:
            raise ValueError(
                f"total_tokens={total_tokens} must be a positive multiple of "
                f"cp_world={self.cp_world}"
            )
        chunk = total_tokens // self.cp_world
        return self.cp_rank * chunk, (self.cp_rank + 1) * chunk

    def local_positions(self, total_tokens: int, device: torch.device | str) -> torch.Tensor:
        """Absolute int32 positions of this rank's chunk."""
        start, end = self.local_token_range(total_tokens)
        return torch.arange(start, end, dtype=torch.int32, device=device)

    def canonical_chunk_ids(self) -> tuple[int, ...]:
        """Canonical query chunks (of :data:`CP_CANONICAL_QUERY_CHUNKS`) owned by this rank."""
        per_rank = CP_CANONICAL_QUERY_CHUNKS // self.cp_world
        return tuple(range(self.cp_rank * per_rank, (self.cp_rank + 1) * per_rank))

    def validate_local_positions(self, positions_local: torch.Tensor, total_tokens: int) -> None:
        """Fail closed unless ``positions_local`` is exactly this rank's contiguous chunk."""
        if positions_local.ndim != 1:
            raise CPLayoutError("positions_local must be a 1-D tensor of absolute positions")
        if positions_local.dtype not in (torch.int32, torch.int64):
            raise CPLayoutError("positions_local must be int32 or int64")
        start, end = self.local_token_range(total_tokens)
        if positions_local.numel() != end - start:
            raise CPLayoutError(
                f"rank {self.cp_rank}/{self.cp_world} must hold {end - start} tokens, got "
                f"{positions_local.numel()}"
            )
        expected = torch.arange(
            start, end, dtype=positions_local.dtype, device=positions_local.device
        )
        if not torch.equal(positions_local, expected):
            raise CPLayoutError(
                "positions_local is not the contiguous allgather chunk "
                f"[{start}, {end}) of rank {self.cp_rank}; zigzag/striped/permuted token layouts "
                "are rejected (fail closed)"
            )

    def provenance(self) -> dict[str, Any]:
        return {
            "cp_world": self.cp_world,
            "cp_rank": self.cp_rank,
            "cp_layout": self.layout,
            "cp_canonical_query_chunks": CP_CANONICAL_QUERY_CHUNKS,
            "cp_local_canonical_chunks": list(self.canonical_chunk_ids()),
        }


@dataclass(frozen=True)
class TPCPTopology:
    """One TP x CP topology: TP shards heads, CP shards the sequence (``allgather`` layout).

    Global rank ``r`` maps to ``tp_rank = r % tp_world`` and ``cp_rank = r // tp_world``, so the
    TP group is contiguous ranks (NVLink-friendly) and the CP group is strided. TP rank ``t``
    owns Q heads ``[t*32/tp, (t+1)*32/tp)``, KV head(s) ``[t*2/tp, (t+1)*2/tp)`` (``tp <= 2``:
    every TP rank owns exactly its KV head, so attention needs no TP collective at all) and the
    gate channels of its Q heads.
    """

    tp_world: int
    tp_rank: int
    cp_world: int
    cp_rank: int

    def __post_init__(self) -> None:
        if self.tp_world not in (1, 2):
            raise ValueError(
                "TPCPTopology supports tp_world in (1, 2): with 2 KV heads every TP rank owns "
                "its own KV head; TP4/TP8 replicate KV heads (kv_head_replication) and are out "
                f"of scope for this POC; got {self.tp_world}"
            )
        if not 0 <= self.tp_rank < self.tp_world:
            raise ValueError(f"tp_rank {self.tp_rank} out of range for tp_world {self.tp_world}")
        CPAttentionPlan(self.cp_world, self.cp_rank)  # validates the CP axis

    @classmethod
    def from_global_rank(cls, rank: int, *, tp_world: int, cp_world: int) -> "TPCPTopology":
        if rank < 0 or rank >= tp_world * cp_world:
            raise ValueError(
                f"rank {rank} out of range for tp_world*cp_world={tp_world * cp_world}"
            )
        return cls(
            tp_world=tp_world,
            tp_rank=rank % tp_world,
            cp_world=cp_world,
            cp_rank=rank // tp_world,
        )

    @property
    def plan(self) -> CPAttentionPlan:
        return CPAttentionPlan(self.cp_world, self.cp_rank)

    def cp_group_ranks(self) -> tuple[int, ...]:
        """Global ranks sharing this TP shard (the CP process group of this rank)."""
        return tuple(self.tp_rank + j * self.tp_world for j in range(self.cp_world))

    @staticmethod
    def cp_group_ranks_for(tp_rank: int, *, tp_world: int, cp_world: int) -> tuple[int, ...]:
        return tuple(tp_rank + j * tp_world for j in range(cp_world))

    @property
    def q_heads(self) -> int:
        return NUM_Q_HEADS // self.tp_world

    @property
    def kv_heads(self) -> int:
        return NUM_KV_HEADS // self.tp_world

    def q_head_range(self) -> tuple[int, int]:
        return self.tp_rank * self.q_heads, (self.tp_rank + 1) * self.q_heads

    def kv_head_range(self) -> tuple[int, int]:
        return self.tp_rank * self.kv_heads, (self.tp_rank + 1) * self.kv_heads

    def gate_channel_range(self) -> tuple[int, int]:
        lo, hi = self.q_head_range()
        return lo * HEAD_DIM, hi * HEAD_DIM

    def provenance(self) -> dict[str, Any]:
        return {
            "tp_world": self.tp_world,
            "tp_rank": self.tp_rank,
            "q_head_range": list(self.q_head_range()),
            "kv_head_range": list(self.kv_head_range()),
            "gate_channel_range": list(self.gate_channel_range()),
            **self.plan.provenance(),
        }


# --------------------------------------------------------------------------------------------
# Collective helpers (pure data movement)
# --------------------------------------------------------------------------------------------
def make_cp_collective(group: Any = None, device: torch.device | int | None = None) -> Any:
    """Deterministic rank-ordered collective for the CP group.

    Prefers the repo's CUDA-IPC fixed-tree collective (``rl_engine._C``). When the compiled
    extension is absent (this POC environment) it uses the torch.distributed transport
    ``TorchDistributedDeterministicCollective``: ``all_gather`` is ``all_gather_into_tensor``
    (bitwise data movement, rank order), reductions are local balanced-rank-tree sums. Both
    expose ``all_gather``, ``all_gather_many`` and ``reduce_scatter`` with identical semantics.
    """
    try:
        import rl_engine._C  # noqa: F401

        have_extension = True
    except ImportError:
        have_extension = False
    if have_extension and getattr(torch.version, "hip", None) is None:
        from rl_engine.distributed.algorithms.collectives import create_deterministic_collective

        return create_deterministic_collective(group=group, device=device)
    from rl_engine.distributed.transports.rccl import TorchDistributedDeterministicCollective

    return TorchDistributedDeterministicCollective(group=group, device=device)


def all_gather_sequence(local: torch.Tensor, collective: Any) -> torch.Tensor:
    """Rank-ordered gather of a sequence-leading tensor; a pure copy, no arithmetic."""
    return collective.all_gather(local.contiguous())


def _all_gather_many(tensors: Sequence[torch.Tensor], collective: Any) -> tuple[torch.Tensor, ...]:
    values = tuple(t.contiguous() for t in tensors)
    many = getattr(collective, "all_gather_many", None)
    if callable(many):
        return tuple(many(values))
    return tuple(collective.all_gather(v) for v in values)


# --------------------------------------------------------------------------------------------
# Fallback kernel (pure torch, fixed shapes, O5 block order) - NOT the strict kernel
# --------------------------------------------------------------------------------------------
def _disable_tf32() -> None:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def _check_single_sequence(
    cu_seqlens_q: torch.Tensor, cu_seqlens_k: torch.Tensor, tq: int, tk: int
) -> None:
    cq = cu_seqlens_q.tolist()
    ck = cu_seqlens_k.tolist()
    if cq != [0, tq] or ck != [0, tk]:
        raise NotImplementedError(
            "the fallback gated-attention kernel handles one sequence per call "
            f"(cu_seqlens_q={cq}, cu_seqlens_k={ck}); packed varlen needs the strict kernel"
        )


def _tile_rows(x: torch.Tensor, group: int) -> torch.Tensor:
    """``(TILE_GROUP*BLOCK_M, G, D) -> (G*TILE_GROUP, BLOCK_M, D)`` head-major, tile-minor."""
    return (
        x.permute(1, 0, 2)
        .reshape(group, FALLBACK_TILE_GROUP, BLOCK_M, x.size(-1))
        .reshape(group * FALLBACK_TILE_GROUP, BLOCK_M, x.size(-1))
    )


def _untile_rows(x: torch.Tensor, group: int) -> torch.Tensor:
    """Inverse of :func:`_tile_rows`: ``(G*TILES, BLOCK_M, D) -> (TILES*BLOCK_M, G, D)``."""
    return (
        x.reshape(group, FALLBACK_TILE_GROUP * BLOCK_M, x.size(-1)).permute(1, 0, 2).contiguous()
    )


def _pairwise_sum_last(x: torch.Tensor) -> torch.Tensor:
    """Fixed balanced-tree sum over a power-of-two last dim using only elementwise adds.

    Elementwise ops are batch-invariant by construction, so the per-row result does not depend
    on how many rows share the tensor (unlike ``torch.sum`` whose kernel config does).
    """
    while x.size(-1) > 1:
        x = x[..., 0::2] + x[..., 1::2]
    return x[..., 0]


def _block_range(p_min: int, p_max: int, tk: int, window: Optional[int]) -> tuple[int, int]:
    first = 0 if window is None else max(p_min - window + 1, 0) // BLOCK_N
    last = min(p_max, tk - 1) // BLOCK_N
    return first, last


def fallback_gated_attention_fwd(
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
    """FALLBACK for ``strict_gated_attention_fwd_varlen``: ``(y bf16, o bf16, lse fp32)``.

    Pure torch fp32 with the O5 block order: KV blocks of 64 aligned at absolute key position 0,
    ascending from ``floor(max(0, p_min-W+1)/64)`` (local) or 0 (global) to ``floor(p_max/64)``;
    ``m_new = max(m, rowmax)``; ``alpha = exp(m - m_new)``; ``p = exp(s - m_new)``;
    ``l = l*alpha + rowsum(p)``; ``acc = acc*alpha + (p -> bf16) @ V``; ``o = acc / l -> bf16``;
    ``y = (o -> fp32) * sigmoid(g) -> bf16``; ``lse = m + log(l)``. Every GEMM has the FIXED
    shape (GQA_GROUP * 8 tiles * 64 rows) so a row's bits never depend on the launch.
    Shapes: ``q (Tq, Hq, 128)``, ``k/v (Tk, Hkv, 128)``, ``gate (Tq, Hq*128)``, ``q_pos (Tq,)``.
    """
    _disable_tf32()
    tq, hq, d = q.shape
    tk, hkv, _ = k.shape
    group = hq // hkv
    if hq % hkv or group != GQA_GROUP or d != HEAD_DIM:
        raise ValueError(f"unsupported head geometry Hq={hq}, Hkv={hkv}, D={d}")
    _check_single_sequence(cu_seqlens_q, cu_seqlens_k, tq, tk)
    device = q.device
    sm_scale = float(d) ** -0.5
    rows_per_group = FALLBACK_TILE_GROUP * BLOCK_M
    n_row_groups = -(-tq // rows_per_group)
    tq_pad = n_row_groups * rows_per_group
    tk_pad = -(-tk // BLOCK_N) * BLOCK_N

    q_pad = torch.zeros((tq_pad, hq, d), dtype=torch.bfloat16, device=device)
    q_pad[:tq] = q
    g_pad = torch.zeros((tq_pad, hq, d), dtype=torch.bfloat16, device=device)
    g_pad[:tq] = gate_logits.view(tq, hq, d)
    pos_pad = torch.full((tq_pad,), -1, dtype=torch.int64, device=device)
    pos_pad[:tq] = q_pos.to(torch.int64)
    k_pad = torch.zeros((tk_pad, hkv, d), dtype=torch.bfloat16, device=device)
    k_pad[:tk] = k
    v_pad = torch.zeros((tk_pad, hkv, d), dtype=torch.bfloat16, device=device)
    v_pad[:tk] = v

    y_out = torch.empty((tq_pad, hq, d), dtype=torch.bfloat16, device=device)
    o_out = torch.empty((tq_pad, hq, d), dtype=torch.bfloat16, device=device)
    lse_out = torch.empty((tq_pad, hq), dtype=torch.float32, device=device)
    batch = group * FALLBACK_TILE_GROUP
    pos_host = pos_pad.tolist()

    for kv_head in range(hkv):
        heads = slice(kv_head * group, (kv_head + 1) * group)
        kf_all = k_pad[:, kv_head].float()
        vf_all = v_pad[:, kv_head].float()
        for rg in range(n_row_groups):
            rows = slice(rg * rows_per_group, (rg + 1) * rows_per_group)
            qf = _tile_rows(q_pad[rows, heads].float(), group)  # (B, 64, D)
            pos = pos_pad[rows].view(FALLBACK_TILE_GROUP, BLOCK_M).expand(group, -1, -1)
            pos = pos.reshape(batch, BLOCK_M)
            valid = pos >= 0
            valid_pos = [p for p in pos_host[rows] if p >= 0]
            m_i = torch.full((batch, BLOCK_M), _NEG_INF, dtype=torch.float32, device=device)
            l_i = torch.zeros((batch, BLOCK_M), dtype=torch.float32, device=device)
            acc = torch.zeros((batch, BLOCK_M, d), dtype=torch.float32, device=device)
            if valid_pos:
                first, last = _block_range(min(valid_pos), max(valid_pos), tk, window)
                for blk in range(first, last + 1):
                    kf = kf_all[blk * BLOCK_N : (blk + 1) * BLOCK_N]
                    vf = vf_all[blk * BLOCK_N : (blk + 1) * BLOCK_N]
                    k_pos = torch.arange(blk * BLOCK_N, (blk + 1) * BLOCK_N, device=device)
                    scores = torch.matmul(qf, kf.t()) * sm_scale  # (B, 64, 64) fp32
                    keep = valid[:, :, None] & (k_pos < tk)[None, None, :]
                    keep = keep & (k_pos[None, None, :] <= pos[:, :, None])
                    if window is not None:
                        keep = keep & (k_pos[None, None, :] > pos[:, :, None] - window)
                    scores = torch.where(keep, scores, _NEG_INF)
                    m_blk = scores.amax(dim=-1)
                    m_new = torch.maximum(m_i, m_blk)
                    m_safe = torch.where(m_new == _NEG_INF, 0.0, m_new)
                    alpha = torch.exp(m_i - m_safe)
                    p = torch.exp(scores - m_safe[:, :, None])
                    l_i = l_i * alpha + p.sum(dim=-1)
                    acc = acc * alpha[:, :, None] + torch.matmul(p.to(torch.bfloat16).float(), vf)
                    m_i = m_new
            o_f = acc / l_i[:, :, None]
            o_bf = o_f.to(torch.bfloat16)
            g = _tile_rows(g_pad[rows, heads].float(), group)
            gate = 1.0 / (1.0 + torch.exp(-g))
            y_bf = (o_bf.float() * gate).to(torch.bfloat16)
            o_out[rows, heads] = _untile_rows(o_bf, group)
            y_out[rows, heads] = _untile_rows(y_bf, group)
            lse = m_i + torch.log(l_i)
            lse_out[rows, heads] = lse.view(group, rows_per_group).t()
    return (
        y_out[:tq].reshape(tq, hq * d).contiguous(),
        o_out[:tq].reshape(tq, hq * d).contiguous(),
        lse_out[:tq].contiguous(),
    )


def fallback_gated_attention_bwd_partials(
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
    """FALLBACK for ``strict_gated_attention_bwd_partials``.

    Returns ``(dq bf16 (Tq,Hq,128), dk_partials fp32 (Hq//4, Tk, Hkv, 128), dv_partials fp32,
    dgate bf16 (Tq, Hq*128))``. FA2 order per (query tile, KV block ascending): ``p = exp(s -
    lse)``; ``dV += (p->bf16)^T dO``; ``dP = dO V^T``; ``dS = p (dP - delta)``; ``dQ += (dS->bf16)
    K``; ``dK += (dS->bf16)^T Q``; ``dO = (dy * sigmoid(g)) -> bf16``; ``delta = rowsum(dO * o)``
    (fixed pairwise tree); ``dgate = dy * o * sigmoid(g) * (1 - sigmoid(g)) -> bf16``.
    dK/dV partial of 4-head group ``j`` for a key block is the left fold, in ascending query-tile
    order, of ``((h0 + h1) + h2) + h3`` over its heads; the softmax scale is applied once in
    fp32 at the end. Fixed GEMM shapes as in :func:`fallback_gated_attention_fwd`.
    """
    _disable_tf32()
    tq, hq, d = q.shape
    tk, hkv, _ = k.shape
    group = hq // hkv
    if hq % hkv or group != GQA_GROUP or d != HEAD_DIM or hq % HEADS_PER_CANONICAL_GROUP:
        raise ValueError(f"unsupported head geometry Hq={hq}, Hkv={hkv}, D={d}")
    _check_single_sequence(cu_seqlens_q, cu_seqlens_k, tq, tk)
    device = q.device
    sm_scale = float(d) ** -0.5
    groups_per_kv = group // HEADS_PER_CANONICAL_GROUP  # 4
    n_groups = hq // HEADS_PER_CANONICAL_GROUP  # 8 at TP1, 4 at TP2
    rows_per_group = FALLBACK_TILE_GROUP * BLOCK_M
    n_row_groups = -(-tq // rows_per_group)
    tq_pad = n_row_groups * rows_per_group
    tk_pad = -(-tk // BLOCK_N) * BLOCK_N
    batch = group * FALLBACK_TILE_GROUP

    # Row-local gate backward (fp32) and the attention output gradient.
    g_f = gate_logits.view(tq, hq, d).float()
    sig = 1.0 / (1.0 + torch.exp(-g_f))
    o_f = o.view(tq, hq, d).float()
    dy_f = dy.view(tq, hq, d).float()
    dgate = (dy_f * o_f * sig * (1.0 - sig)).to(torch.bfloat16).reshape(tq, hq * d)
    do_bf = (dy_f * sig).to(torch.bfloat16)
    delta = _pairwise_sum_last(do_bf.float() * o_f)  # (Tq, Hq) fp32

    def _pad_rows(x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        out = torch.zeros((tq_pad, *x.shape[1:]), dtype=dtype, device=device)
        out[:tq] = x
        return out

    q_pad = _pad_rows(q, torch.bfloat16)
    do_pad = _pad_rows(do_bf, torch.bfloat16)
    lse_pad = _pad_rows(lse, torch.float32)
    delta_pad = _pad_rows(delta, torch.float32)
    pos_pad = torch.full((tq_pad,), -1, dtype=torch.int64, device=device)
    pos_pad[:tq] = q_pos.to(torch.int64)
    k_pad = torch.zeros((tk_pad, hkv, d), dtype=torch.bfloat16, device=device)
    k_pad[:tk] = k
    v_pad = torch.zeros((tk_pad, hkv, d), dtype=torch.bfloat16, device=device)
    v_pad[:tk] = v
    pos_host = pos_pad.tolist()

    dq_out = torch.empty((tq_pad, hq, d), dtype=torch.bfloat16, device=device)
    dk_partials = torch.zeros((n_groups, tk_pad, hkv, d), dtype=torch.float32, device=device)
    dv_partials = torch.zeros((n_groups, tk_pad, hkv, d), dtype=torch.float32, device=device)

    for kv_head in range(hkv):
        heads = slice(kv_head * group, (kv_head + 1) * group)
        kf_all = k_pad[:, kv_head].float()
        vf_all = v_pad[:, kv_head].float()
        dk_acc = torch.zeros((groups_per_kv, tk_pad, d), dtype=torch.float32, device=device)
        dv_acc = torch.zeros((groups_per_kv, tk_pad, d), dtype=torch.float32, device=device)
        for rg in range(n_row_groups):
            rows = slice(rg * rows_per_group, (rg + 1) * rows_per_group)
            qf = _tile_rows(q_pad[rows, heads].float(), group)
            dof = _tile_rows(do_pad[rows, heads].float(), group)
            lse_t = lse_pad[rows, heads].t().reshape(batch, BLOCK_M)
            delta_t = delta_pad[rows, heads].t().reshape(batch, BLOCK_M)
            pos = pos_pad[rows].view(FALLBACK_TILE_GROUP, BLOCK_M).expand(group, -1, -1)
            pos = pos.reshape(batch, BLOCK_M)
            valid = pos >= 0
            lse_safe = torch.where(valid, lse_t, 0.0)
            valid_pos = [p for p in pos_host[rows] if p >= 0]
            dq_acc = torch.zeros((batch, BLOCK_M, d), dtype=torch.float32, device=device)
            if valid_pos:
                first, last = _block_range(min(valid_pos), max(valid_pos), tk, window)
                for blk in range(first, last + 1):
                    kslice = slice(blk * BLOCK_N, (blk + 1) * BLOCK_N)
                    kf = kf_all[kslice]
                    vf = vf_all[kslice]
                    k_pos = torch.arange(blk * BLOCK_N, (blk + 1) * BLOCK_N, device=device)
                    scores = torch.matmul(qf, kf.t()) * sm_scale
                    keep = valid[:, :, None] & (k_pos < tk)[None, None, :]
                    keep = keep & (k_pos[None, None, :] <= pos[:, :, None])
                    if window is not None:
                        keep = keep & (k_pos[None, None, :] > pos[:, :, None] - window)
                    p = torch.where(keep, torch.exp(scores - lse_safe[:, :, None]), 0.0)
                    pf = p.to(torch.bfloat16).float()
                    dv_c = torch.bmm(pf.transpose(1, 2), dof)  # (B, 64 keys, D)
                    dp = torch.matmul(dof, vf.t())  # (B, 64, 64)
                    ds = p * (dp - delta_t[:, :, None])
                    dsf = ds.to(torch.bfloat16).float()
                    dq_acc = dq_acc + torch.matmul(dsf, kf)
                    dk_c = torch.bmm(dsf.transpose(1, 2), qf)  # (B, 64 keys, D)
                    for contrib, target in ((dk_c, dk_acc), (dv_c, dv_acc)):
                        c = contrib.view(
                            groups_per_kv,
                            HEADS_PER_CANONICAL_GROUP,
                            FALLBACK_TILE_GROUP,
                            BLOCK_N,
                            d,
                        )
                        hsum = ((c[:, 0] + c[:, 1]) + c[:, 2]) + c[:, 3]  # (groups, tiles, 64, D)
                        for tile in range(FALLBACK_TILE_GROUP):
                            target[:, kslice] += hsum[:, tile]
            dq_out[rows, heads] = _untile_rows((dq_acc * sm_scale).to(torch.bfloat16), group)
        groups = slice(kv_head * groups_per_kv, (kv_head + 1) * groups_per_kv)
        dk_partials[groups, :, kv_head] = dk_acc * sm_scale
        dv_partials[groups, :, kv_head] = dv_acc
    return (
        dq_out[:tq].contiguous(),
        dk_partials[:, :tk].contiguous(),
        dv_partials[:, :tk].contiguous(),
        dgate,
    )


def gated_attention_rows(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    q_pos: torch.Tensor,
    *,
    window: Optional[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Strict kernel (sibling) or fallback on ONE sequence: ``(y, o, lse)`` for the given rows.

    ``q``/``gate``/``q_pos`` hold any subset of the sequence's query rows (ABSOLUTE positions);
    ``k``/``v`` hold the full key sequence. This is exactly the call a CP rank makes.
    """
    device = q.device
    cu_q = torch.tensor([0, q.size(0)], dtype=torch.int32, device=device)
    cu_k = torch.tensor([0, k.size(0)], dtype=torch.int32, device=device)
    fwd = _strict_fwd if _strict_fwd is not None else fallback_gated_attention_fwd
    return fwd(q, k, v, gate, cu_q, cu_k, q_pos.to(torch.int32), window=window)


def _kernel_backward_partials(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    o: torch.Tensor,
    lse: torch.Tensor,
    dy: torch.Tensor,
    q_pos: torch.Tensor,
    *,
    window: Optional[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    device = q.device
    cu_q = torch.tensor([0, q.size(0)], dtype=torch.int32, device=device)
    cu_k = torch.tensor([0, k.size(0)], dtype=torch.int32, device=device)
    bwd = _strict_bwd_partials
    if bwd is None:
        bwd = fallback_gated_attention_bwd_partials
    return bwd(q, k, v, gate, o, lse, dy, cu_q, cu_k, q_pos.to(torch.int32), window=window)


# --------------------------------------------------------------------------------------------
# CP forward / backward
# --------------------------------------------------------------------------------------------
def _validate_local_tensors(
    q_local: torch.Tensor,
    k_local: torch.Tensor,
    v_local: torch.Tensor,
    gate_local: torch.Tensor,
    positions_local: torch.Tensor,
    *,
    q_heads: int,
    kv_heads: int,
) -> None:
    if q_local.ndim != 3 or q_local.shape[1:] != (q_heads, HEAD_DIM):
        raise ValueError(
            f"q_local must be (T_local, {q_heads}, {HEAD_DIM}), got {tuple(q_local.shape)}"
        )
    for name, t in (("k_local", k_local), ("v_local", v_local)):
        if t.ndim != 3 or t.shape != (q_local.size(0), kv_heads, HEAD_DIM):
            raise ValueError(
                f"{name} must be (T_local, {kv_heads}, {HEAD_DIM}) with T_local={q_local.size(0)}, "
                f"got {tuple(t.shape)}"
            )
    if gate_local.shape != (q_local.size(0), q_heads * HEAD_DIM):
        raise ValueError(
            f"gate_local must be (T_local, {q_heads * HEAD_DIM}), got {tuple(gate_local.shape)}"
        )
    named = (("q_local", q_local), ("k_local", k_local), ("v_local", v_local))
    for name, t in (*named, ("gate_local", gate_local)):
        if t.dtype != torch.bfloat16:
            raise TypeError(f"{name} must be bf16, got {t.dtype}")
    if positions_local.shape != (q_local.size(0),):
        raise CPLayoutError(
            f"positions_local must be (T_local,)={q_local.size(0)}, "
            f"got {tuple(positions_local.shape)}"
        )


def _gather_kv_positions(
    k_local: torch.Tensor,
    v_local: torch.Tensor,
    positions_local: torch.Tensor,
    plan: CPAttentionPlan,
    collective: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """AG(K, V, positions) in rank order (pure copy) and check the allgather-layout invariant."""
    total = k_local.size(0) * plan.cp_world
    if plan.cp_world == 1:
        k_full, v_full, pos_full = k_local.contiguous(), v_local.contiguous(), positions_local
    else:
        if collective is None:
            raise ValueError("a deterministic collective is required for cp_world > 1")
        if int(getattr(collective, "world_size", plan.cp_world)) != plan.cp_world:
            raise ValueError(
                f"collective world_size {getattr(collective, 'world_size', None)} != "
                f"cp_world {plan.cp_world}"
            )
        k_full, v_full = _all_gather_many((k_local, v_local), collective)
        pos_full = all_gather_sequence(positions_local.to(torch.int32), collective)
    expected = torch.arange(total, dtype=pos_full.dtype, device=pos_full.device)
    if not torch.equal(pos_full, expected):
        raise CPLayoutError(
            "gathered key positions are not arange(T): the CP ranks do not hold contiguous "
            "rank-ordered chunks (allgather layout); zigzag/striped layouts fail closed"
        )
    return k_full, v_full, pos_full


def _cp_forward_core(
    q_local: torch.Tensor,
    k_local: torch.Tensor,
    v_local: torch.Tensor,
    gate_local: torch.Tensor,
    positions_local: torch.Tensor,
    plan: CPAttentionPlan,
    collective: Any,
    *,
    window: Optional[int],
    q_heads: int,
    kv_heads: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    bad_window = isinstance(window, bool) or not isinstance(window, int) or window <= 0
    if window is not None and bad_window:
        raise ValueError(f"window must be None (global layer) or a positive int, got {window!r}")
    _validate_local_tensors(
        q_local, k_local, v_local, gate_local, positions_local, q_heads=q_heads, kv_heads=kv_heads
    )
    total = q_local.size(0) * plan.cp_world
    plan.validate_local_positions(positions_local, total)
    k_full, v_full, _ = _gather_kv_positions(k_local, v_local, positions_local, plan, collective)
    y, o, lse = gated_attention_rows(
        q_local.contiguous(),
        k_full,
        v_full,
        gate_local.contiguous(),
        positions_local,
        window=window,
    )
    return y, o, lse, k_full, v_full


def cp_gated_attention_forward(
    q_local: torch.Tensor,
    k_local: torch.Tensor,
    v_local: torch.Tensor,
    gate_local: torch.Tensor,
    positions_local: torch.Tensor,
    plan: CPAttentionPlan,
    collective: Any,
    *,
    window: Optional[int],
    return_aux: bool = False,
):
    """CP forward of one strict gated-attention layer on the ``allgather`` layout.

    ``q_local (T/cp, 32, 128)``, ``k_local``/``v_local (T/cp, 2, 128)``, ``gate_local (T/cp,
    4096)`` bf16; ``positions_local (T/cp,)`` int32/int64 ABSOLUTE positions (must equal the
    rank's contiguous chunk); ``window=None`` for global (full-causal NoPE) layers, ``2048`` for
    local layers. Returns ``y_local (T/cp, 4096)`` bf16, or ``(y, o, lse)`` with ``return_aux``.
    Deterministic AG of K/V/positions (rank order, pure copy), then the strict kernel on the
    local queries against the full K/V with the gate applied in the kernel epilogue.
    """
    y, o, lse, _, _ = _cp_forward_core(
        q_local, k_local, v_local, gate_local, positions_local, plan, collective,
        window=window, q_heads=NUM_Q_HEADS, kv_heads=NUM_KV_HEADS,
    )
    return (y, o, lse) if return_aux else y


def cp_tp_gated_attention_forward(
    q_local: torch.Tensor,
    k_local: torch.Tensor,
    v_local: torch.Tensor,
    gate_local: torch.Tensor,
    positions_local: torch.Tensor,
    topology: TPCPTopology,
    cp_collective: Any,
    *,
    window: Optional[int],
    return_aux: bool = False,
):
    """TP x CP forward for ONE topology (TP2 x CP2 on 4 GPUs is the tested instance).

    TP rank ``t`` passes its Q-head shard ``q_local (T/cp, 32/tp, 128)``, its KV head
    ``k_local``/``v_local (T/cp, 2/tp, 128)`` and the gate channels of its heads
    ``gate_local (T/cp, 4096/tp)``; ``cp_collective`` is the deterministic collective of this
    rank's CP group (ranks sharing ``tp_rank``). Attention needs no TP communication: every TP
    rank owns the KV head its Q heads read (``tp <= 2``), so the only collective is the CP AG of
    this rank's K/V shard. Per-head arithmetic is unchanged by head sharding, so the result is
    bitwise the TP1 x CP1 result restricted to this rank's heads and rows.
    """
    y, o, lse, _, _ = _cp_forward_core(
        q_local, k_local, v_local, gate_local, positions_local, topology.plan, cp_collective,
        window=window, q_heads=topology.q_heads, kv_heads=topology.kv_heads,
    )
    return (y, o, lse) if return_aux else y


def cp_gated_attention_backward(
    q_local: torch.Tensor,
    k_full: torch.Tensor,
    v_full: torch.Tensor,
    gate_local: torch.Tensor,
    o_local: torch.Tensor,
    lse_local: torch.Tensor,
    dy_local: torch.Tensor,
    positions_local: torch.Tensor,
    plan: CPAttentionPlan,
    collective: Any,
    *,
    window: Optional[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """CP backward: ``(dq_local bf16, dk_local bf16, dv_local bf16, dgate_local bf16)``.

    Reduction order (see module docstring): per canonical query chunk owned by this rank (in
    ascending order) the kernel returns ``dq``/``dgate`` rows and fp32 dK/dV partials per 4-head
    group; inner tree over the head groups; local tree over the owned chunks; rank-ordered
    all_gather of the (T, Hkv, 128) fp32 rank roots; outer tree over the ranks evaluated
    locally; one rounding to bf16; this rank's key slice is returned. ``dq``/``dgate`` are the
    concatenation of the per-chunk kernel outputs (row-local, CP-invariant by O5).
    """
    total = k_full.size(0)
    t_local = q_local.size(0)
    if t_local * plan.cp_world != total:
        raise ValueError(f"q_local rows {t_local} x cp_world {plan.cp_world} != T {total}")
    if total % CP_CANONICAL_QUERY_CHUNKS:
        raise ValueError(
            f"T={total} must be a multiple of {CP_CANONICAL_QUERY_CHUNKS} canonical query chunks"
        )
    plan.validate_local_positions(positions_local, total)
    start, _ = plan.local_token_range(total)
    chunk_len = total // CP_CANONICAL_QUERY_CHUNKS
    if chunk_len < t_local and t_local % chunk_len:
        raise ValueError("local chunk must be a whole number of canonical query chunks")

    dq_parts: list[torch.Tensor] = []
    dgate_parts: list[torch.Tensor] = []
    dk_roots: list[torch.Tensor] = []
    dv_roots: list[torch.Tensor] = []
    for chunk in plan.canonical_chunk_ids():
        lo = chunk * chunk_len - start
        rows = slice(lo, lo + chunk_len)
        dq_c, dk_p, dv_p, dgate_c = _kernel_backward_partials(
            q_local[rows].contiguous(),
            k_full,
            v_full,
            gate_local[rows].contiguous(),
            o_local[rows].contiguous(),
            lse_local[rows].contiguous(),
            dy_local[rows].contiguous(),
            positions_local[rows],
            window=window,
        )
        dq_parts.append(dq_c)
        dgate_parts.append(dgate_c)
        dk_roots.append(combine_tree(list(dk_p.unbind(0))))  # inner tree: head groups
        dv_roots.append(combine_tree(list(dv_p.unbind(0))))
    dk_rank = combine_tree(dk_roots)  # local part of the outer tree (owned chunks)
    dv_rank = combine_tree(dv_roots)
    if plan.cp_world > 1:
        dk_all, dv_all = _all_gather_many((dk_rank[None], dv_rank[None]), collective)
        dk_full = combine_tree(list(dk_all.unbind(0)))  # upper levels of the outer tree
        dv_full = combine_tree(list(dv_all.unbind(0)))
    else:
        dk_full, dv_full = dk_rank, dv_rank
    end = start + t_local
    return (
        torch.cat(dq_parts, dim=0),
        dk_full[start:end].to(torch.bfloat16).contiguous(),
        dv_full[start:end].to(torch.bfloat16).contiguous(),
        torch.cat(dgate_parts, dim=0),
    )


class CPGatedAttentionFn(torch.autograd.Function):
    """Autograd binding of the CP forward/backward above (``y_local`` out, CP grads in)."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        q_local: torch.Tensor,
        k_local: torch.Tensor,
        v_local: torch.Tensor,
        gate_local: torch.Tensor,
        positions_local: torch.Tensor,
        plan: CPAttentionPlan,
        collective: Any,
        window: Optional[int],
    ) -> torch.Tensor:
        y, o, lse, k_full, v_full = _cp_forward_core(
            q_local, k_local, v_local, gate_local, positions_local, plan, collective,
            window=window, q_heads=q_local.size(1), kv_heads=k_local.size(1),
        )
        ctx.save_for_backward(q_local, k_full, v_full, gate_local, o, lse, positions_local)
        ctx.plan = plan
        ctx.collective = collective
        ctx.window = window
        return y

    @staticmethod
    def backward(ctx, dy: torch.Tensor):  # type: ignore[override]
        q_local, k_full, v_full, gate_local, o, lse, positions_local = ctx.saved_tensors
        dq, dk, dv, dgate = cp_gated_attention_backward(
            q_local, k_full, v_full, gate_local, o, lse, dy.contiguous(), positions_local,
            ctx.plan, ctx.collective, window=ctx.window,
        )
        return dq, dk, dv, dgate, None, None, None, None


def cp_gated_attention(
    q_local: torch.Tensor,
    k_local: torch.Tensor,
    v_local: torch.Tensor,
    gate_local: torch.Tensor,
    positions_local: torch.Tensor,
    plan: CPAttentionPlan,
    collective: Any,
    *,
    window: Optional[int],
) -> torch.Tensor:
    """``y_local`` with the CP-aware deterministic backward attached."""
    return CPGatedAttentionFn.apply(
        q_local, k_local, v_local, gate_local, positions_local, plan, collective, window
    )


def trace(plan: Optional[CPAttentionPlan] = None) -> dict[str, Any]:
    """Provenance record (contract R4) of the CP attention path."""
    record: dict[str, Any] = {
        "op": "muse_glimmer.cp_gated_attention",
        "rows": ["cp_global_attention", "cp_local_attention"],
        "dtype_in": "bf16",
        "dtype_acc": "fp32",
        "dtype_out": "bf16",
        "cp_layout": SUPPORTED_LAYOUTS[0],
        "cp_rejected_layouts": list(REJECTED_LAYOUTS),
        "cp_forward": (
            "ag_kv_positions_rank_order -> strict_core(local_q, full_kv, abs_pos); "
            "no (out,lse) merge"
        ),
        "cp_gate": "kernel epilogue on owner rows (O5): y = (o->bf16->fp32) * sigmoid(g) -> bf16",
        "cp_backward_dkdv": (
            "inner tree over 8 Q-head groups ((0+1)+(2+3))+((4+5)+(6+7)); outer tree over "
            f"{CP_CANONICAL_QUERY_CHUNKS} canonical query chunks (same tree); rank roots exchanged "
            "by rank-ordered all_gather and combined locally; one bf16 rounding"
        ),
        "cp_backward_dq_dgate": "row-local on the owner rank (O5)",
        "reduction_order": (
            "o5_kv_blocks_64_ascending_from_abs_0; canonical_two_level_tree_for_dkdv"
        ),
        "tile_config": {"BLOCK_N": BLOCK_N, "BLOCK_M": BLOCK_M},
        "forward_impl": FORWARD_IMPL,
        "forward_is_fallback": _strict_fwd is None,
        "backward_impl": BACKWARD_IMPL,
        "backward_is_fallback": _strict_bwd_partials is None,
        "tree_impl": TREE_IMPL,
        "fallback_fixed_gemm_rows": GQA_GROUP * FALLBACK_TILE_GROUP * BLOCK_M,
    }
    if plan is not None:
        record.update(plan.provenance())
    return record
