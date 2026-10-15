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
``dK``/``dV`` follow the kernel's query-chunk rule (``triton/gated_attention_bwd.py``): leaves
``(query chunk c, Q-head group g)`` with ``chunk c = [floor(c*T/8), floor((c+1)*T/8))``, folded
per group over the 8 chunks in the O3 tree, then over the groups in the same tree. CP rank ``i``
of ``c`` owns chunks ``[8i/c, 8(i+1)/c)`` (rank boundaries are chunk boundaries whenever
``T % c == 0``, no ``T % 8`` requirement). One kernel call on the rank's rows against the full
K/V yields, per group, the sub-tree root of its chunks (the other chunks are exact zero leaves);
the rank roots travel by a deterministic rank-ordered all_gather and are folded in the same tree;
then the group tree; one bf16 rounding. The expression is identical at CP1/2/4/8 and to the
one-shot kernel (``strict_gated_attention_bwd``), so all of them agree bit for bit. No NCCL
reduce-scatter is ever used.

Cast points: K/V/positions move as bits; the fp32 tree root of dK/dV is rounded to bf16 once;
``y``, ``o``, ``lse``, ``dq`` and ``dgate`` carry the kernel's own casts (O5).

Kernels (no fallback: strict ops fail closed): forward
``ops.gated_attention.strict_gated_attention_fwd_varlen`` (head counts as parameters, so TP head
shards use the same wrapper), backward ``triton.gated_attention_bwd``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import torch

from rl_engine.models.muse_glimmer.ops.canonical_tree import combine_tree
from rl_engine.models.muse_glimmer.ops.gated_attention import strict_gated_attention_fwd_varlen
from rl_engine.models.muse_glimmer.triton import gated_attention_bwd as _bwd

__all__ = [
    "BACKWARD_IMPL",
    "CP_CANONICAL_QUERY_CHUNKS",
    "CPAttentionPlan",
    "CPGatedAttentionFn",
    "CPLayoutError",
    "FORWARD_IMPL",
    "SUPPORTED_CP_WORLDS",
    "SUPPORTED_LAYOUTS",
    "TPCPTopology",
    "all_gather_sequence",
    "combine_tree",
    "cp_gated_attention",
    "cp_gated_attention_backward",
    "cp_gated_attention_forward",
    "cp_tp_gated_attention_forward",
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
HEADS_PER_CANONICAL_GROUP = _bwd.HEADS_PER_GROUP  # O5: dK/dV partials per 4-head group
SLIDING_WINDOW = 2048

SUPPORTED_CP_WORLDS = (1, 2, 4, 8)
SUPPORTED_LAYOUTS = ("allgather",)
CP_CANONICAL_QUERY_CHUNKS = _bwd.NUM_QUERY_CHUNKS
REJECTED_LAYOUTS = ("zigzag", "striped", "ring", "interleaved")

BLOCK_N = _bwd.BLOCK_N
BLOCK_M = _bwd.BLOCK_M

FORWARD_IMPL = "rl_engine.models.muse_glimmer.ops.gated_attention.strict_gated_attention_fwd_varlen"
BACKWARD_IMPL = (
    "rl_engine.models.muse_glimmer.triton.gated_attention_bwd."
    "gated_attention_bwd_preprocess + strict_gated_attention_bwd_partials (query-chunk rule)"
)
TREE_IMPL = "rl_engine.models.muse_glimmer.ops.canonical_tree.combine_tree"


class CPLayoutError(ValueError):
    """Raised when a CP layout other than the contiguous ``allgather`` layout is requested."""


# --------------------------------------------------------------------------------------------
# Plan / topology
# --------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class CPAttentionPlan:
    """Contiguous-chunk (``allgather``) context-parallel plan for one logical sequence.

    ``cp_rank`` owns tokens ``[cp_rank*T/cp_world, (cp_rank+1)*T/cp_world)``; ``T`` must be a
    multiple of ``cp_world`` (then the rank boundaries are canonical query-chunk boundaries).
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
                "its own KV head; TP4/TP8 replicate KV heads (kv_head_replication) and need the "
                "cross-rank dK/dV tree, out of scope for this POC topology; "
                f"got {self.tp_world}"
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
    (bitwise data movement, rank order). Only ``all_gather`` / ``all_gather_many`` are used here;
    every reduction is the canonical tree evaluated locally.
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
# Kernel calls on one logical sequence
# --------------------------------------------------------------------------------------------
def _single_sequence_offsets(tq: int, tk: int, device: torch.device):
    cu_q = torch.tensor([0, tq], dtype=torch.int32, device=device)
    cu_k = torch.tensor([0, tk], dtype=torch.int32, device=device)
    return cu_q, cu_k


def gated_attention_rows(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    q_pos: torch.Tensor,
    *,
    window: Optional[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Strict kernel on ONE sequence: ``(y, o, lse)`` for the given rows.

    ``q``/``gate``/``q_pos`` hold any subset of the sequence's query rows (ABSOLUTE positions);
    ``k``/``v`` hold the full key sequence. This is exactly the call a CP rank makes. Head counts
    come from the tensors (full 32/2 or a TP shard such as 16/1) and are validated by the wrapper.
    """
    cu_q, cu_k = _single_sequence_offsets(q.size(0), k.size(0), q.device)
    return strict_gated_attention_fwd_varlen(
        q, k, v, gate, cu_q, cu_k, q_pos.to(torch.int32), window=window,
        num_q_heads=q.size(1), num_kv_heads=k.size(1),
    )


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
    this rank's K/V shard. Per-head arithmetic is unchanged by head sharding (the strict wrapper
    takes the head counts), so the result is bitwise the TP1 x CP1 result restricted to this
    rank's heads and rows.
    """
    y, o, lse, _, _ = _cp_forward_core(
        q_local, k_local, v_local, gate_local, positions_local, topology.plan, cp_collective,
        window=window, q_heads=topology.q_heads, kv_heads=topology.kv_heads,
    )
    return (y, o, lse) if return_aux else y


def _compact_groups(partials: torch.Tensor) -> torch.Tensor:
    """``(G, T, H_KV, D)`` group partials -> ``(G, T, D)``: each group's own KV head slice."""
    n_groups, _, hkv, _ = partials.shape
    kv_of_group = torch.arange(n_groups, device=partials.device) // (n_groups // hkv)
    return partials[torch.arange(n_groups, device=partials.device), :, kv_of_group]


def _expand_groups(compact: torch.Tensor, hkv: int) -> torch.Tensor:
    n_groups, total, d = compact.shape
    out = torch.zeros((n_groups, total, hkv, d), dtype=compact.dtype, device=compact.device)
    kv_of_group = torch.arange(n_groups, device=compact.device) // (n_groups // hkv)
    out[torch.arange(n_groups, device=compact.device), :, kv_of_group] = compact
    return out


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

    Reduction order (module docstring): ONE kernel call on this rank's rows against the full K/V
    gives ``dq``/``dgate`` rows and fp32 dK/dV group partials whose chunk axis is already folded
    (the chunks of other ranks are exact zero leaves, so each group partial is this rank's
    sub-tree root); rank-ordered all_gather of the ``(G, T, 128)`` roots; per group the canonical
    tree over the ranks; then the canonical tree over the groups; one rounding to bf16; this
    rank's key slice is returned. Head counts come from the tensors (32/2, or a TP2 shard 16/1).
    """
    total = k_full.size(0)
    t_local = q_local.size(0)
    if t_local * plan.cp_world != total:
        raise ValueError(f"q_local rows {t_local} x cp_world {plan.cp_world} != T {total}")
    plan.validate_local_positions(positions_local, total)
    hq, hkv = q_local.size(1), k_full.size(1)
    if hq // hkv != GQA_GROUP:
        raise ValueError(
            f"cp_gated_attention_backward needs whole KV-head groups per call, got {hq}/{hkv}"
        )
    start, end = plan.local_token_range(total)
    cu_q, cu_k = _single_sequence_offsets(t_local, total, q_local.device)
    dgate, do, delta = _bwd.gated_attention_bwd_preprocess(
        dy_local.contiguous(), o_local.contiguous(), gate_local.contiguous(), num_q_heads=hq
    )
    dq, dk_part, dv_part = _bwd.strict_gated_attention_bwd_partials(
        q_local.contiguous(), k_full, v_full, do, lse_local.contiguous(), delta, cu_q, cu_k,
        positions_local.to(torch.int32), window=window, num_q_heads=hq, num_kv_heads=hkv,
    )
    if plan.cp_world > 1:
        dk_all, dv_all = _all_gather_many(
            (_compact_groups(dk_part)[None], _compact_groups(dv_part)[None]), collective
        )
        dk_part = _expand_groups(combine_tree(list(dk_all.unbind(0))), hkv)  # tree over ranks
        dv_part = _expand_groups(combine_tree(list(dv_all.unbind(0))), hkv)
    dk_full = _bwd.combine_kv_partials(dk_part)  # tree over the Q-head groups -> bf16
    dv_full = _bwd.combine_kv_partials(dv_part)
    return (
        dq,
        dk_full[start:end].contiguous(),
        dv_full[start:end].contiguous(),
        dgate,
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
            f"kernel query-chunk rule: leaves (chunk c of {CP_CANONICAL_QUERY_CHUNKS} = "
            "[floor(cT/8), floor((c+1)T/8)), Q-head group g); one kernel call per rank folds its "
            "own chunks per group (other chunks exact zero leaves); rank roots (G,T,128) by "
            "rank-ordered all_gather, canonical tree over ranks per group, canonical tree over "
            "groups, one bf16 rounding == one-shot strict_gated_attention_bwd"
        ),
        "cp_backward_dq_dgate": "row-local on the owner rank (O5)",
        "reduction_order": (
            "o5_kv_blocks_64_ascending_from_abs_0; dkdv_query_chunk_tree_then_group_tree"
        ),
        "tile_config": {"BLOCK_N": BLOCK_N, "BLOCK_M": BLOCK_M},
        "forward_impl": FORWARD_IMPL,
        "forward_is_fallback": False,
        "backward_impl": BACKWARD_IMPL,
        "backward_is_fallback": False,
        "backward_kernel_id": _bwd.KERNEL_ID,
        "tree_impl": TREE_IMPL,
        "tp_head_shards": "same wrapper with num_q_heads/num_kv_heads (16/1 at TP2)",
    }
    if plan is not None:
        record.update(plan.provenance())
    return record
