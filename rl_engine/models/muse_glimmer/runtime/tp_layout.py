# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Declared tensor-parallel ownership for the Muse-Glimmer-30B strict POC (RFC #501, WS2).

One object answers, for a physical TP degree ``tp`` in {1, 2, 4, 8} and a rank ``r``:

* which Q heads the rank owns (``32 / tp`` consecutive heads),
* which KV head(s) it holds (``q_head // 16``; at TP2 exactly one KV head per rank, at TP4 /
  TP8 each KV head is replicated 2x / 4x, at TP1 both heads are local),
* which canonical K-chunks of the row-parallel ``o_proj`` (512 wide) and ``down_proj``
  (2496 wide) it owns (``8 / tp`` consecutive chunks),
* which canonical vocabulary tiles (25344 rows each) of the padded LM head it owns.

Every number is derived from ``OFFICIAL_PARALLEL_LAYOUT`` / ``OFFICIAL_FINGERPRINT`` in
``workload.py``; a layout that disagrees with the pinned manifest fails closed. The canonical
degree is 8 so TP1 / TP2 / TP4 / TP8 are nested prefixes of one 8-leaf balanced tree
``((0+1)+(2+3))+((4+5)+(6+7))`` over chunks (row-parallel sums) and over tiles (vocab partials).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from rl_engine.models.muse_glimmer.workload import OFFICIAL_FINGERPRINT, OFFICIAL_PARALLEL_LAYOUT

SUPPORTED_TP = (1, 2, 4, 8)
CANONICAL_TREE_ORDER = "((0+1)+(2+3))+((4+5)+(6+7))"


class TPLayoutError(ValueError):
    """Raised when a requested TP layout is not the declared Muse-Glimmer layout."""


@dataclass(frozen=True)
class TPLayout:
    """Rank ownership at physical TP ``tp`` under canonical TP ``canonical_tp``.

    All fields default to the pinned Muse-Glimmer-30B values; tests may shrink the head
    counts / chunk widths only through the explicit keyword arguments, and such a layout is
    flagged ``official=False`` in :meth:`provenance`.
    """

    tp: int
    canonical_tp: int = int(OFFICIAL_PARALLEL_LAYOUT["canonical_tp"])
    num_q_heads: int = int(OFFICIAL_FINGERPRINT["num_attention_heads"])
    num_kv_heads: int = int(OFFICIAL_FINGERPRINT["num_key_value_heads"])
    head_dim: int = int(OFFICIAL_FINGERPRINT["head_dim"])
    o_proj_chunk: int = int(OFFICIAL_PARALLEL_LAYOUT["row_parallel_shards"]["o_proj"])
    down_chunk: int = int(OFFICIAL_PARALLEL_LAYOUT["row_parallel_shards"]["down_proj"])
    vocab_size: int = int(OFFICIAL_FINGERPRINT["vocab_size"])
    padded_vocab_size: int = int(OFFICIAL_PARALLEL_LAYOUT["padded_vocab_size"])

    def __post_init__(self) -> None:
        if self.tp not in SUPPORTED_TP:
            raise TPLayoutError(f"tp must be one of {SUPPORTED_TP}, got {self.tp}")
        if self.canonical_tp not in SUPPORTED_TP or self.canonical_tp % self.tp:
            raise TPLayoutError(
                f"canonical_tp {self.canonical_tp} must be in {SUPPORTED_TP} and a multiple "
                f"of tp={self.tp}"
            )
        if self.num_q_heads % self.canonical_tp:
            raise TPLayoutError("num_q_heads must divide into canonical_tp head groups")
        if self.num_q_heads % self.num_kv_heads:
            raise TPLayoutError("num_q_heads must be a multiple of num_kv_heads")
        if self.tp > self.num_kv_heads and self.tp % self.num_kv_heads:
            raise TPLayoutError("tp must be a multiple of num_kv_heads when replicating KV")
        if self.padded_vocab_size % self.canonical_tp:
            raise TPLayoutError("padded_vocab_size must split into canonical_tp equal tiles")
        if self.padded_vocab_size < self.vocab_size:
            raise TPLayoutError("padded_vocab_size must not be smaller than vocab_size")
        if self.o_proj_chunk * self.canonical_tp != self.num_q_heads * self.head_dim:
            raise TPLayoutError(
                "o_proj_chunk * canonical_tp must equal the Q/O width num_q_heads * head_dim"
            )

    # ------------------------------------------------------------------ #
    # Derived constants
    # ------------------------------------------------------------------ #

    @property
    def chunks_per_rank(self) -> int:
        """Canonical K-chunks (and vocab tiles) each physical rank owns: ``8 / tp``."""
        return self.canonical_tp // self.tp

    @property
    def q_heads_per_rank(self) -> int:
        return self.num_q_heads // self.tp

    @property
    def gqa_group(self) -> int:
        """Q heads per KV head (16): q head ``h`` reads kv head ``h // gqa_group``."""
        return self.num_q_heads // self.num_kv_heads

    @property
    def kv_replication(self) -> int:
        """How many ranks hold a copy of each KV head (1 at TP1/TP2, 2 at TP4, 4 at TP8)."""
        return max(1, self.tp // self.num_kv_heads)

    @property
    def vocab_tile_rows(self) -> int:
        """Rows per canonical vocab tile (25344 = 202752 / 8)."""
        return self.padded_vocab_size // self.canonical_tp

    @property
    def o_proj_k(self) -> int:
        return self.o_proj_chunk * self.canonical_tp

    @property
    def down_k(self) -> int:
        return self.down_chunk * self.canonical_tp

    # ------------------------------------------------------------------ #
    # Per-rank ownership
    # ------------------------------------------------------------------ #

    def _check_rank(self, rank: int) -> int:
        rank = int(rank)
        if not 0 <= rank < self.tp:
            raise TPLayoutError(f"rank {rank} out of range for tp={self.tp}")
        return rank

    def q_heads(self, rank: int) -> range:
        """Owned Q heads, consecutive: ``[rank * 32/tp, (rank + 1) * 32/tp)``."""
        rank = self._check_rank(rank)
        return range(rank * self.q_heads_per_rank, (rank + 1) * self.q_heads_per_rank)

    def q_columns(self, rank: int) -> tuple[int, int]:
        """Column range of the owned heads in the flattened 4096-wide Q/O/gate width."""
        heads = self.q_heads(rank)
        return heads.start * self.head_dim, heads.stop * self.head_dim

    def kv_heads(self, rank: int) -> tuple[int, ...]:
        """KV heads the rank holds: ``{h // 16 for h in q_heads(rank)}`` in ascending order."""
        heads = self.q_heads(rank)
        return tuple(sorted({h // self.gqa_group for h in heads}))

    def kv_head(self, rank: int) -> int:
        """The single replicated KV head at TP >= 2 (``floor(2 * rank / tp)``)."""
        heads = self.kv_heads(rank)
        if len(heads) != 1:
            raise TPLayoutError(
                f"rank {rank} at tp={self.tp} holds {len(heads)} KV heads; use kv_heads()"
            )
        return heads[0]

    def canonical_chunks(self, rank: int) -> range:
        """Owned canonical K-chunk ids (also the owned vocab tile ids): ``8/tp`` consecutive."""
        rank = self._check_rank(rank)
        return range(rank * self.chunks_per_rank, (rank + 1) * self.chunks_per_rank)

    def chunk_owner(self, chunk: int) -> int:
        """Physical rank that owns canonical chunk / tile ``chunk``."""
        if not 0 <= int(chunk) < self.canonical_tp:
            raise TPLayoutError(f"chunk {chunk} out of range for canonical_tp={self.canonical_tp}")
        return int(chunk) // self.chunks_per_rank

    def k_range(self, rank: int, chunk_width: int) -> tuple[int, int]:
        chunks = self.canonical_chunks(rank)
        return chunks.start * chunk_width, chunks.stop * chunk_width

    def o_proj_k_range(self, rank: int) -> tuple[int, int]:
        """K columns of ``o_proj`` (and the 4096-wide attention output) owned by the rank."""
        return self.k_range(rank, self.o_proj_chunk)

    def down_k_range(self, rank: int) -> tuple[int, int]:
        """K columns of ``down_proj`` (and the 19968-wide SwiGLU output) owned by the rank."""
        return self.k_range(rank, self.down_chunk)

    def vocab_tiles(self, rank: int) -> range:
        return self.canonical_chunks(rank)

    def vocab_range(self, rank: int) -> tuple[int, int]:
        """Padded vocabulary rows ``[start, stop)`` owned by the rank."""
        return self.k_range(rank, self.vocab_tile_rows)

    def row_parallel_chunk_widths(self) -> tuple[int, ...]:
        return (self.o_proj_chunk, self.down_chunk)

    # ------------------------------------------------------------------ #
    # Provenance
    # ------------------------------------------------------------------ #

    def kv_ownership_map(self) -> dict[str, Any]:
        """The declared replicated-KV ownership map (RFC #501 ``kv_head_replication``)."""
        return {
            "tp": self.tp,
            "canonical_tp": self.canonical_tp,
            "num_q_heads": self.num_q_heads,
            "num_kv_heads": self.num_kv_heads,
            "gqa_group": self.gqa_group,
            "kv_replication": self.kv_replication,
            "rule": "q head h reads kv head h // 16; rank r owns q heads "
            "[32r/tp, 32(r+1)/tp) and holds kv head floor(2r/tp) (both heads at TP1)",
            "ranks": [
                {
                    "rank": r,
                    "q_heads": [self.q_heads(r).start, self.q_heads(r).stop],
                    "kv_heads": list(self.kv_heads(r)),
                    "canonical_chunks": [
                        self.canonical_chunks(r).start,
                        self.canonical_chunks(r).stop,
                    ],
                    "o_proj_k": list(self.o_proj_k_range(r)),
                    "down_k": list(self.down_k_range(r)),
                    "vocab_tiles": [self.vocab_tiles(r).start, self.vocab_tiles(r).stop],
                    "vocab_rows": list(self.vocab_range(r)),
                }
                for r in range(self.tp)
            ],
        }

    def provenance(self) -> dict[str, Any]:
        """Trace-style dictionary recorded next to every WS2 result."""
        return {
            "op": "tp_layout",
            "official": self.is_official(),
            "tp": self.tp,
            "canonical_tp": self.canonical_tp,
            "chunks_per_rank": self.chunks_per_rank,
            "reduction_order": CANONICAL_TREE_ORDER,
            "row_parallel_shards": {"o_proj": self.o_proj_chunk, "down_proj": self.down_chunk},
            "vocab": {
                "vocab_size": self.vocab_size,
                "padded_vocab_size": self.padded_vocab_size,
                "tile_rows": self.vocab_tile_rows,
                "padded_logits": "excluded_from_logsumexp",
                "softcap_before_vocab_partials": True,
            },
            "kv_heads_per_canonical_rank": "replicated",
            "kv_ownership": self.kv_ownership_map(),
        }

    def is_official(self) -> bool:
        return (
            self.canonical_tp == int(OFFICIAL_PARALLEL_LAYOUT["canonical_tp"])
            and self.num_q_heads == int(OFFICIAL_FINGERPRINT["num_attention_heads"])
            and self.num_kv_heads == int(OFFICIAL_FINGERPRINT["num_key_value_heads"])
            and self.head_dim == int(OFFICIAL_FINGERPRINT["head_dim"])
            and self.o_proj_chunk == int(OFFICIAL_PARALLEL_LAYOUT["row_parallel_shards"]["o_proj"])
            and self.down_chunk
            == int(OFFICIAL_PARALLEL_LAYOUT["row_parallel_shards"]["down_proj"])
            and self.vocab_size == int(OFFICIAL_FINGERPRINT["vocab_size"])
            and self.padded_vocab_size == int(OFFICIAL_PARALLEL_LAYOUT["padded_vocab_size"])
        )

    def fail_closed_if_undeclared(
        self,
        declared: Mapping[str, Any] | None = None,
        *,
        world_size: int | None = None,
        require_official: bool = True,
    ) -> None:
        """Raise unless this layout is exactly the declared (manifest) layout.

        ``declared`` defaults to ``OFFICIAL_PARALLEL_LAYOUT``; ``world_size`` (if given) must
        equal ``tp``. Both runtimes call this before any strict TP collective so a silently
        different ownership map (for example Megatron's QKV all-gather instead of KV
        replication, or a vocab padding other than 202752) aborts instead of drifting.
        """

        layout = OFFICIAL_PARALLEL_LAYOUT if declared is None else declared
        problems: list[str] = []
        if world_size is not None and int(world_size) != self.tp:
            problems.append(f"process-group world_size {world_size} != layout tp {self.tp}")
        if layout.get("kv_heads_per_canonical_rank") != "replicated":
            problems.append(
                "declared kv_heads_per_canonical_rank is "
                f"{layout.get('kv_heads_per_canonical_rank')!r}, not 'replicated'"
            )
        if int(layout.get("canonical_tp", -1)) != self.canonical_tp:
            problems.append(
                f"declared canonical_tp {layout.get('canonical_tp')} != {self.canonical_tp}"
            )
        shards = layout.get("row_parallel_shards", {})
        if int(shards.get("o_proj", -1)) != self.o_proj_chunk:
            problems.append(f"declared o_proj shard {shards.get('o_proj')} != {self.o_proj_chunk}")
        if int(shards.get("down_proj", -1)) != self.down_chunk:
            problems.append(
                f"declared down_proj shard {shards.get('down_proj')} != {self.down_chunk}"
            )
        if int(layout.get("padded_vocab_size", -1)) != self.padded_vocab_size:
            problems.append(
                f"declared padded_vocab_size {layout.get('padded_vocab_size')} != "
                f"{self.padded_vocab_size}"
            )
        if layout.get("padded_logits") != "excluded_from_logsumexp":
            problems.append("declared layout does not exclude padded logits from the logsumexp")
        if layout.get("softcap_before_vocab_partials") is not True:
            problems.append("declared layout does not apply the soft cap before vocab partials")
        if require_official and not self.is_official():
            problems.append("layout head counts / chunk widths differ from the official pin")
        if problems:
            raise TPLayoutError("undeclared TP layout: " + "; ".join(problems))


def official_layout(tp: int) -> TPLayout:
    """The pinned Muse-Glimmer-30B layout at physical TP ``tp``, validated against the manifest."""
    layout = TPLayout(tp=tp)
    layout.fail_closed_if_undeclared()
    return layout


__all__ = [
    "CANONICAL_TREE_ORDER",
    "SUPPORTED_TP",
    "TPLayout",
    "TPLayoutError",
    "official_layout",
]
