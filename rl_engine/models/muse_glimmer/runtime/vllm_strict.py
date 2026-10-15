# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Bind the Muse-Glimmer-30B strict operator set into vLLM 0.31's native model (RFC #501 row
``vllm_runtime_adapter``).

What is strict and what stays native
------------------------------------
Strict (contract ``docs/contracts/muse-glimmer-30b-strict-contract.md``, "Model assembly"):
embedding norm (O1), the four sandwich norms + final norm (O2), q/k/v/gate as FOUR det GEMMs on
canonical slices of vLLM's fused ``qkv_proj`` + ``output_gate_proj`` weights (O3), qk-norm + scale
+ RoPE (O4), paged gated attention with the gate inside the op (O5), o_proj / down_proj as
row-parallel fp32 sub-tree partials + rank-ordered all-gather + canonical tree (O3), SwiGLU (O6)
and the selected-token log-probability through the vocab-tile API (O7).  Every selected-token
logprob vLLM returns (sampled tokens via the V2 ``Sampler.__call__``; prompt tokens via
``compute_prompt_logprobs_with_chunking``) is overwritten with the strict value.

Native (out of the strict contract, recorded in provenance): token SELECTION.  vLLM keeps computing
its own bf16 soft-capped logits (``MuseGlimmerForCausalLM.compute_logits``) and samples / argmaxes
from them; ``selected_token_ranks`` and any top-k logprob columns beyond the selected token also come
from those native logits.  KV-cache writes stay native (``reshape_and_cache_flash`` is a copy).

Fused-weight slice mapping (per TP rank ``r`` of ``t``; H = 6656, D = 128)
----------------------------------------------------------------------
``qkv_proj.weight`` rows: ``[0, q_size)`` = q heads ``[32r/t, 32(r+1)/t)``; ``[q_size, q_size +
kv_size)`` = k head(s) of this rank; ``[q_size + kv_size, q_size + 2 kv_size)`` = v head(s)
(``q_size = 4096/t``, ``kv_size = 128 * max(1, 2/t)``; at TP > 2 the kv head is replicated).
``output_gate_proj.weight`` rows ``[0, q_size)`` = gate channels of the same q heads.
``gate_up_proj.weight`` rows ``[0, F/t)`` = mlp gate, ``[F/t, 2F/t)`` = mlp up (F = 19968).
``o_proj.weight`` ``(H, 4096/t)`` = canonical K chunks ``rank_chunk_ids(t, r)`` (chunk = 512);
``down_proj.weight`` ``(H, F/t)`` = chunks ``rank_chunk_ids(t, r)`` (chunk = 2496).
``lm_head.weight`` is re-padded to 202752 rows (``ParallelLMHead`` patch) so the shard of rank ``r``
is exactly vocab tiles ``[8r/t, 8(r+1)/t)`` of 25344 rows.

LoRA (strict unmerged slots)
----------------------------
``load_lora`` stores per-layer ``(A, B)`` for q/k/v/gate/o and mlp gate/up/down.  The declared
unmerged form is the one of the Megatron binding (``runtime/megatron_lora.py``), so trainer and
rollout compute the same bits::

    z = bf16( acc32(A x) )                         # canonical 8-chunk det GEMM, one rounding
    y = bf16( acc32(W x) + sum_{k<r} B[:, k] z[k] )  # base fp32 accumulator (NOT rounded) +
                                                   # fixed-order fp32 loop over the rank
                                                   # (``strict_kernels.small_k_fp32``), one rounding

Column-parallel slots keep ``A`` replicated and shard ``B`` rows like the base weight; row-parallel
slots shard ``A`` columns like the base weight (K chunks + tree: ``acc32`` is the cross-rank tree of
the fp32 sub-tree partials) and keep ``B`` replicated.  ``B = 0`` is a bitwise no-op
(``acc + 0.0 == acc``; the one rounding is the same as without LoRA).

Fail-closed checks: ``check_runtime_binding`` (native class, four-GEMM grouping, speculative off),
config pins, ``enforce_eager`` / no torch.compile, bf16 KV cache, canonical lm_head tiling.
"""

from __future__ import annotations

import atexit
import json
import os
import re
import threading
import time
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from rl_engine.models.muse_glimmer.loader import check_runtime_binding
from rl_engine.models.muse_glimmer.runtime import fallback_ops as fb
from rl_engine.models.muse_glimmer.runtime import vllm_attention_backend as attn_backend
from rl_engine.models.muse_glimmer.runtime.strict_ops import StrictOps, probe_ops, resolve_ops
from rl_engine.models.muse_glimmer.workload import (
    FULL_ATTENTION,
    OFFICIAL_FINGERPRINT,
    OFFICIAL_PARALLEL_LAYOUT,
    OFFICIAL_PROJECTION_GROUPING,
    MuseGlimmerWorkloadError,
    expected_layer_types,
)

STRICT_ENV = "RL_KERNEL_MUSEGLIMMER_STRICT"
READBACK_ENV = "RL_KERNEL_READBACK_DIR"
SCHEMA = "rlkernel.museglimmer.vllm_strict.v1"
REAL_VOCAB = int(OFFICIAL_FINGERPRINT["vocab_size"])
PADDED_VOCAB = int(OFFICIAL_PARALLEL_LAYOUT["padded_vocab_size"])
CANONICAL_TP = int(OFFICIAL_PARALLEL_LAYOUT["canonical_tp"])
TILE_ROWS = PADDED_VOCAB // CANONICAL_TP
HEAD_DIM = int(OFFICIAL_FINGERPRINT["head_dim"])
HIDDEN = int(OFFICIAL_FINGERPRINT["hidden_size"])
FFN = int(OFFICIAL_FINGERPRINT["intermediate_size"])
NUM_HEADS = int(OFFICIAL_FINGERPRINT["num_attention_heads"])
NUM_KV_HEADS = int(OFFICIAL_FINGERPRINT["num_key_value_heads"])
LOGP_CHUNK_ROWS = 1024

LORA_NAMES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "gate_proj",
    "o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)
_COLUMN_PARALLEL = frozenset(
    {"q_proj", "k_proj", "v_proj", "gate_proj", "mlp.gate_proj", "mlp.up_proj"}
)
# name -> (K, N) of the full (unsharded) weight.
LORA_SHAPES: dict[str, tuple[int, int]] = {
    "q_proj": (HIDDEN, NUM_HEADS * HEAD_DIM),
    "k_proj": (HIDDEN, NUM_KV_HEADS * HEAD_DIM),
    "v_proj": (HIDDEN, NUM_KV_HEADS * HEAD_DIM),
    "gate_proj": (HIDDEN, NUM_HEADS * HEAD_DIM),
    "o_proj": (NUM_HEADS * HEAD_DIM, HIDDEN),
    "mlp.gate_proj": (HIDDEN, FFN),
    "mlp.up_proj": (HIDDEN, FFN),
    "mlp.down_proj": (FFN, HIDDEN),
}
_LORA_KEY = re.compile(
    r"^layers\.(\d+)\.(q_proj|k_proj|v_proj|gate_proj|o_proj|mlp\.gate_proj|mlp\.up_proj|"
    r"mlp\.down_proj)\.lora_(A|B)$"
)
_MARK = "__rl_kernel_museglimmer_strict_original__"
_SAMPLING_NOTE = (
    "token selection uses vLLM's native bf16 soft-capped logits (cuBLAS lm_head); only the "
    "selected-token logprob values are recomputed strictly; selected_token_ranks and extra top-k "
    "logprob columns stay native"
)

_STATE: StrictBindingState | None = None


# ----------------------------------------------------------------------------------------------
# State and provenance
# ----------------------------------------------------------------------------------------------


@dataclass
class StrictBindingState:
    ops: StrictOps
    hooks: list[str] = field(default_factory=list)
    call_counts: Counter = field(default_factory=Counter)
    lora_sha256: str | None = None
    lora: dict[int, dict[str, tuple[torch.Tensor, torch.Tensor]]] = field(default_factory=dict)
    tp_size: int = 1
    tp_rank: int = 0
    device_group: Any = None
    collective: Any = None
    collective_label: str = "none (TP1)"
    model: Any = None
    lm_head: Any = None
    model_class: str | None = None
    cfg: dict[str, Any] = field(default_factory=dict)
    rope_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = field(default_factory=dict)
    logits_ctx: torch.Tensor | None = None
    role: str = "engine"
    notes: list[str] = field(default_factory=list)
    checks: dict[str, str] = field(default_factory=dict)
    installed_at: float = field(default_factory=time.time)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def note(self, message: str) -> None:
        if message not in self.notes:
            self.notes.append(message)

    def take_logits_ctx(self) -> torch.Tensor | None:
        ctx = self.logits_ctx
        self.logits_ctx = None
        return ctx

    def kv_ownership(self) -> dict[str, Any]:
        t, r = self.tp_size, self.tp_rank
        heads_per_rank = NUM_HEADS // t
        replicas = max(1, t // NUM_KV_HEADS)
        kv_per_rank = max(1, NUM_KV_HEADS // t)
        kv_first = (r // replicas) * kv_per_rank
        chunks = list(fb.rank_chunk_ids(t, r))
        tiles_per_rank = CANONICAL_TP // t
        return {
            "q_heads": [r * heads_per_rank, (r + 1) * heads_per_rank],
            "kv_heads_projected": [kv_first, kv_first + kv_per_rank],
            "kv_heads_cached": [0, NUM_KV_HEADS],
            "kv_heads_replicated": True,
            "attention": (
                "TP1"
                if t == 1
                else "full 32/2-head attention run redundantly on every rank after an "
                "exact all-gather of q/k/v/gate (bit-identical to TP1)"
            ),
            "o_proj_chunks": chunks,
            "down_proj_chunks": chunks,
            "vocab_tiles": [r * tiles_per_rank, (r + 1) * tiles_per_rank],
            "chunk_widths": dict(OFFICIAL_PARALLEL_LAYOUT["row_parallel_shards"]),
        }

    def provenance(self) -> dict[str, Any]:
        try:
            import vllm

            vllm_version = str(vllm.__version__)
        except Exception:  # noqa: BLE001
            vllm_version = "unknown"
        counts = dict(self.call_counts)
        counts.update({k: int(v) for k, v in attn_backend.CALL_COUNTS.items()})
        ops_prov = self.ops.provenance()
        return {
            "schema": SCHEMA,
            "role": self.role,
            "pid": os.getpid(),
            "strict": True,
            "vllm_version": vllm_version,
            "model_class": self.model_class,
            "hooks": list(self.hooks),
            "attention_backend": attn_backend.BACKEND_CLASS_PATH,
            "op_impls": ops_prov["op_impls"],
            "fallbacks": ops_prov["fallbacks"],
            "demotions": ops_prov["demotions"],
            "all_ops_strict": ops_prov["all_strict"],
            "backend_ids": {
                "gemm": self.ops.labels["det_gemm"],
                "attention": self.ops.labels["gated_attention_paged"],
                "logp": self.ops.labels["softcap_tile_stats"],
                "collective": self.collective_label,
            },
            "call_counts": counts,
            "lora_sha256": self.lora_sha256,
            "lora_slots": {str(k): sorted(v) for k, v in self.lora.items()},
            "tp": self.tp_size,
            "tp_rank": self.tp_rank,
            "canonical_tp": CANONICAL_TP,
            "kv_ownership": self.kv_ownership(),
            "sampling": _SAMPLING_NOTE,
            "config_pins": dict(self.cfg),
            "checks": dict(self.checks),
            "notes": list(self.notes),
        }

    def persist(self) -> Path | None:
        directory = os.getenv(READBACK_ENV, "").strip()
        if not directory:
            return None
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        path = target / f"museglimmer-vllm-{self.role}-rank{self.tp_rank}-pid{os.getpid()}.json"
        temporary = path.with_suffix(".tmp")
        with self.lock:
            temporary.write_text(
                json.dumps(self.provenance(), indent=2, sort_keys=True, default=str),
                encoding="utf-8",
            )
            temporary.replace(path)
        return path


def get_state() -> StrictBindingState | None:
    return _STATE


def require_state() -> StrictBindingState:
    if _STATE is None:
        raise RuntimeError("Muse Glimmer strict vLLM binding is not installed")
    return _STATE


# ----------------------------------------------------------------------------------------------
# Pure helpers (unit-testable without vLLM)
# ----------------------------------------------------------------------------------------------


def canonical_qkv_slices(
    fused_weight: torch.Tensor, q_size: int, kv_size: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split vLLM's per-rank fused ``qkv_proj.weight`` rows into (Wq, Wk, Wv) views (no copy)."""

    if fused_weight.shape[0] != q_size + 2 * kv_size:
        raise ValueError(
            f"fused qkv weight has {fused_weight.shape[0]} rows, expected {q_size + 2 * kv_size}"
        )
    return (
        fused_weight[:q_size],
        fused_weight[q_size : q_size + kv_size],
        fused_weight[q_size + kv_size : q_size + 2 * kv_size],
    )


def canonical_gate_up_slices(fused_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Split vLLM's per-rank ``gate_up_proj.weight`` rows into (W_gate, W_up) views."""

    if fused_weight.shape[0] % 2:
        raise ValueError("fused gate_up weight must have an even number of rows")
    half = fused_weight.shape[0] // 2
    return fused_weight[:half], fused_weight[half:]


def lm_head_tiles_for_rank(tp_size: int, rank: int) -> list[int]:
    """Canonical vocab tiles owned by ``rank`` once the lm_head is padded to 202752 rows."""

    if CANONICAL_TP % tp_size:
        raise ValueError(f"tp_size {tp_size} must divide the canonical TP {CANONICAL_TP}")
    per_rank = CANONICAL_TP // tp_size
    return list(range(rank * per_rank, (rank + 1) * per_rank))


def shard_lora(
    name: str,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    *,
    tp_size: int,
    tp_rank: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Shard one full LoRA pair like the base weight (see module docstring)."""

    if name not in LORA_SHAPES:
        raise KeyError(f"unknown LoRA slot {name!r}")
    k_full, n_full = LORA_SHAPES[name]
    if lora_a.dim() != 2 or lora_b.dim() != 2:
        raise ValueError(f"{name}: lora_A/lora_B must be 2-D")
    rank = lora_a.shape[0]
    if lora_a.shape != (rank, k_full) or lora_b.shape != (n_full, rank):
        raise ValueError(
            f"{name}: expected lora_A ({rank}, {k_full}) and lora_B ({n_full}, {rank}), got "
            f"{tuple(lora_a.shape)} and {tuple(lora_b.shape)}"
        )
    if name in _COLUMN_PARALLEL:
        if name in ("k_proj", "v_proj"):
            replicas = max(1, tp_size // NUM_KV_HEADS)
            kv_per_rank = max(1, NUM_KV_HEADS // tp_size)
            n_local = kv_per_rank * HEAD_DIM
            start = (tp_rank // replicas) * n_local
        else:
            n_local = n_full // tp_size
            start = tp_rank * n_local
        return lora_a, lora_b[start : start + n_local]
    k_local = k_full // tp_size
    start = tp_rank * k_local
    return lora_a[:, start : start + k_local], lora_b


def parse_lora_state_dict(
    state_dict: Mapping[str, torch.Tensor],
) -> dict[tuple[int, str], tuple[torch.Tensor, torch.Tensor]]:
    """Group ``layers.{i}.{name}.lora_A/B`` keys into ``{(layer, name): (A, B)}``; fail on junk."""

    grouped: dict[tuple[int, str], dict[str, torch.Tensor]] = {}
    for key, value in state_dict.items():
        match = _LORA_KEY.match(key)
        if match is None:
            raise KeyError(f"unsupported LoRA key {key!r}")
        layer, name, which = int(match.group(1)), match.group(2), match.group(3)
        grouped.setdefault((layer, name), {})[which] = value
    out: dict[tuple[int, str], tuple[torch.Tensor, torch.Tensor]] = {}
    for (layer, name), pair in grouped.items():
        if set(pair) != {"A", "B"}:
            raise KeyError(f"layers.{layer}.{name}: both lora_A and lora_B are required")
        out[(layer, name)] = (pair["A"], pair["B"])
    return out


# ----------------------------------------------------------------------------------------------
# TP collectives
# ----------------------------------------------------------------------------------------------


def _init_collective(state: StrictBindingState) -> None:
    if state.tp_size == 1 or state.device_group is None:
        state.collective = None
        state.collective_label = "none (TP1)"
        return
    try:
        from rl_engine.distributed.algorithms.collectives import collective_for_group

        collective = collective_for_group(state.device_group)
        if collective is None:
            raise RuntimeError("collective_for_group returned None")
        state.collective = collective
        state.collective_label = f"rl_engine DeterministicCollective ({collective.backend_id})"
    except Exception as exc:  # noqa: BLE001 - the compiled _C extension is absent here
        state.collective = None
        state.collective_label = (
            "torch.distributed.all_gather_into_tensor over the TP device group (NCCL; pure "
            "data movement, bit-exact) + local canonical tree; repo DeterministicCollective "
            f"unavailable: {type(exc).__name__}: {exc}"
        )


def all_gather_stack(state: StrictBindingState, x: torch.Tensor) -> torch.Tensor:
    """Rank-ordered gather of ``x`` -> ``(tp, *x.shape)``; exact (no arithmetic)."""

    if state.tp_size == 1:
        return x.unsqueeze(0)
    x = x.contiguous()
    if state.collective is not None:
        gathered = state.collective.all_gather(x.reshape(1, -1))
        return gathered.reshape(state.tp_size, *x.shape)
    import torch.distributed as dist

    out = torch.empty((state.tp_size, *x.shape), dtype=x.dtype, device=x.device)
    dist.all_gather_into_tensor(out, x, group=state.device_group)
    return out


def tree_over_ranks(state: StrictBindingState, partial: torch.Tensor) -> torch.Tensor:
    """fp32 sub-tree partial of this rank -> canonical full tree over all ranks."""

    if state.tp_size == 1:
        return partial
    gathered = all_gather_stack(state, partial)
    return state.ops.combine_tree([gathered[i] for i in range(state.tp_size)])


# ----------------------------------------------------------------------------------------------
# Strict projections
# ----------------------------------------------------------------------------------------------


_PAD = 64  # M/N padding of the LoRA A GEMM (same as strict_kernels._GemmBackend.partial)


def _padded_partial(
    state: StrictBindingState,
    a: torch.Tensor,
    w: torch.Tensor,
    chunk_ids: tuple[int, ...],
) -> torch.Tensor:
    """fp32 canonical sub-tree partial of ``a @ w.T`` with M and N zero-padded to 64.

    This is the exact call pattern of ``strict_kernels._GemmBackend.partial`` (the Megatron
    binding's GEMM entry), needed for the rank-r LoRA ``A`` GEMM whose N (= r) is below one tile.
    Zero rows/columns cannot change the bits of the other outputs; K is never padded.
    """

    m, n = a.shape[0], w.shape[0]
    m_pad = -(-m // _PAD) * _PAD
    n_pad = -(-n // _PAD) * _PAD
    a_in = a if m_pad == m else torch.nn.functional.pad(a, (0, 0, 0, m_pad - m))
    w_in = w if n_pad == n else torch.nn.functional.pad(w, (0, 0, 0, n_pad - n))
    out = state.ops.det_gemm_partial(a_in.contiguous(), w_in, tuple(chunk_ids))
    if m_pad != m or n_pad != n:
        out = out[:m, :n]
    return out


def lora_small_k(z: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``sum_k B[:, k] z[k]`` in fp32, fixed order over the rank (Megatron's ``small_k_fp32``)."""

    from rl_engine.models.muse_glimmer.runtime.strict_kernels import small_k_fp32

    return small_k_fp32(z.contiguous(), b.contiguous())


def column_parallel_project(
    state: StrictBindingState,
    x: torch.Tensor,
    w_shard: torch.Tensor,
    lora: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    """``bf16(x @ W_shard^T)`` through det_gemm (+ strict unmerged LoRA, one rounding)."""

    if lora is None:
        return state.ops.det_gemm(x, w_shard, out_dtype=torch.bfloat16)
    lora_a, lora_b = lora
    acc = state.ops.det_gemm(x, w_shard, out_dtype=torch.float32)
    z = _padded_partial(state, x, lora_a, tuple(range(CANONICAL_TP))).to(torch.bfloat16)
    state.call_counts["lora.column"] += 1
    return (acc + lora_small_k(z, lora_b)).to(torch.bfloat16)


def row_parallel_project(
    state: StrictBindingState,
    x_local: torch.Tensor,
    w_local: torch.Tensor,
    lora: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Row-parallel GEMM: local canonical-chunk partial -> all-gather -> tree -> one bf16 cast."""

    chunk_ids = fb.rank_chunk_ids(state.tp_size, state.tp_rank)
    acc = tree_over_ranks(state, state.ops.det_gemm_partial(x_local, w_local, chunk_ids))
    if lora is None:
        return acc.to(torch.bfloat16)
    lora_a_local, lora_b = lora
    z = tree_over_ranks(state, _padded_partial(state, x_local, lora_a_local, chunk_ids))
    state.call_counts["lora.row"] += 1
    return (acc + lora_small_k(z.to(torch.bfloat16), lora_b)).to(torch.bfloat16)


def strict_selected_logp(
    state: StrictBindingState, hidden: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    """O7 for the given rows: fp32 tile logits -> softcap tile stats -> gathered tree -> logp."""

    lm_head = state.lm_head
    if lm_head is None:
        raise RuntimeError("strict logp requires the bound Muse Glimmer lm_head")
    weight = lm_head.weight
    tiles = lm_head_tiles_for_rank(state.tp_size, state.tp_rank)
    if weight.shape[0] != len(tiles) * TILE_ROWS:
        raise RuntimeError(
            f"lm_head shard has {weight.shape[0]} rows; strict mode requires the canonical "
            f"{len(tiles)} tiles x {TILE_ROWS} rows (ParallelLMHead padding patch missing?)"
        )
    cfg = state.cfg
    multiplier = float(cfg["output_multiplier"])
    softcap = float(cfg["final_logit_softcapping"])
    rows = hidden.shape[0]
    outputs = []
    for start in range(0, rows, LOGP_CHUNK_ROWS):
        h = hidden[start : start + LOGP_CHUNK_ROWS]
        tgt = targets[start : start + LOGP_CHUNK_ROWS].to(torch.int64)
        local_stats = []
        for i, tile_id in enumerate(tiles):
            w_tile = weight[i * TILE_ROWS : (i + 1) * TILE_ROWS]
            z = state.ops.det_gemm(h, w_tile, out_dtype=torch.float32)
            m_t, s_t, target_logit, in_tile = state.ops.softcap_tile_stats(
                z,
                tile_id,
                tgt,
                multiplier=multiplier,
                softcap=softcap,
                vocab_size=REAL_VOCAB,
            )
            local_stats.append(torch.stack([m_t, s_t, target_logit, in_tile.to(torch.float32)]))
        local = torch.stack(local_stats)  # (tiles_per_rank, 4, T)
        gathered = all_gather_stack(state, local).reshape(CANONICAL_TP, 4, h.shape[0])
        stats = [
            (gathered[j, 0], gathered[j, 1], gathered[j, 2], gathered[j, 3] > 0.5)
            for j in range(CANONICAL_TP)
        ]
        # combine_tile_stats consumes (M_t, S_t) pairs in tile order; the target logit is the
        # single non-zero entry over the tiles (fixed tree, exact).
        _, log_z = state.ops.combine_tile_stats([(m, s) for m, s, _tl, _in in stats])
        outputs.append(fb.target_logit_from_stats(stats) - log_z)
    if not outputs:
        return hidden.new_empty((0,), dtype=torch.float32)
    return torch.cat(outputs) if len(outputs) > 1 else outputs[0]


def _rope_for(
    state: StrictBindingState, positions: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    key = str(positions.device)
    tables = state.rope_cache.get(key)
    if tables is None:
        max_pos = int(state.cfg["rope_table_positions"])
        cos, sin = state.ops.rope_table(
            torch.arange(max_pos, device=positions.device),
            theta=float(state.cfg["rope_theta"]),
            dim=HEAD_DIM,
            device=positions.device,
        )
        tables = (cos.contiguous(), sin.contiguous())
        state.rope_cache[key] = tables
    cos, sin = tables
    idx = positions.to(torch.long)
    return cos[idx], sin[idx]


# ----------------------------------------------------------------------------------------------
# vLLM patches
# ----------------------------------------------------------------------------------------------


def _check_pins(text: Any) -> dict[str, Any]:
    """Fail closed unless the served text config is the pinned Muse-Glimmer-30B."""

    pins = {}
    for key in (
        "num_hidden_layers",
        "hidden_size",
        "intermediate_size",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "sliding_window",
        "vocab_size",
        "rms_norm_eps",
        "post_norm_eps",
        "qk_scale_factor",
        "output_multiplier",
        "final_logit_softcapping",
    ):
        value = getattr(text, key, None)
        expected = OFFICIAL_FINGERPRINT[key]
        if value is None or value != expected:
            raise MuseGlimmerWorkloadError(
                f"vLLM text_config {key}={value!r} is not the pinned {expected!r}"
            )
        pins[key] = value
    layer_types = list(getattr(text, "layer_types", None) or [])
    if layer_types != expected_layer_types():
        raise MuseGlimmerWorkloadError("vLLM text_config layer_types deviate from the pin")
    rope = getattr(text, "rope_parameters", None) or {}
    theta = (
        rope.get("rope_theta") if isinstance(rope, Mapping) else getattr(rope, "rope_theta", None)
    )
    if float(theta or 0.0) != float(OFFICIAL_FINGERPRINT["rope_parameters"]["rope_theta"]):
        raise MuseGlimmerWorkloadError(f"vLLM text_config rope_theta={theta!r} is not pinned")
    pins["rope_theta"] = float(theta)
    pins["max_position_embeddings"] = int(getattr(text, "max_position_embeddings"))
    return pins


def _bind_model(state: StrictBindingState, model: Any, vllm_config: Any) -> None:
    speculative = (
        vllm_config is not None and getattr(vllm_config, "speculative_config", None) is not None
    )
    check_runtime_binding(
        model_class=type(model).__name__,
        attention_projection_grouping=OFFICIAL_PROJECTION_GROUPING["attention"],
        speculative_decoding=bool(speculative),
    )
    state.checks["check_runtime_binding"] = "passed"
    pins = _check_pins(model.text_config)
    state.checks["config_pins"] = "passed"
    if vllm_config is not None:
        from vllm.config import CompilationMode

        model_config = vllm_config.model_config
        compilation = vllm_config.compilation_config
        eager = bool(getattr(model_config, "enforce_eager", False)) or (
            getattr(compilation, "mode", None) == CompilationMode.NONE
        )
        if not eager:
            raise RuntimeError(
                "strict Muse Glimmer binding requires enforce_eager=True (the strict ops are "
                "plain Python/Triton calls, not torch.compile custom ops)"
            )
        cache_dtype = str(getattr(vllm_config.cache_config, "cache_dtype", "auto"))
        if cache_dtype not in ("auto", "bfloat16"):
            raise RuntimeError(
                f"strict Muse Glimmer binding requires a bf16 KV cache, got {cache_dtype}"
            )
        state.cfg["max_model_len"] = int(model_config.max_model_len)
        state.checks["eager"] = "passed"
        state.checks["kv_cache_dtype"] = cache_dtype
    try:
        from vllm.distributed.parallel_state import get_tp_group

        group = get_tp_group()
        state.tp_size = int(group.world_size)
        state.tp_rank = int(group.rank_in_group)
        state.device_group = group.device_group
    except Exception:  # noqa: BLE001 - single-process unit tests
        state.tp_size, state.tp_rank, state.device_group = 1, 0, None
    if CANONICAL_TP % state.tp_size:
        raise RuntimeError(f"TP {state.tp_size} does not divide the canonical TP {CANONICAL_TP}")
    _init_collective(state)
    device = next(model.parameters()).device
    probe_ops(state.ops, device)
    attn_backend.bind_ops(state.ops)

    layers = model.model.layers
    first_attention = None
    for layer in layers:
        attention = getattr(layer, "self_attn", None)
        if attention is None:
            continue  # pipeline-parallel placeholder
        if not getattr(attention, "use_output_gate", False):
            raise RuntimeError("strict Muse Glimmer requires the attention output gate")
        if not getattr(attention, "use_qk_norm", False):
            raise RuntimeError("strict Muse Glimmer requires qk-norm")
        if abs(float(attention.scale_query_by) - float(pins["qk_scale_factor"])) > 1e-9:
            raise RuntimeError(
                f"vLLM derived qk scale {attention.scale_query_by} != pinned "
                f"{pins['qk_scale_factor']}"
            )
        expected_local = expected_layer_types()[int(attention.layer_idx)] != FULL_ATTENTION
        if bool(attention.use_rope) != expected_local:
            raise RuntimeError(f"layer {attention.layer_idx}: rope/window schedule mismatch")
        impl = attention.attn.impl
        if int(impl.num_heads) != NUM_HEADS or int(impl.num_kv_heads) != NUM_KV_HEADS:
            raise RuntimeError(
                f"layer {attention.layer_idx}: vLLM Attention has {impl.num_heads}/"
                f"{impl.num_kv_heads} heads; strict mode needs the full {NUM_HEADS}/{NUM_KV_HEADS}"
            )
        if type(impl).__name__ != "MuseGlimmerStrictAttentionImpl":
            raise RuntimeError(
                f"layer {attention.layer_idx}: attention impl is {type(impl).__name__}, not the "
                "strict backend (select attention_config backend FLASH_ATTN)"
            )
        layer.mlp._rlk_layer_idx = int(attention.layer_idx)
        setattr(attention.attn, attn_backend.GATE_ATTR, None)
        first_attention = first_attention or attention
    if first_attention is None:
        raise RuntimeError("no Muse Glimmer attention layers found")
    _bind_final_norm(state, model.model.norm)
    state.model = model
    state.lm_head = model.lm_head
    state.model_class = type(model).__name__
    state.role = "worker"
    state.cfg.update(pins)
    state.cfg["rope_table_positions"] = int(
        min(
            pins["max_position_embeddings"],
            state.cfg.get("max_model_len", pins["max_position_embeddings"]),
        )
    )
    state.cfg["window_keys_incl_self"] = int(pins["sliding_window"])
    state.cfg["sampling"] = "native"
    state.persist()


def _bind_final_norm(state: StrictBindingState, norm: Any) -> None:
    """Route the final norm (``MuseGlimmerModel.norm``, y = x_hat * w) to the strict O2 op.

    vLLM's native ``MuseGlimmerRMSNorm`` uses ``torch.mean`` (unspecified reduction order), so it
    is replaced on the instance; the module object, its weight and ``MuseGlimmerModel.forward``
    stay native.
    """

    if int(getattr(norm, "weight_offset", 0)) != 0 or not bool(getattr(norm, "with_scale", True)):
        raise RuntimeError("strict Muse Glimmer: final norm is not the weighted (w, no +1) form")
    eps = float(norm.eps)

    def final_norm_forward(hidden_states: torch.Tensor) -> torch.Tensor:
        state.call_counts["final_norm"] += 1
        return state.ops.weighted_rmsnorm(hidden_states, norm.weight, eps=eps)

    norm.forward = final_norm_forward
    hook = "vllm.model_executor.models.muse_glimmer.MuseGlimmerModel.norm.forward (instance)"
    if hook not in state.hooks:
        state.hooks.append(hook)


def _patch_lm_head_padding(state: StrictBindingState) -> None:
    from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead

    if hasattr(ParallelLMHead, _MARK):
        return
    original_init = ParallelLMHead.__init__
    original_loader = ParallelLMHead.weight_loader

    def init_wrapped(self: Any, num_embeddings: int, embedding_dim: int, *args: Any, **kwargs: Any):
        if int(num_embeddings) != REAL_VOCAB or len(args) > 2:
            original_init(self, num_embeddings, embedding_dim, *args, **kwargs)
            return
        kwargs["org_num_embeddings"] = PADDED_VOCAB
        kwargs["padding_size"] = PADDED_VOCAB
        original_init(self, PADDED_VOCAB, embedding_dim, *args, **kwargs)
        if (
            int(self.num_embeddings_padded) != PADDED_VOCAB
            or int(self.org_vocab_size) != PADDED_VOCAB
        ):
            raise RuntimeError("strict lm_head padding did not produce the canonical 202752 layout")
        self._rlk_museglimmer_real_vocab = REAL_VOCAB
        state.call_counts["lm_head.padded_init"] += 1

    def loader_wrapped(self: Any, param: Any, loaded_weight: torch.Tensor) -> None:
        real = getattr(self, "_rlk_museglimmer_real_vocab", None)
        if (
            real is None
            or getattr(param, "output_dim", None) != 0
            or getattr(param, "packed_dim", None) is not None
            or loaded_weight.shape[0] != int(real)
        ):
            original_loader(self, param, loaded_weight)
            return
        start = int(self.shard_indices.org_vocab_start_index)
        shard = int(self.shard_indices.org_vocab_end_index) - start
        available = max(0, min(shard, int(real) - start))
        if available:
            param.data[:available].copy_(loaded_weight[start : start + available])
        if available < param.shape[0]:
            param.data[available:].zero_()

    setattr(ParallelLMHead, _MARK, original_init)
    ParallelLMHead.__init__ = init_wrapped
    ParallelLMHead.weight_loader = loader_wrapped
    state.hooks.append(
        "vllm.model_executor.layers.vocab_parallel_embedding.ParallelLMHead.__init__"
    )
    state.hooks.append(
        "vllm.model_executor.layers.vocab_parallel_embedding.ParallelLMHead.weight_loader"
    )


def gather_heads(
    state: StrictBindingState, local: torch.Tensor, *, replicas: int = 1
) -> torch.Tensor:
    """TP > 1: all-gather a per-rank head shard ``(T, n_local)`` into the full ``(T, n)`` tensor.

    Rank order is head order (column-parallel shards).  ``replicas`` > 1 (KV heads replicated
    across consecutive ranks) keeps one copy per distinct shard.  Pure data movement: exact.
    """

    if state.tp_size == 1:
        return local
    gathered = all_gather_stack(state, local)  # (tp, T, n_local)
    ranks = range(0, state.tp_size, replicas)
    return torch.cat([gathered[r] for r in ranks], dim=-1)


def _patch_attention_factory(state: StrictBindingState) -> None:
    """At TP > 1 build the vLLM ``Attention`` layer with the FULL head set (KV replicated).

    The strict kernel pins 32 query / 2 KV heads, so every rank keeps both KV heads in its own
    cache and runs the whole attention redundantly on exactly the TP1 inputs (bit-identical).
    """

    from vllm.model_executor.models import muse_glimmer as mg

    original_attention = mg.Attention
    if getattr(original_attention, _MARK, False):
        return

    def attention_factory(
        num_heads: int, head_size: int, scale: float, num_kv_heads=None, **kwargs
    ):
        from vllm.distributed import get_tensor_model_parallel_world_size

        tp = int(get_tensor_model_parallel_world_size())
        if tp > 1 and int(num_heads) * tp == NUM_HEADS and int(head_size) == HEAD_DIM:
            num_heads, num_kv_heads = NUM_HEADS, NUM_KV_HEADS
            state.note("TP>1: vLLM Attention layers built with the full 32/2 heads (KV replicated)")
        return original_attention(num_heads, head_size, scale, num_kv_heads, **kwargs)

    setattr(attention_factory, _MARK, True)
    mg.Attention = attention_factory
    state.hooks.append("vllm.model_executor.models.muse_glimmer.Attention (KV-replicated factory)")


def _patch_model(state: StrictBindingState) -> None:
    from vllm.model_executor.models import muse_glimmer as mg

    cls = mg.MuseGlimmerForCausalLM
    if hasattr(cls, _MARK):
        return
    _patch_attention_factory(state)
    original_init = cls.__init__
    original_logits = cls.compute_logits
    original_embed = mg.MuseGlimmerModel.embed_input_ids
    original_layer = mg.MuseGlimmerDecoderLayer.forward
    original_attention = mg.MuseGlimmerAttention.forward
    original_mlp = mg.MuseGlimmerMLP.forward

    def init_wrapped(self: Any, *, vllm_config: Any, prefix: str = "") -> None:
        original_init(self, vllm_config=vllm_config, prefix=prefix)
        _bind_model(state, self, vllm_config)

    def compute_logits(self: Any, hidden_states: torch.Tensor) -> torch.Tensor | None:
        logits = original_logits(self, hidden_states)
        state.logits_ctx = hidden_states
        state.call_counts["compute_logits.native_for_sampling"] += 1
        return logits

    def embed_input_ids(self: Any, input_ids: torch.Tensor) -> torch.Tensor:
        embedding = self.embed_tokens
        eps = float(self.embed_norm.eps)
        if state.tp_size == 1 or int(getattr(embedding, "tp_size", 1)) == 1:
            out = state.ops.normalized_embedding(embedding.weight, input_ids, eps=eps)
        else:
            start = int(embedding.shard_indices.org_vocab_start_index)
            end = int(embedding.shard_indices.org_vocab_end_index)
            shard = end - start
            ids = input_ids.to(torch.long)
            in_range = (ids >= start) & (ids < end)
            local = embedding.weight[(ids - start).clamp(0, shard - 1)]
            local = torch.where(in_range.unsqueeze(-1), local, torch.zeros_like(local))
            gathered = all_gather_stack(state, local)  # (tp, T, H); exact rows
            owner = (ids // shard).clamp(0, state.tp_size - 1)
            rows = gathered[owner, torch.arange(ids.shape[0], device=ids.device)]
            out = state.ops.normalized_embedding(
                rows, torch.arange(ids.shape[0], device=ids.device), eps=eps
            )
        state.call_counts["embed_input_ids"] += 1
        return out

    def layer_forward(
        self: Any,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ops = state.ops
        residual = hidden_states
        x_n = ops.centered_rmsnorm(
            hidden_states, self.input_layernorm.weight, eps=float(self.input_layernorm.eps)
        )
        branch = self.self_attn(positions=positions, hidden_states=x_n)
        hidden_states, x_f = ops.fused_post_norm_residual_pre_norm(
            branch,
            residual,
            self.post_attention_layernorm.weight,
            float(self.post_attention_layernorm.eps),
            self.pre_feedforward_layernorm.weight,
            float(self.pre_feedforward_layernorm.eps),
            pre_form="centered",
        )
        residual = hidden_states
        ffn = self.mlp(x_f)
        post = ops.centered_rmsnorm(
            ffn,
            self.post_feedforward_layernorm.weight,
            eps=float(self.post_feedforward_layernorm.eps),
        )
        hidden_states = (residual.float() + post.float()).to(torch.bfloat16)
        state.call_counts["decoder_layer"] += 1
        return hidden_states, residual

    def attention_forward(
        self: Any, positions: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        ops = state.ops
        layer_idx = int(self.layer_idx)
        lora = state.lora.get(layer_idx, {})
        w_q, w_k, w_v = canonical_qkv_slices(self.qkv_proj.weight, self.q_size, self.kv_size)
        w_g = self.output_gate_proj.weight
        q = column_parallel_project(state, hidden_states, w_q, lora.get("q_proj"))
        k = column_parallel_project(state, hidden_states, w_k, lora.get("k_proj"))
        v = column_parallel_project(state, hidden_states, w_v, lora.get("v_proj"))
        gate = column_parallel_project(state, hidden_states, w_g, lora.get("gate_proj"))
        tokens = hidden_states.shape[0]
        if state.tp_size > 1:
            # KV-replicated attention: exact all-gather of the per-rank head shards so the strict
            # kernel always runs the full 32/2-head problem on TP1-identical inputs.
            kv_replicas = max(1, state.tp_size // NUM_KV_HEADS)
            q = gather_heads(state, q)
            k = gather_heads(state, k, replicas=kv_replicas)
            v = gather_heads(state, v, replicas=kv_replicas)
            gate = gather_heads(state, gate)
        q_heads = q.view(tokens, NUM_HEADS, HEAD_DIM)
        k_heads = k.view(tokens, NUM_KV_HEADS, HEAD_DIM)
        if self.use_rope:
            cos, sin = _rope_for(state, positions)
        else:
            cos = sin = None
        q_heads, k_heads = ops.qknorm_scaled_rope(
            q_heads,
            k_heads,
            cos,
            sin,
            eps=float(self.qk_norm.eps),
            qk_scale=float(self.scale_query_by),
        )
        setattr(self.attn, attn_backend.GATE_ATTR, gate.contiguous())
        try:
            attended = self.attn(
                q_heads.reshape(tokens, NUM_HEADS * HEAD_DIM),
                k_heads.reshape(tokens, NUM_KV_HEADS * HEAD_DIM),
                v,
            )
        finally:
            setattr(self.attn, attn_backend.GATE_ATTR, None)
        if state.tp_size > 1:
            # Keep this rank's heads (= the o_proj weight shard's K columns).
            attended = attended[:, state.tp_rank * self.q_size : (state.tp_rank + 1) * self.q_size]
            attended = attended.contiguous()
        state.call_counts["attention_forward"] += 1
        return row_parallel_project(state, attended, self.o_proj.weight, lora.get("o_proj"))

    def mlp_forward(self: Any, x: torch.Tensor) -> torch.Tensor:
        lora = state.lora.get(int(getattr(self, "_rlk_layer_idx", -1)), {})
        w_gate, w_up = canonical_gate_up_slices(self.gate_up_proj.weight)
        gate = column_parallel_project(state, x, w_gate, lora.get("mlp.gate_proj"))
        up = column_parallel_project(state, x, w_up, lora.get("mlp.up_proj"))
        hidden = state.ops.swiglu(gate, up)
        state.call_counts["mlp_forward"] += 1
        return row_parallel_project(state, hidden, self.down_proj.weight, lora.get("mlp.down_proj"))

    setattr(cls, _MARK, original_init)
    cls.__init__ = init_wrapped
    cls.compute_logits = compute_logits
    mg.MuseGlimmerModel.embed_input_ids = embed_input_ids
    mg.MuseGlimmerDecoderLayer.forward = layer_forward
    mg.MuseGlimmerAttention.forward = attention_forward
    mg.MuseGlimmerMLP.forward = mlp_forward
    setattr(mg.MuseGlimmerModel, _MARK, original_embed)
    setattr(mg.MuseGlimmerDecoderLayer, _MARK, original_layer)
    setattr(mg.MuseGlimmerAttention, _MARK, original_attention)
    setattr(mg.MuseGlimmerMLP, _MARK, original_mlp)
    prefix = "vllm.model_executor.models.muse_glimmer."
    state.hooks.extend(
        [
            prefix + "MuseGlimmerForCausalLM.__init__",
            prefix + "MuseGlimmerForCausalLM.compute_logits",
            prefix + "MuseGlimmerModel.embed_input_ids",
            prefix + "MuseGlimmerDecoderLayer.forward",
            prefix + "MuseGlimmerAttention.forward",
            prefix + "MuseGlimmerMLP.forward",
        ]
    )


def _patch_sampler(state: StrictBindingState) -> None:
    from vllm.v1.worker.gpu.sample.sampler import Sampler

    if hasattr(Sampler, _MARK):
        return
    original_call = Sampler.__call__

    def call_wrapped(self: Any, logits: torch.Tensor, input_batch: Any) -> Any:
        output = original_call(self, logits, input_batch)
        hidden = state.take_logits_ctx()
        logprobs_tensors = getattr(output, "logprobs_tensors", None)
        if logprobs_tensors is None:
            return output
        if hidden is None or hidden.shape[0] != logits.shape[0]:
            raise RuntimeError(
                "strict Muse Glimmer logp: sampler rows do not match the hidden states published "
                "by MuseGlimmerForCausalLM.compute_logits (fail closed)"
            )
        targets = output.sampled_token_ids.reshape(-1).to(torch.int64)
        if targets.shape[0] != logits.shape[0]:
            raise RuntimeError("strict Muse Glimmer logp: unexpected sampled-token layout")
        logprobs_tensors.logprobs[:, 0] = strict_selected_logp(state, hidden, targets)
        state.call_counts["sampler.strict_logp_calls"] += 1
        state.call_counts["sampler.strict_logp_rows"] += int(logits.shape[0])
        if state.call_counts["sampler.strict_logp_calls"] == 1:
            state.persist()
        return output

    setattr(Sampler, _MARK, original_call)
    Sampler.__call__ = call_wrapped
    state.hooks.append("vllm.v1.worker.gpu.sample.sampler.Sampler.__call__")


def _patch_prompt_logprobs(state: StrictBindingState) -> None:
    from vllm.v1.worker.gpu.sample import prompt_logprob as module

    if hasattr(module, _MARK):
        return
    original = module.compute_prompt_logprobs_with_chunking

    def wrapped(
        prompt_token_ids: torch.Tensor,
        prompt_hidden_states: torch.Tensor,
        logits_fn: Callable[[torch.Tensor], torch.Tensor],
        num_prompt_logprobs: int,
        logprobs_mode: Any = "raw_logprobs",
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        token_ids, scores, ranks = original(
            prompt_token_ids, prompt_hidden_states, logits_fn, num_prompt_logprobs, logprobs_mode
        )
        state.take_logits_ctx()
        targets = prompt_token_ids.to(torch.int64)
        scores[:, 0] = strict_selected_logp(
            state, prompt_hidden_states[: targets.shape[0]], targets
        )
        state.call_counts["prompt_logprobs.strict_logp_calls"] += 1
        state.call_counts["prompt_logprobs.strict_logp_rows"] += int(targets.shape[0])
        if state.call_counts["prompt_logprobs.strict_logp_calls"] == 1:
            state.persist()
        return token_ids, scores, ranks

    setattr(module, _MARK, original)
    module.compute_prompt_logprobs_with_chunking = wrapped
    state.hooks.append(
        "vllm.v1.worker.gpu.sample.prompt_logprob.compute_prompt_logprobs_with_chunking"
    )


# ----------------------------------------------------------------------------------------------
# Install + worker RPC entry points
# ----------------------------------------------------------------------------------------------


def install_museglimmer_vllm_strict(*, force_fallback: bool | None = None) -> StrictBindingState:
    """Install every hook (idempotent); fails closed later at model init if the model is not native."""

    global _STATE
    if _STATE is not None:
        return _STATE
    state = StrictBindingState(ops=resolve_ops(force_fallback))
    _STATE = state
    state.hooks.append(attn_backend.register(state.ops))
    _patch_lm_head_padding(state)
    _patch_model(state)
    _patch_sampler(state)
    _patch_prompt_logprobs(state)
    atexit.register(state.persist)
    state.persist()
    return state


def worker_provenance(_worker: Any = None) -> dict[str, Any]:
    """``LLM.collective_rpc`` target: this worker's provenance (persisted as a side effect)."""

    state = require_state()
    state.persist()
    return state.provenance()


def worker_load_lora(
    _worker: Any, state_dict: Mapping[str, torch.Tensor], sha256: str
) -> dict[str, Any]:
    """``LLM.collective_rpc`` target: install strict unmerged LoRA slots on this rank."""

    state = require_state()
    if state.model is None:
        raise RuntimeError("LoRA can only be loaded after the Muse Glimmer model is bound")
    device = next(state.model.parameters()).device
    t0 = time.perf_counter()
    if state_dict:
        from rl_engine.models.muse_glimmer.e2e.sync import lora_sha256

        received = lora_sha256(state_dict)
        if received != str(sha256):
            raise RuntimeError(
                f"rank {state.tp_rank}: received LoRA hashes to {received}, declared {sha256} "
                "(muse_glimmer_lora_sha256_v1; fail closed)"
            )
    hash_ms = (time.perf_counter() - t0) * 1e3
    pairs = parse_lora_state_dict(state_dict)
    new_slots: dict[int, dict[str, tuple[torch.Tensor, torch.Tensor]]] = {}
    for (layer, name), (lora_a, lora_b) in pairs.items():
        a_local, b_local = shard_lora(
            name,
            lora_a.to(torch.bfloat16),
            lora_b.to(torch.bfloat16),
            tp_size=state.tp_size,
            tp_rank=state.tp_rank,
        )
        new_slots.setdefault(layer, {})[name] = (
            a_local.to(device).contiguous(),
            b_local.to(device).contiguous(),
        )
    state.lora = new_slots
    state.lora_sha256 = str(sha256) if pairs else None
    state.call_counts["load_lora"] += 1
    state.persist()
    return {
        "rank": state.tp_rank,
        "layers": len(new_slots),
        "slots": sum(len(v) for v in new_slots.values()),
        "sha256": state.lora_sha256,
        "rehash_ms": hash_ms,
        "install_ms": (time.perf_counter() - t0) * 1e3,
    }


def worker_check_lora_rule(
    _worker: Any, layer: int, tokens: int = 7, seed: int = 0
) -> dict[str, Any]:
    """Check the declared unmerged LoRA rule on this rank's REAL layer weights (TP1 only).

    For each of the 8 slots of ``layer`` (with the LoRA currently loaded) and random bf16 inputs
    ``x`` of ``tokens`` rows, compares bitwise:

    * ``binding``: the forward path the engine uses (``column_parallel_project`` /
      ``row_parallel_project``);
    * ``formula``: ``bf16(acc32(W x) + loop_k B[:, k] z[k])`` with ``z = bf16(acc32(A x))``, the
      fp32 loop written out here independently (exact bf16 products, sequential fp32 adds);
    * ``megatron``: ``runtime.megatron_lora.StrictColumnLinear/StrictRowLinear.forward`` (the
      trainer's implementation, imported read-only; it has no Megatron dependency) on the same
      tensors at TP1.

    Also reports the max |difference| against an fp64 evaluation of ``W x + B (A x)`` (tolerance
    view) and whether the LoRA term changes any bit of the base projection.
    """

    state = require_state()
    if state.tp_size != 1:
        raise RuntimeError("worker_check_lora_rule compares against TP1 references; run at TP1")
    slots = state.lora.get(int(layer))
    if not slots:
        raise RuntimeError(f"no LoRA loaded for layer {layer}")
    model = state.model
    decoder = model.model.layers[int(layer)]
    attn, mlp = decoder.self_attn, decoder.mlp
    w_q, w_k, w_v = canonical_qkv_slices(attn.qkv_proj.weight, attn.q_size, attn.kv_size)
    w_gate_mlp, w_up = canonical_gate_up_slices(mlp.gate_up_proj.weight)
    weights = {
        "q_proj": w_q,
        "k_proj": w_k,
        "v_proj": w_v,
        "gate_proj": attn.output_gate_proj.weight,
        "o_proj": attn.o_proj.weight,
        "mlp.gate_proj": w_gate_mlp,
        "mlp.up_proj": w_up,
        "mlp.down_proj": mlp.down_proj.weight,
    }
    from rl_engine.models.muse_glimmer.runtime import megatron_lora as ml
    from rl_engine.models.muse_glimmer.runtime.strict_kernels import TPComm

    comm = TPComm(None)
    device = w_q.device
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    out: dict[str, Any] = {"layer": int(layer), "tokens": int(tokens), "slots": {}}
    for name, w in weights.items():
        if name not in slots:
            continue
        lora_a, lora_b = slots[name]
        k_in = w.shape[1]
        x = torch.randn(int(tokens), k_in, generator=gen).to(device=device, dtype=torch.bfloat16)
        if name in _COLUMN_PARALLEL:
            y_bind = column_parallel_project(state, x, w, (lora_a, lora_b))
            y_base = column_parallel_project(state, x, w)
            acc = state.ops.det_gemm(x, w, out_dtype=torch.float32)
            y_meg = ml.StrictColumnLinear.apply(
                x, lora_a.float(), lora_b.float(), comm, int(w.shape[0]), w
            )
        else:
            y_bind = row_parallel_project(state, x, w, (lora_a, lora_b))
            y_base = row_parallel_project(state, x, w)
            acc = state.ops.det_gemm_partial(x, w, tuple(range(CANONICAL_TP)))
            y_meg = ml.StrictRowLinear.apply(
                x, w, lora_a.float(), lora_b.float(), comm, int(w.shape[1])
            )
        z = _padded_partial(state, x, lora_a, tuple(range(CANONICAL_TP))).to(torch.bfloat16)
        zf, bf = z.float(), lora_b.float()
        loop = torch.zeros_like(acc)
        for k in range(zf.shape[1]):
            loop = loop + zf[:, k : k + 1] * bf[:, k].unsqueeze(0)
        y_formula = (acc + loop).to(torch.bfloat16)
        ref64 = x.double() @ w.double().t() + (x.double() @ lora_a.double().t()) @ (
            lora_b.double().t()
        )
        out["slots"][name] = {
            "shape_x": list(x.shape),
            "shape_w": list(w.shape),
            "rank": int(lora_a.shape[0]),
            "binding_equals_formula": bool(torch.equal(y_bind, y_formula)),
            "binding_equals_megatron": bool(torch.equal(y_bind, y_meg.to(torch.bfloat16))),
            "lora_changed_elements": int((y_bind != y_base).sum().item()),
            "elements": int(y_bind.numel()),
            "max_abs_vs_fp64": float((y_bind.double() - ref64).abs().max().item()),
            "max_abs_base_vs_fp64": float(
                (y_base.double() - x.double() @ w.double().t()).abs().max().item()
            ),
            "max_abs_lora_term": float(loop.abs().max().item()),
        }
    out["all_equal"] = all(
        v["binding_equals_formula"] and v["binding_equals_megatron"]
        for v in out["slots"].values()
    )
    return out


__all__ = [
    "LORA_NAMES",
    "LORA_SHAPES",
    "PADDED_VOCAB",
    "READBACK_ENV",
    "REAL_VOCAB",
    "STRICT_ENV",
    "TILE_ROWS",
    "StrictBindingState",
    "all_gather_stack",
    "canonical_gate_up_slices",
    "canonical_qkv_slices",
    "column_parallel_project",
    "get_state",
    "install_museglimmer_vllm_strict",
    "lm_head_tiles_for_rank",
    "parse_lora_state_dict",
    "require_state",
    "row_parallel_project",
    "shard_lora",
    "strict_selected_logp",
    "tree_over_ranks",
    "lora_small_k",
    "worker_check_lora_rule",
    "worker_load_lora",
    "worker_provenance",
]
