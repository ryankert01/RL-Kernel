# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Unmerged strict LoRA for the Muse Glimmer Megatron Bridge binding.

Declared LoRA form (RFC #501 open question 3), identical on every TP size and intended to be
mirrored bit-for-bit by the rollout engine::

    z  = bf16( A·x )                      # det GEMM, fp32 accumulate, one rounding
    y  = bf16( acc32(W·x) + Σ_k B[:,k]·z[k] )   # base fp32 accumulator + fixed-order fp32 loop
                                               # over the rank, ONE rounding at the end

With ``B = 0`` the LoRA term is an exact (signed) zero, so ``y`` equals the base projection
bit-for-bit (verified by :func:`lora_is_bitwise_noop`). LoRA A/B are kept as fp32 masters and
cast to bf16 inside the autograd functions, so parameter gradients stay fp32.

Sharding (``tp`` = tensor-parallel size):

* column-parallel projections (q, k, v, attention gate, MLP gate, MLP up): ``A (r, K)`` is
  replicated, ``B (N, r)`` is split by output rows exactly like the base weight;
* row-parallel projections (o, MLP down): ``A (r, K)`` is split by input columns like the base
  weight, ``B (N, r)`` is replicated.

Backward reduction orders (contract gap, declared here): the input gradient of a
column-parallel projection is the fixed tree over the 8 canonical output-row chunks of the
FULL ``N`` (one whole-chunk GEMM each); the LoRA contributions use the same chunks; base and
LoRA fp32 input gradients are added once (``dx = bf16(dx_base + dx_lora)``); parameter
gradients reduce over the (zero-padded) token axis with the canonical 8-chunk tree.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import torch
from torch import nn

from rl_engine.models.muse_glimmer.runtime.strict_kernels import (
    CANONICAL_CHUNKS,
    TPComm,
    ceil_to,
    gemm_backend,
    gemm_fp32,
    gemm_single_chunk_fp32,
    local_chunk_ids,
    small_k_fp32,
)

COLUMN_PROJECTIONS = ("q_proj", "k_proj", "v_proj", "gate_proj", "mlp.gate_proj", "mlp.up_proj")
ROW_PROJECTIONS = ("o_proj", "mlp.down_proj")
PROJECTIONS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "gate_proj",
    "o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)
_TOKEN_PAD = CANONICAL_CHUNKS * 64
DEFAULT_LORA_RANK = 16


def _pad_tokens_t(x_t: torch.Tensor, m_pad: int) -> torch.Tensor:
    """Zero-pad the token axis (dim 1 of a transposed activation) to ``m_pad``."""
    m = x_t.shape[1]
    if m == m_pad:
        return x_t.contiguous()
    return torch.nn.functional.pad(x_t, (0, m_pad - m)).contiguous()


def _token_grad_gemm(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    """``lhs.T @ rhs`` reducing over tokens: ``lhs (M, P)``, ``rhs (M, Q)`` → fp32 ``(P, Q)``.

    Tokens are zero-padded to a multiple of 8*64 so the reduction forms 8 canonical chunks.
    """
    m_pad = ceil_to(lhs.shape[0], _TOKEN_PAD)
    a = _pad_tokens_t(lhs.t(), m_pad)
    w = _pad_tokens_t(rhs.t(), m_pad)
    return gemm_fp32(a, w)


class StrictColumnLinear(torch.autograd.Function):
    """``y_local = bf16(x·W_localᵀ + B_local·bf16(A·x))`` for a column-parallel shard.

    ``blocks`` are row-block views whose concatenation is this rank's shard of the weight in
    HF row order (e.g. the q rows of each local KV group inside the fused ``linear_qkv``). Each
    block is projected separately and the fp32 outputs concatenated: output columns are
    independent, so this is bit-identical to one GEMM over the concatenated rows. Every block
    must be a whole number of canonical ``N/8`` chunks.
    """

    @staticmethod
    def forward(ctx, x, a32, b32_local, comm: TPComm, n_total: int, *blocks):
        if x.dtype != torch.bfloat16 or any(w.dtype != torch.bfloat16 for w in blocks):
            raise TypeError("strict column linear needs bf16 activations and weights")
        parts = [gemm_fp32(x, w) for w in blocks]
        y32 = parts[0] if len(parts) == 1 else torch.cat(parts, dim=1)
        a = b = z = None
        if a32 is not None:
            a = a32.to(torch.bfloat16)
            b = b32_local.to(torch.bfloat16)
            z = gemm_fp32(x, a).to(torch.bfloat16)
            y32 = y32 + small_k_fp32(z, b)
        ctx.save_for_backward(x, a, b, z, *blocks)
        ctx.comm = comm
        ctx.n_total = n_total
        ctx.n_blocks = len(blocks)
        return y32.to(torch.bfloat16)

    @staticmethod
    def backward(ctx, dy):
        x, a, b, z, *blocks = ctx.saved_tensors
        comm: TPComm = ctx.comm
        dy = dy.contiguous()
        if dy.dtype != torch.bfloat16:
            dy = dy.to(torch.bfloat16)
        n_chunk = ctx.n_total // CANONICAL_CHUNKS
        ids = local_chunk_ids(comm.rank, comm.world)
        sizes = [w.shape[0] for w in blocks]
        if sum(sizes) != n_chunk * len(ids) or any(s % n_chunk for s in sizes):
            raise RuntimeError("column blocks do not tile this rank's canonical chunks")
        base_parts = []
        lora_parts = []
        block_index = 0
        block_start = 0
        for j, c in enumerate(ids):
            start = j * n_chunk
            while start >= block_start + sizes[block_index]:
                block_start += sizes[block_index]
                block_index += 1
            local = slice(start - block_start, start - block_start + n_chunk)
            dy_c = dy[:, start : start + n_chunk].contiguous()
            w_c = blocks[block_index][local].t().contiguous()
            base_parts.append(gemm_single_chunk_fp32(dy_c, w_c, c))
            if a is not None:
                lora_parts.append(
                    gemm_single_chunk_fp32(dy_c, b[start : start + n_chunk].t().contiguous(), c)
                )
        backend = gemm_backend()
        dx32 = comm.all_gather_tree(backend.combine(base_parts))
        da32 = db32 = None
        if a is not None:
            dz = comm.all_gather_tree(backend.combine(lora_parts)).to(torch.bfloat16)
            dx32 = dx32 + small_k_fp32(dz, a.t().contiguous())
            if ctx.needs_input_grad[1]:
                da32 = _token_grad_gemm(dz, x)  # (r, K)
            if ctx.needs_input_grad[2]:
                db32 = _token_grad_gemm(dy, z)  # (N_local, r)
        return (dx32.to(torch.bfloat16), da32, db32, None, None, *([None] * ctx.n_blocks))


class StrictRowLinear(torch.autograd.Function):
    """``y = bf16(tree_r(a_r·W_rᵀ) + B·bf16(tree_r(a_r·A_rᵀ)))`` for a row-parallel shard."""

    @staticmethod
    def forward(ctx, a_local, w_local, a32_local, b32, comm: TPComm, k_total: int):
        if a_local.dtype != torch.bfloat16 or w_local.dtype != torch.bfloat16:
            raise TypeError("strict row linear needs bf16 activations and weights")
        ids = local_chunk_ids(comm.rank, comm.world)
        if a_local.shape[1] * comm.world != k_total:
            raise RuntimeError("row shard width does not match k_total / tp")
        backend = gemm_backend()
        y32 = comm.all_gather_tree(backend.partial(a_local, w_local, ids))
        a = b = z = None
        if a32_local is not None:
            a = a32_local.to(torch.bfloat16)
            b = b32.to(torch.bfloat16)
            z = comm.all_gather_tree(backend.partial(a_local, a, ids)).to(torch.bfloat16)
            y32 = y32 + small_k_fp32(z, b)
        ctx.save_for_backward(a_local, w_local, a, b, z)
        return y32.to(torch.bfloat16)

    @staticmethod
    def backward(ctx, dy):
        a_local, w_local, a, b, z = ctx.saved_tensors
        dy = dy.contiguous()
        if dy.dtype != torch.bfloat16:
            dy = dy.to(torch.bfloat16)
        din32 = gemm_fp32(dy, w_local.t().contiguous())  # (M, K_local), 8 chunks over N
        da32 = db32 = None
        if a is not None:
            dz = gemm_fp32(dy, b.t().contiguous()).to(torch.bfloat16)  # (M, r), replicated
            din32 = din32 + small_k_fp32(dz, a.t().contiguous())
            if ctx.needs_input_grad[2]:
                da32 = _token_grad_gemm(dz, a_local)  # (r, K_local)
            if ctx.needs_input_grad[3]:
                db32 = _token_grad_gemm(dy, z)  # (N, r), identical on every rank
        return din32.to(torch.bfloat16), None, da32, db32, None, None


@dataclass(frozen=True)
class ProjectionShape:
    name: str
    n_total: int
    k_total: int
    mode: str  # "column" | "row"


def muse_glimmer_projection_shapes(
    *, hidden: int = 6656, q_width: int = 4096, kv_width: int = 256, ffn: int = 19968
) -> tuple[ProjectionShape, ...]:
    return (
        ProjectionShape("q_proj", q_width, hidden, "column"),
        ProjectionShape("k_proj", kv_width, hidden, "column"),
        ProjectionShape("v_proj", kv_width, hidden, "column"),
        ProjectionShape("gate_proj", q_width, hidden, "column"),
        ProjectionShape("o_proj", hidden, q_width, "row"),
        ProjectionShape("mlp.gate_proj", ffn, hidden, "column"),
        ProjectionShape("mlp.up_proj", ffn, hidden, "column"),
        ProjectionShape("mlp.down_proj", hidden, ffn, "row"),
    )


def _init_a_full(shape: ProjectionShape, rank: int, seed: int, layer: int) -> torch.Tensor:
    """Full ``A (r, K)`` drawn on CPU from a seed that depends only on (seed, layer, name)."""
    digest = hashlib.sha256(f"{seed}:{layer}:{shape.name}".encode()).digest()
    gen = torch.Generator(device="cpu").manual_seed(int.from_bytes(digest[:8], "little") % (2**63))
    bound = 1.0 / math.sqrt(shape.k_total)  # kaiming_uniform(a=sqrt(5)) bound for fan_in=K
    return (torch.rand(rank, shape.k_total, generator=gen, dtype=torch.float32) * 2.0 - 1.0) * bound


class LayerLoRA(nn.Module):
    """fp32 LoRA masters for one decoder layer, sharded like the base projections."""

    def __init__(
        self,
        layer_index: int,
        *,
        rank: int,
        shapes: Iterable[ProjectionShape],
        tp_rank: int,
        tp_world: int,
        seed: int,
        device: torch.device | str,
    ) -> None:
        super().__init__()
        self.layer_index = layer_index
        self.rank = rank
        self.tp_rank = tp_rank
        self.tp_world = tp_world
        self.shapes = {s.name: s for s in shapes}
        self.a = nn.ParameterDict()
        self.b = nn.ParameterDict()
        for s in self.shapes.values():
            key = s.name.replace(".", "__")
            a_full = _init_a_full(s, rank, seed, layer_index)
            if s.mode == "column":
                if s.n_total % tp_world:
                    raise ValueError(f"{s.name}: N={s.n_total} not divisible by tp={tp_world}")
                a_local = a_full
                b_local = torch.zeros(s.n_total // tp_world, rank, dtype=torch.float32)
            else:
                if s.k_total % tp_world:
                    raise ValueError(f"{s.name}: K={s.k_total} not divisible by tp={tp_world}")
                k_l = s.k_total // tp_world
                a_local = a_full[:, tp_rank * k_l : (tp_rank + 1) * k_l].contiguous()
                b_local = torch.zeros(s.n_total, rank, dtype=torch.float32)
            self.a[key] = nn.Parameter(a_local.to(device))
            self.b[key] = nn.Parameter(b_local.to(device))

    def pair(self, name: str) -> tuple[nn.Parameter, nn.Parameter]:
        key = name.replace(".", "__")
        return self.a[key], self.b[key]


def lora_state_bytes(state: Mapping[str, torch.Tensor]) -> bytes:
    """Canonical byte stream: sorted keys, ``key\\n`` + shape + bf16 little-endian payload."""
    stream = bytearray()
    for key in sorted(state):
        t = state[key]
        if t.dtype != torch.bfloat16:
            raise TypeError(f"{key}: LoRA export must be bf16, got {t.dtype}")
        stream += key.encode("utf-8") + b"\n"
        stream += (",".join(str(d) for d in t.shape) + "\n").encode("utf-8")
        stream += t.detach().cpu().contiguous().view(torch.int16).numpy().tobytes()
    return bytes(stream)


def lora_sha256(state: Mapping[str, torch.Tensor]) -> str:
    return hashlib.sha256(lora_state_bytes(state)).hexdigest()


def export_lora(layers: Iterable[LayerLoRA], comm: TPComm) -> tuple[dict[str, torch.Tensor], str]:
    """Gather full ``(r, K)`` / ``(N, r)`` bf16 tensors on every rank and hash them."""
    state: dict[str, torch.Tensor] = {}
    for lora in layers:
        for s in lora.shapes.values():
            a, b = lora.pair(s.name)
            a16 = a.detach().to(torch.bfloat16)
            b16 = b.detach().to(torch.bfloat16)
            if s.mode == "column":
                b_full = torch.cat(comm.all_gather(b16.contiguous()), dim=0)
                a_full = a16
            else:
                a_full = torch.cat(comm.all_gather(a16.contiguous()), dim=1)
                b_full = b16
            prefix = f"layers.{lora.layer_index}.{s.name}"
            state[f"{prefix}.lora_A"] = a_full.cpu().contiguous()
            state[f"{prefix}.lora_B"] = b_full.cpu().contiguous()
    return state, lora_sha256(state)


def import_lora(layers: Iterable[LayerLoRA], state: Mapping[str, torch.Tensor], sha256: str) -> None:
    """Verify the hash and load each rank's shard of the full tensors into the fp32 masters."""
    actual = lora_sha256(state)
    if actual != sha256:
        raise ValueError(f"LoRA state sha256 mismatch: expected {sha256}, got {actual}")
    with torch.no_grad():
        for lora in layers:
            for s in lora.shapes.values():
                a, b = lora.pair(s.name)
                prefix = f"layers.{lora.layer_index}.{s.name}"
                a_full = state[f"{prefix}.lora_A"].to(device=a.device, dtype=torch.float32)
                b_full = state[f"{prefix}.lora_B"].to(device=b.device, dtype=torch.float32)
                if s.mode == "column":
                    n_l = s.n_total // lora.tp_world
                    a.copy_(a_full)
                    b.copy_(b_full[lora.tp_rank * n_l : (lora.tp_rank + 1) * n_l])
                else:
                    k_l = s.k_total // lora.tp_world
                    a.copy_(a_full[:, lora.tp_rank * k_l : (lora.tp_rank + 1) * k_l])
                    b.copy_(b_full)


def lora_parameters(layers: Iterable[LayerLoRA]) -> list[nn.Parameter]:
    params: list[nn.Parameter] = []
    for lora in layers:
        params.extend(lora.parameters())
    return params


def lora_is_bitwise_noop(
    fn_with_lora: Any, fn_without_lora: Any, *inputs: torch.Tensor
) -> bool:
    """``torch.equal`` of two forwards; used to prove the zero-B initialisation is a no-op."""
    with torch.no_grad():
        return bool(torch.equal(fn_with_lora(*inputs), fn_without_lora(*inputs)))


__all__ = [
    "COLUMN_PROJECTIONS",
    "DEFAULT_LORA_RANK",
    "LayerLoRA",
    "PROJECTIONS",
    "ProjectionShape",
    "ROW_PROJECTIONS",
    "StrictColumnLinear",
    "StrictRowLinear",
    "export_lora",
    "import_lora",
    "lora_is_bitwise_noop",
    "lora_parameters",
    "lora_sha256",
    "lora_state_bytes",
    "muse_glimmer_projection_shapes",
]
