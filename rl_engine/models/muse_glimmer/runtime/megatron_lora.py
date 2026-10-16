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
LoRA fp32 input gradients are added once (``dx = bf16(dx_base + dx_lora)``).

LoRA weight gradients (reduction over TOKENS), :data:`LORA_GRAD_RULE`:

* per sequence ``s`` of the packed micro-batch (``segments``), ``g_s`` = the det_gemm dW rule
  over its ``T_s`` tokens: ``T_pad = roundup(T_s, 256)``, 8 chunks of ``T_pad/8`` rows, rows
  ``>= T_s`` masked to zero, fp32 tree (``fast_det_gemm(..., ragged_k=True)``, the rule
  ``ops.det_gemm.DW_CHUNK_RULE`` declares);
* the step gradient is the LEFT FOLD in fp32 over the request's sequences in sample order,
  ``((g_0 + g_1) + g_2) + ...``, written straight into the fp32 master's ``.grad`` (the op
  accumulates; it returns ``None`` to autograd for these inputs).

So the gradient bits do not depend on how the sequences are packed into micro-batches
(``micro_batch_tokens``) nor on the TP size; they depend on the sample order, which is part of
the data. Without ``segments`` (single-sequence callers, the vLLM rule check) the whole input is
one segment and the gradient is returned through autograd as usual.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import torch
from torch import nn

from rl_engine.models.muse_glimmer.e2e.sync import lora_sha256 as _canonical_lora_sha256
from rl_engine.models.muse_glimmer.runtime.strict_kernels import (
    CANONICAL_CHUNKS,
    TOKEN_PAD,
    TPComm,
    TreeAccumulator,
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
DEFAULT_LORA_RANK = 16
LORA_GRAD_RULE = (
    "per sequence: det_gemm dW rule over its tokens (T_pad = roundup(T, %d), 8 chunks of "
    "T_pad/8 rows, fp32 tree); step gradient = left fold in fp32 over the request's sequences "
    "in sample order, accumulated into the fp32 master .grad (packing- and TP-invariant)"
    % TOKEN_PAD
)

Segments = tuple[tuple[int, int], ...]


def _token_grad_gemm(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    """``lhs.T @ rhs`` reducing over tokens: ``lhs (T, P)``, ``rhs (T, Q)`` -> fp32 ``(P, Q)``.

    The det_gemm dW rule: ``T_pad = roundup(T, 256)``, 8 chunks of ``T_pad/8`` token rows
    (masked tail), fp32 chunk tree (``strict_kernels._GemmBackend.token_grad``).
    """
    return gemm_backend().token_grad(lhs, rhs)


def normalize_segments(segments: Any, tokens: int) -> Segments | None:
    """Validate ``((start, end), ...)`` covering ``[0, tokens)`` contiguously, in order."""
    if segments is None:
        return None
    out = tuple((int(s), int(e)) for s, e in segments)
    pos = 0
    for s, e in out:
        if s != pos or e <= s:
            raise ValueError(f"segments {out} do not tile [0, {tokens}) in order")
        pos = e
    if pos != tokens:
        raise ValueError(f"segments end at {pos}, the input has {tokens} tokens")
    return out


def _accumulate_token_grad(
    param: torch.Tensor, lhs: torch.Tensor, rhs: torch.Tensor, segments: Segments
) -> None:
    """``param.grad += g_s`` for each segment in order (``g_s`` = :func:`_token_grad_gemm`)."""
    with torch.no_grad():
        for start, end in segments:
            part = _token_grad_gemm(lhs[start:end], rhs[start:end])
            if part.shape != param.shape:
                raise RuntimeError(f"LoRA grad shape {tuple(part.shape)} != {tuple(param.shape)}")
            if param.grad is None:
                param.grad = part
            else:
                param.grad.add_(part)


class _StrictColumnLinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, a32, b32_local, comm: TPComm, n_total: int, segments, *blocks):
        if x.dtype != torch.bfloat16 or any(w.dtype != torch.bfloat16 for w in blocks):
            raise TypeError("strict column linear needs bf16 activations and weights")
        parts = [gemm_fp32(x, w) for w in blocks]
        y32 = parts[0] if len(parts) == 1 else torch.cat(parts, dim=1)
        del parts
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
        ctx.segments = segments
        ctx.params = (a32, b32_local) if segments is not None else None
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
        # Streaming canonical tree: same bits as combine_tree over the list, fewer live partials.
        base_tree = TreeAccumulator()
        lora_tree = TreeAccumulator() if a is not None else None
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
            base_tree.push(gemm_single_chunk_fp32(dy_c, w_c, c))
            del w_c
            if lora_tree is not None:
                lora_tree.push(
                    gemm_single_chunk_fp32(dy_c, b[start : start + n_chunk].t().contiguous(), c)
                )
        dx32 = comm.all_gather_tree(base_tree.result())
        del base_tree
        da32 = db32 = None
        if a is not None:
            dz = comm.all_gather_tree(lora_tree.result()).to(torch.bfloat16)
            dx32 = dx32 + small_k_fp32(dz, a.t().contiguous())
            if ctx.segments is None:
                if ctx.needs_input_grad[1]:
                    da32 = _token_grad_gemm(dz, x)  # (r, K)
                if ctx.needs_input_grad[2]:
                    db32 = _token_grad_gemm(dy, z)  # (N_local, r)
            else:
                a_param, b_param = ctx.params
                if ctx.needs_input_grad[1]:
                    _accumulate_token_grad(a_param, dz, x, ctx.segments)
                if ctx.needs_input_grad[2]:
                    _accumulate_token_grad(b_param, dy, z, ctx.segments)
        return (dx32.to(torch.bfloat16), da32, db32, None, None, None, *([None] * ctx.n_blocks))


class StrictColumnLinear:
    """``y_local = bf16(x·W_localᵀ + B_local·bf16(A·x))`` for a column-parallel shard.

    ``blocks`` are row-block views whose concatenation is this rank's shard of the weight in
    HF row order (e.g. the q rows of each local KV group inside the fused ``linear_qkv``). Each
    block is projected separately and the fp32 outputs concatenated: output columns are
    independent, so this is bit-identical to one GEMM over the concatenated rows. Every block
    must be a whole number of canonical ``N/8`` chunks. ``segments`` (the packed sequences'
    ``(start, end)`` token ranges) selects the packing-invariant LoRA gradient rule (module
    docstring); the positional signature is the one the vLLM rule check calls.
    """

    @staticmethod
    def apply(x, a32, b32_local, comm: TPComm, n_total: int, *blocks, segments=None):
        segs = normalize_segments(segments, x.shape[0])
        return _StrictColumnLinearFn.apply(x, a32, b32_local, comm, n_total, segs, *blocks)


class _StrictRowLinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a_local, w_local, a32_local, b32, comm: TPComm, k_total: int, segments):
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
        ctx.segments = segments
        ctx.params = (a32_local, b32) if segments is not None else None
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
            if ctx.segments is None:
                if ctx.needs_input_grad[2]:
                    da32 = _token_grad_gemm(dz, a_local)  # (r, K_local)
                if ctx.needs_input_grad[3]:
                    db32 = _token_grad_gemm(dy, z)  # (N, r), identical on every rank
            else:
                a_param, b_param = ctx.params
                if ctx.needs_input_grad[2]:
                    _accumulate_token_grad(a_param, dz, a_local, ctx.segments)
                if ctx.needs_input_grad[3]:
                    _accumulate_token_grad(b_param, dy, z, ctx.segments)
        return din32.to(torch.bfloat16), None, da32, db32, None, None, None


class StrictRowLinear:
    """``y = bf16(tree_r(a_r·W_rᵀ) + B·bf16(tree_r(a_r·A_rᵀ)))`` for a row-parallel shard.

    ``segments``: see :class:`StrictColumnLinear` (positional signature unchanged).
    """

    @staticmethod
    def apply(a_local, w_local, a32_local, b32, comm: TPComm, k_total: int, *, segments=None):
        segs = normalize_segments(segments, a_local.shape[0])
        return _StrictRowLinearFn.apply(a_local, w_local, a32_local, b32, comm, k_total, segs)


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


def _cpu_generator(seed: int, layer: int, name: str, which: str) -> torch.Generator:
    tag = f"{seed}:{layer}:{name}" + ("" if which == "A" else f":{which}")
    digest = hashlib.sha256(tag.encode()).digest()
    return torch.Generator(device="cpu").manual_seed(int.from_bytes(digest[:8], "little") % (2**63))


def _init_a_full(shape: ProjectionShape, rank: int, seed: int, layer: int) -> torch.Tensor:
    """Full ``A (r, K)`` drawn on CPU from a seed that depends only on (seed, layer, name)."""
    gen = _cpu_generator(seed, layer, shape.name, "A")
    bound = 1.0 / math.sqrt(shape.k_total)  # kaiming_uniform(a=sqrt(5)) bound for fan_in=K
    return (torch.rand(rank, shape.k_total, generator=gen, dtype=torch.float32) * 2.0 - 1.0) * bound


def _init_b_full(shape: ProjectionShape, rank: int, seed: int, layer: int, std: float) -> torch.Tensor:
    """Full ``B (N, r)``: zeros (the default, an exact no-op) or ``N(0, std)`` drawn on CPU.

    The random form exists for parity tests only (it makes every LoRA gradient non-trivial);
    it depends on (seed, layer, name) and never on the TP layout.
    """
    if std == 0.0:
        return torch.zeros(shape.n_total, rank, dtype=torch.float32)
    gen = _cpu_generator(seed, layer, shape.name, "B")
    return torch.randn(shape.n_total, rank, generator=gen, dtype=torch.float32) * float(std)


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
        b_init_std: float = 0.0,
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
            b_full = _init_b_full(s, rank, seed, layer_index, b_init_std)
            if s.mode == "column":
                if s.n_total % tp_world:
                    raise ValueError(f"{s.name}: N={s.n_total} not divisible by tp={tp_world}")
                n_l = s.n_total // tp_world
                a_local = a_full
                b_local = b_full[tp_rank * n_l : (tp_rank + 1) * n_l].contiguous()
            else:
                if s.k_total % tp_world:
                    raise ValueError(f"{s.name}: K={s.k_total} not divisible by tp={tp_world}")
                k_l = s.k_total // tp_world
                a_local = a_full[:, tp_rank * k_l : (tp_rank + 1) * k_l].contiguous()
                b_local = b_full
            self.a[key] = nn.Parameter(a_local.to(device))
            self.b[key] = nn.Parameter(b_local.to(device))

    def pair(self, name: str) -> tuple[nn.Parameter, nn.Parameter]:
        key = name.replace(".", "__")
        return self.a[key], self.b[key]

    @torch.no_grad()
    def reset_b(self, std: float, seed: int) -> None:
        """Re-draw every ``B`` from the TP-invariant generator (``std=0`` → zeros)."""
        for s in self.shapes.values():
            _, b = self.pair(s.name)
            b_full = _init_b_full(s, self.rank, seed, self.layer_index, std)
            if s.mode == "column":
                n_l = s.n_total // self.tp_world
                b_full = b_full[self.tp_rank * n_l : (self.tp_rank + 1) * n_l]
            b.copy_(b_full.to(device=b.device))


def lora_state_bytes(state: Mapping[str, torch.Tensor]) -> bytes:
    """Legacy byte stream (sorted keys, ``key\\n`` + shape + bf16 payload); NOT the sync hash.

    Kept for reference only: :func:`lora_sha256` is the E2E canonical
    ``muse_glimmer_lora_sha256_v1`` hash of ``rl_engine.models.muse_glimmer.e2e.sync``.
    """
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
    """The E2E canonical hash (schema ``muse_glimmer_lora_sha256_v1``, ``e2e/sync.py``)."""
    return _canonical_lora_sha256(state)


def gather_lora(
    layers: Iterable[LayerLoRA],
    comm: TPComm,
    *,
    source: str = "param",
    dtype: torch.dtype = torch.bfloat16,
    to_cpu: bool = True,
) -> dict[str, torch.Tensor]:
    """Full (TP-gathered) LoRA tensors keyed ``layers.{i}.{proj}.lora_{A,B}``.

    ``source="param"`` gathers the fp32 masters, ``source="grad"`` their gradients (zeros when a
    gradient is absent). Sharded tensors (column ``B`` by rows, row ``A`` by columns) are
    all-gathered in rank order (data movement only); replicated tensors are taken from this rank.
    Every rank must call this (collectives); every rank receives the same tensors.
    """
    if source not in ("param", "grad"):
        raise ValueError(f"source must be 'param' or 'grad', got {source!r}")
    state: dict[str, torch.Tensor] = {}
    for lora in layers:
        for s in lora.shapes.values():
            a, b = lora.pair(s.name)
            if source == "grad":
                a_t = a.grad if a.grad is not None else torch.zeros_like(a)
                b_t = b.grad if b.grad is not None else torch.zeros_like(b)
            else:
                a_t, b_t = a, b
            a_t = a_t.detach().to(dtype).contiguous()
            b_t = b_t.detach().to(dtype).contiguous()
            if s.mode == "column":
                b_full = torch.cat(comm.all_gather(b_t), dim=0) if comm.world > 1 else b_t
                a_full = a_t
            else:
                a_full = torch.cat(comm.all_gather(a_t), dim=1) if comm.world > 1 else a_t
                b_full = b_t
            prefix = f"layers.{lora.layer_index}.{s.name}"
            state[f"{prefix}.lora_A"] = (a_full.cpu() if to_cpu else a_full).contiguous()
            state[f"{prefix}.lora_B"] = (b_full.cpu() if to_cpu else b_full).contiguous()
    return state


def replicated_lora_grads_agree(layers: Iterable[LayerLoRA], comm: TPComm) -> tuple[int, int]:
    """``(tensors_checked, tensors_differing)`` for the TP-replicated LoRA gradients.

    Column ``A`` and row ``B`` are replicated on every rank and must receive bitwise-identical
    gradients (each rank's backward sees the same full upstream values). Checked by gathering
    each replicated gradient from all ranks and comparing with ``torch.equal``.
    """
    checked = differing = 0
    if comm.world == 1:
        return 0, 0
    for lora in layers:
        for s in lora.shapes.values():
            a, b = lora.pair(s.name)
            p = a if s.mode == "column" else b
            g = p.grad if p.grad is not None else torch.zeros_like(p)
            parts = comm.all_gather(g.detach().float().contiguous())
            checked += 1
            if not all(torch.equal(parts[0], q) for q in parts[1:]):
                differing += 1
    return checked, differing


def export_lora(layers: Iterable[LayerLoRA], comm: TPComm) -> tuple[dict[str, torch.Tensor], str]:
    """Gather full ``(r, K)`` / ``(N, r)`` bf16 tensors on every rank and hash them (sync v1)."""
    state = gather_lora(layers, comm, source="param", dtype=torch.bfloat16)
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
    "LORA_GRAD_RULE",
    "LayerLoRA",
    "PROJECTIONS",
    "ProjectionShape",
    "ROW_PROJECTIONS",
    "StrictColumnLinear",
    "StrictRowLinear",
    "export_lora",
    "gather_lora",
    "import_lora",
    "lora_is_bitwise_noop",
    "lora_parameters",
    "lora_sha256",
    "lora_state_bytes",
    "muse_glimmer_projection_shapes",
    "normalize_segments",
    "replicated_lora_grads_agree",
]
