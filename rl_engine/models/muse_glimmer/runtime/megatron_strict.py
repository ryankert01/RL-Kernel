# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Strict binding into Megatron Bridge's native Muse Glimmer ``HybridModel``.

The binding does not patch Megatron classes. It reads the Bridge model's parameters (fused
``linear_qkv``, fused ``linear_fc1``, row-parallel ``linear_proj``/``linear_fc2``, TE norms,
vocab-parallel embedding and output layer) and runs the 52-layer text model through the strict
operator set in the contract's "Model assembly" order, with unmerged strict LoRA and autograd.

Bridge module → strict op (per layer ``L = decoder.layers[i]``):

===================================  =============================================================
Bridge parameter / module            strict path
===================================  =============================================================
``embedding.word_embeddings``        O1 ``normalized_embedding`` (full table on every rank)
``L.input_layernorm`` (TE, 1+w)      O2 ``centered_rmsnorm`` eps=1e-5 (fused into previous layer)
``L.self_attention.linear_qkv``      O3 FOUR column GEMMs on views of the fused weight (+LoRA)
``L.self_attention.q/k_layernorm``   O4 ``qknorm_scaled_rope`` (3.87 on q, RoPE on sliding layers)
``L.self_attention.core_attention``  O5 ``gated_attention`` (gate inside, window from config)
``L.self_attention.linear_proj``     O3 row-parallel partial + all-gather + tree (+LoRA)
``L.self_attention.post_layernorm``  O2 fused post-norm/residual/pre-norm, eps 1e-8 / 1e-5
``L.mlp.linear_fc1`` ([gate; up])    O3 two column GEMMs on the two halves (+LoRA)
activation                           O6 ``silu_mul`` (fp32, one rounding)
``L.mlp.linear_fc2``                 O3 row-parallel partial + all-gather + tree (+LoRA)
``L.mlp.post_layernorm``             O2 fused with the next layer's input norm
``decoder.final_norm`` (w-1 in bf16) O2 ``weighted_rmsnorm`` with the HF tensor (reloaded)
``output_layer`` + softcap + CE      O7 canonical vocab tiles, tile stats, tree, logp
===================================  =============================================================

Fused ``linear_qkv`` row layout (Bridge ``MuseGlimmerQKVGMapping`` + MCore ``merge_qkv_weights``
with ``attention_output_gate``), per local KV group ``g`` (4352 rows each, ``D=128``)::

    [ q heads 16g..16g+15 (2048 rows) | gate heads 16g..16g+15 (2048) | k head g (128) | v head g (128) ]

so HF ``q_proj`` = concat over g of the first block, ``gate_proj`` the second, ``k_proj``/``v_proj``
the third/fourth. At TP ``t <= 2`` rank ``r``'s MCore shard is groups ``[2r/t, 2(r+1)/t)``
and its canonical q/gate/k/v row chunks are views of it. At TP 4/8 MCore splits the fused rows
evenly (no clean head partition); the binding re-shards them to canonical chunks at bind time
(see ``MuseGlimmerStrictBinding._qkv_views``). Attention runs on all 32/2 heads on every rank
(head shards all-gathered), so it is the TP1 computation at every TP.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from rl_engine.models.muse_glimmer.runtime import strict_kernels as sk
from rl_engine.models.muse_glimmer.runtime.megatron_lora import (
    DEFAULT_LORA_RANK,
    LayerLoRA,
    StrictColumnLinear,
    StrictRowLinear,
    export_lora,
    lora_parameters,
    muse_glimmer_projection_shapes,
)
from rl_engine.models.muse_glimmer.workload import (
    FULL_ATTENTION,
    OFFICIAL_FINGERPRINT,
    OFFICIAL_PARALLEL_LAYOUT,
    MuseGlimmerWorkloadError,
    expected_layer_types,
)

NATIVE_BRIDGE_MODEL_CLASS = "MuseGlimmerModel"
NATIVE_BRIDGE_MODULE = "megatron.bridge.models.muse_glimmer.modeling_muse_glimmer"
_FINAL_NORM_HF_KEY = "model.language_model.norm.weight"
SUPPORTED_TP = (1, 2, 4, 8)


class StrictBindingError(MuseGlimmerWorkloadError):
    """Raised when the Bridge model cannot be bound strictly (fail closed)."""


@dataclass
class LayerWeights:
    """Views into one Bridge decoder layer's parameters, in HF row order."""

    index: int
    sliding: bool
    q_blocks: tuple[torch.Tensor, ...]
    gate_blocks: tuple[torch.Tensor, ...]
    k_blocks: tuple[torch.Tensor, ...]
    v_blocks: tuple[torch.Tensor, ...]
    o_local: torch.Tensor
    fc1_gate_local: torch.Tensor
    fc1_up_local: torch.Tensor
    down_local: torch.Tensor
    w_input: torch.Tensor
    w_post_attn: torch.Tensor
    w_pre_ffn: torch.Tensor
    w_post_ffn: torch.Tensor


@dataclass
class ModelDims:
    hidden: int
    layers: int
    heads: int
    kv_heads: int
    head_dim: int
    ffn: int
    vocab: int
    padded_vocab: int
    window: int
    rms_eps: float
    post_eps: float
    qk_scale: float
    multiplier: float
    softcap: float
    rope_theta: float

    @property
    def q_width(self) -> int:
        return self.heads * self.head_dim

    @property
    def kv_width(self) -> int:
        return self.kv_heads * self.head_dim

    @property
    def tile_rows(self) -> int:
        return self.padded_vocab // sk.CANONICAL_CHUNKS


def _unwrap(model: Any) -> Any:
    inner = model
    seen = 0
    while hasattr(inner, "module") and seen < 4:
        inner = inner.module
        seen += 1
    return inner


def _tp_group(model: Any) -> Any | None:
    pg = getattr(model, "pg_collection", None)
    group = getattr(pg, "tp", None)
    if group is not None:
        return group
    try:
        from megatron.core import parallel_state

        return parallel_state.get_tensor_model_parallel_group()
    except Exception:  # noqa: BLE001
        return None


def _check_native_model(model: Any) -> None:
    cls = type(model)
    if cls.__name__ != NATIVE_BRIDGE_MODEL_CLASS or cls.__module__ != NATIVE_BRIDGE_MODULE:
        raise StrictBindingError(
            f"model class {cls.__module__}.{cls.__name__} is not Bridge's native "
            f"{NATIVE_BRIDGE_MODULE}.{NATIVE_BRIDGE_MODEL_CLASS}; refusing strict binding"
        )


def _check_config(cfg: Any, dims: ModelDims, tp_world: int) -> None:
    exp_types = expected_layer_types(dims.layers)
    want_window = [t != FULL_ATTENTION for t in exp_types]
    want_nope = [t == FULL_ATTENTION for t in exp_types]
    checks: list[tuple[str, Any, Any]] = [
        ("num_layers", cfg.num_layers, dims.layers),
        ("hidden_size", cfg.hidden_size, dims.hidden),
        ("num_attention_heads", cfg.num_attention_heads, dims.heads),
        ("num_query_groups", cfg.num_query_groups, dims.kv_heads),
        ("kv_channels", cfg.kv_channels, dims.head_dim),
        ("ffn_hidden_size", cfg.ffn_hidden_size, dims.ffn),
        ("window_size", tuple(cfg.window_size or ()), (dims.window - 1, 0)),
        ("window_attn_skip_freq", list(cfg.window_attn_skip_freq or []), want_window),
        ("no_rope_freq", list(cfg.no_rope_freq or []), want_nope),
        ("layernorm_zero_centered_gamma", bool(cfg.layernorm_zero_centered_gamma), True),
        ("layernorm_epsilon", float(cfg.layernorm_epsilon), dims.rms_eps),
        ("post_norm_epsilon", float(getattr(cfg, "post_norm_epsilon", -1.0)), dims.post_eps),
        ("output_multiplier", float(getattr(cfg, "output_multiplier", -1.0)), dims.multiplier),
        ("final_logit_softcapping", float(getattr(cfg, "final_logit_softcapping", -1.0)), dims.softcap),
        ("attention_output_gate", bool(cfg.attention_output_gate), True),
        ("qk_layernorm", bool(cfg.qk_layernorm), True),
        ("gated_linear_unit", bool(cfg.gated_linear_unit), True),
        ("add_bias_linear", bool(cfg.add_bias_linear), False),
        ("add_qkv_bias", bool(cfg.add_qkv_bias), False),
        ("rotary_interleaved", bool(cfg.rotary_interleaved), False),
        ("context_parallel_size", int(cfg.context_parallel_size or 1), 1),
        ("sequence_parallel", bool(cfg.sequence_parallel), False),
        ("tensor_model_parallel_size", int(cfg.tensor_model_parallel_size), tp_world),
        ("params_dtype", cfg.params_dtype, torch.bfloat16),
    ]
    for name, got, want in checks:
        if got != want:
            raise StrictBindingError(f"Bridge config {name}={got!r} is not the pinned {want!r}")
    if not math.isclose(float(cfg.softmax_scale), dims.qk_scale / math.sqrt(dims.head_dim), rel_tol=1e-12):
        raise StrictBindingError(
            f"softmax_scale {cfg.softmax_scale} is not qk_scale/sqrt(head_dim) "
            f"({dims.qk_scale}/sqrt({dims.head_dim}))"
        )
    if int(cfg.context_parallel_size or 1) > 1 or getattr(cfg, "attention_cp_layout", "zigzag") != "zigzag":
        # CP>1 is rejected above; a non-zigzag layout with CP=1 is irrelevant but recorded.
        pass
    if tp_world not in SUPPORTED_TP:
        raise StrictBindingError(
            f"tp={tp_world}: the strict binding supports TP {SUPPORTED_TP} (8 canonical chunks)"
        )


def _dims_from_manifest() -> ModelDims:
    f = OFFICIAL_FINGERPRINT
    return ModelDims(
        hidden=int(f["hidden_size"]),
        layers=int(f["num_hidden_layers"]),
        heads=int(f["num_attention_heads"]),
        kv_heads=int(f["num_key_value_heads"]),
        head_dim=int(f["head_dim"]),
        ffn=int(f["intermediate_size"]),
        vocab=int(f["vocab_size"]),
        padded_vocab=int(OFFICIAL_PARALLEL_LAYOUT["padded_vocab_size"]),
        window=int(f["sliding_window"]),
        rms_eps=float(f["rms_norm_eps"]),
        post_eps=float(f["post_norm_eps"]),
        qk_scale=float(f["qk_scale_factor"]),
        multiplier=float(f["output_multiplier"]),
        softcap=float(f["final_logit_softcapping"]),
        rope_theta=float(f["rope_parameters"]["rope_theta"]),
    )


def fused_qkv_blocks(
    qkv: torch.Tensor, heads_per_group: int, head_dim: int, groups_local: int
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...], tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    """Views ``(q_blocks, gate_blocks, k_blocks, v_blocks)`` of a Bridge fused ``linear_qkv`` shard.

    Per local KV group ``g`` the fused rows are
    ``[q heads (hpg*D) | gate heads (hpg*D) | k head (D) | v head (D)]``; concatenating a block
    list over ``g`` gives the HF-ordered ``q_proj``/``gate_proj``/``k_proj``/``v_proj`` rows of the
    rank's groups (no copy).
    """
    per_group = (2 * heads_per_group + 2) * head_dim
    qw = heads_per_group * head_dim
    if qkv.ndim != 2 or qkv.shape[0] != groups_local * per_group:
        raise StrictBindingError(f"fused qkv shape {tuple(qkv.shape)} is not {groups_local} groups")
    qb, gb, kb, vb = [], [], [], []
    for g in range(groups_local):
        base = g * per_group
        qb.append(qkv[base : base + qw])
        gb.append(qkv[base + qw : base + 2 * qw])
        kb.append(qkv[base + 2 * qw : base + 2 * qw + head_dim])
        vb.append(qkv[base + 2 * qw + head_dim : base + per_group])
    return tuple(qb), tuple(gb), tuple(kb), tuple(vb)


def fused_qkv_to_hf(
    qkv_full: torch.Tensor, heads: int, kv_heads: int, head_dim: int
) -> dict[str, torch.Tensor]:
    """Un-fuse a FULL (all groups) Bridge ``linear_qkv`` weight into HF ``q/k/v/gate_proj``."""
    hpg = heads // kv_heads
    qb, gb, kb, vb = fused_qkv_blocks(qkv_full, hpg, head_dim, kv_heads)
    return {
        "q_proj": torch.cat(qb, dim=0),
        "k_proj": torch.cat(kb, dim=0),
        "v_proj": torch.cat(vb, dim=0),
        "gate_proj": torch.cat(gb, dim=0),
    }


def _load_hf_tensor(hf_path: str | Path, name: str) -> torch.Tensor:
    from safetensors import safe_open

    root = Path(hf_path)
    index = json.loads((root / "model.safetensors.index.json").read_text(encoding="utf-8"))
    shard = index["weight_map"][name]
    with safe_open(str(root / shard), "pt", device="cpu") as fh:
        return fh.get_tensor(name)


class MuseGlimmerStrictBinding:
    """State of the strict binding attached to a Bridge ``MuseGlimmerModel``."""

    def __init__(
        self,
        model: Any,
        *,
        hf_path: str | Path | None,
        lora_rank: int,
        seed: int,
        checkpoint_layers: bool,
        drop_vision: bool,
        lora_b_init_std: float = 0.0,
    ) -> None:
        self.model = model
        self.dims = _dims_from_manifest()
        self.ops = sk.resolve_ops()
        self.flags = sk.set_deterministic_torch()
        group = _tp_group(model)
        self.comm = sk.TPComm(group)
        self.tp_world = self.comm.world
        self.tp_rank = self.comm.rank
        _check_native_model(model)
        _check_config(model.config, self.dims, self.tp_world)
        self.device = model.decoder.layers[0].input_layernorm.weight.device
        self.checkpoint_layers = checkpoint_layers
        self.lora_rank = int(lora_rank)
        self.seed = int(seed)
        self._rope_cache: dict[tuple[int, ...], tuple[torch.Tensor, torch.Tensor]] = {}
        self.notes: list[str] = []

        for p in model.parameters():
            p.requires_grad_(False)
        gib = 2**30
        self.memory_gib: dict[str, float] = {"bound_start": torch.cuda.memory_allocated() / gib}
        if drop_vision:
            for name in ("vision_tower", "vision_adapter", "vision_projection"):
                if hasattr(model, name):
                    delattr(model, name)
                    self.notes.append(f"dropped {name} (text-only trainer)")
            torch.cuda.empty_cache()
        self.memory_gib["after_vision_drop"] = torch.cuda.memory_allocated() / gib

        self.layers = [self._layer_weights(i) for i in range(self.dims.layers)]
        self.embed_full = self._gather_embedding()
        self.lm_tiles, self.lm_tile_ids = self._gather_lm_head_tiles()
        self.final_norm_weight = self._final_norm_weight(hf_path)
        self.memory_gib["after_weight_views"] = torch.cuda.memory_allocated() / gib
        self.memory_gib["model_params"] = sum(
            p.numel() * p.element_size() for p in model.parameters()
        ) / gib
        shapes = muse_glimmer_projection_shapes(
            hidden=self.dims.hidden,
            q_width=self.dims.q_width,
            kv_width=self.dims.kv_width,
            ffn=self.dims.ffn,
        )
        self.lora: list[LayerLoRA] | None = None
        if self.lora_rank > 0:
            self.lora = [
                LayerLoRA(
                    i,
                    rank=self.lora_rank,
                    shapes=shapes,
                    tp_rank=self.tp_rank,
                    tp_world=self.tp_world,
                    seed=self.seed,
                    device=self.device,
                    b_init_std=lora_b_init_std,
                )
                for i in range(self.dims.layers)
            ]
        self.lora_enabled = self.lora is not None

    # ------------------------------------------------------------------ weights
    def _layer_weights(self, i: int) -> LayerWeights:
        d = self.dims
        layer = self.model.decoder.layers[i]
        attn = layer.self_attention
        qb, gb, kb, vb = self._qkv_views(i, attn.linear_qkv.weight)
        fc1 = layer.mlp.linear_fc1.weight
        ffn_local = d.ffn // self.tp_world
        if tuple(fc1.shape) != (2 * ffn_local, d.hidden):
            raise StrictBindingError(f"layer {i}: linear_fc1 shape {tuple(fc1.shape)} unexpected")
        o = attn.linear_proj.weight
        down = layer.mlp.linear_fc2.weight
        if tuple(o.shape) != (d.hidden, d.q_width // self.tp_world):
            raise StrictBindingError(f"layer {i}: linear_proj shape {tuple(o.shape)} unexpected")
        if tuple(down.shape) != (d.hidden, ffn_local):
            raise StrictBindingError(f"layer {i}: linear_fc2 shape {tuple(down.shape)} unexpected")
        for norm, eps in (
            (layer.input_layernorm, d.rms_eps),
            (attn.post_layernorm, d.post_eps),
            (layer.pre_mlp_layernorm, d.rms_eps),
            (layer.mlp.post_layernorm, d.post_eps),
        ):
            if not math.isclose(float(norm.eps), eps, rel_tol=0, abs_tol=0):
                raise StrictBindingError(f"layer {i}: norm eps {norm.eps} != {eps}")
            if not bool(getattr(norm, "zero_centered_gamma", True)):
                raise StrictBindingError(f"layer {i}: decoder norm is not zero-centred")
        return LayerWeights(
            index=i,
            sliding=expected_layer_types(d.layers)[i] != FULL_ATTENTION,
            q_blocks=tuple(qb),
            gate_blocks=tuple(gb),
            k_blocks=tuple(kb),
            v_blocks=tuple(vb),
            o_local=o,
            fc1_gate_local=fc1[:ffn_local],
            fc1_up_local=fc1[ffn_local:],
            down_local=down,
            w_input=layer.input_layernorm.weight,
            w_post_attn=attn.post_layernorm.weight,
            w_pre_ffn=layer.pre_mlp_layernorm.weight,
            w_post_ffn=layer.mlp.post_layernorm.weight,
        )

    def _qkv_views(self, i: int, qkv: torch.Tensor):
        """This rank's CANONICAL row blocks of q/gate/k/v (HF row order) from the fused shard.

        Canonical ownership (contract O3 column rule): at TP ``t`` rank ``r`` owns output-row
        chunks ``[8r/t, 8(r+1)/t)`` of EACH projection (q/gate: 512-row chunks = 4 heads,
        k/v: 32-row chunks). For ``t <= num_kv_heads`` MCore's fused shard holds whole KV groups
        and those chunks are exactly its blocks (zero-copy views). For ``t > num_kv_heads``
        MCore (attention.py, ``num_query_groups < world_size``) splits the FUSED rows evenly, so a
        shard holds e.g. "q heads 0-15 + 1 gate head" at TP4; the binding all-gathers the fused
        weight once at bind time (data movement only), un-fuses it and keeps contiguous copies of
        its canonical rows (same bytes as the checkpoint; memory equal to the MCore shard).
        """
        d = self.dims
        t = self.tp_world
        heads_per_group = d.heads // d.kv_heads
        per_group = (2 * heads_per_group + 2) * d.head_dim
        rows_total = d.kv_heads * per_group
        if rows_total % t or tuple(qkv.shape) != (rows_total // t, d.hidden):
            raise StrictBindingError(f"layer {i}: linear_qkv shape {tuple(qkv.shape)} unexpected")
        if t <= d.kv_heads:
            return fused_qkv_blocks(qkv, heads_per_group, d.head_dim, d.kv_heads // t)
        full = torch.cat(self.comm.all_gather(qkv.detach().contiguous()), dim=0)
        parts = fused_qkv_to_hf(full, d.heads, d.kv_heads, d.head_dim)
        r = self.tp_rank
        out = []
        for name in ("q_proj", "gate_proj", "k_proj", "v_proj"):
            w = parts[name]
            n_l = w.shape[0] // t
            out.append((w[r * n_l : (r + 1) * n_l].contiguous(),))
        del full, parts
        if i == 0:
            self.notes.append(
                f"tp={t} > kv_heads={d.kv_heads}: fused linear_qkv re-sharded to canonical "
                "per-projection row chunks at bind time (all-gather, data movement only)"
            )
        q_b, g_b, k_b, v_b = out
        return q_b, g_b, k_b, v_b

    def _gather_embedding(self) -> torch.Tensor:
        emb = self.model.embedding.word_embeddings
        w = emb.weight.detach()
        if self.tp_world == 1:
            if tuple(w.shape) != (self.dims.vocab, self.dims.hidden):
                raise StrictBindingError(f"embedding shape {tuple(w.shape)} unexpected")
            return w
        shards = self.comm.all_gather(w.contiguous())
        full = torch.cat(shards, dim=0)
        if tuple(full.shape) != (self.dims.vocab, self.dims.hidden):
            raise StrictBindingError(f"gathered embedding shape {tuple(full.shape)} unexpected")
        self.notes.append("embedding table gathered on every rank (TP>1)")
        return full

    def _gather_lm_head_tiles(self) -> tuple[torch.Tensor, list[int]]:
        d = self.dims
        out = self.model.output_layer
        w = out.weight.detach()
        if getattr(out, "bias", None) is not None:
            raise StrictBindingError("output layer has a bias; Muse Glimmer lm_head is bias-free")
        ids = sk.local_chunk_ids(self.tp_rank, self.tp_world)
        pad_rows = d.padded_vocab - d.vocab
        if self.tp_world == 1:
            if tuple(w.shape) != (d.vocab, d.hidden):
                raise StrictBindingError(f"output layer shape {tuple(w.shape)} unexpected")
            padded = torch.nn.functional.pad(w, (0, 0, 0, pad_rows)).contiguous()
            # Share storage: the native output layer keeps the first `vocab` rows as a view.
            out.weight = nn.Parameter(padded[: d.vocab], requires_grad=False)
            self.notes.append("output_layer.weight re-pointed to a view of the padded tile tensor")
            tiles = padded
        else:
            shards = self.comm.all_gather(w.contiguous())
            full = torch.cat(shards, dim=0)
            if tuple(full.shape) != (d.vocab, d.hidden):
                raise StrictBindingError(f"gathered lm_head shape {tuple(full.shape)} unexpected")
            padded = torch.nn.functional.pad(full, (0, 0, 0, pad_rows))
            rows = d.tile_rows
            tiles = padded[ids[0] * rows : (ids[-1] + 1) * rows].contiguous()
            del full, padded, shards
            torch.cuda.empty_cache()
            self.notes.append(f"lm_head redistributed to canonical tiles {ids} on this rank")
        return tiles, ids

    def _final_norm_weight(self, hf_path: str | Path | None) -> torch.Tensor:
        meg = self.model.decoder.final_norm.weight.detach()
        if hf_path is None:
            raise StrictBindingError(
                "hf_path is required: Bridge stores final_norm as (w-1) in bf16, which does not "
                "round-trip to the HF tensor; the strict path reloads model.language_model.norm.weight"
            )
        hf = _load_hf_tensor(hf_path, _FINAL_NORM_HF_KEY).to(device=meg.device)
        if hf.dtype != torch.bfloat16 or tuple(hf.shape) != (self.dims.hidden,):
            raise StrictBindingError("HF final norm tensor has an unexpected dtype/shape")
        recon = (meg.float() + 1.0).to(torch.bfloat16)
        diff = int((recon != hf).sum().item())
        self.final_norm_roundtrip_mismatches = diff
        if diff:
            self.notes.append(
                f"final_norm: Bridge (w-1)+1 differs from HF at {diff}/{hf.numel()} elements; "
                "strict path uses the HF tensor"
            )
        return hf

    # ------------------------------------------------------------------ helpers
    def _lora_pair(self, layer: int, name: str):
        if self.lora is None or not self.lora_enabled:
            return None, None
        return self.lora[layer].pair(name)

    def _col(self, x: torch.Tensor, blocks: Sequence[torch.Tensor], layer: int, name: str, n_total: int):
        a, b = self._lora_pair(layer, name)
        return StrictColumnLinear.apply(x, a, b, self.comm, n_total, *blocks)

    def _row(self, x: torch.Tensor, w_local: torch.Tensor, layer: int, name: str, k_total: int):
        a, b = self._lora_pair(layer, name)
        return StrictRowLinear.apply(x, w_local, a, b, self.comm, k_total)

    def rope(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        key = tuple(positions.tolist()) if positions.numel() <= 4096 else None
        if key is not None and key in self._rope_cache:
            return self._rope_cache[key]
        cos, sin = self.ops.rope_table(
            positions, theta=self.dims.rope_theta, dim=self.dims.head_dim, device=self.device
        )
        if key is not None:
            self._rope_cache[key] = (cos, sin)
        return cos, sin

    # ------------------------------------------------------------------ forward
    def _layer(
        self,
        i: int,
        resid: torch.Tensor,
        x_n: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cu_seqlens: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        d = self.dims
        lw = self.layers[i]
        t = x_n.shape[0]
        q = self._col(x_n, lw.q_blocks, i, "q_proj", d.q_width)
        k = self._col(x_n, lw.k_blocks, i, "k_proj", d.kv_width)
        v = self._col(x_n, lw.v_blocks, i, "v_proj", d.kv_width)
        g = self._col(x_n, lw.gate_blocks, i, "gate_proj", d.q_width)
        if self.tp_world > 1:
            # The strict O4/O5 kernels take the full 32 Q / 2 KV heads: gather the canonical
            # column shards (data movement only), compute attention redundantly (= TP1 bits and
            # TP1 dQ/dK/dV on every rank), keep the local head columns.
            q = GatherHeads.apply(q, self.comm)
            k = GatherHeads.apply(k, self.comm)
            v = GatherHeads.apply(v, self.comm)
            g = GatherHeads.apply(g, self.comm)
        q3 = q.view(t, d.heads, d.head_dim)
        k3 = k.view(t, d.kv_heads, d.head_dim)
        if lw.sliding:
            qh, kh = self.ops.qknorm_scaled_rope(q3, k3, cos, sin, eps=d.rms_eps, qk_scale=d.qk_scale)
            window: int | None = d.window
        else:
            qh, kh = self.ops.qknorm_scaled_rope(q3, k3, None, None, eps=d.rms_eps, qk_scale=d.qk_scale)
            window = None
        y = self.ops.gated_attention(
            qh, kh, v.view(t, d.kv_heads, d.head_dim), g, cu_seqlens, cu_seqlens, positions, window=window
        )
        y_local = SliceLocalHeads.apply(y.reshape(t, d.q_width), self.comm)
        o = self._row(y_local, lw.o_local, i, "o_proj", d.q_width)
        resid, x_f = self.ops.fused_post_norm_residual_pre_norm(
            o, resid, lw.w_post_attn, d.post_eps, lw.w_pre_ffn, d.rms_eps, pre_form="centered"
        )
        gm = self._col(x_f, (lw.fc1_gate_local,), i, "mlp.gate_proj", d.ffn)
        up = self._col(x_f, (lw.fc1_up_local,), i, "mlp.up_proj", d.ffn)
        h = self.ops.silu_mul(gm, up)
        f = self._row(h, lw.down_local, i, "mlp.down_proj", d.ffn)
        if i + 1 < d.layers:
            resid, x_next = self.ops.fused_post_norm_residual_pre_norm(
                f, resid, lw.w_post_ffn, d.post_eps, self.layers[i + 1].w_input, d.rms_eps,
                pre_form="centered",
            )
        else:
            resid, x_next = self.ops.fused_post_norm_residual_pre_norm(
                f, resid, lw.w_post_ffn, d.post_eps, self.final_norm_weight, d.rms_eps,
                pre_form="weighted",
            )
        return resid, x_next

    def hidden_states(
        self, tokens: torch.Tensor, cu_seqlens: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        """Final-normed hidden states ``(T, H)`` bf16 for packed tokens."""
        d = self.dims
        resid = self.ops.normalized_embedding(self.embed_full, tokens, eps=d.rms_eps)
        x_n = self.ops.centered_rmsnorm(resid, self.layers[0].w_input, eps=d.rms_eps)
        cos, sin = self.rope(positions)
        cu = cu_seqlens.to(device=self.device, dtype=torch.int32)
        pos = positions.to(device=self.device, dtype=torch.int32)
        for i in range(d.layers):
            if self.checkpoint_layers and torch.is_grad_enabled():
                resid, x_n = torch_checkpoint(
                    self._layer, i, resid, x_n, cos, sin, cu, pos, use_reentrant=False
                )
            else:
                resid, x_n = self._layer(i, resid, x_n, cos, sin, cu, pos)
        return x_n  # final norm output (weighted form)

    def logprobs(self, h: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """fp32 ``logp[row] = softcap(z)[target] - logsumexp(softcap(z))`` over the real vocab."""
        d = self.dims
        return StrictLMHeadLogprob.apply(
            h,
            targets.to(device=h.device, dtype=torch.int64),
            self.lm_tiles,
            tuple(self.lm_tile_ids),
            self.comm,
            d.vocab,
            d.tile_rows,
            d.multiplier,
            d.softcap,
        )

    def forward_packed(
        self,
        tokens: torch.Tensor,
        cu_seqlens: torch.Tensor,
        positions: torch.Tensor,
        select_rows: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        h = self.hidden_states(tokens, cu_seqlens, positions)
        return self.logprobs(h.index_select(0, select_rows.to(h.device)), targets)

    # ------------------------------------------------------------------ provenance
    def provenance(self) -> dict[str, Any]:
        d = self.dims
        return {
            "binding": "megatron_bridge_native_hybridmodel_strict",
            "model_class": f"{NATIVE_BRIDGE_MODULE}.{NATIVE_BRIDGE_MODEL_CLASS}",
            "tp": self.tp_world,
            "tp_collective": self.comm.backend,
            "ops": self.ops.provenance(),
            "torch_flags": self.flags,
            "lora": {
                "rank": self.lora_rank,
                "form": "unmerged: y = bf16(acc32(W x) + sum_k B[:,k] * bf16(A x)[k])",
                "enabled": self.lora_enabled,
                "seed": self.seed,
            },
            "checkpoint_layers": self.checkpoint_layers,
            "canonical_chunks": sk.CANONICAL_CHUNKS,
            "vocab_tiles": {"rows": d.tile_rows, "local": self.lm_tile_ids, "padded_vocab": d.padded_vocab},
            "final_norm_roundtrip_mismatches": getattr(self, "final_norm_roundtrip_mismatches", None),
            "memory_gib": dict(self.memory_gib),
            "notes": list(self.notes),
        }


class GatherHeads(torch.autograd.Function):
    """TP all-gather of a head-sharded ``(T, N_local)`` activation along dim 1 (rank order).

    Forward is data movement only; backward keeps this rank's slice of the full gradient.
    """

    @staticmethod
    def forward(ctx, x_local, comm):
        ctx.comm = comm
        ctx.n_local = x_local.shape[1]
        if comm.world == 1:
            return x_local
        return torch.cat(comm.all_gather(x_local.contiguous()), dim=1)

    @staticmethod
    def backward(ctx, grad_full):
        comm = ctx.comm
        if comm.world == 1:
            return grad_full, None
        start = comm.rank * ctx.n_local
        return grad_full[:, start : start + ctx.n_local].contiguous(), None


class SliceLocalHeads(torch.autograd.Function):
    """Keep this rank's ``N/t`` columns of a replicated ``(T, N)`` activation.

    Backward all-gathers the local gradients so every rank sees the full upstream gradient
    (the replicated attention backward then produces identical dK/dV on every rank).
    """

    @staticmethod
    def forward(ctx, y_full, comm):
        ctx.comm = comm
        if comm.world == 1:
            return y_full
        n_local = y_full.shape[1] // comm.world
        start = comm.rank * n_local
        return y_full[:, start : start + n_local].contiguous()

    @staticmethod
    def backward(ctx, grad_local):
        comm = ctx.comm
        if comm.world == 1:
            return grad_local, None
        return torch.cat(comm.all_gather(grad_local.contiguous()), dim=1), None


class StrictLMHeadLogprob(torch.autograd.Function):
    """O7 at TP ``t``: this rank's ``8/t`` canonical vocab tiles → per-row ``(M_t, S_t)`` and the
    target's soft-capped logit → rank-ordered all-gather (data movement) → fixed tree over the 8
    tiles → ``logp = zhat[y] - (M + log S)``.  Backward recomputes each local tile's logits,
    forms ``dz`` with the tile kernel, and reduces ``dh`` with the per-tile 8-chunk GEMM followed
    by the same tile tree (bit-identical to the single-rank sibling op at TP1).

    ``tiles`` is ``(len(tile_ids) * tile_rows, H)`` bf16 holding the rank's consecutive tiles.
    """

    @staticmethod
    def forward(ctx, h, targets, tiles, tile_ids, comm, vocab, tile_rows, multiplier, softcap):
        ops = sk.lm_head_tile_ops()
        m = h.shape[0]
        stats = torch.empty(len(tile_ids), 2, m, dtype=torch.float32, device=h.device)
        zy = torch.zeros(m, dtype=torch.float32, device=h.device)
        for j, tile in enumerate(tile_ids):
            w_t = tiles[j * tile_rows : (j + 1) * tile_rows]
            z = sk.gemm_fp32(h, w_t)
            mx, s, zy_t, hit = ops.tile_stats(
                z, tile, targets, multiplier=multiplier, softcap=softcap, vocab_size=vocab,
                tile_size=tile_rows,
            )
            stats[j, 0] = mx
            stats[j, 1] = s
            zy = torch.where(hit, zy_t, zy)
            del z
        if comm.world > 1:
            gathered = torch.cat(comm.all_gather(stats), dim=0)
            zy_parts = comm.all_gather(zy)
            zy_total = zy_parts[0]
            for part in zy_parts[1:]:
                zy_total = zy_total + part  # exactly one non-zero contribution per row
        else:
            gathered = stats
            zy_total = zy
        pairs = [(gathered[t, 0], gathered[t, 1]) for t in range(sk.CANONICAL_CHUNKS)]
        mx_all, s_all = ops.merge(pairs)
        logz = mx_all + torch.log(s_all)
        logp = zy_total - logz
        ctx.save_for_backward(h, targets, logz, tiles)
        ctx.meta = (tuple(tile_ids), comm, int(vocab), int(tile_rows), float(multiplier), float(softcap))
        return logp

    @staticmethod
    def backward(ctx, dlogp):
        h, targets, logz, tiles = ctx.saved_tensors
        tile_ids, comm, vocab, tile_rows, multiplier, softcap = ctx.meta
        ops = sk.lm_head_tile_ops()
        dlogp32 = dlogp.to(torch.float32).contiguous()
        parts = []
        for j, tile in enumerate(tile_ids):
            w_t = tiles[j * tile_rows : (j + 1) * tile_rows]
            z = sk.gemm_fp32(h, w_t)
            col_begin = tile * tile_rows
            real = min(max(vocab - col_begin, 0), tile_rows)
            dz = ops.tile_dz(
                z, logz, dlogp32, targets, tile_size=tile_rows, real_cols=real,
                col_begin=col_begin, multiplier=multiplier, softcap=softcap,
            )
            del z
            parts.append(sk.gemm_fp32(dz, w_t.t()))  # 8-chunk tree inside the tile (sibling rule)
            del dz
        dh = comm.all_gather_tree(sk.gemm_backend().combine(parts))
        return (dh.to(h.dtype), None, None, None, None, None, None, None, None)


# ------------------------------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------------------------------


def install_museglimmer_megatron_strict(
    model: Any,
    *,
    hf_path: str | Path | None = None,
    lora_rank: int = DEFAULT_LORA_RANK,
    seed: int = 0,
    checkpoint_layers: bool = True,
    drop_vision: bool = True,
    lora_b_init_std: float = 0.0,
) -> MuseGlimmerStrictBinding:
    """Bind the strict operator set to a Bridge ``MuseGlimmerModel`` (fails closed).

    ``lora_b_init_std > 0`` draws LoRA ``B`` from ``N(0, std)`` (TP-invariant CPU generator)
    instead of zeros; parity tests only (makes every LoRA gradient non-trivial).
    """
    inner = _unwrap(model)
    binding = MuseGlimmerStrictBinding(
        inner,
        hf_path=hf_path,
        lora_rank=lora_rank,
        seed=seed,
        checkpoint_layers=checkpoint_layers,
        drop_vision=drop_vision,
        lora_b_init_std=lora_b_init_std,
    )
    inner._muse_strict_binding = binding
    return binding


def get_binding(model: Any) -> MuseGlimmerStrictBinding:
    binding = getattr(_unwrap(model), "_muse_strict_binding", None)
    if binding is None:
        raise StrictBindingError("install_museglimmer_megatron_strict was not called on this model")
    return binding


def pack_sequences(
    sequences: Sequence[Sequence[int]], device: torch.device | str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Concatenate sequences → (tokens, cu_seqlens int32, positions int64) on ``device``."""
    lengths = [len(s) for s in sequences]
    if any(n < 2 for n in lengths):
        raise ValueError("every sequence needs at least two tokens")
    tokens = torch.tensor([t for s in sequences for t in s], dtype=torch.int64, device=device)
    cu = torch.tensor([0, *torch.cumsum(torch.tensor(lengths), 0).tolist()], dtype=torch.int32, device=device)
    positions = torch.cat([torch.arange(n, dtype=torch.int64) for n in lengths]).to(device)
    return tokens, cu, positions


def strict_token_logprobs(
    model: Any, token_ids: torch.Tensor, targets: torch.Tensor | None = None
) -> torch.Tensor:
    """fp32 ``(B, T-1)`` log-probabilities of ``targets`` (default: next tokens), with autograd."""
    binding = get_binding(model)
    if token_ids.ndim != 2:
        raise ValueError("token_ids must be (B, T)")
    b, t = token_ids.shape
    if targets is None:
        targets = token_ids[:, 1:]
    if tuple(targets.shape) != (b, t - 1):
        raise ValueError("targets must be (B, T-1)")
    seqs = token_ids.tolist()
    tokens, cu, positions = pack_sequences(seqs, binding.device)
    select = torch.cat([torch.arange(i * t, i * t + t - 1) for i in range(b)]).to(binding.device)
    logp = binding.forward_packed(tokens, cu, positions, select, targets.reshape(-1))
    return logp.view(b, t - 1)


def native_token_logprobs(model: Any, token_ids: torch.Tensor) -> torch.Tensor:
    """Unpatched Bridge forward → fp32 next-token logprobs ``(B, T-1)`` (no grad)."""
    inner = _unwrap(model)
    device = inner.decoder.layers[0].input_layernorm.weight.device
    ids = token_ids.to(device)
    b, t = ids.shape
    pos = torch.arange(t, device=device).unsqueeze(0).expand(b, -1).contiguous()
    mask = torch.triu(torch.ones(t, t, dtype=torch.bool, device=device), diagonal=1)[None, None]
    was_training = inner.training
    inner.eval()
    try:
        with torch.no_grad():
            logits = inner(input_ids=ids, position_ids=pos, attention_mask=mask, runtime_gather_output=True)
    finally:
        inner.train(was_training)
    from rl_engine.models.muse_glimmer.runtime.megatron_native_lora import chunked_token_logprobs

    rows = [chunked_token_logprobs(logits[i, :-1], ids[i, 1:]) for i in range(b)]
    return torch.stack(rows)


def export_binding_lora(binding: MuseGlimmerStrictBinding) -> tuple[dict[str, torch.Tensor], str]:
    if binding.lora is None:
        return {}, export_lora([], binding.comm)[1]
    return export_lora(binding.lora, binding.comm)


def binding_lora_parameters(binding: MuseGlimmerStrictBinding) -> list[nn.Parameter]:
    return lora_parameters(binding.lora or [])


def hex_bits(values: torch.Tensor) -> list[str]:
    """IEEE-754 fp32 bit patterns as 8 hex chars (the vLLM dump format)."""
    return [f"{v:08x}" for v in values.detach().float().cpu().contiguous().view(torch.int32).tolist()]


def load_bridge_model(
    model_path: str | Path, *, tensor_parallel_size: int, seed: int = 0
) -> tuple[Any, Any, float]:
    """``AutoBridge`` → native Bridge model (TP shards on this rank). Returns (model, bridge, secs)."""
    from megatron.bridge import AutoBridge

    t0 = time.time()
    bridge = AutoBridge.from_hf_pretrained(str(model_path))
    cfg = bridge.get_model_config()
    cfg.transformer.tensor_model_parallel_size = int(tensor_parallel_size)
    models = bridge.get_model(cfg, wrap_with_ddp=False, mixed_precision_wrapper=None)
    if len(models) != 1:
        raise StrictBindingError("expected a single pipeline stage")
    return _unwrap(models[0]), bridge, time.time() - t0


__all__ = [
    "GatherHeads",
    "LayerWeights",
    "ModelDims",
    "MuseGlimmerStrictBinding",
    "NATIVE_BRIDGE_MODEL_CLASS",
    "SliceLocalHeads",
    "StrictBindingError",
    "StrictLMHeadLogprob",
    "binding_lora_parameters",
    "export_binding_lora",
    "fused_qkv_blocks",
    "fused_qkv_to_hf",
    "get_binding",
    "hex_bits",
    "install_museglimmer_megatron_strict",
    "load_bridge_model",
    "native_token_logprobs",
    "pack_sequences",
    "strict_token_logprobs",
]
