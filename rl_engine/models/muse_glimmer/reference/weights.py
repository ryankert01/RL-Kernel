# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Muse Glimmer strict reference: configuration and weights.

:class:`StrictConfig` carries the pinned text-model hyperparameters (official values come from
``MuseGlimmerSpec``; ``tiny()`` is a 4-layer unit-test shape with the real [L, L, L, G]
schedule). :class:`MuseGlimmerWeights` holds bf16 text tensors on one device, loaded from the
pinned HF safetensors (after :func:`verify_checkpoint`) or drawn at random for tests. Vision
tensors are never read. Weights are stored as :class:`Linear` holders in either the natural
``(N, K)`` layout (sibling ops, backward) or the ``(K, N)`` layout the golden GEMM consumes;
forward-only models keep exactly one copy. ``lm_head`` is zero-padded to the 8 canonical tiles.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

import torch

from rl_engine.models.muse_glimmer.loader import CheckpointReport, verify_checkpoint
from rl_engine.models.muse_glimmer.operator_trace import MuseGlimmerSpec
from rl_engine.models.muse_glimmer.reference.golden_ops import (
    Linear,
    LmHead,
    pad_lm_head_weight,
)
from rl_engine.models.muse_glimmer.workload import (
    FULL_ATTENTION,
    SLIDING_ATTENTION,
    expected_layer_types,
    padded_vocab_size,
    text_weight_schema,
)

_PREFIX = "model.language_model."
LAYOUTS = ("nk", "kn", "both")

# Trace node suffixes per layer, identical to operator_trace._layer_node_suffixes.
_LAYER_NODE_SUFFIXES = (
    "input_layernorm",
    "q_proj",
    "k_proj",
    "v_proj",
    "attn_gate_proj",
    "q_norm",
    "k_norm",
    "rope_q",
    "rope_k",
    "attn",
    "output_gate",
    "o_proj",
    "post_attention_layernorm",
    "residual_attn",
    "pre_feedforward_layernorm",
    "mlp_gate_proj",
    "up_proj",
    "swiglu",
    "down_proj",
    "post_feedforward_layernorm",
    "residual_mlp",
)
_GEMM_SUFFIXES = {
    "q_proj",
    "k_proj",
    "v_proj",
    "attn_gate_proj",
    "o_proj",
    "mlp_gate_proj",
    "up_proj",
    "down_proj",
}


@dataclass(frozen=True)
class StrictConfig:
    """Text-model hyperparameters the strict reference needs. Official or tiny."""

    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    padded_vocab_size: int
    sliding_window: int
    rope_theta: float
    rms_norm_eps: float
    post_norm_eps: float
    qk_scale_factor: float
    output_multiplier: float
    final_logit_softcapping: float
    layer_types: tuple[str, ...]
    name: str = "official"

    @classmethod
    def from_spec(cls, spec: MuseGlimmerSpec) -> StrictConfig:
        return cls(
            hidden_size=spec.hidden_size,
            intermediate_size=spec.intermediate_size,
            num_hidden_layers=spec.num_hidden_layers,
            num_attention_heads=spec.num_attention_heads,
            num_key_value_heads=spec.num_key_value_heads,
            head_dim=spec.head_dim,
            vocab_size=spec.vocab_size,
            padded_vocab_size=spec.padded_vocab_size,
            sliding_window=spec.sliding_window,
            rope_theta=spec.rope_theta,
            rms_norm_eps=spec.rms_norm_eps,
            post_norm_eps=spec.post_norm_eps,
            qk_scale_factor=spec.qk_scale_factor,
            output_multiplier=spec.output_multiplier,
            final_logit_softcapping=spec.final_logit_softcapping,
            layer_types=tuple(spec.layer_types),
            name="official",
        )

    @classmethod
    def official(cls) -> StrictConfig:
        return cls.from_spec(MuseGlimmerSpec.from_manifest())

    @classmethod
    def tiny(
        cls,
        *,
        num_hidden_layers: int = 4,
        hidden_size: int = 256,
        num_attention_heads: int = 8,
        num_key_value_heads: int = 2,
        head_dim: int = 32,
        intermediate_size: int = 512,
        vocab_size: int = 1024,
        sliding_window: int = 100,
        rope_theta: float = 500000.0,
    ) -> StrictConfig:
        """Unit-test shape: real [L, L, L, G] schedule and real eps/scales, small widths.

        ``sliding_window=100`` is deliberately not a multiple of the 64-key block so window
        starts fall inside blocks; ``vocab_size`` other than a multiple of 1024 exercises the
        padded lm_head columns.
        """
        return cls(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            vocab_size=vocab_size,
            padded_vocab_size=padded_vocab_size(vocab_size),
            sliding_window=sliding_window,
            rope_theta=rope_theta,
            rms_norm_eps=1e-5,
            post_norm_eps=1e-8,
            qk_scale_factor=3.87,
            output_multiplier=0.19611613513818404,
            final_logit_softcapping=20.0,
            layer_types=tuple(expected_layer_types(num_hidden_layers)),
            name="tiny",
        )

    # -- derived ------------------------------------------------------------------------
    @property
    def q_width(self) -> int:
        return self.num_attention_heads * self.head_dim

    @property
    def kv_width(self) -> int:
        return self.num_key_value_heads * self.head_dim

    def is_local(self, layer: int) -> bool:
        return self.layer_types[layer] == SLIDING_ATTENTION

    def layer_rope_theta(self) -> list[float]:
        return [0.0 if t == FULL_ATTENTION else self.rope_theta for t in self.layer_types]

    def node_names(self) -> tuple[str, ...]:
        """Operator-trace node names (same spelling as ``MuseGlimmerSpec.node_names``)."""
        names = ["embedding"]
        for i in range(self.num_hidden_layers):
            local = self.is_local(i)
            for suffix in _LAYER_NODE_SUFFIXES:
                if suffix in ("rope_q", "rope_k") and not local:
                    continue
                names.append(f"layers.{i}.{suffix}")
        names.extend(["final_norm", "lm_head", "logit_softcap", "logprob"])
        return tuple(names)

    def node_kind(self, name: str) -> str:
        top = {
            "embedding": "normalized_embedding",
            "final_norm": "final_rms_norm",
            "lm_head": "lm_head",
            "logit_softcap": "logit_softcap",
            "logprob": "logprob",
        }
        if name in top:
            return top[name]
        parts = name.split(".")
        index, suffix = int(parts[1]), parts[2]
        scope = "local" if self.is_local(index) else "global"
        if suffix.endswith("layernorm"):
            return "centered_rms_norm"
        if suffix in _GEMM_SUFFIXES:
            return "det_gemm"
        if suffix in ("q_norm", "k_norm"):
            return "qk_norm"
        if suffix in ("rope_q", "rope_k"):
            return "rope_local"
        if suffix == "attn":
            return f"attention_{scope}"
        if suffix == "output_gate":
            return f"output_gate_{scope}"
        if suffix == "swiglu":
            return "swiglu"
        if suffix in ("residual_attn", "residual_mlp"):
            return "residual_add"
        raise KeyError(name)

    def hf_text_config_kwargs(self) -> dict[str, Any]:
        """Keyword arguments for ``transformers.MuseGlimmerTextConfig`` with these shapes."""
        return {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "head_dim": self.head_dim,
            "hidden_activation": "silu",
            "max_position_embeddings": 131072,
            "rms_norm_eps": self.rms_norm_eps,
            "post_norm_eps": self.post_norm_eps,
            "qk_scale_factor": self.qk_scale_factor,
            "output_multiplier": self.output_multiplier,
            "final_logit_softcapping": self.final_logit_softcapping,
            "sliding_window": self.sliding_window,
            "layer_types": list(self.layer_types),
            "layer_rope_theta": self.layer_rope_theta(),
            "rope_parameters": {"rope_type": "default", "rope_theta": self.rope_theta},
            "attention_bias": False,
            "attention_dropout": 0.0,
            "tie_word_embeddings": False,
            "use_cache": False,
            "pad_token_id": None,
            "bos_token_id": 200000 if self.vocab_size > 200001 else None,
            "eos_token_id": 200001 if self.vocab_size > 200001 else None,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            k: (list(v) if isinstance(v, tuple) else v) for k, v in self.__dict__.items()
        }


@dataclass
class LayerWeights:
    index: int
    is_local: bool
    input_layernorm: torch.Tensor
    post_attention_layernorm: torch.Tensor
    pre_feedforward_layernorm: torch.Tensor
    post_feedforward_layernorm: torch.Tensor
    q_proj: Linear
    k_proj: Linear
    v_proj: Linear
    attn_gate_proj: Linear
    o_proj: Linear
    mlp_gate_proj: Linear
    up_proj: Linear
    down_proj: Linear

    def linears(self) -> dict[str, Linear]:
        return {
            "self_attn.q_proj": self.q_proj,
            "self_attn.k_proj": self.k_proj,
            "self_attn.v_proj": self.v_proj,
            "self_attn.gate_proj": self.attn_gate_proj,
            "self_attn.o_proj": self.o_proj,
            "mlp.gate_proj": self.mlp_gate_proj,
            "mlp.up_proj": self.up_proj,
            "mlp.down_proj": self.down_proj,
        }

    def norms(self) -> dict[str, torch.Tensor]:
        return {
            "input_layernorm": self.input_layernorm,
            "post_attention_layernorm": self.post_attention_layernorm,
            "pre_feedforward_layernorm": self.pre_feedforward_layernorm,
            "post_feedforward_layernorm": self.post_feedforward_layernorm,
        }


def _make_linear(name: str, w: torch.Tensor, layout: str) -> Linear:
    if layout == "nk":
        return Linear(name, w, None)
    w_t = w.t().contiguous()
    if layout == "kn":
        return Linear(name, None, w_t)
    return Linear(name, w, w_t)


def _make_lm_head(w: torch.Tensor, cfg: StrictConfig, layout: str) -> LmHead:
    padded = pad_lm_head_weight(w, cfg.padded_vocab_size)
    tiles_t = None
    if layout in ("kn", "both"):
        tile = cfg.padded_vocab_size // 8
        tiles_t = [padded[t * tile : (t + 1) * tile].t().contiguous() for t in range(8)]
    return LmHead(
        padded if layout in ("nk", "both") else None,
        tiles_t,
        vocab_size=cfg.vocab_size,
        padded_vocab_size=cfg.padded_vocab_size,
    )


@dataclass
class MuseGlimmerWeights:
    """bf16 text weights of one Muse Glimmer model on one device."""

    cfg: StrictConfig
    embed_tokens: torch.Tensor
    layers: list[LayerWeights]
    final_norm: torch.Tensor
    lm_head: LmHead
    layout: str
    source: str
    device: torch.device
    checkpoint_report: CheckpointReport | None = None
    load_seconds: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)

    # -- construction -----------------------------------------------------------------------
    @classmethod
    def load_checkpoint(
        cls,
        path: str | Path,
        *,
        device: torch.device | str,
        layout: str = "kn",
        verify_shards: bool = False,
        require_pinned_bytes: bool = True,
        progress: Callable[[str], None] | None = None,
    ) -> MuseGlimmerWeights:
        """Verify the pinned checkpoint, then stream the 627 text tensors to ``device``.

        Vision tensors are skipped. ``layout="kn"`` transposes each projection on the device
        as it arrives and keeps only the transposed copy (golden GEMM operand), so the peak is
        one extra tensor, not a second model.
        """
        if layout not in LAYOUTS:
            raise ValueError(f"layout must be one of {LAYOUTS}")
        root = Path(path).expanduser()
        t0 = time.perf_counter()
        report = verify_checkpoint(
            root, require_pinned_bytes=require_pinned_bytes, verify_shard_sha256=verify_shards
        )
        cfg = StrictConfig.official()
        schema = text_weight_schema()
        index = json.loads((root / "model.safetensors.index.json").read_text(encoding="utf-8"))
        weight_map: dict[str, str] = index["weight_map"]
        by_shard: dict[str, list[str]] = {}
        for name in schema:
            by_shard.setdefault(weight_map[name], []).append(name)
        dev = torch.device(device)
        raw: dict[str, torch.Tensor] = {}
        try:
            from safetensors import safe_open
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("safetensors is required to load the checkpoint") from exc
        for shard, names in sorted(by_shard.items()):
            if progress:
                progress(f"loading {len(names)} text tensors from {shard}")
            with safe_open(str(root / shard), framework="pt", device="cpu") as fh:
                for name in names:
                    t = fh.get_tensor(name)
                    if t.dtype != torch.bfloat16 or tuple(t.shape) != schema[name]:
                        raise RuntimeError(f"{name}: unexpected {t.dtype} {tuple(t.shape)}")
                    raw[name] = t.to(dev, non_blocking=False)
        weights = cls._assemble(cfg, raw, layout=layout, device=dev, source=f"hf:{root}")
        weights.checkpoint_report = report
        weights.load_seconds = time.perf_counter() - t0
        return weights

    @classmethod
    def random(
        cls,
        cfg: StrictConfig,
        *,
        device: torch.device | str,
        seed: int = 0,
        layout: str = "nk",
        requires_grad: bool = False,
        std: float = 0.02,
    ) -> MuseGlimmerWeights:
        """Seeded random bf16 weights of shape ``cfg`` (tests). Norm weights are small and
        centered (HF init is zeros for centered norms, ones for the final norm)."""
        if layout not in LAYOUTS:
            raise ValueError(f"layout must be one of {LAYOUTS}")
        dev = torch.device(device)
        gen = torch.Generator(device="cpu").manual_seed(int(seed))
        schema = text_weight_schema(cfg.__dict__ | {"num_hidden_layers": cfg.num_hidden_layers})
        raw: dict[str, torch.Tensor] = {}
        for name, shape in schema.items():
            if name.endswith("norm.weight"):
                if name == f"{_PREFIX}norm.weight":
                    t = 1.0 + 0.1 * torch.randn(shape, generator=gen)
                else:
                    t = 0.1 * torch.randn(shape, generator=gen)
            else:
                t = std * torch.randn(shape, generator=gen)
            raw[name] = t.to(dtype=torch.bfloat16).to(dev)
        weights = cls._assemble(cfg, raw, layout=layout, device=dev, source=f"random:seed={seed}")
        if requires_grad:
            for t in weights.parameters().values():
                t.requires_grad_(True)
        return weights

    @classmethod
    def _assemble(
        cls,
        cfg: StrictConfig,
        raw: dict[str, torch.Tensor],
        *,
        layout: str,
        device: torch.device,
        source: str,
    ) -> MuseGlimmerWeights:
        def take(name: str) -> torch.Tensor:
            return raw.pop(name)

        layers: list[LayerWeights] = []
        for i in range(cfg.num_hidden_layers):
            p = f"{_PREFIX}layers.{i}."
            layers.append(
                LayerWeights(
                    index=i,
                    is_local=cfg.is_local(i),
                    input_layernorm=take(p + "input_layernorm.weight"),
                    post_attention_layernorm=take(p + "post_attention_layernorm.weight"),
                    pre_feedforward_layernorm=take(p + "pre_feedforward_layernorm.weight"),
                    post_feedforward_layernorm=take(p + "post_feedforward_layernorm.weight"),
                    q_proj=_make_linear("q_proj", take(p + "self_attn.q_proj.weight"), layout),
                    k_proj=_make_linear("k_proj", take(p + "self_attn.k_proj.weight"), layout),
                    v_proj=_make_linear("v_proj", take(p + "self_attn.v_proj.weight"), layout),
                    attn_gate_proj=_make_linear(
                        "attn_gate_proj", take(p + "self_attn.gate_proj.weight"), layout
                    ),
                    o_proj=_make_linear("o_proj", take(p + "self_attn.o_proj.weight"), layout),
                    mlp_gate_proj=_make_linear(
                        "mlp_gate_proj", take(p + "mlp.gate_proj.weight"), layout
                    ),
                    up_proj=_make_linear("up_proj", take(p + "mlp.up_proj.weight"), layout),
                    down_proj=_make_linear("down_proj", take(p + "mlp.down_proj.weight"), layout),
                )
            )
        lm_head = _make_lm_head(take("lm_head.weight"), cfg, layout)
        return cls(
            cfg=cfg,
            embed_tokens=take(f"{_PREFIX}embed_tokens.weight"),
            layers=layers,
            final_norm=take(f"{_PREFIX}norm.weight"),
            lm_head=lm_head,
            layout=layout,
            source=source,
            device=device,
        )

    # -- introspection ----------------------------------------------------------------------
    def parameters(self) -> dict[str, torch.Tensor]:
        """HF-named leaf tensors (natural layout for Linear when kept, else the (K,N) copy)."""
        out: dict[str, torch.Tensor] = {f"{_PREFIX}embed_tokens.weight": self.embed_tokens}
        for layer in self.layers:
            p = f"{_PREFIX}layers.{layer.index}."
            for name, t in layer.norms().items():
                out[p + name + ".weight"] = t
            for name, lin in layer.linears().items():
                out[p + name + ".weight"] = lin.w if lin.w is not None else lin.w_t
        out[f"{_PREFIX}norm.weight"] = self.final_norm
        if self.lm_head.w_padded is not None:
            out["lm_head.weight"] = self.lm_head.w_padded
        else:
            for t, tile in enumerate(self.lm_head.tiles_t):
                out[f"lm_head.weight.tile{t}"] = tile
        return out

    def iter_tensors(self) -> Iterator[torch.Tensor]:
        yield self.embed_tokens
        for layer in self.layers:
            yield from layer.norms().values()
            for lin in layer.linears().values():
                yield from lin.parameters()
        yield self.final_norm
        if self.lm_head.w_padded is not None:
            yield self.lm_head.w_padded
        if self.lm_head.tiles_t is not None:
            yield from self.lm_head.tiles_t

    def memory_bytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.iter_tensors())

    def hf_text_state_dict(self) -> dict[str, torch.Tensor]:
        """State dict for ``transformers`` ``MuseGlimmerTextModel`` plus ``lm_head.weight``."""
        if self.layout == "kn":
            raise RuntimeError("hf_text_state_dict needs natural-layout weights (layout nk/both)")
        sd: dict[str, torch.Tensor] = {"embed_tokens.weight": self.embed_tokens.detach()}
        for layer in self.layers:
            p = f"layers.{layer.index}."
            for name, t in layer.norms().items():
                sd[p + name + ".weight"] = t.detach()
            for name, lin in layer.linears().items():
                sd[p + name + ".weight"] = lin.w.detach()
        sd["norm.weight"] = self.final_norm.detach()
        sd["lm_head.weight"] = self.lm_head.w_padded[: self.cfg.vocab_size].detach()
        return sd

    def describe(self) -> dict[str, Any]:
        return {
            "config": self.cfg.to_dict(),
            "layout": self.layout,
            "source": self.source,
            "device": str(self.device),
            "memory_gib": self.memory_bytes() / 2**30,
            "load_seconds": self.load_seconds,
            "checkpoint_report": (
                self.checkpoint_report.__dict__ if self.checkpoint_report else None
            ),
        }


__all__ = ["LAYOUTS", "LayerWeights", "MuseGlimmerWeights", "StrictConfig"]
