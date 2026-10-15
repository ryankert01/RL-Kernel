# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""``weight_sync_identity`` (RFC #501): prove Bridge's HF→Megatron conversion keeps every bit.

For each of the 627 logical text tensors of the pinned checkpoint, the Megatron-side value is
reconstructed from the Bridge model's parameters (un-fused, un-sharded: fused ``linear_qkv`` →
q/k/v/gate rows, fused ``linear_fc1`` → gate/up, TP shards gathered with a rank-ordered
all-gather), hashed with SHA-256 over its bf16 bytes and compared (``torch.equal``) with the
tensor read from the HF safetensors shards. Tensors that differ are reported with the number of
differing elements, the max abs difference and the reason when known.

Known exception: ``decoder.final_norm.weight`` is stored by Bridge as ``w - 1`` in bf16 (the
``RMSNorm2ZeroCenteredRMSNormMapping``); ``(w_meg + 1)`` does not reproduce the HF bf16 tensor for
every element, so the strict path reloads the HF tensor (see ``megatron_strict``).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator

import torch

from rl_engine.models.muse_glimmer.runtime import strict_kernels as sk
from rl_engine.models.muse_glimmer.runtime.megatron_strict import (
    _unwrap,
    fused_qkv_to_hf,
)
from rl_engine.models.muse_glimmer.workload import OFFICIAL_FINGERPRINT, text_weight_schema

_PREFIX = "model.language_model."


@dataclass
class TensorMismatch:
    name: str
    shape: tuple[int, ...]
    differing_elements: int
    max_abs_diff: float
    megatron_sha256: str
    hf_sha256: str
    reason: str = ""


@dataclass
class IdentityReport:
    total: int
    equal: int
    mismatches: list[TensorMismatch] = field(default_factory=list)
    aggregate_sha256_megatron: str = ""
    aggregate_sha256_hf: str = ""
    tp: int = 1

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


def sha256_bf16(t: torch.Tensor) -> str:
    if t.dtype != torch.bfloat16:
        raise TypeError("sha256_bf16 expects a bf16 tensor")
    return hashlib.sha256(t.detach().cpu().contiguous().view(torch.int16).numpy().tobytes()).hexdigest()


def _hf_reader(hf_path: str | Path):
    from safetensors import safe_open

    root = Path(hf_path)
    index = json.loads((root / "model.safetensors.index.json").read_text(encoding="utf-8"))
    weight_map = index["weight_map"]
    handles: dict[str, Any] = {}

    def read(name: str) -> torch.Tensor:
        shard = weight_map[name]
        if shard not in handles:
            handles[shard] = safe_open(str(root / shard), "pt", device="cpu")
        return handles[shard].get_tensor(name)

    return read


def iter_megatron_text_tensors(model: Any, comm: sk.TPComm) -> Iterator[tuple[str, torch.Tensor, str]]:
    """Yield ``(hf_name, bf16 tensor on CPU, note)`` for every logical text tensor, gathered."""
    f = OFFICIAL_FINGERPRINT
    heads, kv_heads, head_dim = int(f["num_attention_heads"]), int(f["num_key_value_heads"]), int(f["head_dim"])
    ffn = int(f["intermediate_size"])
    inner = _unwrap(model)
    tp = comm.world

    def gather_rows(t: torch.Tensor) -> torch.Tensor:
        return t if tp == 1 else torch.cat(comm.all_gather(t.contiguous()), dim=0)

    def gather_cols(t: torch.Tensor) -> torch.Tensor:
        return t if tp == 1 else torch.cat(comm.all_gather(t.contiguous()), dim=1)

    emb = inner.embedding.word_embeddings.weight.detach()
    yield f"{_PREFIX}embed_tokens.weight", gather_rows(emb).cpu(), ""
    out_w = inner.output_layer.weight.detach()
    yield "lm_head.weight", gather_rows(out_w).cpu(), ""
    fn = inner.decoder.final_norm.weight.detach()
    yield (
        f"{_PREFIX}norm.weight",
        (fn.float() + 1.0).to(torch.bfloat16).cpu(),
        "Bridge stores (w-1) in bf16 via RMSNorm2ZeroCenteredRMSNormMapping; reconstructed as "
        "bf16(w_meg + 1)",
    )
    for i, layer in enumerate(inner.decoder.layers):
        p = f"{_PREFIX}layers.{i}."
        attn = layer.self_attention
        yield p + "input_layernorm.weight", layer.input_layernorm.weight.detach().cpu(), ""
        yield p + "post_attention_layernorm.weight", attn.post_layernorm.weight.detach().cpu(), ""
        yield p + "pre_feedforward_layernorm.weight", layer.pre_mlp_layernorm.weight.detach().cpu(), ""
        yield p + "post_feedforward_layernorm.weight", layer.mlp.post_layernorm.weight.detach().cpu(), ""
        qkv_full = gather_rows(attn.linear_qkv.weight.detach())
        parts = fused_qkv_to_hf(qkv_full, heads, kv_heads, head_dim)
        for name in ("q_proj", "k_proj", "v_proj", "gate_proj"):
            yield p + f"self_attn.{name}.weight", parts[name].cpu(), "un-fused from linear_qkv"
        del qkv_full, parts
        yield p + "self_attn.o_proj.weight", gather_cols(attn.linear_proj.weight.detach()).cpu(), ""
        fc1 = layer.mlp.linear_fc1.weight.detach()
        ffn_l = ffn // tp
        gate_full = gather_rows(fc1[:ffn_l])
        up_full = gather_rows(fc1[ffn_l:])
        yield p + "mlp.gate_proj.weight", gate_full.cpu(), "un-fused from linear_fc1[:F/tp]"
        yield p + "mlp.up_proj.weight", up_full.cpu(), "un-fused from linear_fc1[F/tp:]"
        del gate_full, up_full
        yield p + "mlp.down_proj.weight", gather_cols(layer.mlp.linear_fc2.weight.detach()).cpu(), ""


def check_weight_identity(model: Any, hf_path: str | Path, comm: sk.TPComm | None = None) -> IdentityReport:
    """Compare every logical text tensor of the Bridge model with the HF checkpoint bytes.

    All TP ranks must call this (they take part in the gathers); the report is meaningful on
    every rank (identical data).
    """
    comm = comm if comm is not None else sk.TPComm(None)
    schema = text_weight_schema()
    read_hf = _hf_reader(hf_path)
    seen: set[str] = set()
    report = IdentityReport(total=len(schema), equal=0, tp=comm.world)
    agg_m = hashlib.sha256()
    agg_h = hashlib.sha256()
    for name, meg, note in iter_megatron_text_tensors(model, comm):
        if name not in schema:
            raise KeyError(f"{name} is not in the pinned text schema")
        seen.add(name)
        hf = read_hf(name)
        if tuple(meg.shape) != schema[name] or tuple(hf.shape) != schema[name]:
            raise ValueError(f"{name}: shape {tuple(meg.shape)} / {tuple(hf.shape)} != {schema[name]}")
        sha_m = sha256_bf16(meg)
        sha_h = sha256_bf16(hf)
        agg_m.update(sha_m.encode())
        agg_h.update(sha_h.encode())
        if sha_m == sha_h and torch.equal(meg, hf):
            report.equal += 1
        else:
            diff = meg.float() - hf.float()
            report.mismatches.append(
                TensorMismatch(
                    name=name,
                    shape=tuple(meg.shape),
                    differing_elements=int((meg != hf).sum().item()),
                    max_abs_diff=float(diff.abs().max().item()),
                    megatron_sha256=sha_m,
                    hf_sha256=sha_h,
                    reason=note,
                )
            )
        del meg, hf
    missing = sorted(set(schema) - seen)
    if missing:
        raise RuntimeError(f"identity check did not cover {len(missing)} tensors: {missing[:4]}")
    report.aggregate_sha256_megatron = agg_m.hexdigest()
    report.aggregate_sha256_hf = agg_h.hexdigest()
    return report


__all__ = ["IdentityReport", "TensorMismatch", "check_weight_identity", "iter_megatron_text_tensors", "sha256_bf16"]
