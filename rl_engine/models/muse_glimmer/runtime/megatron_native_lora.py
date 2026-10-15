# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""NATIVE LoRA on Megatron Bridge's own forward (the E2E ``native`` trainer mode).

The base model runs exactly as Bridge runs it (Transformer Engine / cuBLAS kernels, fused
``linear_qkv`` and ``linear_fc1``, native attention, bf16 logits); the LoRA uses the SAME
parameterisation, initialisation and export schema as the strict path
(:class:`~rl_engine.models.muse_glimmer.runtime.megatron_lora.LayerLoRA`: separate ``A``/``B``
per HF projection, unscaled ``y = W x + B (A x)``), added by forward hooks with plain bf16
``torch.matmul`` (no batch invariance, no fixed reduction order: this is the native reference).

Hooks per decoder layer:

* ``self_attention.linear_qkv`` (column, fused): ``x`` enters through
  ``copy_to_tensor_model_parallel_region`` (identity forward, all-reduce backward); each
  projection's delta ``B_p (A_p x)`` is interleaved into the fused row layout of the local shard
  (per local KV group ``[q (16 heads) | gate (16 heads) | k | v]``) and added to the output.
* ``mlp.linear_fc1`` (column, fused ``[gate_local; up_local]``): same with two projections.
* ``self_attention.linear_proj`` / ``mlp.linear_fc2`` (row): ``z = all_reduce(x_local A_local^T)``
  via ``reduce_from_tensor_model_parallel_region``, delta ``B z`` added to the reduced output.

TP 1 and 2 only (at TP 4/8 MCore splits the fused qkv rows mid-group).
"""

from __future__ import annotations

from typing import Any

import torch

from rl_engine.models.muse_glimmer.runtime.megatron_lora import LayerLoRA

NATIVE_LORA_FORM = "native: y = W x (TE) + bf16(bf16(x A^T) B^T) via forward hooks, no scaling"


def _column_delta(x: torch.Tensor, pairs: list[tuple[torch.Tensor, torch.Tensor]]) -> list[torch.Tensor]:
    out = []
    for a, b in pairs:
        z = torch.matmul(x, a.to(x.dtype).t())
        out.append(torch.matmul(z, b.to(x.dtype).t()))
    return out


class NativeLoRAHooks:
    """Install/remove the LoRA forward hooks on a Bridge ``MuseGlimmerModel``."""

    def __init__(self, model: Any, loras: list[LayerLoRA], *, tp_world: int, heads: int,
                 kv_heads: int, head_dim: int) -> None:
        if tp_world not in (1, 2):
            raise ValueError("native LoRA hooks support TP 1/2 (fused qkv shard = whole KV groups)")
        from megatron.core.tensor_parallel.mappings import (
            copy_to_tensor_model_parallel_region,
            reduce_from_tensor_model_parallel_region,
        )

        self._copy = copy_to_tensor_model_parallel_region
        self._reduce = reduce_from_tensor_model_parallel_region
        self.model = model
        self.loras = loras
        self.enabled = True
        self.groups_local = kv_heads // tp_world
        self.q_rows = (heads // kv_heads) * head_dim  # per group
        self.kv_rows = head_dim
        self.handles: list[Any] = []
        for i, layer in enumerate(model.decoder.layers):
            lora = loras[i]
            attn = layer.self_attention
            self.handles.append(attn.linear_qkv.register_forward_hook(self._qkv_hook(lora)))
            self.handles.append(layer.mlp.linear_fc1.register_forward_hook(self._fc1_hook(lora)))
            self.handles.append(attn.linear_proj.register_forward_hook(self._row_hook(lora, "o_proj")))
            self.handles.append(
                layer.mlp.linear_fc2.register_forward_hook(self._row_hook(lora, "mlp.down_proj"))
            )

    @staticmethod
    def _add(output: Any, delta: torch.Tensor) -> Any:
        if isinstance(output, tuple):
            return (output[0] + delta.to(output[0].dtype), *output[1:])
        return output + delta.to(output.dtype)

    def _qkv_hook(self, lora: LayerLoRA):
        def hook(_module, inputs, output):
            if not self.enabled:
                return output
            x = self._copy(inputs[0])
            dq, dg, dk, dv = _column_delta(
                x, [lora.pair(p) for p in ("q_proj", "gate_proj", "k_proj", "v_proj")]
            )
            parts = []
            for g in range(self.groups_local):
                qs = slice(g * self.q_rows, (g + 1) * self.q_rows)
                ks = slice(g * self.kv_rows, (g + 1) * self.kv_rows)
                parts.extend([dq[..., qs], dg[..., qs], dk[..., ks], dv[..., ks]])
            return self._add(output, torch.cat(parts, dim=-1))

        return hook

    def _fc1_hook(self, lora: LayerLoRA):
        def hook(_module, inputs, output):
            if not self.enabled:
                return output
            x = self._copy(inputs[0])
            dg, du = _column_delta(x, [lora.pair("mlp.gate_proj"), lora.pair("mlp.up_proj")])
            return self._add(output, torch.cat([dg, du], dim=-1))

        return hook

    def _row_hook(self, lora: LayerLoRA, name: str):
        def hook(_module, inputs, output):
            if not self.enabled:
                return output
            a, b = lora.pair(name)
            x = inputs[0]
            z = self._reduce(torch.matmul(x, a.to(x.dtype).t()))
            return self._add(output, torch.matmul(z, b.to(x.dtype).t()))

        return hook

    def remove(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles.clear()


_LOGP_CHUNK = 1024


def _chunk_logp(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    lp = torch.log_softmax(logits.float(), dim=-1)
    return lp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)


def chunked_token_logprobs(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """``log_softmax(logits.float())[t, targets[t]]`` over row chunks of 1024 (native reference).

    Same per-row arithmetic as one full ``torch.log_softmax`` (row-wise kernel); the fp32 working
    set is one chunk. With autograd each chunk is checkpointed (only the bf16 logits view is
    saved), so the backward never holds a (T, V) fp32 tensor either.
    """
    from torch.utils.checkpoint import checkpoint

    outs = []
    for s in range(0, logits.shape[0], _LOGP_CHUNK):
        z = logits[s : s + _LOGP_CHUNK]
        y = targets[s : s + _LOGP_CHUNK]
        if torch.is_grad_enabled() and z.requires_grad:
            outs.append(checkpoint(_chunk_logp, z, y, use_reentrant=False))
        else:
            outs.append(_chunk_logp(z, y))
    return torch.cat(outs)


def native_logprobs_with_grad(model: Any, seq: list[int], start: int, device: torch.device,
                              *, grad: bool, recompute: bool = True) -> torch.Tensor:
    """fp32 logp of ``seq[start:]`` from Bridge's native forward (one sequence, B=1).

    With ``grad`` the model runs in train mode with MCore full per-layer activation recompute
    (base weights frozen; gradients flow to the LoRA hooks' parameters).
    """
    inner = model
    cfg = inner.config
    saved = (cfg.recompute_granularity, cfg.recompute_method, cfg.recompute_num_layers)
    was_training = inner.training
    ids = torch.tensor([seq], dtype=torch.int64, device=device)
    t = ids.shape[1]
    pos = torch.arange(t, device=device).unsqueeze(0)
    mask = torch.triu(torch.ones(t, t, dtype=torch.bool, device=device), 1)[None, None]
    handle = None
    try:
        if grad:
            # MCore's full recompute is reentrant: parameter grads inside a checkpointed layer
            # only flow if the layer INPUT requires grad. The embedding is frozen, so mark its
            # output as a grad-requiring leaf (no gradient is ever taken w.r.t. it).
            handle = inner.embedding.register_forward_hook(
                lambda _m, _i, out: out.detach().requires_grad_(True)
            )
            inner.train()
            if recompute:
                cfg.recompute_granularity, cfg.recompute_method, cfg.recompute_num_layers = (
                    "full", "uniform", 1,
                )
        else:
            inner.eval()
        with torch.set_grad_enabled(grad):
            logits = inner(input_ids=ids, position_ids=pos, attention_mask=mask,
                           runtime_gather_output=True)
            out = chunked_token_logprobs(logits[0, start - 1 : -1], ids[0, start:])
    finally:
        if handle is not None:
            handle.remove()
        inner.train(was_training)
        cfg.recompute_granularity, cfg.recompute_method, cfg.recompute_num_layers = saved
    return out


__all__ = [
    "NATIVE_LORA_FORM",
    "NativeLoRAHooks",
    "chunked_token_logprobs",
    "native_logprobs_with_grad",
]
