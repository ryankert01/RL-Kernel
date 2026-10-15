# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""vLLM 0.31 attention backend that routes Muse Glimmer decoder attention to the strict op.

The backend subclasses vLLM's ``FlashAttentionBackend`` so that metadata building, the hybrid KV
cache groups (sliding vs full layers get their own block tables / specs) and the exact
``reshape_and_cache_flash`` KV write stay native, and only the attention arithmetic is replaced by
``strict_gated_attention_fwd_paged`` (contract O5).  It is registered as the override of
``AttentionBackendEnum.FLASH_ATTN``; the strict engine selects it with
``attention_config={"backend": "FLASH_ATTN"}``.

vLLM 0.31 facts this file depends on (RFC notes):
* ``FlashAttentionImpl.forward(layer, query, key, value, kv_cache, attn_metadata, output,
  output_scale, output_block_scale)``; the per-layer ``kv_cache`` is logically ``(blocks, kv_heads,
  block_size, 2 * head_dim)`` with K and V packed along the last dim
  (``kv_cache.transpose(1, 2).split(head_dim, -1)`` yields the ``(blocks, block_size, kv_heads,
  head_dim)`` K/V views the strict op consumes; they are strided, not contiguous).
* ``AttentionBackend.get_kv_cache_shape`` is gone; the layout is described by ``KVCacheLayout``.
* ``FlashAttentionMetadata`` carries ``block_table``, ``seq_lens``, ``query_start_loc`` and
  ``num_actual_tokens`` per KV-cache group, so each layer sees the block table of its own group.
* Sliding-window layers may hold null-block ids for evicted pages; the strict op masks by absolute
  position and only reads blocks from ``floor(max(0, p - W + 1) / 64)`` on, so an evicted (null,
  all-zero) page can only be touched where its keys are masked anyway.

The gate pre-activation (contract: gate on the ``input_layernorm`` output, applied inside the
attention op) travels on a side channel: the patched ``MuseGlimmerAttention.forward`` stores it on
the ``Attention`` layer under :data:`GATE_ATTR` right before calling the layer.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import torch

GATE_ATTR = "_rlk_museglimmer_gate_logits"
BACKEND_CLASS_PATH = (
    "rl_engine.models.muse_glimmer.runtime.vllm_attention_backend."
    "MuseGlimmerStrictAttentionBackend"
)
CALL_COUNTS: Counter[str] = Counter()
_OPS: Any = None


def bind_ops(ops: Any) -> None:
    """Give the backend the resolved op table (set by the installer)."""

    global _OPS
    _OPS = ops


def _ops() -> Any:
    if _OPS is None:
        raise RuntimeError("Muse Glimmer strict attention backend used before install")
    return _OPS


def _build_classes() -> tuple[type, type, type]:
    from vllm.v1.attention.backend import AttentionType
    from vllm.v1.attention.backends.flash_attn import (
        FlashAttentionBackend,
        FlashAttentionImpl,
        FlashAttentionMetadataBuilder,
    )

    class MuseGlimmerStrictAttentionImpl(FlashAttentionImpl):
        """Decoder attention through the strict gated op; encoder (vision) attention stays native."""

        supports_dcp = False

        def forward(
            self,
            layer: torch.nn.Module,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            kv_cache: torch.Tensor,
            attn_metadata: Any,
            output: torch.Tensor,
            output_scale: torch.Tensor | None = None,
            output_block_scale: torch.Tensor | None = None,
        ) -> torch.Tensor:
            if self.attn_type != AttentionType.DECODER:
                return super().forward(
                    layer,
                    query,
                    key,
                    value,
                    kv_cache,
                    attn_metadata,
                    output,
                    output_scale,
                    output_block_scale,
                )
            if output_scale is not None or output_block_scale is not None:
                raise NotImplementedError("strict Muse Glimmer attention has no fused output quant")
            if attn_metadata is None:
                # Memory-profiling run: no KV cache yet.
                return output.fill_(0)
            if getattr(attn_metadata, "use_cascade", False):
                raise RuntimeError("strict Muse Glimmer attention forbids cascade attention")
            if self.dcp_world_size > 1:
                raise RuntimeError("strict Muse Glimmer attention does not support decode CP")
            gate = getattr(layer, GATE_ATTR, None)
            if gate is None:
                raise RuntimeError(
                    "strict Muse Glimmer attention requires the gate side channel; the native "
                    "MuseGlimmerAttention.forward was not patched"
                )
            num_tokens = int(attn_metadata.num_actual_tokens)
            # Logical (blocks, kv_heads, block_size, 2*D) -> K/V views (blocks, block_size, kv_heads, D)
            key_cache, value_cache = kv_cache.transpose(1, 2).split(self.head_size, dim=-1)
            window = None if self.sliding_window[0] < 0 else int(self.sliding_window[0]) + 1
            y, _o, _lse = _ops().gated_attention_paged(
                query[:num_tokens],
                gate[:num_tokens],
                key_cache,
                value_cache,
                attn_metadata.block_table,
                attn_metadata.seq_lens,
                attn_metadata.query_start_loc,
                window=window,
            )
            output[:num_tokens].copy_(y.view(num_tokens, self.num_heads, self.head_size))
            CALL_COUNTS["attention.local" if window is not None else "attention.global"] += 1
            return output

    class MuseGlimmerStrictAttentionMetadataBuilder(FlashAttentionMetadataBuilder):
        """Native FA metadata; cascade (shared-prefix) attention is disabled for bit stability."""

        def use_cascade_attention(self, *args: Any, **kwargs: Any) -> bool:
            return False

    class MuseGlimmerStrictAttentionBackend(FlashAttentionBackend):
        @staticmethod
        def get_impl_cls() -> type:
            return MuseGlimmerStrictAttentionImpl

        @staticmethod
        def get_builder_cls() -> type:
            return MuseGlimmerStrictAttentionMetadataBuilder

    for cls in (
        MuseGlimmerStrictAttentionImpl,
        MuseGlimmerStrictAttentionMetadataBuilder,
        MuseGlimmerStrictAttentionBackend,
    ):
        cls.__module__ = __name__
        cls.__qualname__ = cls.__name__
    return (
        MuseGlimmerStrictAttentionBackend,
        MuseGlimmerStrictAttentionImpl,
        MuseGlimmerStrictAttentionMetadataBuilder,
    )


def __getattr__(name: str) -> Any:
    # vLLM imports the backend class lazily by path (registry override); build it on demand.
    if name in {
        "MuseGlimmerStrictAttentionBackend",
        "MuseGlimmerStrictAttentionImpl",
        "MuseGlimmerStrictAttentionMetadataBuilder",
    }:
        backend, impl, builder = _build_classes()
        globals()["MuseGlimmerStrictAttentionBackend"] = backend
        globals()["MuseGlimmerStrictAttentionImpl"] = impl
        globals()["MuseGlimmerStrictAttentionMetadataBuilder"] = builder
        return globals()[name]
    raise AttributeError(name)


def register(ops: Any) -> str:
    """Override vLLM's FLASH_ATTN enum entry with the strict backend; returns the hook id."""

    from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend

    bind_ops(ops)
    register_backend(AttentionBackendEnum.FLASH_ATTN, BACKEND_CLASS_PATH)
    return f"vllm.v1.attention.backends.registry.FLASH_ATTN->{BACKEND_CLASS_PATH}"


__all__ = ["BACKEND_CLASS_PATH", "CALL_COUNTS", "GATE_ATTR", "bind_ops", "register"]
