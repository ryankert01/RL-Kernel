# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""vLLM 0.31 attention backend that routes Muse Glimmer decoder attention to the strict op.

The backend subclasses vLLM's ``FlashAttentionBackend`` so that metadata building and the hybrid
KV-cache groups (sliding vs full layers get their own block tables / specs) stay native, and only
the attention arithmetic is replaced by ``strict_gated_attention_fwd_paged`` (contract O5).  It is
registered as the override of ``AttentionBackendEnum.FLASH_ATTN``; the strict engine selects it
with ``attention_config={"backend": "FLASH_ATTN"}``.

vLLM 0.31 facts this file depends on (RFC notes):
* ``FlashAttentionImpl.forward(layer, query, key, value, kv_cache, attn_metadata, output,
  output_scale, output_block_scale)``; ``AttentionBackend.get_kv_cache_shape`` is gone and the
  per-layer ``kv_cache`` is logically ``(blocks, kv_heads, block_size, 2 * head_dim)`` with K and V
  packed along the last dim (``KVCacheLayout``).  The strict op needs separate CONTIGUOUS
  ``(blocks, 16, H_kv, 128)`` K and V caches, so this backend reinterprets the layer's
  (contiguous) buffer as ``(2, blocks, block_size, kv_heads, head_dim)`` and owns the KV write
  (``do_kv_cache_update`` -> ``reshape_and_cache_flash`` on those views; a pure copy).
  The strict page ``b`` is therefore NOT the native page ``b``: native page ``b`` is the bytes
  ``[b*N*H*2D, (b+1)*N*H*2D)`` of the layer region, strict K page ``b`` is ``[b*N*H*D,
  (b+1)*N*H*D)`` and strict V page ``b`` sits in the second half.  This is safe only because
  nothing but this backend touches the region: the binding rejects prefix caching (its partial-hit
  copy-on-write copies native page ranges), KV connectors / transfer / offloading, speculative
  decoding, context parallelism, KV sharing and KV-block zeroing at model init
  (``vllm_strict.check_strict_engine_config``) and makes the native page writers raise
  (``copy_kv_cache_blocks_inplace``, ``KVBlockZeroer.zero_block_ids``); cascade attention is off
  in the metadata builder and rejected in :meth:`forward`.  Layers of different KV-cache groups
  that share one tensor use the same strict view and disjoint block ids, so they cannot overlap.
* ``FlashAttentionMetadata`` carries ``block_table``, ``seq_lens``, ``query_start_loc``,
  ``max_query_len`` and ``num_actual_tokens`` per KV-cache group.  vLLM marks never-allocated or
  evicted (sliding-window) pages with the null block id ``0``; the strict op wants ``-1``, and
  masks by absolute position so an evicted page is only ever touched where its keys are masked.
* Heads per rank: vLLM builds each rank's ``Attention`` with ``32/t`` query heads and
  ``max(1, 2/t)`` KV heads (TP2 16/1, TP4 8/1 with the KV head replicated on two ranks); this
  backend passes exactly those counts to the strict op (``num_q_heads``/``num_kv_heads``).  Heads
  never interact in O5, so the rank's output is the matching slice of the TP1 output bit for bit
  (``ops/REPORT_gated_attention_fwd.md``, wave A: TP2/4/8 shard parity).

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
PAGE_SIZE = 16
CALL_COUNTS: Counter[str] = Counter()
_OPS: Any = None
# Opt-in evidence counter (measurement mode, a few extra GPU ops per local-layer call): how many
# block-table entries inside a sequence's page range are vLLM's null block 0, i.e. sliding-window
# pages vLLM has already evicted when the strict op reads the table.  Read with
# :func:`null_page_counts` (one host sync).
COUNT_NULL_PAGES_ENV = "RL_KERNEL_MUSEGLIMMER_COUNT_NULL_PAGES"
_NULL_PAGE_ACC: dict[str, Any] = {}


def _count_null_pages_enabled() -> bool:
    import os

    return os.getenv(COUNT_NULL_PAGES_ENV, "").strip() in ("1", "true", "yes")


def null_page_counts() -> dict[str, int]:
    """Accumulated in-range null-block entries seen by local / global layers (opt-in)."""

    return {key: int(acc.item()) for key, acc in _NULL_PAGE_ACC.items()}


def _accumulate_null_pages(
    key: str, block_table: torch.Tensor, seq_lens: torch.Tensor, num_reqs: int
) -> None:
    table = block_table[:num_reqs]
    pages = (seq_lens[:num_reqs].to(torch.int64) + PAGE_SIZE - 1) // PAGE_SIZE
    cols = torch.arange(table.shape[1], device=table.device)
    hits = ((table == 0) & (cols[None, :] < pages[:, None])).sum()
    acc = _NULL_PAGE_ACC.get(key)
    if acc is None:
        acc = torch.zeros((), dtype=torch.int64, device=table.device)
        _NULL_PAGE_ACC[key] = acc
    acc.add_(hits)


def bind_ops(ops: Any) -> None:
    """Give the backend the resolved op table (set by the installer)."""

    global _OPS
    _OPS = ops


def _ops() -> Any:
    if _OPS is None:
        raise RuntimeError("Muse Glimmer strict attention backend used before install")
    return _OPS


def strict_kv_views(kv_cache: torch.Tensor, head_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Reinterpret a layer's KV region as contiguous strict K/V caches ``(B, N, H, D)`` each.

    vLLM 0.31 hands every layer a LOGICAL ``(B, H, N, 2D)`` view whose strides follow the
    resolved ``KVCacheLayout`` (default ``LBNHC``: physical ``[B, N, H, 2D]`` per layer, i.e. the
    logical view is a permutation, not contiguous).  When the layer's region is dense (the
    physical-order view is contiguous; true for the layer-outermost layouts ``LBNHC``/``LBHNC``),
    its ``B * N * H * 2D`` elements are re-viewed as ``(2, B, N, H, D)``: K of every page in the
    first half, V in the second.  The strict page ``b`` is NOT the native page ``b`` (different
    byte ranges), so this view is only valid while nothing else addresses the region by block id:
    prefix caching, zeroing and KV connectors are rejected at init and the native page writers
    raise (see the module docstring).  Layers of different KV groups that share a tensor use this
    same view with disjoint block ids, so they cannot overlap.
    """

    if kv_cache.dim() != 4:
        raise RuntimeError(f"unexpected KV cache rank {kv_cache.dim()} (expected (B, H, N, 2D))")
    blocks, heads, block_size, channels = kv_cache.shape
    if channels != 2 * head_size:
        raise RuntimeError(f"KV cache last dim {channels} != 2 * head_size {2 * head_size}")
    physical = kv_cache.transpose(1, 2)  # LBNHC per-layer physical order (B, N, H, 2D)
    if not physical.is_contiguous():
        physical = kv_cache  # LBHNC: the logical view itself is dense
        if not physical.is_contiguous():
            raise RuntimeError(
                "strict Muse Glimmer attention requires a dense per-layer KV region (layer-"
                f"outermost KVCacheLayout such as LBNHC); got strides {tuple(kv_cache.stride())}"
            )
    kv = physical.reshape(-1).view(2, blocks, block_size, heads, head_size)
    return kv[0], kv[1]


def strict_block_table(block_table: torch.Tensor) -> torch.Tensor:
    """vLLM null/absent page id ``0`` -> ``-1`` (the strict op's convention)."""

    table = block_table.to(torch.int32)
    return torch.where(table == 0, torch.full_like(table, -1), table)


def _build_classes() -> tuple[type, type, type]:
    from vllm.v1.attention.backend import AttentionType
    from vllm.v1.attention.backends import flash_attn as fa_module
    from vllm.v1.attention.backends.flash_attn import (
        FlashAttentionBackend,
        FlashAttentionImpl,
        FlashAttentionMetadataBuilder,
    )

    class MuseGlimmerStrictAttentionImpl(FlashAttentionImpl):
        """Decoder attention through the strict gated op; encoder (vision) attention stays native."""

        supports_dcp = False

        def do_kv_cache_update(
            self,
            layer: torch.nn.Module,
            key: torch.Tensor,
            value: torch.Tensor,
            kv_cache: torch.Tensor,
            slot_mapping: torch.Tensor,
        ) -> None:
            if self.attn_type != AttentionType.DECODER:
                super().do_kv_cache_update(layer, key, value, kv_cache, slot_mapping)
                return
            key_cache, value_cache = strict_kv_views(kv_cache, self.head_size)
            fa_module.reshape_and_cache_flash(
                key,
                value,
                key_cache,
                value_cache,
                slot_mapping,
                self.kv_cache_dtype,
                layer._k_scale,
                layer._v_scale,
            )
            CALL_COUNTS["kv_cache_write"] += 1

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
            key_cache, value_cache = strict_kv_views(kv_cache, self.head_size)
            if key_cache.shape[1] != PAGE_SIZE:
                raise RuntimeError(
                    f"strict Muse Glimmer attention requires {PAGE_SIZE}-token pages, got "
                    f"block_size={key_cache.shape[1]}"
                )
            seq_lens = attn_metadata.seq_lens
            num_reqs = int(seq_lens.shape[0])
            query_start_loc = attn_metadata.query_start_loc
            if query_start_loc.shape[0] < num_reqs + 1:
                raise RuntimeError("attention metadata query_start_loc is shorter than seq_lens")
            cu_seqlens_q = query_start_loc[: num_reqs + 1]
            block_table = strict_block_table(attn_metadata.block_table[:num_reqs])
            window = None if self.sliding_window[0] < 0 else int(self.sliding_window[0]) + 1
            if _count_null_pages_enabled():
                _accumulate_null_pages(
                    "local" if window is not None else "global",
                    attn_metadata.block_table,
                    seq_lens,
                    num_reqs,
                )
            y, _o, _lse = _ops().gated_attention_paged(
                query[:num_tokens],
                gate[:num_tokens],
                key_cache,
                value_cache,
                block_table,
                seq_lens.to(torch.int32),
                cu_seqlens_q.to(torch.int32),
                window=window,
                max_seqlen_q=int(attn_metadata.max_query_len),
                num_q_heads=int(self.num_heads),
                num_kv_heads=int(self.num_kv_heads),
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

        @classmethod
        def get_supported_kernel_block_sizes(cls, kv_cache_spec: Any = None) -> list[Any]:
            return [PAGE_SIZE]

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


__all__ = [
    "BACKEND_CLASS_PATH",
    "CALL_COUNTS",
    "COUNT_NULL_PAGES_ENV",
    "GATE_ATTR",
    "PAGE_SIZE",
    "bind_ops",
    "null_page_counts",
    "register",
    "strict_block_table",
    "strict_kv_views",
]
