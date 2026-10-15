# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""The same batch / decode / chunking factors on the strict reference path (RFC #501 row
``museglimmer_ablation``): it must stay at 0 differing tokens.

Single-sequence modes use the reference API unchanged
(``MuseGlimmerStrictModel.logprobs(mode=train|prefill|chunked|decode)``).  Batch composition
needs several sequences in one pass, which the reference entry points do not expose, so
:func:`packed_pass` runs the reference's own operator set (``model.ops``) in the exact order of
``strict_model.MuseGlimmerStrictModel._run`` over several packed segments:

* varlen form (training / one-shot prefill): ``cu_seqlens`` over the segments, absolute
  positions restart at 0 in every segment, attention over the segment's own K/V;
* paged form (rollout): each segment is appended to its slot of a multi-sequence
  :class:`PagedKVCache` and attention reads the cache with the slots' block-table rows
  (continuous batching: a decode step packs one token of every still-active sequence).

Every op is row-local or segment-local by contract (O1-O7), so the selected-token logp of a
sequence must be byte-equal however it is batched.  A strict "padded" batch has no key-padding
mask: right padding appends pad tokens to a segment (causality hides them from real rows); left
padding puts the pad run in its own segment in front of the sequence (moves the sequence's rows
to a different offset in the packed batch, positions still start at 0).
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

from rl_engine.models.muse_glimmer.ablation.ablation_hf import (
    DEFAULT_PAD_ID,
    SeqScore,
    SeqSpec,
)


@dataclass
class Segment:
    """One packed segment of a strict pass.

    ``tokens`` are fed at absolute positions ``start .. start+n-1``; ``targets`` (same length,
    ``-1`` = none) select the logp rows.  ``slot`` is the paged-KV sequence slot (paged form).
    """

    tokens: torch.Tensor
    start: int = 0
    targets: torch.Tensor | None = None
    slot: int | None = None


def _ops_call_lm_head(model: Any, hidden: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    cfg, ops, w = model.cfg, model.ops, model.weights
    valid = targets >= 0
    out = torch.full((hidden.shape[0],), float("nan"), dtype=torch.float32, device=hidden.device)
    if not bool(valid.any()):
        return out
    rows = hidden if bool(valid.all()) else hidden[valid]
    lp = ops.lm_head_softcap_logprob(
        rows,
        ops.lm_head_arg(w.lm_head),
        targets[valid],
        multiplier=cfg.output_multiplier,
        softcap=cfg.final_logit_softcapping,
        vocab_size=cfg.vocab_size,
        gemm=ops.gemm,
    )
    return lp if rows is hidden else out.masked_scatter(valid, lp)


@torch.no_grad()
def packed_pass(model: Any, segments: Sequence[Segment], cache: Any = None) -> list[torch.Tensor]:
    """One forward over packed ``segments``; returns each segment's fp32 logp (NaN = no target).

    Mirrors ``MuseGlimmerStrictModel._run`` op for op (same callables, same order, same cast
    points); only ``cu_seqlens`` / positions / block tables describe several sequences.
    """
    from rl_engine.models.muse_glimmer.reference.strict_model import _row_parallel

    cfg, ops, w = model.cfg, model.ops, model.weights
    dev = model.device
    segs = list(segments)
    if not segs or any(int(s.tokens.numel()) == 0 for s in segs):
        raise ValueError("packed_pass needs at least one segment and no empty segments")
    lens = [int(s.tokens.numel()) for s in segs]
    token_ids = torch.cat([s.tokens.to(dev, torch.long) for s in segs])
    positions = torch.cat(
        [torch.arange(s.start, s.start + n, device=dev) for s, n in zip(segs, lens)]
    )
    targets = torch.cat([
        (s.targets.to(dev, torch.long) if s.targets is not None
         else torch.full((n,), -1, dtype=torch.long, device=dev))
        for s, n in zip(segs, lens)
    ])
    offs = [0]
    for n in lens:
        offs.append(offs[-1] + n)
    cu = torch.tensor(offs, dtype=torch.int32, device=dev)
    if cache is None and any(s.start != 0 for s in segs):
        raise ValueError("varlen (cache-less) form needs every segment to start at position 0")
    if cache is not None:
        slots = [s.slot for s in segs]
        if any(x is None for x in slots) or len(set(slots)) != len(slots):
            raise ValueError("paged form needs one distinct slot per segment")
        for s in segs:
            if cache.seq_len(s.slot) != s.start:
                raise ValueError(f"slot {s.slot}: cache has {cache.seq_len(s.slot)} tokens, "
                                 f"segment starts at {s.start}")
        slot_idx = torch.tensor(slots, dtype=torch.long, device=dev)
        seq_lens = torch.tensor([s.start + n for s, n in zip(segs, lens)], dtype=torch.int32,
                                device=dev)
    window = cfg.sliding_window
    n_tok = int(token_ids.shape[0])

    r = ops.normalized_embedding(w.embed_tokens, token_ids, eps=cfg.rms_norm_eps)
    x_n = ops.centered_rmsnorm(r, w.layers[0].input_layernorm, eps=cfg.rms_norm_eps)
    for layer in w.layers:
        i = layer.index
        q = ops.gemm(x_n, ops.weight_arg(layer.q_proj))
        k = ops.gemm(x_n, ops.weight_arg(layer.k_proj))
        v = ops.gemm(x_n, ops.weight_arg(layer.v_proj))
        g = ops.gemm(x_n, ops.weight_arg(layer.attn_gate_proj))
        q = q.view(n_tok, cfg.num_attention_heads, cfg.head_dim)
        k = k.view(n_tok, cfg.num_key_value_heads, cfg.head_dim)
        v = v.view(n_tok, cfg.num_key_value_heads, cfg.head_dim)
        if layer.is_local:
            cos, sin = ops.rope_table(positions, theta=cfg.rope_theta, dim=cfg.head_dim,
                                      device=dev)
        else:
            cos = sin = None
        q_hat, k_hat = ops.qknorm_scaled_rope(q, k, cos, sin, eps=cfg.rms_norm_eps,
                                              qk_scale=cfg.qk_scale_factor)
        layer_window = window if layer.is_local else None
        if cache is None:
            y, _o, _lse = ops.attention_varlen(q_hat, k_hat, v, g, cu, cu,
                                               positions.to(torch.int32), window=layer_window)
        else:
            for b, s in enumerate(segs):
                a, e = offs[b], offs[b + 1]
                cache.append(i, s.slot, k_hat[a:e], v[a:e], s.start)
            k_cache, v_cache, block_table = cache.layer_views(i)
            bt = block_table.index_select(0, slot_idx).contiguous()
            y, _o, _lse = ops.attention_paged(q_hat, g, k_cache, v_cache, bt, seq_lens, cu,
                                              window=layer_window)
        attn_out = _row_parallel(ops, y, layer.o_proj)
        r, x_f = ops.fused_post_norm_residual_pre_norm(
            attn_out, r, layer.post_attention_layernorm, cfg.post_norm_eps,
            layer.pre_feedforward_layernorm, cfg.rms_norm_eps, pre_form="centered",
        )
        f = ops.swiglu_ffn(x_f, ops.weight_arg(layer.mlp_gate_proj), ops.weight_arg(layer.up_proj),
                           ops.weight_arg(layer.down_proj), gemm=ops.gemm)
        last = i == cfg.num_hidden_layers - 1
        r, x_n = ops.fused_post_norm_residual_pre_norm(
            f, r, layer.post_feedforward_layernorm, cfg.post_norm_eps,
            w.final_norm if last else w.layers[i + 1].input_layernorm, cfg.rms_norm_eps,
            pre_form="weighted" if last else "centered",
        )
    if cache is not None:
        for s, n in zip(segs, lens):
            cache.advance(s.slot, n)
        for layer in w.layers:
            if layer.is_local:
                for s in segs:
                    cache.evict_window(layer.index, s.slot, window)
    logp = _ops_call_lm_head(model, x_n, targets)
    return [logp[offs[b] : offs[b + 1]] for b in range(len(segs))]


def _ids(s: SeqSpec) -> torch.Tensor:
    return torch.tensor(s.token_ids, dtype=torch.long)


def _train_targets(s: SeqSpec, n_rows: int | None = None) -> torch.Tensor:
    """Targets of a segment holding the sequence from position 0: completion rows only."""
    n = s.length if n_rows is None else n_rows
    t = torch.full((n,), -1, dtype=torch.long)
    ids = _ids(s)
    t[s.completion_start - 1 : s.length - 1] = ids[s.completion_start :]
    return t


def _sync(dev: torch.device) -> None:
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


def _timed(dev: torch.device):
    _sync(dev)
    return time.perf_counter()


STRICT_ALONE_MODES = ("train", "prefill", "decode", "chunked")


def score_alone(model: Any, seqs: Sequence[SeqSpec], mode: str,
                chunk: int | None = None) -> list[SeqScore]:
    """Reference API per sequence: ``model.logprobs(prompt, completion, mode, chunk_sizes)``."""
    from rl_engine.models.muse_glimmer.reference.chain_harness import _cycle_chunks

    if mode not in STRICT_ALONE_MODES:
        raise ValueError(f"mode must be one of {STRICT_ALONE_MODES}")
    dev = model.device
    out = []
    for s in seqs:
        ids = _ids(s)
        prompt, completion = ids[: s.completion_start], ids[s.completion_start :]
        sizes = None
        if mode == "chunked":
            sizes = _cycle_chunks(s.length - 1, (int(chunk),))
        t0 = _timed(dev)
        lp = model.logprobs(prompt, completion, mode=mode, chunk_sizes=sizes)
        _sync(dev)
        out.append(SeqScore(lp.detach().float().cpu(), None, (time.perf_counter() - t0) * 1e3))
    return out


STRICT_BATCH_LAYOUTS = ("packed", "packed_reversed", "right_pad", "left_pad")


def score_packed_batch(model: Any, seqs: Sequence[SeqSpec], layout: str,
                       pad_id: int = DEFAULT_PAD_ID) -> list[SeqScore]:
    """All sequences in ONE varlen (training-form) pass; returns scores in ``seqs`` order."""
    if layout not in STRICT_BATCH_LAYOUTS:
        raise ValueError(f"layout must be one of {STRICT_BATCH_LAYOUTS}")
    dev = model.device
    lmax = max(s.length for s in seqs)
    order = list(range(len(seqs)))
    if layout == "packed_reversed":
        order = order[::-1]
    segments: list[Segment] = []
    owner: list[int | None] = []
    for b in order:
        s = seqs[b]
        if layout == "right_pad":
            pad = torch.full((lmax - s.length,), pad_id, dtype=torch.long)
            segments.append(Segment(torch.cat([_ids(s), pad]), 0, _train_targets(s, lmax)))
            owner.append(b)
            continue
        if layout == "left_pad" and lmax > s.length:
            segments.append(Segment(torch.full((lmax - s.length,), pad_id, dtype=torch.long)))
            owner.append(None)
        segments.append(Segment(_ids(s), 0, _train_targets(s)))
        owner.append(b)
    t0 = _timed(dev)
    outs = packed_pass(model, segments)
    _sync(dev)
    ms = (time.perf_counter() - t0) * 1e3 / len(seqs)
    res: dict[int, SeqScore] = {}
    for lp, b in zip(outs, owner):
        if b is not None:
            s = seqs[b]
            res[b] = SeqScore(lp[s.completion_start - 1 : s.length - 1].float().cpu(), None, ms)
    return [res[b] for b in range(len(seqs))]


def _new_cache(model: Any, max_len: int, n: int):
    from rl_engine.models.muse_glimmer.reference.paged_kv import PagedKVCache

    cfg = model.cfg
    return PagedKVCache(num_layers=cfg.num_hidden_layers, n_kv_heads=cfg.num_key_value_heads,
                        head_dim=cfg.head_dim, max_seq_len=max_len, device=model.device,
                        dtype=torch.bfloat16, max_seqs=n)


def score_paged_batch_prefill(model: Any, seqs: Sequence[SeqSpec]) -> list[SeqScore]:
    """Rollout form: one paged prefill call holding ``L-1`` tokens of all 9 sequences."""
    dev = model.device
    cache = _new_cache(model, max(s.length for s in seqs), len(seqs))
    segments = [Segment(_ids(s)[:-1], 0, _train_targets(s, s.length - 1), slot=b)
                for b, s in enumerate(seqs)]
    t0 = _timed(dev)
    outs = packed_pass(model, segments, cache)
    _sync(dev)
    ms = (time.perf_counter() - t0) * 1e3 / len(seqs)
    return [SeqScore(lp[s.completion_start - 1 :].float().cpu(), None, ms)
            for s, lp in zip(seqs, outs)]


def score_batched_decode(model: Any, seqs: Sequence[SeqSpec]) -> tuple[list[SeqScore], int]:
    """Continuous batching: one paged prefill of all 9 prompts, then decode steps that pack
    one teacher-forced token of every still-active sequence (sequences drop out as their
    completion ends).  Returns scores and the number of decode steps."""
    dev = model.device
    cache = _new_cache(model, max(s.length for s in seqs), len(seqs))
    rows: list[list[torch.Tensor]] = [[] for _ in seqs]
    t0 = _timed(dev)
    prompt_segments = []
    for b, s in enumerate(seqs):
        ids = _ids(s)
        tgt = torch.full((s.completion_start,), -1, dtype=torch.long)
        tgt[-1] = ids[s.completion_start]
        prompt_segments.append(Segment(ids[: s.completion_start], 0, tgt, slot=b))
    for b, lp in enumerate(packed_pass(model, prompt_segments, cache)):
        rows[b].append(lp[-1:])
    steps = 0
    pos = {b: s.completion_start for b, s in enumerate(seqs)}  # next position to feed
    while True:
        active = [b for b, s in enumerate(seqs) if pos[b] < s.length - 1]
        if not active:
            break
        segs = []
        for b in active:
            s, p = seqs[b], pos[b]
            segs.append(Segment(torch.tensor([s.token_ids[p]]), p,
                                torch.tensor([s.token_ids[p + 1]]), slot=b))
        for b, lp in zip(active, packed_pass(model, segs, cache)):
            rows[b].append(lp)
            pos[b] += 1
        steps += 1
    _sync(dev)
    ms = (time.perf_counter() - t0) * 1e3 / len(seqs)
    return [SeqScore(torch.cat(r).float().cpu(), None, ms) for r in rows], steps


__all__ = [
    "STRICT_ALONE_MODES",
    "STRICT_BATCH_LAYOUTS",
    "Segment",
    "packed_pass",
    "score_alone",
    "score_batched_decode",
    "score_packed_batch",
    "score_paged_batch_prefill",
]
