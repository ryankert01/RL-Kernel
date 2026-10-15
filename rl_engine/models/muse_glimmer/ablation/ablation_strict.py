# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""The same batch / decode / chunking factors on the strict reference path (RFC #501 row
``museglimmer_ablation``): it must stay at 0 differing tokens.

Only PUBLIC reference names are used (``MuseGlimmerStrictModel`` methods without a leading
underscore, ``Session``, ``golden_ops.row_parallel_gemm``); a test parses this module and fails
on any private import or private attribute access.

* Single-sequence modes: ``MuseGlimmerStrictModel.logprobs(mode=train|prefill|chunked|decode)``.
* Continuous batching (:func:`score_continuous_batching`, ``mixed=False``): prompts through
  ``model.prefill(Session(cache, slot), ...)`` and every engine step through ONE
  ``model.decode_batch(cache, running_slots, ...)`` call -- public API only.
* Several sequences with several tokens each in ONE call (packed varlen training pass, one paged
  prefill of all sequences, a vLLM-style mixed prefill+decode step) have no public entry point,
  so :func:`packed_pass` runs the reference's operator set (``model.ops``) over packed
  segments.  It is a re-composition, so :func:`selfcheck_packed_pass` proves it byte-equal to
  the reference's own public paths (``forward_train``, ``prefill``, ``decode_batch``) on the
  model being measured before any packed setting runs, and raises if not (fail closed: a fork
  of the reference never gets measured as "the reference").

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
    chunk_plan,
)
from rl_engine.models.muse_glimmer.reference.golden_ops import row_parallel_gemm
from rl_engine.models.muse_glimmer.reference.strict_model import Session


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

    The contract's "Model assembly" over the reference's ``model.ops`` (same callables, same
    order, same cast points as the reference composition); only ``cu_seqlens`` / positions /
    block tables describe several sequences.  :func:`selfcheck_packed_pass` verifies it against
    the reference's public entry points.
    """
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
        attn_out = row_parallel_gemm(ops.gemm, y, ops.weight_arg(layer.o_proj))
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


def _prompt_targets(s: SeqSpec) -> torch.Tensor:
    """Prompt rows: only the last one is scored (it predicts the first completion token)."""
    t = torch.full((s.completion_start,), -1, dtype=torch.long)
    t[-1] = s.token_ids[s.completion_start]
    return t


def _sync(dev: torch.device) -> None:
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


def _timed(dev: torch.device):
    _sync(dev)
    return time.perf_counter()


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.detach().to("cpu", torch.float32).contiguous().view(torch.int32)


# --------------------------------------------------------------------------------------------
# Self-check: packed_pass == the reference's public entry points, byte for byte
# --------------------------------------------------------------------------------------------
@torch.no_grad()
def selfcheck_packed_pass(model: Any, seqs: Sequence[SeqSpec], *,
                          raise_on_mismatch: bool = True) -> dict[str, Any]:
    """Compare :func:`packed_pass` with the reference's public paths on ``seqs[:2]``:

    * ``varlen_vs_forward_train``: one segment, cache-less, vs ``model.forward_train``;
    * ``paged_vs_prefill``: one L-1 segment into a batch cache vs ``model.prefill`` on a fresh
      session;
    * ``paged_multi_vs_decode_batch``: the prompts prefilled through ``model.prefill`` into two
      identical batch caches, then one decode token per sequence through ``model.decode_batch``
      (cache A) and through ``packed_pass`` (cache B).

    Returns ``{check: {"equal", "n_rows", "n_rows_differ"}, "all_equal"}``; raises
    ``RuntimeError`` on any difference unless ``raise_on_mismatch=False``.
    """
    if len(seqs) < 2:
        raise ValueError("selfcheck needs two sequences")
    s1, s2 = seqs[0], seqs[1]
    dev = model.device
    res: dict[str, Any] = {"sequences": [s1.id, s2.id]}

    def cmp(name: str, a: torch.Tensor, b: torch.Tensor, valid: torch.Tensor) -> None:
        va, vb = _bits(a[valid.to(a.device)]), _bits(b[valid.to(b.device)])
        if va.shape != vb.shape:
            raise RuntimeError(f"selfcheck {name}: shapes {tuple(va.shape)} vs {tuple(vb.shape)}")
        nd = int((va != vb).sum())
        res[name] = {"equal": nd == 0, "n_rows": int(va.numel()), "n_rows_differ": nd}

    ids, tgt = _ids(s1), _train_targets(s1)
    a = packed_pass(model, [Segment(ids, 0, tgt)])[0]
    b = model.forward_train(ids, tgt)["logp"]
    cmp("varlen_vs_forward_train", a, b, tgt >= 0)

    n = s1.length - 1
    tgt1 = _train_targets(s1, n)
    cache = model.new_batch_cache(s1.length, 2)
    a = packed_pass(model, [Segment(ids[:n], 0, tgt1, slot=1)], cache)[0]
    b = model.prefill(model.new_session(s1.length), ids[:n], targets=tgt1)
    cmp("paged_vs_prefill", a, b, tgt1 >= 0)

    caches = [model.new_batch_cache(max(s1.length, s2.length), 2) for _ in range(2)]
    for c in caches:
        for slot, s in enumerate((s1, s2)):
            model.prefill(Session(cache=c, seq=slot), _ids(s)[: s.completion_start],
                          targets=_prompt_targets(s))
    toks = [s.token_ids[s.completion_start] for s in (s1, s2)]
    tgts = [s.token_ids[s.completion_start + 1] for s in (s1, s2)]
    a = model.decode_batch(caches[0], [0, 1], toks, tgts)
    b = torch.cat(packed_pass(model, [
        Segment(torch.tensor([toks[j]]), s.completion_start, torch.tensor([tgts[j]]), slot=j)
        for j, s in enumerate((s1, s2))], caches[1]))
    cmp("paged_multi_vs_decode_batch", a, b, torch.ones(2, dtype=torch.bool))
    del cache, caches
    res["all_equal"] = all(v["equal"] for k, v in res.items() if isinstance(v, dict))
    if raise_on_mismatch and not res["all_equal"]:
        raise RuntimeError(f"packed_pass is not byte-equal to the reference's public paths: {res}")
    return res


# --------------------------------------------------------------------------------------------
# Runners
# --------------------------------------------------------------------------------------------
STRICT_ALONE_MODES = ("train", "prefill", "decode", "chunked")


def score_alone(model: Any, seqs: Sequence[SeqSpec], mode: str,
                chunk: int | None = None) -> list[SeqScore]:
    """Reference API per sequence: ``model.logprobs(prompt, completion, mode, chunk_sizes)``."""
    if mode not in STRICT_ALONE_MODES:
        raise ValueError(f"mode must be one of {STRICT_ALONE_MODES}")
    if mode == "chunked" and not chunk:
        raise ValueError("mode 'chunked' needs a positive chunk")
    dev = model.device
    out = []
    for s in seqs:
        ids = _ids(s)
        prompt, completion = ids[: s.completion_start], ids[s.completion_start :]
        sizes = None
        if mode == "chunked":
            sizes = [b - a for a, b in chunk_plan(s.length - 1, int(chunk))]
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


def score_paged_batch_prefill(model: Any, seqs: Sequence[SeqSpec]) -> list[SeqScore]:
    """Rollout form: one paged prefill call holding ``L-1`` tokens of all sequences."""
    dev = model.device
    cache = model.new_batch_cache(max(s.length for s in seqs), len(seqs))
    segments = [Segment(_ids(s)[:-1], 0, _train_targets(s, s.length - 1), slot=b)
                for b, s in enumerate(seqs)]
    t0 = _timed(dev)
    outs = packed_pass(model, segments, cache)
    _sync(dev)
    ms = (time.perf_counter() - t0) * 1e3 / len(seqs)
    return [SeqScore(lp[s.completion_start - 1 :].float().cpu(), None, ms)
            for s, lp in zip(seqs, outs)]


@dataclass
class ContinuousRun:
    """Result of :func:`score_continuous_batching`: ``per_copy[c][b]`` is copy ``c`` of
    ``seqs[b]``; ``steps`` engine steps; per step the number of decode rows and of prompts."""

    per_copy: list[list[SeqScore]]
    steps: int
    decode_rows_per_step: list[int]
    prefill_seqs_per_step: list[int]

    @property
    def first_copy(self) -> list[SeqScore]:
        return self.per_copy[0]

    @property
    def max_decode_rows(self) -> int:
        return max(self.decode_rows_per_step, default=0)


@torch.no_grad()
def score_continuous_batching(model: Any, seqs: Sequence[SeqSpec], *, copies: int = 1,
                              stagger: int = 0, mixed: bool = False) -> ContinuousRun:
    """Continuous batching with teacher forcing over ``copies`` copies of every sequence.

    Slot ``c * len(seqs) + b`` holds copy ``c`` of ``seqs[b]`` and is admitted at engine step
    ``c * stagger`` (copies join while earlier ones run and finish: the decode batch grows,
    shrinks and the rows change position).  At its admission step a slot's prompt is prefilled
    (its last row scores the first completion token); from the next step on, every step feeds
    one teacher-forced token of every running slot.  Running slots are ordered by slot id,
    reversed on odd steps.

    ``mixed=False``: public reference API only -- each admitted prompt through
    ``model.prefill(Session(cache, slot))``, each step's decode rows through ONE
    ``model.decode_batch`` call.  ``mixed=True``: each step is ONE :func:`packed_pass` call
    holding the admitted prompts AND the decode rows (a vLLM-style mixed prefill/decode batch).
    """
    if copies < 1 or stagger < 0:
        raise ValueError("copies >= 1 and stagger >= 0")
    nb = len(seqs)
    n_slots = copies * nb
    cache = model.new_batch_cache(max(s.length for s in seqs), n_slots)
    seq_of = [seqs[slot % nb] for slot in range(n_slots)]
    admit = [(slot // nb) * stagger for slot in range(n_slots)]
    pos = [0] * n_slots  # next position to feed once running
    started = [False] * n_slots
    rows: list[list[torch.Tensor]] = [[] for _ in range(n_slots)]
    decode_rows: list[int] = []
    prefills: list[int] = []
    step = 0
    while True:
        new = [x for x in range(n_slots) if admit[x] == step]
        running = [x for x in range(n_slots)
                   if started[x] and pos[x] < seq_of[x].length - 1]
        if step % 2 == 1:
            running = running[::-1]
        if not new and not running and all(started):
            break
        dec_tok = [seq_of[x].token_ids[pos[x]] for x in running]
        dec_tgt = [seq_of[x].token_ids[pos[x] + 1] for x in running]
        if mixed:
            segs = [Segment(_ids(seq_of[x])[: seq_of[x].completion_start], 0,
                            _prompt_targets(seq_of[x]), slot=x) for x in new]
            segs += [Segment(torch.tensor([t]), pos[x], torch.tensor([g]), slot=x)
                     for x, t, g in zip(running, dec_tok, dec_tgt)]
            outs = packed_pass(model, segs, cache) if segs else []
            for x, lp in zip(new, outs[: len(new)]):
                rows[x].append(lp[-1:].float().cpu())
            for x, lp in zip(running, outs[len(new):]):
                rows[x].append(lp.float().cpu())
        else:
            for x in new:
                s = seq_of[x]
                lp = model.prefill(Session(cache=cache, seq=x), _ids(s)[: s.completion_start],
                                   targets=_prompt_targets(s))
                rows[x].append(lp[-1:].float().cpu())
            if running:
                lp = model.decode_batch(cache, running, dec_tok, dec_tgt)
                for j, x in enumerate(running):
                    rows[x].append(lp[j : j + 1].float().cpu())
        for x in new:
            started[x] = True
            pos[x] = seq_of[x].completion_start
        for x in running:
            pos[x] += 1
        decode_rows.append(len(running))
        prefills.append(len(new))
        step += 1
    per_copy = []
    for c in range(copies):
        per_copy.append([SeqScore(torch.cat(rows[c * nb + b]), None, 0.0) for b in range(nb)])
    for b, s in enumerate(seqs):
        for c in range(copies):
            if per_copy[c][b].logp.numel() != s.n_completion:
                raise RuntimeError(f"slot {c * nb + b}: {per_copy[c][b].logp.numel()} rows, "
                                   f"expected {s.n_completion}")
    return ContinuousRun(per_copy, step, decode_rows, prefills)


def score_batched_decode(model: Any, seqs: Sequence[SeqSpec]) -> tuple[list[SeqScore], int]:
    """One copy of every sequence, all admitted at step 0, public API only (see
    :func:`score_continuous_batching`).  Returns scores and the number of decode steps."""
    run = score_continuous_batching(model, seqs)
    return run.first_copy, sum(1 for r in run.decode_rows_per_step if r)


def copies_bits_differ(base: Sequence[SeqScore], run: ContinuousRun) -> list[list[int]]:
    """``[b][c]``: tokens of copy ``c`` of sequence ``b`` whose fp32 bits differ from ``base``."""
    return [[int((_bits(run.per_copy[c][b].logp) != _bits(base[b].logp)).sum())
             for c in range(len(run.per_copy))] for b in range(len(base))]


__all__ = [
    "STRICT_ALONE_MODES",
    "STRICT_BATCH_LAYOUTS",
    "ContinuousRun",
    "Segment",
    "copies_bits_differ",
    "packed_pass",
    "score_alone",
    "score_batched_decode",
    "score_continuous_batching",
    "score_packed_batch",
    "score_paged_batch_prefill",
    "selfcheck_packed_pass",
]
