# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""One-factor-at-a-time drift probes on the native HF Muse-Glimmer model (RFC #501,
row ``museglimmer_ablation``).

Question: where does train/rollout drift come from in the *native* stacks?  We score the nine
sequences of ``reports/muse_glimmer_poc/data/xrt_tokens.json`` with transformers 5.17.0
``MuseGlimmerTextModel`` (bf16, one GPU) and flip ONE factor at a time against a fixed baseline:

baseline (``hf_baseline``)
    eager attention, TF32 off, every sequence alone in its own forward (all ``L`` tokens, the
    way a trainer scores a sample), one-shot prefill, transformers-5.17 matmul RoPE table,
    HF lm head (bf16 GEMM, then ``* m``, ``/ T``, ``tanh``, ``* T`` in bf16), fp32
    ``log_softmax`` of those logits, first run on the first GPU of the process.

Metric per completion token ``i`` (``completion_start <= i < L``), with
``d = logp_variant - logp_baseline`` of the gold token, both fp32:
``bits_differ`` (fp32 bit patterns differ), ``|d|``, ``k3 = exp(d) - d - 1`` (fp64) and
``argmax_flip`` (argmax of the variant logits != argmax of the baseline logits).  Pure
functions here are CPU-testable; the model-dependent pieces take any ``MuseGlimmerTextModel``
(tiny random configs in the tests, the real checkpoint in ``run_ablation.py``).

Cast points of the baseline (HF 5.17.0): q/k RMSNorm fp32 -> bf16, ``q * 3.87`` in bf16, RoPE in
bf16 with cos/sin built in fp32 by ``inv_freq @ position_ids`` (a cuBLAS fp32 batched GEMM with
K = 1, hence TF32-sensitive) and cast to bf16; eager scores bf16 GEMM, softmax fp32 -> bf16;
gate ``sigmoid`` and product in bf16; logits and soft cap in bf16.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import struct
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

TEXT_PREFIX = "model.language_model."
DEFAULT_PAD_ID = 200001  # eos of the pinned tokenizer; config.pad_token_id is null
HEAD_MODES = ("bf16", "softcap_fp32", "logits_fp32")


# --------------------------------------------------------------------------------------------
# Token file
# --------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class SeqSpec:
    """One scored sequence: ``logp(token_ids[i] | token_ids[:i])`` for i in [cs, L)."""

    id: str
    token_ids: tuple[int, ...]
    completion_start: int

    @property
    def length(self) -> int:
        return len(self.token_ids)

    @property
    def n_completion(self) -> int:
        return self.length - self.completion_start

    @property
    def targets(self) -> list[int]:
        return list(self.token_ids[self.completion_start :])


def load_xrt_tokens(path: str | Path) -> tuple[list[SeqSpec], dict[str, Any]]:
    """Read ``muse_glimmer_xrt_v1``; returns the sequences and a provenance dict."""
    path = Path(path)
    raw = path.read_bytes()
    data = json.loads(raw)
    if data.get("schema") != "muse_glimmer_xrt_v1":
        raise ValueError(f"{path}: expected schema muse_glimmer_xrt_v1, got {data.get('schema')}")
    seqs = []
    for s in data["sequences"]:
        cs = int(s["completion_start"])
        ids = tuple(int(t) for t in s["token_ids"])
        if not 0 < cs < len(ids):
            raise ValueError(f"{s['id']}: completion_start {cs} outside (0, {len(ids)})")
        seqs.append(SeqSpec(id=str(s["id"]), token_ids=ids, completion_start=cs))
    meta = {
        "path": str(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "schema": data["schema"],
        "checkpoint_revision": data.get("checkpoint_revision"),
        "sequences": [
            {"id": s.id, "length": s.length, "completion_start": s.completion_start,
             "n_completion": s.n_completion}
            for s in seqs
        ],
    }
    return seqs, meta


# --------------------------------------------------------------------------------------------
# Metrics (pure, CPU)
# --------------------------------------------------------------------------------------------
def fp32_bits(t: torch.Tensor) -> torch.Tensor:
    """int32 view of the fp32 bit patterns (CPU)."""
    return t.detach().to("cpu", torch.float32).contiguous().view(torch.int32)


def fp32_hex(t: torch.Tensor) -> list[str]:
    """``'%08x'`` of every fp32 bit pattern (the muse_glimmer_xrt_logprobs_v1 encoding)."""
    return ["%08x" % struct.unpack("<I", struct.pack("<f", float(x)))[0]
            for x in t.detach().to("cpu", torch.float32).tolist()]


def _r(x: float, digits: int = 6) -> float:
    """Compact JSON float (full bits live in the hex arrays)."""
    if x == 0.0 or not math.isfinite(x):
        return float(x)
    return float(f"{x:.{digits}g}")


def compare_tokens(
    base_logp: torch.Tensor,
    var_logp: torch.Tensor,
    base_argmax: torch.Tensor | None = None,
    var_argmax: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Per-token comparison of one sequence; returns summary numbers and per-token arrays.

    ``d = var - base`` (fp32 values, difference in fp64); ``k3 = exp(d) - d - 1`` (fp64);
    ``bits_differ`` compares the fp32 bit patterns (so -0.0 vs 0.0 or two NaN payloads count).
    """
    b = base_logp.detach().to("cpu", torch.float32).reshape(-1)
    v = var_logp.detach().to("cpu", torch.float32).reshape(-1)
    if b.shape != v.shape:
        raise ValueError(f"shape mismatch {tuple(b.shape)} vs {tuple(v.shape)}")
    bits = (fp32_bits(b) != fp32_bits(v))
    d = v.double() - b.double()
    k3 = torch.expm1(d) - d
    ad = d.abs()
    n = int(b.numel())
    out: dict[str, Any] = {
        "n_tokens": n,
        "n_bits_differ": int(bits.sum()),
        "max_abs_dlogp": float(ad.max()) if n else 0.0,
        "mean_abs_dlogp": float(ad.mean()) if n else 0.0,
        "mean_k3": float(k3.mean()) if n else 0.0,
        "max_k3": float(k3.max()) if n else 0.0,
        "first_differing_token": int(torch.nonzero(bits)[0]) if bool(bits.any()) else None,
        "nonfinite_tokens": int((~torch.isfinite(v)).sum()),
        "dlogp": [_r(float(x)) for x in d.tolist()],
        "k3": [_r(float(x), 4) for x in k3.tolist()],
        "bits_differ": [int(x) for x in bits.tolist()],
        "logp_hex": fp32_hex(v),
    }
    if base_argmax is not None and var_argmax is not None:
        flips = base_argmax.reshape(-1).cpu() != var_argmax.reshape(-1).cpu()
        out["argmax_flips"] = int(flips.sum())
        out["argmax_flip"] = [int(x) for x in flips.tolist()]
    else:
        out["argmax_flips"] = None
        out["argmax_flip"] = None
    return out


def aggregate(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Token-weighted totals over the sequences of one (engine, factor, setting)."""
    n = sum(r["n_tokens"] for r in records)
    if n == 0:
        return {"n_tokens": 0}
    flips = [r["argmax_flips"] for r in records]
    k3_sum = sum(r["mean_k3"] * r["n_tokens"] for r in records)
    return {
        "n_sequences": len(records),
        "n_tokens": n,
        "n_bits_differ": sum(r["n_bits_differ"] for r in records),
        "frac_bits_differ": sum(r["n_bits_differ"] for r in records) / n,
        "max_abs_dlogp": max(r["max_abs_dlogp"] for r in records),
        "mean_abs_dlogp": sum(r["mean_abs_dlogp"] * r["n_tokens"] for r in records) / n,
        "mean_k3": k3_sum / n,
        "argmax_flips": None if any(f is None for f in flips) else sum(flips),
        "sequences_with_any_diff": sum(1 for r in records if r["n_bits_differ"]),
    }


# --------------------------------------------------------------------------------------------
# Batch / chunk layouts (pure)
# --------------------------------------------------------------------------------------------
def padded_batch(
    seqs: Sequence[SeqSpec], side: str, pad_id: int = DEFAULT_PAD_ID
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
    """All sequences in ONE padded batch.

    Returns ``(input_ids, attention_mask, position_ids, offsets)`` with shapes ``(B, Lmax)``;
    ``offsets[b]`` is the column of token 0 of sequence ``b`` (0 for right padding,
    ``Lmax - L_b`` for left padding).  Position ids count real tokens from 0 (left padding:
    ``cumsum(mask) - 1``, pads clamped to 0), so RoPE sees the same positions as alone.
    """
    if side not in ("right", "left"):
        raise ValueError("side must be 'right' or 'left'")
    lmax = max(s.length for s in seqs)
    ids = torch.full((len(seqs), lmax), int(pad_id), dtype=torch.long)
    mask = torch.zeros((len(seqs), lmax), dtype=torch.long)
    offsets = []
    for b, s in enumerate(seqs):
        off = 0 if side == "right" else lmax - s.length
        ids[b, off : off + s.length] = torch.tensor(s.token_ids)
        mask[b, off : off + s.length] = 1
        offsets.append(off)
    if side == "right":
        pos = torch.arange(lmax)[None].expand(len(seqs), lmax).clone()
    else:
        pos = (mask.cumsum(-1) - 1).clamp(min=0)
    return ids, mask, pos, offsets


def chunk_plan(n: int, chunk: int) -> list[tuple[int, int]]:
    """``[(start, end), ...]`` covering ``range(n)`` in pieces of ``chunk`` (last may be short)."""
    if n <= 0 or chunk <= 0:
        raise ValueError("n and chunk must be positive")
    return [(s, min(s + chunk, n)) for s in range(0, n, chunk)]


# --------------------------------------------------------------------------------------------
# Model loading (text weights only, bf16)
# --------------------------------------------------------------------------------------------
def hf_rope_inv_freq(cfg: Any, device: torch.device | str | None = None) -> torch.Tensor:
    """transformers 5.17 ``compute_default_rope_parameters`` (fp32; CPU when device is None)."""
    from transformers.models.muse_glimmer import modeling_muse_glimmer as hf

    inv_freq, _ = hf.MuseGlimmerTextRotaryEmbedding.compute_default_rope_parameters(cfg)
    return inv_freq if device is None else inv_freq.to(device)


def _set_rope_buffers(model: nn.Module, inv_freq: torch.Tensor) -> None:
    rot = model.rotary_emb
    rot.inv_freq = inv_freq.to(torch.float32).clone()
    rot.original_inv_freq = inv_freq.to(torch.float32).clone()


def load_text_model(
    checkpoint: str | Path, device: torch.device | str = "cuda", attn: str = "eager"
) -> tuple[nn.Module, torch.Tensor, Any]:
    """``MuseGlimmerTextModel`` + lm_head weight from the safetensors shards, bf16 on ``device``.

    The model is built on the meta device directly in bf16 (no fp32 copy, the vision tower is
    never instantiated), then materialized and filled with ``model.language_model.*`` and
    ``lm_head.weight``.  The non-persistent RoPE ``inv_freq`` is recomputed by HF's own
    ``compute_default_rope_parameters`` on CPU in fp32 (what ``__init__`` does) and kept fp32.
    ``trust_remote_code`` is never used: the classes are transformers' built-ins.
    """
    from safetensors import safe_open
    from transformers import AutoConfig
    from transformers.models.muse_glimmer import modeling_muse_glimmer as hf

    checkpoint = Path(checkpoint).expanduser()
    device = torch.device(device)
    cfg = AutoConfig.from_pretrained(str(checkpoint), trust_remote_code=False).text_config
    cfg._attn_implementation = attn
    with torch.device("meta"):
        model = hf.MuseGlimmerTextModel(cfg)
    model = model.to(torch.bfloat16).to_empty(device=device)
    lm_head = torch.empty((cfg.vocab_size, cfg.hidden_size), dtype=torch.bfloat16, device=device)
    index = json.loads((checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    by_file: dict[str, list[str]] = {}
    for key, fname in index.items():
        if key.startswith(TEXT_PREFIX) or key == "lm_head.weight":
            by_file.setdefault(fname, []).append(key)
    state = model.state_dict()
    loaded: set[str] = set()
    with torch.no_grad():
        for fname, keys in sorted(by_file.items()):
            with safe_open(str(checkpoint / fname), framework="pt", device=str(device)) as f:
                for key in keys:
                    t = f.get_tensor(key)
                    if t.dtype != torch.bfloat16:
                        raise TypeError(f"{key}: expected bf16, got {t.dtype}")
                    if key == "lm_head.weight":
                        lm_head.copy_(t)
                        loaded.add(key)
                        continue
                    name = key[len(TEXT_PREFIX) :]
                    if name not in state:
                        raise KeyError(f"unexpected text tensor {key}")
                    state[name].copy_(t)
                    loaded.add(name)
    missing = sorted(set(state) - loaded)
    if missing or "lm_head.weight" not in loaded:
        raise RuntimeError(f"text weights missing from checkpoint: {missing[:5]}")
    _set_rope_buffers(model, hf_rope_inv_freq(cfg).to(device))
    model.eval()
    return model, lm_head, cfg


# --------------------------------------------------------------------------------------------
# Factor context managers (each restores on exit)
# --------------------------------------------------------------------------------------------
@contextlib.contextmanager
def tf32(enabled: bool) -> Iterator[None]:
    a, b = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = enabled
    torch.backends.cudnn.allow_tf32 = enabled
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = a
        torch.backends.cudnn.allow_tf32 = b


@contextlib.contextmanager
def bf16_reduced_precision_reduction(enabled: bool) -> Iterator[None]:
    saved = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = enabled
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = saved


@contextlib.contextmanager
def attn_implementation(model: nn.Module, impl: str) -> Iterator[None]:
    saved = model.config._attn_implementation
    model.config._attn_implementation = impl
    try:
        yield
    finally:
        model.config._attn_implementation = saved


def rope_forward_elementwise(self: Any, x: torch.Tensor, position_ids: torch.Tensor):
    """transformers-5.19-style table: ``position * inv_freq`` as an fp32 elementwise product
    (no GEMM, so TF32 cannot touch it); everything else as in 5.17."""
    inv = self.inv_freq.float().to(x.device)
    freqs = position_ids[:, :, None].float() * inv[None, None, :]
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos() * self.attention_scaling
    sin = emb.sin() * self.attention_scaling
    return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


@contextlib.contextmanager
def rope_elementwise() -> Iterator[None]:
    from transformers.models.muse_glimmer import modeling_muse_glimmer as hf

    cls = hf.MuseGlimmerTextRotaryEmbedding
    orig = cls.forward
    cls.forward = torch.no_grad()(rope_forward_elementwise)
    try:
        yield
    finally:
        cls.forward = orig


@contextlib.contextmanager
def rope_inv_freq_on(model: nn.Module, device: torch.device | str) -> Iterator[None]:
    """Recompute ``inv_freq`` with HF's formula on ``device`` (CUDA pow vs CPU pow)."""
    rot = model.rotary_emb
    saved = (rot.inv_freq, rot.original_inv_freq)
    _set_rope_buffers(model, hf_rope_inv_freq(model.config, device).to(saved[0].device))
    try:
        yield
    finally:
        rot.inv_freq, rot.original_inv_freq = saved


# --------------------------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------------------------
@dataclass
class SeqScore:
    logp: torch.Tensor  # (n_completion,) fp32 CPU
    argmax: torch.Tensor  # (n_completion,) int64 CPU
    wall_ms: float


def head_logits(
    h: torch.Tensor, lm_head: torch.Tensor, cfg: Any, mode: str = "bf16"
) -> torch.Tensor:
    """Soft-capped logits as fp32 ``(n, V)``.

    ``bf16``: HF 5.17 (bf16 GEMM output, then ``*m``, ``/T``, ``tanh``, ``*T`` all in bf16).
    ``softcap_fp32``: same bf16 GEMM output, soft cap evaluated in fp32.
    ``logits_fp32``: fp32 GEMM of the upcast bf16 operands (no bf16 rounding of z; TF32 off),
    soft cap in fp32.
    """
    mult = float(cfg.output_multiplier)
    cap = float(cfg.final_logit_softcapping)
    if mode == "bf16":
        z = torch.nn.functional.linear(h, lm_head)
        z = z * mult
        z = z / cap
        z = torch.tanh(z)
        z = z * cap
        return z.float()
    if mode == "softcap_fp32":
        z = torch.nn.functional.linear(h, lm_head).float()
    elif mode == "logits_fp32":
        hf32 = h.float()
        step = 32768
        z = torch.cat(
            [hf32 @ lm_head[s : s + step].float().t() for s in range(0, lm_head.shape[0], step)],
            dim=-1,
        )
    else:
        raise ValueError(f"head mode must be one of {HEAD_MODES}")
    return torch.tanh((z * mult) / cap) * cap


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _logp_argmax(
    h: torch.Tensor, targets: Sequence[int], lm_head: torch.Tensor, cfg: Any, head: str
) -> tuple[torch.Tensor, torch.Tensor]:
    logits = head_logits(h, lm_head, cfg, head)
    lp = torch.log_softmax(logits, dim=-1)
    tgt = torch.tensor(list(targets), device=h.device, dtype=torch.long)
    sel = lp.gather(-1, tgt[:, None])[:, 0]
    return sel.cpu(), logits.argmax(-1).cpu()


@torch.no_grad()
def hidden_alone(model: nn.Module, s: SeqSpec) -> torch.Tensor:
    """Trainer-style: one forward over all ``L`` tokens; rows ``cs-1 .. L-2``."""
    dev = model.embed_tokens.weight.device
    ids = torch.tensor([s.token_ids], device=dev)
    h = model(input_ids=ids, use_cache=False).last_hidden_state[0]
    return h[s.completion_start - 1 : s.length - 1]


@torch.no_grad()
def hidden_prefill_minus_last(model: nn.Module, s: SeqSpec) -> torch.Tensor:
    """Rollout-style one-shot prefill of ``L-1`` tokens (the last token is never fed)."""
    dev = model.embed_tokens.weight.device
    ids = torch.tensor([s.token_ids[:-1]], device=dev)
    h = model(input_ids=ids, use_cache=False).last_hidden_state[0]
    return h[s.completion_start - 1 :]


@torch.no_grad()
def hidden_chunked(model: nn.Module, s: SeqSpec, chunk: int) -> torch.Tensor:
    """Chunked prefill of ``L-1`` tokens through a ``DynamicCache`` (HF sliding layers keep
    ``W-1`` keys); rows ``cs-1 .. L-2``."""
    from transformers import DynamicCache

    dev = model.embed_tokens.weight.device
    ids = torch.tensor([s.token_ids[:-1]], device=dev)
    cache = DynamicCache(config=model.config)
    outs = []
    for a, b in chunk_plan(ids.shape[1], chunk):
        pos = torch.arange(a, b, device=dev)[None]
        o = model(input_ids=ids[:, a:b], position_ids=pos, past_key_values=cache, use_cache=True)
        outs.append(o.last_hidden_state[0])
    return torch.cat(outs, dim=0)[s.completion_start - 1 :]


@torch.no_grad()
def hidden_decode(model: nn.Module, s: SeqSpec) -> torch.Tensor:
    """Prefill the prompt ``[0, cs)`` once, then teacher-forced decode one token at a time
    with the KV cache; rows ``cs-1 .. L-2`` (the first from the prefill)."""
    from transformers import DynamicCache

    dev = model.embed_tokens.weight.device
    ids = torch.tensor([s.token_ids], device=dev)
    cs = s.completion_start
    cache = DynamicCache(config=model.config)
    pos = torch.arange(0, cs, device=dev)[None]
    o = model(input_ids=ids[:, :cs], position_ids=pos, past_key_values=cache, use_cache=True)
    rows = [o.last_hidden_state[0, -1:]]
    for i in range(cs, s.length - 1):
        pos = torch.tensor([[i]], device=dev)
        o = model(input_ids=ids[:, i : i + 1], position_ids=pos, past_key_values=cache,
                  use_cache=True)
        rows.append(o.last_hidden_state[0, -1:])
    return torch.cat(rows, dim=0)


@torch.no_grad()
def hidden_padded(
    model: nn.Module, seqs: Sequence[SeqSpec], side: str, pad_id: int | None = None
) -> list[torch.Tensor]:
    """All sequences in one padded batch (2-D attention mask, explicit position ids).
    ``pad_id`` defaults to the pinned eos id (token 0 for vocabularies smaller than that)."""
    dev = model.embed_tokens.weight.device
    if pad_id is None:
        pad_id = DEFAULT_PAD_ID if DEFAULT_PAD_ID < model.config.vocab_size else 0
    ids, mask, pos, offsets = padded_batch(seqs, side, pad_id)
    out = model(input_ids=ids.to(dev), attention_mask=mask.to(dev), position_ids=pos.to(dev),
                use_cache=False).last_hidden_state
    rows = []
    for b, s in enumerate(seqs):
        off = offsets[b]
        rows.append(out[b, off + s.completion_start - 1 : off + s.length - 1])
    return rows


@torch.no_grad()
def hidden_batched_decode(
    model: nn.Module, seqs: Sequence[SeqSpec], pad_id: int | None = None
) -> list[torch.Tensor]:
    """Rollout-engine style for HF: the 9 prompts prefilled together LEFT-padded (2-D mask),
    then teacher-forced decode steps that feed one token of every sequence at once; a
    sequence whose completion is done is fed a masked pad.  Rows ``cs-1 .. L-2`` per sequence."""
    from transformers import DynamicCache

    dev = model.embed_tokens.weight.device
    if pad_id is None:
        pad_id = DEFAULT_PAD_ID if DEFAULT_PAD_ID < model.config.vocab_size else 0
    prompts = [SeqSpec(s.id, s.token_ids[: s.completion_start], s.completion_start - 1)
               for s in seqs]
    ids, mask, pos, offsets = padded_batch(prompts, "left", pad_id)
    cache = DynamicCache(config=model.config)
    out = model(input_ids=ids.to(dev), attention_mask=mask.to(dev), position_ids=pos.to(dev),
                past_key_values=cache, use_cache=True).last_hidden_state
    rows: list[list[torch.Tensor]] = [[out[b, -1:]] for b in range(len(seqs))]
    mask = mask.to(dev)
    steps = max(s.n_completion for s in seqs) - 1
    for j in range(steps):
        tok, msk, p_ids = [], [], []
        for s in seqs:
            p = s.completion_start + j
            live = p < s.length - 1
            tok.append(s.token_ids[p] if live else pad_id)
            msk.append(1 if live else 0)
            p_ids.append(p if live else 0)
        mask = torch.cat([mask, torch.tensor(msk, device=dev)[:, None]], dim=1)
        o = model(input_ids=torch.tensor(tok, device=dev)[:, None], attention_mask=mask,
                  position_ids=torch.tensor(p_ids, device=dev)[:, None], past_key_values=cache,
                  use_cache=True).last_hidden_state
        for b, live in enumerate(msk):
            if live:
                rows[b].append(o[b, -1:])
    return [torch.cat(r, dim=0) for r in rows]


DRIVERS = ("alone", "prefill_minus_last", "chunked", "decode", "padded_right", "padded_left",
           "batched_decode_left")


# --------------------------------------------------------------------------------------------
# Per-layer drift (where in the network do the bits start to differ, and how fast it grows)
# --------------------------------------------------------------------------------------------
class LayerTap:
    """Forward hooks copying the residual stream of ONE batch row to CPU after the embedding,
    after every decoder layer and after the final norm.  Repeated calls (chunked prefill,
    decode steps) are concatenated along the token axis."""

    def __init__(self, model: nn.Module, batch_row: int = 0):
        self.row = int(batch_row)
        n = len(model.layers)
        self.names = ["embedding"] + [f"layer_{i}" for i in range(n)] + ["final_norm"]
        self.parts: dict[str, list[torch.Tensor]] = {k: [] for k in self.names}
        mods = [model.embed_tokens, *model.layers, model.norm]
        self.handles = [m.register_forward_hook(self._hook(k)) for k, m in zip(self.names, mods)]

    def _hook(self, name: str):
        def fn(_m: Any, _i: Any, out: Any) -> None:
            t = out[0] if isinstance(out, tuple) else out
            self.parts[name].append(t[self.row].detach().to("cpu", copy=True))

        return fn

    def close(self) -> dict[str, torch.Tensor]:
        for h in self.handles:
            h.remove()
        return {k: torch.cat(v, dim=0) for k, v in self.parts.items()}


@torch.no_grad()
def layer_trace(
    model: nn.Module,
    seqs: Sequence[SeqSpec],
    target: int,
    driver: str = "alone",
    chunk: int | None = None,
) -> dict[str, torch.Tensor]:
    """Residual stream of ``seqs[target]`` at rows ``0 .. L-2`` (the rows every driver
    computes) for one driver; padded drivers run the whole batch and keep the target row."""
    s = seqs[target]
    if driver in ("padded_right", "padded_left"):
        side = driver.split("_")[1]
        lmax = max(x.length for x in seqs)
        off = 0 if side == "right" else lmax - s.length
        tap = LayerTap(model, batch_row=target)
        try:
            hidden_padded(model, seqs, side)
        finally:
            got = tap.close()
        return {k: v[off : off + s.length - 1] for k, v in got.items()}
    tap = LayerTap(model, 0)
    try:
        if driver == "alone":
            hidden_alone(model, s)
        elif driver == "prefill_minus_last":
            hidden_prefill_minus_last(model, s)
        elif driver == "chunked":
            hidden_chunked(model, s, int(chunk))
        elif driver == "decode":
            hidden_decode(model, s)
        else:
            raise ValueError(f"unknown driver {driver!r}")
    finally:
        got = tap.close()
    return {k: v[: s.length - 1] for k, v in got.items()}


def layer_diff(base: dict[str, torch.Tensor], var: dict[str, torch.Tensor],
               completion_start: int | None = None) -> list[dict[str, Any]]:
    """Per tapped point: bf16 elements with different bits, mean/max |delta| (fp32) and the
    baseline rms, over all rows and over the completion rows (``>= completion_start - 1``)."""
    out = []
    for name in base:
        b, v = base[name], var[name]
        if b.shape != v.shape:
            raise ValueError(f"{name}: {tuple(b.shape)} vs {tuple(v.shape)}")
        diff_bits = b.contiguous().view(torch.int16) != v.contiguous().view(torch.int16)
        d = (v.float() - b.float()).abs()
        rows_any = diff_bits.any(dim=-1)
        rec = {
            "point": name,
            "n_rows": int(b.shape[0]),
            "frac_elems_differ": float(diff_bits.float().mean()),
            "rows_with_any_diff": int(rows_any.sum()),
            "first_row_with_diff": int(torch.nonzero(rows_any)[0]) if bool(rows_any.any())
            else None,
            "mean_abs": float(d.mean()),
            "max_abs": float(d.max()),
            "ref_rms": float(b.float().pow(2).mean().sqrt()),
        }
        if completion_start is not None:
            c = completion_start - 1
            rec["completion_frac_elems_differ"] = float(diff_bits[c:].float().mean())
            rec["completion_mean_abs"] = float(d[c:].mean())
        out.append(rec)
    return out


def tf32_probe(device: torch.device | str, seq_len: int = 2439) -> dict[str, Any]:
    """Does TF32 reach the only fp32 GEMM of the bf16 model (the RoPE ``inv_freq @ positions``
    batched GEMM with K = 1)?  Compares bits with TF32 off/on for that GEMM and for a square
    fp32 GEMM (control that TF32 is really switched on)."""
    dev = torch.device(device)
    gen = torch.Generator(device="cpu").manual_seed(0)
    a = torch.randn(1024, 1024, generator=gen).to(dev)
    b = torch.randn(1024, 1024, generator=gen).to(dev)
    base = 1.0 / (500000.0 ** (torch.arange(0, 128, 2, dtype=torch.float) / 128))
    inv = base[None, :, None].to(dev)
    pos = torch.arange(seq_len, device=dev)[None, None, :].float()
    res: dict[str, Any] = {}
    outs = {}
    for flag in (False, True):
        with tf32(flag):
            res[f"fp32_precision_tf32_{'on' if flag else 'off'}"] = str(
                getattr(torch.backends.cuda.matmul, "fp32_precision", "n/a"))
            outs[flag] = (a @ b, (inv @ pos).transpose(1, 2))
            torch.cuda.synchronize(dev)
    elem = (pos.transpose(1, 2) * base.to(dev)[None, None, :])
    res["square_gemm_1024_elems_differ"] = int((outs[False][0] != outs[True][0]).sum())
    res["square_gemm_1024_max_abs"] = float((outs[False][0] - outs[True][0]).abs().max())
    res["rope_gemm_k1_elems_differ_tf32_on_vs_off"] = int((outs[False][1] != outs[True][1]).sum())
    res["rope_gemm_k1_elems_differ_matmul_vs_elementwise"] = int((outs[False][1] != elem).sum())
    res["rope_gemm_k1_shape"] = [1, 64, 1, seq_len]
    tf = 1.0 / (500000.0 ** (torch.arange(0, 128, 2, dtype=torch.float) / 128))
    tf_bits = tf.view(torch.int32) & ~0x1FFF  # what a TF32 operand rounding (truncation) keeps
    res["inv_freq_entries_with_low13_mantissa_bits"] = int((tf.view(torch.int32) != tf_bits).sum())
    return res


def score_all(
    model: nn.Module,
    lm_head: torch.Tensor,
    cfg: Any,
    seqs: Sequence[SeqSpec],
    *,
    driver: str = "alone",
    chunk: int | None = None,
    head: str = "bf16",
) -> tuple[list[SeqScore], dict[str, float]]:
    """Selected-token fp32 logp + argmax of every completion token of every sequence.

    Returns per-sequence scores and ``{"wall_ms", "peak_mem_gib"}`` for the whole setting.
    Per-sequence ``wall_ms`` is the forward(s) + head of that sequence (padded drivers: the
    batch forward split evenly, head per sequence).
    """
    dev = lm_head.device
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    _sync(dev)
    t_all = time.perf_counter()
    scores: list[SeqScore] = []
    if driver in ("padded_right", "padded_left", "batched_decode_left"):
        t0 = time.perf_counter()
        if driver == "batched_decode_left":
            rows = hidden_batched_decode(model, seqs)
        else:
            rows = hidden_padded(model, seqs, driver.split("_")[1])
        _sync(dev)
        fwd_ms = (time.perf_counter() - t0) * 1e3 / len(seqs)
        for s, h in zip(seqs, rows):
            t1 = time.perf_counter()
            lp, am = _logp_argmax(h, s.targets, lm_head, cfg, head)
            scores.append(SeqScore(lp, am, fwd_ms + (time.perf_counter() - t1) * 1e3))
    else:
        for s in seqs:
            t0 = time.perf_counter()
            if driver == "alone":
                h = hidden_alone(model, s)
            elif driver == "prefill_minus_last":
                h = hidden_prefill_minus_last(model, s)
            elif driver == "chunked":
                if chunk is None:
                    raise ValueError("driver 'chunked' needs chunk")
                h = hidden_chunked(model, s, chunk)
            elif driver == "decode":
                h = hidden_decode(model, s)
            else:
                raise ValueError(f"unknown driver {driver!r}; expected one of {DRIVERS}")
            lp, am = _logp_argmax(h, s.targets, lm_head, cfg, head)
            _sync(dev)
            scores.append(SeqScore(lp, am, (time.perf_counter() - t0) * 1e3))
    _sync(dev)
    stats = {
        "wall_ms": (time.perf_counter() - t_all) * 1e3,
        "peak_mem_gib": (torch.cuda.max_memory_allocated(dev) / 2**30
                         if dev.type == "cuda" else 0.0),
    }
    return scores, stats


# --------------------------------------------------------------------------------------------
# Settings registry: one factor flipped per setting
# --------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Setting:
    name: str
    factor: str
    factor_id: int
    description: str
    driver: str = "alone"
    chunk: int | None = None
    head: str = "bf16"
    contexts: tuple[str, ...] = ()  # names resolved by ``enter_contexts``


HF_SETTINGS: tuple[Setting, ...] = (
    Setting("repeat_immediate", "repeat_null", 9, "baseline again, right after the baseline"),
    Setting("padded_batch9_right", "batch_composition", 1,
            "all 9 sequences in one right-padded batch (2-D mask)", driver="padded_right"),
    Setting("padded_batch9_left", "batch_composition", 1,
            "all 9 sequences in one left-padded batch (2-D mask, positions from 0)",
            driver="padded_left"),
    Setting("kv_cache_decode", "prefill_vs_decode", 2,
            "prompt prefill + teacher-forced 1-token decode steps with DynamicCache",
            driver="decode"),
    Setting("prefill_L_minus_1", "chunked_prefill", 3,
            "one-shot prefill of L-1 tokens (rollout prefill; only the GEMM M changes)",
            driver="prefill_minus_last"),
    Setting("chunked_prefill_64", "chunked_prefill", 3, "chunks of 64 through DynamicCache",
            driver="chunked", chunk=64),
    Setting("chunked_prefill_256", "chunked_prefill", 3, "chunks of 256 through DynamicCache",
            driver="chunked", chunk=256),
    Setting("chunked_prefill_1024", "chunked_prefill", 3, "chunks of 1024 through DynamicCache",
            driver="chunked", chunk=1024),
    Setting("sdpa", "attention_impl", 4, "torch SDPA instead of eager", contexts=("sdpa",)),
    Setting("tf32_on", "tf32", 5, "allow_tf32=True (cuBLAS + cuDNN)", contexts=("tf32_on",)),
    Setting("softcap_fp32", "logits_dtype", 6, "bf16 lm_head GEMM, soft cap in fp32",
            head="softcap_fp32"),
    Setting("logits_fp32", "logits_dtype", 6, "fp32 lm_head GEMM (no bf16 rounding) + fp32 cap",
            head="logits_fp32"),
    Setting("rope_elementwise", "rope_table", 7,
            "cos/sin from fp32 elementwise position*inv_freq (5.19) instead of the 5.17 matmul",
            contexts=("rope_elementwise",)),
    Setting("rope_elementwise_tf32_on", "rope_table", 7,
            "two factors: TF32 on AND elementwise RoPE (is TF32's effect all in the RoPE GEMM?)",
            contexts=("rope_elementwise", "tf32_on")),
    Setting("rope_inv_freq_cuda", "rope_table", 7,
            "inv_freq computed by HF's formula on the GPU instead of the CPU",
            contexts=("rope_inv_freq_cuda",)),
    Setting("bf16_reduced_reduction_off", "gemm_reduction", 10,
            "allow_bf16_reduced_precision_reduction=False (cuBLAS bf16 split-K in fp32)",
            contexts=("bf16_rr_off",)),
    Setting("rollout_like_batched_decode", "combination_rollout_like", 11,
            "factors 1+2 together: 9 prompts left-padded prefill + batched teacher-forced decode",
            driver="batched_decode_left"),
    Setting("rollout_like_batched_decode_sdpa", "combination_rollout_like", 11,
            "factors 1+2+4 together: the same with SDPA", driver="batched_decode_left",
            contexts=("sdpa",)),
)
SKIPPED_SETTINGS = (
    {"name": "flash_attention_2", "factor": "attention_impl", "factor_id": 4,
     "reason": "flash_attn is not installed in ~/envs/vllm (import fails); not installed per "
               "the no-install rule"},
)


@contextlib.contextmanager
def enter_contexts(model: nn.Module, names: Sequence[str]) -> Iterator[None]:
    with contextlib.ExitStack() as stack:
        for name in names:
            if name == "sdpa":
                stack.enter_context(attn_implementation(model, "sdpa"))
            elif name == "tf32_on":
                stack.enter_context(tf32(True))
            elif name == "rope_elementwise":
                stack.enter_context(rope_elementwise())
            elif name == "rope_inv_freq_cuda":
                stack.enter_context(rope_inv_freq_on(model, model.embed_tokens.weight.device))
            elif name == "bf16_rr_off":
                stack.enter_context(bf16_reduced_precision_reduction(False))
            else:
                raise ValueError(f"unknown context {name!r}")
        yield


def run_setting(
    model: nn.Module,
    lm_head: torch.Tensor,
    cfg: Any,
    seqs: Sequence[SeqSpec],
    setting: Setting,
) -> tuple[list[SeqScore], dict[str, float]]:
    with enter_contexts(model, setting.contexts):
        return score_all(model, lm_head, cfg, seqs, driver=setting.driver, chunk=setting.chunk,
                         head=setting.head)


def make_records(
    engine: str,
    setting_name: str,
    factor: str,
    factor_id: int,
    baseline_name: str,
    seqs: Sequence[SeqSpec],
    base: Sequence[SeqScore],
    var: Sequence[SeqScore],
    extra: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """One record per sequence with summary numbers and per-token arrays."""
    out = []
    for s, b, v in zip(seqs, base, var):
        rec = {
            "engine": engine,
            "factor": factor,
            "factor_id": factor_id,
            "setting": setting_name,
            "baseline": baseline_name,
            "sequence": s.id,
            "wall_ms": v.wall_ms,
        }
        rec.update(compare_tokens(b.logp, v.logp,
                                  None if b.argmax is None else b.argmax,
                                  None if v.argmax is None else v.argmax))
        if extra:
            rec.update(extra)
        out.append(rec)
    return out


def baseline_payload(seqs: Sequence[SeqSpec], scores: Sequence[SeqScore]) -> list[dict[str, Any]]:
    """Raw baseline numbers per sequence (logp values + exact hex, argmax, targets)."""
    return [
        {
            "sequence": s.id,
            "completion_start": s.completion_start,
            "targets": s.targets,
            "logp": [float(x) for x in sc.logp.tolist()],
            "logp_hex": fp32_hex(sc.logp),
            "argmax": None if sc.argmax is None else [int(x) for x in sc.argmax.tolist()],
            "wall_ms": sc.wall_ms,
        }
        for s, sc in zip(seqs, scores)
    ]


def env_info(device: torch.device | str | None = None) -> dict[str, Any]:
    import os
    import platform

    import triton

    info: dict[str, Any] = {
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "triton": triton.__version__,
        "python": platform.python_version(),
        "gpu_lease": os.environ.get("GPU_LEASE"),
        "tf32_matmul_default": bool(torch.backends.cuda.matmul.allow_tf32),
    }
    for mod in ("transformers", "vllm"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception as exc:  # pragma: no cover - env dependent
            info[mod] = f"unavailable ({type(exc).__name__})"
    if device is not None and torch.cuda.is_available():
        dev = torch.device(device)
        props = torch.cuda.get_device_properties(dev)
        info["gpu"] = props.name
        info["gpu_uuid"] = str(getattr(props, "uuid", ""))
        info["sm"] = f"{props.major}{props.minor}"
        info["gpu_index_in_process"] = dev.index
    return info


__all__ = [
    "DEFAULT_PAD_ID",
    "DRIVERS",
    "LayerTap",
    "HEAD_MODES",
    "HF_SETTINGS",
    "SKIPPED_SETTINGS",
    "SeqScore",
    "SeqSpec",
    "Setting",
    "aggregate",
    "baseline_payload",
    "chunk_plan",
    "compare_tokens",
    "enter_contexts",
    "env_info",
    "fp32_hex",
    "head_logits",
    "hidden_alone",
    "hidden_batched_decode",
    "hidden_chunked",
    "hidden_decode",
    "hidden_padded",
    "hidden_prefill_minus_last",
    "layer_diff",
    "layer_trace",
    "load_text_model",
    "load_xrt_tokens",
    "make_records",
    "padded_batch",
    "rope_elementwise",
    "run_setting",
    "score_all",
    "tf32",
    "tf32_probe",
]

