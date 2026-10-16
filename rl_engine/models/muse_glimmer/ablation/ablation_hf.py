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
``bits_differ`` (fp32 bit patterns differ), ``|d|``, ``k3 = exp(d) - d - 1`` (fp64; a size of
the gold-token mismatch, NOT a KL), ``kl_full = KL(base || variant)`` over the full vocabulary
(fp64 sum of the two fp32 log-softmax rows), ``argmax_flip`` (argmax id of the variant logits !=
the baseline's) split into flips where both top-1 logits are unique and flips at a top-1 tie
(``top2_gap == 0`` in either run: the bf16 soft-capped logits have exact ties, where
``argmax`` returns the lowest tied id).  Pure functions here are CPU-testable; the
model-dependent pieces take any ``MuseGlimmerTextModel`` (tiny random configs in the tests, the
real checkpoint in ``run_ablation.py``).

Cast points of the baseline (HF 5.17.0): q/k RMSNorm fp32 -> bf16, ``q * 3.87`` in bf16, RoPE in
bf16 with cos/sin built in fp32 by ``inv_freq @ position_ids`` (a cuBLAS fp32 batched GEMM with
K = 1: one rounded product per element, so TF32 on/off and an elementwise product give the same
bits, see :func:`tf32_probe`) and cast to bf16; ``inv_freq`` itself is built by HF on the CPU;
eager scores bf16 GEMM, softmax fp32 -> bf16; gate ``sigmoid`` and product in bf16; logits and
soft cap in bf16.
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


def hex_to_fp32(hexes: Sequence[str]) -> torch.Tensor:
    """Inverse of :func:`fp32_hex` (exact)."""
    vals = [int(h, 16) for h in hexes]
    return torch.tensor([v - (1 << 32) if v >= (1 << 31) else v for v in vals],
                        dtype=torch.int32).view(torch.float32)


def compare_tokens(
    base_logp: torch.Tensor,
    var_logp: torch.Tensor,
    base_argmax: torch.Tensor | None = None,
    var_argmax: torch.Tensor | None = None,
    *,
    base_gap: torch.Tensor | None = None,
    var_gap: torch.Tensor | None = None,
    kl: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Per-token comparison of one sequence; returns summary numbers and per-token arrays.

    ``d = var - base`` (fp32 values, difference in fp64); ``k3 = exp(d) - d - 1`` (fp64);
    ``bits_differ`` compares the fp32 bit patterns (so -0.0 vs 0.0 or two NaN payloads count).
    With both argmax ids, ``argmax_flip`` is the raw id comparison; with both top-1 minus top-2
    logit gaps, a flip where either gap is 0 (a tie, the id is a tie-break) is counted in
    ``argmax_flips_at_ties`` and the others in ``argmax_flips_unique``.  ``kl`` is the per-token
    full-vocabulary ``KL(base || var)`` when the caller had both distributions.
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
    out.update({"argmax_flips": None, "argmax_flip": None, "var_argmax": None,
                "argmax_flips_unique": None, "argmax_flips_at_ties": None, "n_top1_ties": None,
                "top1_tie": None, "var_top2_gap": None})
    if base_argmax is not None and var_argmax is not None:
        ba, va = base_argmax.reshape(-1).cpu(), var_argmax.reshape(-1).cpu()
        flips = ba != va
        out["argmax_flips"] = int(flips.sum())
        out["argmax_flip"] = [int(x) for x in flips.tolist()]
        out["var_argmax"] = [int(x) for x in va.tolist()]
        if base_gap is not None and var_gap is not None:
            bg = base_gap.reshape(-1).cpu().float()
            vg = var_gap.reshape(-1).cpu().float()
            tie = (bg == 0) | (vg == 0)
            out["argmax_flips_unique"] = int((flips & ~tie).sum())
            out["argmax_flips_at_ties"] = int((flips & tie).sum())
            out["n_top1_ties"] = int(tie.sum())
            out["top1_tie"] = [int(x) for x in tie.tolist()]
            out["var_top2_gap"] = [_r(float(x)) for x in vg.tolist()]
    if kl is not None:
        k = kl.detach().to("cpu", torch.float64).reshape(-1)
        if k.shape != b.shape:
            raise ValueError("kl must have one value per token")
        out["kl_full"] = [_r(float(x)) for x in k.tolist()]
        out["mean_kl_full"] = float(k.mean()) if n else 0.0
        out["max_kl_full"] = float(k.max()) if n else 0.0
    else:
        out["kl_full"] = out["mean_kl_full"] = out["max_kl_full"] = None
    return out


def _sum_or_none(records: Sequence[dict[str, Any]], key: str) -> int | None:
    vals = [r.get(key) for r in records]
    return None if any(v is None for v in vals) else sum(vals)


def aggregate(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Token-weighted totals over the sequences of one (engine, factor, setting)."""
    n = sum(r["n_tokens"] for r in records)
    if n == 0:
        return {"n_tokens": 0}
    k3_sum = sum(r["mean_k3"] * r["n_tokens"] for r in records)
    kl = [r.get("mean_kl_full") for r in records]
    has_kl = all(k is not None for k in kl)
    return {
        "n_sequences": len(records),
        "n_tokens": n,
        "n_bits_differ": sum(r["n_bits_differ"] for r in records),
        "frac_bits_differ": sum(r["n_bits_differ"] for r in records) / n,
        "max_abs_dlogp": max(r["max_abs_dlogp"] for r in records),
        "mean_abs_dlogp": sum(r["mean_abs_dlogp"] * r["n_tokens"] for r in records) / n,
        "mean_k3": k3_sum / n,
        "mean_kl_full": (sum(r["mean_kl_full"] * r["n_tokens"] for r in records) / n
                         if has_kl else None),
        "max_kl_full": max(r["max_kl_full"] for r in records) if has_kl else None,
        "argmax_flips": _sum_or_none(records, "argmax_flips"),
        "argmax_flips_unique": _sum_or_none(records, "argmax_flips_unique"),
        "argmax_flips_at_ties": _sum_or_none(records, "argmax_flips_at_ties"),
        "n_top1_ties": _sum_or_none(records, "n_top1_ties"),
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
    """transformers 5.17 ``compute_default_rope_parameters``: the pow is ALWAYS evaluated on the
    CPU in fp32 (HF builds ``arange`` on the CPU and only copies the result to ``device``)."""
    from transformers.models.muse_glimmer import modeling_muse_glimmer as hf

    inv_freq, _ = hf.MuseGlimmerTextRotaryEmbedding.compute_default_rope_parameters(cfg)
    return inv_freq if device is None else inv_freq.to(device)


def rope_inv_freq_formula(cfg: Any, device: torch.device | str) -> torch.Tensor:
    """HF 5.17's ``inv_freq`` expression ``1 / (theta ** (arange(0, D, 2, fp32) / D))``
    evaluated by torch ON ``device`` (``arange`` is created there, so the division and the pow
    run in that device's kernels).  On the CPU this is bitwise HF's own result (tested); on a
    GPU the CUDA ``pow`` rounds some entries differently."""
    dev = torch.device(device)
    base = cfg.rope_parameters["rope_theta"]
    dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    inv = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float, device=dev) / dim))
    if inv.device.type != dev.type or inv.dtype != torch.float32:
        raise RuntimeError(f"inv_freq computed on {inv.device}/{inv.dtype}, wanted {dev}/fp32")
    return inv


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
    """Install an ``inv_freq`` whose pow really ran on the CUDA ``device``
    (:func:`rope_inv_freq_formula`; HF evaluates the same expression on the CPU).  Fails closed
    on a non-CUDA device instead of silently reproducing the CPU table."""
    dev = torch.device(device)
    if dev.type != "cuda":
        raise ValueError(f"rope_inv_freq_on needs a CUDA device, got {dev}")
    rot = model.rotary_emb
    saved = (rot.inv_freq, rot.original_inv_freq)
    _set_rope_buffers(model, rope_inv_freq_formula(model.config, dev).to(saved[0].device))
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
    argmax: torch.Tensor | None  # (n_completion,) int64 CPU
    wall_ms: float
    top2_gap: torch.Tensor | None = None  # (n_completion,) fp32: top-1 minus top-2 logit
    kl: torch.Tensor | None = None  # (n_completion,) fp64 KL(reference || this), full vocab
    full_logp: torch.Tensor | None = None  # (n_completion, V) fp32 CPU, only when kept


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


def _head(
    h: torch.Tensor, targets: Sequence[int], lm_head: torch.Tensor, cfg: Any, head: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """(selected logp, argmax id, logits, log_softmax), all on the device (the timed part)."""
    logits = head_logits(h, lm_head, cfg, head)
    lp = torch.log_softmax(logits, dim=-1)
    tgt = torch.tensor(list(targets), device=h.device, dtype=torch.long)
    sel = lp.gather(-1, tgt[:, None])[:, 0]
    return sel, logits.argmax(-1), logits, lp


KL_ROW_CHUNK = 64


def kl_rows(ref_logp: torch.Tensor, logp: torch.Tensor) -> torch.Tensor:
    """Per-row ``KL(ref || this) = sum_v p_ref (log p_ref - log p)`` in fp64 (exactly 0 for
    identical rows); ``ref_logp`` may live on the CPU, rows are moved in chunks."""
    if ref_logp.shape != logp.shape:
        raise ValueError(f"KL shapes {tuple(ref_logp.shape)} vs {tuple(logp.shape)}")
    out = []
    for a in range(0, logp.shape[0], KL_ROW_CHUNK):
        r = ref_logp[a : a + KL_ROW_CHUNK].to(logp.device, torch.float64)
        v = logp[a : a + KL_ROW_CHUNK].double()
        out.append((r.exp() * (r - v)).sum(-1))
    return torch.cat(out).cpu() if out else torch.zeros(0, dtype=torch.float64)


def _score(
    sel: torch.Tensor,
    am: torch.Tensor,
    logits: torch.Tensor,
    lp: torch.Tensor,
    wall_ms: float,
    ref_full: torch.Tensor | None,
    keep_full: bool,
) -> SeqScore:
    """Untimed metrics of one sequence: top-2 gap, full-vocab KL vs ``ref_full``, kept rows."""
    top = logits.topk(2, dim=-1).values
    return SeqScore(
        logp=sel.cpu(),
        argmax=am.cpu(),
        wall_ms=wall_ms,
        top2_gap=(top[:, 0] - top[:, 1]).float().cpu(),
        kl=None if ref_full is None else kl_rows(ref_full, lp),
        full_logp=lp.float().cpu() if keep_full else None,
    )


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
# Points inside one decoder layer, in the order HF 5.17 runs them: ``<module>`` taps that
# module's OUTPUT, ``<module>.input`` its first INPUT (a pre-hook).  The attention core (QK^T,
# softmax, PV, after RoPE) is a function, not a module: ``self_attn.o_proj.input`` is its gated
# output ``attn * sigmoid(gate_proj(x))``, so a first difference there and not in q/k/v/gate
# proj is the attention core (or RoPE); one at ``self_attn.o_proj`` only is the o_proj GEMM.
SUBLAYER_POINTS = (
    "input_layernorm",
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.gate_proj",
    "self_attn.o_proj.input",
    "self_attn.o_proj",
    "post_attention_layernorm",
    "pre_feedforward_layernorm",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
    "post_feedforward_layernorm",
)
RESIDUAL_POINT_PREFIXES = ("embedding", "final_norm")


def is_residual_point(name: str) -> bool:
    """``embedding``, ``layer_<i>`` (decoder layer output) and ``final_norm``."""
    return name in RESIDUAL_POINT_PREFIXES or (name.startswith("layer_") and "." not in name)


class LayerTap:
    """Forward hooks copying ONE batch row to CPU after the embedding, after every decoder
    layer and after the final norm.  Options: ``rope=True`` also taps the bf16 RoPE tables
    (``rotary_emb`` outputs, points ``rope_cos`` / ``rope_sin``); ``sublayers=(i, ...)`` also
    taps :data:`SUBLAYER_POINTS` inside those layers (points ``layer_i.<point>``, placed before
    ``layer_i`` in execution order).  Repeated calls (chunked prefill, decode steps) are
    concatenated along the token axis."""

    def __init__(self, model: nn.Module, batch_row: int = 0, *, rope: bool = False,
                 sublayers: Sequence[int] = ()):
        self.row = int(batch_row)
        n = len(model.layers)
        subs = set(int(i) for i in sublayers)
        if any(i < 0 or i >= n for i in subs):
            raise ValueError(f"sublayers {sorted(subs)} outside 0..{n - 1}")
        # (name, module, what): what = output index, or "input" for the first forward input
        named: list[tuple[str, nn.Module, Any]] = [("embedding", model.embed_tokens, 0)]
        if rope:
            named += [("rope_cos", model.rotary_emb, 0), ("rope_sin", model.rotary_emb, 1)]
        for i, layer in enumerate(model.layers):
            if i in subs:
                for p in SUBLAYER_POINTS:
                    if p.endswith(".input"):
                        named.append((f"layer_{i}.{p}", layer.get_submodule(p[:-6]), "input"))
                    else:
                        named.append((f"layer_{i}.{p}", layer.get_submodule(p), 0))
            named.append((f"layer_{i}", layer, 0))
        named.append(("final_norm", model.norm, 0))
        self.names = [k for k, _, _ in named]
        self.parts: dict[str, list[torch.Tensor]] = {k: [] for k in self.names}
        self.handles = [
            m.register_forward_pre_hook(self._pre_hook(k)) if what == "input"
            else m.register_forward_hook(self._hook(k, what))
            for k, m, what in named
        ]

    def _keep(self, name: str, t: torch.Tensor) -> None:
        self.parts[name].append(t[self.row].detach().to("cpu", copy=True))

    def _hook(self, name: str, index: int = 0):
        def fn(_m: Any, _i: Any, out: Any) -> None:
            self._keep(name, out[index] if isinstance(out, tuple) else out)

        return fn

    def _pre_hook(self, name: str):
        def fn(_m: Any, args: Any) -> None:
            self._keep(name, args[0])

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
    *,
    rope: bool = False,
    sublayers: Sequence[int] = (),
) -> dict[str, torch.Tensor]:
    """Residual stream of ``seqs[target]`` at rows ``0 .. L-2`` (the rows every driver
    computes) for one driver; padded drivers run the whole batch and keep the target row.
    ``rope`` / ``sublayers`` add the :class:`LayerTap` points."""
    s = seqs[target]
    if driver in ("padded_right", "padded_left"):
        side = driver.split("_")[1]
        lmax = max(x.length for x in seqs)
        off = 0 if side == "right" else lmax - s.length
        tap = LayerTap(model, batch_row=target, rope=rope, sublayers=sublayers)
        try:
            hidden_padded(model, seqs, side)
        finally:
            got = tap.close()
        return {k: v[off : off + s.length - 1] for k, v in got.items()}
    tap = LayerTap(model, 0, rope=rope, sublayers=sublayers)
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
    ref_full: Sequence[torch.Tensor] | None = None,
    keep_full: bool = False,
) -> tuple[list[SeqScore], dict[str, float]]:
    """Selected-token fp32 logp + argmax of every completion token of every sequence.

    Returns per-sequence scores and ``{"wall_ms", "peak_mem_gib", "metrics_ms"}`` for the whole
    setting.  Per-sequence ``wall_ms`` is the forward(s) + head (logits, log_softmax, gather,
    argmax) of that sequence (padded drivers: the batch forward split evenly, head per
    sequence); ``wall_ms`` of the setting is the sum of those.  The analysis metrics (top-2 gap,
    full-vocab KL against ``ref_full[i]``, the kept ``full_logp`` rows when ``keep_full``) run
    after the timed region and are reported separately as ``metrics_ms``.
    """
    dev = lm_head.device
    if ref_full is not None and len(ref_full) != len(seqs):
        raise ValueError("ref_full needs one (n_completion, V) tensor per sequence")
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    timed_ms = 0.0
    metrics_ms = 0.0
    scores: list[SeqScore] = []

    def finish(i: int, s: SeqSpec, h: torch.Tensor, extra_ms: float) -> None:
        nonlocal timed_ms, metrics_ms
        _sync(dev)
        t1 = time.perf_counter()
        sel, am, logits, lp = _head(h, s.targets, lm_head, cfg, head)
        _sync(dev)
        ms = extra_ms + (time.perf_counter() - t1) * 1e3
        timed_ms += ms
        t2 = time.perf_counter()
        scores.append(_score(sel, am, logits, lp, ms,
                             None if ref_full is None else ref_full[i], keep_full))
        del logits, lp
        _sync(dev)
        metrics_ms += (time.perf_counter() - t2) * 1e3

    if driver in ("padded_right", "padded_left", "batched_decode_left"):
        _sync(dev)
        t0 = time.perf_counter()
        if driver == "batched_decode_left":
            rows = hidden_batched_decode(model, seqs)
        else:
            rows = hidden_padded(model, seqs, driver.split("_")[1])
        _sync(dev)
        fwd_ms = (time.perf_counter() - t0) * 1e3 / len(seqs)
        for i, (s, h) in enumerate(zip(seqs, rows)):
            finish(i, s, h, fwd_ms)
    else:
        for i, s in enumerate(seqs):
            _sync(dev)
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
            _sync(dev)
            finish(i, s, h, (time.perf_counter() - t0) * 1e3)
    stats = {
        "wall_ms": timed_ms,
        "metrics_ms": metrics_ms,
        "peak_mem_gib": (torch.cuda.max_memory_allocated(dev) / 2**30
                         if dev.type == "cuda" else 0.0),
    }
    return scores, stats


# --------------------------------------------------------------------------------------------
# Settings registry: one factor flipped per setting
# --------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Setting:
    """One HF variant.  It is scored against ``hf_baseline``; ``n_factors`` says how many
    factors that comparison flips.  When it is more than one, ``one_factor_ref`` names the
    setting that differs from this one in exactly ``one_factor`` and the merge derives that
    comparison from the stored exact per-token values (no extra forward)."""

    name: str
    factor: str
    factor_id: int
    description: str
    driver: str = "alone"
    chunk: int | None = None
    head: str = "bf16"
    contexts: tuple[str, ...] = ()  # names resolved by ``enter_contexts``
    n_factors: int = 1
    one_factor_ref: str | None = None
    one_factor: str | None = None

    def summary(self) -> dict[str, Any]:
        return {"setting": self.name, "factor": self.factor, "factor_id": self.factor_id,
                "description": self.description, "driver": self.driver, "chunk": self.chunk,
                "head": self.head, "contexts": list(self.contexts), "n_factors": self.n_factors,
                "one_factor_ref": self.one_factor_ref, "one_factor": self.one_factor}


HF_SETTINGS: tuple[Setting, ...] = (
    Setting("repeat_immediate", "repeat_null", 9, "baseline again, right after the baseline",
            n_factors=0),
    Setting("padded_batch9_right", "batch_composition", 1,
            "all 9 sequences in one right-padded batch (2-D mask)", driver="padded_right"),
    Setting("padded_batch9_left", "batch_composition", 1,
            "all 9 sequences in one left-padded batch (2-D mask, positions from 0)",
            driver="padded_left"),
    Setting("kv_cache_decode", "prefill_vs_decode", 2,
            "prompt prefill + teacher-forced 1-token decode steps with DynamicCache",
            driver="decode"),
    Setting("prefill_L_minus_1", "prefill_length", 3,
            "one-shot prefill of L-1 tokens (rollout prefill never feeds the last token; the "
            "GEMM M changes from L to L-1)", driver="prefill_minus_last"),
    Setting("chunked_prefill_64", "chunked_prefill", 3,
            "L-1 tokens in chunks of 64 through DynamicCache", driver="chunked", chunk=64,
            n_factors=2, one_factor_ref="prefill_L_minus_1", one_factor="chunked_prefill"),
    Setting("chunked_prefill_256", "chunked_prefill", 3,
            "L-1 tokens in chunks of 256 through DynamicCache", driver="chunked", chunk=256,
            n_factors=2, one_factor_ref="prefill_L_minus_1", one_factor="chunked_prefill"),
    Setting("chunked_prefill_1024", "chunked_prefill", 3,
            "L-1 tokens in chunks of 1024 through DynamicCache", driver="chunked", chunk=1024,
            n_factors=2, one_factor_ref="prefill_L_minus_1", one_factor="chunked_prefill"),
    Setting("sdpa", "attention_impl", 4, "torch SDPA instead of eager", contexts=("sdpa",)),
    Setting("tf32_on", "tf32", 5, "allow_tf32=True (cuBLAS + cuDNN)", contexts=("tf32_on",)),
    Setting("softcap_fp32", "logits_dtype", 6, "bf16 lm_head GEMM, soft cap in fp32",
            head="softcap_fp32"),
    Setting("logits_fp32", "logits_dtype", 6, "fp32 lm_head GEMM (no bf16 rounding) + fp32 cap",
            head="logits_fp32", n_factors=2, one_factor_ref="softcap_fp32",
            one_factor="lm_head_gemm_output_fp32"),
    Setting("rope_elementwise", "rope_table", 7,
            "cos/sin from fp32 elementwise position*inv_freq (5.19) instead of the 5.17 matmul",
            contexts=("rope_elementwise",)),
    Setting("rope_elementwise_tf32_on", "rope_table", 7,
            "two factors: TF32 on AND elementwise RoPE (is TF32's effect all in the RoPE GEMM?)",
            contexts=("rope_elementwise", "tf32_on"), n_factors=2,
            one_factor_ref="rope_elementwise", one_factor="tf32"),
    Setting("rope_inv_freq_cuda", "rope_table", 7,
            "inv_freq = 1/(theta ** (arange(0,128,2)/128)) evaluated by torch on the GPU "
            "(HF evaluates the same expression on the CPU)", contexts=("rope_inv_freq_cuda",)),
    Setting("bf16_reduced_reduction_off", "gemm_reduction", 10,
            "allow_bf16_reduced_precision_reduction=False (cuBLAS bf16 split-K in fp32)",
            contexts=("bf16_rr_off",)),
    Setting("rollout_like_batched_decode", "combination_rollout_like", 11,
            "factors 1+2 together: 9 prompts left-padded prefill + batched teacher-forced decode",
            driver="batched_decode_left", n_factors=2),
    Setting("rollout_like_batched_decode_sdpa", "combination_rollout_like", 11,
            "factors 1+2+4 together: the same with SDPA", driver="batched_decode_left",
            contexts=("sdpa",), n_factors=3, one_factor_ref="rollout_like_batched_decode",
            one_factor="attention_impl"),
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
                # raises on a CPU model: the factor is "pow on the GPU", never a CPU copy
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
    ref_full: Sequence[torch.Tensor] | None = None,
) -> tuple[list[SeqScore], dict[str, float]]:
    with enter_contexts(model, setting.contexts):
        return score_all(model, lm_head, cfg, seqs, driver=setting.driver, chunk=setting.chunk,
                         head=setting.head, ref_full=ref_full)


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
    if not (len(seqs) == len(base) == len(var)):
        raise ValueError("make_records needs one baseline and one variant score per sequence")
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
        rec.update(compare_tokens(b.logp, v.logp, b.argmax, v.argmax, base_gap=b.top2_gap,
                                  var_gap=v.top2_gap, kl=v.kl))
        if extra:
            rec.update(extra)
        out.append(rec)
    return out


def score_from_record(rec: dict[str, Any]) -> SeqScore:
    """The variant side of a record as a :class:`SeqScore` (exact logp from ``logp_hex``;
    argmax ids and top-2 gaps when the record has them; no KL, no timing)."""
    am, gap = rec.get("var_argmax"), rec.get("var_top2_gap")
    return SeqScore(
        logp=hex_to_fp32(rec["logp_hex"]),
        argmax=None if am is None else torch.tensor(am, dtype=torch.long),
        wall_ms=float(rec.get("wall_ms") or 0.0),
        top2_gap=None if gap is None else torch.tensor(gap, dtype=torch.float32),
    )


def derive_one_factor_records(
    records: Sequence[dict[str, Any]],
    settings: Sequence[dict[str, Any]],
    engine: str = "hf",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """For every setting summary with a ``one_factor_ref``, compare it against that reference
    setting token by token from the stored exact values (``logp_hex``, argmax ids, top-2 gaps).
    Returns ``(records, setting_summaries)`` named ``<setting>__vs__<ref>`` with
    ``derived_from_stored_values=True``; the full-vocabulary KL is ``None`` (it needs both
    distributions, which are not stored).  A reference that was not measured is skipped."""
    by: dict[str, dict[str, dict[str, Any]]] = {}
    for r in records:
        if r["engine"] == engine:
            by.setdefault(r["setting"], {})[r["sequence"]] = r
    out_recs: list[dict[str, Any]] = []
    out_sets: list[dict[str, Any]] = []
    for st in settings:
        ref = st.get("one_factor_ref")
        if st.get("engine") != engine or not ref or ref not in by or st["setting"] not in by:
            continue
        name = f"{st['setting']}__vs__{ref}"
        recs = []
        for sid, vr in by[st["setting"]].items():
            rr = by[ref].get(sid)
            if rr is None:
                continue
            b, v = score_from_record(rr), score_from_record(vr)
            rec = {"engine": engine, "factor": st.get("one_factor") or st["factor"],
                   "factor_id": st["factor_id"], "setting": name, "baseline": f"{engine}/{ref}",
                   "sequence": sid, "wall_ms": None, "derived_from_stored_values": True}
            rec.update(compare_tokens(b.logp, v.logp, b.argmax, v.argmax, base_gap=b.top2_gap,
                                      var_gap=v.top2_gap))
            recs.append(rec)
        if not recs:
            continue
        out_recs.extend(recs)
        summ = {"engine": engine, "setting": name, "factor": st.get("one_factor") or st["factor"],
                "factor_id": st["factor_id"],
                "description": f"{st['setting']} against {ref} (one factor: "
                               f"{st.get('one_factor')}); derived from the stored per-token values",
                "n_factors": 1, "one_factor_ref": None, "derived_from_stored_values": True,
                "variant_setting": st["setting"], "reference_setting": ref}
        summ.update(aggregate(recs))
        out_sets.append(summ)
    return out_recs, out_sets


def baseline_payload(seqs: Sequence[SeqSpec], scores: Sequence[SeqScore]) -> list[dict[str, Any]]:
    """Raw baseline numbers per sequence (logp values + exact hex, argmax, top-2 logit gap,
    targets)."""
    return [
        {
            "sequence": s.id,
            "completion_start": s.completion_start,
            "targets": s.targets,
            "logp": [float(x) for x in sc.logp.tolist()],
            "logp_hex": fp32_hex(sc.logp),
            "argmax": None if sc.argmax is None else [int(x) for x in sc.argmax.tolist()],
            "top2_gap": None if sc.top2_gap is None else [float(x) for x in sc.top2_gap.tolist()],
            "top1_tie": None if sc.top2_gap is None
            else [int(x == 0) for x in sc.top2_gap.tolist()],
            "wall_ms": sc.wall_ms,
        }
        for s, sc in zip(seqs, scores)
    ]


def token_char_spans(decode: Any, ids: Sequence[int]) -> tuple[str, list[tuple[int, int]]]:
    """Character span of every token of ``ids`` in ``decode(ids)``, from cumulative prefix
    decodes (byte-level BPE pieces can only be placed that way).  ``decode`` maps a list of ids
    to text (e.g. ``tokenizers.Tokenizer.decode``)."""
    spans: list[tuple[int, int]] = []
    prev = ""
    for i in range(len(ids)):
        cur = decode(list(ids[: i + 1]))
        spans.append((len(prev), len(cur)))
        prev = cur
    return prev, spans


def calc_annotation_flags(text: str, spans: Sequence[tuple[int, int]]) -> list[int]:
    """1 where a token's character span overlaps a GSM8K ``<<...>>`` calculator annotation."""
    import re

    calc = [(m.start(), m.end()) for m in re.finditer(r"<<.*?>>", text, flags=re.S)]
    return [int(any(a < ce and b > cs for cs, ce in calc)) for a, b in spans]


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
    "SUBLAYER_POINTS",
    "SeqScore",
    "SeqSpec",
    "Setting",
    "aggregate",
    "baseline_payload",
    "calc_annotation_flags",
    "chunk_plan",
    "compare_tokens",
    "derive_one_factor_records",
    "enter_contexts",
    "env_info",
    "fp32_hex",
    "head_logits",
    "is_residual_point",
    "hex_to_fp32",
    "hf_rope_inv_freq",
    "hidden_alone",
    "hidden_batched_decode",
    "hidden_chunked",
    "hidden_decode",
    "hidden_padded",
    "hidden_prefill_minus_last",
    "kl_rows",
    "layer_diff",
    "layer_trace",
    "load_text_model",
    "load_xrt_tokens",
    "make_records",
    "padded_batch",
    "rope_elementwise",
    "rope_inv_freq_formula",
    "rope_inv_freq_on",
    "run_setting",
    "score_all",
    "score_from_record",
    "tf32",
    "tf32_probe",
    "token_char_spans",
]

