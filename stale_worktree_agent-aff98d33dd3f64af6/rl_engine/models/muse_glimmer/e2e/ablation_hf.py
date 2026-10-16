# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""One-factor mismatch probes for Muse-Glimmer-30B on the HF eager bf16 reference (RFC #501,
row ``museglimmer_ablation``).

Every factor is a context manager that monkeypatches the *installed*
``transformers.models.muse_glimmer.modeling_muse_glimmer`` module (or a config attribute) and
restores it on exit.  The baseline is the unpatched eager bf16 text model with TF32 off.  The
attention/head patches are implemented as ONE replacement attention forward driven by a knob
set; with every knob at its default the replacement is bitwise equal to the HF forward (the
harness asserts this before running any factor).

Metric per variant: selected-token log-probabilities of the 64 completion tokens versus the
baseline -- max |dlogp|, mean |dlogp|, fraction of positions whose argmax changes, index of the
first token with |dlogp| > 1e-3, and whether any decoder-layer output or logit is non-finite.

Cast points of the reference path (HF 5.17.0): q/k RMSNorm in fp32 -> bf16; q * 3.87 in bf16;
RoPE in bf16 (cos/sin cast to bf16); scores = bf16 matmul * 128^-0.5 in bf16; softmax fp32 ->
bf16; gate = bf16 sigmoid, bf16 multiply; logits bf16 -> * mult -> / T -> tanh -> * T in bf16;
log-softmax of the bf16 logits in fp32 (harness).
"""

from __future__ import annotations

import contextlib
import json
import math
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from transformers.models.muse_glimmer import modeling_muse_glimmer as hf

TEXT_PREFIX = "model.language_model."
PROMPT_TOKENS = 512
COMPLETION_TOKENS = 64
DRIFT_THRESHOLD = 1e-3

# --------------------------------------------------------------------------------------------
# Knobs for the replacement attention forward / LM head.
# --------------------------------------------------------------------------------------------


@dataclass
class Knobs:
    """Every value at its default reproduces the HF 5.17.0 arithmetic bitwise."""

    gate: str = "elementwise"  # elementwise | off | per_head_mean
    q_scale: float | None = None  # None -> module.qk_scale_factor
    scale_k: bool = False
    fold_scale_into_softmax: bool = False  # q unscaled, eager scaling = q_scale * d^-0.5
    fp32_scores: bool = False  # scores in fp32 (Megatron/flash style), softmax fp32
    single_rounding_qk: bool = False  # fp32 norm -> scale -> RoPE -> one bf16 cast
    # LM head
    softcap: str = "hf"  # hf | swapped | off
    head_fp32: bool = False  # keep the lm_head GEMM output in fp32 through the soft cap
    softcap_fp32: bool = False  # bf16 logits, soft cap evaluated in fp32
    rope_table: str = "matmul"  # matmul (5.17.0) | elementwise (5.19.0)


KNOBS = Knobs()


def _rmsnorm_fp32(x: torch.Tensor, eps: float) -> torch.Tensor:
    xf = x.float()
    return xf * torch.pow(xf.pow(2).mean(-1, keepdim=True) + eps, -0.5)


def _apply_gate(attn_output: torch.Tensor, gate_pre: torch.Tensor, head_dim: int) -> torch.Tensor:
    if KNOBS.gate == "off":
        return attn_output
    g = torch.sigmoid(gate_pre)
    if KNOBS.gate == "per_head_mean":
        shape = g.shape
        g = g.view(*shape[:-1], shape[-1] // head_dim, head_dim).mean(-1, keepdim=True)
        g = g.expand(*shape[:-1], shape[-1] // head_dim, head_dim).reshape(shape)
    return attn_output * g


def patched_attention_forward(
    self: hf.MuseGlimmerTextAttention,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
    attention_mask: torch.Tensor | None = None,
    past_key_values: Any = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Replacement for ``MuseGlimmerTextAttention.forward`` driven by ``KNOBS``."""
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)
    q = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    k = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    v = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    q_scale = self.qk_scale_factor if KNOBS.q_scale is None else KNOBS.q_scale
    scaling = self.scaling
    apply_q_scale = not KNOBS.fold_scale_into_softmax
    if KNOBS.fold_scale_into_softmax:
        scaling = float(q_scale) * self.scaling

    if KNOBS.single_rounding_qk:
        qf = _rmsnorm_fp32(q, self.qk_norm.eps)
        kf = _rmsnorm_fp32(k, self.qk_norm.eps)
        if apply_q_scale:
            qf = qf * q_scale
        if KNOBS.scale_k:
            kf = kf * q_scale
        if position_embeddings is not None:
            cos, sin = position_embeddings
            # fp32 cos/sin: recompute from the attached fp32 table when present
            cos32 = getattr(self, "_ablation_cos32", None)
            sin32 = getattr(self, "_ablation_sin32", None)
            if cos32 is None:
                cos32, sin32 = cos.float(), sin.float()
            qf, kf = hf.apply_rotary_pos_emb(qf, kf, cos32, sin32)
        q = qf.to(hidden_states.dtype)
        k = kf.to(hidden_states.dtype)
    else:
        q = self.qk_norm(q)
        if apply_q_scale:
            q = q * q_scale
        k = self.qk_norm(k)
        if KNOBS.scale_k:
            k = k * q_scale
        if position_embeddings is not None:
            cos, sin = position_embeddings
            q, k = hf.apply_rotary_pos_emb(q, k, cos, sin)

    if past_key_values is not None:
        k, v = past_key_values.update(k, v, self.layer_idx)

    if KNOBS.fp32_scores:
        ks = hf.repeat_kv(k, self.num_key_value_groups)
        vs = hf.repeat_kv(v, self.num_key_value_groups)
        w = torch.matmul(q.float(), ks.float().transpose(2, 3)) * scaling
        if attention_mask is not None:
            w = w + attention_mask.float()
        w = torch.softmax(w, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(w, vs).transpose(1, 2).contiguous()
    else:
        interface = hf.ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, hf.eager_attention_forward
        )
        attn_output, _ = interface(
            self,
            q,
            k,
            v,
            attention_mask,
            dropout=0.0,
            scaling=scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = _apply_gate(attn_output, self.gate_proj(hidden_states), self.head_dim)
    return self.o_proj(attn_output), None


def lm_head_softcap(hidden: torch.Tensor, lm_head: nn.Linear, cfg: Any) -> torch.Tensor:
    """HF 5.17.0 head: ``T * tanh(z * mult / T)`` in bf16, modified by ``KNOBS``."""
    mult = cfg.output_multiplier
    cap = cfg.final_logit_softcapping
    if KNOBS.head_fp32:
        z = torch.matmul(hidden.float(), lm_head.weight.float().t())
    else:
        z = lm_head(hidden)
        if KNOBS.softcap_fp32:
            z = z.float()
    if KNOBS.softcap == "off":
        return z * mult
    if KNOBS.softcap == "swapped":
        return torch.tanh(z / cap) * cap * mult
    return torch.tanh(z * mult / cap) * cap


def rope_forward_elementwise(self: Any, x: torch.Tensor, position_ids: torch.Tensor):
    """transformers 5.19.0-style table: fp32 elementwise multiply instead of a matmul."""
    inv = self.inv_freq.float()
    freqs = position_ids[:, :, None].float() * inv[None, None, :]
    emb = torch.cat((freqs, freqs), dim=-1)
    return (emb.cos() * self.attention_scaling).to(x.dtype), (emb.sin() * self.attention_scaling).to(
        x.dtype
    )


# --------------------------------------------------------------------------------------------
# Factor context managers.
# --------------------------------------------------------------------------------------------


@contextlib.contextmanager
def _knobs(**values: Any) -> Iterator[None]:
    saved = {k: getattr(KNOBS, k) for k in values}
    orig = hf.MuseGlimmerTextAttention.forward
    hf.MuseGlimmerTextAttention.forward = patched_attention_forward
    for k, v in values.items():
        setattr(KNOBS, k, v)
    try:
        yield
    finally:
        for k, v in saved.items():
            setattr(KNOBS, k, v)
        hf.MuseGlimmerTextAttention.forward = orig


@contextlib.contextmanager
def _config_attr(model: nn.Module, name: str, value: Any) -> Iterator[None]:
    cfg = model.config
    saved = getattr(cfg, name)
    setattr(cfg, name, value)
    try:
        yield
    finally:
        setattr(cfg, name, saved)


@contextlib.contextmanager
def _module_attr(modules: Sequence[nn.Module], name: str, value: Any) -> Iterator[None]:
    saved = [getattr(m, name) for m in modules]
    for m in modules:
        setattr(m, name, value)
    try:
        yield
    finally:
        for m, s in zip(modules, saved):
            setattr(m, name, s)


@contextlib.contextmanager
def _centered_norm_plain_w() -> Iterator[None]:
    cls = hf.MuseGlimmerTextCenteredRMSNorm
    orig = cls.forward

    def fwd(self: Any, x: torch.Tensor) -> torch.Tensor:
        return (self._norm(x.float()) * self.weight.float()).type_as(x)

    cls.forward = fwd
    try:
        yield
    finally:
        cls.forward = orig


@contextlib.contextmanager
def _rope_elementwise() -> Iterator[None]:
    cls = hf.MuseGlimmerTextRotaryEmbedding
    orig = cls.forward
    cls.forward = torch.no_grad()(rope_forward_elementwise)
    try:
        yield
    finally:
        cls.forward = orig


@contextlib.contextmanager
def _tf32(enabled: bool) -> Iterator[None]:
    a, b = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = enabled
    torch.backends.cudnn.allow_tf32 = enabled
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = a
        torch.backends.cudnn.allow_tf32 = b


@contextlib.contextmanager
def _attn_impl(model: nn.Module, impl: str) -> Iterator[None]:
    saved = model.config._attn_implementation
    model.config._attn_implementation = impl
    try:
        yield
    finally:
        model.config._attn_implementation = saved


def _attn_modules(model: nn.Module) -> list[nn.Module]:
    return [m for m in model.modules() if isinstance(m, hf.MuseGlimmerTextAttention)]


def _post_norms(model: nn.Module) -> list[nn.Module]:
    return [
        getattr(layer, n)
        for layer in model.layers
        for n in ("post_attention_layernorm", "post_feedforward_layernorm")
    ]


def factor_registry(model: nn.Module) -> dict[str, Callable[[], Any]]:
    """name -> zero-arg callable returning a context manager that applies the factor."""
    n = model.config.num_hidden_layers
    theta = model.config.rope_parameters["rope_theta"]
    embed_norm = model.embed_tokens.embed_norm
    reg: dict[str, Callable[[], Any]] = {
        "baseline": lambda: contextlib.nullcontext(),
        "patched_identity": lambda: _knobs(),
        "rope_on_global_layers": lambda: _config_attr(model, "layer_rope_theta", [theta] * n),
        "nope_all_layers": lambda: _config_attr(model, "layer_rope_theta", [0] * n),
        "sliding_window_256": lambda: _config_attr(model, "sliding_window", 256),
        "sliding_window_512": lambda: _config_attr(model, "sliding_window", 512),
        "sliding_window_1024": lambda: _config_attr(model, "sliding_window", 1024),
        "sliding_window_4096": lambda: _config_attr(model, "sliding_window", 4096),
        "sliding_window_none": lambda: _config_attr(model, "sliding_window", 1 << 30),
        "gate_off": lambda: _knobs(gate="off"),
        "gate_per_head_mean": lambda: _knobs(gate="per_head_mean"),
        "qk_scale_43.78": lambda: _knobs(q_scale=43.7840518911),
        "qk_scale_on_k_too": lambda: _knobs(scale_k=True),
        "qk_scale_folded_bf16": lambda: _knobs(fold_scale_into_softmax=True),
        "qk_scale_folded_fp32_scores": lambda: _knobs(
            fold_scale_into_softmax=True, fp32_scores=True
        ),
        "fp32_scores_only": lambda: _knobs(fp32_scores=True),
        "embed_norm_off": lambda: _module_attr([embed_norm], "forward", lambda x: x),
        "embed_norm_eps_1e-6": lambda: _module_attr([embed_norm], "eps", 1e-6),
        "softcap_swapped": lambda: _knobs(softcap="swapped"),
        "softcap_off": lambda: _knobs(softcap="off"),
        "softcap_fp32": lambda: _knobs(softcap_fp32=True),
        "post_norm_eps_1e-5": lambda: _module_attr(_post_norms(model), "eps", 1e-5),
        "centered_norm_plain_w": lambda: _centered_norm_plain_w(),
        "qk_single_rounding": lambda: _knobs(single_rounding_qk=True),
        "lm_head_fp32": lambda: _knobs(head_fp32=True),
        "tf32_on": lambda: _tf32(True),
        "sdpa": lambda: _attn_impl(model, "sdpa"),
        "rope_table_elementwise": lambda: _rope_elementwise(),
    }
    return reg


# Factors that change the *driver* rather than the arithmetic (batching / chunking).
DRIVER_FACTORS = ("batch_of_8", "chunked_prefill_256")

# --------------------------------------------------------------------------------------------
# Model loading (text weights only, bf16, vision tower never instantiated).
# --------------------------------------------------------------------------------------------


def load_text_model(checkpoint: str | Path, device: str = "cuda") -> tuple[nn.Module, nn.Linear]:
    """Build ``MuseGlimmerTextModel`` on the meta device and fill it from the safetensors shards.

    Only keys under ``model.language_model.`` and ``lm_head.weight`` are read.  Non-persistent
    RoPE buffers are recomputed on CPU in fp32 (same formula as HF init) and copied over.
    """
    from safetensors import safe_open
    from transformers import AutoConfig

    checkpoint = Path(checkpoint)
    cfg = AutoConfig.from_pretrained(checkpoint).text_config
    cfg._attn_implementation = "eager"
    with torch.device("meta"):
        model = hf.MuseGlimmerTextModel(cfg)
        lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
    model = model.to_empty(device=device).to(torch.bfloat16)
    lm_head = lm_head.to_empty(device=device).to(torch.bfloat16)
    index = json.loads((checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    by_file: dict[str, list[str]] = {}
    for key, fname in index.items():
        if key.startswith(TEXT_PREFIX) or key == "lm_head.weight":
            by_file.setdefault(fname, []).append(key)
    state = model.state_dict()
    loaded = set()
    with torch.no_grad():
        for fname, keys in by_file.items():
            with safe_open(str(checkpoint / fname), framework="pt", device=device) as f:
                for key in keys:
                    t = f.get_tensor(key)
                    if key == "lm_head.weight":
                        lm_head.weight.copy_(t)
                        continue
                    state[key[len(TEXT_PREFIX) :]].copy_(t)
                    loaded.add(key[len(TEXT_PREFIX) :])
    missing = set(state) - loaded
    if missing:
        raise RuntimeError(f"text weights missing from checkpoint: {sorted(missing)[:5]}")
    inv_freq, _ = hf.MuseGlimmerTextRotaryEmbedding.compute_default_rope_parameters(cfg)
    with torch.no_grad():
        model.rotary_emb.inv_freq.copy_(inv_freq)
        model.rotary_emb.original_inv_freq.copy_(inv_freq)
    model.eval()
    lm_head.eval()
    return model, lm_head


# --------------------------------------------------------------------------------------------
# Prompts and scoring.
# --------------------------------------------------------------------------------------------

QUESTIONS = [
    "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in"
    " May. How many clips did Natalia sell altogether in April and May?",
    "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting."
    " How much did she earn?",
    "Betty is saving money for a new wallet which costs $100. Betty has only half of the money"
    " she needs. Her parents decided to give her $15 for that purpose, and her grandparents"
    " twice as much as her parents. How much more money does Betty need to buy the wallet?",
    "Julie is reading a 120-page book. Yesterday, she was able to read 12 pages and today, she"
    " read twice as many pages as yesterday. If she wants to read half of the remaining pages"
    " tomorrow, how many pages should she read?",
    "James writes a 3-page letter to 2 different friends twice a week. How many pages does he"
    " write a year?",
    "Mark has a garden with flowers. He planted plants of three different colors in it. Ten of"
    " them are yellow, and there are 80% more of those in purple. There are only 25% as many"
    " green flowers as there are yellow and purple flowers. How many flowers does Mark have?",
    "Albert is wondering how much pizza he can eat in one day. He buys 2 large pizzas and 2"
    " small pizzas. A large pizza has 16 slices and a small pizza has 8 slices. If he eats it"
    " all, how many pieces does he eat that day?",
    "Ken created a care package to send to his brother. He placed a box on a scale, and then he"
    " poured into the box enough jelly beans to bring the weight to 2 pounds. Then, he added"
    " enough brownies to cause the weight to triple. Next, he added another 2 pounds of jelly"
    " beans. And finally, he added enough gummy worms to double the weight once again. What"
    " was the final weight of the box of goodies, in pounds?",
]

FILLER = (
    "Work through the problem step by step, state every intermediate quantity with its unit, "
    "check the arithmetic once more before answering, and finish with the final number alone. "
)


def build_prompts(tokenizer: Any, prompt_len: int = PROMPT_TOKENS) -> list[list[int]]:
    """Chat-templated GSM8K-style prompts padded with a fixed instruction passage to exactly
    ``prompt_len`` tokens (the passage is grown/shrunk one word at a time)."""
    out = []
    for q in QUESTIONS:
        words = (FILLER * 40).split()
        n = len(words)
        ids: list[int] = []
        for _ in range(4000):
            msg = [{"role": "user", "content": " ".join(words[:n]) + "\n\n" + q}]
            ids = tokenizer.apply_chat_template(msg, tokenize=True, add_generation_prompt=True)
            if hasattr(ids, "input_ids"):
                ids = ids["input_ids"]
            ids = list(ids)
            if len(ids) == prompt_len:
                break
            n += -1 if len(ids) > prompt_len else 1
        if len(ids) != prompt_len:
            raise RuntimeError(f"could not hit {prompt_len} tokens for prompt {q[:30]!r}")
        out.append(ids)
    return out


@torch.no_grad()
def greedy_complete(
    model: nn.Module, lm_head: nn.Linear, prompt: list[int], n_new: int = COMPLETION_TOKENS
) -> list[int]:
    """Greedy continuation (no cache: re-runs the growing sequence; fine for 64 tokens)."""
    device = model.embed_tokens.weight.device
    ids = list(prompt)
    for _ in range(n_new):
        x = torch.tensor([ids], device=device)
        h = model(input_ids=x, use_cache=False).last_hidden_state[:, -1:]
        logits = lm_head_softcap(h, lm_head, model.config)
        ids.append(int(logits[0, -1].float().argmax()))
    return ids


@dataclass
class Scored:
    logp: torch.Tensor  # (64,) fp32 selected-token logp
    argmax: torch.Tensor  # (64,) int64
    finite: bool
    wall_s: float


@torch.no_grad()
def score_sequences(
    model: nn.Module,
    lm_head: nn.Linear,
    seqs: Sequence[Sequence[int]],
    batched: bool = False,
    chunk: int | None = None,
) -> list[Scored]:
    """Selected-token logp of the last COMPLETION_TOKENS tokens for every sequence."""
    device = model.embed_tokens.weight.device
    finite = {"ok": True}

    def hook(_m: Any, _i: Any, out: Any) -> None:
        t = out[0] if isinstance(out, tuple) else out
        if not bool(torch.isfinite(t).all()):
            finite["ok"] = False

    handles = [layer.register_forward_hook(hook) for layer in model.layers]
    try:
        groups = [list(seqs)] if batched else [[s] for s in seqs]
        results: list[Scored] = []
        for group in groups:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            x = torch.tensor(group, device=device)
            if chunk is None:
                h = model(input_ids=x, use_cache=False).last_hidden_state
                h = h[:, -COMPLETION_TOKENS - 1 : -1]
            else:
                from transformers import DynamicCache

                cache = DynamicCache(config=model.config)
                outs = []
                for s in range(0, x.shape[1], chunk):
                    piece = x[:, s : s + chunk]
                    pos = torch.arange(s, s + piece.shape[1], device=device)[None]
                    o = model(
                        input_ids=piece, position_ids=pos, past_key_values=cache, use_cache=True
                    )
                    outs.append(o.last_hidden_state)
                h = torch.cat(outs, dim=1)[:, -COMPLETION_TOKENS - 1 : -1]
            logits = lm_head_softcap(h, lm_head, model.config).float()
            fin = finite["ok"] and bool(torch.isfinite(logits).all())
            lp = torch.log_softmax(logits, dim=-1)
            tgt = x[:, -COMPLETION_TOKENS:]
            sel = lp.gather(-1, tgt[..., None])[..., 0]
            am = logits.argmax(-1)
            torch.cuda.synchronize()
            wall = time.perf_counter() - t0
            for b in range(x.shape[0]):
                results.append(Scored(sel[b].cpu(), am[b].cpu(), fin, wall / x.shape[0]))
        return results
    finally:
        for hnd in handles:
            hnd.remove()


def compare(base: list[Scored], var: list[Scored]) -> dict[str, Any]:
    d = torch.cat([abs(b.logp - v.logp) for b, v in zip(base, var)])
    am = torch.cat([(b.argmax != v.argmax) for b, v in zip(base, var)]).float()
    firsts = []
    for b, v in zip(base, var):
        idx = torch.nonzero(abs(b.logp - v.logp) > DRIFT_THRESHOLD)
        firsts.append(int(idx[0]) if idx.numel() else None)
    present = [f for f in firsts if f is not None]
    return {
        "max_abs_dlogp": float(d.max()),
        "mean_abs_dlogp": float(d.mean()),
        "argmax_change_frac": float(am.mean()),
        "first_drift_index": min(present) if present else None,
        "first_drift_per_prompt": firsts,
        "nonfinite": not all(v.finite for v in var),
        "wall_s_per_prompt": sum(v.wall_s for v in var) / len(var),
        "bitwise_equal": bool(d.max() == 0.0),
    }


# --------------------------------------------------------------------------------------------
# Driver.
# --------------------------------------------------------------------------------------------


@dataclass
class AblationResult:
    checkpoint: str
    factors: dict[str, dict[str, Any]] = field(default_factory=dict)
    completions: list[list[int]] = field(default_factory=list)


def run_ablation(
    checkpoint: str | Path,
    prompts: Sequence[Sequence[int]] | None,
    factors: Sequence[str] | None,
    out_json: str | Path,
    device: str = "cuda",
) -> AblationResult:
    """Load the text model once, build baseline completions, then run every factor.

    Results are written to ``out_json`` after every factor so a cut-off run leaves a partial
    file; ``factors=None`` runs the whole registry plus the driver factors.
    """
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    from transformers import AutoTokenizer

    out_json = Path(out_json)
    t0 = time.perf_counter()
    model, lm_head = load_text_model(checkpoint, device)
    load_s = time.perf_counter() - t0
    if prompts is None:
        tok = AutoTokenizer.from_pretrained(checkpoint)
        prompts = build_prompts(tok)
    seqs = [greedy_complete(model, lm_head, p) for p in prompts]
    result = AblationResult(checkpoint=str(checkpoint), completions=[s[PROMPT_TOKENS:] for s in seqs])
    base = score_sequences(model, lm_head, seqs)
    result.factors["baseline"] = {
        "load_s": load_s,
        "wall_s_per_prompt": sum(b.wall_s for b in base) / len(base),
        "mean_logp": float(torch.cat([b.logp for b in base]).mean()),
        "nonfinite": not all(b.finite for b in base),
    }
    reg = factor_registry(model)
    names = list(factors) if factors else [n for n in reg if n != "baseline"] + list(DRIVER_FACTORS)
    # Sanity gate: the replacement forward with default knobs must be bitwise HF.
    with reg["patched_identity"]():
        ident = compare(base, score_sequences(model, lm_head, seqs))
    result.factors["patched_identity"] = ident
    if not ident["bitwise_equal"]:
        raise RuntimeError("replacement attention forward is not bitwise equal to HF at defaults")
    for name in names:
        if name == "patched_identity":
            continue
        if name in DRIVER_FACTORS:
            kw = {"batched": True} if name == "batch_of_8" else {"chunk": 256}
            var = score_sequences(model, lm_head, seqs, **kw)
        else:
            with reg[name]():
                var = score_sequences(model, lm_head, seqs)
        result.factors[name] = compare(base, var)
        out_json.write_text(json.dumps(result.__dict__, indent=2))
    # Restore check: after every factor exits, the baseline must still be bitwise reproducible.
    result.factors["restore_check"] = compare(base, score_sequences(model, lm_head, seqs))
    out_json.write_text(json.dumps(result.__dict__, indent=2))
    return result


def rank_factors(factors: dict[str, dict[str, Any]]) -> list[tuple[str, float]]:
    rows = [(k, v["max_abs_dlogp"]) for k, v in factors.items() if "max_abs_dlogp" in v]
    return sorted(rows, key=lambda kv: -kv[1] if math.isfinite(kv[1]) else -float("inf"))
