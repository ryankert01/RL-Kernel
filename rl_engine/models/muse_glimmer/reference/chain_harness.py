# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Full-model chain gate for the Muse Glimmer strict reference (RFC #501 ``full_model_chain``).

``compare_paths`` runs the selected-token log-probabilities of one (prompt, completion) pair
through training forward, one-shot prefill, several chunked prefills and token-by-token decode on
the SAME strict operator set and asserts byte equality per token. When a path disagrees with the
training forward, ``localize_first_drift`` re-runs both with per-node capture and names the first
operator-trace node (and token) whose bits differ. ``compare_to_hf`` anchors the strict path to
the Hugging Face bf16 eager model at tolerance (NOT bitwise): max |Δlogit|, |Δlogp| and the
per-layer growth of the residual-stream difference.
"""

from __future__ import annotations

import gc
import math
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

import torch

from rl_engine.models.muse_glimmer.reference.strict_model import MuseGlimmerStrictModel
from rl_engine.models.muse_glimmer.reference.weights import MuseGlimmerWeights

WINDOW_BOUNDARY_CHUNKS = (2047, 2048, 2049)
_RAGGED_PATTERN = (1, 63, 100, 7, 129, 30)


# --------------------------------------------------------------------------------------------
# Path specs
# --------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class PathSpec:
    name: str
    mode: str
    chunk_sizes: tuple[int, ...] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "mode": self.mode,
            "chunk_sizes": None if self.chunk_sizes is None else list(self.chunk_sizes),
        }


def _cycle_chunks(n: int, pattern: Sequence[int]) -> tuple[int, ...]:
    out: list[int] = []
    i = 0
    while n > 0:
        c = min(int(pattern[i % len(pattern)]), n)
        out.append(c)
        n -= c
        i += 1
    return tuple(out)


def default_path_specs(
    prompt_len: int,
    completion_len: int,
    *,
    include_per_token: bool | None = None,
    window: int = 2048,
) -> list[PathSpec]:
    """Train, one-shot prefill, chunkings (prompt boundary, halves, 64-aligned, ragged, and the
    2047/2048/2049 window boundaries when the sequence is long enough) and decode."""
    n = prompt_len + completion_len - 1  # scored tokens
    specs = [PathSpec("train", "train"), PathSpec("prefill", "prefill")]
    if 0 < prompt_len < n:
        specs.append(PathSpec("chunked:prompt_boundary", "chunked", (prompt_len, n - prompt_len)))
    if n >= 2:
        specs.append(PathSpec("chunked:halves", "chunked", (n // 2, n - n // 2)))
    if n > 64:
        specs.append(PathSpec("chunked:block64", "chunked", _cycle_chunks(n, (64,))))
    specs.append(PathSpec("chunked:ragged", "chunked", _cycle_chunks(n, _RAGGED_PATTERN)))
    for boundary in (window - 1, window, window + 1):
        if n > boundary + 1:
            specs.append(
                PathSpec(f"chunked:window_boundary_{boundary}", "chunked", (boundary, n - boundary))
            )
    specs.append(PathSpec("decode", "decode"))
    if include_per_token is None:
        include_per_token = n <= 64
    if include_per_token:
        specs.append(PathSpec("per_token", "per_token"))
    return specs


# --------------------------------------------------------------------------------------------
# Running paths
# --------------------------------------------------------------------------------------------
@dataclass
class PathResult:
    spec: PathSpec
    logp: torch.Tensor  # (Lc,) fp32 CPU
    seconds: float
    peak_alloc_gib: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec": self.spec.to_dict(),
            "seconds": self.seconds,
            "peak_alloc_gib": self.peak_alloc_gib,
            "logp_head": [float(x) for x in self.logp[:4].tolist()],
            "logp_sum": float(self.logp.double().sum().item()),
            "logp": [float(x) for x in self.logp.tolist()],
        }


def run_path(
    model: MuseGlimmerStrictModel,
    prompt: torch.Tensor,
    completion: torch.Tensor,
    spec: PathSpec,
) -> PathResult:
    dev = model.device
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
        torch.cuda.reset_peak_memory_stats(dev)
    t0 = time.perf_counter()
    logp = model.logprobs(prompt, completion, mode=spec.mode, chunk_sizes=spec.chunk_sizes)
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    seconds = time.perf_counter() - t0
    peak = torch.cuda.max_memory_allocated(dev) / 2**30 if dev.type == "cuda" else 0.0
    return PathResult(spec=spec, logp=logp.detach().float().cpu(), seconds=seconds,
                      peak_alloc_gib=peak)


# --------------------------------------------------------------------------------------------
# Equality
# --------------------------------------------------------------------------------------------
@dataclass
class Equality:
    lhs: str
    rhs: str
    equal: bool
    n_tokens: int
    n_equal: int
    first_mismatch: int | None
    max_abs_diff: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compare_logp(lhs: PathResult, rhs: PathResult) -> Equality:
    a, b = lhs.logp, rhs.logp
    same = torch.eq(a, b) | (torch.isnan(a) & torch.isnan(b))
    n_equal = int(same.sum().item())
    first = None if n_equal == a.numel() else int((~same).nonzero()[0].item())
    diff = (a.double() - b.double()).abs()
    diff = torch.where(torch.isnan(diff), torch.zeros_like(diff), diff)
    return Equality(
        lhs=lhs.spec.name,
        rhs=rhs.spec.name,
        equal=bool(n_equal == a.numel()),
        n_tokens=int(a.numel()),
        n_equal=n_equal,
        first_mismatch=first,
        max_abs_diff=float(diff.max().item()) if diff.numel() else 0.0,
    )


# --------------------------------------------------------------------------------------------
# First-drift localization
# --------------------------------------------------------------------------------------------
@dataclass
class DriftReport:
    path: str
    reference: str
    node: str | None
    kind: str | None
    first_row: int | None
    rows_compared: int
    rows_differing: int
    max_abs_diff: float
    nodes_checked: int
    nodes_equal_before: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["nodes_equal_before"] = self.nodes_equal_before[-5:]
        return d


def _captured_rows(
    model: MuseGlimmerStrictModel, prompt: torch.Tensor, completion: torch.Tensor, spec: PathSpec
) -> dict[str, torch.Tensor]:
    model.begin_capture()
    try:
        model.logprobs(prompt, completion, mode=spec.mode, chunk_sizes=spec.chunk_sizes)
    finally:
        cap = model.end_capture()
    return cap


def localize_first_drift(
    model: MuseGlimmerStrictModel,
    prompt: torch.Tensor,
    completion: torch.Tensor,
    spec: PathSpec,
    *,
    reference: PathSpec = PathSpec("train", "train"),
) -> DriftReport:
    """Re-run ``reference`` and ``spec`` with node capture; report the first trace node whose
    rows differ (rows = token positions; the first ``T−1`` positions are comparable)."""
    ref_cap = _captured_rows(model, prompt, completion, reference)
    cap = _captured_rows(model, prompt, completion, spec)
    n_rows = int(prompt.numel() + completion.numel() - 1)
    equal_before: list[str] = []
    checked = 0
    for name in model.cfg.node_names():
        if name not in ref_cap or name not in cap:
            continue
        a = ref_cap[name][:n_rows]
        b = cap[name][:n_rows]
        rows = min(a.shape[0], b.shape[0])
        a, b = a[:rows], b[:rows]
        checked += 1
        same = torch.eq(a, b) | (torch.isnan(a) & torch.isnan(b))
        if a.dim() > 1:
            row_ok = same.flatten(1).all(dim=1)
        else:
            row_ok = same
        if bool(row_ok.all()):
            equal_before.append(name)
            continue
        bad = (~row_ok).nonzero()[:, 0]
        diff = (a.float() - b.float()).abs()
        diff = torch.where(torch.isnan(diff), torch.zeros_like(diff), diff)
        return DriftReport(
            path=spec.name,
            reference=reference.name,
            node=name,
            kind=model.cfg.node_kind(name),
            first_row=int(bad[0].item()),
            rows_compared=rows,
            rows_differing=int(bad.numel()),
            max_abs_diff=float(diff.max().item()),
            nodes_checked=checked,
            nodes_equal_before=equal_before,
        )
    return DriftReport(
        path=spec.name,
        reference=reference.name,
        node=None,
        kind=None,
        first_row=None,
        rows_compared=n_rows,
        rows_differing=0,
        max_abs_diff=0.0,
        nodes_checked=checked,
        nodes_equal_before=equal_before,
    )


# --------------------------------------------------------------------------------------------
# The gate
# --------------------------------------------------------------------------------------------
@dataclass
class ChainReport:
    prompt_len: int
    completion_len: int
    backend: str
    results: list[PathResult]
    equalities: list[Equality]
    all_equal: bool
    drifts: list[DriftReport]
    ops_provenance: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_len": self.prompt_len,
            "completion_len": self.completion_len,
            "backend": self.backend,
            "all_equal": self.all_equal,
            "results": [r.to_dict() for r in self.results],
            "equalities": [e.to_dict() for e in self.equalities],
            "drifts": [d.to_dict() for d in self.drifts],
            "ops_provenance": dict(self.ops_provenance),
        }

    def result(self, name: str) -> PathResult:
        for r in self.results:
            if r.spec.name == name:
                return r
        raise KeyError(name)

    def markdown(self) -> str:
        lines = [
            f"prompt {self.prompt_len} + completion {self.completion_len} tokens, backend "
            f"`{self.backend}`, all paths byte-equal to train: **{self.all_equal}**",
            "",
            "| path | chunk sizes | equal to train | equal tokens | max abs diff | seconds | "
            "peak GiB |",
            "|---|---|---|---|---|---|---|",
        ]
        eq_by = {e.rhs: e for e in self.equalities}
        for r in self.results:
            e = eq_by.get(r.spec.name)
            chunks = "-" if r.spec.chunk_sizes is None else _fmt_chunks(r.spec.chunk_sizes)
            if e is None:
                verdict, n_eq, mx = "(reference)", f"{r.logp.numel()}/{r.logp.numel()}", "0"
            else:
                verdict = "yes" if e.equal else "NO"
                n_eq = f"{e.n_equal}/{e.n_tokens}"
                mx = f"{e.max_abs_diff:.3e}"
            lines.append(
                f"| {r.spec.name} | {chunks} | {verdict} | {n_eq} | {mx} | {r.seconds:.2f} | "
                f"{r.peak_alloc_gib:.2f} |"
            )
        if self.drifts:
            lines += ["", "First drift per differing path:"]
            for d in self.drifts:
                lines.append(
                    f"- {d.path}: node `{d.node}` ({d.kind}) first row {d.first_row}, "
                    f"{d.rows_differing}/{d.rows_compared} rows differ, "
                    f"max |Δ|={d.max_abs_diff:.3e}"
                )
        return "\n".join(lines)


def _fmt_chunks(chunks: Sequence[int]) -> str:
    if len(chunks) > 6:
        return f"{list(chunks[:3])}…{list(chunks[-2:])} ({len(chunks)} chunks)"
    return str(list(chunks))


def compare_paths(
    model: MuseGlimmerStrictModel,
    prompt_ids: torch.Tensor,
    completion_ids: torch.Tensor,
    *,
    specs: Sequence[PathSpec] | None = None,
    localize: bool = True,
    log=None,
) -> ChainReport:
    """Run every path and compare selected-token logp bitwise against the training forward."""
    prompt = prompt_ids.to(model.device, dtype=torch.long)
    completion = completion_ids.to(model.device, dtype=torch.long)
    if specs is None:
        specs = default_path_specs(
            int(prompt.numel()), int(completion.numel()), window=model.cfg.sliding_window
        )
    results: list[PathResult] = []
    for spec in specs:
        res = run_path(model, prompt, completion, spec)
        results.append(res)
        if log:
            log(f"[{spec.name}] {res.seconds:.2f}s peak {res.peak_alloc_gib:.2f} GiB "
                f"logp[:3]={[round(x, 6) for x in res.logp[:3].tolist()]}")
    ref = results[0]
    equalities = [compare_logp(ref, r) for r in results[1:]]
    drifts: list[DriftReport] = []
    if localize:
        for eq, res in zip(equalities, results[1:], strict=True):
            if not eq.equal:
                if log:
                    log(f"[{res.spec.name}] differs from {ref.spec.name}; localizing first drift")
                drifts.append(localize_first_drift(model, prompt, completion, res.spec,
                                                   reference=ref.spec))
    return ChainReport(
        prompt_len=int(prompt.numel()),
        completion_len=int(completion.numel()),
        backend=model.ops.backend,
        results=results,
        equalities=equalities,
        all_equal=all(e.equal for e in equalities),
        drifts=drifts,
        ops_provenance=dict(model.ops.provenance),
    )


# --------------------------------------------------------------------------------------------
# Hugging Face anchor (tolerance, NOT bitwise)
# --------------------------------------------------------------------------------------------
def strict_reference_outputs(
    model: MuseGlimmerStrictModel, token_ids: torch.Tensor
) -> dict[str, Any]:
    """Training forward with capture: fp32 logits (T, V), logp (T−1,), residual stream per
    layer and the embedding / final hidden, all on CPU."""
    ids = token_ids.to(model.device, dtype=torch.long)
    t = int(ids.numel())
    targets = torch.cat([ids[1:], torch.full((1,), -1, device=model.device)])
    model.begin_capture()
    try:
        with torch.no_grad():
            out = model.forward_train(ids, targets, return_logits=True)
    finally:
        cap = model.end_capture()
    residuals = [cap[f"layers.{i}.residual_mlp"] for i in range(model.cfg.num_hidden_layers)]
    return {
        "logits": out["logits"].detach().float().cpu(),
        "logp": out["logp"].detach().float().cpu()[: t - 1],
        "embedding": cap["embedding"],
        "residuals": residuals,
        "final_hidden": cap["final_norm"],
        "token_ids": ids.cpu(),
    }


def _tensor_diff(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    a = a.float()
    b = b.float()
    d = (a - b).abs()
    rms = float(b.pow(2).mean().sqrt().item())
    return {
        "max_abs": float(d.max().item()),
        "mean_abs": float(d.mean().item()),
        "ref_rms": rms,
        "max_abs_over_rms": float(d.max().item()) / rms if rms > 0 else math.inf,
    }


def compare_strict_to_hf_outputs(
    ref: dict[str, Any],
    hf_logits: torch.Tensor,
    hf_hidden_states: Sequence[torch.Tensor] | None,
    *,
    completion_start: int,
) -> dict[str, Any]:
    """Diff strict outputs against HF logits (T, V) fp32 and hidden states (list, CPU)."""
    token_ids = ref["token_ids"]
    t = int(token_ids.numel())
    logits = ref["logits"]
    vocab = min(logits.shape[1], hf_logits.shape[1])
    logits, hf_logits = logits[:, :vocab], hf_logits[:, :vocab]
    targets = token_ids[1:]
    hf_logp_all = torch.log_softmax(hf_logits.double(), dim=-1)
    hf_logp = hf_logp_all[torch.arange(t - 1), targets].float()
    strict_logp = ref["logp"]
    comp = slice(completion_start - 1, t - 1)
    report: dict[str, Any] = {
        "positions": t,
        "completion_start": completion_start,
        "logits_all_positions": _tensor_diff(logits, hf_logits),
        "logits_completion_positions": _tensor_diff(logits[comp], hf_logits[comp]),
        "argmax_agreement_all": float((logits.argmax(-1) == hf_logits.argmax(-1)).float().mean()),
        "argmax_agreement_completion": float(
            (logits[comp].argmax(-1) == hf_logits[comp].argmax(-1)).float().mean()
        ),
        "logp_all": _tensor_diff(strict_logp, hf_logp),
        "logp_completion": _tensor_diff(strict_logp[comp], hf_logp[comp]),
        "strict_logp_completion_mean": float(strict_logp[comp].mean()),
        "hf_logp_completion_mean": float(hf_logp[comp].mean()),
    }
    if hf_hidden_states is not None:
        hs = [h.reshape(-1, h.shape[-1]).float() for h in hf_hidden_states]
        layers: list[dict[str, Any]] = []
        layers.append({"layer": "embedding", **_tensor_diff(ref["embedding"], hs[0])})
        n_layers = len(ref["residuals"])
        # HF: hs[0]=embeddings, hs[i+1]=layer i output for i < L−1, hs[L]=final norm output.
        for i in range(min(n_layers - 1, len(hs) - 2)):
            layers.append({"layer": i, **_tensor_diff(ref["residuals"][i], hs[i + 1])})
        layers.append({"layer": "final_norm", **_tensor_diff(ref["final_hidden"], hs[-1])})
        report["per_layer"] = layers
    return report


def compare_to_hf(
    ref: dict[str, Any],
    *,
    checkpoint: str,
    device: torch.device | str,
    completion_start: int,
    dtype: torch.dtype = torch.bfloat16,
    log=None,
    save_path=None,
) -> dict[str, Any]:
    """Load transformers' ``MuseGlimmerForConditionalGeneration`` (text-only input, eager
    attention, bf16), run the same tokens and diff against the strict reference outputs.

    The strict model should already be freed: the HF model needs ~56 GiB on the device.
    ``save_path`` additionally stores the HF logits and hidden states (fp32 CPU) for later
    cross-comparisons (e.g. against an fp32 HF run).
    """
    from transformers import MuseGlimmerForConditionalGeneration

    dev = torch.device(device)
    t0 = time.perf_counter()
    model = MuseGlimmerForConditionalGeneration.from_pretrained(
        checkpoint, dtype=dtype, attn_implementation="eager"
    )
    model.eval()
    model.to(dev)
    load_s = time.perf_counter() - t0
    if log:
        log(f"[hf] loaded in {load_s:.1f}s; running {ref['token_ids'].numel()} tokens")
    ids = ref["token_ids"].to(dev)[None]
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    t1 = time.perf_counter()
    with torch.no_grad():
        out = model(input_ids=ids, output_hidden_states=True, use_cache=False)
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    fwd_s = time.perf_counter() - t1
    hf_logits = out.logits[0].float().cpu()
    hidden = [h[0].detach().cpu() for h in out.hidden_states]
    del out, model
    gc.collect()
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    if save_path is not None:
        torch.save({"logits": hf_logits, "hidden_states": hidden, "dtype": str(dtype),
                    "token_ids": ref["token_ids"]}, save_path)
        if log:
            log(f"[hf] saved {save_path}")
    report = compare_strict_to_hf_outputs(ref, hf_logits, hidden, completion_start=completion_start)
    report["hf"] = {
        "class": "MuseGlimmerForConditionalGeneration",
        "attn_implementation": "eager",
        "dtype": str(dtype).replace("torch.", ""),
        "load_seconds": load_s,
        "forward_seconds": fwd_s,
        "hidden_states": len(hidden),
    }
    return report


def hf_text_model_from_weights(weights: MuseGlimmerWeights, device: torch.device | str):
    """Instantiate transformers' ``MuseGlimmerTextModel`` with ``weights.cfg`` (tiny configs)
    and load the same random bf16 weights; returns ``(text_model, lm_head_weight)``."""
    from transformers import MuseGlimmerTextConfig
    from transformers.models.muse_glimmer.modeling_muse_glimmer import MuseGlimmerTextModel

    kwargs = weights.cfg.hf_text_config_kwargs()
    cfg = MuseGlimmerTextConfig(**kwargs)
    cfg._attn_implementation = "eager"
    model = MuseGlimmerTextModel(cfg)
    sd = weights.hf_text_state_dict()
    lm_head = sd.pop("lm_head.weight")
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [m for m in missing if "rotary_emb" not in m]
    if missing or unexpected:
        raise RuntimeError(
            f"HF tiny state dict mismatch: missing={missing} unexpected={unexpected}"
        )
    model = model.to(device=torch.device(device), dtype=torch.bfloat16).eval()
    return model, lm_head.to(device=torch.device(device), dtype=torch.bfloat16)


@torch.no_grad()
def hf_text_logits(
    model, lm_head: torch.Tensor, token_ids: torch.Tensor, cfg
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """HF-style head: bf16 linear, ×multiplier, ÷T, tanh, ×T (all bf16) → fp32 logits (T, V)."""
    ids = token_ids.to(lm_head.device, dtype=torch.long)[None]
    out = model(input_ids=ids, output_hidden_states=True, use_cache=False)
    h = out.last_hidden_state[0]
    z = torch.nn.functional.linear(h, lm_head)
    z = z * cfg.output_multiplier
    z = z / cfg.final_logit_softcapping
    z = torch.tanh(z)
    z = z * cfg.final_logit_softcapping
    return z.float().cpu(), [hs[0].detach().cpu() for hs in out.hidden_states]


__all__ = [
    "ChainReport",
    "DriftReport",
    "Equality",
    "PathResult",
    "PathSpec",
    "WINDOW_BOUNDARY_CHUNKS",
    "compare_logp",
    "compare_paths",
    "compare_strict_to_hf_outputs",
    "compare_to_hf",
    "default_path_specs",
    "hf_text_logits",
    "hf_text_model_from_weights",
    "localize_first_drift",
    "run_path",
    "strict_reference_outputs",
]
