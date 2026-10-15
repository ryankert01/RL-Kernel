# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Op-class profiling of the Muse Glimmer strict model and of the Hugging Face bf16 baseline.

Method: CUDA events. Every op call of the strict composition (``strict_model._run``) runs inside
a named region (``MuseGlimmerStrictModel.set_timer``); the HF model gets the same regions from
module forward hooks plus two function wrappers (``apply_rotary_pos_emb`` and the ``sdpa``
attention interface). A region records one CUDA event on entry and one on exit on the current
stream; regions nest, and each region's EXCLUSIVE time (its elapsed time minus its direct
children's) is charged to its class, so the classes of one forward sum exactly to the root
region (the whole call). Times are GPU-timeline intervals: a region whose kernels are short is
charged the CPU launch gaps inside it as well (decode at batch 1 is launch-bound), which is why
every configuration is also timed "clean" (no regions, events around the call only).

Classes (``kind`` = ``global`` / ``local`` layer, or ``model`` outside the layer stack):
  embedding, norm (strict: the fused post-norm + residual + pre-norm op; HF: the four norm
  modules), layer_glue (HF only: residual adds outside the norm modules), qkvg_gemm (q, k, v,
  attention-gate projections), qknorm_rope (strict: rope table + fused Q/K norm, scale, RoPE;
  HF: rotary table, qk_norm module calls, apply_rotary_pos_emb), kv_write (strict paged modes:
  paged cache append + block-table view), attention (strict: the gated attention kernel incl.
  the sigmoid gate; HF: the sdpa interface call), attn_glue (HF only: transposes, q scale,
  cache update, sigmoid gate multiply, reshapes inside the attention module), o_gemm,
  ffn_gemm (gate, up, down), ffn_act (SwiGLU activation: strict swiglu kernel; HF act_fn and
  multiply), lm_head_gemm, softcap_logprob (strict: 8-tile softcap + logsumexp tree +
  selected-token logp; HF: softcap ops in the model forward + log_softmax + gather),
  kv_evict (strict paged commit: lengths + window eviction), other (anything else inside the
  call: Python glue, masks, views).
"""

from __future__ import annotations

import gc
import math
import statistics
import time
from contextlib import contextmanager
from typing import Any, Callable, Iterator, Sequence

import torch

CLASSES = (
    "embedding",
    "norm",
    "layer_glue",
    "qkvg_gemm",
    "qknorm_rope",
    "kv_write",
    "attention",
    "attn_glue",
    "o_gemm",
    "ffn_gemm",
    "ffn_act",
    "lm_head_gemm",
    "softcap_logprob",
    "kv_evict",
    "other",
)
KINDS = ("model", "global", "local")
# Task-level groups for the stacked bars (class -> group; attention split by layer kind).
GROUPS = (
    "embedding",
    "norms_residual",
    "qkvgo_gemm",
    "qknorm_rope",
    "attention_global",
    "attention_local",
    "ffn_gemm",
    "ffn_act",
    "lm_head_softcap_logprob",
    "other",
)
_GROUP_OF = {
    "embedding": "embedding",
    "norm": "norms_residual",
    "layer_glue": "norms_residual",
    "qkvg_gemm": "qkvgo_gemm",
    "o_gemm": "qkvgo_gemm",
    "qknorm_rope": "qknorm_rope",
    "ffn_gemm": "ffn_gemm",
    "ffn_act": "ffn_act",
    "lm_head_gemm": "lm_head_softcap_logprob",
    "softcap_logprob": "lm_head_softcap_logprob",
    "kv_evict": "other",
    "other": "other",
}
_ROOT = "forward"
ANNOT_PREFIX = "mgrg"


def group_of(cls: str, kind: str) -> str:
    if cls in ("attention", "kv_write", "attn_glue"):
        return "attention_local" if kind == "local" else "attention_global"
    return _GROUP_OF.get(cls, "other")


# --------------------------------------------------------------------------------------------
# Region timer
# --------------------------------------------------------------------------------------------
class RegionTimer:
    """Nested named regions timed with CUDA events (see the module docstring)."""

    def __init__(self) -> None:
        self._recs: list[dict[str, Any]] = []
        self._stack: list[int] = []
        # annotate=True also opens a torch.profiler record_function range per region (named
        # ANNOT_PREFIX|cls|kind|layer) so kernels can be attributed to regions afterwards.
        self.annotate = False

    def reset(self) -> None:
        if self._stack:
            raise RuntimeError("reset() inside an open region")
        self._recs = []

    def begin(self, cls: str, kind: str | None = None, layer: int | None = None) -> int:
        parent = self._stack[-1] if self._stack else None
        if parent is not None:
            if kind is None:
                kind = self._recs[parent]["kind"]
            if layer is None:
                layer = self._recs[parent]["layer"]
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        idx = len(self._recs)
        rec = {"cls": cls, "kind": kind or "model", "layer": layer,
               "parent": parent, "start": start, "end": end, "annot": None}
        if self.annotate:
            rf = torch.autograd.profiler.record_function(
                f"{ANNOT_PREFIX}|{cls}|{rec['kind']}|{-1 if layer is None else layer}"
            )
            rf.__enter__()
            rec["annot"] = rf
        self._recs.append(rec)
        self._stack.append(idx)
        start.record()
        return idx

    def end(self, idx: int) -> None:
        if not self._stack or self._stack[-1] != idx:
            raise RuntimeError(f"region {idx} closed out of order (stack {self._stack[-3:]})")
        rec = self._recs[idx]
        rec["end"].record()
        if rec["annot"] is not None:
            rec["annot"].__exit__(None, None, None)
            rec["annot"] = None
        self._stack.pop()

    @contextmanager
    def region(
        self, cls: str, kind: str | None = None, layer: int | None = None
    ) -> Iterator[None]:
        idx = self.begin(cls, kind, layer)
        try:
            yield
        finally:
            self.end(idx)

    def wrap(
        self, fn: Callable[..., Any], cls: str, kind: str | None = None, layer: int | None = None
    ) -> Callable[..., Any]:
        """A timed proxy of a GEMM callable. It forwards every attribute, so duck-typed
        dispatch on the wrapped object (``GoldenGemm.row_parallel`` / ``.matmul_kn``) is
        preserved; those methods are timed too. Bits are unchanged."""
        return _TimedCallable(self, fn, cls, kind, layer)

    def resolve(self) -> list[dict[str, Any]]:
        """Synchronize and return one row per region with inclusive and exclusive ms."""
        if self._stack:
            raise RuntimeError("resolve() inside an open region")
        torch.cuda.synchronize()
        incl = [r["start"].elapsed_time(r["end"]) for r in self._recs]
        child = [0.0] * len(self._recs)
        for i, r in enumerate(self._recs):
            if r["parent"] is not None:
                child[r["parent"]] += incl[i]
        rows = []
        for i, r in enumerate(self._recs):
            rows.append({
                "cls": r["cls"], "kind": r["kind"], "layer": r["layer"],
                "root": r["parent"] is None, "inclusive_ms": incl[i],
                "exclusive_ms": incl[i] - child[i],
            })
        return rows


class _TimedCallable:
    _TIMED_METHODS = ("row_parallel", "matmul_kn")

    def __init__(self, timer: RegionTimer, fn: Any, cls: str, kind: str | None,
                 layer: int | None) -> None:
        self._timer, self._fn = timer, fn
        self._cls, self._kind, self._layer = cls, kind, layer

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        with self._timer.region(self._cls, self._kind, self._layer):
            return self._fn(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._fn, name)  # AttributeError when the wrapped object lacks it
        if name in self._TIMED_METHODS and callable(attr):
            def timed(*args: Any, **kwargs: Any) -> Any:
                with self._timer.region(self._cls, self._kind, self._layer):
                    return attr(*args, **kwargs)

            return timed
        return attr


def summarize_regions(rows: Sequence[dict[str, Any]], num_layers: int) -> dict[str, Any]:
    """Aggregate one call's regions: per (kind, class) and per (class, layer) exclusive ms."""
    roots = [r for r in rows if r["root"]]
    if len(roots) != 1:
        raise RuntimeError(f"expected exactly one root region, got {len(roots)}")
    total = roots[0]["inclusive_ms"]
    by_kind: dict[str, dict[str, float]] = {k: {c: 0.0 for c in CLASSES} for k in KINDS}
    per_layer: dict[str, list[float]] = {c: [0.0] * num_layers for c in CLASSES}
    for r in rows:
        cls = "other" if r["cls"] in (_ROOT, "model_glue") else r["cls"]
        if cls not in by_kind["model"]:
            cls = "other"
        kind = r["kind"] if r["kind"] in KINDS else "model"
        by_kind[kind][cls] += r["exclusive_ms"]
        if r["layer"] is not None and kind != "model" and 0 <= r["layer"] < num_layers:
            per_layer[cls][r["layer"]] += r["exclusive_ms"]
    return {"total_ms": total, "by_kind_ms": by_kind, "per_layer_ms": per_layer}


def _median(xs: Sequence[float]) -> float:
    return float(statistics.median(xs)) if xs else float("nan")


def combine_repeats(summaries: Sequence[dict[str, Any]], num_layers: int) -> dict[str, Any]:
    """Median over repeats of every class / layer cell; plus group sums and fractions."""
    totals = [s["total_ms"] for s in summaries]
    by_kind = {
        k: {c: _median([s["by_kind_ms"][k][c] for s in summaries]) for c in CLASSES}
        for k in KINDS
    }
    per_layer = {
        c: [_median([s["per_layer_ms"][c][i] for s in summaries]) for i in range(num_layers)]
        for c in CLASSES
    }
    by_class = {c: sum(by_kind[k][c] for k in KINDS) for c in CLASSES}
    by_group = {g: 0.0 for g in GROUPS}
    for k in KINDS:
        for c in CLASSES:
            by_group[group_of(c, k)] += by_kind[k][c]
    sum_ms = sum(by_class.values())
    return {
        "total_ms_repeats": totals,
        "total_ms_median": _median(totals),
        "sum_of_classes_ms": sum_ms,
        "by_kind_class_ms": by_kind,
        "by_class_ms": by_class,
        "by_class_frac": {c: (v / sum_ms if sum_ms else 0.0) for c, v in by_class.items()},
        "by_group_ms": by_group,
        "by_group_frac": {g: (v / sum_ms if sum_ms else 0.0) for g, v in by_group.items()},
        "per_layer_ms": per_layer,
    }


def per_layer_type_means(by_kind: dict[str, dict[str, float]], n_global: int,
                         n_local: int) -> dict[str, dict[str, float]]:
    """ms per ONE layer of each type (sum over the layers of that type / their count)."""
    return {
        "global": {c: by_kind["global"][c] / max(1, n_global) for c in CLASSES},
        "local": {c: by_kind["local"][c] / max(1, n_local) for c in CLASSES},
    }


# --------------------------------------------------------------------------------------------
# torch.profiler view: GPU kernel time per region (no CPU gaps)
# --------------------------------------------------------------------------------------------
_LAUNCH_PREFIXES = ("cudaLaunch", "cuLaunch", "cudaMemcpy", "cudaMemset", "cuMemcpy",
                    "cuMemset", "cudaGraphLaunch")


def attributed_kernel_profile(
    fn: Callable[[], Any],
    timer: RegionTimer,
    install: Callable[[RegionTimer | None], None],
    *,
    num_layers: int,
    root_cls: str = _ROOT,
    top: int = 12,
) -> dict[str, Any]:
    """One extra call under torch.profiler with every region also opened as a record_function
    range. Each GPU kernel (and memcpy/memset) is charged, with its device duration, to the
    innermost region that was open on the CPU when its launch API call was made (CUPTI
    correlation id kernel <-> launch). Gives the per-class GPU kernel time without the CPU
    launch gaps that the CUDA-event view includes; ``gpu_busy_ms`` vs the wall time shows how
    launch-bound a configuration is."""
    from torch.profiler import ProfilerActivity, profile

    torch.cuda.synchronize()
    install(timer)
    timer.reset()
    timer.annotate = True
    try:
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            idx = timer.begin(root_cls, "model", None)
            fn()
            timer.end(idx)
            torch.cuda.synchronize()
    finally:
        timer.annotate = False
        install(None)
    timer.reset()
    ranges: list[tuple[float, float, str, str, int]] = []
    launches: dict[int, float] = {}
    kernels: list[tuple[int, float, str]] = []
    for e in prof.events():
        dt = str(getattr(e, "device_type", ""))
        if dt.endswith("CPU"):
            if e.name.startswith(ANNOT_PREFIX + "|"):
                _, cls, kind, layer = e.name.split("|")
                ranges.append((e.time_range.start, e.time_range.end, cls, kind, int(layer)))
            elif e.name.startswith(_LAUNCH_PREFIXES):
                launches[int(e.id)] = e.time_range.start
        elif dt.endswith("CUDA"):
            if getattr(e, "is_user_annotation", False):
                continue
            kernels.append((int(e.id), float(e.time_range.elapsed_us()), e.name))
    by_kind = {k: {c: 0.0 for c in CLASSES} for k in KINDS}
    per_layer = {c: [0.0] * num_layers for c in CLASSES}
    names: dict[str, list[float]] = {}
    unattributed = 0.0
    timed = []
    for corr, dur, name in kernels:
        names.setdefault(name, []).append(dur)
        t = launches.get(corr)
        if t is None:
            unattributed += dur
        else:
            timed.append((t, dur))
    timed.sort()
    ranges.sort(key=lambda r: (r[0], -r[1]))
    stack: list[tuple[float, float, str, str, int]] = []
    ri = 0
    for t, dur in timed:
        while ri < len(ranges) and ranges[ri][0] <= t:
            while stack and stack[-1][1] < ranges[ri][0]:
                stack.pop()
            stack.append(ranges[ri])
            ri += 1
        while stack and stack[-1][1] < t:
            stack.pop()
        if not stack:
            unattributed += dur
            continue
        _, _, cls, kind, layer = stack[-1]
        cls = cls if cls in by_kind["model"] else "other"
        kind = kind if kind in KINDS else "model"
        by_kind[kind][cls] += dur / 1e3
        if kind != "model" and 0 <= layer < num_layers:
            per_layer[cls][layer] += dur / 1e3
    busy = sum(sum(v) for v in names.values()) / 1e3
    by_group = {g: 0.0 for g in GROUPS}
    for k in KINDS:
        for c in CLASSES:
            by_group[group_of(c, k)] += by_kind[k][c]
    ranked = sorted(names.items(), key=lambda kv: -sum(kv[1]))[:top]
    return {
        "method": "torch.profiler CUDA activity, one call, kernels charged to the innermost "
                  "region open at launch (correlation id)",
        "gpu_busy_ms": busy,
        "unattributed_ms": unattributed / 1e3,
        "kernel_launches": len(kernels),
        "launch_events_matched": len(timed),
        "by_kind_class_gpu_ms": by_kind,
        "by_class_gpu_ms": {c: sum(by_kind[k][c] for k in KINDS) for c in CLASSES},
        "by_group_gpu_ms": by_group,
        "per_layer_gpu_ms": per_layer,
        "top_kernels": [
            {"name": k[:160], "calls": len(v), "ms": sum(v) / 1e3} for k, v in ranked
        ],
    }


# --------------------------------------------------------------------------------------------
# Measurement helpers
# --------------------------------------------------------------------------------------------
def _gib(x: float) -> float:
    return x / 2**30


def timed_call(fn: Callable[[], Any]) -> tuple[Any, float]:
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    out = fn()
    e.record()
    torch.cuda.synchronize()
    return out, s.elapsed_time(e)


def measure(
    fn: Callable[[], Any],
    timer: RegionTimer,
    install: Callable[[RegionTimer | None], None],
    *,
    num_layers: int,
    warmup: int,
    repeats: int,
    root_cls: str = _ROOT,
    profiler_check: bool = False,
) -> dict[str, Any]:
    """Warm up, then ``repeats`` instrumented calls (regions) and ``repeats`` clean calls."""
    dev = torch.device("cuda")
    install(None)
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    mem_before = torch.cuda.memory_allocated(dev)
    torch.cuda.reset_peak_memory_stats(dev)
    clean = []
    for _ in range(repeats):
        _, ms = timed_call(fn)
        clean.append(ms)
    peak = torch.cuda.max_memory_allocated(dev)
    summaries = []
    install(timer)
    try:
        for _ in range(repeats):
            timer.reset()
            idx = timer.begin(root_cls, "model", None)
            fn()
            timer.end(idx)
            summaries.append(summarize_regions(timer.resolve(), num_layers))
    finally:
        install(None)
    out = combine_repeats(summaries, num_layers)
    out["clean_total_ms_repeats"] = clean
    out["clean_total_ms_median"] = _median(clean)
    out["instrumentation_overhead_frac"] = (
        out["total_ms_median"] / out["clean_total_ms_median"] - 1.0
        if out["clean_total_ms_median"] > 0 else float("nan")
    )
    out["mem_allocated_before_gib"] = _gib(mem_before)
    out["peak_allocated_gib"] = _gib(peak)
    out["activation_peak_gib"] = _gib(peak - mem_before)
    if profiler_check:
        try:
            out["kernel_profile"] = attributed_kernel_profile(
                fn, timer, install, num_layers=num_layers, root_cls=root_cls
            )
        except Exception as exc:  # the cross-check must never sink the measurement
            timer.annotate = False
            install(None)
            timer._stack.clear()
            timer._recs = []
            out["kernel_profile"] = {"error": f"{type(exc).__name__}: {exc}"}
    return out


# --------------------------------------------------------------------------------------------
# Strict model
# --------------------------------------------------------------------------------------------
def _tokens(n: int, seed: int, vocab: int = 200000) -> torch.Tensor:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    ids = torch.randint(0, vocab, (n,), generator=gen)
    ids[0] = 200000
    return ids


def profile_strict(
    model: Any,
    *,
    prefill_lens: Sequence[int] = (512, 2048, 4096),
    decode_batches: Sequence[int] = (1, 16),
    decode_context: int = 2048,
    warmup: int = 1,
    repeats: int = 3,
    decode_repeats: int = 5,
    paged_prefill: bool = True,
    batch_invariance: bool = True,
    profiler_check: bool = True,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Profile train-forward (varlen) and one-shot paged prefill at each ``prefill_lens`` and
    batched decode at ``decode_context``; also the decode batch-invariance gate (G2)."""
    from rl_engine.models.muse_glimmer.reference.strict_model import Session

    cfg = model.cfg
    nl = cfg.num_hidden_layers
    timer = RegionTimer()
    runs: list[dict[str, Any]] = []
    dev = model.device

    for t in prefill_lens:
        ids = _tokens(t, seed=1000 + t).to(dev)
        targets = torch.cat([ids[1:], torch.full((1,), -1, device=dev)])

        def train_call(ids=ids, targets=targets):
            with torch.no_grad():
                return model.forward_train(ids, targets)["logp"]

        log(f"[strict] train forward T={t}")
        res = measure(train_call, timer, model.set_timer, num_layers=nl, warmup=warmup,
                      repeats=repeats, profiler_check=profiler_check)
        runs.append({"phase": "prefill_train", "T": t, "batch": 1, "context": 0,
                     "tokens": t, **res})
        log(f"  total {res['total_ms_median']:.1f} ms (clean {res['clean_total_ms_median']:.1f})"
            f" peak {res['peak_allocated_gib']:.2f} GiB")
        if paged_prefill:
            # fresh sessions are allocated up front so the cache zero-fill is not timed
            sessions = [model.new_session(t)
                        for _ in range(warmup + 2 * repeats + int(profiler_check))]

            def paged_call(ids=ids, targets=targets, sessions=sessions):
                return model.prefill(sessions.pop(), ids, targets=targets)

            log(f"[strict] paged prefill T={t}")
            res = measure(paged_call, timer, model.set_timer, num_layers=nl, warmup=warmup,
                          repeats=repeats, profiler_check=profiler_check)
            del sessions
            runs.append({"phase": "prefill_paged", "T": t, "batch": 1, "context": 0,
                         "tokens": t, **res})
            log(f"  total {res['total_ms_median']:.1f} ms "
                f"(clean {res['clean_total_ms_median']:.1f})")
        del ids, targets
        gc.collect()
        torch.cuda.empty_cache()

    gate: dict[str, Any] = {}
    if decode_batches:
        bmax = max(decode_batches)
        steps = len(decode_batches) * (warmup + 2 * decode_repeats + 2) + 4
        cache = model.new_batch_cache(decode_context + steps, bmax)
        t0 = time.perf_counter()
        prompts = []
        for b in range(bmax):
            ids = _tokens(decode_context, seed=5000 + b).to(dev)
            tgt = torch.full((decode_context,), -1, dtype=torch.long, device=dev)
            tgt[-1] = int(ids[0].item())
            model.prefill(Session(cache=cache, seq=b), ids, targets=tgt)
            prompts.append(ids)
        torch.cuda.synchronize()
        log(f"[strict] filled {bmax} sequences x {decode_context} tokens in "
            f"{time.perf_counter() - t0:.1f}s")
        gen = torch.Generator(device="cpu").manual_seed(77)
        if batch_invariance:
            gate = decode_batch_invariance(model, cache, bmax, gen)
            log(f"[strict] decode batch invariance: {gate['summary']}")
        for b in sorted(decode_batches):
            seqs = list(range(b))
            toks = torch.randint(0, 200000, (b,), generator=gen)
            tgts = torch.randint(0, 200000, (b,), generator=gen)

            def decode_call(seqs=seqs, toks=toks, tgts=tgts):
                return model.decode_batch(cache, seqs, toks, tgts, commit=True)

            log(f"[strict] decode batch {b} context {decode_context}")
            res = measure(decode_call, timer, model.set_timer, num_layers=nl, warmup=warmup,
                          repeats=decode_repeats, profiler_check=profiler_check)
            runs.append({"phase": "decode", "T": 1, "batch": b, "context": decode_context,
                         "tokens": b, **res})
            log(f"  step {res['total_ms_median']:.2f} ms (clean "
                f"{res['clean_total_ms_median']:.2f}) peak {res['peak_allocated_gib']:.2f} GiB")
        del cache
        gc.collect()
        torch.cuda.empty_cache()
    n_global = sum(1 for i in range(nl) if not cfg.is_local(i))
    for r in runs:
        r["per_layer_type_ms"] = per_layer_type_means(r["by_kind_class_ms"], n_global,
                                                      nl - n_global)
        r["tokens_per_s_clean"] = r["tokens"] / (r["clean_total_ms_median"] / 1e3)
    return {"target": "strict_triton", "runs": runs, "decode_batch_invariance": gate,
            "layers": {"global": n_global, "local": nl - n_global}}


@torch.no_grad()
def decode_batch_invariance(model: Any, cache: Any, bmax: int,
                            gen: torch.Generator) -> dict[str, Any]:
    """Contract G2 at the full-model level: the same decode step of sequence b gives the same
    logp bits at batch 1, inside a batch of ``bmax`` and inside a permuted batch of 4."""
    toks = torch.randint(0, 200000, (bmax,), generator=gen)
    tgts = torch.randint(0, 200000, (bmax,), generator=gen)
    full = model.decode_batch(cache, list(range(bmax)), toks, tgts, commit=False).cpu()
    single = torch.cat([
        model.decode_batch(cache, [b], toks[b : b + 1], tgts[b : b + 1], commit=False).cpu()
        for b in range(bmax)
    ])
    perm = [3, 0, bmax - 1, bmax // 2] if bmax >= 4 else list(range(bmax))[::-1]
    sub = model.decode_batch(cache, perm, toks[perm], tgts[perm], commit=False).cpu()
    eq_full = int(torch.eq(full, single).sum())
    eq_perm = int(torch.eq(sub, single[perm]).sum())
    return {
        "batch": bmax,
        "batch1_vs_batchN_equal": eq_full,
        "permuted_batch4_vs_batch1_equal": eq_perm,
        "permutation": perm,
        "all_equal": eq_full == bmax and eq_perm == len(perm),
        "max_abs_diff": float((full.double() - single.double()).abs().max()),
        "logp_batchN": [float(x) for x in full.tolist()],
        "logp_hex_batchN": [f"{x & 0xFFFFFFFF:08x}" for x in full.view(torch.int32).tolist()],
        "summary": f"batch1 == batch{bmax}: {eq_full}/{bmax}, permuted batch-{len(perm)}: "
                   f"{eq_perm}/{len(perm)}",
    }


# --------------------------------------------------------------------------------------------
# Hugging Face bf16 baseline
# --------------------------------------------------------------------------------------------
class HFInstrument:
    """Forward hooks + function wrappers that open/close RegionTimer regions on an HF
    ``MuseGlimmerForConditionalGeneration``; ``install(timer)`` / ``install(None)``."""

    def __init__(self, hf_model: Any) -> None:
        self.model = hf_model
        self.timer: RegionTimer | None = None
        self._handles: list[Any] = []
        self._patched: list[tuple[Any, str, Any]] = []
        self._open: dict[int, list[int]] = {}

    def _hook_module(self, module: Any, cls: str, kind: str | None, layer: int | None) -> None:
        key = id(module)

        def pre(_mod, _args):
            if self.timer is not None:
                self._open.setdefault(key, []).append(self.timer.begin(cls, kind, layer))

        def post(_mod, _args, _out):
            if self.timer is not None and self._open.get(key):
                self.timer.end(self._open[key].pop())

        self._handles.append(module.register_forward_pre_hook(pre))
        self._handles.append(module.register_forward_hook(post))

    def _wrap_fn(self, owner: Any, name: str, cls: str, kind_of: Callable[..., Any] | None):
        orig = getattr(owner, name) if not isinstance(owner, dict) else owner[name]

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            if self.timer is None:
                return orig(*args, **kwargs)
            kind, layer = kind_of(*args) if kind_of is not None else (None, None)
            idx = self.timer.begin(cls, kind, layer)
            try:
                return orig(*args, **kwargs)
            finally:
                self.timer.end(idx)

        return orig, wrapped

    def attach(self) -> None:
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        from transformers.models.muse_glimmer import modeling_muse_glimmer as mm

        lm = self.model.model.language_model
        # exclusive time of the top module = the softcap ops after lm_head (+ slicing)
        self._hook_module(self.model, "softcap_logprob", "model", None)
        self._hook_module(self.model.model, "other", "model", None)
        self._hook_module(lm, "other", "model", None)
        self._hook_module(lm.embed_tokens, "embedding", "model", None)
        self._hook_module(lm.rotary_emb, "qknorm_rope", "model", None)
        self._hook_module(lm.norm, "norm", "model", None)
        self._hook_module(self.model.lm_head, "lm_head_gemm", "model", None)
        for i, layer in enumerate(lm.layers):
            kind = "local" if layer.self_attn.is_local_attention else "global"
            self._hook_module(layer, "layer_glue", kind, i)
            for nm in ("input_layernorm", "post_attention_layernorm",
                       "pre_feedforward_layernorm", "post_feedforward_layernorm"):
                self._hook_module(getattr(layer, nm), "norm", kind, i)
            at = layer.self_attn
            self._hook_module(at, "attn_glue", kind, i)
            for nm in ("q_proj", "k_proj", "v_proj", "gate_proj"):
                self._hook_module(getattr(at, nm), "qkvg_gemm", kind, i)
            self._hook_module(at.qk_norm, "qknorm_rope", kind, i)
            self._hook_module(at.o_proj, "o_gemm", kind, i)
            self._hook_module(layer.mlp, "ffn_act", kind, i)
            for nm in ("gate_proj", "up_proj", "down_proj"):
                self._hook_module(getattr(layer.mlp, nm), "ffn_gemm", kind, i)

        orig_rope, rope = self._wrap_fn(mm, "apply_rotary_pos_emb", "qknorm_rope", None)
        mm.apply_rotary_pos_emb = rope
        self._patched.append((mm, "apply_rotary_pos_emb", orig_rope))

        impl = self.model.config.text_config._attn_implementation
        orig_attn = ALL_ATTENTION_FUNCTIONS[impl]

        def attn_kind(module, *_args):
            return ("local" if module.is_local_attention else "global"), module.layer_idx

        def attn(*args: Any, **kwargs: Any) -> Any:
            if self.timer is None:
                return orig_attn(*args, **kwargs)
            kind, layer = attn_kind(*args)
            idx = self.timer.begin("attention", kind, layer)
            try:
                return orig_attn(*args, **kwargs)
            finally:
                self.timer.end(idx)

        had_local = impl in ALL_ATTENTION_FUNCTIONS._local_mapping
        prev_local = ALL_ATTENTION_FUNCTIONS._local_mapping.get(impl)
        ALL_ATTENTION_FUNCTIONS[impl] = attn
        self._patched.append((ALL_ATTENTION_FUNCTIONS, impl, (had_local, prev_local)))

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []
        for owner, name, orig in self._patched:
            if owner.__class__.__name__ == "AttentionInterface":
                had_local, prev = orig
                if had_local:
                    owner._local_mapping[name] = prev
                else:
                    owner._local_mapping.pop(name, None)
            else:
                setattr(owner, name, orig)
        self._patched = []

    def install(self, timer: RegionTimer | None) -> None:
        self.timer = timer


def hf_logprob(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Native selected-token logp from HF's (softcapped, bf16) logits: fp32 log_softmax."""
    lp = torch.log_softmax(logits.float(), dim=-1)
    return lp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)


def profile_hf(
    hf_model: Any,
    *,
    prefill_lens: Sequence[int] = (512, 2048, 4096),
    decode_batches: Sequence[int] = (1, 16),
    decode_context: int = 2048,
    warmup: int = 1,
    repeats: int = 3,
    decode_repeats: int = 5,
    profiler_check: bool = True,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """The same configurations on HF bf16 (sdpa): forward over T tokens with full logits and
    the fp32 log_softmax + gather of the next tokens; decode steps on a DynamicCache."""
    from transformers import DynamicCache

    text_cfg = hf_model.config.text_config
    nl = text_cfg.num_hidden_layers
    dev = next(hf_model.parameters()).device
    inst = HFInstrument(hf_model)
    inst.attach()
    timer = RegionTimer()
    runs: list[dict[str, Any]] = []

    def logprob_region(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if inst.timer is None:
            return hf_logprob(logits, targets)
        with inst.timer.region("softcap_logprob", "model", None):
            return hf_logprob(logits, targets)

    try:
        for t in prefill_lens:
            ids = _tokens(t, seed=1000 + t).to(dev)[None]

            def call(ids=ids):
                with torch.no_grad():
                    out = hf_model(input_ids=ids, use_cache=False)
                    return logprob_region(out.logits[0, :-1], ids[0, 1:])

            log(f"[hf] forward T={t}")
            res = measure(call, timer, inst.install, num_layers=nl, warmup=warmup,
                          repeats=repeats, profiler_check=profiler_check)
            runs.append({"phase": "prefill_train", "T": t, "batch": 1, "context": 0,
                         "tokens": t, **res})
            log(f"  total {res['total_ms_median']:.1f} ms (clean "
                f"{res['clean_total_ms_median']:.1f}) peak {res['peak_allocated_gib']:.2f} GiB")
            del ids
            gc.collect()
            torch.cuda.empty_cache()

        gen = torch.Generator(device="cpu").manual_seed(77)
        for b in sorted(decode_batches):
            prompts = torch.stack(
                [_tokens(decode_context, seed=5000 + i) for i in range(b)]
            ).to(dev)
            cache = DynamicCache(config=text_cfg)
            with torch.no_grad():
                hf_model(input_ids=prompts, past_key_values=cache, use_cache=True,
                         logits_to_keep=1)
            torch.cuda.synchronize()
            toks = torch.randint(0, 200000, (b, 1), generator=gen).to(dev)
            tgts = torch.randint(0, 200000, (b,), generator=gen).to(dev)

            def step(cache=cache, toks=toks, tgts=tgts):
                with torch.no_grad():
                    out = hf_model(input_ids=toks, past_key_values=cache, use_cache=True)
                    return logprob_region(out.logits[:, -1], tgts)

            log(f"[hf] decode batch {b} context {decode_context}")
            res = measure(step, timer, inst.install, num_layers=nl, warmup=warmup,
                          repeats=decode_repeats, profiler_check=profiler_check)
            runs.append({"phase": "decode", "T": 1, "batch": b, "context": decode_context,
                         "tokens": b, **res})
            log(f"  step {res['total_ms_median']:.2f} ms (clean "
                f"{res['clean_total_ms_median']:.2f}) peak {res['peak_allocated_gib']:.2f} GiB")
            del cache, prompts
            gc.collect()
            torch.cuda.empty_cache()
    finally:
        inst.install(None)
        inst.detach()
    n_global = sum(1 for t_ in text_cfg.layer_types if t_ != "sliding_attention")
    for r in runs:
        r["per_layer_type_ms"] = per_layer_type_means(r["by_kind_class_ms"], n_global,
                                                      nl - n_global)
        r["tokens_per_s_clean"] = r["tokens"] / (r["clean_total_ms_median"] / 1e3)
    return {"target": "hf_bf16_sdpa", "runs": runs,
            "layers": {"global": n_global, "local": nl - n_global},
            "attn_implementation": text_cfg._attn_implementation}


def finite_or_none(x: float) -> float | None:
    return None if (x is None or not math.isfinite(x)) else x


__all__ = [
    "CLASSES",
    "GROUPS",
    "HFInstrument",
    "KINDS",
    "RegionTimer",
    "combine_repeats",
    "decode_batch_invariance",
    "group_of",
    "attributed_kernel_profile",
    "measure",
    "profile_hf",
    "profile_strict",
    "summarize_regions",
]
