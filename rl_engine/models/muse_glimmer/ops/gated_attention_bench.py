# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Measurement driver for the strict gated attention (O5): bit digests, timing series, bit diffs.

Writes machine-readable records for the POC report (``reports/muse_glimmer_poc/data/
kernels_attention.json``). Three modes, all single-GPU (run under ``~/bin/gpu-lease -n 1``)::

    python -m rl_engine.models.muse_glimmer.ops.gated_attention_bench digest --out d.json
    python -m rl_engine.models.muse_glimmer.ops.gated_attention_bench bench --out b.json
    python -m rl_engine.models.muse_glimmer.ops.gated_attention_bench bwd-diff \
        --old-module /path/to/old_gated_attention_bwd.py --out diff.json
    python -m rl_engine.models.muse_glimmer.ops.gated_attention_bench bwd-ab \
        --old-module /path/to/old_gated_attention_bwd.py --out ab.json [--rounds 5]
    python -m rl_engine.models.muse_glimmer.ops.gated_attention_bench parity --out p.json

* ``digest``: sha256 of the forward outputs ``(y, o, lse)`` over a matrix of shapes and paths
  (one-shot varlen, TOKEN/HEAD tiles, packed batch, chunked prefill, paged prefill and decode)
  and of the backward outputs ``(dq, dk, dv, dgate)``. Inputs are generated on the CPU from
  fixed seeds, so two commits can be compared by digest alone ("forward bits unchanged").
* ``bench``: CUDA-event timings in ms (prefill forward, paged decode forward, backward) for the
  strict kernels and the torch SDPA baseline, the ratio, and the backward's transient peak
  memory in GiB; optional TP head-shard geometries when the wrappers accept head counts.
  Forward rows time strict / SDPA / vLLM FlashAttention-2 (windowed on local layers, paged for
  decode) in alternating rounds and keep the raw per-round series (median, range). Local-layer
  SDPA rows carry ``ratio: null`` + ``baseline_not_comparable`` (dense-mask SDPA, and a
  full-causal SDPA backward); the FA2 ``flash_ratio`` is the comparable local baseline.
* ``bwd-diff``: per shape, mismatch counts / max relative difference between the backward of an
  OLD module file (e.g. ``git show <sha>:rl_engine/.../triton/gated_attention_bwd.py``) and the
  current one, on the same strict-forward ``o``/``lse`` (bit-pattern counts), split into the
  preprocess (sigmoid) and the dK/dV-rule causes.
* ``bwd-ab``: backward before/after timing, old and new module alternated in one process.
* ``parity``: bit-pattern mismatch counts for TP head shards (TP2/4/8 vs TP1 slices, replica
  tree), the CP1 runtime path and simulated CP2/4/8 ranks vs the one-shot kernel, plus
  association controls (teeth).

Only public wrappers are called; nothing here is part of the strict arithmetic.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import platform
import sys
import time
from typing import Any, Callable, Optional

import torch
import torch.nn.functional as F

from rl_engine.models.muse_glimmer.ops import gated_attention as ga
from rl_engine.models.muse_glimmer.triton import gated_attention_bwd as bwd

W = 2048
HQ, HKV, D, PAGE = 32, 2, 128, 16
DEV = torch.device("cuda")


# --------------------------------------------------------------------------------- helpers
def _sha(t: torch.Tensor) -> str:
    raw = t.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()[:16]


def _inputs(n: int, seed: int, hq: int = HQ, hkv: int = HKV) -> dict[str, torch.Tensor]:
    g = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.randn(n, hq, D, generator=g)
    q = q * torch.rsqrt(q.pow(2).mean(-1, keepdim=True) + 1e-5) * 3.87
    k = torch.randn(n, hkv, D, generator=g)
    k = k * torch.rsqrt(k.pow(2).mean(-1, keepdim=True) + 1e-5)
    v = torch.randn(n, hkv, D, generator=g)
    gate = torch.randn(n, hq * D, generator=g) * 2.0
    dy = torch.randn(n, hq * D, generator=g)
    to = {"device": DEV, "dtype": torch.bfloat16}
    return {"q": q.to(**to), "k": k.to(**to), "v": v.to(**to), "gate": gate.to(**to),
            "dy": dy.to(**to)}


def _i32(values) -> torch.Tensor:
    return torch.tensor(list(values), dtype=torch.int32, device=DEV)


def _time_ms(fn: Callable[[], Any], iters: int, warmup: int = 3) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def _supports_heads() -> bool:
    return "num_q_heads" in inspect.signature(ga.strict_gated_attention_fwd_varlen).parameters


def _env() -> dict[str, Any]:
    import triton

    return {
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "triton": triton.__version__,
        "cuda": torch.version.cuda,
        "python": platform.python_version(),
    }


def _fwd_varlen(inp, cu_q, cu_k, pos, window, **kw):
    return ga.strict_gated_attention_fwd_varlen(
        inp["q"], inp["k"], inp["v"], inp["gate"], cu_q, cu_k, pos, window=window, **kw
    )


def _paged_from_contiguous(k_all: torch.Tensor, v_all: torch.Tensor, bsz: int, ctx: int):
    """Contiguous page layout: sequence i owns pages [i*P, (i+1)*P)."""
    pages = -(-ctx // PAGE)
    hkv = k_all.shape[1]
    kc = torch.zeros((bsz * pages * PAGE, hkv, D), dtype=torch.bfloat16, device=DEV)
    vc = torch.zeros_like(kc)
    for i in range(bsz):
        kc[i * pages * PAGE : i * pages * PAGE + ctx] = k_all[i * ctx : (i + 1) * ctx]
        vc[i * pages * PAGE : i * pages * PAGE + ctx] = v_all[i * ctx : (i + 1) * ctx]
    kc = kc.view(bsz * pages, PAGE, hkv, D)
    vc = vc.view(bsz * pages, PAGE, hkv, D)
    table = torch.arange(bsz * pages, dtype=torch.int32, device=DEV).view(bsz, pages)
    lens = torch.full((bsz,), ctx, dtype=torch.int32, device=DEV)
    return kc, vc, table, lens


# --------------------------------------------------------------------------------- digest
def run_digest() -> dict[str, Any]:
    out: dict[str, Any] = {"forward": {}, "backward": {}}
    fwd = out["forward"]
    for window in (None, W):
        kind = "global" if window is None else "local"
        for n in (1, 17, 64, 300, 2047, 2048, 2049, 2100, 4096, 8192):
            inp = _inputs(n, seed=10_000 + n)
            cu = _i32([0, n])
            pos = torch.arange(n, dtype=torch.int32, device=DEV)
            modes = ("auto", "token", "head") if n <= 2100 else ("auto",)
            for mode in modes:
                kw = {} if mode == "auto" else {"tile_mode": mode}
                y, o, lse = _fwd_varlen(inp, cu, cu, pos, window, **kw)
                fwd[f"{kind}/varlen/T={n}/{mode}"] = [_sha(y), _sha(o), _sha(lse)]
        lens = [5, 130, 64, 2100, 1, 333]
        seqs = [_inputs(n, seed=20_000 + i) for i, n in enumerate(lens)]
        packed = {k: torch.cat([s[k] for s in seqs]) for k in ("q", "k", "v", "gate")}
        cu = _i32([0, *torch.tensor(lens).cumsum(0).tolist()])
        pos = torch.cat([torch.arange(n, dtype=torch.int32, device=DEV) for n in lens])
        y, o, lse = _fwd_varlen(packed, cu, cu, pos, window)
        fwd[f"{kind}/packed/{lens}"] = [_sha(y), _sha(o), _sha(lse)]
        # chunked prefill (varlen, keys 0..end-1)
        n = 4096
        inp = _inputs(n, seed=30_000)
        parts = []
        for a, b in ((0, 2048), (2048, 4088), (4088, 4096)):
            sub = {"q": inp["q"][a:b].contiguous(), "gate": inp["gate"][a:b].contiguous(),
                   "k": inp["k"][:b].contiguous(), "v": inp["v"][:b].contiguous()}
            parts.append(_fwd_varlen(sub, _i32([0, b - a]), _i32([0, b]),
                                     torch.arange(a, b, dtype=torch.int32, device=DEV), window))
        fwd[f"{kind}/chunked_varlen/T=4096/[2048,4088]"] = [
            _sha(torch.cat([p[i] for p in parts])) for i in range(3)
        ]
        # paged decode B=8 ctx in {1, 2049, 4096} + one paged prefill
        for bsz, ctx in ((8, 2049), (8, 4096), (1, 1), (4, 300)):
            seqs = [_inputs(ctx, seed=40_000 + 97 * i + ctx) for i in range(bsz)]
            k_all = torch.cat([s["k"] for s in seqs])
            v_all = torch.cat([s["v"] for s in seqs])
            kc, vc, table, seq_lens = _paged_from_contiguous(k_all, v_all, bsz, ctx)
            q = torch.cat([s["q"][-1:] for s in seqs]).contiguous()
            gate = torch.cat([s["gate"][-1:] for s in seqs]).contiguous()
            y, o, lse = ga.strict_gated_attention_fwd_paged(
                q, gate, kc, vc, table, seq_lens, _i32(range(bsz + 1)), window=window
            )
            fwd[f"{kind}/paged_decode/B={bsz}/ctx={ctx}"] = [_sha(y), _sha(o), _sha(lse)]
        inp = _inputs(300, seed=50_000)
        kc, vc, table, seq_lens = _paged_from_contiguous(inp["k"], inp["v"], 1, 300)
        y, o, lse = ga.strict_gated_attention_fwd_paged(
            inp["q"], inp["gate"], kc, vc, table, seq_lens, _i32([0, 300]), window=window
        )
        fwd[f"{kind}/paged_prefill/T=300"] = [_sha(y), _sha(o), _sha(lse)]
        _digest_extra(fwd, kind, window)

    back = out["backward"]
    for window in (None, W):
        kind = "global" if window is None else "local"
        for n in (1, 2, 64, 300, 2048, 2100, 4096):
            inp = _inputs(n, seed=60_000 + n)
            cu = _i32([0, n])
            pos = torch.arange(n, dtype=torch.int32, device=DEV)
            _y, o, lse = _fwd_varlen(inp, cu, cu, pos, window)
            try:
                grads = bwd.strict_gated_attention_bwd(
                    inp["q"], inp["k"], inp["v"], inp["gate"], o, lse, inp["dy"], cu, cu, pos,
                    window=window,
                )
            except Exception as exc:  # noqa: BLE001 - recorded, e.g. a compile error at T=1
                back[f"{kind}/T={n}"] = f"ERROR {type(exc).__name__}: {str(exc)[-200:]}"
                continue
            back[f"{kind}/T={n}"] = [_sha(t) for t in grads]
    return out


def _paged_layout(ctxs: list[int], seed: int, hkv: int = HKV, evict_window: Optional[int] = None):
    """Several sequences in one cache with interleaved (non-contiguous) page ids.

    Sequence ``i`` has ``ctxs[i]`` keys; its logical page ``j`` lives at physical page
    ``j * B + i``. With ``evict_window`` the pages whose keys all sit at ``<= ctx - 1 - W`` are
    marked ``-1`` (evicted for a decode query at ``ctx - 1``) and NaN-filled.
    """
    bsz = len(ctxs)
    max_pages = max(-(-c // PAGE) for c in ctxs)
    kc = torch.zeros((max_pages * bsz, PAGE, hkv, D), dtype=torch.bfloat16, device=DEV)
    vc = torch.zeros_like(kc)
    table = torch.full((bsz, max_pages), -1, dtype=torch.int32, device=DEV)
    seqs = []
    for i, ctx in enumerate(ctxs):
        s = _inputs(ctx, seed=seed + 13 * i, hq=HQ if hkv == HKV else 4, hkv=hkv)
        seqs.append(s)
        for j in range(-(-ctx // PAGE)):
            phys = j * bsz + i
            rows = slice(j * PAGE, min((j + 1) * PAGE, ctx))
            n = rows.stop - rows.start
            kc[phys, :n] = s["k"][rows]
            vc[phys, :n] = s["v"][rows]
            table[i, j] = phys
            if evict_window is not None and (j + 1) * PAGE - 1 <= ctx - 1 - evict_window:
                table[i, j] = -1
                kc[phys].fill_(float("nan"))
                vc[phys].fill_(float("nan"))
    lens = torch.tensor(ctxs, dtype=torch.int32, device=DEV)
    return kc, vc, table, lens


def _digest_extra(fwd: dict[str, Any], kind: str, window: Optional[int]) -> None:
    """Paths added in wave B (validation / in-kernel paged positions must not move bits):
    TP head shards (TOKEN + HEAD tiles, varlen and paged), paged chunked prefill with queries
    that are NOT the whole sequence (TOKEN tiles), multi-token HEAD tiles on the paged path,
    interleaved page ids and evicted (-1, NaN-filled) pages on local layers."""
    lens = [5, 130, 2100, 1, 333]
    for hq in (16, 8, 4):
        seqs = [_inputs(n, seed=21_000 + 7 * hq + i, hq=hq, hkv=1) for i, n in enumerate(lens)]
        packed = {k: torch.cat([s[k] for s in seqs]) for k in ("q", "k", "v", "gate")}
        cu = _i32([0, *torch.tensor(lens).cumsum(0).tolist()])
        pos = torch.cat([torch.arange(n, dtype=torch.int32, device=DEV) for n in lens])
        for mode in ("token", "head"):
            y, o, lse = _fwd_varlen(packed, cu, cu, pos, window, tile_mode=mode,
                                    num_q_heads=hq, num_kv_heads=1)
            fwd[f"{kind}/tp_varlen/H_Q={hq}/{lens}/{mode}"] = [_sha(y), _sha(o), _sha(lse)]
        ctxs = [2049, 1, 700, 4096]
        kc, vc, table, seq_lens = _paged_layout(ctxs, seed=22_000 + hq, hkv=1)
        g = torch.Generator(device="cpu").manual_seed(23_000 + hq)
        q = (torch.randn(len(ctxs), hq, D, generator=g) * 3.0).to(DEV, torch.bfloat16)
        gate = torch.randn(len(ctxs), hq * D, generator=g).to(DEV, torch.bfloat16)
        y, o, lse = ga.strict_gated_attention_fwd_paged(
            q, gate, kc, vc, table, seq_lens, _i32(range(len(ctxs) + 1)), window=window,
            num_q_heads=hq, num_kv_heads=1,
        )
        fwd[f"{kind}/tp_paged_decode/H_Q={hq}/{ctxs}"] = [_sha(y), _sha(o), _sha(lse)]
    # paged chunked prefill: the queries are the last len_q of each sequence (TOKEN tiles)
    for ctxs, lq in (([2100, 300, 4096], [100, 17, 64]), ([2049, 70, 4096], [3, 16, 1])):
        kc, vc, table, seq_lens = _paged_layout(ctxs, seed=24_000 + sum(lq))
        g = torch.Generator(device="cpu").manual_seed(25_000 + sum(lq))
        tq = sum(lq)
        q = (torch.randn(tq, HQ, D, generator=g) * 3.0).to(DEV, torch.bfloat16)
        gate = torch.randn(tq, HQ * D, generator=g).to(DEV, torch.bfloat16)
        cu_q = _i32([0, *torch.tensor(lq).cumsum(0).tolist()])
        for mode in (None, "token", "head"):
            kw = {} if mode is None else {"tile_mode": mode}
            y, o, lse = ga.strict_gated_attention_fwd_paged(
                q, gate, kc, vc, table, seq_lens, cu_q, window=window, **kw
            )
            fwd[f"{kind}/paged_chunk/{ctxs}/{lq}/{mode or 'auto'}"] = [
                _sha(y), _sha(o), _sha(lse)
            ]
    if window is not None:  # evicted pages (-1 in the table, NaN-filled) at decode
        ctxs = [4096, 2100, 2049, 9000]
        kc, vc, table, seq_lens = _paged_layout(ctxs, seed=26_000, evict_window=window)
        g = torch.Generator(device="cpu").manual_seed(26_500)
        q = (torch.randn(len(ctxs), HQ, D, generator=g) * 3.0).to(DEV, torch.bfloat16)
        gate = torch.randn(len(ctxs), HQ * D, generator=g).to(DEV, torch.bfloat16)
        y, o, lse = ga.strict_gated_attention_fwd_paged(
            q, gate, kc, vc, table, seq_lens, _i32(range(len(ctxs) + 1)), window=window
        )
        fwd[f"{kind}/paged_decode_evicted/{ctxs}"] = [_sha(y), _sha(o), _sha(lse)]


# --------------------------------------------------------------------------------- bench
def _sdpa_gated(q4, k4, v4, gate, *, is_causal: bool, mask):
    out = F.scaled_dot_product_attention(
        q4, k4, v4, attn_mask=mask, is_causal=is_causal, enable_gqa=True
    )
    o = out.transpose(1, 2).reshape(gate.shape)
    return (o.float() * torch.sigmoid(gate.float())).to(torch.bfloat16)


#: Why a baseline is not comparable (stored instead of a ratio; review kernels_attention:R3).
LOCAL_SDPA_NOT_COMPARABLE = (
    "local layer: torch SDPA has no sliding-window kernel; the baseline is SDPA with a dense "
    "bool mask (math/efficient path, O(T^2) memory), not a flash-class kernel"
)
LOCAL_SDPA_BWD_NOT_COMPARABLE = (
    "local layer: the SDPA backward baseline is the FULL-causal flash backward (no window), "
    "i.e. different work from the windowed strict backward"
)


def _flash_fwd():
    """vLLM's FlashAttention-2 varlen forward (windowed, GQA, paged), or None if unavailable."""
    try:
        from vllm.vllm_flash_attn import flash_attn_varlen_func
    except Exception:  # noqa: BLE001 - optional baseline (only in the vLLM env)
        return None
    return flash_attn_varlen_func


def _flash_window(window: Optional[int]) -> list[int]:
    return [-1, -1] if window is None else [window - 1, 0]


def _gated(o: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    return (o.reshape(gate.shape).float() * torch.sigmoid(gate.float())).to(torch.bfloat16)


def _rounds(fns: dict[str, Callable[[], Any]], iters: int, rounds: int) -> dict[str, Any]:
    """Time every fn ``rounds`` times, alternating (review kernels_attention:R6): raw per-round
    ms, median, min, max per fn."""
    import statistics

    raw: dict[str, list[float]] = {name: [] for name in fns}
    for _ in range(rounds):
        for name, fn in fns.items():
            raw[name].append(_time_ms(fn, iters))
    out: dict[str, Any] = {}
    for name, xs in raw.items():
        out[f"{name}_ms"] = statistics.median(xs)
        out[f"{name}_ms_rounds"] = xs
        out[f"{name}_ms_range"] = [min(xs), max(xs)]
    return out


def _baseline_diffs(fns: dict[str, Callable[[], Any]]) -> dict[str, float]:
    """max |y_baseline - y_strict| per baseline: the baselines compute the same function."""
    y = fns["strict"]()[0].float()
    return {f"{name}_max_abs_diff_y": float((fn().float() - y).abs().max().item())
            for name, fn in fns.items() if name != "strict"}


def bench_prefill(n: int, window: Optional[int], iters: int, hq: int = HQ, hkv: int = HKV,
                  rounds: int = 5):
    inp = _inputs(n, seed=1, hq=hq, hkv=hkv)
    cu = _i32([0, n])
    pos = torch.arange(n, dtype=torch.int32, device=DEV)
    kw: dict[str, Any] = {"max_seqlen_q": n}
    if (hq, hkv) != (HQ, HKV):
        kw.update(num_q_heads=hq, num_kv_heads=hkv)
    fns: dict[str, Callable[[], Any]] = {
        "strict": lambda: _fwd_varlen(inp, cu, cu, pos, window, **kw)
    }
    rec: dict[str, Any] = {"T": n, "window": window, "H_Q": hq, "H_KV": hkv,
                           "iters": iters, "rounds": rounds}
    flash = _flash_fwd()
    if (hq, hkv) == (HQ, HKV):
        q4 = inp["q"].transpose(0, 1).unsqueeze(0).contiguous()
        k4 = inp["k"].transpose(0, 1).unsqueeze(0).contiguous()
        v4 = inp["v"].transpose(0, 1).unsqueeze(0).contiguous()
        if window is None:
            mask, causal = None, True
        else:
            i = torch.arange(n, device=DEV)
            mask, causal = (i[None, :] <= i[:, None]) & (i[None, :] > i[:, None] - window), False
        fns["sdpa"] = lambda: _sdpa_gated(q4, k4, v4, inp["gate"], is_causal=causal, mask=mask)
        if flash is not None:
            fns["flash"] = lambda: _gated(
                flash(inp["q"], inp["k"], inp["v"], n, cu, n, cu_seqlens_k=cu,
                      causal=True, window_size=_flash_window(window)),
                inp["gate"],
            )
    rec.update(_baseline_diffs(fns))
    rec.update(_rounds(fns, iters, rounds))
    if "sdpa_ms" in rec:
        if window is None:
            rec["ratio"] = rec["strict_ms"] / rec["sdpa_ms"]
        else:
            rec["ratio"] = None
            rec["baseline_not_comparable"] = LOCAL_SDPA_NOT_COMPARABLE
    if "flash_ms" in rec:
        rec["flash_ratio"] = rec["strict_ms"] / rec["flash_ms"]
        rec["flash_baseline"] = "vllm.vllm_flash_attn.flash_attn_varlen_func (FA2), " + (
            "causal" if window is None else f"causal, window_size=[{window - 1}, 0]"
        ) + " + the same gate"
    return rec


def bench_decode(bsz: int, ctx: int, window: Optional[int], iters: int, rounds: int = 5):
    g = torch.Generator(device="cpu").manual_seed(bsz * 7 + ctx)
    k_all = torch.randn(bsz * ctx, HKV, D, generator=g).to(DEV, torch.bfloat16)
    v_all = torch.randn(bsz * ctx, HKV, D, generator=g).to(DEV, torch.bfloat16)
    q = (torch.randn(bsz, HQ, D, generator=g) * 3.87).to(DEV, torch.bfloat16)
    gate = torch.randn(bsz, HQ * D, generator=g).to(DEV, torch.bfloat16)
    kc, vc, table, seq_lens = _paged_from_contiguous(k_all, v_all, bsz, ctx)
    cu_q = _i32(range(bsz + 1))
    q4 = q.unsqueeze(2).contiguous()
    k4 = k_all.view(bsz, ctx, HKV, D).transpose(1, 2).contiguous()
    v4 = v_all.view(bsz, ctx, HKV, D).transpose(1, 2).contiguous()
    if window is None:
        mask = None
    else:
        j = torch.arange(ctx, device=DEV)
        mask = ((j <= ctx - 1) & (j > ctx - 1 - window)).view(1, 1, 1, ctx)
    fns: dict[str, Callable[[], Any]] = {
        "strict": lambda: ga.strict_gated_attention_fwd_paged(
            q, gate, kc, vc, table, seq_lens, cu_q, window=window, max_seqlen_q=1
        ),
        "sdpa": lambda: _sdpa_gated(q4, k4, v4, gate, is_causal=False, mask=mask),
    }
    flash = _flash_fwd()
    if flash is not None:
        fns["flash"] = lambda: _gated(
            flash(q, kc, vc, 1, cu_q, ctx, seqused_k=seq_lens, block_table=table, causal=True,
                  window_size=_flash_window(window)),
            gate,
        )
    rec: dict[str, Any] = {"B": bsz, "ctx": ctx, "window": window, "iters": iters,
                           "rounds": rounds}
    rec.update(_baseline_diffs(fns))
    rec.update(_rounds(fns, iters, rounds))
    if window is None:
        rec["ratio"] = rec["strict_ms"] / rec["sdpa_ms"]
    else:
        rec["ratio"] = None
        rec["baseline_not_comparable"] = LOCAL_SDPA_NOT_COMPARABLE
    if "flash_ms" in rec:
        rec["flash_ratio"] = rec["strict_ms"] / rec["flash_ms"]
        rec["flash_baseline"] = (
            "vllm.vllm_flash_attn.flash_attn_varlen_func (FA2) on the same paged cache "
            "(block_table, page 16, seqused_k), " + (
                "causal" if window is None else f"causal, window_size=[{window - 1}, 0]"
            ) + " + the same gate"
        )
    return rec


def bench_backward(n: int, window: Optional[int], iters: int, hq: int = HQ, hkv: int = HKV):
    inp = _inputs(n, seed=2, hq=hq, hkv=hkv)
    cu = _i32([0, n])
    pos = torch.arange(n, dtype=torch.int32, device=DEV)
    kw: dict[str, Any] = {}
    if (hq, hkv) != (HQ, HKV):
        kw.update(num_q_heads=hq, num_kv_heads=hkv)
    _y, o, lse = _fwd_varlen(inp, cu, cu, pos, window, **kw)

    def strict():
        if hq // hkv == HQ // HKV:  # TP1 / TP2: the rank owns whole KV heads, fused entry
            return bwd.strict_gated_attention_bwd(
                inp["q"], inp["k"], inp["v"], inp["gate"], o, lse, inp["dy"], cu, cu, pos,
                window=window, **kw,
            )
        # TP4 / TP8 rank: its local work = preprocess + kernels + its node of the KV-head tree
        # (the cross-rank fold of t/2 nodes is a few elementwise adds, not timed here)
        _dg, do, delta = bwd.gated_attention_bwd_preprocess(inp["dy"], o, inp["gate"],
                                                             num_q_heads=hq)
        dq, dkp, dvp = bwd.strict_gated_attention_bwd_partials(
            inp["q"], inp["k"], inp["v"], do, lse, delta, cu, cu, pos, window=window, **kw
        )
        return dq, bwd.kv_partials_subtree(dkp), bwd.kv_partials_subtree(dvp)

    strict()
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    strict()
    torch.cuda.synchronize()
    peak_gib = (torch.cuda.max_memory_allocated() - base) / 2**30
    strict_ms = _time_ms(strict, iters)
    rec: dict[str, Any] = {"T": n, "window": window, "H_Q": hq, "H_KV": hkv,
                           "strict_ms": strict_ms, "transient_peak_GiB": peak_gib}
    if (hq, hkv) == (HQ, HKV):
        qs = inp["q"].permute(1, 0, 2).unsqueeze(0).contiguous().requires_grad_(True)
        ks = inp["k"].permute(1, 0, 2).unsqueeze(0).contiguous().requires_grad_(True)
        vs = inp["v"].permute(1, 0, 2).unsqueeze(0).contiguous().requires_grad_(True)
        out = F.scaled_dot_product_attention(qs, ks, vs, is_causal=True, enable_gqa=True)
        dout = inp["dy"].view(n, hq, D).permute(1, 0, 2).unsqueeze(0).contiguous()
        sdpa_ms = _time_ms(
            lambda: torch.autograd.grad(out, (qs, ks, vs), grad_outputs=dout, retain_graph=True),
            iters,
        )
        rec.update(sdpa_causal_bwd_ms=sdpa_ms)
        if window is None:
            rec["ratio"] = strict_ms / sdpa_ms
        else:
            rec["ratio"] = None
            rec["baseline_not_comparable"] = LOCAL_SDPA_BWD_NOT_COMPARABLE
    return rec


SECTIONS = ("prefill", "decode", "backward", "tp")


def run_bench(iters: int, quick: bool, sections=SECTIONS, rounds: int = 5) -> dict[str, Any]:
    lengths = (512, 1024, 2048, 4096, 8192) if not quick else (1024, 4096)
    out: dict[str, Any] = {"prefill_forward": [], "decode_forward": [], "backward": [],
                           "tp_shard_forward": [], "tp_shard_backward": [],
                           "sections": list(sections), "iters": iters, "rounds": rounds}
    for window in (None, W):
        for n in lengths:
            if "prefill" in sections:
                out["prefill_forward"].append(bench_prefill(n, window, iters, rounds=rounds))
            if "backward" in sections:
                out["backward"].append(bench_backward(n, window, max(5, iters // 2)))
        if "decode" in sections:
            for bsz in ((1, 16, 64, 256) if not quick else (1, 64)):
                out["decode_forward"].append(
                    bench_decode(bsz, 4096, window, iters, rounds=rounds)
                )
    if "tp" in sections and _supports_heads():
        for hq, hkv in ((32, 2), (16, 1), (8, 1), (4, 1)):
            for window in (None, W):
                kw = {} if (hq, hkv) == (HQ, HKV) else {"hq": hq, "hkv": hkv}
                rec = bench_prefill(4096, window, iters, rounds=rounds, **kw)
                rec["tp"] = HQ // hq
                out["tp_shard_forward"].append(rec)
                rec = bench_backward(4096, window, max(5, iters // 2), **kw)
                rec["tp"] = HQ // hq
                out["tp_shard_backward"].append(rec)
    return out


# --------------------------------------------------------------------------------- bwd-diff
def _load_module(path: str):
    spec = importlib.util.spec_from_file_location("_old_gated_attention_bwd", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def run_bwd_ab(old_path: str, iters: int, rounds: int) -> dict[str, Any]:
    """Backward before/after in ONE process: old and new module timed alternately per round on
    the same inputs and the same strict-forward o/lse (medians over rounds; robust to drift)."""
    import statistics

    old = _load_module(old_path)
    records = []
    for window in (None, W):
        for n in (512, 1024, 2048, 4096, 8192):
            inp = _inputs(n, seed=2)
            cu = _i32([0, n])
            pos = torch.arange(n, dtype=torch.int32, device=DEV)
            _y, o, lse = _fwd_varlen(inp, cu, cu, pos, window)
            args = (inp["q"], inp["k"], inp["v"], inp["gate"], o, lse, inp["dy"], cu, cu, pos)
            runs = {"v1": [], "v2": []}
            fns = {"v1": lambda: old.strict_gated_attention_bwd(*args, window=window),
                   "v2": lambda: bwd.strict_gated_attention_bwd(*args, window=window)}
            for _ in range(rounds):
                for name in ("v1", "v2"):
                    runs[name].append(_time_ms(fns[name], iters))
            rec = {"T": n, "window": window, "iters": iters, "rounds": rounds,
                   "v1_ms": runs["v1"], "v2_ms": runs["v2"],
                   "v1_median_ms": statistics.median(runs["v1"]),
                   "v2_median_ms": statistics.median(runs["v2"])}
            rec["v2_over_v1"] = rec["v2_median_ms"] / rec["v1_median_ms"]
            qs = inp["q"].permute(1, 0, 2).unsqueeze(0).contiguous().requires_grad_(True)
            ks = inp["k"].permute(1, 0, 2).unsqueeze(0).contiguous().requires_grad_(True)
            vs = inp["v"].permute(1, 0, 2).unsqueeze(0).contiguous().requires_grad_(True)
            out = F.scaled_dot_product_attention(qs, ks, vs, is_causal=True, enable_gqa=True)
            dout = inp["dy"].view(n, HQ, D).permute(1, 0, 2).unsqueeze(0).contiguous()
            sdpa = [
                _time_ms(lambda: torch.autograd.grad(out, (qs, ks, vs), grad_outputs=dout,
                                                     retain_graph=True), iters)
                for _ in range(rounds)
            ]
            rec["sdpa_causal_bwd_median_ms"] = statistics.median(sdpa)
            if window is None:
                rec["v1_ratio_to_sdpa"] = rec["v1_median_ms"] / rec["sdpa_causal_bwd_median_ms"]
                rec["v2_ratio_to_sdpa"] = rec["v2_median_ms"] / rec["sdpa_causal_bwd_median_ms"]
            else:
                rec["v1_ratio_to_sdpa"] = rec["v2_ratio_to_sdpa"] = None
                rec["baseline_not_comparable"] = LOCAL_SDPA_BWD_NOT_COMPARABLE
            records.append(rec)
    return {"records": records, "old_module": old_path}


def run_bwd_diff(old_path: str) -> dict[str, Any]:
    old = _load_module(old_path)
    records = []
    for window in (None, W):
        for n in (1, 2, 7, 8, 64, 65, 300, 512, 1024, 2047, 2048, 2049, 2100, 4096, 8192):
            inp = _inputs(n, seed=70_000 + n)
            cu = _i32([0, n])
            pos = torch.arange(n, dtype=torch.int32, device=DEV)
            _y, o, lse = _fwd_varlen(inp, cu, cu, pos, window)
            args = (inp["q"], inp["k"], inp["v"], inp["gate"], o, lse, inp["dy"], cu, cu, pos)
            new_g = bwd.strict_gated_attention_bwd(*args, window=window)
            rec: dict[str, Any] = {"T": n, "window": window}
            try:
                old_g = old.strict_gated_attention_bwd(*args, window=window)
            except Exception as exc:  # noqa: BLE001 - the old kernel cannot compile Tk == 1
                rec["old_error"] = f"{type(exc).__name__}: {str(exc)[-160:]}"
                records.append(rec)
                continue
            _diff_into(rec, "full", ("dq", "dk", "dv", "dgate"), new_g, old_g)
            # isolate the two causes: (a) preprocess sigmoid (finding 4) on dgate/do/delta,
            # (b) the dK/dV query-chunk rule on identical do/delta (dq kernel is unchanged)
            new_pre = bwd.gated_attention_bwd_preprocess(inp["dy"], o, inp["gate"])
            old_pre = old.gated_attention_bwd_preprocess(inp["dy"], o, inp["gate"])
            _diff_into(rec, "preprocess", ("dgate", "do", "delta"), new_pre, old_pre)
            _dg, do, delta = new_pre
            core = (inp["q"], inp["k"], inp["v"], do, lse, delta, cu, cu, pos)
            new_c = bwd.strict_gated_attention_bwd_partials(*core, window=window)
            old_c = old.strict_gated_attention_bwd_partials(*core, window=window)
            _diff_into(
                rec, "core_same_do", ("dq", "dk", "dv"),
                (new_c[0], bwd.combine_kv_partials(new_c[1]), bwd.combine_kv_partials(new_c[2])),
                (old_c[0], old.combine_kv_partials(old_c[1]), old.combine_kv_partials(old_c[2])),
            )
            if n % 8 == 0:
                # the contract defect: OLD one-shot kernel vs the OLD CP1 path (8 canonical query
                # chunks, each through the old kernel, group tree then chunk tree), and the NEW
                # one-shot kernel vs the same CP1 leaves folded in the new order
                old_cp = _old_cp1_path(old, inp, do, lse, delta, pos, window)
                _diff_into(
                    rec, "defect_old_oneshot_vs_old_cp1", ("dk", "dv"),
                    (old.combine_kv_partials(old_c[1]), old.combine_kv_partials(old_c[2])),
                    old_cp,
                )
            records.append(rec)
    return {"records": records, "old_module": old_path}


def _old_cp1_path(old, inp, do, lse, delta, pos, window):
    """v1 CP1 path (runtime/cp_attention.py at 6a61f6e): per canonical chunk the old kernel's 8
    group partials folded by the group tree, then the 8 chunk roots folded by the same tree."""
    from rl_engine.models.muse_glimmer.ops.canonical_tree import combine_tree

    n = inp["q"].shape[0]
    chunk = n // 8
    cu_k = _i32([0, n])
    roots_k, roots_v = [], []
    for c in range(8):
        rows = slice(c * chunk, (c + 1) * chunk)
        _dq, dkp, dvp = old.strict_gated_attention_bwd_partials(
            inp["q"][rows].contiguous(), inp["k"], inp["v"], do[rows].contiguous(),
            lse[rows].contiguous(), delta[rows].contiguous(), _i32([0, chunk]), cu_k,
            pos[rows].contiguous(), window=window,
        )
        roots_k.append(combine_tree(list(dkp.unbind(0))))
        roots_v.append(combine_tree(list(dvp.unbind(0))))
    return (combine_tree(roots_k).to(torch.bfloat16), combine_tree(roots_v).to(torch.bfloat16))


# --------------------------------------------------------------------------------- parity
def _bits(t: torch.Tensor) -> torch.Tensor:
    """Integer view of a float tensor: mismatches are counted on bit patterns (-0.0 != +0.0)."""
    if not t.is_floating_point():
        return t
    return t.contiguous().view({2: torch.int16, 4: torch.int32, 8: torch.int64}[t.element_size()])


def _mm(a: torch.Tensor, b: torch.Tensor) -> int:
    """Bit-pattern mismatch count (-1 if shape/dtype differ)."""
    if a.shape != b.shape or a.dtype != b.dtype:
        return -1
    return int((_bits(a) != _bits(b)).sum().item())


def _packed_inputs(lens: list[int], seed: int, hq: int = HQ, hkv: int = HKV):
    seqs = [_inputs(n, seed=seed + i, hq=hq, hkv=hkv) for i, n in enumerate(lens)]
    inp = {k: torch.cat([s[k] for s in seqs]).contiguous() for k in ("q", "k", "v", "gate", "dy")}
    cu = _i32([0, *torch.tensor(lens).cumsum(0).tolist()])
    pos = torch.cat([torch.arange(n, dtype=torch.int32, device=DEV) for n in lens])
    return inp, cu, pos


def _tp_rank_slices(tp: int, r: int):
    """Rank r of TP t: Q heads [32r/t, 32(r+1)/t), gate/y/o columns, its KV head, its groups."""
    hq = HQ // tp
    heads = slice(r * hq, (r + 1) * hq)
    cols = slice(r * hq * D, (r + 1) * hq * D)
    kvh = (r * hq) // (HQ // HKV)
    groups = slice(r * (8 // tp), (r + 1) * (8 // tp))
    return hq, heads, cols, kvh, groups


def _tp_parity_varlen(case: str, lens: list[int], seed: int, window) -> list[dict[str, Any]]:
    """Forward (y, o, lse) and backward (dq, dgate, dK/dV group partials, tree) per TP rank."""
    from rl_engine.models.muse_glimmer.runtime.tp_canonical_tree import combine_tree as tp_tree

    inp, cu, pos = _packed_inputs(lens, seed)
    n = inp["q"].shape[0]
    y, o, lse = _fwd_varlen(inp, cu, cu, pos, window)
    dgate, do, delta = bwd.gated_attention_bwd_preprocess(inp["dy"], o, inp["gate"])
    dq, dkp, dvp = bwd.strict_gated_attention_bwd_partials(
        inp["q"], inp["k"], inp["v"], do, lse, delta, cu, cu, pos, window=window
    )
    dk, dv = bwd.combine_kv_partials(dkp), bwd.combine_kv_partials(dvp)
    out = []
    for tp in (2, 4, 8):
        rec: dict[str, Any] = {"case": case, "tp": tp, "window": window, "Tq": n, "ranks": []}
        roots_k, roots_v = [], []
        for r in range(tp):
            hq, heads, cols, kvh, groups = _tp_rank_slices(tp, r)
            sh = {"q": inp["q"][:, heads].contiguous(),
                  "k": inp["k"][:, kvh : kvh + 1].contiguous(),
                  "v": inp["v"][:, kvh : kvh + 1].contiguous(),
                  "gate": inp["gate"][:, cols].contiguous(),
                  "dy": inp["dy"][:, cols].contiguous()}
            heads_kw = {"num_q_heads": hq, "num_kv_heads": 1}
            y_r, o_r, l_r = _fwd_varlen(sh, cu, cu, pos, window, **heads_kw)
            dg_r, do_r, de_r = bwd.gated_attention_bwd_preprocess(
                sh["dy"], o_r, sh["gate"], num_q_heads=hq
            )
            dq_r, dkp_r, dvp_r = bwd.strict_gated_attention_bwd_partials(
                sh["q"], sh["k"], sh["v"], do_r, l_r, de_r, cu, cu, pos, window=window, **heads_kw
            )
            rank_rec = {
                "rank": r, "q_heads": [heads.start, heads.stop], "kv_head": kvh,
                "y": _mm(y_r, y[:, cols]), "o": _mm(o_r, o[:, cols]),
                "lse": _mm(l_r, lse[:, heads]), "dq": _mm(dq_r, dq[:, heads]),
                "dgate": _mm(dg_r, dgate[:, cols]), "delta": _mm(de_r, delta[:, heads]),
                "dk_group_partials": _mm(dkp_r[:, :, 0], dkp[groups, :, kvh]),
                "dv_group_partials": _mm(dvp_r[:, :, 0], dvp[groups, :, kvh]),
                "numel_y": y_r.numel(), "numel_dq": dq_r.numel(),
                "numel_partials": dkp_r.numel(),
            }
            if tp == 2:  # a TP2 rank owns its whole KV head: the fused entry is final
                f = bwd.strict_gated_attention_bwd(
                    sh["q"], sh["k"], sh["v"], sh["gate"], o_r, l_r, sh["dy"], cu, cu, pos,
                    window=window, **heads_kw,
                )
                rank_rec.update(
                    fused_dq=_mm(f[0], dq[:, heads]), fused_dk=_mm(f[1][:, 0], dk[:, kvh]),
                    fused_dv=_mm(f[2][:, 0], dv[:, kvh]), fused_dgate=_mm(f[3], dgate[:, cols]),
                )
            rec["ranks"].append(rank_rec)
            for roots, part in ((roots_k, dkp_r), (roots_v, dvp_r)):
                root = bwd.kv_partials_subtree(part)  # (Tk, 1, D) fp32: this rank's tree node
                full = torch.zeros((n, HKV, D), dtype=torch.float32, device=DEV)
                full[:, kvh] = root[:, 0]
                roots.append(full)
        # replicated-KV rule: every rank's sub-tree root (zeros at the other KV head) folded in
        # the canonical TP tree over the ranks, rounded once == TP1 dk/dv
        rec["dk_tp_tree_vs_tp1"] = _mm(tp_tree(roots_k).to(torch.bfloat16), dk)
        rec["dv_tp_tree_vs_tp1"] = _mm(tp_tree(roots_v).to(torch.bfloat16), dv)
        # binding recipe: per KV head, only the replicas (ranks r with (2r)//t == h) are
        # gathered in rank order and folded in the canonical tree (no zero padding)
        mk = mv = 0
        for h in range(HKV):
            reps = [r for r in range(tp) if _tp_rank_slices(tp, r)[3] == h]
            got_k = tp_tree([roots_k[r][:, h] for r in reps]).to(torch.bfloat16)
            got_v = tp_tree([roots_v[r][:, h] for r in reps]).to(torch.bfloat16)
            mk += _mm(got_k, dk[:, h])
            mv += _mm(got_v, dv[:, h])
        rec["dk_replica_tree_vs_tp1"] = mk
        rec["dv_replica_tree_vs_tp1"] = mv
        rec["numel_dk"] = dk.numel()
        out.append(rec)
    return out


def _tp_parity_paged_decode(bsz: int, ctx: int, window) -> list[dict[str, Any]]:
    """HEAD-tile forward (paged decode, BLOCK_M=16 rows, group 16/8/4) per TP rank."""
    seqs = [_inputs(ctx, seed=85_000 + 31 * i + ctx) for i in range(bsz)]
    k_all = torch.cat([s["k"] for s in seqs])
    v_all = torch.cat([s["v"] for s in seqs])
    kc, vc, table, seq_lens = _paged_from_contiguous(k_all, v_all, bsz, ctx)
    q = torch.cat([s["q"][-1:] for s in seqs]).contiguous()
    gate = torch.cat([s["gate"][-1:] for s in seqs]).contiguous()
    cu_q = _i32(range(bsz + 1))
    y, o, lse = ga.strict_gated_attention_fwd_paged(
        q, gate, kc, vc, table, seq_lens, cu_q, window=window
    )
    out = []
    for tp in (2, 4, 8):
        rec: dict[str, Any] = {"case": f"paged_decode/B={bsz}/ctx={ctx}", "tile": "head",
                               "tp": tp, "window": window, "ranks": []}
        for r in range(tp):
            hq, heads, cols, kvh, _groups = _tp_rank_slices(tp, r)
            y_r, o_r, l_r = ga.strict_gated_attention_fwd_paged(
                q[:, heads].contiguous(), gate[:, cols].contiguous(),
                kc[:, :, kvh : kvh + 1].contiguous(), vc[:, :, kvh : kvh + 1].contiguous(),
                table, seq_lens, cu_q, window=window, num_q_heads=hq, num_kv_heads=1,
            )
            rec["ranks"].append({"rank": r, "y": _mm(y_r, y[:, cols]), "o": _mm(o_r, o[:, cols]),
                                 "lse": _mm(l_r, lse[:, heads]), "numel_y": y_r.numel()})
        out.append(rec)
    return out


def run_parity(cp_lengths=(8, 300, 2048, 2100, 4096, 8192)) -> dict[str, Any]:
    """Bit-pattern mismatch counts (expected all 0) for TP head shards and CP, one GPU.

    TP t in {2, 4, 8}: rank r runs the strict forward/backward on its head shard (32/t Q heads,
    one KV head, replicated when t > 2) and is compared with the TP1 call's slices (y, o, lse,
    dq, dgate, the dK/dV group partials); the ranks' sub-tree roots folded in
    runtime/tp_canonical_tree.combine_tree must equal TP1 dk/dv. Forward also on the HEAD tile
    (paged decode). CP: the runtime/cp_attention backward at CP1 (the plan the trainer runs) and
    simulated CP2/4/8 ranks (each rank's kernel call on its contiguous rows against the full
    K/V; rank roots folded in the canonical tree; all_gather is a pure copy) vs the one-shot
    kernel. The real two-GPU CP2 run is runtime/check_cp_parity.py.
    """
    from rl_engine.models.muse_glimmer.ops.canonical_tree import combine_tree
    from rl_engine.models.muse_glimmer.runtime import cp_attention as cpa

    tp_records: list[dict[str, Any]] = []
    cp_records: list[dict[str, Any]] = []
    controls: list[dict[str, Any]] = []
    for window in (None, W):
        controls.append(_association_controls(2100, window))
        tp_records += _tp_parity_varlen("T=2100", [2100], 80_000, window)
        tp_records += _tp_parity_varlen("packed[1500,600,333]", [1500, 600, 333], 81_000, window)
        tp_records += _tp_parity_varlen("T=8192", [8192], 82_000, window)
        tp_records += _tp_parity_paged_decode(8, 2049, window)
        for n in cp_lengths:
            inp = _inputs(n, seed=90_000 + n)
            cu = _i32([0, n])
            pos = torch.arange(n, dtype=torch.int32, device=DEV)
            _y, o, lse = _fwd_varlen(inp, cu, cu, pos, window)
            one = bwd.strict_gated_attention_bwd(
                inp["q"], inp["k"], inp["v"], inp["gate"], o, lse, inp["dy"], cu, cu, pos,
                window=window,
            )
            cp1 = cpa.cp_gated_attention_backward(
                inp["q"], inp["k"], inp["v"], inp["gate"], o, lse, inp["dy"], pos,
                cpa.CPAttentionPlan(1, 0), None, window=window,
            )
            cp_records.append({
                "cp": 1, "path": "runtime/cp_attention.cp_gated_attention_backward", "T": n,
                "window": window, "dq": _mm(cp1[0], one[0]), "dk": _mm(cp1[1], one[1]),
                "dv": _mm(cp1[2], one[2]), "dgate": _mm(cp1[3], one[3]),
                "numel_dq": one[0].numel(), "numel_dk": one[1].numel(),
            })
            for cp in (2, 4, 8):
                if n % cp:
                    continue
                span = n // cp
                rk, rv, dqs, dgs = [], [], [], []
                for i in range(cp):
                    rows = slice(i * span, (i + 1) * span)
                    dg_i, do_i, de_i = bwd.gated_attention_bwd_preprocess(
                        inp["dy"][rows].contiguous(), o[rows].contiguous(),
                        inp["gate"][rows].contiguous(),
                    )
                    dq_i, dkp_i, dvp_i = bwd.strict_gated_attention_bwd_partials(
                        inp["q"][rows].contiguous(), inp["k"], inp["v"], do_i,
                        lse[rows].contiguous(), de_i, _i32([0, span]), cu,
                        pos[rows].contiguous(), window=window,
                    )
                    rk.append(dkp_i)
                    rv.append(dvp_i)
                    dqs.append(dq_i)
                    dgs.append(dg_i)
                got_k = bwd.combine_kv_partials(combine_tree(rk))
                got_v = bwd.combine_kv_partials(combine_tree(rv))
                cp_records.append({
                    "cp": cp, "path": "simulated ranks (kernel calls + canonical rank tree)",
                    "T": n, "window": window, "rows_per_rank": span,
                    "dq": _mm(torch.cat(dqs), one[0]), "dk": _mm(got_k, one[1]),
                    "dv": _mm(got_v, one[2]), "dgate": _mm(torch.cat(dgs), one[3]),
                    "numel_dq": one[0].numel(), "numel_dk": one[1].numel(),
                })
    return {"tp": tp_records, "cp": cp_records, "controls": controls}


def _association_controls(n: int, window) -> dict[str, Any]:
    """Teeth for the zero counts above: the same fp32 leaves folded in another association
    (sequential left folds instead of the canonical tree) change the bf16 dK/dV bits."""
    inp = _inputs(n, seed=80_000)
    cu = _i32([0, n])
    pos = torch.arange(n, dtype=torch.int32, device=DEV)
    _y, o, lse = _fwd_varlen(inp, cu, cu, pos, window)
    _dg, do, delta = bwd.gated_attention_bwd_preprocess(inp["dy"], o, inp["gate"])
    _dq, leaf_k, _leaf_v = bwd.strict_gated_attention_bwd_chunk_leaves(
        inp["q"], inp["k"], inp["v"], do, lse, delta, cu, cu, pos, window=window
    )
    dk = bwd.combine_kv_partials(bwd.fold_query_chunks(leaf_k, HKV))

    def left_fold(parts):
        acc = parts[0]
        for p in parts[1:]:
            acc = acc + p
        return acc

    # (a) chunks folded left-to-right (the v1 "query blocks ascending" association), groups tree
    chunk_seq = left_fold(list(leaf_k.unbind(0)))  # (G, Tk, D)
    part_seq = torch.zeros((chunk_seq.shape[0], n, HKV, D), dtype=torch.float32, device=DEV)
    for g in range(chunk_seq.shape[0]):
        part_seq[g, :, g // (chunk_seq.shape[0] // HKV)] = chunk_seq[g]
    dk_a = bwd.combine_kv_partials(part_seq)
    # (b) canonical chunk tree, groups folded left-to-right (an all_reduce-like ring order)
    part = bwd.fold_query_chunks(leaf_k, HKV)
    dk_b = left_fold(list(part.unbind(0))).to(torch.bfloat16)
    return {"T": n, "window": window, "numel_dk": dk.numel(),
            "dk_chunks_left_fold_vs_tree": _mm(dk_a, dk),
            "dk_groups_left_fold_vs_tree": _mm(dk_b, dk)}


def _diff_into(rec: dict[str, Any], prefix: str, names, new, old) -> None:
    for name, a, b in zip(names, new, old):
        diff = (a.float() - b.float()).abs()
        key = f"{prefix}/{name}"
        rec[f"{key}_mismatch"] = _mm(a, b)
        rec[f"{key}_numel"] = a.numel()
        rec[f"{key}_max_abs_diff"] = float(diff.max().item())
        rec[f"{key}_max_rel_diff"] = float(
            (diff.max() / b.float().abs().max().clamp_min(1e-30)).item()
        )
        rec[f"{key}_max_ulp"] = _max_ulp(a, b)


def _max_ulp(a: torch.Tensor, b: torch.Tensor) -> int:
    """Largest distance in units in the last place (bf16 / fp32), over all elements."""
    if a.shape != b.shape or a.dtype != b.dtype or not a.is_floating_point():
        return -1
    width = a.element_size() * 8
    ia, ib = _bits(a).to(torch.int64), _bits(b).to(torch.int64)
    mag = (1 << (width - 1)) - 1
    # sign-magnitude -> monotonic integer line (+0 and -0 both map to 0)
    ka = torch.where(ia < 0, -(ia & mag), ia & mag)
    kb = torch.where(ib < 0, -(ib & mag), ib & mag)
    return int((ka - kb).abs().max().item()) if a.numel() else 0


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["digest", "bench", "bwd-diff", "bwd-ab", "parity"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--rounds", type=int, default=5,
                    help="bwd-ab / bench forward rows: alternating timing rounds")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--old-module", default=None)
    ap.add_argument("--sections", nargs="+", choices=SECTIONS, default=list(SECTIONS),
                    help="bench mode: which timing series to run")
    args = ap.parse_args(argv)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    t0 = time.time()
    if args.mode == "digest":
        payload = run_digest()
    elif args.mode == "bench":
        payload = run_bench(args.iters, args.quick, tuple(args.sections), args.rounds)
    elif args.mode == "parity":
        payload = run_parity()
    else:
        if not args.old_module:
            ap.error(f"{args.mode} needs --old-module")
        if args.mode == "bwd-ab":
            payload = run_bwd_ab(args.old_module, args.iters, args.rounds)
        else:
            payload = run_bwd_diff(args.old_module)
    payload["env"] = _env()
    payload["modules"] = {"forward": ga.__file__, "backward": bwd.__file__,
                          "backward_kernel_id": getattr(bwd, "KERNEL_ID", None)}
    payload["argv"] = sys.argv
    payload["head_count_params"] = _supports_heads()
    payload["wall_s"] = time.time() - t0
    with open(args.out, "w") as fh:
        json.dump(payload, fh, indent=1)
    print(f"wrote {args.out} in {payload['wall_s']:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
