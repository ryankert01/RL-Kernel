# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Measurement driver for the strict gated attention (O5): bit digests, timing series, bit diffs.

Writes machine-readable records for the POC report (``reports/muse_glimmer_poc/data/
kernels_attention.json``). Three modes, all single-GPU (run under ``~/bin/gpu-lease -n 1``)::

    python -m rl_engine.models.muse_glimmer.ops.gated_attention_bench digest --out d.json
    python -m rl_engine.models.muse_glimmer.ops.gated_attention_bench bench --out b.json
    python -m rl_engine.models.muse_glimmer.ops.gated_attention_bench bwd-diff \
        --old-module /path/to/old_gated_attention_bwd.py --out diff.json

* ``digest``: sha256 of the forward outputs ``(y, o, lse)`` over a matrix of shapes and paths
  (one-shot varlen, TOKEN/HEAD tiles, packed batch, chunked prefill, paged prefill and decode)
  and of the backward outputs ``(dq, dk, dv, dgate)``. Inputs are generated on the CPU from
  fixed seeds, so two commits can be compared by digest alone ("forward bits unchanged").
* ``bench``: CUDA-event timings in ms (prefill forward, paged decode forward, backward) for the
  strict kernels and the torch SDPA baseline, the ratio, and the backward's transient peak
  memory in GiB; optional TP head-shard geometries when the wrappers accept head counts.
* ``bwd-diff``: per shape, mismatch counts / max relative difference between the backward of an
  OLD module file (e.g. ``git show <sha>:rl_engine/.../triton/gated_attention_bwd.py``) and the
  current one, on the same strict-forward ``o``/``lse``.

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


# --------------------------------------------------------------------------------- bench
def _sdpa_gated(q4, k4, v4, gate, *, is_causal: bool, mask):
    out = F.scaled_dot_product_attention(
        q4, k4, v4, attn_mask=mask, is_causal=is_causal, enable_gqa=True
    )
    o = out.transpose(1, 2).reshape(gate.shape)
    return (o.float() * torch.sigmoid(gate.float())).to(torch.bfloat16)


def bench_prefill(n: int, window: Optional[int], iters: int, hq: int = HQ, hkv: int = HKV):
    inp = _inputs(n, seed=1, hq=hq, hkv=hkv)
    cu = _i32([0, n])
    pos = torch.arange(n, dtype=torch.int32, device=DEV)
    kw: dict[str, Any] = {"max_seqlen_q": n}
    if (hq, hkv) != (HQ, HKV):
        kw.update(num_q_heads=hq, num_kv_heads=hkv)
    strict_ms = _time_ms(lambda: _fwd_varlen(inp, cu, cu, pos, window, **kw), iters)
    rec: dict[str, Any] = {"T": n, "window": window, "H_Q": hq, "H_KV": hkv,
                           "strict_ms": strict_ms}
    if (hq, hkv) == (HQ, HKV):
        q4 = inp["q"].transpose(0, 1).unsqueeze(0).contiguous()
        k4 = inp["k"].transpose(0, 1).unsqueeze(0).contiguous()
        v4 = inp["v"].transpose(0, 1).unsqueeze(0).contiguous()
        if window is None:
            mask, causal = None, True
        else:
            i = torch.arange(n, device=DEV)
            mask, causal = (i[None, :] <= i[:, None]) & (i[None, :] > i[:, None] - window), False
        sdpa_ms = _time_ms(
            lambda: _sdpa_gated(q4, k4, v4, inp["gate"], is_causal=causal, mask=mask), iters
        )
        rec.update(sdpa_ms=sdpa_ms, ratio=strict_ms / sdpa_ms)
    return rec


def bench_decode(bsz: int, ctx: int, window: Optional[int], iters: int):
    g = torch.Generator(device="cpu").manual_seed(bsz * 7 + ctx)
    k_all = torch.randn(bsz * ctx, HKV, D, generator=g).to(DEV, torch.bfloat16)
    v_all = torch.randn(bsz * ctx, HKV, D, generator=g).to(DEV, torch.bfloat16)
    q = (torch.randn(bsz, HQ, D, generator=g) * 3.87).to(DEV, torch.bfloat16)
    gate = torch.randn(bsz, HQ * D, generator=g).to(DEV, torch.bfloat16)
    kc, vc, table, seq_lens = _paged_from_contiguous(k_all, v_all, bsz, ctx)
    cu_q = _i32(range(bsz + 1))
    strict_ms = _time_ms(
        lambda: ga.strict_gated_attention_fwd_paged(
            q, gate, kc, vc, table, seq_lens, cu_q, window=window, max_seqlen_q=1
        ),
        iters,
    )
    q4 = q.unsqueeze(2).contiguous()
    k4 = k_all.view(bsz, ctx, HKV, D).transpose(1, 2).contiguous()
    v4 = v_all.view(bsz, ctx, HKV, D).transpose(1, 2).contiguous()
    if window is None:
        mask = None
    else:
        j = torch.arange(ctx, device=DEV)
        mask = ((j <= ctx - 1) & (j > ctx - 1 - window)).view(1, 1, 1, ctx)
    sdpa_ms = _time_ms(lambda: _sdpa_gated(q4, k4, v4, gate, is_causal=False, mask=mask), iters)
    return {"B": bsz, "ctx": ctx, "window": window, "strict_ms": strict_ms, "sdpa_ms": sdpa_ms,
            "ratio": strict_ms / sdpa_ms}


def bench_backward(n: int, window: Optional[int], iters: int, hq: int = HQ, hkv: int = HKV):
    inp = _inputs(n, seed=2, hq=hq, hkv=hkv)
    cu = _i32([0, n])
    pos = torch.arange(n, dtype=torch.int32, device=DEV)
    kw: dict[str, Any] = {}
    if (hq, hkv) != (HQ, HKV):
        kw.update(num_q_heads=hq, num_kv_heads=hkv)
    _y, o, lse = _fwd_varlen(inp, cu, cu, pos, window, **kw)

    def strict():
        return bwd.strict_gated_attention_bwd(
            inp["q"], inp["k"], inp["v"], inp["gate"], o, lse, inp["dy"], cu, cu, pos,
            window=window, **kw,
        )

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
        rec.update(sdpa_causal_bwd_ms=sdpa_ms, ratio=strict_ms / sdpa_ms)
    return rec


def run_bench(iters: int, quick: bool) -> dict[str, Any]:
    lengths = (512, 1024, 2048, 4096, 8192) if not quick else (1024, 4096)
    out: dict[str, Any] = {"prefill_forward": [], "decode_forward": [], "backward": [],
                           "tp_shard_forward": [], "tp_shard_backward": []}
    for window in (None, W):
        for n in lengths:
            out["prefill_forward"].append(bench_prefill(n, window, iters))
            out["backward"].append(bench_backward(n, window, max(5, iters // 2)))
        for bsz in ((1, 16, 64, 256) if not quick else (1, 64)):
            out["decode_forward"].append(bench_decode(bsz, 4096, window, iters))
    if _supports_heads():
        for hq, hkv in ((32, 2), (16, 1), (8, 1), (4, 1)):
            for window in (None, W):
                kw = {} if (hq, hkv) == (HQ, HKV) else {"hq": hq, "hkv": hkv}
                rec = bench_prefill(4096, window, iters, **kw)
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
            records.append(rec)
    return {"records": records, "old_module": old_path}


def _diff_into(rec: dict[str, Any], prefix: str, names, new, old) -> None:
    for name, a, b in zip(names, new, old):
        diff = (a.float() - b.float()).abs()
        key = f"{prefix}/{name}"
        rec[f"{key}_mismatch"] = int((a != b).sum().item())
        rec[f"{key}_numel"] = a.numel()
        rec[f"{key}_max_abs_diff"] = float(diff.max().item())
        rec[f"{key}_max_rel_diff"] = float(
            (diff.max() / b.float().abs().max().clamp_min(1e-30)).item()
        )


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["digest", "bench", "bwd-diff"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--old-module", default=None)
    args = ap.parse_args(argv)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    t0 = time.time()
    if args.mode == "digest":
        payload = run_digest()
    elif args.mode == "bench":
        payload = run_bench(args.iters, args.quick)
    else:
        if not args.old_module:
            ap.error("bwd-diff needs --old-module")
        payload = run_bwd_diff(args.old_module)
    payload["env"] = _env()
    payload["argv"] = sys.argv
    payload["head_count_params"] = _supports_heads()
    payload["wall_s"] = time.time() - t0
    with open(args.out, "w") as fh:
        json.dump(payload, fh, indent=1)
    print(f"wrote {args.out} in {payload['wall_s']:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
