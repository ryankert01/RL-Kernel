"""Before/after benchmark for the GEMM row (fast_det_gemm, swiglu_ffn, lm_head_softcap_logprob).

Writes one JSON with raw per-shape series.  Uses only APIs present before and after the fixes.
Usage: python bench_rows.py <out_json> [--quick]
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import time

import torch
import triton

from rl_engine.models.muse_glimmer.ops import det_gemm as dg
from rl_engine.models.muse_glimmer.ops import lm_head_softcap_logprob as lm
from rl_engine.models.muse_glimmer.ops import swiglu_ffn as sf
from rl_engine.models.muse_glimmer.triton import det_gemm_tree as kt

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
DEV = "cuda"
BF = torch.bfloat16
QUICK = "--quick" in sys.argv
H, F = 6656, 19968
SHAPES = [
    ("q_proj", 4096, 6656),
    ("k_proj", 256, 6656),
    ("v_proj", 256, 6656),
    ("attn_gate", 4096, 6656),
    ("o_proj", 6656, 4096),
    ("mlp_gate", 19968, 6656),
    ("up_proj", 19968, 6656),
    ("down_proj", 6656, 19968),
    ("lm_head", 202752, 6656),
]


def t_ms(fn, iters=10, warmup=2):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        samples.append(s.elapsed_time(e))
    return statistics.median(samples), samples


def iters_for(flop):
    if flop > 5e12:
        return 5, 1
    if flop > 5e11:
        return 10, 2
    return 30, 5


def rnd(*s, scale=1.0, gen=None):
    return (torch.randn(*s, device=DEV, generator=gen) * scale).to(BF)


def peak_of(fn):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    fn()
    torch.cuda.synchronize()
    return (torch.cuda.max_memory_allocated() - base) / 2**30


def safe(rec, key, fn):
    try:
        rec[key] = fn()
    except Exception as exc:  # noqa: BLE001
        rec[key] = None
        rec[key + "_error"] = f"{type(exc).__name__}: {str(exc)[:240]}"


def gemm_forward(out):
    ms_list = (1, 16, 64, 256, 1024, 4096) if not QUICK else (16, 4096)
    gen = torch.Generator(device=DEV).manual_seed(0)
    for name, n, k in SHAPES:
        w = rnd(n, k, scale=0.02, gen=gen)
        for m in ms_list:
            a = rnd(m, k, gen=gen)
            flop = 2.0 * m * n * k
            it, wu = iters_for(flop)
            rec = {"shape": name, "M": m, "N": n, "K": k, "flop": flop}
            ms, samples = t_ms(lambda: dg.fast_det_gemm(a, w), it, wu)
            launch = kt.last_launch()
            rec.update(strict_ms=ms, strict_samples_ms=samples,
                       strict_tflops=flop / ms / 1e9, shape_class=launch["shape_class"],
                       block_k=launch["plan"]["block_k"], n_regs=launch["n_regs"],
                       n_spills=launch["n_spills"])
            ms_c, _ = t_ms(lambda: torch.matmul(a, w.t()), it, wu)
            rec.update(cublas_ms=ms_c, cublas_tflops=flop / ms_c / 1e9,
                       strict_over_cublas=ms / ms_c)
            out.append(rec)
            print("fwd", {k_: v for k_, v in rec.items() if k_ != "strict_samples_ms"},
                  flush=True)
            del a
        del w
        torch.cuda.empty_cache()


def gemm_backward_kernels(out):
    """dA (reduce over N) and dW (reduce over tokens, ragged rule) kernel calls."""
    gen = torch.Generator(device=DEV).manual_seed(1)
    for name, n, k in SHAPES:
        if name == "lm_head":
            continue
        w = rnd(n, k, scale=0.02, gen=gen)
        m = 4096
        g = rnd(m, n, gen=gen)
        flop = 2.0 * m * n * k
        it, wu = iters_for(flop)
        rec = {"kind": "dA", "shape": name, "M": m, "reduce_len": n, "N_out": k, "flop": flop}
        ms, _ = t_ms(lambda: kt.tree_gemm(g, w, allow_pad=True), it, wu)
        launch = kt.last_launch()
        rec.update(strict_ms=ms, strict_tflops=flop / ms / 1e9, block_k=launch["plan"]["block_k"],
                   chunk_len=launch["plan"]["chunk_len"])
        ms_c, _ = t_ms(lambda: torch.matmul(g, w), it, wu)
        rec.update(cublas_ms=ms_c, cublas_tflops=flop / ms_c / 1e9)
        out.append(rec)
        print(rec, flush=True)
        del g
        tokens_list = (256, 300, 512, 1024, 4001, 4096) if not QUICK else (256, 4096)
        if name in ("k_proj", "mlp_gate", "down_proj", "o_proj"):
            for t in tokens_list:
                a = rnd(t, k, gen=gen)
                g = rnd(t, n, gen=gen)
                flop = 2.0 * t * n * k
                it, wu = iters_for(flop)
                rec = {"kind": "dW", "shape": name, "tokens": t, "N": n, "K": k, "flop": flop}
                ms, _ = t_ms(lambda: kt.tree_gemm(g.t(), a, allow_pad=True), it, wu)
                launch = kt.last_launch()
                rec.update(strict_ms=ms, strict_tflops=flop / ms / 1e9,
                           block_k=launch["plan"]["block_k"],
                           chunk_len=launch["plan"]["chunk_len"], k_pad=launch["plan"]["k_pad"])
                ms_c, _ = t_ms(lambda: torch.matmul(g.t(), a), it, wu)
                rec.update(cublas_ms=ms_c, cublas_tflops=flop / ms_c / 1e9)
                out.append(rec)
                print(rec, flush=True)
                del a, g
        del w
        torch.cuda.empty_cache()


def gemm_autograd(out):
    gen = torch.Generator(device=DEV).manual_seed(2)
    for name, n, k in SHAPES:
        if name in ("lm_head", "v_proj", "up_proj", "attn_gate"):
            continue
        m = 4096
        a = rnd(m, k, gen=gen).requires_grad_(True)
        w = rnd(n, k, scale=0.02, gen=gen).requires_grad_(True)
        g = rnd(m, n, gen=gen)
        flop = 3 * 2.0 * m * n * k
        rec = {"shape": name, "M": m, "N": n, "K": k, "flop_fwd_bwd": flop}

        def strict():
            dg.fast_det_gemm(a, w).backward(g)

        def native():
            torch.nn.functional.linear(a, w).backward(g)

        ms, _ = t_ms(strict, 5, 2)
        ms_n, _ = t_ms(native, 5, 2)
        rec.update(strict_ms=ms, strict_tflops=flop / ms / 1e9, native_ms=ms_n,
                   native_tflops=flop / ms_n / 1e9)
        out.append(rec)
        print(rec, flush=True)
        del a, w, g
        torch.cuda.empty_cache()


def ffn(out):
    gen = torch.Generator(device=DEV).manual_seed(3)
    x_all = rnd(4096, H, gen=gen)
    wg = rnd(F, H, scale=H**-0.5, gen=gen)
    wu = rnd(F, H, scale=H**-0.5, gen=gen)
    wd = rnd(H, F, scale=F**-0.5, gen=gen)
    gemm = sf.default_gemm()
    for t in ((1, 16, 64, 256, 1024, 4096) if not QUICK else (16, 4096)):
        x = x_all[:t].contiguous()
        flop = 2.0 * t * H * F * 3
        it, wu_ = iters_for(flop)
        rec = {"pass": "forward", "T": t, "flop": flop}
        with torch.no_grad():
            safe(rec, "strict_ms", lambda: t_ms(lambda: sf.swiglu_ffn(x, wg, wu, wd, gemm=gemm),
                                                it, wu_)[0])
            safe(rec, "native_ms", lambda: t_ms(lambda: torch.nn.functional.linear(
                torch.nn.functional.silu(torch.nn.functional.linear(x, wg))
                * torch.nn.functional.linear(x, wu), wd), it, wu_)[0])
        if rec.get("strict_ms"):
            rec["strict_tflops"] = flop / rec["strict_ms"] / 1e9
        out.append(rec)
        print(rec, flush=True)
    for t in ((64, 256, 300, 4001, 4096) if not QUICK else (256, 4096)):
        x = x_all[:t].contiguous() if t <= 4096 else None
        dy = rnd(t, H, scale=t**-0.5, gen=gen)
        flop = 2.0 * t * H * F * 3 * 3
        rec = {"pass": "forward+backward", "T": t, "flop": flop}
        leaves = [x.clone().requires_grad_(True), wg.clone().requires_grad_(True),
                  wu.clone().requires_grad_(True), wd.clone().requires_grad_(True)]

        def strict():
            for leaf in leaves:
                leaf.grad = None
            sf.swiglu_ffn(*leaves, gemm=gemm).backward(dy)

        def native():
            for leaf in leaves:
                leaf.grad = None
            xx, a1, a2, a3 = leaves
            torch.nn.functional.linear(
                torch.nn.functional.silu(torch.nn.functional.linear(xx, a1))
                * torch.nn.functional.linear(xx, a2), a3).backward(dy)

        safe(rec, "strict_ms", lambda: t_ms(strict, 3, 1)[0])
        safe(rec, "strict_peak_gib", lambda: peak_of(strict))
        safe(rec, "native_ms", lambda: t_ms(native, 3, 1)[0])
        safe(rec, "native_peak_gib", lambda: peak_of(native))
        if rec.get("strict_ms"):
            rec["strict_tflops"] = flop / rec["strict_ms"] / 1e9
        out.append(rec)
        print(rec, flush=True)
        del leaves
        torch.cuda.empty_cache()


def lm_head(out):
    gen = torch.Generator(device=DEV).manual_seed(4)
    w = rnd(lm.VOCAB_SIZE, H, scale=0.02, gen=gen)
    w_padded = lm.pad_lm_head_weight(w)
    del w
    h_all = rnd(4096, H, gen=gen)
    tg_all = torch.randint(0, lm.VOCAB_SIZE, (4096,), device=DEV, generator=gen)
    dl_all = torch.randn(4096, device=DEV, generator=gen)
    for t in ((1, 64, 256, 1024, 4096) if not QUICK else (256, 4096)):
        h, tg = h_all[:t].contiguous(), tg_all[:t].contiguous()
        flop = 2.0 * t * H * lm.PADDED_VOCAB_SIZE
        it, wu_ = iters_for(flop)
        rec = {"pass": "forward", "T": t, "flop": flop}
        with torch.no_grad():
            safe(rec, "strict_ms", lambda: t_ms(lambda: lm.lm_head_softcap_logprob(
                h, w_padded, tg), it, wu_)[0])
            safe(rec, "strict_peak_gib", lambda: peak_of(lambda: lm.lm_head_softcap_logprob(
                h, w_padded, tg)))
        if rec.get("strict_ms"):
            rec["strict_tflops"] = flop / rec["strict_ms"] / 1e9
        out.append(rec)
        print(rec, flush=True)
    for t in ((256, 300, 4001, 4096) if not QUICK else (256,)):
        h, tg, dl = h_all[:t].contiguous(), tg_all[:t].contiguous(), dl_all[:t].contiguous()
        flop = 3 * 2.0 * t * H * lm.PADDED_VOCAB_SIZE + 2.0 * t * H * lm.PADDED_VOCAB_SIZE
        rec = {"pass": "forward+backward", "T": t, "flop": flop}
        hh = h.clone().requires_grad_(True)
        ww = w_padded.clone().requires_grad_(True)

        def strict():
            hh.grad = None
            ww.grad = None
            lm.lm_head_softcap_logprob(hh, ww, tg).backward(dl)

        safe(rec, "strict_ms", lambda: t_ms(strict, 3, 1)[0])
        safe(rec, "strict_peak_gib", lambda: peak_of(strict))
        if rec.get("strict_ms"):
            rec["strict_tflops"] = flop / rec["strict_ms"] / 1e9
        out.append(rec)
        print(rec, flush=True)
        del hh, ww
        torch.cuda.empty_cache()


def main():
    path = sys.argv[1]
    rec = {
        "env": {
            "gpu": torch.cuda.get_device_name(),
            "capability": list(torch.cuda.get_device_capability()),
            "torch": torch.__version__, "triton": triton.__version__,
            "dg_module": dg.__file__,
            "commit": os.environ.get("BENCH_COMMIT", "unknown"),
        },
        "gemm_forward": [], "gemm_backward_kernels": [], "gemm_autograd": [], "ffn": [],
        "lm_head": [],
    }
    t0 = time.time()
    gemm_forward(rec["gemm_forward"])
    gemm_backward_kernels(rec["gemm_backward_kernels"])
    gemm_autograd(rec["gemm_autograd"])
    ffn(rec["ffn"])
    lm_head(rec["lm_head"])
    rec["wall_s"] = time.time() - t0
    with open(path, "w") as f:
        json.dump(rec, f, indent=1)
    print("wrote", path, "in", rec["wall_s"], "s")


if __name__ == "__main__":
    main()
