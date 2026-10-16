"""Reviewer probes on the reviewed tree (b5b77b0).

Usage: PYTHONPATH=<worktree> python probe_extra.py <out_json>
"""

import json
import sys
import traceback

import torch

from rl_engine.models.muse_glimmer.ops import det_gemm as dg
from rl_engine.models.muse_glimmer.ops import lm_head_softcap_logprob as lm
from rl_engine.models.muse_glimmer.ops import swiglu_ffn as sf
from rl_engine.models.muse_glimmer.ops.canonical_tree import combine_tree
from rl_engine.models.muse_glimmer.triton import det_gemm_tree as kt

DEV, BF, FP32 = "cuda", torch.bfloat16, torch.float32
res = {}


def rnd(shape, seed, scale=1.0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    return (torch.randn(*shape, device=DEV, generator=g) * scale).to(BF)


def run(key, fn):
    try:
        res[key] = fn()
    except Exception as exc:  # noqa: BLE001
        res[key] = f"ERROR {type(exc).__name__}: {str(exc)[:300]}"
        traceback.print_exc()
    print(key, res[key], flush=True)


# 1. k_proj dA (reduce N=256) at decode M: fast kernel vs an independent kernel (adapter: one
#    launch per chunk, so no in-kernel tree adds), both fp32.
def dA_vs_adapter():
    w = rnd((256, 6656), 1, 0.02)
    adapter = sf.TritonDetGemmAdapter()
    out = {}
    for m in (1, 5, 16, 17, 64, 65, 256):
        gy = rnd((m, 256), 10 + m)
        fast = kt.tree_gemm(gy, w, out_dtype=FP32, allow_pad=True)
        indep = adapter(gy, w.t(), out_dtype=FP32)
        out[m] = int((fast != indep).sum())
    return out


run("dA_k_proj_fast_vs_adapter_mismatch_by_M", dA_vs_adapter)


# 2. dW ragged at several T, fast vs adapter twin (fp32), q_proj geometry reduced N.
def dW_vs_adapter():
    adapter = sf.TritonDetGemmAdapter()
    out = {}
    for t in (1, 31, 33, 200, 257, 300, 511, 513, 777, 1000, 4001, 8192):
        dy = rnd((t, 512), 20 + t)
        a = rnd((t, 640), 30 + t)
        fast = dg.fast_det_gemm(dy.t(), a.t(), out_dtype=FP32, ragged_k=True)
        twin = adapter(dy.t(), a.t(), out_dtype=FP32, ragged_k=True)
        plan = kt.last_launch()["plan"]
        out[t] = {"mismatch": int((fast != twin).sum()), "block_k": plan["block_k"],
                  "chunk_len": plan["chunk_len"]}
    return out


run("dW_fast_vs_adapter", dW_vs_adapter)


# 3. CUDA graph capture of decode GEMMs (vLLM decode is graph-captured).
def graph_capture():
    w = rnd((4096, 6656), 40, 0.02)
    wk = rnd((256, 6656), 41, 0.02)
    out = {}
    for m in (1, 16, 64, 128, 256):
        a = rnd((m, 6656), 50 + m)
        with torch.inference_mode():
            ref = dg.fast_det_gemm(a, w)
            refk = dg.fast_det_gemm(a, wk)
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(2):
                    dg.fast_det_gemm(a, w)
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                y = dg.fast_det_gemm(a, w)
                yk = dg.fast_det_gemm(a, wk)
            g.replay()
            torch.cuda.synchronize()
        out[m] = bool(torch.equal(y, ref) and torch.equal(yk, refk))
    return out


run("cuda_graph_capture_equal", graph_capture)


# 4. lm_head small geometry under CUDA graph capture with validate_targets=False.
def lm_graph():
    h_size, vocab, vp = 256, 1900, 2048
    h = rnd((4, h_size), 60)
    w = rnd((vocab, h_size), 61, 0.05)
    wp = torch.zeros(vp, h_size, device=DEV, dtype=BF)
    wp[:vocab] = w
    tg = torch.randint(0, vocab, (4,), device=DEV)
    with torch.inference_mode():
        ref = lm.lm_head_softcap_logprob(h, wp, tg, vocab_size=vocab, validate_targets=False)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            lm.lm_head_softcap_logprob(h, wp, tg, vocab_size=vocab, validate_targets=False)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            y = lm.lm_head_softcap_logprob(h, wp, tg, vocab_size=vocab, validate_targets=False)
        g.replay()
        torch.cuda.synchronize()
    return bool(torch.equal(y, ref))


run("lm_head_graph_capture_novalidate", lm_graph)


# 5. Column-parallel autograd at decode M (decode16 / decode64 classes), simulated TP2/4.
class Gather:
    def __init__(self, partials, rank):
        self.partials, self.rank = partials, rank

    def all_gather(self, x):
        return torch.stack([x[0] if r == self.rank else p for r, p in enumerate(self.partials)])


def col_par_decode():
    out = {}
    for name, (n, k) in (("q_proj", (4096, 6656)), ("k_proj", (256, 6656))):
        w = rnd((n, k), 70, 0.02)
        for m in (1, 16, 33):
            a = rnd((m, k), 71 + m)
            gy = rnd((m, n), 72 + m)
            a1 = a.clone().requires_grad_(True)
            dg.fast_det_gemm(a1, w).backward(gy)
            for tp in (2, 4):
                nl = n // tp
                rows = [slice(r * nl, (r + 1) * nl) for r in range(tp)]
                parts = [dg.column_parallel_da_partial(gy[:, s], w[s], tp) for s in rows]
                bad = 0
                for r, s in enumerate(rows):
                    ar = a.clone().requires_grad_(True)
                    y = dg.column_parallel_fast_det_gemm(
                        ar, w[s], tp_size=tp, da_reduce=dg.tree_reduce_over_ranks(Gather(parts, r))
                    )
                    y.backward(gy[:, s])
                    bad += int((ar.grad != a1.grad).sum())
                out[f"{name}/M{m}/TP{tp}"] = bad
    return out


run("column_parallel_decode_dA_mismatch", col_par_decode)


# 6. trace naming when the strict reference model passes the raw fast_det_gemm.
def trace_names():
    return {
        "lm_backend_name_raw": lm.gemm_backend_name(dg.fast_det_gemm),
        "lm_gemm_order_raw": lm.trace(gemm=dg.fast_det_gemm)["reduction_order"]["gemm"][:80],
        "lm_backend_name_default": lm.gemm_backend_name(),
        "ffn_gemm_name_raw": sf._gemm_name(dg.fast_det_gemm),
        "ffn_gemm_name_default": sf._gemm_name(sf.default_gemm()),
        "verified_ir_variants": len(kt._VERIFIED_IR),
    }


run("trace_names", trace_names)


# 7. swiglu_ffn backward with raw fast_det_gemm (strict reference model) at ragged T.
def ffn_raw():
    hid, ffn = 256, 512
    out = {}
    for t in (1, 300):
        x = rnd((t, hid), 80).requires_grad_(True)
        wg = rnd((ffn, hid), 81, 0.06).requires_grad_(True)
        wu = rnd((ffn, hid), 82, 0.06).requires_grad_(True)
        wd = rnd((hid, ffn), 83, 0.04).requires_grad_(True)
        y = sf.swiglu_ffn(x, wg, wu, wd, gemm=dg.fast_det_gemm)
        y.backward(rnd((t, hid), 84))
        grads1 = [p.grad.clone() for p in (x, wg, wu, wd)]
        for p in (x, wg, wu, wd):
            p.grad = None
        y = sf.swiglu_ffn(x, wg, wu, wd)  # default shim
        y.backward(rnd((t, hid), 84))
        out[t] = all(torch.equal(g, p.grad) for g, p in zip(grads1, (x, wg, wu, wd)))
    return out


run("ffn_raw_vs_default_gemm_grads_equal", ffn_raw)

json.dump(res, open(sys.argv[1], "w"), indent=1, default=str)
