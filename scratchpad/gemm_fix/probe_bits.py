"""Bit ledger: SHA-256 of the output bytes of every GEMM path, to diff base vs fixed commit.

Usage: python probe_bits.py <out_json>   (run once per tree; inputs are seeded identically)
"""

import hashlib
import json
import sys

import torch

from rl_engine.models.muse_glimmer.ops import det_gemm as dg
from rl_engine.models.muse_glimmer.ops import lm_head_softcap_logprob as lm
from rl_engine.models.muse_glimmer.ops import swiglu_ffn as sf
from rl_engine.models.muse_glimmer.ops.canonical_tree import rank_chunk_ids

torch.backends.cuda.matmul.allow_tf32 = False
DEV, BF = "cuda", torch.bfloat16
SHAPES = {
    "q_proj": (4096, 6656), "k_proj": (256, 6656), "v_proj": (256, 6656),
    "attn_gate": (4096, 6656), "o_proj": (6656, 4096), "mlp_gate": (19968, 6656),
    "up_proj": (19968, 6656), "down_proj": (6656, 19968), "lm_head": (202752, 6656),
}
out = {}


def h(t):
    return hashlib.sha256(t.detach().contiguous().cpu().view(torch.uint8).numpy()).hexdigest()[:16]


def gen(seed):
    return torch.Generator(device=DEV).manual_seed(seed)


def rnd(shape, seed, scale=1.0):
    return (torch.randn(*shape, device=DEV, generator=gen(seed)) * scale).to(BF)


def run(key, fn):
    try:
        out[key] = fn()
    except Exception as exc:  # noqa: BLE001
        out[key] = f"ERROR {type(exc).__name__}: {str(exc)[:120]}"
    print(key, out[key], flush=True)


for i, (name, (n, k)) in enumerate(SHAPES.items()):
    w = rnd((n, k), 100 + i, 0.02)
    for m in ((1, 16, 64, 300, 4096) if name != "lm_head" else (1, 16, 64, 300)):
        a = rnd((m, k), 200 + m)
        run(f"fwd_bf16/{name}/M{m}", lambda: h(dg.fast_det_gemm(a, w)))
        run(f"fwd_fp32/{name}/M{m}", lambda: h(dg.fast_det_gemm(a, w, out_dtype=torch.float32)))
    if name in ("o_proj", "down_proj"):
        a = rnd((64, k), 300)
        for tp in (2, 4, 8):
            kl = k // tp
            run(f"tp_partial/{name}/TP{tp}", lambda: h(torch.stack([
                dg.fast_det_gemm_partial(a[:, r * kl:(r + 1) * kl], w[:, r * kl:(r + 1) * kl],
                                         rank_chunk_ids(tp, r)) for r in range(tp)])))
    if name != "lm_head":
        for t in (1, 64, 256, 300, 512, 1024, 4096):
            a = rnd((t, k), 400 + t).requires_grad_(True)
            ww = w.clone().requires_grad_(True)
            gy = rnd((t, n), 500 + t)

            def bwd():
                dg.fast_det_gemm(a, ww).backward(gy)
                res = {"dA": h(a.grad), "dW": h(ww.grad)}
                a.grad = None
                ww.grad = None
                return res

            run(f"autograd/{name}/T{t}", bwd)
    del w
    torch.cuda.empty_cache()

H, F = 6656, 19968
wg, wu, wd = rnd((F, H), 600, H**-0.5), rnd((F, H), 601, H**-0.5), rnd((H, F), 602, F**-0.5)
for t in (1, 64, 300, 4096):
    x = rnd((t, H), 700 + t)
    with torch.no_grad():
        run(f"ffn_fwd/T{t}", lambda: h(sf.swiglu_ffn(x, wg, wu, wd)))
for t in (64, 256, 300, 4096):
    x = rnd((t, H), 800 + t)
    dy = rnd((t, H), 900 + t, t**-0.5)

    def ffn_bwd():
        leaves = [v.clone().requires_grad_(True) for v in (x, wg, wu, wd)]
        sf.swiglu_ffn(*leaves).backward(dy)
        return {n_: h(v.grad) for n_, v in zip(("dx", "dw_gate", "dw_up", "dw_down"), leaves)}

    run(f"ffn_bwd/T{t}", ffn_bwd)
del wg, wu, wd
torch.cuda.empty_cache()

w_lm = lm.pad_lm_head_weight(rnd((lm.VOCAB_SIZE, H), 1000, 0.02))
for t in (1, 64, 300):
    hh = rnd((t, H), 1100 + t)
    tg = torch.randint(0, lm.VOCAB_SIZE, (t,), device=DEV, generator=gen(1200 + t))
    with torch.no_grad():
        run(f"lm_head_fwd/T{t}", lambda: h(lm.lm_head_softcap_logprob(hh, w_lm, tg)))
for t in (256, 300):
    hh = rnd((t, H), 1300 + t)
    tg = torch.randint(0, lm.VOCAB_SIZE, (t,), device=DEV, generator=gen(1400 + t))
    dl = torch.randn(t, device=DEV, generator=gen(1500 + t))

    def lm_bwd():
        a_ = hh.clone().requires_grad_(True)
        w_ = w_lm.clone().requires_grad_(True)
        lm.lm_head_softcap_logprob(a_, w_, tg).backward(dl)
        return {"dh": h(a_.grad), "dW": h(w_.grad)}

    run(f"lm_head_bwd/T{t}", lm_bwd)

with open(sys.argv[1], "w") as f:
    json.dump(out, f, indent=1)
