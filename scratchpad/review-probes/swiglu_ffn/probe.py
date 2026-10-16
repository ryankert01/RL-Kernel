import json
import time

import torch

from rl_engine.models.muse_glimmer.ops import det_gemm as dg
from rl_engine.models.muse_glimmer.ops.swiglu_ffn import (
    TritonDetGemmAdapter,
    default_gemm,
    swiglu_ffn,
    trace,
)

torch.backends.cuda.matmul.allow_tf32 = False
H, F = 6656, 19968
t0 = time.time()

# --- 1. trace() of the default (fast_det_gemm) path -------------------------------------
gemm = default_gemm()
print("default gemm type:", type(gemm).__name__, "| fast_det_gemm has .trace attr:",
      hasattr(dg.fast_det_gemm, "trace"))
tr = trace()
print("trace()['tile_config']['gemm'] =", json.dumps(tr["tile_config"]["gemm"]))


def randn(shape, seed, scale=1.0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return (torch.randn(*shape, generator=g, dtype=torch.float32) * scale).to("cuda", torch.bfloat16)


# --- 2. FFN batch invariance across fast_det_gemm shape classes (decode16/decode64/prefill)
x = randn((4096, H), 1)
w_gate = randn((F, H), 2, H**-0.5)
w_up = randn((F, H), 3, H**-0.5)
w_down = randn((H, F), 4, F**-0.5)
with torch.no_grad():
    full = swiglu_ffn(x, w_gate, w_up, w_down)  # default gemm, T=4096 -> prefill class
    torch.cuda.synchronize()
    print(f"full fwd done at {time.time()-t0:.1f}s")
    for T in (1, 16, 64, 65):
        for start in (0, 2048, 4096 - T):
            y = swiglu_ffn(x[start:start + T], w_gate, w_up, w_down)
            ok = torch.equal(y, full[start:start + T])
            nd = (y != full[start:start + T]).sum().item()
            print(f"G2 default gemm T={T:4d} rows[{start}:{start+T}] vs T=4096: equal={ok} mismatches={nd}")
    torch.cuda.synchronize()
    print(f"G2 probes done at {time.time()-t0:.1f}s")

# --- 3. dW chunk rule on K=T: default (fast) gemm vs adapter -----------------------------
for T in (64, 72, 256):
    for label, gm in (("fast_det_gemm", default_gemm()), ("adapter", TritonDetGemmAdapter())):
        leaves = [t[:T].detach().clone().requires_grad_() if t.shape[0] == 4096 else t.detach().clone().requires_grad_()
                  for t in (x, w_gate, w_up, w_down)]
        try:
            y = swiglu_ffn(*leaves, gemm=gm)
            y.backward(randn((T, H), 5, T**-0.5))
            print(f"backward T={T:3d} gemm={label}: OK dW_gate finite={torch.isfinite(leaves[1].grad).all().item()}")
        except Exception as e:  # noqa: BLE001
            print(f"backward T={T:3d} gemm={label}: {type(e).__name__}: {e}")
    # frozen weights: only dx
    leaves = [x[:T].detach().clone().requires_grad_(), w_gate, w_up, w_down]
    try:
        y = swiglu_ffn(*leaves)
        y.backward(randn((T, H), 5, T**-0.5))
        print(f"backward T={T:3d} frozen weights default gemm: OK")
    except Exception as e:  # noqa: BLE001
        print(f"backward T={T:3d} frozen weights default gemm: {type(e).__name__}: {e}")
torch.cuda.synchronize()
print(f"total {time.time()-t0:.1f}s, peak mem {torch.cuda.max_memory_allocated()/2**30:.2f} GiB")
