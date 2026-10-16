import sys
import time

import torch

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

from rl_engine.models.muse_glimmer.ops.gated_attention import (  # noqa: E402
    gated_attention_forward_fp32,
    strict_gated_attention_fwd_paged,
    strict_gated_attention_fwd_varlen,
)
from rl_engine.models.muse_glimmer.ops.paged_kv import PagedKVCache  # noqa: E402

dev = torch.device("cuda")
g = torch.Generator(device=dev).manual_seed(0)


def make(T):
    q = (torch.randn(T, 32, 128, device=dev, generator=g) * 3.87).to(torch.bfloat16)
    k = torch.randn(T, 2, 128, device=dev, generator=g).to(torch.bfloat16)
    v = torch.randn(T, 2, 128, device=dev, generator=g).to(torch.bfloat16)
    gate = (torch.randn(T, 4096, device=dev, generator=g) * 2).to(torch.bfloat16)
    return q, k, v, gate


def run(T, window, tile_mode):
    q, k, v, gate = make(T)
    cu = torch.tensor([0, T], dtype=torch.int32, device=dev)
    pos = torch.arange(T, dtype=torch.int32, device=dev)
    y, o, lse = strict_gated_attention_fwd_varlen(
        q, k, v, gate, cu, cu, pos, window=window, tile_mode=tile_mode
    )
    torch.cuda.synchronize()
    return (q, k, v, gate), (y, o, lse)


for window in (None, 2048):
    for T in (5, 64, 130, 2100):
        inputs, (y, o, lse) = run(T, window, "token")
        q, k, v, gate = inputs
        cu = torch.tensor([0, T], dtype=torch.int32, device=dev)
        pos = torch.arange(T, dtype=torch.int32, device=dev)
        y2, o2, lse2 = strict_gated_attention_fwd_varlen(
            q, k, v, gate, cu, cu, pos, window=window, tile_mode="head"
        )
        yg, og, lseg = gated_attention_forward_fp32(q, k, v, gate, pos, window)
        err_y = (y.float() - yg).abs().max().item()
        err_o = (o.float() - og).abs().max().item()
        err_l = (lse - lseg).abs().max().item()
        print(
            f"window={window} T={T}: token-vs-head equal y={torch.equal(y, y2)} "
            f"o={torch.equal(o, o2)} lse={torch.equal(lse, lse2)} | vs golden "
            f"max|dy|={err_y:.3e} max|do|={err_o:.3e} max|dlse|={err_l:.3e} "
            f"nan={torch.isnan(y).any().item()}"
        )
        if not torch.equal(o, o2):
            diff = (o.float() - o2.float()).abs()
            print("   mismatch rows:", (diff.max(dim=1).values > 0).nonzero().flatten()[:10].tolist(),
                  "max", diff.max().item())
            dl = (lse - lse2).abs()
            print("   lse mismatch count", (dl > 0).sum().item(), "max", dl.max().item())

# paged decode sanity
T = 200
q, k, v, gate = make(T)
cu = torch.tensor([0, T], dtype=torch.int32, device=dev)
pos = torch.arange(T, dtype=torch.int32, device=dev)
y, o, lse = strict_gated_attention_fwd_varlen(q, k, v, gate, cu, cu, pos, window=None)
cache = PagedKVCache(num_pages=64, num_seqs=1, max_pages_per_seq=32, device=dev)
cache.append(0, k[:150], v[:150])
cu1 = torch.tensor([0, 1], dtype=torch.int32, device=dev)
ok = True
for t in range(150, T):
    cache.append(0, k[t : t + 1], v[t : t + 1])
    yt, ot, lt = strict_gated_attention_fwd_paged(
        q[t : t + 1], gate[t : t + 1], cache.k_cache, cache.v_cache, cache.block_table,
        cache.seq_lens, cu1, window=None,
    )
    ok &= torch.equal(yt[0], y[t]) and torch.equal(ot[0], o[t]) and torch.equal(lt[0], lse[t])
print("paged decode == varlen prefill:", ok)

# dump layouts
from rl_engine.models.muse_glimmer.triton.gated_attention_fwd import _gated_attention_fwd_kernel  # noqa

for key, kern in _gated_attention_fwd_kernel.cache[0].items() if hasattr(_gated_attention_fwd_kernel, "cache") else []:
    pass
dc = _gated_attention_fwd_kernel.device_caches if hasattr(_gated_attention_fwd_kernel, "device_caches") else None
caches = dc[0][0] if dc else _gated_attention_fwd_kernel.cache[0]
for i, (key, kern) in enumerate(caches.items()):
    ttgir = kern.asm["ttgir"]
    mma = sorted({ln.strip() for ln in ttgir.splitlines() if "#mma" in ln and "=" in ln and ln.strip().startswith("#")})
    reds = [ln.strip()[:160] for ln in ttgir.splitlines() if "tt.reduce" in ln and "arith.addf" not in ln][:4]
    print(f"--- variant {i}: {kern.metadata.num_warps} warps; constants: ", {k: v for k, v in kern.metadata.__dict__.items() if k in ('num_warps','num_stages')} if hasattr(kern.metadata,'__dict__') else kern.metadata)
    for ln in mma:
        print("   ", ln[:200])
    print("    name:", kern.name)
    # Show where the reductions over axis 1 happen
    for ln in ttgir.splitlines():
        s = ln.strip()
        if s.startswith('"tt.reduce"') or "tt.reduce" in s and "axis = 1" in s:
            print("    ", s[:220])
    # print encodings used by reduce
    for ln in ttgir.splitlines():
        if "tt.reduce" in ln:
            print("    R:", ln.strip()[:200])
            break
