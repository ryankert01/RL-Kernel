"""Review probe 3: PTX lowering of exp/log/div in every compiled variant + oracle sharpness
restricted to rows whose window is actually clipped (pos >= W)."""
import torch
from rl_engine.models.muse_glimmer.ops import gated_attention as ga
from rl_engine.models.muse_glimmer.ops.paged_kv import PagedKVCache
from rl_engine.models.muse_glimmer.triton import gated_attention_fwd as kern

DEV = torch.device("cuda")
HQ, HKV, D = 32, 2, 128


def inputs(n, seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    q = (torch.randn(n, HQ, D, device=DEV, generator=g) * 3.87).to(torch.bfloat16)
    k = torch.randn(n, HKV, D, device=DEV, generator=g).to(torch.bfloat16)
    v = torch.randn(n, HKV, D, device=DEV, generator=g).to(torch.bfloat16)
    gate = (torch.randn(n, HQ * D, device=DEV, generator=g) * 2.0).to(torch.bfloat16)
    return q, k, v, gate


def dense_lse(q, k, pos, window):
    qf, kf = q.double(), k.double()
    hq = q.shape[1]
    kv = torch.arange(hq, device=DEV) // (hq // k.shape[1])
    s = torch.einsum("qhd,khd->hqk", qf, kf[:, kv, :]) * (128 ** -0.5)
    kpos = torch.arange(k.shape[0], device=DEV)
    keep = kpos[None, :] <= pos[:, None]
    if window is not None:
        keep = keep & (kpos[None, :] > pos[:, None] - window)
    s = s.masked_fill(~keep[None], float("-inf"))
    m = s.amax(-1, keepdim=True)
    return (m.squeeze(-1) + torch.log(torch.exp(s - m).sum(-1))).transpose(0, 1)


def one_shot(q, k, v, gate, window, tile_mode=None):
    n = q.shape[0]
    cu = torch.tensor([0, n], dtype=torch.int32, device=DEV)
    pos = torch.arange(n, dtype=torch.int32, device=DEV)
    return ga.strict_gated_attention_fwd_varlen(q, k, v, gate, cu, cu, pos, window=window, tile_mode=tile_mode)


print("== oracle sharpness on clipped rows only (pos >= W)")
for n, window in [(100, 20), (300, 64), (2300, 2048)]:
    q, k, v, gate = inputs(n, 7 + n)
    pos = torch.arange(n, device=DEV)
    rows = torch.arange(window, n, device=DEV)
    lr = dense_lse(q[rows], k, pos[rows], window)
    lm1 = dense_lse(q[rows], k, pos[rows], window - 1)
    lp1 = dense_lse(q[rows], k, pos[rows], window + 1)
    y, o, lse = one_shot(q, k, v, gate, window)
    print(f"n={n} W={window}: kernel max|dlse|={(lse[rows].double()-lr).abs().max().item():.2e} | median over rows of max_h |lse(W)-lse(W-1)| = {(lr-lm1).abs().amax(1).median().item():.2e}, min = {(lr-lm1).abs().amax(1).min().item():.2e}; W+1: median {(lr-lp1).abs().amax(1).median().item():.2e}")

# compile paged variants too
cache = PagedKVCache(num_pages=40, num_seqs=1, max_pages_per_seq=20, device=DEV)
cache.append(0, k[:300], v[:300])
cu = torch.tensor([0, 1], dtype=torch.int32, device=DEV)
for w in (None, 2048):
    ga.strict_gated_attention_fwd_paged(q[-1:].contiguous(), gate[-1:].contiguous(), cache.k_cache, cache.v_cache, cache.block_table, cache.seq_lens, cu, window=w)
    ga.strict_gated_attention_fwd_paged(q[-1:].contiguous(), gate[-1:].contiguous(), cache.k_cache, cache.v_cache, cache.block_table, cache.seq_lens, cu, window=w, tile_mode="token")

print("\n== compiled variants: PTX instruction census")
caches = kern._gated_attention_fwd_kernel.device_caches[torch.cuda.current_device()][0]
for key, compiled in caches.items():
    ptx = compiled.asm["ptx"]
    ttgir = compiled.asm["ttgir"]
    splits = sorted({ln.split("warpsPerCTA = ")[1].split("]")[0] + "]" for ln in ttgir.splitlines() if ln.strip().startswith("#mma")})
    md = compiled.metadata
    consts = getattr(md, "constants", None) or {}
    tag = {k_: consts[k_] for k_ in consts if k_ in ("BLOCK_M", "LOCAL", "PAGED", "HEAD_TILE")}
    print(f"  {tag} num_warps={getattr(md,'num_warps',None)} num_stages={getattr(md,'num_stages',None)}: ex2.approx={ptx.count('ex2.approx')} lg2.approx={ptx.count('lg2.approx')} div.rn.f32={ptx.count('div.rn.f32')} div.full={ptx.count('div.full')} div.approx={ptx.count('div.approx')} rcp.approx={ptx.count('rcp.approx')} mma.sync.m16n8k16={ptx.count('mma.sync.aligned.m16n8k16')} shfl.sync={ptx.count('shfl.sync')} warpsPerCTA={splits}")
