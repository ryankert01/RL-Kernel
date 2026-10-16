"""Independent dense fp64 oracle for the strict gated-attention forward (review probe).

Checks that the author's blocked golden does not just mirror the kernel: compare the kernel's
fp32 lse (unaffected by the p->bf16 cast) against a dense fp64 softmax with an explicit mask,
for global and local layers (small window so an off-by-one key is visible), both tile modes.
Also: (a) sharpness of the oracle (lse with W-1 / W+1), (b) silent masking of a -1 page inside
the window, (c) PTX inspection (exp lowering, div.rn), (d) rows beyond cu_seqlens_q[-1].
"""
import torch
from rl_engine.models.muse_glimmer.ops import gated_attention as ga
from rl_engine.models.muse_glimmer.ops.paged_kv import PagedKVCache
from rl_engine.models.muse_glimmer.triton import gated_attention_fwd as kern

torch.backends.cuda.matmul.allow_tf32 = False
DEV = torch.device("cuda")
HQ, HKV, D = 32, 2, 128


def inputs(n, seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    q = (torch.randn(n, HQ, D, device=DEV, generator=g) * 3.87).to(torch.bfloat16)
    k = torch.randn(n, HKV, D, device=DEV, generator=g).to(torch.bfloat16)
    v = torch.randn(n, HKV, D, device=DEV, generator=g).to(torch.bfloat16)
    gate = (torch.randn(n, HQ * D, device=DEV, generator=g) * 2.0).to(torch.bfloat16)
    return q, k, v, gate


def dense_ref(q, k, v, gate, pos, window):
    qf, kf, vf = q.double(), k.double(), v.double()
    tq, hq, d = q.shape
    tk = k.shape[0]
    kv = torch.arange(hq, device=DEV) // (hq // k.shape[1])
    kh, vh = kf[:, kv, :], vf[:, kv, :]
    s = torch.einsum("qhd,khd->hqk", qf, kh) * (d ** -0.5)
    kpos = torch.arange(tk, device=DEV)
    keep = kpos[None, :] <= pos[:, None]
    if window is not None:
        keep = keep & (kpos[None, :] > pos[:, None] - window)
    s = s.masked_fill(~keep[None], float("-inf"))
    m = s.amax(-1, keepdim=True)
    p = torch.exp(s - m)
    l = p.sum(-1)
    lse = (m.squeeze(-1) + torch.log(l)).transpose(0, 1).contiguous()
    o = torch.einsum("hqk,khd->qhd", p / l[..., None], vh)
    y = o.to(torch.bfloat16).double() * torch.sigmoid(gate.double().view(tq, hq, d))
    return y.reshape(tq, -1), o.reshape(tq, -1), lse


def one_shot(q, k, v, gate, window, tile_mode=None):
    n = q.shape[0]
    cu = torch.tensor([0, n], dtype=torch.int32, device=DEV)
    pos = torch.arange(n, dtype=torch.int32, device=DEV)
    return ga.strict_gated_attention_fwd_varlen(q, k, v, gate, cu, cu, pos, window=window, tile_mode=tile_mode)


print("== A. kernel vs independent dense fp64 oracle (lse is sharp: p is not bf16-rounded in l)")
for n, window in [(100, None), (100, 20), (300, 64), (2300, 2048), (2300, None)]:
    q, k, v, gate = inputs(n, 7 + n)
    pos = torch.arange(n, dtype=torch.int32, device=DEV)
    rows = torch.arange(n, device=DEV) if n <= 300 else torch.arange(n - 260, n, device=DEV)
    yr, orf, lr = dense_ref(q[rows], k, v, gate[rows], pos[rows].long(), window)
    for tm in ("token", "head"):
        y, o, lse = one_shot(q, k, v, gate, window, tile_mode=tm)
        dl = (lse[rows].double() - lr).abs().max().item()
        do = (o[rows].double() - orf).abs().max().item()
        dy = (y[rows].double() - yr).abs().max().item()
        line = f"n={n:5d} W={window} tile={tm:5s}: max|dlse|={dl:.3e} max|do|={do:.3e} max|dy|={dy:.3e}"
        if window is not None and window < n:
            _, _, lm1 = dense_ref(q[rows], k, v, gate[rows], pos[rows].long(), window - 1)
            _, _, lp1 = dense_ref(q[rows], k, v, gate[rows], pos[rows].long(), window + 1)
            line += f"  | oracle sharpness: min|lse(W)-lse(W-1)|={(lr - lm1).abs().amax(1).min().item():.2e} min|lse(W)-lse(W+1)|={(lr - lp1).abs().amax(1).min().item():.2e}"
        print(line)

print("\n== B. a block-table entry of -1 INSIDE the window is silently masked (no error)")
n = 200
q, k, v, gate = inputs(n, 99)
cache = PagedKVCache(num_pages=16, num_seqs=1, max_pages_per_seq=13, device=DEV)
cache.append(0, k, v)
cu = torch.tensor([0, 1], dtype=torch.int32, device=DEV)
ref = ga.strict_gated_attention_fwd_paged(q[-1:], gate[-1:], cache.k_cache, cache.v_cache, cache.block_table, cache.seq_lens, cu, window=None)
bt = cache.block_table.clone()
bt[0, 5] = -1  # logical page 5 = keys 80..95, well inside the causal/global window of query 199
got = ga.strict_gated_attention_fwd_paged(q[-1:], gate[-1:], cache.k_cache, cache.v_cache, bt, cache.seq_lens, cu, window=None)
print("raised:", False, "| bits equal to intact table:", torch.equal(got[0], ref[0]), "| max|dlse|:", (got[2] - ref[2]).abs().max().item(), "| any NaN:", bool(torch.isnan(got[0]).any()))

print("\n== C. seq_lens beyond block_table width (max_pages) is not validated")
bt2 = cache.block_table[:, :6].contiguous()  # only 6 pages = 96 keys, but seq_lens says 200
try:
    got2 = ga.strict_gated_attention_fwd_paged(q[-1:], gate[-1:], cache.k_cache, cache.v_cache, bt2, cache.seq_lens, cu, window=None)
    torch.cuda.synchronize()
    print("no error raised; y finite:", bool(torch.isfinite(got2[0]).all()), "| equal to intact:", torch.equal(got2[0], ref[0]))
except Exception as e:  # noqa: BLE001
    print("raised:", type(e).__name__, str(e)[:120])

print("\n== D. rows beyond cu_seqlens_q[-1] are left uninitialized (torch.empty)")
q, k, v, gate = inputs(70, 5)
cu_q = torch.tensor([0, 64], dtype=torch.int32, device=DEV)
cu_k = torch.tensor([0, 70], dtype=torch.int32, device=DEV)
pos = torch.arange(70, dtype=torch.int32, device=DEV)
y, o, lse = ga.strict_gated_attention_fwd_varlen(q, k, v, gate, cu_q, cu_k, pos, window=None)
print("rows 64..69 of y written by kernel? (all zero / finite):", bool((y[64:] == 0).all()), bool(torch.isfinite(y[64:]).all()), "| lse[64:] finite:", bool(torch.isfinite(lse[64:]).all()))

print("\n== E. compiled variants: exp/log/div lowering and MMA warp split")
caches = kern._gated_attention_fwd_kernel.device_caches[torch.cuda.current_device()][0]
for key, compiled in caches.items():
    ptx = compiled.asm["ptx"]
    ttgir = compiled.asm["ttgir"]
    splits = sorted({ln.split("warpsPerCTA = ")[1].split("]")[0] + "]" for ln in ttgir.splitlines() if ln.strip().startswith("#mma")})
    consts = {k_: v_ for k_, v_ in compiled.metadata.__dict__.items() if k_ in ("num_warps", "num_stages")} if hasattr(compiled, "metadata") else {}
    print(f"  variant {compiled.name if hasattr(compiled,'name') else ''}: ex2.approx={ptx.count('ex2.approx')} lg2.approx={ptx.count('lg2.approx')} div.rn.f32={ptx.count('div.rn.f32')} div.full={ptx.count('div.full')} div.approx={ptx.count('div.approx')} mma.sync={ptx.count('mma.sync')} fma={ptx.count('fma.rn.f32')>0} warpsPerCTA={splits} {consts}")
    try:
        sig = compiled.metadata.constants if hasattr(compiled.metadata, "constants") else None
        if sig:
            print("     constants:", {k_: v_ for k_, v_ in sig.items() if k_ in ("BLOCK_M", "LOCAL", "PAGED", "HEAD_TILE", "window", "max_pages")})
    except Exception:  # noqa: BLE001
        pass
