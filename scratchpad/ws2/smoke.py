import json
import time

import torch

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

from rl_engine.models.muse_glimmer.runtime import kv_replication as kvr
from rl_engine.models.muse_glimmer.runtime import tp_canonical_tree as tct
from rl_engine.models.muse_glimmer.runtime import vocab_parallel_softcap_logp as vp
from rl_engine.models.muse_glimmer.runtime.tp_layout import TPLayout, TPLayoutError, official_layout

dev = torch.device("cuda")
g = torch.Generator(device="cpu").manual_seed(2026)

print("== layout")
for tp in (1, 2, 4, 8):
    lay = official_layout(tp)
    print(tp, [(r["rank"], r["q_heads"], r["kv_heads"], r["canonical_chunks"]) for r in lay.kv_ownership_map()["ranks"]])
try:
    TPLayout(tp=3)
except TPLayoutError as e:
    print("tp=3 rejected:", e)
try:
    TPLayout(tp=2).fail_closed_if_undeclared({"canonical_tp": 4})
except TPLayoutError as e:
    print("undeclared rejected:", str(e)[:80])
print(json.dumps(tct.trace(official_layout(4)), indent=None)[:300])

print("== row-parallel tree")
for name, (M, K, N) in {"o_proj": (256, 4096, 6656), "down": (256, 19968, 6656)}.items():
    a = torch.randn(M, K, generator=g).to(dev, torch.bfloat16)
    w = torch.randn(N, K, generator=g).to(dev, torch.bfloat16)
    t0 = time.perf_counter()
    ref = tct.row_parallel_tree_gemm_tp1(a, w)
    torch.cuda.synchronize()
    t_ref = (time.perf_counter() - t0) * 1e3
    rep = tct.row_parallel_tree_gemm_tp1(a, w)
    print(f"{name}: tp1 {t_ref:.1f} ms repeat_equal={torch.equal(ref, rep)}")
    for tp in (2, 4, 8):
        lay = official_layout(tp)
        roots = []
        for r in range(tp):
            k0, k1 = lay.k_range(r, K // 8)
            roots.append(tct.subtree_partial_fp32(a[:, k0:k1], w[:, k0:k1], lay.canonical_chunks(r)))
        sim = tct.combine_tree(torch.stack(roots)).to(torch.bfloat16)
        leftfold = roots[0].clone()
        for x in roots[1:]:
            leftfold = leftfold + x
        print(f"  simulated tp{tp}: equal={torch.equal(sim, ref)} leftfold_equal={torch.equal(leftfold.to(torch.bfloat16), ref)} "
              f"leftfold_mismatch={(leftfold.to(torch.bfloat16) != ref).sum().item()}")
    # cuBLAS fp32 reference closeness
    cub = (a.float() @ w.float().t())
    print(f"  max|tree_bf16 - cublas_fp32| = {(ref.float() - cub).abs().max().item():.4f} (bf16 rounding of |y|~{cub.abs().max().item():.0f})")

print("== vocab")
T, H = 64, 6656
lay1 = official_layout(1)
h = torch.randn(T, H, generator=g).to(dev, torch.bfloat16)
gc = torch.Generator(device="cuda").manual_seed(7)
w_lm = torch.randn(lay1.vocab_size, H, device=dev, dtype=torch.bfloat16, generator=gc)
w_pad = vp.pad_lm_head_weight(w_lm)
print("padded", tuple(w_pad.shape), "pad rows zero:", bool((w_pad[lay1.vocab_size:] == 0).all()))
targets = torch.randint(0, lay1.vocab_size, (T,), generator=g).to(dev)
special = [0, 25343, 25344, 50687, 101375, 177407, 177408, 202047]
targets[: len(special)] = torch.tensor(special, device=dev)
t0 = time.perf_counter()
logp1 = vp.vocab_parallel_softcap_logp(h, w_pad, targets, lay1)
torch.cuda.synchronize()
print(f"tp1 logp {(time.perf_counter()-t0)*1e3:.1f} ms; repeat_equal={torch.equal(logp1, vp.vocab_parallel_softcap_logp(h, w_pad, targets, lay1))}")
tiles = [vp.tile_logits_fp32(h, w_pad[t * 25344:(t + 1) * 25344]) for t in range(8)]
ref64 = vp.vocab_softcap_logp_reference_fp64(tiles, targets, vocab_size=lay1.vocab_size)
print(f"max|logp32 - ref64| = {(logp1.double() - ref64).abs().max().item():.3e}; logp range [{logp1.min().item():.3f}, {logp1.max().item():.3f}]")
for tp in (2, 4, 8):
    lay = official_layout(tp)
    stats = [[], [], [], []]
    for r in range(tp):
        v0, v1 = lay.vocab_range(r)
        loc = []
        for j, t in enumerate(lay.vocab_tiles(r)):
            z = vp.tile_logits_fp32(h, w_pad[t * 25344:(t + 1) * 25344])
            loc.append(vp.softcap_tile_stats(z, t, targets, multiplier=vp.OUTPUT_MULTIPLIER, softcap=vp.FINAL_LOGIT_SOFTCAP, vocab_size=lay.vocab_size))
        for i in range(4):
            stats[i].append(torch.stack([s[i] for s in loc]))
    m_all, s_all, tl_all, own_all = (torch.cat(x) for x in stats)
    _, logz = vp.combine_tile_stats((m_all, s_all))
    owner = targets // 25344
    rows = torch.arange(T, device=dev)
    sim = tl_all[owner, rows] - logz
    print(f"  simulated tp{tp}: equal={torch.equal(sim, logp1)}")
for bad in (lay1.vocab_size, lay1.padded_vocab_size - 1, -1):
    tt = targets.clone(); tt[3] = bad
    try:
        vp.vocab_parallel_softcap_logp(h, w_pad, tt, lay1); print("  NO RAISE for", bad)
    except ValueError as e:
        print(f"  target {bad} raised: {str(e)[:60]}")

print("== kv replication")
for name, (lens, window) in {"global_2seq": ([160, 352], None), "local_window256": ([640], 256)}.items():
    Ttot = sum(lens)
    q = torch.randn(Ttot, 32, 128, generator=g).to(dev, torch.bfloat16)
    k = torch.randn(Ttot, 2, 128, generator=g).to(dev, torch.bfloat16)
    v = torch.randn(Ttot, 2, 128, generator=g).to(dev, torch.bfloat16)
    gate = torch.randn(Ttot, 4096, generator=g).to(dev, torch.bfloat16)
    cu = torch.tensor([0] + list(torch.tensor(lens).cumsum(0).tolist()), device=dev)
    pos = torch.cat([torch.arange(n, device=dev) for n in lens])
    t0 = time.perf_counter()
    rep = kvr.prove_kv_replication(q, k, v, gate, cu, pos, window=window)
    torch.cuda.synchronize()
    print(f"{name}: {(time.perf_counter()-t0):.1f} s", {tp: (r["equal"], r["concat_equal"]) for tp, r in rep["per_tp"].items()})
    # sanity vs SDPA-style fp32 reference (tolerance only)
    y = kvr.gated_attention_fwd(q, k, v, gate, cu, pos, window=window)
    outs = []
    for s in range(len(lens)):
        a0, a1 = int(cu[s]), int(cu[s + 1])
        qq = q[a0:a1].float().permute(1, 0, 2); kk = k[a0:a1].float().repeat_interleave(16, dim=1).permute(1, 0, 2); vv = v[a0:a1].float().repeat_interleave(16, dim=1).permute(1, 0, 2)
        sc = (qq @ kk.transpose(1, 2)) * 128 ** -0.5
        qp = pos[a0:a1]
        vis = qp[None, :, None] >= qp[None, None, :]
        if window is not None:
            vis = vis & (qp[None, :, None] - qp[None, None, :] < window)
        sc = sc.masked_fill(~vis, float("-inf"))
        o = torch.softmax(sc, -1) @ vv
        outs.append(o.permute(1, 0, 2).reshape(a1 - a0, 4096))
    ref = torch.cat(outs) * torch.sigmoid(gate.float())
    print(f"  max|y - sdpa_ref| = {(y.float() - ref).abs().max().item():.4f}, max|ref| = {ref.abs().max().item():.3f}")
