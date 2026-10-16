"""Reviewer probes for kernels_attention (read-only; run with PYTHONPATH=<worktree>)."""
from __future__ import annotations

import json
import sys

import torch

from rl_engine.models.muse_glimmer.ops import gated_attention as ga
from rl_engine.models.muse_glimmer.ops import gated_attention_autograd as ag
from rl_engine.models.muse_glimmer.ops.canonical_tree import combine_tree
from rl_engine.models.muse_glimmer.triton import gated_attention_bwd as bwd

DEV = torch.device("cuda")
W = 2048
out: dict = {}


def bits(t):
    return t.contiguous().view({2: torch.int16, 4: torch.int32}[t.element_size()])


def mm(a, b):
    return int((bits(a) != bits(b)).sum())


def inputs(n, seed, hq=32, hkv=2):
    g = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.randn(n, hq, 128, generator=g)
    q = q * torch.rsqrt(q.pow(2).mean(-1, keepdim=True) + 1e-5) * 3.87
    k = torch.randn(n, hkv, 128, generator=g)
    k = k * torch.rsqrt(k.pow(2).mean(-1, keepdim=True) + 1e-5)
    v = torch.randn(n, hkv, 128, generator=g)
    gate = torch.randn(n, hq * 128, generator=g) * 2.0
    dy = torch.randn(n, hq * 128, generator=g)
    to = dict(device=DEV, dtype=torch.bfloat16)
    return {"q": q.to(**to), "k": k.to(**to), "v": v.to(**to), "gate": gate.to(**to), "dy": dy.to(**to)}


# ---------------------------------------------------------------- 1. host-sync profile
def probe_host_sync(window):
    lens = [300, 64, 2049]
    inp = {k: torch.cat([inputs(n, 50 + i)[k] for i, n in enumerate(lens)]) for k in ("q", "k", "v", "gate", "dy")}
    cu = torch.tensor([0, *torch.tensor(lens).cumsum(0).tolist()], dtype=torch.int32, device=DEV)
    pos = torch.cat([torch.arange(n, dtype=torch.int32, device=DEV) for n in lens])
    meta = bwd.prepare_sequence_meta(cu, cu, pos, num_k_tokens=inp["k"].shape[0])
    q = inp["q"].clone().requires_grad_(True)
    k = inp["k"].clone().requires_grad_(True)
    v = inp["v"].clone().requires_grad_(True)
    g = inp["gate"].clone().requires_grad_(True)
    # warm up (compile) outside the profiler
    ag.gated_attention(q, k, v, g, cu, cu, pos, window, meta).backward(inp["dy"])
    torch.cuda.synchronize()
    res = {}
    for label, m in (("with_meta", meta), ("without_meta", None)):
        for t in (q, k, v, g):
            t.grad = None
        from torch.profiler import ProfilerActivity, profile

        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            y = ag.gated_attention(q, k, v, g, cu, cu, pos, window, m)
            y.backward(inp["dy"])
        names = {}
        for e in prof.events():
            n = e.name
            if any(s in n for s in ("Synchronize", "cudaMemcpy", "Memcpy", "item", "_local_scalar_dense")):
                names[n] = names.get(n, 0) + 1
        res[label] = names
    return res


# ---------------------------------------------------------------- 2. TP2 x CP simulated
def probe_tp2_cp(n, window):
    inp = inputs(n, 777 + n)
    cu = torch.tensor([0, n], dtype=torch.int32, device=DEV)
    pos = torch.arange(n, dtype=torch.int32, device=DEV)
    _y, o, lse = ga.strict_gated_attention_fwd_varlen(inp["q"], inp["k"], inp["v"], inp["gate"], cu, cu, pos, window=window)
    dq, dk, dv, dg = bwd.strict_gated_attention_bwd(inp["q"], inp["k"], inp["v"], inp["gate"], o, lse, inp["dy"], cu, cu, pos, window=window)
    rec = {}
    for tp_rank in (0, 1):
        heads = slice(16 * tp_rank, 16 * tp_rank + 16)
        cols = slice(2048 * tp_rank, 2048 * tp_rank + 2048)
        kvh = tp_rank
        for cp in (2, 4, 8):
            if n % cp:
                continue
            span = n // cp
            rk, rv, dqs, dgs = [], [], [], []
            for r in range(cp):
                rows = slice(r * span, (r + 1) * span)
                dg_r, do_r, de_r = bwd.gated_attention_bwd_preprocess(
                    inp["dy"][rows, cols].contiguous(), o[rows, cols].contiguous(),
                    inp["gate"][rows, cols].contiguous(), num_q_heads=16)
                dq_r, dkp, dvp = bwd.strict_gated_attention_bwd_partials(
                    inp["q"][rows, heads].contiguous(), inp["k"][:, kvh:kvh + 1].contiguous(),
                    inp["v"][:, kvh:kvh + 1].contiguous(), do_r, lse[rows, heads].contiguous(), de_r,
                    torch.tensor([0, span], dtype=torch.int32, device=DEV), cu, pos[rows].contiguous(),
                    window=window, num_q_heads=16, num_kv_heads=1)
                rk.append(dkp)
                rv.append(dvp)
                dqs.append(dq_r)
                dgs.append(dg_r)
            gk = bwd.combine_kv_partials(combine_tree(rk))
            gv = bwd.combine_kv_partials(combine_tree(rv))
            rec[f"tp_rank{tp_rank}_cp{cp}"] = {
                "dq": mm(torch.cat(dqs), dq[:, heads]), "dgate": mm(torch.cat(dgs), dg[:, cols]),
                "dk": mm(gk[:, 0], dk[:, kvh]), "dv": mm(gv[:, 0], dv[:, kvh]),
            }
    return rec


# ---------------------------------------------------------------- 3. signed zeros in dk/dv
def probe_signed_zero(n, window):
    inp = inputs(n, 4242)
    cu = torch.tensor([0, n], dtype=torch.int32, device=DEV)
    pos = torch.arange(n, dtype=torch.int32, device=DEV)
    _y, o, lse = ga.strict_gated_attention_fwd_varlen(inp["q"], inp["k"], inp["v"], inp["gate"], cu, cu, pos, window=window)
    _dg, do, de = bwd.gated_attention_bwd_preprocess(inp["dy"], o, inp["gate"])
    _dq, lk, lv = bwd.strict_gated_attention_bwd_chunk_leaves(inp["q"], inp["k"], inp["v"], do, lse, de, cu, cu, pos, window=window)
    negz = lambda t: int(((t == 0) & torch.signbit(t)).sum())  # noqa: E731
    return {"leaf_dk_neg_zero": negz(lk), "leaf_dv_neg_zero": negz(lv),
            "leaf_numel": lk.numel()}


# ---------------------------------------------------------------- 4. TP shards at decode B=256
def probe_decode_b256(window):
    from rl_engine.models.muse_glimmer.ops.gated_attention import paged_query_positions  # noqa: F401
    bsz, page = 256, 16
    g = torch.Generator(device="cpu").manual_seed(31)
    ctx = torch.randint(1, 600, (bsz,), generator=g).tolist()
    pages_per = [-(-c // page) for c in ctx]
    max_pages = max(pages_per)
    total_pages = sum(pages_per)
    kc = torch.randn(total_pages, page, 2, 128, generator=g).to(DEV, torch.bfloat16)
    vc = torch.randn(total_pages, page, 2, 128, generator=g).to(DEV, torch.bfloat16)
    table = torch.full((bsz, max_pages), -1, dtype=torch.int32)
    base = 0
    for i, p in enumerate(pages_per):
        table[i, :p] = torch.arange(base, base + p, dtype=torch.int32)
        base += p
    table = table.to(DEV)
    seq_lens = torch.tensor(ctx, dtype=torch.int32, device=DEV)
    q = torch.randn(bsz, 32, 128, generator=g).to(DEV, torch.bfloat16)
    gate = torch.randn(bsz, 4096, generator=g).to(DEV, torch.bfloat16)
    cu_q = torch.arange(bsz + 1, dtype=torch.int32, device=DEV)
    full = ga.strict_gated_attention_fwd_paged(q, gate, kc, vc, table, seq_lens, cu_q, window=window)
    res = {}
    for tp in (2, 4, 8):
        hq = 32 // tp
        tot = 0
        for r in range(tp):
            kvh = (2 * r) // tp
            heads = slice(r * hq, (r + 1) * hq)
            cols = slice(r * hq * 128, (r + 1) * hq * 128)
            got = ga.strict_gated_attention_fwd_paged(
                q[:, heads].contiguous(), gate[:, cols].contiguous(),
                kc[:, :, kvh:kvh + 1].contiguous(), vc[:, :, kvh:kvh + 1].contiguous(),
                table, seq_lens, cu_q, window=window, num_q_heads=hq, num_kv_heads=1)
            tot += mm(got[0], full[0][:, cols]) + mm(got[1], full[1][:, cols]) + mm(got[2], full[2][:, heads])
        res[f"tp{tp}"] = tot
    return res


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = False
    for window in (None, W):
        key = "global" if window is None else "local"
        out[f"host_sync_{key}"] = probe_host_sync(window)
        out[f"tp2_cp_{key}"] = {n: probe_tp2_cp(n, window) for n in (2048, 2100, 4096)}
        out[f"signed_zero_{key}"] = probe_signed_zero(4096, window)
        out[f"decode_b256_tp_{key}"] = probe_decode_b256(window)
    json.dump(out, open(sys.argv[1], "w"), indent=1)
    print(json.dumps(out, indent=1))
