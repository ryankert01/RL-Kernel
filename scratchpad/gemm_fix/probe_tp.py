"""det_gemm:2 probe: column-parallel dA at simulated TP2/4/8 vs TP1 (mismatch counts).

naive    = each rank runs the plain autograd fast_det_gemm on its N shard (8 chunks over N_local,
           bf16 dA), rank dAs summed (fp32 sum in rank order, then bf16) -- what a Megatron
           ColumnParallelLinear + all-reduce would do with the strict GEMM.
canonical= column_parallel_fast_det_gemm with fp32 sub-tree partials combined in the canonical
           tree (tp_canonical_tree.combine_tree) before one bf16 rounding.
Also row-parallel forward/backward through row_parallel_fast_det_gemm.
"""

import json
import sys

import torch

from rl_engine.models.muse_glimmer.ops import det_gemm as dg
from rl_engine.models.muse_glimmer.ops.canonical_tree import rank_chunk_ids

DEV, BF = "cuda", torch.bfloat16


class Gather:
    def __init__(self, partials, rank):
        self.p, self.r = partials, rank

    def all_gather(self, x):
        return torch.stack([x[0] if i == self.r else q for i, q in enumerate(self.p)])


def rnd(*s, scale=1.0):
    return (torch.randn(*s, device=DEV) * scale).to(BF)


out = {"column_parallel": [], "row_parallel": []}
torch.manual_seed(0)
M = 256
for name, n, k in (("q_proj", 4096, 6656), ("attn_gate", 4096, 6656), ("k_proj", 256, 6656),
                   ("mlp_gate", 19968, 6656), ("up_proj", 19968, 6656)):
    a, w, gy = rnd(M, k), rnd(n, k, scale=0.02), rnd(M, n)
    a1 = a.clone().requires_grad_(True)
    dg.fast_det_gemm(a1, w).backward(gy)
    for tp in (2, 4, 8):
        nl = n // tp
        rows = [slice(r * nl, (r + 1) * nl) for r in range(tp)]
        naive = torch.zeros(M, k, device=DEV)
        for s in rows:
            ar = a.clone().requires_grad_(True)
            dg.fast_det_gemm(ar, w[s]).backward(gy[:, s])
            naive += ar.grad.float()
        naive = naive.to(BF)
        parts = [dg.column_parallel_da_partial(gy[:, s], w[s], tp) for s in rows]
        mism = []
        for r, s in enumerate(rows):
            ar = a.clone().requires_grad_(True)
            dg.column_parallel_fast_det_gemm(
                ar, w[s], tp_size=tp, da_reduce=dg.tree_reduce_over_ranks(Gather(parts, r))
            ).backward(gy[:, s])
            mism.append(int((ar.grad != a1.grad).sum()))
        rec = {"shape": name, "N": n, "K": k, "M": M, "tp": tp, "numel": a1.grad.numel(),
               "naive_mismatch": int((naive != a1.grad).sum()),
               "naive_max_abs_diff": float((naive.float() - a1.grad.float()).abs().max()),
               "canonical_mismatch_per_rank": mism}
        out["column_parallel"].append(rec)
        print(rec, flush=True)
for name, n, k in (("o_proj", 6656, 4096), ("down_proj", 6656, 19968)):
    a, w, gy = rnd(M, k), rnd(n, k, scale=0.02), rnd(M, n)
    a1, w1 = a.clone().requires_grad_(True), w.clone().requires_grad_(True)
    y1 = dg.fast_det_gemm(a1, w1)
    y1.backward(gy)
    for tp in (2, 4, 8):
        kl = k // tp
        cols = [slice(r * kl, (r + 1) * kl) for r in range(tp)]
        parts = [dg.fast_det_gemm_partial(a[:, s], w[:, s], rank_chunk_ids(tp, r))
                 for r, s in enumerate(cols)]
        nccl_like = sum(p for p in parts).to(BF)  # left fold (ring-like order), control
        fwd, da, dw = [], [], []
        for r, s in enumerate(cols):
            ar, wr = a[:, s].clone().requires_grad_(True), w[:, s].clone().requires_grad_(True)
            yr = dg.row_parallel_fast_det_gemm(
                ar, wr, tp_size=tp, y_reduce=dg.tree_reduce_over_ranks(Gather(parts, r)))
            yr.backward(gy)
            fwd.append(int((yr != y1).sum()))
            da.append(int((ar.grad != a1.grad[:, s]).sum()))
            dw.append(int((wr.grad != w1.grad[:, s]).sum()))
        rec = {"shape": name, "N": n, "K": k, "M": M, "tp": tp, "numel_y": y1.numel(),
               "left_fold_mismatch": int((nccl_like != y1).sum()),
               "canonical_fwd_mismatch_per_rank": fwd, "canonical_dA_mismatch_per_rank": da,
               "canonical_dW_mismatch_per_rank": dw}
        out["row_parallel"].append(rec)
        print(rec, flush=True)
with open(sys.argv[1], "w") as f:
    json.dump(out, f, indent=1)
