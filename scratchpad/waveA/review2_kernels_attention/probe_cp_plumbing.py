"""Reviewer probe: runtime/cp_attention.cp_gated_attention_backward at CP4/CP8 on ONE GPU.

The real CP plumbing (_compact_groups -> all_gather -> combine_tree over ranks -> _expand_groups
-> combine_kv_partials) is driven with a two-pass replay collective: pass 1 records every rank's
all_gather inputs, pass 2 returns the rank-ordered concatenation (what NCCL all_gather returns).
"""
from __future__ import annotations

import json
import sys

import torch

from rl_engine.models.muse_glimmer.ops import gated_attention as ga
from rl_engine.models.muse_glimmer.runtime import cp_attention as cpa
from rl_engine.models.muse_glimmer.triton import gated_attention_bwd as bwd

DEV = torch.device("cuda")


class Replay:
    def __init__(self, world, rank, store, record):
        self.world_size, self.rank, self.store, self.record = world, rank, store, record
        self.calls = 0

    def all_gather(self, t):
        key = self.calls
        self.calls += 1
        if self.record:
            self.store.setdefault(key, {})[self.rank] = t.clone()
            return torch.cat([t] * self.world_size, 0)
        return torch.cat([self.store[key][r] for r in range(self.world_size)], 0)


def bits(t):
    return t.contiguous().view({2: torch.int16, 4: torch.int32}[t.element_size()])


def mm(a, b):
    return int((bits(a) != bits(b)).sum())


def inputs(n, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.randn(n, 32, 128, generator=g)
    q = q * torch.rsqrt(q.pow(2).mean(-1, keepdim=True) + 1e-5) * 3.87
    k = torch.randn(n, 2, 128, generator=g)
    k = k * torch.rsqrt(k.pow(2).mean(-1, keepdim=True) + 1e-5)
    v = torch.randn(n, 2, 128, generator=g)
    gate = torch.randn(n, 4096, generator=g) * 2.0
    dy = torch.randn(n, 4096, generator=g)
    to = dict(device=DEV, dtype=torch.bfloat16)
    return {"q": q.to(**to), "k": k.to(**to), "v": v.to(**to), "gate": gate.to(**to), "dy": dy.to(**to)}


def run(n, cp, window):
    inp = inputs(n, 5000 + n)
    cu = torch.tensor([0, n], dtype=torch.int32, device=DEV)
    pos = torch.arange(n, dtype=torch.int32, device=DEV)
    _y, o, lse = ga.strict_gated_attention_fwd_varlen(inp["q"], inp["k"], inp["v"], inp["gate"], cu, cu, pos, window=window)
    one = bwd.strict_gated_attention_bwd(inp["q"], inp["k"], inp["v"], inp["gate"], o, lse, inp["dy"], cu, cu, pos, window=window)
    store: dict = {}
    outs = {}
    for record in (True, False):
        for r in range(cp):
            plan = cpa.CPAttentionPlan(cp, r)
            s, e = plan.local_token_range(n)
            coll = Replay(cp, r, store, record)
            got = cpa.cp_gated_attention_backward(
                inp["q"][s:e].contiguous(), inp["k"], inp["v"], inp["gate"][s:e].contiguous(),
                o[s:e].contiguous(), lse[s:e].contiguous(), inp["dy"][s:e].contiguous(),
                pos[s:e].contiguous(), plan, coll, window=window,
            )
            if not record:
                outs[r] = got
    res = {}
    for i, name in enumerate(("dq", "dk", "dv", "dgate")):
        cat = torch.cat([outs[r][i] for r in range(cp)], 0)
        res[name] = mm(cat, one[i])
    return res


if __name__ == "__main__":
    out = {}
    for window in (None, 2048):
        for n, cps in ((2048, (4, 8)), (2100, (4,)), (4096, (4, 8)), (8192, (8,))):
            for cp in cps:
                out[f"{'global' if window is None else 'local'}/T={n}/cp{cp}"] = run(n, cp, window)
    json.dump(out, open(sys.argv[1], "w"), indent=1)
    print(json.dumps(out, indent=1))
