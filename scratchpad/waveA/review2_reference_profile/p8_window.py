"""Reviewer probe: batched decode across W=2048 with page eviction, teacher-forced so every
step's logp can be compared with the train forward (real weights, triton)."""
import pathlib
import time

import torch

from rl_engine.models.muse_glimmer.reference.strict_model import MuseGlimmerStrictModel, Session
from rl_engine.models.muse_glimmer.reference.weights import MuseGlimmerWeights
from rl_engine.validation.operators.accelerator import disable_tf32


def bits(t):
    return t.detach().float().contiguous().cpu().view(torch.int32)


dev = torch.device("cuda:0")
disable_tf32("cuda")
w = MuseGlimmerWeights.load_checkpoint(str(pathlib.Path("~/models/Muse-Glimmer-30B").expanduser()),
                                       device=dev, layout="nk")
model = MuseGlimmerStrictModel(w, "triton")
gen = torch.Generator().manual_seed(321)
lens4 = [2100, 2047, 2200, 2064]
steps = 5
pr = [torch.randint(0, 200000, (L,), generator=gen) for L in lens4]
comp = [torch.randint(0, 200000, (steps + 1,), generator=gen) for _ in lens4]
cache = model.new_batch_cache(max(lens4) + steps + 4, 4)
for b in range(4):
    model.prefill(Session(cache=cache, seq=b), pr[b])
ev0 = cache.evictions
got = [[] for _ in range(4)]
eq_b1 = 0
for j in range(steps):
    tk = torch.tensor([int(c[j]) for c in comp])
    tg = torch.tensor([int(c[j + 1]) for c in comp])
    f4 = model.decode_batch(cache, [0, 1, 2, 3], tk, tg, commit=False).cpu()
    s4 = torch.cat([model.decode_batch(cache, [b], tk[b:b + 1], tg[b:b + 1], commit=False).cpu()
                    for b in range(4)])
    eq_b1 += int((bits(f4) == bits(s4)).sum())
    perm = [2, 0, 3, 1]
    fc = model.decode_batch(cache, perm, tk[perm], tg[perm], commit=True).cpu()
    eq_b1 += int((bits(fc) == bits(f4[perm])).sum())
    for b in range(4):
        got[b].append(f4[b:b + 1])
res = {"evictions_prefill": ev0, "evictions_end": cache.evictions,
       "b1_vs_b4_and_perm_commit": f"{eq_b1}/{2 * 4 * steps}"}
for b in range(4):
    lp = model.logprobs(pr[b], comp[b], mode="train").cpu()  # lp[k] = logp(comp[k] | prefix)
    dec = torch.cat(got[b])  # step j: logp(comp[j+1] | pr + comp[:j+1])
    res[f"seq{b}_len{lens4[b]}_train_eq"] = f"{int((bits(lp[1:]) == bits(dec)).sum())}/{steps}"
print(res, flush=True)
