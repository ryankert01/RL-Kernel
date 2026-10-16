"""Probe: can the strict O7 path return logp > 0 when the target is a near-certain argmax?"""
import json

import torch

from rl_engine.models.muse_glimmer.ops import lm_head_softcap_logprob as lm
from rl_engine.models.muse_glimmer.runtime import fallback_ops as fb

torch.manual_seed(0)
dev = "cuda"
rows, tile = 2048, 25344
mult, cap, vocab = 0.19611613513818404, 20.0, 202048
targets = torch.randint(0, vocab, (rows,), device=dev)
stats = []
zs = []
for t in range(8):
    z = torch.randn(rows, tile, device=dev) * 3.0 - 60.0
    zs.append(z)
# put a dominant logit at each row's target (spread over the full zhat range [8, 20))
big = torch.rand(rows, device=dev) * 150.0 + 60.0
for r in range(rows):
    tgt = int(targets[r])
    zs[tgt // tile][r, tgt % tile] = big[r]
for t in range(8):
    stats.append(lm.softcap_tile_stats(zs[t], t, targets, multiplier=mult, softcap=cap, vocab_size=vocab))
_, log_z = lm.combine_tile_stats([(s[0], s[1]) for s in stats])
logp = fb.target_logit_from_stats(stats) - log_z
# Same arithmetic without FMA ambiguity: torch reference of the tile sums (fp32).
zhat_t = fb.target_logit_from_stats(stats)
m_all = torch.stack([s[0] for s in stats]).max(0).values
s_tile0 = torch.stack([s[1] for s in stats])
out = {
    "rows": rows,
    "positive": int((logp > 0).sum()),
    "zero": int((logp == 0).sum()),
    "max_logp": float(logp.max()),
    "target_is_max": int((zhat_t == m_all).sum()),
    "tiles_with_s_below_1": int((s_tile0 < 1).sum()),
    "min_tile_s_of_max_tile": float(
        torch.where(torch.stack([s[0] for s in stats]) == m_all, s_tile0, torch.full_like(s_tile0, 9.0)).min()
    ),
}
print(json.dumps(out))
