import time

import torch

from rl_engine.models.muse_glimmer.ablation import ablation_hf as A
from rl_engine.models.muse_glimmer.reference.chain_harness import hf_text_model_from_weights
from rl_engine.models.muse_glimmer.reference.weights import MuseGlimmerWeights, StrictConfig

cfg = StrictConfig.tiny()
w = MuseGlimmerWeights.random(cfg, device="cpu", seed=1, layout="nk")
m, lm = hf_text_model_from_weights(w, "cpu")
print(type(m), lm.shape, m.rotary_emb.inv_freq.dtype, m.config._attn_implementation)
m = m.float()
lm = lm.float()
print(m.rotary_emb.inv_freq.dtype)
g = torch.Generator().manual_seed(0)
toks = torch.randint(0, 1000, (150,), generator=g).tolist()
s = A.SeqSpec("a", tuple(toks), 120)
t = time.time()
ha = A.hidden_alone(m, s)
hd = A.hidden_decode(m, s)
hc = A.hidden_chunked(m, s, 64)
s2 = A.SeqSpec("b", tuple(toks[:90]), 60)
hp = A.hidden_padded(m, [s, s2], "left")
hr = A.hidden_padded(m, [s, s2], "right")
h2 = A.hidden_alone(m, s2)
print((ha - hd).abs().max(), (ha - hc).abs().max(), (ha - hp[0]).abs().max(),
      (h2 - hp[1]).abs().max(), (h2 - hr[1]).abs().max(), time.time() - t)
