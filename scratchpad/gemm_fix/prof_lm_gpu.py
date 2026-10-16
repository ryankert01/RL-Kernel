import torch
from torch.profiler import ProfilerActivity, profile

from rl_engine.models.muse_glimmer.ops import lm_head_softcap_logprob as lm

DEV, BF = "cuda", torch.bfloat16
g = torch.Generator(device=DEV).manual_seed(0)
wl = lm.pad_lm_head_weight((torch.randn(lm.VOCAB_SIZE, 6656, device=DEV, generator=g) * 0.02).to(BF))
h = torch.randn(1, 6656, device=DEV, generator=g).to(BF)
tg = torch.randint(0, lm.VOCAB_SIZE, (1,), device=DEV, generator=g)
with torch.no_grad():
    for _ in range(20):
        lm.lm_head_softcap_logprob(h, wl, tg)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(20):
            lm.lm_head_softcap_logprob(h, wl, tg)
        torch.cuda.synchronize()
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=12, max_name_column_width=60))
