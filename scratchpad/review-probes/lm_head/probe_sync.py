# Does the strict forward force a device->host sync?  Count cudaStreamSynchronize-class events via the profiler.
import torch
from rl_engine.models.muse_glimmer.ops import lm_head_softcap_logprob as op
dev="cuda"
g = torch.Generator(device=dev).manual_seed(1)
H, V, VP, T = 64, 1900, 2048, 32
w = op.pad_lm_head_weight((torch.randn(V, H, device=dev, generator=g)).to(torch.bfloat16), padded_vocab=VP)
h = torch.randn(T, H, device=dev, generator=g).to(torch.bfloat16)
tg = torch.randint(0, V, (T,), device=dev, generator=g)
with torch.no_grad():
    op.lm_head_softcap_logprob(h, w, tg, vocab_size=V)  # warm/compile
torch.cuda.synchronize()
from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    with torch.no_grad():
        op.lm_head_softcap_logprob(h, w, tg, vocab_size=V)
    torch.cuda.synchronize()
names = [e.name for e in prof.events()]
syncs = [n for n in names if "Synchronize" in n or "aten::_local_scalar_dense" in n or "Memcpy DtoH" in n]
print("sync-class events in one strict forward:", syncs)
# out-of-range target without host validation: what does the kernel produce?
from rl_engine.models.muse_glimmer.triton.softcap_logprob import launch_tile_stats
bad = tg.clone(); bad[0] = -100
z = op.legacy_det_gemm_fp32(h, w[:256])
m, s_, zt, it = launch_tile_stats(z, bad, tile_size=256, real_cols=256, col_begin=0, multiplier=op.OUTPUT_MULTIPLIER, softcap=op.FINAL_LOGIT_SOFTCAP)
print("kernel with target=-100 (no host check): target_in_tile=", it[0].item(), "target_logit=", zt[0].item(), "-> logp would be -logZ, no error")
