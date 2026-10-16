"""Count DtoH memcpys and stream syncs in one GatedAttentionFn forward+backward without seq_meta
(owner claim: '2 D2H per layer'), and with a prebuilt meta."""
import torch
from torch.profiler import ProfilerActivity, profile

from rl_engine.models.muse_glimmer.ops.gated_attention_autograd import gated_attention
from rl_engine.models.muse_glimmer.triton import gated_attention_bwd as bwd

DEV = torch.device("cuda")
lens = [300, 64, 2049]
T = sum(lens)
g = torch.Generator(device=DEV).manual_seed(0)
q = torch.randn(T, 32, 128, device=DEV, generator=g).bfloat16().requires_grad_(True)
k = torch.randn(T, 2, 128, device=DEV, generator=g).bfloat16().requires_grad_(True)
v = torch.randn(T, 2, 128, device=DEV, generator=g).bfloat16().requires_grad_(True)
gate = torch.randn(T, 32 * 128, device=DEV, generator=g).bfloat16().requires_grad_(True)
cu = torch.tensor([0, 300, 364, T], dtype=torch.int32, device=DEV)
pos = torch.cat([torch.arange(n) for n in lens]).int().to(DEV)
dy = torch.randn(T, 32 * 128, device=DEV, generator=g).bfloat16()


def step(meta):
    y = gated_attention(q, k, v, gate, cu, cu, pos, 2048, meta)
    y.backward(dy)


for label, meta_fn in (("no_meta", lambda: None),
                       ("meta", lambda: bwd.prepare_sequence_meta(cu, cu, pos, num_k_tokens=T))):
    meta = meta_fn()
    step(meta)  # warm
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        step(meta)
        torch.cuda.synchronize()
    names = [e.name for e in prof.events()]
    d2h = sum(1 for n in names if "DtoH" in n or "Memcpy DtoH" in n)
    h2d = sum(1 for n in names if "HtoD" in n)
    ss = sum(1 for n in names if n == "cudaStreamSynchronize")
    items = sum(1 for n in names if n == "aten::item" or n == "aten::_local_scalar_dense")
    print(f"{label}: DtoH_events={d2h} HtoD_events={h2d} cudaStreamSynchronize={ss} item_calls={items}")
