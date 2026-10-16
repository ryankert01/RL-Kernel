import torch

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

import rl_engine.models.muse_glimmer.ops.gated_attention as ga  # noqa: E402
from rl_engine.models.muse_glimmer.triton.gated_attention_fwd import (  # noqa: E402
    _gated_attention_fwd_kernel,
)

dev = torch.device("cuda")
g = torch.Generator(device=dev).manual_seed(0)
T = 2100
q = (torch.randn(T, 32, 128, device=dev, generator=g) * 3.87).to(torch.bfloat16)
k = torch.randn(T, 2, 128, device=dev, generator=g).to(torch.bfloat16)
v = torch.randn(T, 2, 128, device=dev, generator=g).to(torch.bfloat16)
gate = (torch.randn(T, 4096, device=dev, generator=g) * 2).to(torch.bfloat16)
cu = torch.tensor([0, T], dtype=torch.int32, device=dev)
pos = torch.arange(T, dtype=torch.int32, device=dev)


def layouts():
    dc = _gated_attention_fwd_kernel.device_caches[0][0]
    out = []
    for key, kern in dc.items():
        ttgir = kern.asm["ttgir"]
        mma = sorted(
            {ln.strip()[:120] for ln in ttgir.splitlines() if ln.strip().startswith("#mma")}
        )
        out.append((kern.metadata.num_warps, kern.metadata.hash[:8], mma))
    return out


for window in (None, 2048):
    ref = ga.strict_gated_attention_fwd_varlen(
        q, k, v, gate, cu, cu, pos, window=window, tile_mode="token"
    )
    for nw in (1, 2, 4, 8):
        ga.NUM_WARPS_HEAD = nw
        _gated_attention_fwd_kernel.device_caches[0][0].clear()
        got = ga.strict_gated_attention_fwd_varlen(
            q, k, v, gate, cu, cu, pos, window=window, tile_mode="head"
        )
        eq = [torch.equal(a, b) for a, b in zip(ref, got)]
        print(f"window={window} head num_warps={nw}: equal(y,o,lse)={eq}  layouts={layouts()}")

# also: token tile with different warps vs itself (8 warps)
ga.NUM_WARPS_HEAD = 4
for nw_tok in (4, 8):
    ga.NUM_WARPS_TOKEN = nw_tok
    _gated_attention_fwd_kernel.device_caches[0][0].clear()
    got = ga.strict_gated_attention_fwd_varlen(q, k, v, gate, cu, cu, pos, window=None, tile_mode="token")
    print(f"token num_warps={nw_tok}: equal to 4-warp ref={[torch.equal(a, b) for a, b in zip(ref, got)]} layouts={layouts()}")
