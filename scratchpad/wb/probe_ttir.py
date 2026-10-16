import sys

import torch

from rl_engine.models.muse_glimmer.ops import qknorm_rope as op
from rl_engine.models.muse_glimmer.triton import qknorm_rope as tk

DEV = torch.device("cuda")
q = torch.randn(2, 32, 128, device=DEV).to(torch.bfloat16)
k = torch.randn(2, 2, 128, device=DEV).to(torch.bfloat16)
cos, sin = op.rope_table(torch.arange(2, device=DEV), device=DEV)
oq, ok = torch.empty_like(q), torch.empty_like(k)
ret = tk._qknorm_rope_fwd_kernel[(2,)](
    q, k, cos, sin, oq, ok, 3.87, 1e-5, HQ=32, HK=2, BLOCK_HQ=32, BLOCK_HK=2, HALF=64,
    HAS_ROPE=True, num_warps=4, num_stages=1, enable_fp_fusion=False)
print(type(ret))
ttir = ret.asm["ttir"]
lines = [ln.strip() for ln in ttir.splitlines() if "tt.addptr" in ln]
print(len(lines))
for ln in lines[:6]:
    print(ln[:200])
pid = [ln.strip() for ln in ttir.splitlines() if "get_program_id" in ln or "extsi" in ln]
print(pid[:5])
if len(sys.argv) > 1:
    print(ttir[:3000])
