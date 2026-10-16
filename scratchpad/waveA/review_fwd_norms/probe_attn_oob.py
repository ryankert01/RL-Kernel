# Run under compute-sanitizer: packed offsets that disagree with the tensor extents are accepted
# by strict_gated_attention_fwd_varlen and the kernel reads/writes out of bounds.
#   case q:  cu_seqlens_q[-1] = 64 > Tq = 8   (reads q/gate/q_pos OOB, WRITES y/o/lse OOB)
#   case k:  cu_seqlens_k[-1] = 4096 > Tk = 64 (reads K/V OOB)
from __future__ import annotations

import sys

import torch

from rl_engine.models.muse_glimmer.ops import gated_attention as ga

DEV = torch.device("cuda")
case = sys.argv[1]
if case == "q":
    tq, tk = 8, 64
    cu_q = torch.tensor([0, 64], dtype=torch.int32, device=DEV)
    cu_k = torch.tensor([0, 64], dtype=torch.int32, device=DEV)
else:
    tq, tk = 8, 64
    cu_q = torch.tensor([0, 8], dtype=torch.int32, device=DEV)
    cu_k = torch.tensor([0, 4096], dtype=torch.int32, device=DEV)
q = torch.randn(tq, 32, 128, device=DEV).to(torch.bfloat16)
k = torch.randn(tk, 2, 128, device=DEV).to(torch.bfloat16)
v = torch.randn(tk, 2, 128, device=DEV).to(torch.bfloat16)
g = torch.randn(tq, 4096, device=DEV).to(torch.bfloat16)
pos = (torch.arange(tq, dtype=torch.int32, device=DEV) + (4088 if case == "k" else 0)).contiguous()
y, o, lse = ga.strict_gated_attention_fwd_varlen(q, k, v, g, cu_q, cu_k, pos, window=None)
torch.cuda.synchronize()
print(f"case {case}: wrapper accepted the call; y finite={bool(torch.isfinite(y.float()).all())}")
