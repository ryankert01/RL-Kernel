"""Dump the HEAD-tile local PTX/TTGIR/LLIR of the strict attention forward for inspection."""
import sys

import torch

from rl_engine.models.muse_glimmer.ops import gated_attention as ga
from rl_engine.models.muse_glimmer.triton import gated_attention_fwd as kern

DEV = torch.device("cuda")
n = 100
q = torch.randn(n, 32, 128, device=DEV).bfloat16()
k = torch.randn(n, 2, 128, device=DEV).bfloat16()
v = torch.randn(n, 2, 128, device=DEV).bfloat16()
g = torch.randn(n, 4096, device=DEV).bfloat16()
cu = torch.tensor([0, n], dtype=torch.int32, device=DEV)
pos = torch.arange(n, dtype=torch.int32, device=DEV)
ga.strict_gated_attention_fwd_varlen(q, k, v, g, cu, cu, pos, window=2048, tile_mode="head")
caches = kern._gated_attention_fwd_kernel.device_caches[torch.cuda.current_device()][0]
compiled = list(caches.values())[0]
for ext in ("ptx", "ttgir", "llir"):
    with open(f"{sys.argv[1]}.{ext}", "w") as fh:
        fh.write(compiled.asm[ext])
print("written")
