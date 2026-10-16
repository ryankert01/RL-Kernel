# Demonstrates that cu_seqlens_q[-1] > Tq makes the strict forward WRITE outside its outputs and
# corrupt an unrelated live tensor (caching allocator places y/o right before a canary tensor).
from __future__ import annotations

import json

import torch

from rl_engine.models.muse_glimmer.ops import gated_attention as ga

DEV = torch.device("cuda")
tq = 8
row_bytes = 32 * 128 * 2
q = torch.randn(tq, 32, 128, device=DEV).to(torch.bfloat16)
k = torch.randn(64, 2, 128, device=DEV).to(torch.bfloat16)
v = torch.randn(64, 2, 128, device=DEV).to(torch.bfloat16)
g = torch.randn(tq, 4096, device=DEV).to(torch.bfloat16)
pos = torch.arange(tq, dtype=torch.int32, device=DEV)
cu_q = torch.tensor([0, 64], dtype=torch.int32, device=DEV)  # claims 64 queries, Tq = 8
cu_k = torch.tensor([0, 64], dtype=torch.int32, device=DEV)
a = torch.empty(tq * row_bytes // 2, dtype=torch.bfloat16, device=DEV)  # slot for y
b = torch.empty(tq * row_bytes // 2, dtype=torch.bfloat16, device=DEV)  # slot for o
canary = torch.full((256 * 1024,), 3.0, dtype=torch.bfloat16, device=DEV)  # 512 KiB, after a/b
pa, pc = a.data_ptr(), canary.data_ptr()
del a, b
raised = None
try:
    y, o, lse = ga.strict_gated_attention_fwd_varlen(q, k, v, g, cu_q, cu_k, pos, window=None,
                                                     tile_mode="token")
    torch.cuda.synchronize()
except Exception as e:  # noqa: BLE001
    raised = repr(e)[:200]
    y = None
print(json.dumps({
    "raised": raised,
    "y_reused_slot_a": (y.data_ptr() == pa) if y is not None else None,
    "canary_offset_from_y_bytes": (pc - y.data_ptr()) if y is not None else None,
    "canary_elements_overwritten": int((canary != 3.0).sum().item()),
    "canary_elements": canary.numel(),
}))
