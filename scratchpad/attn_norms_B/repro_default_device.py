import traceback

import torch

from rl_engine.models.muse_glimmer.ops import gated_attention as ga
from rl_engine.models.muse_glimmer.ops.paged_kv import PagedKVCache

DEV = torch.device("cuda")
n = 40
q = torch.randn(n, 32, 128, device=DEV).bfloat16()
k = torch.randn(n, 2, 128, device=DEV).bfloat16()
g = torch.randn(n, 4096, device=DEV).bfloat16()
cu = torch.tensor([0, n], dtype=torch.int32, device=DEV)
pos = torch.arange(n, dtype=torch.int32, device=DEV)
c = PagedKVCache(num_pages=4, num_seqs=1, max_pages_per_seq=3, device=DEV)
c.append(0, k, k)
previous = torch.get_default_dtype()
torch.set_default_dtype(torch.bfloat16)
with torch.device(DEV):
    for name, fn in (
        ("varlen", lambda: ga.strict_gated_attention_fwd_varlen(q, k, k, g, cu, cu, pos,
                                                                 window=None)),
        ("paged", lambda: ga.strict_gated_attention_fwd_paged(
            q[-1:].contiguous(), g[-1:].contiguous(), c.k_cache, c.v_cache, c.block_table,
            c.seq_lens, cu.new_tensor([0, 1]), window=None)),
    ):
        try:
            fn()
            torch.cuda.synchronize()
            print(name, "OK")
        except Exception:  # noqa: BLE001
            traceback.print_exc()
torch.set_default_dtype(previous)
