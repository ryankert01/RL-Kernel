# Extra probes: (1) Megatron strict binding's resolved attention vs the strict kernel bits;
# (2) the REPORT's warp-split claim (HEAD tile with 1/2/8 warps vs TOKEN tile with 4 warps);
# (3) undersized max_seqlen_q on the PAGED path (the vLLM call passes attn_metadata.max_query_len).
from __future__ import annotations

import json

import torch

from rl_engine.models.muse_glimmer.ops import gated_attention as ga
from rl_engine.models.muse_glimmer.ops.paged_kv import PagedKVCache
from rl_engine.models.muse_glimmer.runtime import strict_kernels as S
from rl_engine.models.muse_glimmer.triton import gated_attention_fwd as kern

torch.backends.cuda.matmul.allow_tf32 = False
DEV = torch.device("cuda")


def inputs(n, seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    q = (torch.randn(n, 32, 128, device=DEV, generator=g) * 3.87).to(torch.bfloat16)
    k = torch.randn(n, 2, 128, device=DEV, generator=g).to(torch.bfloat16)
    v = torch.randn(n, 2, 128, device=DEV, generator=g).to(torch.bfloat16)
    gate = (torch.randn(n, 4096, device=DEV, generator=g) * 2.0).to(torch.bfloat16)
    return q, k, v, gate


res = {}
# (1) ------------------------------------------------------------------------------------------
n = 2300
q, k, v, gate = inputs(n, 1)
cu = torch.tensor([0, n], dtype=torch.int32, device=DEV)
pos = torch.arange(n, dtype=torch.int32, device=DEV)
for window in (None, 2048):
    y_strict, _, _ = ga.strict_gated_attention_fwd_varlen(q, k, v, gate, cu, cu, pos, window=window)
    ops = S.resolve_ops()
    y_meg = ops.gated_attention(q, k, v, gate, cu, cu, pos, window=window)
    d = (y_meg.float() - y_strict.float()).abs()
    res[f"megatron_resolved_attention_vs_strict_{'global' if window is None else 'local'}"] = {
        "source": str(ops.sources["gated_attention"].detail),
        "elements": y_strict.numel(),
        "mismatched_elements": int((d > 0).sum().item()),
        "mismatched_fraction": float((d > 0).float().mean().item()),
        "max_abs": float(d.max().item()),
    }

# (2) ------------------------------------------------------------------------------------------
n = 300
q, k, v, gate = inputs(n, 2)
cu = torch.tensor([0, n], dtype=torch.int32, device=DEV)
pos = torch.arange(n, dtype=torch.int32, device=DEV)
ref = ga.strict_gated_attention_fwd_varlen(q, k, v, gate, cu, cu, pos, window=None, tile_mode="token")
dummy = torch.zeros((1,), dtype=torch.int32, device=DEV)
warp = {}
for nw in (1, 2, 4, 8):
    y = torch.empty_like(ref[0])
    o = torch.empty_like(ref[1])
    lse = torch.empty_like(ref[2])
    kern._gated_attention_fwd_kernel[(n, 1, 2)](
        q, gate, k, v, y, o, lse, pos, cu, cu, dummy, dummy, float(128**-0.5), 0, 0,
        H_Q=32, H_KV=2, D=128, PAGE=16, BLOCK_M=16, BLOCK_N=64, LOCAL=False, PAGED=False,
        HEAD_TILE=True, num_warps=nw, num_stages=2,
    )
    warp[f"head_tile_num_warps_{nw}"] = {
        "y_mismatch": int((y != ref[0]).sum().item()),
        "lse_mismatch": int((lse != ref[2]).sum().item()),
    }
res["warp_split_claim"] = warp

# (3) ------------------------------------------------------------------------------------------
n = 100
q, k, v, gate = inputs(n, 3)
cache = PagedKVCache(num_pages=16, num_seqs=2, max_pages_per_seq=8, device=DEV)
cache.append(0, k, v)
cache.append(1, k[:40], v[:40])
qq = torch.cat([q[60:100], q[39:40]]).contiguous()  # seq0: 40-token chunk, seq1: decode
gg = torch.cat([gate[60:100], gate[39:40]]).contiguous()
cu_q = torch.tensor([0, 40, 41], dtype=torch.int32, device=DEV)
good = ga.strict_gated_attention_fwd_paged(
    qq, gg, cache.k_cache, cache.v_cache, cache.block_table, cache.seq_lens, cu_q, window=2048
)
bad = ga.strict_gated_attention_fwd_paged(
    qq, gg, cache.k_cache, cache.v_cache, cache.block_table, cache.seq_lens, cu_q, window=2048,
    max_seqlen_q=1,
)
res["paged_max_seqlen_q_undersized"] = {
    "rows": int(qq.shape[0]),
    "rows_wrong": int((bad[0] != good[0]).any(dim=1).sum().item()),
    "raised": None,
}
print(json.dumps(res, indent=1))
