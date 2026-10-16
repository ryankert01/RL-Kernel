# Which attention forward do the two runtimes actually execute, and do they agree bitwise?
#  - vLLM binding: strict_ops.resolve_ops() + probe_ops() (run at model bind, vllm_strict.py:641)
#  - Megatron binding: strict_kernels.resolve_ops() (megatron_strict.py:296)
# Compares each resolved op against the strict Triton kernel and against each other.
from __future__ import annotations

import json

import torch

from rl_engine.models.muse_glimmer.ops import gated_attention as ga
from rl_engine.models.muse_glimmer.ops.paged_kv import PagedKVCache
from rl_engine.models.muse_glimmer.runtime import strict_kernels as SK
from rl_engine.models.muse_glimmer.runtime.strict_ops import probe_ops, resolve_ops

torch.backends.cuda.matmul.allow_tf32 = False
DEV = torch.device("cuda")


def inputs(n, seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    q = (torch.randn(n, 32, 128, device=DEV, generator=g) * 3.87).to(torch.bfloat16)
    k = torch.randn(n, 2, 128, device=DEV, generator=g).to(torch.bfloat16)
    v = torch.randn(n, 2, 128, device=DEV, generator=g).to(torch.bfloat16)
    gate = (torch.randn(n, 4096, device=DEV, generator=g) * 2.0).to(torch.bfloat16)
    return q, k, v, gate


def cmp(a, b):
    d = (a.float() - b.float()).abs()
    return {"mismatched_fraction": float((d > 0).float().mean().item()), "max_abs": float(d.max().item())}


vops = probe_ops(resolve_ops(False), DEV)
mops = SK.resolve_ops()
res = {
    "vllm_label_paged": vops.labels["gated_attention_paged"],
    "vllm_demotions": vops.demotions,
    "megatron_source": str(mops.sources["gated_attention"].detail),
}
n = 1100
q, k, v, gate = inputs(n, 5)
cu = torch.tensor([0, n], dtype=torch.int32, device=DEV)
pos = torch.arange(n, dtype=torch.int32, device=DEV)
for window, tag in ((None, "global"), (2048, "local")):
    y_strict = ga.strict_gated_attention_fwd_varlen(q, k, v, gate, cu, cu, pos, window=window)[0]
    # rollout prefill as the vLLM binding would run it (paged op, as resolved after probe_ops)
    cache = PagedKVCache(num_pages=80, num_seqs=1, max_pages_per_seq=70, device=DEV)
    cache.append(0, k, v)
    y_vllm = vops.gated_attention_paged(
        q, gate, cache.k_cache, cache.v_cache, cache.block_table, cache.seq_lens, cu, window=window
    )[0]
    y_meg = mops.gated_attention(q, k, v, gate, cu, cu, pos, window=window)
    res[tag] = {
        "vllm_resolved_vs_strict_kernel": cmp(y_vllm, y_strict),
        "megatron_resolved_vs_strict_kernel": cmp(y_meg, y_strict),
        "vllm_resolved_vs_megatron_resolved": cmp(y_vllm, y_meg),
    }
print(json.dumps(res, indent=1, default=str))
