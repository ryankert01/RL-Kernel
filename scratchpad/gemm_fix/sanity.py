import json

import torch

from rl_engine.models.muse_glimmer.ops import det_gemm as dg
from rl_engine.models.muse_glimmer.ops import lm_head_softcap_logprob as lm
from rl_engine.models.muse_glimmer.ops import swiglu_ffn as sf
from rl_engine.models.muse_glimmer.ops.canonical_tree import combine_tree
from rl_engine.models.muse_glimmer.triton import det_gemm_tree as kt

for k, n in ((256, 8), (512, 8), (768, 8), (1024, 8), (6656, 8), (128, 4), (64, 2), (32, 1)):
    print(k, n, kt.plan_chunks(k, n))
print(kt.plan_chunks(16, allow_pad=True), kt.plan_chunks(300, allow_pad=True))
t = dg.trace()
print(sorted(t))
print(json.dumps(sf.trace()["tile_config"]["gemm"])[:200])
print(lm.gemm_backend_name(), lm.gemm_backend_name(lm.legacy_det_gemm_fp32))
print(dg.fast_det_gemm.trace is dg.trace)

torch.manual_seed(0)
dev = "cuda"
for k in (256, 512, 1024, 4096):
    a = torch.randn(64, k, device=dev).bfloat16()
    w = (torch.randn(384, k, device=dev) * 0.02).bfloat16()
    full = dg.fast_det_gemm(a, w, out_dtype=torch.float32)
    c = k // 8
    parts = [kt.tree_gemm(a[:, i * c:(i + 1) * c], w[:, i * c:(i + 1) * c].t(),
                          out_dtype=torch.float32, num_chunks=1) for i in range(8)]
    print(k, kt.last_launch()["plan"], "kernel==tree", torch.equal(full, combine_tree(parts)))
print(dg.trace()["runtime"])
