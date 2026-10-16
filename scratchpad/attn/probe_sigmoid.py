"""Teeth for test_backward_sigmoid_is_the_forward_sigmoid (finding bwd:4).

Same construction as the test: one-key sequences -> o == v, y = bf16(o * sg_fwd(g)); feeding
dy := o to a preprocess gives do = bf16(o * sg_bwd(g)). Count mismatches for the NEW preprocess
(sigmoid_rn) and the OLD v1 preprocess (fp32 `/`), loaded from the old file.
"""
import importlib.util
import json
import sys

import torch

from rl_engine.models.muse_glimmer.ops import gated_attention as ga
from rl_engine.models.muse_glimmer.triton import gated_attention_bwd as bwd

spec = importlib.util.spec_from_file_location("_old_bwd", sys.argv[1])
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)

n = 4096
g = torch.Generator(device="cpu").manual_seed(4)
dev = torch.device("cuda")
q = torch.randn(n, 32, 128, generator=g).to(dev, torch.bfloat16)
k = torch.randn(n, 2, 128, generator=g).to(dev, torch.bfloat16)
v = torch.randn(n, 2, 128, generator=g).to(dev, torch.bfloat16)
gate = (torch.randn(n, 4096, generator=g) * 3.0).to(dev, torch.bfloat16)
cu = torch.arange(n + 1, dtype=torch.int32, device=dev)
pos = torch.zeros(n, dtype=torch.int32, device=dev)
y, o, _lse = ga.strict_gated_attention_fwd_varlen(q, k, v, gate, cu, cu, pos, window=None)
_dg, do_new, _ = bwd.gated_attention_bwd_preprocess(o, torch.ones_like(o), gate)
_dg, do_old, _ = old.gated_attention_bwd_preprocess(o, torch.ones_like(o), gate)
rec = {
    "numel": y.numel(),
    "new_preprocess_vs_forward_y": int((do_new.view(torch.int16) != y.view(torch.int16)).sum()),
    "old_preprocess_vs_forward_y": int((do_old.view(torch.int16) != y.view(torch.int16)).sum()),
}
print(json.dumps(rec))
json.dump(rec, open(sys.argv[2], "w"))
