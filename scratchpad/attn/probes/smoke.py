"""Smoke: new bwd kernel == old kernel run per canonical chunk (leaves), CP1 path, TP shards."""
import importlib.util
import sys

import torch

torch.backends.cuda.matmul.allow_tf32 = False
from rl_engine.models.muse_glimmer.ops import gated_attention as ga  # noqa: E402
from rl_engine.models.muse_glimmer.ops import gated_attention_autograd as ag  # noqa: E402
from rl_engine.models.muse_glimmer.triton import gated_attention_bwd as bwd  # noqa: E402

spec = importlib.util.spec_from_file_location("oldbwd", sys.argv[1])
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)

dev = "cuda"


def inputs(n, seed, hq=32, hkv=2):
    g = torch.Generator(device="cpu").manual_seed(seed)
    q = (torch.randn(n, hq, 128, generator=g) * 3.87).to(dev, torch.bfloat16)
    k = torch.randn(n, hkv, 128, generator=g).to(dev, torch.bfloat16)
    v = torch.randn(n, hkv, 128, generator=g).to(dev, torch.bfloat16)
    gate = torch.randn(n, hq * 128, generator=g).to(dev, torch.bfloat16)
    dy = torch.randn(n, hq * 128, generator=g).to(dev, torch.bfloat16)
    return q, k, v, gate, dy


print(ag.trace()["forward_impl"], ag.trace()["forward_is_fallback"])
for window in (None, 2048):
    for n in (1, 2, 7, 64, 300, 2048, 2100, 4096):
        q, k, v, gate, dy = inputs(n, n)
        cu = torch.tensor([0, n], dtype=torch.int32, device=dev)
        pos = torch.arange(n, dtype=torch.int32, device=dev)
        y, o, lse = ga.strict_gated_attention_fwd_varlen(q, k, v, gate, cu, cu, pos, window=window)
        dgate, do, delta = bwd.gated_attention_bwd_preprocess(dy, o, gate)
        dq, lk, lv = bwd.strict_gated_attention_bwd_chunk_leaves(
            q, k, v, do, lse, delta, cu, cu, pos, window=window
        )
        # old kernel, called per canonical chunk (the v1 CP path)
        bad = 0
        for c in range(8):
            lo, hi = (c * n) // 8, ((c + 1) * n) // 8
            if hi == lo:
                assert lk[c].abs().max() == 0
                continue
            cuq = torch.tensor([0, hi - lo], dtype=torch.int32, device=dev)
            _dq, dkp, dvp = old.strict_gated_attention_bwd_partials(
                q[lo:hi].contiguous(), k, v, do[lo:hi].contiguous(), lse[lo:hi].contiguous(),
                delta[lo:hi].contiguous(), cuq, cu, pos[lo:hi].contiguous(), window=window,
            ) if n > 1 else (None, None, None)
            if dkp is None:
                continue
            for gidx in range(8):
                kvh = (4 * gidx) // 16
                bad += int((dkp[gidx, :, kvh] != lk[c, gidx]).sum())
                bad += int((dvp[gidx, :, kvh] != lv[c, gidx]).sum())
            bad += int((_dq != dq[lo:hi]).sum())
        full = bwd.strict_gated_attention_bwd(q, k, v, gate, o, lse, dy, cu, cu, pos, window=window)
        print(f"window={window} n={n} leaf-vs-old-per-chunk mismatches={bad}",
              "dk finite", bool(torch.isfinite(full[1].float()).all()))
