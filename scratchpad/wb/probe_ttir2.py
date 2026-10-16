import importlib.util
import re
import sys

import torch

from rl_engine.models.muse_glimmer.ops import qknorm_rope as op

DEV = torch.device("cuda")


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def ttir_of(tk, bwd=False):
    q = torch.randn(2, 32, 128, device=DEV).to(torch.bfloat16)
    k = torch.randn(2, 2, 128, device=DEV).to(torch.bfloat16)
    cos, sin = op.rope_table(torch.arange(2, device=DEV), device=DEV)
    oq, ok = torch.empty_like(q), torch.empty_like(k)
    kw = dict(HQ=32, HK=2, BLOCK_HQ=32, BLOCK_HK=2, HALF=64, HAS_ROPE=True, num_warps=4,
              num_stages=1, enable_fp_fusion=False)
    if bwd:
        ret = tk._qknorm_rope_bwd_kernel[(2,)](q, k, q, k, cos, sin, oq, ok, 3.87, 1e-5, **kw)
    else:
        ret = tk._qknorm_rope_fwd_kernel[(2,)](q, k, cos, sin, oq, ok, 3.87, 1e-5, **kw)
    return ret.asm["ttir"]


new = importlib.import_module("rl_engine.models.muse_glimmer.triton.qknorm_rope")
old = load(sys.argv[1], "old_qk")
for name, tk in (("new", new), ("old", old)):
    for bwd in (False, True):
        ttir = ttir_of(tk, bwd)
        adds = [ln.strip() for ln in ttir.splitlines() if "tt.addptr" in ln]
        pids = re.findall(r"(%\w+) = tt.get_program_id x : i32", ttir)
        ext = [p for p in pids if re.search(re.escape(p) + r" : i32 to i64", ttir)]
        types = [ln.split(":")[-1].split("loc(")[0].strip() for ln in adds]
        print(name, "bwd" if bwd else "fwd", "pids", pids, "extended", ext)
        for t in sorted(set(types)):
            print("   ", types.count(t), t)
