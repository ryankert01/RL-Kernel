"""Reproduce findings bwd:4/5/7/8/9 on the OLD backward module (6a61f6e) and show the NEW one.

Usage: python probe_findings_old_vs_new.py <old_gated_attention_bwd.py> <out.json>
The forward used here is the current one (forward bits are unchanged, see the digests).
"""
import importlib.util
import json
import sys
import traceback

import torch

torch.backends.cuda.matmul.allow_tf32 = False
from rl_engine.models.muse_glimmer.ops import gated_attention as ga  # noqa: E402
from rl_engine.models.muse_glimmer.triton import gated_attention_bwd as new  # noqa: E402

spec = importlib.util.spec_from_file_location("oldbwd", sys.argv[1])
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
dev = torch.device("cuda")
out = {}


def outcome(fn):
    try:
        fn()
        return "no error"
    except Exception as exc:  # noqa: BLE001
        return f"{type(exc).__name__}: {str(exc).splitlines()[0][:160]}"


# ---- finding 4: forward sigmoid vs backward sigmoid on 16.8M gate values (one-key sequences)
n = 4096
g = torch.Generator(device="cpu").manual_seed(4)
q = torch.randn(n, 32, 128, generator=g).to(dev, torch.bfloat16)
k = torch.randn(n, 2, 128, generator=g).to(dev, torch.bfloat16)
v = torch.randn(n, 2, 128, generator=g).to(dev, torch.bfloat16)
gate = (torch.randn(n, 4096, generator=g) * 3.0).to(dev, torch.bfloat16)
cu = torch.arange(n + 1, dtype=torch.int32, device=dev)
pos = torch.zeros(n, dtype=torch.int32, device=dev)
y, o, _ = ga.strict_gated_attention_fwd_varlen(q, k, v, gate, cu, cu, pos, window=None)
ones = torch.ones_like(o)
for name, mod in (("old", old), ("new", new)):
    _dg, do, _d = mod.gated_attention_bwd_preprocess(o, ones, gate)
    out[f"f4_{name}_do_vs_forward_y_mismatch"] = int((do != y).sum())
    dg1, _do1, _ = mod.gated_attention_bwd_preprocess(ones, ones, gate)
    out[f"f4_{name}_dgate_unit_digest"] = float(dg1.float().sum())
_dgo, _, _ = old.gated_attention_bwd_preprocess(ones, ones, gate)
_dgn, _, _ = new.gated_attention_bwd_preprocess(ones, ones, gate)
out["f4_dgate(sg*(1-sg))_old_vs_new_mismatch"] = int((_dgo != _dgn).sum())
out["f4_values"] = n * 4096

# common small inputs
L = 128
g = torch.Generator(device="cpu").manual_seed(7)
q = torch.randn(L, 32, 128, generator=g).to(dev, torch.bfloat16)
k = torch.randn(L, 2, 128, generator=g).to(dev, torch.bfloat16)
v = torch.randn(L, 2, 128, generator=g).to(dev, torch.bfloat16)
gate = torch.randn(L, 4096, generator=g).to(dev, torch.bfloat16)
dy = torch.randn(L, 4096, generator=g).to(dev, torch.bfloat16)
cu = torch.tensor([0, 64, 128], dtype=torch.int32, device=dev)
pos = torch.cat([torch.arange(64), torch.arange(64)]).to(dev, torch.int32)
y, o, lse = ga.strict_gated_attention_fwd_varlen(q, k, v, gate, cu, cu, pos, window=None)

# ---- finding 5: TP shard call
for name, mod in (("old", old), ("new", new)):
    out[f"f5_{name}_tp2_shard_call"] = outcome(
        lambda mod=mod: mod.strict_gated_attention_bwd_partials(
            q[:, :16].contiguous(), k[:, :1].contiguous(), v[:, :1].contiguous(),
            torch.zeros(L, 2048, dtype=torch.bfloat16, device=dev),
            lse[:, :16].contiguous(), lse[:, :16].contiguous(), cu, cu, pos, window=None,
            **({"num_q_heads": 16, "num_kv_heads": 1} if mod is new else {}),
        )
    )

# ---- finding 7: fp32 dy silently cast
for name, mod in (("old", old), ("new", new)):
    out[f"f7_{name}_fp32_dy"] = outcome(
        lambda mod=mod: mod.strict_gated_attention_bwd(
            q, k, v, gate, o, lse, dy.float(), cu, cu, pos, window=None
        )
    )

# ---- finding 8: inconsistent cu_seqlens_k / device
short_k = cu.clone()
short_k[-1] = 100
for name, mod in (("old", old), ("new", new)):
    out[f"f8_{name}_cu_k_last_100_vs_Tk_128"] = outcome(
        lambda mod=mod: mod.strict_gated_attention_bwd(
            q, k, v, gate, o, lse, dy, cu, short_k, pos, window=None
        )
    )
    out[f"f8_{name}_cpu_cu_seqlens_q"] = outcome(
        lambda mod=mod: mod.strict_gated_attention_bwd(
            q, k, v, gate, o, lse, dy, cu.cpu(), cu, pos, window=None
        )
    )

# ---- finding 9: host syncs per backward call
for name, mod in (("old", old), ("new", new)):
    kwargs = {}
    if mod is new:
        kwargs["seq_meta"] = new.prepare_sequence_meta(cu, cu, pos, num_k_tokens=L)
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        out[f"f9_{name}_backward_under_sync_debug_error"] = outcome(
            lambda mod=mod, kwargs=kwargs: mod.strict_gated_attention_bwd(
                q, k, v, gate, o, lse, dy, cu, cu, pos, window=None, **kwargs
            )
        )
    finally:
        torch.cuda.set_sync_debug_mode("default")

# ---- finding: Tk == 1 compile error (not in the review list)
q1, k1, v1 = q[:1].contiguous(), k[:1].contiguous(), v[:1].contiguous()
g1, dy1 = gate[:1].contiguous(), dy[:1].contiguous()
cu1 = torch.tensor([0, 1], dtype=torch.int32, device=dev)
p1 = torch.zeros(1, dtype=torch.int32, device=dev)
y1, o1, l1 = ga.strict_gated_attention_fwd_varlen(q1, k1, v1, g1, cu1, cu1, p1, window=None)
for name, mod in (("old", old), ("new", new)):
    out[f"t1_{name}_backward_T1"] = outcome(
        lambda mod=mod: mod.strict_gated_attention_bwd(
            q1, k1, v1, g1, o1, l1, dy1, cu1, cu1, p1, window=None
        )
    )

print(json.dumps(out, indent=1))
with open(sys.argv[2], "w") as fh:
    json.dump(out, fh, indent=1)
