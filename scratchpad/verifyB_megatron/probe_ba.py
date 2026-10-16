"""Run the conditions of the new regression tests against whichever tree is on PYTHONPATH.

Prints one JSON line per issue: what the code did (raised / mismatch counts). Used to show the
new tests fail on the pre-fix tree (f91e8b6) and pass on the fixed tree.
"""
import json
import os
import sys

import torch

from rl_engine.models.muse_glimmer.runtime import strict_kernels as sk

out = {"tree": os.environ.get("PYTHONPATH")}

# --- R1: missing sibling norms module -> must raise -------------------------------------
real = sk._optional_module


def fake(path):
    if path.endswith("ops.norms"):
        return None
    return real(path)


sk._optional_module = fake
for name in ("_OPS", "_GEMM", "_LM_TILE_OPS"):
    if hasattr(sk, name):
        setattr(sk, name, None)
try:
    ops = sk.resolve_ops()
    out["R1_missing_norms"] = {"raised": False, "all_strict": ops.provenance()["all_strict"]}
except Exception as exc:  # noqa: BLE001
    out["R1_missing_norms"] = {"raised": True, "error": type(exc).__name__}
sk._optional_module = real
for name in ("_OPS", "_GEMM", "_LM_TILE_OPS"):
    if hasattr(sk, name):
        setattr(sk, name, None)
os.environ["MUSE_STRICT_ATTN"] = "off"
try:
    ops = sk.resolve_ops()
    out["R1_attn_off"] = {"raised": False, "all_strict": ops.provenance()["all_strict"]}
except Exception as exc:  # noqa: BLE001
    out["R1_attn_off"] = {"raised": True, "error": type(exc).__name__}
os.environ["MUSE_STRICT_ATTN"] = "auto"
for name in ("_OPS", "_GEMM", "_LM_TILE_OPS"):
    if hasattr(sk, name):
        setattr(sk, name, None)

# --- R3: recompute mismatch -> must raise, grads discarded -----------------------------
from rl_engine.models.muse_glimmer.runtime import strict_trainer as st  # noqa: E402

lora = torch.nn.Linear(2, 1, bias=False)
state = st._RankState.__new__(st._RankState)
state.device = torch.device("cuda", 0)
state.binding = None
state.native_lora = [lora]
state.strict = True
state._pending = {"token_ids": [[1, 2, 3]], "starts": [1], "groups": [[0]],
                  "values": [torch.tensor([-1.0, -2.0], device="cuda")]}
state._values = lambda _i, _s, _g, *, grad: [
    torch.tensor([-1.0, -2.5], device="cuda", requires_grad=grad) * 1.0
]
lora.weight.grad = torch.ones_like(lora.weight)
try:
    info = state.cmd_backward(torch.ones(2))
    out["R3_strict_mismatch"] = {"raised": False, "mismatches": info["recompute_logp_mismatches"],
                                 "grads_kept": lora.weight.grad is not None}
except Exception as exc:  # noqa: BLE001
    out["R3_strict_mismatch"] = {"raised": True, "error": type(exc).__name__,
                                 "grads_kept": lora.weight.grad is not None}

# --- kernels_gemm:R1: token-grad rule at T = 200 vs the det_gemm dW kernel ------------
from rl_engine.models.muse_glimmer.runtime import megatron_lora as ml  # noqa: E402
from rl_engine.models.muse_glimmer.triton.det_gemm_tree import tree_gemm  # noqa: E402

gen = torch.Generator().manual_seed(200)
res = {}
for t in (200, 300, 513):
    lhs = torch.randn(t, 48, generator=gen).to(device="cuda", dtype=torch.bfloat16)
    rhs = torch.randn(t, 512, generator=gen).to(device="cuda", dtype=torch.bfloat16)
    got = ml._token_grad_gemm(lhs, rhs)
    want = tree_gemm(lhs.t(), rhs, out_dtype=torch.float32, allow_pad=True)
    res[t] = int((got.view(torch.int32) != want.view(torch.int32)).sum())
out["kernels_gemm_R1_mismatch_vs_dw_rule"] = res

# --- R2: packing dependence of the LoRA gradients ----------------------------------------
lens = [70, 200, 330]
offs = [0, 70, 270, 600]
k, n, r = 1024, 256, 16
gen = torch.Generator().manual_seed(5)
x = torch.randn(600, k, generator=gen).to(device="cuda", dtype=torch.bfloat16)
w = (torch.randn(n, k, generator=gen) * 0.05).to(device="cuda", dtype=torch.bfloat16)
a0 = (torch.randn(r, k, generator=gen) * 0.02).cuda()
b0 = (torch.randn(n, r, generator=gen) * 0.02).cuda()
gy = torch.randn(600, n, generator=gen).to(device="cuda", dtype=torch.bfloat16)
comm = sk.TPComm(None)
has_segments = "segments" in (ml.StrictColumnLinear.apply.__code__.co_varnames
                              if hasattr(ml.StrictColumnLinear.apply, "__code__") else ())


def run(packing):
    a = a0.clone().requires_grad_(True)
    b = b0.clone().requires_grad_(True)
    for group in packing:
        rows = torch.cat([torch.arange(offs[i], offs[i + 1]) for i in group]).cuda()
        segs, pos = [], 0
        for i in group:
            segs.append((pos, pos + lens[i]))
            pos += lens[i]
        xs = x.index_select(0, rows)
        if has_segments:  # the trainer passes the packed sequences
            y = ml.StrictColumnLinear.apply(xs, a, b, comm, n, w, segments=segs)
        else:  # pre-fix API: no notion of sequences inside a micro-batch
            y = ml.StrictColumnLinear.apply(xs, a, b, comm, n, w)
        (y.float() * gy.index_select(0, rows).float()).sum().backward()
    return a.grad.clone(), b.grad.clone()


ref = run([[0, 1, 2]])
pk = {}
for packing in ([[0, 1], [2]], [[0], [1], [2]]):
    got = run(packing)
    pk[str(packing)] = int((got[0].view(torch.int32) != ref[0].view(torch.int32)).sum()
                           + (got[1].view(torch.int32) != ref[1].view(torch.int32)).sum())
out["R2_packing_grad_mismatches_vs_one_pack"] = pk
out["R2_elements"] = int(a0.numel() + b0.numel())
print(json.dumps(out))
sys.stdout.flush()
