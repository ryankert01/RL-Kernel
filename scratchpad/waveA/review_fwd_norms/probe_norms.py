# Adversarial review probes for the strict norms + normalized embedding (contract O1/O2).
# Read-only: imports repo code, writes nothing into the repo. Prints one JSON line per probe.
from __future__ import annotations

import json
import subprocess
import sys

import torch

from rl_engine.models.muse_glimmer.ops import norms as N
from rl_engine.models.muse_glimmer.triton import norms as nk

DEV = torch.device("cuda")
H = N.HIDDEN
RESULTS: dict[str, dict] = {}


def emit(name: str, **kw) -> None:
    RESULTS[name] = kw
    print(json.dumps({"probe": name, **kw}), flush=True)


def bf16(shape, seed, scale=1.0, shift=0.0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    return (torch.randn(*shape, device=DEV, generator=g) * scale + shift).to(torch.bfloat16)


def misaligned(t: torch.Tensor, elems: int = 1) -> torch.Tensor:
    buf = torch.empty(t.numel() + elems, dtype=t.dtype, device=t.device)
    view = buf[elems:].view(t.shape)
    view.copy_(t)
    assert view.is_contiguous() and torch.equal(view, t)
    return view


def ndiff(a, b) -> int:
    return int((a.float() != b.float()).sum().item())


# ---------------------------------------------------------------- N1 storage alignment
def probe_alignment():
    x = bf16((4096, H), 1, scale=2.0)
    w = bf16((H,), 2, scale=0.1)
    b = bf16((4096, H), 3)
    r = bf16((4096, H), 4, scale=3.0)
    w2 = bf16((H,), 5, scale=0.1)
    out = {}
    ref = N.centered_rmsnorm(x, w, eps=N.RMS_NORM_EPS)
    for elems in (1, 2, 4):
        xm = misaligned(x, elems)
        got = N.centered_rmsnorm(xm, w, eps=N.RMS_NORM_EPS)
        out[f"centered_x_offset_{2 * elems}B"] = {
            "x_data_ptr_mod_16": xm.data_ptr() % 16,
            "elements_differing": ndiff(got, ref),
            "rows_differing": int((got != ref).any(dim=1).sum().item()),
        }
    got = N.centered_rmsnorm(x, misaligned(w), eps=N.RMS_NORM_EPS)
    out["centered_w_offset_2B"] = {"elements_differing": ndiff(got, ref)}
    rn, pre = N.fused_post_norm_residual_pre_norm(b, r, w, N.POST_NORM_EPS, w2, N.RMS_NORM_EPS)
    rn_m, pre_m = N.fused_post_norm_residual_pre_norm(
        misaligned(b), misaligned(r), w, N.POST_NORM_EPS, w2, N.RMS_NORM_EPS
    )
    out["fused_offset_2B"] = {
        "resid_new_elements_differing": ndiff(rn_m, rn),
        "pre_elements_differing": ndiff(pre_m, pre),
    }
    table = bf16((4099, H), 6, scale=0.02)
    ids = torch.randint(0, 4099, (512,), device=DEV)
    e_ref = N.normalized_embedding(table, ids)
    e_m = N.normalized_embedding(misaligned(table), ids)
    out["embedding_table_offset_2B"] = {"elements_differing": ndiff(e_m, e_ref)}
    # How many compiled variants of the row kernel exist now (alignment specializations)?
    caches = nk._norm_fwd_kernel.device_caches[torch.cuda.current_device()][0]
    out["norm_fwd_compiled_variants"] = len(caches)
    # Backward: dx of a misaligned input
    dy = bf16((4096, H), 7)
    xx = x.clone().requires_grad_(True)
    N.centered_rmsnorm(xx, w, eps=N.RMS_NORM_EPS).backward(dy)
    xm = misaligned(x).requires_grad_(True)
    N.centered_rmsnorm(xm, w, eps=N.RMS_NORM_EPS).backward(misaligned(dy))
    out["centered_backward_dx_offset_2B"] = {"elements_differing": ndiff(xm.grad, xx.grad)}
    emit("N1_input_alignment_changes_bits", **out)


# ---------------------------------------------------------------- N2 out-of-range ids
_CHILD = r"""
import sys, torch
from rl_engine.models.muse_glimmer.ops import norms as N
table = torch.randn(1000, 6656, device='cuda').to(torch.bfloat16)
ids = torch.tensor([3, int(sys.argv[1]), 5], device='cuda')
try:
    y = N.normalized_embedding(table, ids)
    torch.cuda.synchronize()
    print('NO_ERROR row1_all_zero=', bool((y[1] == 0).all().item()))
except Exception as e:
    print('RAISED', type(e).__name__, str(e).splitlines()[0][:160])
"""


def probe_out_of_range_ids():
    out = {}
    for bad in (1000, -1, 5000):
        p = subprocess.run(
            [sys.executable, "-c", _CHILD, str(bad)], capture_output=True, text=True, timeout=300
        )
        lines = [ln for ln in (p.stdout + p.stderr).splitlines() if ln.strip()]
        out[f"id_{bad}"] = {"returncode": p.returncode, "tail": lines[-2:]}
    emit("N2_out_of_range_token_ids", **out)


# ---------------------------------------------------------------- N3 golden 'same IEEE ops' claim
def probe_golden_division():
    g = torch.Generator(device=DEV).manual_seed(9)
    ss = torch.rand(1_000_000, device=DEV, generator=g) * 1e5
    golden_ms = ss / float(H)  # what inv_rms_fp32 does (CPU python scalar)
    ieee_ms = ss / torch.full_like(ss, float(H))  # tensor/tensor IEEE division (== div_rn)
    recip_ms = ss * (1.0 / torch.tensor(float(H), dtype=torch.float32)).item()
    # inv chain both ways
    inv_g = 1.0 / torch.sqrt(golden_ms + 1e-5)
    inv_i = 1.0 / torch.sqrt(ieee_ms + 1e-5)
    emit(
        "N3_golden_ms_is_reciprocal_multiply_on_cuda",
        samples=ss.numel(),
        golden_ms_vs_ieee_div_mismatches=ndiff(golden_ms, ieee_ms),
        golden_ms_vs_reciprocal_multiply_mismatches=ndiff(golden_ms, recip_ms),
        inv_mismatches_from_ms_alone=ndiff(inv_g, inv_i),
    )


# ---------------------------------------------------------------- N4 fused op device mixing
def probe_fused_input_validation():
    out = {}
    b = bf16((4, H), 10)
    r = bf16((4, H), 11)
    w = bf16((H,), 12)
    # eps validation: negative eps -> sqrt(negative) for an all-zero row
    z = torch.zeros(2, H, dtype=torch.bfloat16, device=DEV)
    try:
        y = N.centered_rmsnorm(z, w, eps=-1e-5)
        out["negative_eps"] = {"raised": None, "y_nan": bool(torch.isnan(y.float()).any().item())}
    except Exception as e:  # noqa: BLE001
        out["negative_eps"] = {"raised": repr(e)[:200]}
    # zero rows with eps=1e-8: finite?
    y = N.centered_rmsnorm(z, w, eps=N.POST_NORM_EPS)
    out["zero_row_post_eps"] = {"finite": bool(torch.isfinite(y.float()).all().item())}
    # weight of wrong dtype for the fused op
    try:
        N.fused_post_norm_residual_pre_norm(b, r, w.float(), 1e-8, w, 1e-5)
        out["fp32_w_post"] = {"raised": None}
    except Exception as e:  # noqa: BLE001
        out["fp32_w_post"] = {"raised": type(e).__name__}
    emit("N4_input_validation", **out)


if __name__ == "__main__":
    which = sys.argv[1:] or [
        "alignment",
        "out_of_range_ids",
        "golden_division",
        "fused_input_validation",
    ]
    for name in which:
        try:
            globals()[f"probe_{name}"]()
        except Exception as e:  # noqa: BLE001
            emit(name, crashed=repr(e)[:400])
    print("ALL_RESULTS " + json.dumps(RESULTS))
