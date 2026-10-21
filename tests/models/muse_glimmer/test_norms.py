# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Muse-Glimmer-30B strict norms (RFC #501 rows normalized_embedding /
fused_post_norm_residual_pre_norm).

Contract gates: G1 forward vs fp32/fp64 golden (ws1.json tolerances), G2 batch invariance
(torch.equal), G4 backward repeat-determinism + allclose vs fp32 autograd, fusion == composition
(torch.equal), G5 trace(). Real shape H=6656 with T in {1, 16, 4096}; embedding at V=202048.
"""

from __future__ import annotations

import json
import os
from functools import partial
from pathlib import Path

import pytest
import torch

from rl_engine.models.muse_glimmer.ops import norms as N

REPO = Path(__file__).resolve().parents[3]
_PROFILE = json.loads((REPO / "rl_engine/contracts/profiles/precision/ws1.json").read_text())
_TOL_NORM = _PROFILE["accuracy"]["default"]["reduction"]["bfloat16"]  # rms_norm op_class
_TOL_EMBED = _PROFILE["accuracy"]["default"]["elementwise"]["bfloat16"]  # embedding op_class
_TOL_FP32 = _PROFILE["accuracy"]["default"]["reduction"]["float32"]

H = N.HIDDEN  # 6656
V = N.VOCAB  # 202048
EPS_RMS = N.RMS_NORM_EPS
EPS_POST = N.POST_NORM_EPS
CKPT = Path(os.path.expanduser("~/models/Muse-Glimmer-30B"))

pytestmark = pytest.mark.ws1

_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.fixture(autouse=True)
def _no_tf32():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    yield


def _gen(seed: int, device: str = "cuda") -> torch.Generator:
    return torch.Generator(device=device).manual_seed(seed)


def _bf16(shape, *, seed, scale=1.0, shift=0.0, device="cuda"):
    g = _gen(seed, device)
    return (torch.randn(*shape, device=device, generator=g) * scale + shift).to(torch.bfloat16)


def _close(a, b, tol) -> None:
    torch.testing.assert_close(a.float(), b.float(), atol=tol["atol"], rtol=tol["rtol"])


# fp64 references (independent of the goldens' fixed-order helpers)
def _rms64(x, w, eps, form):
    x64 = x.double()
    inv = torch.rsqrt((x64 * x64).mean(-1, keepdim=True) + eps)
    if form == "centered":
        return x64 * inv * (1.0 + w.double())
    if form == "weighted":
        return x64 * inv * w.double()
    return x64 * inv


# --------------------------------------------------------------------------------------------- #
# Goldens (CPU-safe): fp32 fixed-order vs fp64
# --------------------------------------------------------------------------------------------- #


def test_goldens_match_fp64_cpu():
    x = _bf16((5, 1024), seed=1, scale=2.0, device="cpu")
    w = _bf16((1024,), seed=2, scale=0.1, device="cpu")
    wf = _bf16((1024,), seed=3, scale=0.1, shift=1.0, device="cpu")
    y_c = N.centered_rmsnorm_fp32(x, w, eps=EPS_RMS)
    y_w = N.weighted_rmsnorm_fp32(x, wf, eps=EPS_RMS)
    _close(y_c, _rms64(x, w, EPS_RMS, "centered"), _TOL_FP32)
    _close(y_w, _rms64(x, wf, EPS_RMS, "weighted"), _TOL_FP32)
    table = _bf16((37, 1024), seed=4, scale=0.02, device="cpu")
    ids = torch.randint(0, 37, (9,), generator=torch.Generator().manual_seed(5))
    _close(
        N.normalized_embedding_fp32(table, ids, eps=EPS_RMS),
        _rms64(table[ids], None, EPS_RMS, "weightless"),
        _TOL_FP32,
    )
    # fused golden keeps the two contract-declared bf16 casts (post -> bf16, r' -> bf16)
    b, r = _bf16((5, 1024), seed=6, device="cpu"), _bf16((5, 1024), seed=7, scale=3.0, device="cpu")
    rnew, pre = N.fused_post_norm_residual_pre_norm_fp32(b, r, w, EPS_POST, w, EPS_RMS)
    post = _rms64(b, w, EPS_POST, "centered").to(torch.bfloat16)
    rnew64 = (r.float() + post.float()).to(torch.bfloat16)
    assert torch.equal(rnew, rnew64.float())
    _close(pre, _rms64(rnew64, w, EPS_RMS, "centered"), _TOL_FP32)


def _ieee_inv_reference(ss: torch.Tensor, hidden: int, eps: float) -> torch.Tensor:
    """inv from the golden's own ss with a correctly rounded fp32 ss / H (fp64 tensor division on
    the CPU, rounded once to fp32: exact for division since 53 >= 2 * 24 + 2)."""
    ms = torch.div(ss.cpu().double(), torch.tensor(float(hidden), dtype=torch.float64)).float()
    return 1.0 / torch.sqrt(ms.to(ss.device) + float(eps))


@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda", marks=_cuda)]
)
def test_golden_mean_square_is_ieee_division(device):
    """Review norms:R1: the golden's ms = ss / H must be the IEEE division the kernels' div_rn
    performs, on CPU and CUDA alike (torch CUDA `t / float` is t * (1/float))."""
    x = _bf16((4096, H), seed=3, scale=2.0, device=device).float()
    ss = N._sum_squares_fp32(x)
    want = _ieee_inv_reference(ss, H, EPS_RMS)
    got = N.inv_rms_fp32(x, EPS_RMS)
    bad = int((got != want).sum())
    assert bad == 0, f"{bad}/{got.numel()} rows: golden inv is not 1/sqrt(div_rn(ss, H) + eps)"
    ms = N.div_ieee(ss, H)
    ms_other = N.div_ieee(ss.cpu() if device == "cuda" else ss, H).to(ms.device)
    assert torch.equal(ms, ms_other), "div_ieee differs between CPU and CUDA"


@_cuda
def test_embedding_out_of_range_ids_fail_closed():
    """Review norms:R2: the kernel maps an out-of-range id to a zero row, so rejection rests on
    the wrapper's torch._assert_async (device-side assert: poisons the context -> subprocess)."""
    import subprocess
    import sys

    child = (
        "import torch\n"
        "from rl_engine.models.muse_glimmer.ops import norms as N\n"
        "w = torch.randn(100, 64, device='cuda').bfloat16()\n"
        "ok = N.normalized_embedding(w, torch.tensor([3, 99], device='cuda'))\n"
        "torch.cuda.synchronize()\n"
        "print('VALID-OK', flush=True)\n"
        "N.normalized_embedding(w, torch.tensor([3, 100], device='cuda'))\n"
        "torch.cuda.synchronize()\n"
        "print('CHILD-OK')\n"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO) + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [sys.executable, "-c", child], capture_output=True, text=True, env=env, timeout=600
    )
    out = proc.stdout + proc.stderr
    assert "VALID-OK" in out, out[-2000:]
    assert proc.returncode != 0 and "CHILD-OK" not in out, out[-2000:]
    assert "token_ids must be in [0, 100)" in out, out[-2000:]


def test_fused_rejects_mixed_devices():
    """Review norms:R2: the fused wrapper checks that all four tensors share one device (before,
    each tensor was only checked against its own weight)."""
    b = torch.zeros(2, 64, dtype=torch.bfloat16, device="meta")
    w = torch.zeros(64, dtype=torch.bfloat16, device="meta")
    r = torch.zeros(2, 64, dtype=torch.bfloat16, device="cpu")
    w_r = torch.zeros(64, dtype=torch.bfloat16, device="cpu")
    with pytest.raises(RuntimeError, match="share one device"):
        N.fused_post_norm_residual_pre_norm(b, r, w, 1e-8, w_r, 1e-5)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two visible GPUs")
def test_fused_rejects_two_gpus():
    b = torch.zeros(2, 64, dtype=torch.bfloat16, device="cuda:0")
    w = torch.zeros(64, dtype=torch.bfloat16, device="cuda:0")
    r = torch.zeros(2, 64, dtype=torch.bfloat16, device="cuda:1")
    w_r = torch.zeros(64, dtype=torch.bfloat16, device="cuda:1")
    with pytest.raises(RuntimeError, match="share one device"):
        N.fused_post_norm_residual_pre_norm(b, r, w, 1e-8, w_r, 1e-5)


def test_strict_rejects_non_bf16():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    x = torch.randn(2, 64, device="cuda")
    w = torch.randn(64, device="cuda")
    with pytest.raises(TypeError):
        N.centered_rmsnorm(x, w, eps=EPS_RMS)
    with pytest.raises(TypeError):
        N.normalized_embedding(x, torch.zeros(2, dtype=torch.long, device="cuda"))
    with pytest.raises(ValueError):
        N.fused_post_norm_residual_pre_norm(
            x.bfloat16(), x.bfloat16(), w.bfloat16(), 1e-8, w.bfloat16(), 1e-5, pre_form="bogus"
        )


# --------------------------------------------------------------------------------------------- #
# G1: forward correctness at the real hidden size
# --------------------------------------------------------------------------------------------- #


@_cuda
@pytest.mark.parametrize("rows", [1, 16, 4096])
def test_g1_norm_forwards_real_shape(rows):
    x = _bf16((rows, H), seed=10 + rows, scale=2.0)
    w = _bf16((H,), seed=11, scale=0.1)
    wf = _bf16((H,), seed=12, scale=0.1, shift=1.0)
    for eps in (EPS_RMS, EPS_POST):
        y = N.centered_rmsnorm(x, w, eps=eps)
        assert y.dtype == torch.bfloat16 and y.shape == x.shape
        _close(y, N.centered_rmsnorm_fp32(x, w, eps=eps), _TOL_NORM)
        _close(y, _rms64(x, w, eps, "centered"), _TOL_NORM)
    y = N.weighted_rmsnorm(x, wf, eps=EPS_RMS)
    _close(y, N.weighted_rmsnorm_fp32(x, wf, eps=EPS_RMS), _TOL_NORM)
    _close(y, _rms64(x, wf, EPS_RMS, "weighted"), _TOL_NORM)


@_cuda
@pytest.mark.parametrize("rows", [1, 16, 4096])
@pytest.mark.parametrize("pre_form", ["centered", "weighted"])
def test_g1_fused_forward_real_shape(rows, pre_form):
    b = _bf16((rows, H), seed=20 + rows)
    r = _bf16((rows, H), seed=21 + rows, scale=3.0)
    w_post = _bf16((H,), seed=22, scale=0.1)
    w_pre = _bf16((H,), seed=23, scale=0.1, shift=1.0 if pre_form == "weighted" else 0.0)
    rnew, pre = N.fused_post_norm_residual_pre_norm(
        b, r, w_post, EPS_POST, w_pre, EPS_RMS, pre_form=pre_form
    )
    rnew_g, pre_g = N.fused_post_norm_residual_pre_norm_fp32(
        b, r, w_post, EPS_POST, w_pre, EPS_RMS, pre_form=pre_form
    )
    assert rnew.dtype == pre.dtype == torch.bfloat16
    # r' is bf16 on both sides; a reduction-order flip in post can move it by one bf16 ulp.
    _close(rnew, rnew_g, _TOL_NORM)
    _close(pre, pre_g, _TOL_NORM)
    _close(pre, _rms64(rnew, w_pre, EPS_RMS, pre_form), _TOL_NORM)


@_cuda
def test_g1_small_shape_breadth():
    hidden = 1024
    x = _bf16((7, hidden), seed=30, scale=2.0)
    w = _bf16((hidden,), seed=31, scale=0.1)
    _close(N.centered_rmsnorm(x, w, eps=EPS_RMS), _rms64(x, w, EPS_RMS, "centered"), _TOL_NORM)
    _close(N.weighted_rmsnorm(x, w, eps=EPS_RMS), _rms64(x, w, EPS_RMS, "weighted"), _TOL_NORM)
    table = _bf16((513, hidden), seed=32, scale=0.02)
    ids = torch.randint(0, 513, (7,), device="cuda", generator=_gen(33))
    emb = N.normalized_embedding(table, ids)
    _close(emb, _rms64(table[ids], None, EPS_RMS, "weightless"), _TOL_EMBED)
    # 3-D inputs are flattened to rows and reshaped back
    x3 = x.reshape(1, 7, hidden)
    y3 = N.centered_rmsnorm(x3, w, eps=EPS_RMS)
    assert torch.equal(y3[0], N.centered_rmsnorm(x, w, eps=EPS_RMS))
    ids2 = ids.reshape(1, 7)
    assert N.normalized_embedding(table, ids2).shape == (1, 7, hidden)


# --------------------------------------------------------------------------------------------- #
# Fusion == composition (torch.equal)
# --------------------------------------------------------------------------------------------- #


@_cuda
@pytest.mark.parametrize("rows", [1, 16, 4096])
@pytest.mark.parametrize("pre_form", ["centered", "weighted"])
def test_fused_equals_composition(rows, pre_form):
    b = _bf16((rows, H), seed=40 + rows)
    r = _bf16((rows, H), seed=41 + rows, scale=3.0)
    w_post = _bf16((H,), seed=42, scale=0.1)
    w_pre = _bf16((H,), seed=43, scale=0.1, shift=1.0 if pre_form == "weighted" else 0.0)
    rnew, pre = N.fused_post_norm_residual_pre_norm(
        b, r, w_post, EPS_POST, w_pre, EPS_RMS, pre_form=pre_form
    )
    post = N.centered_rmsnorm(b, w_post, eps=EPS_POST)
    rnew_c = N.residual_add(r, post)
    norm = N.centered_rmsnorm if pre_form == "centered" else N.weighted_rmsnorm
    pre_c = norm(rnew_c, w_pre, eps=EPS_RMS)
    assert torch.equal(rnew, rnew_c)
    assert torch.equal(pre, pre_c)
    # residual add in bf16 is the same single fp32 add rounded once
    assert torch.equal(rnew_c, r + post)


# --------------------------------------------------------------------------------------------- #
# G2: batch invariance (torch.equal) — same logical row at batch 1 / 4 / 4096, any position
# --------------------------------------------------------------------------------------------- #

_POSITIONS = (0, 1, 777, 4095)


def _assert_row_invariant(fn, x_big, y_big):
    for p in _POSITIONS:
        y1 = fn(x_big[p : p + 1].clone())
        assert torch.equal(y1[0], y_big[p]), f"batch-1 vs batch-4096 mismatch at row {p}"
        idx = torch.tensor([5, 6, p, 7], device=x_big.device)
        y4 = fn(x_big[idx].clone())
        assert torch.equal(y4[2], y_big[p]), f"batch-4 position-2 mismatch for row {p}"
        idx16 = torch.cat([torch.arange(15, device=x_big.device) + 100, idx[2:3]])
        y16 = fn(x_big[idx16].clone())
        assert torch.equal(y16[15], y_big[p]), f"batch-16 position-15 mismatch for row {p}"


@_cuda
def test_g2_norm_batch_invariance():
    x = _bf16((4096, H), seed=50, scale=2.0)
    w = _bf16((H,), seed=51, scale=0.1)
    wf = _bf16((H,), seed=52, scale=0.1, shift=1.0)
    for eps in (EPS_RMS, EPS_POST):
        fn = partial(N.centered_rmsnorm, w=w, eps=eps)
        _assert_row_invariant(fn, x, fn(x))
    fn = partial(N.weighted_rmsnorm, w=wf, eps=EPS_RMS)
    _assert_row_invariant(fn, x, fn(x))


@_cuda
@pytest.mark.parametrize("pre_form", ["centered", "weighted"])
def test_g2_fused_batch_invariance(pre_form):
    b = _bf16((4096, H), seed=60)
    r = _bf16((4096, H), seed=61, scale=3.0)
    w_post = _bf16((H,), seed=62, scale=0.1)
    w_pre = _bf16((H,), seed=63, scale=0.1)
    big = N.fused_post_norm_residual_pre_norm(
        b, r, w_post, EPS_POST, w_pre, EPS_RMS, pre_form=pre_form
    )
    for p in _POSITIONS:
        for idx, pos in ((torch.tensor([p]), 0), (torch.tensor([5, 6, p, 7]), 2)):
            idx = idx.to("cuda")
            out = N.fused_post_norm_residual_pre_norm(
                b[idx].clone(), r[idx].clone(), w_post, EPS_POST, w_pre, EPS_RMS, pre_form=pre_form
            )
            assert torch.equal(out[0][pos], big[0][p])
            assert torch.equal(out[1][pos], big[1][p])


@_cuda
def test_g2_embedding_batch_invariance_small_vocab():
    table = _bf16((4099, H), seed=70, scale=0.02)
    ids = torch.randint(0, 4099, (4096,), device="cuda", generator=_gen(71))
    ids[1000] = ids[0]  # same token at two positions
    big = N.normalized_embedding(table, ids)
    assert torch.equal(big[0], big[1000])
    _assert_row_invariant(lambda t: N.normalized_embedding(table, t), ids, big)


# --------------------------------------------------------------------------------------------- #
# G4: backward — repeat determinism (torch.equal) and allclose vs fp32 autograd of the golden
# --------------------------------------------------------------------------------------------- #


def _grads_norm(fn, x, w, eps, dy):
    xx, ww = x.clone().requires_grad_(True), w.clone().requires_grad_(True)
    fn(xx, ww, eps=eps).backward(dy)
    return xx.grad, ww.grad


@_cuda
@pytest.mark.parametrize("rows", [1, 16, 4096])
@pytest.mark.parametrize("form", ["centered", "weighted"])
def test_g4_norm_backward(rows, form):
    fn, fn32 = (
        (N.centered_rmsnorm, N.centered_rmsnorm_fp32)
        if form == "centered"
        else (N.weighted_rmsnorm, N.weighted_rmsnorm_fp32)
    )
    x = _bf16((rows, H), seed=80 + rows, scale=2.0)
    w = _bf16((H,), seed=81, scale=0.1, shift=1.0 if form == "weighted" else 0.0)
    dy = _bf16((rows, H), seed=82 + rows)
    dx, dw = _grads_norm(fn, x, w, EPS_RMS, dy)
    dx2, dw2 = _grads_norm(fn, x, w, EPS_RMS, dy)
    assert torch.equal(dx, dx2) and torch.equal(dw, dw2)
    assert dx.dtype == dw.dtype == torch.bfloat16
    x32, w32 = x.float().requires_grad_(True), w.float().requires_grad_(True)
    fn32(x32, w32, eps=EPS_RMS).backward(dy.float())
    _close(dx, x32.grad, _TOL_NORM)
    _close(dw, w32.grad, _TOL_NORM)
    if rows == 4096:  # dx is row-local: batch-1 bits equal batch-4096 bits
        p = 777
        dx1, _ = _grads_norm(fn, x[p : p + 1].clone(), w, EPS_RMS, dy[p : p + 1].clone())
        assert torch.equal(dx1[0], dx[p])


def _grads_fused(b, r, w_post, w_pre, pre_form, d_rnew, d_pre):
    bb, rr = b.clone().requires_grad_(True), r.clone().requires_grad_(True)
    wp, wq = w_post.clone().requires_grad_(True), w_pre.clone().requires_grad_(True)
    rnew, pre = N.fused_post_norm_residual_pre_norm(
        bb, rr, wp, EPS_POST, wq, EPS_RMS, pre_form=pre_form
    )
    outs, grads = [], []
    if d_rnew is not None:
        outs.append(rnew)
        grads.append(d_rnew)
    if d_pre is not None:
        outs.append(pre)
        grads.append(d_pre)
    torch.autograd.backward(outs, grads)
    return bb.grad, rr.grad, wp.grad, wq.grad


def _grads_fused_fp32(b, r, w_post, w_pre, pre_form, d_rnew, d_pre):
    bb, rr = b.float().requires_grad_(True), r.float().requires_grad_(True)
    wp, wq = w_post.float().requires_grad_(True), w_pre.float().requires_grad_(True)
    rnew, pre = N.fused_post_norm_residual_pre_norm_fp32(
        bb, rr, wp, EPS_POST, wq, EPS_RMS, pre_form=pre_form
    )
    outs, grads = [], []
    if d_rnew is not None:
        outs.append(rnew)
        grads.append(d_rnew.float())
    if d_pre is not None:
        outs.append(pre)
        grads.append(d_pre.float())
    torch.autograd.backward(outs, grads)
    # w_pre is outside the graph when only resid_new is consumed; the strict op returns zeros.
    return tuple(t.grad if t.grad is not None else torch.zeros_like(t) for t in (bb, rr, wp, wq))


@_cuda
@pytest.mark.parametrize("rows", [1, 4096])
@pytest.mark.parametrize("pre_form", ["centered", "weighted"])
@pytest.mark.parametrize("which", ["both", "pre_only", "rnew_only"])
def test_g4_fused_backward(rows, pre_form, which):
    b = _bf16((rows, H), seed=90 + rows)
    r = _bf16((rows, H), seed=91 + rows, scale=3.0)
    w_post = _bf16((H,), seed=92, scale=0.1)
    w_pre = _bf16((H,), seed=93, scale=0.1, shift=1.0 if pre_form == "weighted" else 0.0)
    d_rnew = _bf16((rows, H), seed=94 + rows) if which != "pre_only" else None
    d_pre = _bf16((rows, H), seed=95 + rows) if which != "rnew_only" else None
    g1 = _grads_fused(b, r, w_post, w_pre, pre_form, d_rnew, d_pre)
    g2 = _grads_fused(b, r, w_post, w_pre, pre_form, d_rnew, d_pre)
    for a, c in zip(g1, g2, strict=True):
        assert torch.equal(a, c)
        assert a.dtype == torch.bfloat16
    ref = _grads_fused_fp32(b, r, w_post, w_pre, pre_form, d_rnew, d_pre)
    for a, c in zip(g1, ref, strict=True):
        _close(a, c, _TOL_NORM)
    if d_pre is None:
        assert torch.count_nonzero(g1[3]) == 0  # w_pre receives no gradient


# --------------------------------------------------------------------------------------------- #
# Embedding at the real vocabulary (random weights), forward + backward
# --------------------------------------------------------------------------------------------- #


@_cuda
def test_embedding_real_vocab_random_weights():
    table = torch.randn(V, H, device="cuda", dtype=torch.bfloat16, generator=_gen(100)) * 0.02
    ids = torch.randint(0, V, (4096,), device="cuda", generator=_gen(101))
    ids[-1] = V - 1
    ids[0] = 0
    ids[7] = ids[3]
    y = N.normalized_embedding(table, ids)
    assert y.shape == (4096, H) and y.dtype == torch.bfloat16
    _close(y, N.normalized_embedding_fp32(table, ids), _TOL_EMBED)
    _close(y, _rms64(table[ids], None, EPS_RMS, "weightless"), _TOL_EMBED)
    assert torch.equal(y[7], y[3])
    for p in _POSITIONS:
        assert torch.equal(N.normalized_embedding(table, ids[p : p + 1].clone())[0], y[p])

    # backward: repeat-deterministic dweight through the sorted-segment path
    dy = _bf16((4096, H), seed=102)
    grads = []
    for _ in range(2):
        tt = table.clone().requires_grad_(True)
        N.normalized_embedding(tt, ids).backward(dy)
        grads.append(tt.grad)
    assert torch.equal(grads[0], grads[1])
    assert grads[0].dtype == torch.bfloat16 and grads[0].shape == (V, H)
    uniq = torch.unique(ids)
    touched = (grads[0] != 0).any(dim=1)
    assert torch.equal(touched.nonzero().flatten(), uniq)
    t32 = table.float().requires_grad_(True)
    N.normalized_embedding_fp32(t32, ids).backward(dy.float())
    _close(grads[0][uniq], t32.grad[uniq], _TOL_NORM)
    del grads, t32


@_cuda
@pytest.mark.skipif(not (CKPT / "REVISION").exists(), reason="Muse-Glimmer-30B checkpoint absent")
def test_embedding_real_checkpoint_weights():
    from safetensors import safe_open

    index = json.loads((CKPT / "model.safetensors.index.json").read_text())["weight_map"]
    name = "model.language_model.embed_tokens.weight"
    with safe_open(str(CKPT / index[name]), framework="pt", device="cuda") as f:
        table = f.get_tensor(name)
    assert table.dtype == torch.bfloat16 and tuple(table.shape) == (V, H)
    ids = torch.randint(0, V, (4096,), device="cuda", generator=_gen(110))
    y = N.normalized_embedding(table, ids)
    assert torch.isfinite(y.float()).all()
    _close(y, N.normalized_embedding_fp32(table, ids), _TOL_EMBED)
    _close(y, _rms64(table[ids], None, EPS_RMS, "weightless"), _TOL_EMBED)
    rms = y.float().pow(2).mean(-1).sqrt()
    nonzero_rows = table[ids].float().abs().sum(-1) > 0
    assert torch.allclose(rms[nonzero_rows], torch.ones_like(rms[nonzero_rows]), atol=0.02)
    for p in _POSITIONS:
        assert torch.equal(N.normalized_embedding(table, ids[p : p + 1].clone())[0], y[p])


# --------------------------------------------------------------------------------------------- #
# G5: trace()
# --------------------------------------------------------------------------------------------- #


def test_g5_trace():
    all_ops = N.trace()
    assert set(all_ops) == {
        "normalized_embedding",
        "centered_rmsnorm",
        "weighted_rmsnorm",
        "residual_add",
        "fused_post_norm_residual_pre_norm",
    }
    for name, t in all_ops.items():
        assert t["op"] == name
        for key in ("op", "dtype_in", "dtype_acc", "dtype_out", "reduction_order", "tile_config"):
            assert key in t, (name, key)
        assert t["dtype_acc"] == "float32" and t["dtype_out"] == "bfloat16"
        assert t == N.trace(name)
    assert N.trace() == N.trace()
    cfg = N.trace("centered_rmsnorm")["tile_config"]
    assert cfg["BLOCK_H"] == 8192 and cfg["num_warps"] == 8 and cfg["enable_fp_fusion"] is False
    assert N.trace("centered_rmsnorm", hidden=1024)["tile_config"]["BLOCK_H"] == 1024
    with pytest.raises(KeyError):
        N.trace("nope")
