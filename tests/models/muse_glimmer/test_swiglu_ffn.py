# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""RFC #501 row ``swiglu_ffn`` (contract O6 + O3): gates G1, G2, G4, G5, real shapes, real weights.

Tolerances come from ``rl_engine/contracts/profiles/precision/ws1.json`` (read-only):
elementwise rows for SwiGLU, reduction rows for the FFN (its output is a K=F reduction).
Parity gates use ``torch.equal``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from rl_engine.models.muse_glimmer.ops.swiglu_ffn import (
    TritonDetGemmAdapter,
    default_gemm,
    swiglu,
    swiglu_backward_fp32,
    swiglu_ffn,
    swiglu_ffn_fp32,
    swiglu_fp32,
    swiglu_trace,
    trace,
)
from rl_engine.models.muse_glimmer.triton.swiglu import launch_swiglu_bwd, launch_swiglu_fwd

pytestmark = pytest.mark.ws1

REPO = Path(__file__).resolve().parents[3]
_PROFILE = json.loads(
    (REPO / "rl_engine/contracts/profiles/precision/ws1.json").read_text(encoding="utf-8")
)
H, F = 6656, 19968
CKPT = Path.home() / "models" / "Muse-Glimmer-30B"
LAYER0_MLP = {
    name: f"model.language_model.layers.0.mlp.{name}.weight"
    for name in ("gate_proj", "up_proj", "down_proj")
}

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
requires_checkpoint = pytest.mark.skipif(
    not (CKPT / "REVISION").exists(), reason="Muse-Glimmer-30B checkpoint is not complete"
)


def _tol(kind: str, op_class: str, dtype: str = "bfloat16") -> tuple[float, float]:
    row = _PROFILE["judgments"][kind]["by_op_class"][op_class][dtype]
    return float(row["atol"]), float(row["rtol"])


@pytest.fixture(autouse=True)
def _disable_tf32():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    yield


def _randn(shape, *, seed: int, scale: float = 1.0, device="cuda", dtype=torch.bfloat16):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    return (torch.randn(*shape, generator=gen, dtype=torch.float32) * scale).to(device, dtype)


def _ffn_weights(hidden: int, ffn: int, *, seed: int, device="cuda"):
    w_gate = _randn((ffn, hidden), seed=seed, scale=hidden**-0.5, device=device)
    w_up = _randn((ffn, hidden), seed=seed + 1, scale=hidden**-0.5, device=device)
    w_down = _randn((hidden, ffn), seed=seed + 2, scale=ffn**-0.5, device=device)
    return w_gate, w_up, w_down


def _report_error(label: str, got: torch.Tensor, ref: torch.Tensor) -> None:
    diff = (got.double() - ref.double()).abs()
    rel = (diff / ref.double().abs().clamp_min(1e-3)).max().item()
    print(
        f"[{label}] max_abs={diff.max().item():.3e} mean_abs={diff.mean().item():.3e} "
        f"max_rel={rel:.3e}"
    )


# --- CPU-safe: goldens, provenance, validation --------------------------------------------
def test_swiglu_fp32_golden_formula():
    g = torch.linspace(-12, 12, 97, dtype=torch.float64)
    u = torch.linspace(-3, 3, 97, dtype=torch.float64)
    ref = torch.nn.functional.silu(g) * u
    got = swiglu_fp32(g.to(torch.bfloat16), u.to(torch.bfloat16))
    exact = torch.nn.functional.silu(g.to(torch.bfloat16).double()) * u.to(torch.bfloat16).double()
    assert got.dtype == torch.float32
    torch.testing.assert_close(got.double(), exact, atol=1e-6, rtol=1e-6)
    assert torch.isfinite(ref).all()


def test_swiglu_backward_fp32_matches_autograd_of_golden():
    g = torch.linspace(-20, 20, 201, dtype=torch.float32).requires_grad_()
    u = torch.linspace(-2, 2, 201, dtype=torch.float32).requires_grad_()
    dh = torch.cos(torch.arange(201, dtype=torch.float32))
    swiglu_fp32(g, u).backward(dh)
    dg, du = swiglu_backward_fp32(dh, g.detach(), u.detach())
    torch.testing.assert_close(dg, g.grad, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(du, u.grad, atol=1e-6, rtol=1e-6)


def test_g5_trace_is_emitted_and_stable():
    for tracer in (lambda: trace(TritonDetGemmAdapter()), trace, swiglu_trace):
        first, second = tracer(), tracer()
        assert first == second
        for key in ("op", "dtype_in", "dtype_acc", "dtype_out", "reduction_order", "tile_config"):
            assert key in first, key
        json.dumps(first)
    ffn = trace(TritonDetGemmAdapter())
    assert ffn["op"] == "swiglu_ffn"
    assert ffn["nodes"] == ["mlp_gate_proj", "up_proj", "swiglu", "down_proj"]
    assert ffn["tile_config"]["gemm"]["tile_config"]["k_chunks"] == 8
    assert ffn["tile_config"]["swiglu"]["BLOCK_F"] == 1024
    assert "((0+1)+(2+3))+((4+5)+(6+7))" in ffn["reduction_order"]


def test_ffn_shape_validation():
    x = torch.zeros(2, 8, dtype=torch.bfloat16)
    w_gate, w_up, w_down = _ffn_weights(8, 16, seed=1, device="cpu")
    with pytest.raises(ValueError):
        swiglu_ffn_fp32(x, w_gate, w_up, w_down[:, :8])
    with pytest.raises(ValueError):
        swiglu_fp32(torch.zeros(2, 3), torch.zeros(3, 2))


# --- G1: SwiGLU forward vs golden ----------------------------------------------------------
@requires_cuda
@pytest.mark.parametrize("shape", [(3, 40), (17, 1000), (64, F)])
def test_g1_swiglu_forward_vs_golden(shape):
    g = _randn(shape, seed=10, scale=3.0)
    u = _randn(shape, seed=11, scale=2.0)
    golden = swiglu_fp32(g, u)
    golden64 = torch.nn.functional.silu(g.double()) * u.double()
    h = swiglu(g, u)
    assert h.dtype == torch.bfloat16 and h.shape == g.shape
    atol, rtol = _tol("forward_accuracy", "elementwise", "bfloat16")
    torch.testing.assert_close(h.float(), golden, atol=atol, rtol=rtol)
    torch.testing.assert_close(h.double(), golden64, atol=atol, rtol=rtol)
    h32 = launch_swiglu_fwd(g, u, out_dtype=torch.float32)
    atol, rtol = _tol("forward_accuracy", "elementwise", "float32")
    torch.testing.assert_close(h32, golden, atol=atol, rtol=rtol)
    torch.testing.assert_close(h32.double(), golden64, atol=atol, rtol=rtol)
    if shape[1] == F:
        _report_error("G1 swiglu bf16 vs fp64, T=64 x F", h, golden64)


@requires_cuda
def test_g1_swiglu_saturation_corners():
    g = torch.tensor([[-100.0, -30.0, -10.0, 0.0, 10.0, 30.0, 100.0, 1e-3]], device="cuda")
    u = torch.full_like(g, 1.5)
    g, u = g.to(torch.bfloat16), u.to(torch.bfloat16)
    h = swiglu(g, u)
    assert torch.isfinite(h).all()
    torch.testing.assert_close(h.float(), swiglu_fp32(g, u), atol=2e-2, rtol=1.6e-2)
    dh = torch.ones_like(g)
    dg, du = launch_swiglu_bwd(dh, g, u, out_dtype=torch.float32)
    ref_dg, ref_du = swiglu_backward_fp32(dh, g, u)
    assert torch.isfinite(dg).all() and torch.isfinite(du).all()
    torch.testing.assert_close(dg, ref_dg, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(du, ref_du, atol=1e-5, rtol=1e-5)


# --- G2: batch invariance ------------------------------------------------------------------
@requires_cuda
def test_g2_swiglu_batch_invariance_real_width():
    rows = 4096
    g = _randn((rows, F), seed=20, scale=3.0)
    u = _randn((rows, F), seed=21)
    dh = _randn((rows, F), seed=22)
    full = swiglu(g, u)
    full_dg, full_du = launch_swiglu_bwd(dh, g, u)
    for i in (0, 1, 2047, 4095):
        one = swiglu(g[i : i + 1], u[i : i + 1])
        assert torch.equal(one[0], full[i])
        dg1, du1 = launch_swiglu_bwd(dh[i : i + 1], g[i : i + 1], u[i : i + 1])
        assert torch.equal(dg1[0], full_dg[i]) and torch.equal(du1[0], full_du[i])
    idx = torch.tensor([4095, 7, 2047, 1], device="cuda")
    gathered = swiglu(g[idx], u[idx])
    assert torch.equal(gathered, full[idx])
    assert torch.equal(swiglu(g[:16], u[:16]), full[:16])
    assert torch.equal(swiglu(g[100:102], u[100:102]), full[100:102])


@requires_cuda
def test_g2_ffn_batch_invariance_real_shape():
    rows = 4096
    x = _randn((rows, H), seed=30)
    w_gate, w_up, w_down = _ffn_weights(H, F, seed=31)
    gemm = TritonDetGemmAdapter()
    with torch.no_grad():
        full = swiglu_ffn(x, w_gate, w_up, w_down, gemm=gemm)
        assert torch.equal(full, swiglu_ffn(x, w_gate, w_up, w_down, gemm=gemm))
        for i in (0, 1, 2048, 4095):
            one = swiglu_ffn(x[i : i + 1], w_gate, w_up, w_down, gemm=gemm)
            assert torch.equal(one[0], full[i])
        idx = torch.tensor([4095, 1, 2048, 0], device="cuda")
        assert torch.equal(swiglu_ffn(x[idx], w_gate, w_up, w_down, gemm=gemm), full[idx])
        idx16 = torch.arange(37, 37 + 16, device="cuda")
        assert torch.equal(swiglu_ffn(x[idx16], w_gate, w_up, w_down, gemm=gemm), full[idx16])


# --- default GEMM adapter -------------------------------------------------------------------
@requires_cuda
def test_adapter_gemm_accuracy_invariance_and_strides():
    gemm = TritonDetGemmAdapter()
    a = _randn((100, 256), seed=40)
    w = _randn((96, 256), seed=41, scale=256**-0.5)
    ref = a.double() @ w.double().t()
    y = gemm(a, w)
    assert y.dtype == torch.bfloat16 and y.shape == (100, 96)
    atol, rtol = _tol("forward_accuracy", "reduction", "bfloat16")
    torch.testing.assert_close(y.double(), ref, atol=atol, rtol=rtol)
    y32 = gemm(a, w, out_dtype=torch.float32)
    atol, rtol = _tol("forward_accuracy", "reduction", "float32")
    torch.testing.assert_close(y32.double(), ref, atol=atol, rtol=rtol)
    assert torch.equal(y32, gemm(a, w, out_dtype=torch.float32))
    assert torch.equal(gemm(a[5:6], w)[0], y[5])
    assert torch.equal(gemm(a[3:80], w), y[3:80])
    # Transposed (strided) operands read through strides must equal the contiguous result.
    a_t = _randn((256, 100), seed=42).t()
    w_t = _randn((256, 96), seed=43, scale=256**-0.5).t()
    assert torch.equal(gemm(a_t, w_t), gemm(a_t.contiguous(), w_t.contiguous()))
    # K % 8 != 0 is out of contract but must still compute (single chunk).
    a36 = _randn((9, 36), seed=44)
    w36 = _randn((5, 36), seed=45)
    torch.testing.assert_close(
        gemm(a36, w36, out_dtype=torch.float32).double(), a36.double() @ w36.double().t(),
        atol=1e-4, rtol=1e-4,
    )


# --- FFN forward correctness ---------------------------------------------------------------
@requires_cuda
@pytest.mark.parametrize("shape", [(32, 64, 96), (5, 48, 80), (2, 32, 1024)])
def test_ffn_forward_small_vs_golden(shape):
    tokens, hidden, ffn = shape
    x = _randn((tokens, hidden), seed=50)
    w_gate, w_up, w_down = _ffn_weights(hidden, ffn, seed=51)
    y = swiglu_ffn(x, w_gate, w_up, w_down, gemm=TritonDetGemmAdapter())
    assert y.dtype == torch.bfloat16 and y.shape == (tokens, hidden)
    atol, rtol = _tol("forward_accuracy", "reduction", "bfloat16")
    golden = swiglu_ffn_fp32(x, w_gate, w_up, w_down)
    torch.testing.assert_close(y.float(), golden, atol=atol, rtol=rtol)
    # Leading dims are flattened and restored.
    y3 = swiglu_ffn(x.reshape(1, tokens, hidden), w_gate, w_up, w_down, gemm=TritonDetGemmAdapter())
    assert y3.shape == (1, tokens, hidden) and torch.equal(y3[0], y)


@requires_cuda
def test_ffn_forward_real_shape_vs_fp64():
    tokens = 64
    x = _randn((tokens, H), seed=60)
    w_gate, w_up, w_down = _ffn_weights(H, F, seed=61)
    with torch.no_grad():
        y = swiglu_ffn(x, w_gate, w_up, w_down, gemm=default_gemm())
    ref = swiglu_ffn_fp32(x, w_gate, w_up, w_down)  # fp64 compute, fp32 return
    _report_error("FFN real shape T=64 bf16 vs fp64", y, ref)
    atol, rtol = _tol("forward_accuracy", "reduction", "bfloat16")
    torch.testing.assert_close(y.float(), ref, atol=atol, rtol=rtol)


# --- G4: backward --------------------------------------------------------------------------
@requires_cuda
def test_g4_swiglu_backward_repeat_and_reference():
    g = _randn((64, F), seed=70, scale=3.0)
    u = _randn((64, F), seed=71)
    dh = _randn((64, F), seed=72)

    def run():
        gg = g.clone().requires_grad_()
        uu = u.clone().requires_grad_()
        swiglu(gg, uu).backward(dh)
        return gg.grad, uu.grad

    dg, du = run()
    dg2, du2 = run()
    assert torch.equal(dg, dg2) and torch.equal(du, du2)
    assert dg.dtype == torch.bfloat16 and du.dtype == torch.bfloat16
    g32 = g.float().requires_grad_()
    u32 = u.float().requires_grad_()
    swiglu_fp32(g32, u32).backward(dh.float())
    atol, rtol = _tol("gradient_accuracy", "elementwise", "bfloat16")
    torch.testing.assert_close(dg.float(), g32.grad, atol=atol, rtol=rtol)
    torch.testing.assert_close(du.float(), u32.grad, atol=atol, rtol=rtol)
    dg32, du32 = launch_swiglu_bwd(dh, g, u, out_dtype=torch.float32)
    atol, rtol = _tol("gradient_accuracy", "elementwise", "float32")
    torch.testing.assert_close(dg32, g32.grad, atol=atol, rtol=rtol)
    torch.testing.assert_close(du32, u32.grad, atol=atol, rtol=rtol)
    _report_error("G4 swiglu dgate bf16 vs fp32 autograd", dg, g32.grad)


def _ffn_grads(x, w_gate, w_up, w_down, dy, *, gemm, weight_grads: bool = True):
    leaves = [t.detach().clone().requires_grad_(flag) for t, flag in (
        (x, True), (w_gate, weight_grads), (w_up, weight_grads), (w_down, weight_grads),
    )]
    y = swiglu_ffn(*leaves, gemm=gemm)
    y.backward(dy)
    return y.detach(), [leaf.grad for leaf in leaves]


def _ffn_grads_fp32_reference(x, w_gate, w_up, w_down, dy):
    leaves = [t.detach().float().requires_grad_() for t in (x, w_gate, w_up, w_down)]
    xx, wg, wu, wd = leaves
    g = xx @ wg.t()
    u = xx @ wu.t()
    h = (g / (1.0 + torch.exp(-g))) * u
    y = h @ wd.t()
    y.backward(dy.float())
    return y.detach(), [leaf.grad for leaf in leaves]


@requires_cuda
@pytest.mark.parametrize("shape", [(32, 64, 96), (24, 40, 56)])
def test_g4_ffn_backward_small_repeat_and_reference(shape):
    tokens, hidden, ffn = shape
    x = _randn((tokens, hidden), seed=80)
    w_gate, w_up, w_down = _ffn_weights(hidden, ffn, seed=81)
    dy = _randn((tokens, hidden), seed=82, scale=tokens**-0.5)
    gemm = TritonDetGemmAdapter()
    y, grads = _ffn_grads(x, w_gate, w_up, w_down, dy, gemm=gemm)
    y2, grads2 = _ffn_grads(x, w_gate, w_up, w_down, dy, gemm=gemm)
    assert torch.equal(y, y2)
    for got, again in zip(grads, grads2, strict=True):
        assert got.dtype == torch.bfloat16 and torch.equal(got, again)
    _, refs = _ffn_grads_fp32_reference(x, w_gate, w_up, w_down, dy)
    atol, rtol = _tol("gradient_accuracy", "reduction", "bfloat16")
    for name, got, ref in zip(("dx", "dw_gate", "dw_up", "dw_down"), grads, refs, strict=True):
        torch.testing.assert_close(got.float(), ref, atol=atol, rtol=rtol, msg=name)
    # Frozen weights (LoRA-style): only dx is produced, bitwise equal to the full-backward dx.
    _, frozen = _ffn_grads(x, w_gate, w_up, w_down, dy, gemm=gemm, weight_grads=False)
    assert torch.equal(frozen[0], grads[0]) and all(g is None for g in frozen[1:])


@requires_cuda
def test_g4_ffn_forward_backward_real_shape_repeat_determinism():
    tokens = 256
    x = _randn((tokens, H), seed=90)
    w_gate, w_up, w_down = _ffn_weights(H, F, seed=91)
    dy = _randn((tokens, H), seed=92, scale=tokens**-0.5)
    gemm = default_gemm()
    y, grads = _ffn_grads(x, w_gate, w_up, w_down, dy, gemm=gemm)
    y2, grads2 = _ffn_grads(x, w_gate, w_up, w_down, dy, gemm=gemm)
    assert torch.equal(y, y2)
    for got, again in zip(grads, grads2, strict=True):
        assert torch.isfinite(got).all()
        assert torch.equal(got, again)
    assert grads[0].shape == (tokens, H)
    assert grads[1].shape == (F, H) and grads[2].shape == (F, H) and grads[3].shape == (H, F)


# --- real layer-0 MLP weights ----------------------------------------------------------------
@requires_cuda
@requires_checkpoint
def test_real_layer0_mlp_weights_forward():
    from safetensors import safe_open

    weight_map = json.loads((CKPT / "model.safetensors.index.json").read_text())["weight_map"]
    weights = {}
    for name, key in LAYER0_MLP.items():
        with safe_open(CKPT / weight_map[key], framework="pt", device="cuda") as shard:
            weights[name] = shard.get_tensor(key)
    w_gate, w_up, w_down = weights["gate_proj"], weights["up_proj"], weights["down_proj"]
    assert w_gate.shape == (F, H) and w_up.shape == (F, H) and w_down.shape == (H, F)
    assert {w.dtype for w in weights.values()} == {torch.bfloat16}
    x = _randn((8, H), seed=100)
    with torch.no_grad():
        y = swiglu_ffn(x, w_gate, w_up, w_down)
    assert y.shape == (8, H) and y.dtype == torch.bfloat16 and torch.isfinite(y).all()
    ref = swiglu_ffn_fp32(x, w_gate, w_up, w_down)
    _report_error("real layer-0 MLP weights, T=8, bf16 vs fp64", y, ref)
    atol, rtol = _tol("forward_accuracy", "reduction", "bfloat16")
    torch.testing.assert_close(y.float(), ref, atol=atol, rtol=rtol)
    with torch.no_grad():
        native = torch.nn.functional.linear(
            torch.nn.functional.silu(torch.nn.functional.linear(x, w_gate))
            * torch.nn.functional.linear(x, w_up),
            w_down,
        )
    _report_error("real layer-0 MLP weights, HF-style bf16 cuBLAS vs fp64 (reference)", native, ref)
