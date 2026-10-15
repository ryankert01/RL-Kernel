# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Contract gates G1-G5 for ``fast_det_gemm`` (Muse-Glimmer-30B O3, row dense_gemm_qualification).

G1 correctness vs an fp64 reference within the ws1 reduction tolerances; G2 batch invariance
(decode M=16 vs M=64 vs prefill M=4096, chunked prefill, padding) with ``torch.equal``; TP1 vs
simulated TP2/4/8 (K sliced on one GPU, partials combined with ``combine_tree``) bitwise; G4
backward repeatability and accuracy vs fp32 autograd; G5 ``trace()``.  Small shapes give breadth,
every Muse Glimmer GEMM shape (incl. the padded lm_head, N=202752) is covered at least once.
"""

from __future__ import annotations

import json

import pytest
import torch

from rl_engine.contracts.numerical import load_contract
from rl_engine.models.muse_glimmer.ops import det_gemm as dg
from rl_engine.models.muse_glimmer.ops.canonical_tree import (
    CANONICAL_K_CHUNKS,
    combine_tree,
    rank_chunk_ids,
)
from rl_engine.models.muse_glimmer.triton import det_gemm_tree as kt

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 8,
    reason="fast_det_gemm requires CUDA SM80+ (bf16 mma.sync)",
)

DEV = "cuda"
BF16 = torch.bfloat16
FP32 = torch.float32

# (N, K) of every Muse Glimmer GEMM (contract O3); lm_head uses the padded vocab.
MUSE_SHAPES: dict[str, tuple[int, int]] = {
    "q_proj": (4096, 6656),
    "k_proj": (256, 6656),
    "v_proj": (256, 6656),
    "attn_gate_proj": (4096, 6656),
    "o_proj": (6656, 4096),
    "mlp_gate_proj": (19968, 6656),
    "up_proj": (19968, 6656),
    "down_proj": (6656, 19968),
    "lm_head": (202752, 6656),
}
MUSE_NAMES = tuple(MUSE_SHAPES)
# (M, N, K): ragged M, N not a multiple of the tile, every K a multiple of 256.
SMALL_SHAPES = [(1, 64, 256), (16, 96, 512), (100, 192, 1024), (257, 320, 2048)]

_CONTRACT = load_contract()
_FWD_TOL = _CONTRACT["accuracy"]["default"]["reduction"]
_GRAD_TOL = _CONTRACT["judgments"]["gradient_accuracy"]["by_op_class"]["reduction"]


def _rand(*shape: int, scale: float = 1.0, seed: int | None = None) -> torch.Tensor:
    if seed is not None:
        torch.manual_seed(seed)
    return (torch.randn(*shape, device=DEV) * scale).to(BF16)


def _operands(m_size: int, n_size: int, k_size: int, seed: int = 0):
    """Activations ~N(0,1), weights ~N(0, 0.02^2) as in the real model."""

    torch.manual_seed(seed)
    return _rand(m_size, k_size), _rand(n_size, k_size, scale=0.02)


def _assert_close(actual: torch.Tensor, ref: torch.Tensor, tol: dict, what: str) -> None:
    torch.testing.assert_close(
        actual.float(),
        ref.float(),
        atol=tol["atol"],
        rtol=tol["rtol"],
        msg=lambda m: f"{what}: {m}",
    )


def _big_m(name: str) -> int:
    return 512 if name == "lm_head" else 4096


# --------------------------------------------------------------------------------------- G1 ---


@pytest.mark.parametrize("shape", SMALL_SHAPES)
def test_g1_forward_correctness_small(shape):
    m_size, n_size, k_size = shape
    a, w = _operands(m_size, n_size, k_size, seed=1)
    ref = dg.forward_fp32(a, w)
    _assert_close(dg.fast_det_gemm(a, w), ref, _FWD_TOL["bfloat16"], "bf16 out")
    _assert_close(dg.fast_det_gemm(a, w, out_dtype=FP32), ref, _FWD_TOL["float32"], "fp32 out")


@pytest.mark.parametrize("name", MUSE_NAMES)
def test_g1_forward_correctness_muse_shape(name):
    n_size, k_size = MUSE_SHAPES[name]
    m_size = 256
    a, w = _operands(m_size, n_size, k_size, seed=2)
    ref = dg.forward_fp32(a, w)
    y32 = dg.fast_det_gemm(a, w, out_dtype=FP32)
    y16 = dg.fast_det_gemm(a, w)
    _assert_close(y32, ref, _FWD_TOL["float32"], f"{name} fp32 out")
    _assert_close(y16, ref, _FWD_TOL["bfloat16"], f"{name} bf16 out")
    # fp32 path and bf16 path share one accumulation; bf16 is a single final rounding.
    assert torch.equal(y32.to(BF16), y16)


def test_g1_forward_fp32_golden_is_fp64():
    a, w = _operands(8, 32, 256, seed=3)
    expected = (a.double() @ w.double().t()).float()
    assert torch.equal(dg.forward_fp32(a, w), expected)
    assert torch.equal(dg.forward_fp32(a, w.t().contiguous(), w_kn=True), expected)


# --------------------------------------------------------------------------------------- G2 ---


@pytest.mark.parametrize("name", MUSE_NAMES)
def test_g2_batch_invariance_decode_vs_prefill(name):
    """One logical row: decode M=16, decode M=64 (position 37), prefill (positions 0 and 2049)."""

    n_size, k_size = MUSE_SHAPES[name]
    big_m = _big_m(name)
    torch.manual_seed(4)
    w = _rand(n_size, k_size, scale=0.02)
    row = _rand(1, k_size)
    big = _rand(big_m, k_size)
    big[0] = row[0]
    big[big_m // 2 + 1] = row[0]
    dec16 = _rand(16, k_size)
    dec16[5] = row[0]
    dec64 = _rand(64, k_size)
    dec64[37] = row[0]
    y_big = dg.fast_det_gemm(big, w)
    y1 = dg.fast_det_gemm(row, w)
    y16 = dg.fast_det_gemm(dec16, w)
    y64 = dg.fast_det_gemm(dec64, w)
    assert torch.equal(y1[0], y_big[0]), f"{name}: M=1 vs prefill"
    assert torch.equal(y16[5], y_big[0]), f"{name}: decode16 vs prefill"
    assert torch.equal(y64[37], y_big[0]), f"{name}: decode64 vs prefill"
    assert torch.equal(y_big[big_m // 2 + 1], y_big[0]), f"{name}: batch position"


@pytest.mark.parametrize("shape", SMALL_SHAPES + [(300, 6656, 4096)])
def test_g2_chunked_prefill_and_padding(shape):
    m_size, n_size, k_size = shape
    a, w = _operands(m_size, n_size, k_size, seed=5)
    full = dg.fast_det_gemm(a, w)
    split = max(1, m_size // 3)
    chunked = torch.cat([dg.fast_det_gemm(a[:split], w), dg.fast_det_gemm(a[split:], w)])
    assert torch.equal(full, chunked), "chunked prefill changed bits"
    padded = dg.fast_det_gemm(torch.cat([a, _rand(29, k_size)]), w)
    assert torch.equal(full, padded[:m_size]), "padding rows changed valid rows"


def test_g2_leading_dims_are_flattened():
    a, w = _operands(6 * 7, 128, 512, seed=6)
    y3 = dg.fast_det_gemm(a.view(6, 7, 512), w)
    assert y3.shape == (6, 7, 128)
    assert torch.equal(y3.reshape(42, 128), dg.fast_det_gemm(a, w))


_TILE_VARIANTS = [
    kt.TileConfig(16, 64, 64, 4, 4),  # decode16 pinned
    kt.TileConfig(32, 64, 64, 4, 4),  # decode64 pinned
    kt.TileConfig(64, 128, 64, 8, 3),  # prefill pinned
    kt.TileConfig(128, 64, 64, 8, 3),
    kt.TileConfig(64, 64, 32, 4, 4),  # BLOCK_K = 32
    kt.TileConfig(64, 64, 64, 8, 2),
]


@pytest.mark.parametrize(
    "cfg", _TILE_VARIANTS, ids=lambda c: f"{c.block_m}x{c.block_n}x{c.block_k}w{c.num_warps}"
)
def test_g2_tile_constants_do_not_change_bits(cfg):
    """Tile constants are performance-only: the MMA chain per chunk and the tree are fixed."""

    a, w = _operands(256, 4096, 6656, seed=7)
    pinned = dg.fast_det_gemm(a, w, out_dtype=FP32)
    variant = kt.tree_gemm(a, w.t(), out_dtype=FP32, config=cfg)
    assert torch.equal(pinned, variant)


def test_g2_weight_layout_kn_is_bitwise():
    a, w = _operands(256, 6656, 4096, seed=8)
    y_nk = dg.fast_det_gemm(a, w, out_dtype=FP32)
    y_kn = dg.fast_det_gemm(a, w.t().contiguous(), w_kn=True, out_dtype=FP32)
    assert torch.equal(y_nk, y_kn)


def test_repeat_determinism_muse_shape():
    a, w = _operands(4096, 6656, 19968, seed=9)
    first = dg.fast_det_gemm(a, w)
    _ = torch.randn(4096, 4096, device=DEV)  # perturb allocator / scheduling between runs
    second = dg.fast_det_gemm(a, w)
    assert torch.equal(first, second)


# ------------------------------------------------------------------------ simulated TP tree ---


def _simulated_tp(a: torch.Tensor, w: torch.Tensor, tp_size: int) -> torch.Tensor:
    """Row-parallel TP on one GPU: slice K per rank, sub-tree partials, canonical combine."""

    chunk = a.size(1) // CANONICAL_K_CHUNKS
    partials = []
    for rank in range(tp_size):
        ids = rank_chunk_ids(tp_size, rank)
        lo, hi = ids[0] * chunk, (ids[-1] + 1) * chunk
        partials.append(dg.fast_det_gemm_partial(a[:, lo:hi], w[:, lo:hi], ids))
    return combine_tree(partials)


@pytest.mark.parametrize("tp_size", [2, 4, 8])
@pytest.mark.parametrize("name", MUSE_NAMES)
def test_tp_simulated_matches_tp1_bitwise(name, tp_size):
    n_size, k_size = MUSE_SHAPES[name]
    a, w = _operands(256, n_size, k_size, seed=10)
    tp1_fp32 = dg.fast_det_gemm(a, w, out_dtype=FP32)
    tp1_bf16 = dg.fast_det_gemm(a, w)
    combined = _simulated_tp(a, w, tp_size)
    assert torch.equal(combined, tp1_fp32), f"{name}: TP{tp_size} fp32 tree != TP1"
    assert torch.equal(combined.to(BF16), tp1_bf16), f"{name}: TP{tp_size} bf16 != TP1"


@pytest.mark.parametrize("tp_size", [2, 4, 8])
def test_tp_partial_is_batch_invariant(tp_size):
    n_size, k_size = MUSE_SHAPES["down_proj"]
    torch.manual_seed(11)
    w = _rand(n_size, k_size, scale=0.02)
    row = _rand(1, k_size)
    big = _rand(256, k_size)
    big[77] = row[0]
    assert torch.equal(_simulated_tp(row, w, tp_size)[0], _simulated_tp(big, w, tp_size)[77])


def test_tp8_left_fold_is_not_the_tree():
    """Pins that the tree is load-bearing: a left fold of the 8 chunk partials differs."""

    a, w = _operands(256, 6656, 4096, seed=12)
    chunk = a.size(1) // CANONICAL_K_CHUNKS
    partials = []
    for c in range(CANONICAL_K_CHUNKS):
        cols = slice(c * chunk, (c + 1) * chunk)
        partials.append(dg.fast_det_gemm_partial(a[:, cols], w[:, cols], [c]))
    tree = combine_tree(partials)
    folded = partials[0]
    for part in partials[1:]:
        folded = folded + part
    assert torch.equal(tree, dg.fast_det_gemm(a, w, out_dtype=FP32))
    assert not torch.equal(tree, folded)


def test_combine_tree_order_and_validation():
    torch.manual_seed(13)
    parts = [torch.randn(1000, device=DEV) for _ in range(8)]
    p = parts
    expected8 = ((p[0] + p[1]) + (p[2] + p[3])) + ((p[4] + p[5]) + (p[6] + p[7]))
    assert torch.equal(combine_tree(parts), expected8)
    assert torch.equal(combine_tree(parts[:4]), (p[0] + p[1]) + (p[2] + p[3]))
    assert torch.equal(combine_tree(parts[:2]), p[0] + p[1])
    assert combine_tree(parts[:1]) is parts[0]
    # TP2 over TP4 sub-trees is the same tree.
    assert torch.equal(combine_tree([combine_tree(parts[:4]), combine_tree(parts[4:])]), expected8)
    with pytest.raises(ValueError):
        combine_tree(parts[:3])
    with pytest.raises(TypeError):
        combine_tree([x.to(BF16) for x in parts])
    assert rank_chunk_ids(2, 1) == (4, 5, 6, 7)
    assert rank_chunk_ids(8, 3) == (3,)
    with pytest.raises(ValueError):
        dg.fast_det_gemm_partial(_rand(4, 512), _rand(8, 512), [1, 2])  # not aligned
    with pytest.raises(ValueError):
        dg.fast_det_gemm_partial(_rand(4, 512), _rand(8, 512), [0, 1, 2])  # not a power of two


# --------------------------------------------------------------------------------------- G4 ---


def _fp32_autograd_reference(a, w, grad_y, *, w_kn=False):
    a32 = a.detach().float().requires_grad_(True)
    w32 = w.detach().float().requires_grad_(True)
    y = a32 @ w32 if w_kn else a32 @ w32.t()
    y.backward(grad_y.float())
    return a32.grad, w32.grad


@pytest.mark.parametrize(
    "shape",
    [(16, 96, 512), (100, 192, 1024), (256, 6656, 4096), (256, 256, 6656), (4096, 256, 6656)],
    ids=["ragged16", "ragged100", "o_proj", "k_proj", "k_proj_m4096"],
)
def test_g4_backward_repeatable_and_accurate(shape):
    m_size, n_size, k_size = shape
    a, w = _operands(m_size, n_size, k_size, seed=14)
    grad_y = _rand(m_size, n_size)
    grads = []
    for _ in range(2):
        a_ = a.clone().requires_grad_(True)
        w_ = w.clone().requires_grad_(True)
        dg.fast_det_gemm(a_, w_).backward(grad_y)
        grads.append((a_.grad, w_.grad))
    assert torch.equal(grads[0][0], grads[1][0]), "dA not repeatable"
    assert torch.equal(grads[0][1], grads[1][1]), "dW not repeatable"
    ref_da, ref_dw = _fp32_autograd_reference(a, w, grad_y)
    _assert_close(grads[0][0], ref_da, _GRAD_TOL["bfloat16"], "dA")
    _assert_close(grads[0][1], ref_dw, _GRAD_TOL["bfloat16"], "dW")


def test_g4_backward_w_kn_layout():
    torch.manual_seed(15)
    a = _rand(128, 1024)
    w_kn = _rand(1024, 384, scale=0.02)
    grad_y = _rand(128, 384)
    a_ = a.clone().requires_grad_(True)
    w_ = w_kn.clone().requires_grad_(True)
    dg.fast_det_gemm(a_, w_, w_kn=True).backward(grad_y)
    ref_da, ref_dw = _fp32_autograd_reference(a, w_kn, grad_y, w_kn=True)
    assert w_.grad.shape == w_kn.shape
    _assert_close(a_.grad, ref_da, _GRAD_TOL["bfloat16"], "dA (w_kn)")
    _assert_close(w_.grad, ref_dw, _GRAD_TOL["bfloat16"], "dW (w_kn)")
    # Same arithmetic as the (N, K) layout: dW bits agree after a transpose.
    a2 = a.clone().requires_grad_(True)
    w2 = w_kn.t().contiguous().requires_grad_(True)
    dg.fast_det_gemm(a2, w2).backward(grad_y)
    assert torch.equal(a2.grad, a_.grad)
    assert torch.equal(w2.grad.t(), w_.grad)


def test_g4_da_is_batch_invariant():
    """dA reduces over N with the canonical rule, so it is batch-invariant like the forward."""

    n_size, k_size = MUSE_SHAPES["q_proj"]
    torch.manual_seed(16)
    w = _rand(n_size, k_size, scale=0.02)
    grad_row = _rand(1, n_size)
    grad_big = _rand(300, n_size)
    grad_big[123] = grad_row[0]
    row = _rand(1, k_size).requires_grad_(True)
    big = _rand(300, k_size).requires_grad_(True)
    dg.fast_det_gemm(row, w).backward(grad_row)
    dg.fast_det_gemm(big, w).backward(grad_big)
    assert torch.equal(row.grad[0], big.grad[123])


def test_g4_dw_ragged_rule_equals_zero_padded_tokens():
    """dW for M=16 tokens equals dW for the same tokens zero-padded to M=256 (x + 0 == x)."""

    n_size, k_size = 384, 1024
    a, w = _operands(16, n_size, k_size, seed=17)
    grad_y = _rand(16, n_size)
    w1 = w.clone().requires_grad_(True)
    dg.fast_det_gemm(a, w1).backward(grad_y)
    a_pad = torch.zeros(256, k_size, device=DEV, dtype=BF16)
    g_pad = torch.zeros(256, n_size, device=DEV, dtype=BF16)
    a_pad[:16], g_pad[:16] = a, grad_y
    w2 = w.clone().requires_grad_(True)
    dg.fast_det_gemm(a_pad, w2).backward(g_pad)
    assert torch.equal(w1.grad, w2.grad)
    assert kt.plan_chunks(16, allow_pad=True) == kt.ChunkPlan(16, 256, 8, 32, 32, True)


def test_g4_fp32_output_path_backward():
    """lm_head path: fp32 logits out, fp32 upstream grad cast once to bf16 for the GEMMs."""

    n_size, k_size = 25344, 6656  # one canonical vocab tile
    a, w = _operands(64, n_size, k_size, seed=18)
    grad_y = torch.randn(64, n_size, device=DEV, dtype=FP32)
    outs = []
    for _ in range(2):
        a_ = a.clone().requires_grad_(True)
        w_ = w.clone().requires_grad_(True)
        y = dg.fast_det_gemm(a_, w_, out_dtype=FP32)
        assert y.dtype == FP32
        y.backward(grad_y)
        outs.append((a_.grad, w_.grad))
    assert torch.equal(outs[0][0], outs[1][0]) and torch.equal(outs[0][1], outs[1][1])
    ref_da, ref_dw = _fp32_autograd_reference(a, w, grad_y.to(BF16))
    _assert_close(outs[0][0], ref_da, _GRAD_TOL["bfloat16"], "dA (fp32 out)")
    _assert_close(outs[0][1], ref_dw, _GRAD_TOL["bfloat16"], "dW (fp32 out)")


# --------------------------------------------------------------------------------------- G5 ---


def test_g5_trace_is_emitted_and_stable():
    tr = dg.trace()
    for key in ("op", "dtype_in", "dtype_acc", "dtype_out", "reduction_order", "tile_config"):
        assert key in tr, key
    assert tr["dtype_in"] == "bfloat16" and tr["dtype_acc"] == "float32"
    assert tr["k_chunks"] == 8 and tr["tree_order"] == "((0+1)+(2+3))+((4+5)+(6+7))"
    assert tr["split_k"] is False and tr["autotune"] is False and tr["allow_tf32"] is False
    assert set(tr["tile_config"]) == {"decode16", "decode64", "prefill"}
    assert tr == dg.trace()
    json.dumps(tr)  # serialisable provenance
    assert list(dg.trace(16)["tile_config"]) == ["decode16"]
    assert list(dg.trace(64)["tile_config"]) == ["decode64"]
    assert list(dg.trace(4096)["tile_config"]) == ["prefill"]
    assert dg.trace(4096, out_dtype=FP32)["dtype_out"] == "float32"
    a, w = _operands(4096, 256, 6656, seed=19)
    dg.fast_det_gemm(a, w)
    launch = kt.last_launch()
    assert launch["shape_class"] == "prefill"
    assert launch["config"] == dg.trace(4096)["tile_config"]["prefill"]
    assert launch["plan"]["block_k"] == 64 and launch["n_spills"] is not None


# -------------------------------------------------------------------------------- contract ---


def test_contract_rejects_out_of_contract_inputs():
    a, w = _operands(8, 64, 512, seed=20)
    with pytest.raises(ValueError, match="not a multiple of 256"):
        dg.fast_det_gemm(_rand(8, 500), _rand(64, 500))
    with pytest.raises(TypeError):
        dg.fast_det_gemm(a.float(), w)
    with pytest.raises(TypeError):
        dg.fast_det_gemm(a, w, out_dtype=torch.float16)
    with pytest.raises(ValueError):
        dg.fast_det_gemm(a, w, k_chunks=3)
    with pytest.raises(ValueError, match="K mismatch"):
        dg.fast_det_gemm(a, _rand(64, 256))
    with pytest.raises(ValueError, match="K mismatch"):
        dg.fast_det_gemm(a, w, w_kn=True)
    with pytest.raises(ValueError):
        kt.plan_chunks(500)
    assert kt.plan_chunks(6656) == kt.ChunkPlan(6656, 6656, 8, 832, 64, False)
    assert kt.plan_chunks(256) == kt.ChunkPlan(256, 256, 8, 32, 32, False)
    assert kt.plan_chunks(19968).block_k == 64 and kt.plan_chunks(4096).block_k == 64
    assert kt.plan_chunks(202752).block_k == 64
