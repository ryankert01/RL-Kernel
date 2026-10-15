# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Contract gates for the strict Muse-Glimmer gated attention backward (O5, rows
``global_gated_attention`` / ``local_gated_attention_train``).

G4  deterministic repeat (torch.equal) and accuracy vs the FP64 autograd reference;
G2  batch invariance of dq/dgate rows and of dk/dv (alone vs packed, several positions);
TP  the 8 canonical Q-head-group partials combined by the fixed tree equal the TP2/4/8
    sub-tree combination bit for bit;
G5  trace() emitted and stable.

Accuracy tolerance: the ws1 ``gradient_accuracy/attention/bfloat16`` row (atol 5e-2,
rtol 2e-2) is calibrated for O(1) tensors, while the magnitude of an attention gradient is
set by the arbitrary scale of ``dy`` (dK sums 16 heads x up to 4096 queries; with unit dy its
max is ~26 and the contract-mandated bf16 OUTPUT rounding alone is 0.0625 there). The row is
therefore applied at unit scale (both tensors divided by max|reference|), and a stricter
explicit bound max|err| / max|ref| <= 1e-2 (observed <= 6e-3, FA2 bf16 level) is asserted
on top of it. Forward accuracy of the fallback golden uses the ``forward_accuracy`` row.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
pytestmark = [pytest.mark.ws1, requires_cuda]

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

_PROFILE = Path(__file__).resolve().parents[3] / "rl_engine/contracts/profiles/precision/ws1.json"
NORMALIZED_BOUND = 1e-2
WINDOW = 2048


def _tol(judgment: str, op_class: str = "attention") -> tuple[float, float]:
    row = json.loads(_PROFILE.read_text())["judgments"][judgment]["by_op_class"][op_class]
    row = row["bfloat16"]
    return float(row["atol"]), float(row["rtol"])


@pytest.fixture(scope="module")
def mods():
    from rl_engine.models.muse_glimmer.ops import attention_golden_bwd as gold
    from rl_engine.models.muse_glimmer.ops import gated_attention_autograd as ag
    from rl_engine.models.muse_glimmer.triton import gated_attention_bwd as bwd

    return gold, ag, bwd


def make_inputs(lens: list[int], *, seed: int = 0, pos0: list[int] | None = None, k_lens=None):
    """Random bf16 inputs at the real per-token shapes; q rms-normalised and scaled by 3.87."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    tq = sum(lens)
    k_lens = list(lens) if k_lens is None else list(k_lens)
    tk = sum(k_lens)
    q = torch.randn(tq, 32, 128, generator=g)
    q = q * torch.rsqrt(q.pow(2).mean(-1, keepdim=True) + 1e-5) * 3.87
    k = torch.randn(tk, 2, 128, generator=g)
    k = k * torch.rsqrt(k.pow(2).mean(-1, keepdim=True) + 1e-5)
    v = torch.randn(tk, 2, 128, generator=g)
    gate = torch.randn(tq, 4096, generator=g)
    dy = torch.randn(tq, 4096, generator=g)
    cu_q = torch.tensor([0] + torch.tensor(lens).cumsum(0).tolist(), dtype=torch.int32)
    cu_k = torch.tensor([0] + torch.tensor(k_lens).cumsum(0).tolist(), dtype=torch.int32)
    pos0 = [0] * len(lens) if pos0 is None else pos0
    pos = torch.cat([torch.arange(p, p + n, dtype=torch.int32) for p, n in zip(pos0, lens)])
    dev = torch.device("cuda")
    return dict(
        q=q.to(dev, torch.bfloat16),
        k=k.to(dev, torch.bfloat16),
        v=v.to(dev, torch.bfloat16),
        gate=gate.to(dev, torch.bfloat16),
        dy=dy.to(dev, torch.bfloat16),
        cu_q=cu_q.to(dev),
        cu_k=cu_k.to(dev),
        pos=pos.to(dev),
    )


def strict_bwd(mods, inp, window):
    gold, _ag, bwd = mods
    _y, o, lse = gold.golden_forward_fixed_order(
        inp["q"], inp["k"], inp["v"], inp["gate"], inp["cu_q"], inp["cu_k"], inp["pos"],
        window=window,
    )
    grads = bwd.strict_gated_attention_bwd(
        inp["q"], inp["k"], inp["v"], inp["gate"], o, lse, inp["dy"], inp["cu_q"], inp["cu_k"],
        inp["pos"], window=window,
    )
    return grads, (o, lse)


def _check_accuracy(name, got: torch.Tensor, ref: torch.Tensor) -> float:
    got = got.float()
    ref = ref.float()
    assert not torch.isnan(got).any(), f"{name} has NaN"
    scale = ref.abs().max().clamp_min(1e-12)
    norm_err = float(((got - ref).abs().max() / scale).item())
    assert norm_err <= NORMALIZED_BOUND, f"{name}: normalized max error {norm_err:.3e}"
    atol, rtol = _tol("gradient_accuracy")
    assert torch.allclose(got / scale, ref / scale, atol=atol, rtol=rtol), (
        f"{name}: ws1 attention gradient row failed at unit scale"
    )
    return norm_err


@pytest.mark.parametrize("window", [None, WINDOW], ids=["global", "local"])
@pytest.mark.parametrize("length", [64, 2047, 2048, 2049, 4096])
def test_backward_matches_reference(mods, window, length):
    gold, _ag, _bwd = mods
    inp = make_inputs([length], seed=length)
    (dq, dk, dv, dgate), _ = strict_bwd(mods, inp, window)
    rq, rk, rv, rg = gold.reference_backward(
        inp["q"], inp["k"], inp["v"], inp["gate"], inp["cu_q"], inp["cu_k"], inp["pos"],
        inp["dy"], window=window,
    )
    errs = {
        "dq": _check_accuracy("dq", dq, rq),
        "dk": _check_accuracy("dk", dk, rk),
        "dv": _check_accuracy("dv", dv, rv),
        "dgate": _check_accuracy("dgate", dgate, rg),
    }
    print(f"\n[accuracy window={window} L={length}] normalized max errors: {errs}")


@pytest.mark.parametrize("window", [None, WINDOW], ids=["global", "local"])
def test_backward_matches_reference_packed_offset_positions(mods, window):
    """Packed batch + chunked-prefill style layout (queries at positions 200.., keys 0..Lk)."""
    gold, _ag, _bwd = mods
    inp = make_inputs([100, 70, 2100], seed=7, pos0=[200, 0, 0], k_lens=[300, 70, 2100])
    (dq, dk, dv, dgate), _ = strict_bwd(mods, inp, window)
    rq, rk, rv, rg = gold.reference_backward(
        inp["q"], inp["k"], inp["v"], inp["gate"], inp["cu_q"], inp["cu_k"], inp["pos"],
        inp["dy"], window=window,
    )
    for name, got, ref in (("dq", dq, rq), ("dk", dk, rk), ("dv", dv, rv), ("dgate", dgate, rg)):
        _check_accuracy(name, got, ref)


@pytest.mark.parametrize("window", [None, WINDOW], ids=["global", "local"])
def test_repeat_determinism(mods, window):
    _gold, _ag, bwd = mods
    inp = make_inputs([2049, 64, 300], seed=3)
    (dq, dk, dv, dgate), (o, lse) = strict_bwd(mods, inp, window)
    dgate2, do2, delta2 = bwd.gated_attention_bwd_preprocess(inp["dy"], o, inp["gate"])
    dq2, dkp2, dvp2 = bwd.strict_gated_attention_bwd_partials(
        inp["q"], inp["k"], inp["v"], do2, lse, delta2, inp["cu_q"], inp["cu_k"], inp["pos"],
        window=window,
    )
    dq3, dk3, dv3, dgate3 = bwd.strict_gated_attention_bwd(
        inp["q"], inp["k"], inp["v"], inp["gate"], o, lse, inp["dy"], inp["cu_q"], inp["cu_k"],
        inp["pos"], window=window,
    )
    assert torch.equal(dq, dq2) and torch.equal(dq, dq3)
    assert torch.equal(dgate, dgate2) and torch.equal(dgate, dgate3)
    assert torch.equal(dk, bwd.combine_kv_partials(dkp2)) and torch.equal(dk, dk3)
    assert torch.equal(dv, bwd.combine_kv_partials(dvp2)) and torch.equal(dv, dv3)
    assert dkp2.shape == (8, inp["k"].shape[0], 2, 128) and dkp2.dtype == torch.float32


@pytest.mark.parametrize("window", [None, WINDOW], ids=["global", "local"])
def test_batch_invariance(mods, window):
    """Sequence A alone vs packed at batch positions 1 and 2 with different neighbours."""
    _gold, _ag, bwd = mods
    len_a, seed_a = 777, 11
    alone = make_inputs([len_a], seed=seed_a)
    (dq_a, dk_a, dv_a, dgate_a), (o_a, lse_a) = strict_bwd(mods, alone, window)
    _dgate, do_a, delta_a = bwd.gated_attention_bwd_preprocess(alone["dy"], o_a, alone["gate"])
    _dq, dkp_a, dvp_a = bwd.strict_gated_attention_bwd_partials(
        alone["q"], alone["k"], alone["v"], do_a, lse_a, delta_a, alone["cu_q"], alone["cu_k"],
        alone["pos"], window=window,
    )
    for neighbours, slot in (([130, 2049], 1), ([2049, 64, 5], 3)):
        lens = list(neighbours)
        lens.insert(slot, len_a)
        packed = make_inputs(lens, seed=100 + slot)
        start = sum(lens[:slot])
        sl = slice(start, start + len_a)
        for name in ("q", "gate", "dy"):
            packed[name][sl] = alone[name]
        for name in ("k", "v"):
            packed[name][sl] = alone[name]
        (dq_p, dk_p, dv_p, dgate_p), (o_p, lse_p) = strict_bwd(mods, packed, window)
        assert torch.equal(o_p[sl], o_a) and torch.equal(lse_p[sl], lse_a)
        assert torch.equal(dq_p[sl], dq_a), f"dq rows differ when packed at slot {slot}"
        assert torch.equal(dgate_p[sl], dgate_a), f"dgate rows differ at slot {slot}"
        assert torch.equal(dk_p[sl], dk_a), f"dk rows differ at slot {slot}"
        assert torch.equal(dv_p[sl], dv_a), f"dv rows differ at slot {slot}"
        _dg, do_p, delta_p = bwd.gated_attention_bwd_preprocess(packed["dy"], o_p, packed["gate"])
        _dq, dkp_p, dvp_p = bwd.strict_gated_attention_bwd_partials(
            packed["q"], packed["k"], packed["v"], do_p, lse_p, delta_p, packed["cu_q"],
            packed["cu_k"], packed["pos"], window=window,
        )
        assert torch.equal(dkp_p[:, sl], dkp_a) and torch.equal(dvp_p[:, sl], dvp_a)


def _tree(leaves: list[torch.Tensor]) -> torch.Tensor:
    """Fixed binary tree by recursive halving: 8 leaves -> ((0+1)+(2+3))+((4+5)+(6+7))."""
    if len(leaves) == 1:
        return leaves[0]
    half = len(leaves) // 2
    return _tree(leaves[:half]) + _tree(leaves[half:])


@pytest.mark.parametrize("window", [None, WINDOW], ids=["global", "local"])
def test_tp_simulation_group_partials(mods, window):
    """TP1 tree over the 8 group partials == per-rank sub-trees + tree over ranks (TP2/4/8)."""
    _gold, _ag, bwd = mods
    inp = make_inputs([1500, 600], seed=5)
    (_dq, dk, dv, _dgate), (o, lse) = strict_bwd(mods, inp, window)
    _dgate, do, delta = bwd.gated_attention_bwd_preprocess(inp["dy"], o, inp["gate"])
    _dq, dkp, dvp = bwd.strict_gated_attention_bwd_partials(
        inp["q"], inp["k"], inp["v"], do, lse, delta, inp["cu_q"], inp["cu_k"], inp["pos"],
        window=window,
    )
    for part, combined in ((dkp, dk), (dvp, dv)):
        # group g only touches KV head (4g)//16; the other slice is exactly zero
        for g in range(8):
            other = 1 - (4 * g) // 16
            assert torch.count_nonzero(part[g, :, other]) == 0
        full = _tree([part[g] for g in range(8)])
        assert torch.equal(full.to(torch.bfloat16), combined)
        for tp in (2, 4, 8):
            per_rank = 8 // tp
            rank_partials = [
                _tree([part[g] for g in range(r * per_rank, (r + 1) * per_rank)])
                for r in range(tp)
            ]
            assert torch.equal(_tree(rank_partials), full), f"TP{tp} tree differs from TP1"


@pytest.mark.parametrize("window", [None, WINDOW], ids=["global", "local"])
def test_autograd_function_binds_strict_backward(mods, window):
    gold, ag, _bwd = mods
    inp = make_inputs([300, 64], seed=9)
    q = inp["q"].clone().requires_grad_(True)
    k = inp["k"].clone().requires_grad_(True)
    v = inp["v"].clone().requires_grad_(True)
    gate = inp["gate"].clone().requires_grad_(True)
    y = ag.gated_attention(q, k, v, gate, inp["cu_q"], inp["cu_k"], inp["pos"], window)
    assert y.dtype == torch.bfloat16 and y.shape == (364, 4096)
    y.backward(inp["dy"])
    y_ref, o, lse = gold.golden_forward_fixed_order(
        inp["q"], inp["k"], inp["v"], inp["gate"], inp["cu_q"], inp["cu_k"], inp["pos"],
        window=window,
    )
    if ag.trace()["forward_is_fallback"]:
        assert torch.equal(y.detach(), y_ref)
    else:
        atol, rtol = _tol("forward_accuracy")
        assert torch.allclose(y.detach().float(), y_ref.float(), atol=atol, rtol=rtol)
    dq, dk, dv, dgate = ag.gated_attention_backward(
        inp["q"], inp["k"], inp["v"], inp["gate"], o, lse, inp["dy"], inp["cu_q"], inp["cu_k"],
        inp["pos"], window=window,
    )
    assert torch.equal(q.grad, dq) and torch.equal(k.grad, dk)
    assert torch.equal(v.grad, dv) and torch.equal(gate.grad, dgate)
    assert q.grad.dtype == k.grad.dtype == v.grad.dtype == gate.grad.dtype == torch.bfloat16


@pytest.mark.parametrize("window", [None, WINDOW], ids=["global", "local"])
def test_golden_forward_matches_fp64_reference(mods, window):
    gold, _ag, _bwd = mods
    inp = make_inputs([2049, 64], seed=21)
    y, _o, lse = gold.golden_forward_fixed_order(
        inp["q"], inp["k"], inp["v"], inp["gate"], inp["cu_q"], inp["cu_k"], inp["pos"],
        window=window,
    )
    y_ref = gold.reference_forward(
        inp["q"].double(), inp["k"].double(), inp["v"].double(), inp["gate"].double(),
        inp["cu_q"], inp["cu_k"], inp["pos"], window=window,
    )
    atol, rtol = _tol("forward_accuracy")
    assert torch.allclose(y.float(), y_ref.float(), atol=atol, rtol=rtol)
    assert torch.isfinite(lse).all()


def test_rejects_non_contiguous_positions(mods):
    _gold, _ag, bwd = mods
    inp = make_inputs([64], seed=1)
    pos = inp["pos"].clone()
    pos[10] = 50
    do = torch.zeros(64, 4096, dtype=torch.bfloat16, device="cuda")
    lse = torch.zeros(64, 32, dtype=torch.float32, device="cuda")
    with pytest.raises(ValueError, match="contiguous"):
        bwd.strict_gated_attention_bwd_partials(
            inp["q"], inp["k"], inp["v"], do, lse, lse.clone(), inp["cu_q"], inp["cu_k"], pos,
            window=None,
        )


def test_trace_is_stable(mods):
    _gold, ag, bwd = mods
    t1, t2 = bwd.trace(), bwd.trace()
    assert t1 == t2
    for key in ("op", "dtype_in", "dtype_acc", "dtype_out", "reduction_order", "tile_config"):
        assert key in t1
    assert t1["tile_config"]["BLOCK_N"] == 64 and t1["tile_config"]["BLOCK_M"] == 64
    assert t1["tile_config"]["allow_tf32"] is False and t1["tile_config"]["autotune"] is False
    a = ag.trace()
    assert a["op"] == "muse_glimmer.gated_attention" and "forward_impl" in a
    assert a["kernel_id"] == bwd.KERNEL_ID
