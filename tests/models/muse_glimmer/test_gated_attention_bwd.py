# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Contract gates for the strict Muse-Glimmer gated attention backward (O5, rows
``global_gated_attention`` / ``local_gated_attention_train``).

G4  deterministic repeat (torch.equal) and accuracy vs the FP64 autograd reference;
G2  batch invariance of dq/dgate rows and of dk/dv (alone vs packed, several positions);
CP  query-chunk rule: the one-shot kernel's leaves equal per-chunk calls, and simulated CP
    ranks combined in the canonical tree equal the one-shot dK/dV (any T, incl. T % 8 != 0);
TP  real head-shard calls (TP2/4/8 per-rank geometries 16/1, 8/1, 4/1) equal the TP1 slices
    bitwise and their dK/dV partials, combined in the canonical TP tree, equal TP1 dk/dv;
G5  trace() emitted and stable; plus the fail-closed input checks and the review findings
    gated_attention_bwd:1..9 (see REPORT_gated_attention_bwd.md).

Accuracy gate: max|got - ref| / max|ref| <= 1e-2 per gradient (observed <= 6e-3, FA2 bf16
level). The magnitude of an attention gradient is set by the arbitrary scale of ``dy`` (dK sums
16 heads x up to 8192 queries); with unit dy, max|dK| ~ 26 where the contract-mandated bf16
OUTPUT rounding alone is 0.0625 > the ws1 ``gradient_accuracy/attention`` atol (5e-2), so the
absolute ws1 row cannot gate a bf16 gradient. The normalized bound is strictly tighter than the
ws1 row applied at unit scale (atol 5e-2 >= 1e-2), so no separate ws1 allclose is asserted (it
could never fail; review finding gated_attention_bwd:6).
"""

from __future__ import annotations

import pytest
import torch

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
pytestmark = [pytest.mark.ws1, requires_cuda]

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

NORMALIZED_BOUND = 1e-2
WINDOW = 2048
WINDOWS = pytest.mark.parametrize("window", [None, WINDOW], ids=["global", "local"])


@pytest.fixture(scope="module")
def mods():
    from rl_engine.models.muse_glimmer.ops import attention_golden_bwd as gold
    from rl_engine.models.muse_glimmer.ops import gated_attention_autograd as ag
    from rl_engine.models.muse_glimmer.triton import gated_attention_bwd as bwd

    return gold, ag, bwd


def make_inputs(
    lens: list[int], *, seed: int = 0, pos0: list[int] | None = None, k_lens=None,
    hq: int = 32, hkv: int = 2,
):
    """Random bf16 inputs at the real per-token shapes; q rms-normalised and scaled by 3.87."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    tq = sum(lens)
    k_lens = list(lens) if k_lens is None else list(k_lens)
    tk = sum(k_lens)
    q = torch.randn(tq, hq, 128, generator=g)
    q = q * torch.rsqrt(q.pow(2).mean(-1, keepdim=True) + 1e-5) * 3.87
    k = torch.randn(tk, hkv, 128, generator=g)
    k = k * torch.rsqrt(k.pow(2).mean(-1, keepdim=True) + 1e-5)
    v = torch.randn(tk, hkv, 128, generator=g)
    gate = torch.randn(tq, hq * 128, generator=g)
    dy = torch.randn(tq, hq * 128, generator=g)
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


def strict_fwd(inp, window, **heads):
    from rl_engine.models.muse_glimmer.ops import gated_attention as ga

    return ga.strict_gated_attention_fwd_varlen(
        inp["q"], inp["k"], inp["v"], inp["gate"], inp["cu_q"], inp["cu_k"], inp["pos"],
        window=window, **heads,
    )


def strict_bwd(mods, inp, window):
    """Strict forward kernel -> saved (o, lse) -> strict backward (what training runs)."""
    _gold, _ag, bwd = mods
    _y, o, lse = strict_fwd(inp, window)
    grads = bwd.strict_gated_attention_bwd(
        inp["q"], inp["k"], inp["v"], inp["gate"], o, lse, inp["dy"], inp["cu_q"], inp["cu_k"],
        inp["pos"], window=window,
    )
    return grads, (o, lse)


def _check_accuracy(name, got: torch.Tensor, ref: torch.Tensor) -> float:
    got = got.float()
    ref = ref.float()
    assert not torch.isnan(got).any(), f"{name} has NaN"
    scale = float(ref.abs().max())
    abs_err = float((got - ref).abs().max())
    if scale < 1e-6:
        # exactly-zero gradient (e.g. dq/dk of a one-token sequence: p == 1 for every q, k);
        # the kernel leaves only fp32 cancellation residue (dP - delta)
        assert abs_err <= 1e-4, f"{name}: zero reference but max |got| = {abs_err:.3e}"
        return abs_err
    norm_err = abs_err / scale
    assert norm_err <= NORMALIZED_BOUND, f"{name}: normalized max error {norm_err:.3e}"
    return norm_err


# ------------------------------------------------------------------------------- G4 accuracy
@WINDOWS
@pytest.mark.parametrize("length", [1, 2, 64, 2047, 2048, 2049, 4096])
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


@WINDOWS
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


# ------------------------------------------------------------------------------- G4 repeat
@WINDOWS
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


# ------------------------------------------------------------------------------- G2 batch
@WINDOWS
def test_batch_invariance(mods, window):
    """Sequence A alone vs packed at batch positions 1 and 3 with different neighbours."""
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
        for name in ("q", "gate", "dy", "k", "v"):
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


# ------------------------------------------------------------------------------- CP rule
def _chunk_bounds(n: int) -> list[tuple[int, int]]:
    return [((c * n) // 8, ((c + 1) * n) // 8) for c in range(8)]


@WINDOWS
@pytest.mark.parametrize("n", [1, 7, 300, 2048, 2100, 4096])
def test_chunk_leaves_equal_per_chunk_calls(mods, window, n):
    """Leaf (c, g) of the one-shot call == the same kernel called on chunk c's rows alone.

    This is the CP property: a rank holding chunk c computes exactly the one-shot leaves.
    """
    _gold, _ag, bwd = mods
    inp = make_inputs([n], seed=900 + n)
    _y, o, lse = strict_fwd(inp, window)
    _dg, do, delta = bwd.gated_attention_bwd_preprocess(inp["dy"], o, inp["gate"])
    dq, leaf_k, leaf_v = bwd.strict_gated_attention_bwd_chunk_leaves(
        inp["q"], inp["k"], inp["v"], do, lse, delta, inp["cu_q"], inp["cu_k"], inp["pos"],
        window=window,
    )
    assert leaf_k.shape == (8, 8, n, 128)
    for c, (lo, hi) in enumerate(_chunk_bounds(n)):
        if lo == hi:
            assert torch.count_nonzero(leaf_k[c]) == 0 and torch.count_nonzero(leaf_v[c]) == 0
            continue
        cu_c = torch.tensor([0, hi - lo], dtype=torch.int32, device="cuda")
        dq_c, lk_c, lv_c = bwd.strict_gated_attention_bwd_chunk_leaves(
            inp["q"][lo:hi].contiguous(), inp["k"], inp["v"], do[lo:hi].contiguous(),
            lse[lo:hi].contiguous(), delta[lo:hi].contiguous(), cu_c, inp["cu_k"],
            inp["pos"][lo:hi].contiguous(), window=window,
        )
        assert torch.equal(dq_c, dq[lo:hi]), f"dq rows of chunk {c}"
        assert torch.equal(lk_c[c], leaf_k[c]) and torch.equal(lv_c[c], leaf_v[c]), c
        others = [i for i in range(8) if i != c]
        assert torch.count_nonzero(lk_c[others]) == 0 and torch.count_nonzero(lv_c[others]) == 0


@WINDOWS
@pytest.mark.parametrize("n", [8, 300, 2048, 2100, 4096])
def test_one_shot_equals_simulated_cp_ranks(mods, window, n):
    """CP2/CP4 on one GPU: each rank's call on its contiguous rows, rank roots combined in the
    canonical tree (all_gather is a copy), equals the one-shot dk/dv bitwise; dq/dgate rows too.
    """
    from rl_engine.models.muse_glimmer.ops.canonical_tree import combine_tree

    _gold, _ag, bwd = mods
    inp = make_inputs([n], seed=1000 + n)
    (dq, dk, dv, dgate), (o, lse) = strict_bwd(mods, inp, window)
    for cp in (2, 4, 8):
        if n % cp:
            continue
        rank_dk, rank_dv = [], []
        span = n // cp
        for r in range(cp):
            rows = slice(r * span, (r + 1) * span)
            dg_r, do_r, delta_r = bwd.gated_attention_bwd_preprocess(
                inp["dy"][rows].contiguous(), o[rows].contiguous(), inp["gate"][rows].contiguous()
            )
            cu_r = torch.tensor([0, span], dtype=torch.int32, device="cuda")
            dq_r, dkp_r, dvp_r = bwd.strict_gated_attention_bwd_partials(
                inp["q"][rows].contiguous(), inp["k"], inp["v"], do_r, lse[rows].contiguous(),
                delta_r, cu_r, inp["cu_k"], inp["pos"][rows].contiguous(), window=window,
            )
            assert torch.equal(dq_r, dq[rows]) and torch.equal(dg_r, dgate[rows]), (cp, r)
            rank_dk.append(dkp_r)
            rank_dv.append(dvp_r)
        got_k = bwd.combine_kv_partials(combine_tree(rank_dk))  # ranks, then groups
        got_v = bwd.combine_kv_partials(combine_tree(rank_dv))
        assert torch.equal(got_k, dk), f"CP{cp} dk differs from the one-shot kernel"
        assert torch.equal(got_v, dv), f"CP{cp} dv differs from the one-shot kernel"


@WINDOWS
@pytest.mark.parametrize("n", [2048, 2100])
def test_one_shot_equals_cp1_path(mods, window, n):
    """runtime/cp_attention backward at CP1 (the plan WS2 runs) == the one-shot kernel."""
    from rl_engine.models.muse_glimmer.runtime import cp_attention as cpa

    _gold, _ag, _bwd = mods
    inp = make_inputs([n], seed=1100 + n)
    (dq, dk, dv, dgate), (o, lse) = strict_bwd(mods, inp, window)
    got = cpa.cp_gated_attention_backward(
        inp["q"], inp["k"], inp["v"], inp["gate"], o, lse, inp["dy"], inp["pos"],
        cpa.CPAttentionPlan(1, 0), None, window=window,
    )
    for name, a, b in zip(("dq", "dk", "dv", "dgate"), got, (dq, dk, dv, dgate)):
        assert torch.equal(a, b), f"CP1 path {name} differs from the one-shot kernel"


# ------------------------------------------------------------------------------- TP shards
@WINDOWS
@pytest.mark.parametrize("tp", [2, 4, 8])
def test_tp_head_shards_equal_tp1_slices(mods, window, tp):
    """Per-rank calls on real TP head shards (16/1, 8/1, 4/1) == the TP1 call's slices.

    dq/dgate rows and the per-group dK/dV partials are compared bitwise; the ranks' sub-tree
    roots, combined in the canonical TP tree (runtime/tp_canonical_tree), equal TP1 dk/dv.
    (Replaces the old simulation that re-associated the same tensors: finding bwd:5.)
    """
    from rl_engine.models.muse_glimmer.runtime.tp_canonical_tree import combine_tree as tp_tree

    _gold, _ag, bwd = mods
    inp = make_inputs([1500, 600], seed=5)
    _y, o, lse = strict_fwd(inp, window)
    dgate, do, delta = bwd.gated_attention_bwd_preprocess(inp["dy"], o, inp["gate"])
    dq, dkp, dvp = bwd.strict_gated_attention_bwd_partials(
        inp["q"], inp["k"], inp["v"], do, lse, delta, inp["cu_q"], inp["cu_k"], inp["pos"],
        window=window,
    )
    dk = bwd.combine_kv_partials(dkp)
    dv = bwd.combine_kv_partials(dvp)
    hq_r, groups_r = 32 // tp, 8 // tp
    roots_k, roots_v = [], []
    for r in range(tp):
        heads = slice(r * hq_r, (r + 1) * hq_r)
        cols = slice(r * hq_r * 128, (r + 1) * hq_r * 128)
        kvh = (r * hq_r) // 16
        shard = dict(inp)
        shard.update(
            q=inp["q"][:, heads].contiguous(),
            k=inp["k"][:, kvh : kvh + 1].contiguous(),
            v=inp["v"][:, kvh : kvh + 1].contiguous(),
            gate=inp["gate"][:, cols].contiguous(),
            dy=inp["dy"][:, cols].contiguous(),
        )
        heads_kw = dict(num_q_heads=hq_r, num_kv_heads=1)
        _y_r, o_r, lse_r = strict_fwd(shard, window, **heads_kw)
        assert torch.equal(o_r, o[:, cols]) and torch.equal(lse_r, lse[:, heads]), (tp, r)
        dg_r, do_r, delta_r = bwd.gated_attention_bwd_preprocess(
            shard["dy"], o_r, shard["gate"], num_q_heads=hq_r
        )
        assert torch.equal(dg_r, dgate[:, cols]) and torch.equal(delta_r, delta[:, heads])
        dq_r, dkp_r, dvp_r = bwd.strict_gated_attention_bwd_partials(
            shard["q"], shard["k"], shard["v"], do_r, lse_r, delta_r, inp["cu_q"], inp["cu_k"],
            inp["pos"], window=window, **heads_kw,
        )
        assert torch.equal(dq_r, dq[:, heads]), f"TP{tp} rank {r} dq"
        groups = slice(r * groups_r, (r + 1) * groups_r)
        assert dkp_r.shape == (groups_r, inp["k"].shape[0], 1, 128)
        assert torch.equal(dkp_r[:, :, 0], dkp[groups, :, kvh]), f"TP{tp} rank {r} dK partials"
        assert torch.equal(dvp_r[:, :, 0], dvp[groups, :, kvh]), f"TP{tp} rank {r} dV partials"
        for roots, part in ((roots_k, dkp_r), (roots_v, dvp_r)):
            root = bwd.kv_partials_subtree(part)  # (Tk, 1, 128) fp32, this rank's tree node
            full = torch.zeros((root.shape[0], 2, 128), dtype=torch.float32, device="cuda")
            full[:, kvh] = root[:, 0]
            roots.append(full)
        if tp == 2:  # a TP2 rank owns its whole KV head: the fused entry is final
            dq2, dk2, dv2, dg2 = bwd.strict_gated_attention_bwd(
                shard["q"], shard["k"], shard["v"], shard["gate"], o_r, lse_r, shard["dy"],
                inp["cu_q"], inp["cu_k"], inp["pos"], window=window, **heads_kw,
            )
            assert torch.equal(dk2[:, 0], dk[:, kvh]) and torch.equal(dv2[:, 0], dv[:, kvh])
        else:  # TP4/TP8 ranks hold partials only: the fused entry fails closed
            with pytest.raises(ValueError, match="partials"):
                bwd.strict_gated_attention_bwd(
                    shard["q"], shard["k"], shard["v"], shard["gate"], o_r, lse_r, shard["dy"],
                    inp["cu_q"], inp["cu_k"], inp["pos"], window=window, **heads_kw,
                )
    assert torch.equal(tp_tree(roots_k).to(torch.bfloat16), dk), f"TP{tp} tree dk != TP1"
    assert torch.equal(tp_tree(roots_v).to(torch.bfloat16), dv), f"TP{tp} tree dv != TP1"


def test_rejects_unsupported_head_geometry(mods):
    _gold, _ag, bwd = mods
    inp = make_inputs([64], seed=2, hq=16, hkv=2)
    do = torch.zeros(64, 16 * 128, dtype=torch.bfloat16, device="cuda")
    stats = torch.zeros(64, 16, dtype=torch.float32, device="cuda")
    with pytest.raises(ValueError, match="TP shard"):
        bwd.strict_gated_attention_bwd_partials(
            inp["q"], inp["k"], inp["v"], do, stats, stats, inp["cu_q"], inp["cu_k"],
            inp["pos"], window=None, num_q_heads=16, num_kv_heads=2,
        )


# ------------------------------------------------------------------------------- findings 1-3
def test_autograd_forward_is_the_strict_kernel(mods):
    """Finding bwd:1: GatedAttentionFn must run the strict Triton forward, never the golden."""
    _gold, ag, _bwd = mods
    rec = ag.trace()
    assert rec["forward_is_fallback"] is False
    assert rec["forward_impl"].startswith("rl_engine.models.muse_glimmer.ops.gated_attention.")
    inp = make_inputs([300, 64], seed=9)
    for window in (None, WINDOW):
        y = ag.gated_attention(
            inp["q"], inp["k"], inp["v"], inp["gate"], inp["cu_q"], inp["cu_k"], inp["pos"],
            window,
        )
        y_k, _o, _lse = strict_fwd(inp, window)
        assert torch.equal(y, y_k), "autograd forward is not the strict kernel"


@WINDOWS
def test_autograd_function_binds_strict_backward(mods, window):
    """Finding bwd:2: grads of the Function == direct entry on the SAME saved o/lse, bitwise."""
    gold, ag, _bwd = mods
    inp = make_inputs([300, 64], seed=9)
    q = inp["q"].clone().requires_grad_(True)
    k = inp["k"].clone().requires_grad_(True)
    v = inp["v"].clone().requires_grad_(True)
    gate = inp["gate"].clone().requires_grad_(True)
    y = ag.gated_attention(q, k, v, gate, inp["cu_q"], inp["cu_k"], inp["pos"], window)
    assert y.dtype == torch.bfloat16 and y.shape == (364, 4096)
    y.backward(inp["dy"])
    y_k, o, lse = strict_fwd(inp, window)  # exactly what the Function saved
    assert torch.equal(y.detach(), y_k)
    dq, dk, dv, dgate = ag.gated_attention_backward(
        inp["q"], inp["k"], inp["v"], inp["gate"], o, lse, inp["dy"], inp["cu_q"], inp["cu_k"],
        inp["pos"], window=window,
    )
    assert torch.equal(q.grad, dq) and torch.equal(k.grad, dk)
    assert torch.equal(v.grad, dv) and torch.equal(gate.grad, dgate)
    assert q.grad.dtype == k.grad.dtype == v.grad.dtype == gate.grad.dtype == torch.bfloat16
    refs = gold.reference_backward(
        inp["q"], inp["k"], inp["v"], inp["gate"], inp["cu_q"], inp["cu_k"], inp["pos"],
        inp["dy"], window=window,
    )
    for name, got, ref in zip(("dq", "dk", "dv", "dgate"), (dq, dk, dv, dgate), refs):
        _check_accuracy(name, got, ref)


@WINDOWS
def test_golden_adapter_is_the_forward_golden(mods, window):
    """Finding bwd:3: one O5 golden; the varlen adapter equals it bitwise per sequence."""
    from rl_engine.models.muse_glimmer.ops import gated_attention as ga

    gold, _ag, _bwd = mods
    lens, pos0, k_lens = [100, 70], [200, 0], [300, 70]
    inp = make_inputs(lens, seed=21, pos0=pos0, k_lens=k_lens)
    y, o, lse = gold.golden_forward_fixed_order(
        inp["q"], inp["k"], inp["v"], inp["gate"], inp["cu_q"], inp["cu_k"], inp["pos"],
        window=window,
    )
    q0 = k0 = 0
    for n, nk in zip(lens, k_lens):
        yg, og, lg = ga.gated_attention_forward_fp32(
            inp["q"][q0 : q0 + n], inp["k"][k0 : k0 + nk], inp["v"][k0 : k0 + nk],
            inp["gate"][q0 : q0 + n], inp["pos"][q0 : q0 + n], window,
        )
        assert torch.equal(y[q0 : q0 + n], yg.to(torch.bfloat16))
        assert torch.equal(o[q0 : q0 + n], og.to(torch.bfloat16))
        assert torch.equal(lse[q0 : q0 + n], lg)
        q0, k0 = q0 + n, k0 + nk


@WINDOWS
def test_golden_forward_matches_fp64_reference(mods, window):
    from rl_engine.contracts.numerical import load_contract, resolve_tolerance

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
    tol = resolve_tolerance(
        load_contract(), judgment="forward_accuracy", op_class="attention", dtype=torch.bfloat16
    )
    assert torch.allclose(y.float(), y_ref.float(), atol=tol.atol, rtol=tol.rtol)
    assert torch.isfinite(lse).all()


# ------------------------------------------------------------------------------- finding 4
def test_backward_sigmoid_is_the_forward_sigmoid(mods):
    """Finding bwd:4: preprocess must use the forward's sigmoid (div_rn), bit for bit.

    One-key sequences make the forward exact: p = 1, l = 1, o = v, y = bf16(o * sg_fwd(g)).
    Feeding dy := o to the preprocess gives do = bf16(o * sg_bwd(g)); equal iff the sigmoids
    agree on every one of the 16.8M gate values (the approximate fp32 `/` flips dozens).
    """
    _gold, _ag, bwd = mods
    n = 4096
    g = torch.Generator(device="cpu").manual_seed(4)
    dev = torch.device("cuda")
    q = torch.randn(n, 32, 128, generator=g).to(dev, torch.bfloat16)
    k = torch.randn(n, 2, 128, generator=g).to(dev, torch.bfloat16)
    v = torch.randn(n, 2, 128, generator=g).to(dev, torch.bfloat16)
    gate = (torch.randn(n, 4096, generator=g) * 3.0).to(dev, torch.bfloat16)
    cu = torch.arange(n + 1, dtype=torch.int32, device=dev)
    pos = torch.zeros(n, dtype=torch.int32, device=dev)
    inp = dict(q=q, k=k, v=v, gate=gate, cu_q=cu, cu_k=cu, pos=pos)
    y, o, _lse = strict_fwd(inp, None)
    assert torch.equal(o.view(n, 32, 128), v.repeat_interleave(16, dim=1))  # o == v exactly
    _dgate, do, _delta = bwd.gated_attention_bwd_preprocess(o, torch.ones_like(o), gate)
    mismatches = int((do != y).sum())
    assert mismatches == 0, f"backward sigmoid differs from the forward's on {mismatches} values"


# ------------------------------------------------------------------------------- findings 7-9
def test_rejects_non_bf16_dy(mods):
    """Finding bwd:7: no silent fp32 -> bf16 cast of dy (fail closed)."""
    _gold, _ag, bwd = mods
    inp = make_inputs([64], seed=1)
    _y, o, lse = strict_fwd(inp, None)
    with pytest.raises(ValueError, match="bf16"):
        bwd.strict_gated_attention_bwd(
            inp["q"], inp["k"], inp["v"], inp["gate"], o, lse, inp["dy"].float(), inp["cu_q"],
            inp["cu_k"], inp["pos"], window=None,
        )


def test_rejects_inconsistent_cu_seqlens_k_and_devices(mods):
    """Finding bwd:8: cu_seqlens_k[-1] must equal Tk; index tensors must be on q's device."""
    _gold, _ag, bwd = mods
    inp = make_inputs([64, 64], seed=1)
    _y, o, lse = strict_fwd(inp, None)
    args = (inp["q"], inp["k"], inp["v"], inp["gate"], o, lse, inp["dy"])
    short_k = inp["cu_k"].clone()
    short_k[-1] = 100  # claims fewer keys than k holds
    with pytest.raises(ValueError, match="cu_seqlens_k"):
        bwd.strict_gated_attention_bwd(
            *args, inp["cu_q"], short_k, inp["pos"], window=None
        )
    long_k = inp["cu_k"].clone()
    long_k[-1] = 200  # claims more keys than k holds (would read out of bounds)
    with pytest.raises(ValueError, match="cu_seqlens_k"):
        bwd.strict_gated_attention_bwd(*args, inp["cu_q"], long_k, inp["pos"], window=None)
    with pytest.raises(ValueError, match="device"):
        bwd.strict_gated_attention_bwd(
            *args, inp["cu_q"].cpu(), inp["cu_k"], inp["pos"], window=None
        )


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


@WINDOWS
def test_seq_meta_removes_host_sync(mods, window):
    """Finding bwd:9: with a prepared SequenceMeta the backward performs no host sync."""
    _gold, ag, bwd = mods
    inp = make_inputs([300, 64, 2049], seed=13)
    _y, o, lse = strict_fwd(inp, window)
    args = (inp["q"], inp["k"], inp["v"], inp["gate"], o, lse, inp["dy"], inp["cu_q"],
            inp["cu_k"], inp["pos"])
    ref = bwd.strict_gated_attention_bwd(*args, window=window)
    meta = bwd.prepare_sequence_meta(
        inp["cu_q"], inp["cu_k"], inp["pos"], num_k_tokens=inp["k"].shape[0]
    )
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        got = bwd.strict_gated_attention_bwd(*args, window=window, seq_meta=meta)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    for a, b in zip(got, ref):
        assert torch.equal(a, b)
    # the meta is bound to its tensors: a copy of cu_seqlens (or an in-place edit) is rejected
    with pytest.raises(ValueError, match="seq_meta"):
        bwd.strict_gated_attention_bwd(
            *args[:7], inp["cu_q"].clone(), inp["cu_k"], inp["pos"], window=window,
            seq_meta=meta,
        )
    # autograd entry accepts the meta too
    q = inp["q"].clone().requires_grad_(True)
    y = ag.gated_attention(
        q, inp["k"], inp["v"], inp["gate"], inp["cu_q"], inp["cu_k"], inp["pos"], window, meta
    )
    y.backward(inp["dy"])
    assert torch.equal(q.grad, ref[0])


# ------------------------------------------------------------------------------- G5
def test_trace_is_stable(mods):
    _gold, ag, bwd = mods
    t1, t2 = bwd.trace(), bwd.trace()
    assert t1 == t2
    for key in ("op", "dtype_in", "dtype_acc", "dtype_out", "reduction_order", "tile_config"):
        assert key in t1
    assert t1["tile_config"]["BLOCK_N"] == 64 and t1["tile_config"]["BLOCK_M"] == 64
    assert t1["tile_config"]["allow_tf32"] is False and t1["tile_config"]["autotune"] is False
    assert t1["query_chunk_rule"]["num_chunks"] == 8
    a = ag.trace()
    assert a["op"] == "muse_glimmer.gated_attention" and "forward_impl" in a
    assert a["kernel_id"] == bwd.KERNEL_ID
    assert a["forward"]["op"] == "gated_attention"
