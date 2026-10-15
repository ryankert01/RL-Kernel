# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Gates G1/G2/G4/G5 (and the WS2 tile/TP parity) for ``lm_head_softcap_logprob`` (O7).

Small geometry: H=256, V=1900 padded to Vp=2048 = 8 tiles x 256 (148 padded columns in the
last tile); H and the tile are multiples of 256 so the default ``fast_det_gemm`` path runs (a
K % 256 != 0 reduction now fails closed instead of silently using the legacy kernel).  Real
geometry: H=6656, V=202048 padded to 202752 = 8 x 25344.

Tolerances: parity gates are ``torch.equal``.  The forward-accuracy gate uses the ws1.json
``logprob`` bfloat16 row (``atol=0.06``, read from the read-only profile) because the strict
path consumes bf16 operands; on top of that we assert a much tighter sanity bound
(``TIGHT_ATOL = 2e-3`` vs an fp64 golden) that a wrong multiplier, cap, mask or tree would
blow through.  Gradients: bf16 outputs are compared to an fp32 autograd reference with
``rtol=2e-2`` and ``atol = 2e-2 * max|ref|`` (the bf16 cast of ``dz`` and of the result are
each 2^-8 relative).
"""

from __future__ import annotations

import json
import math
import os
import random
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from rl_engine.models.muse_glimmer.ops import lm_head_softcap_logprob as op

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")

REPO = Path(__file__).resolve().parents[3]
WS1_JSON = REPO / "rl_engine" / "contracts" / "profiles" / "precision" / "ws1.json"
TIGHT_ATOL = 2e-3
GRAD_RTOL = 2e-2

SMALL_H = 256
SMALL_V = 1900
SMALL_VP = 2048  # 8 tiles x 256; 148 padded columns in tile 7
MULT = op.OUTPUT_MULTIPLIER
CAP = op.FINAL_LOGIT_SOFTCAP

CKPT = Path(os.path.expanduser("~/models/Muse-Glimmer-30B"))


def _ws1_logprob_atol() -> float:
    profile = json.loads(WS1_JSON.read_text())
    return float(profile["accuracy"]["default"]["logprob"]["bfloat16"]["atol"])


def _small_inputs(rows: int, *, seed: int = 0, device: str = "cpu", vocab: int = SMALL_V):
    g = torch.Generator().manual_seed(seed)
    h = torch.randn(rows, SMALL_H, generator=g).to(torch.bfloat16)
    # logits ~ N(0, 30): the soft cap is neither linear nor saturated (u = z*m/T ~ N(0, 0.3))
    w = (torch.randn(vocab, SMALL_H, generator=g) * (30.0 / math.sqrt(SMALL_H))).to(torch.bfloat16)
    targets = torch.randint(0, vocab, (rows,), generator=g)
    w_padded = op.pad_lm_head_weight(w, padded_vocab=SMALL_VP)
    return h.to(device), w_padded.to(device), targets.to(device)


def _strict(h, w_padded, targets, **kw):
    vocab_size = kw.pop("vocab_size", SMALL_V)
    return op.lm_head_softcap_logprob(
        h, w_padded, targets, multiplier=MULT, softcap=CAP, vocab_size=vocab_size, **kw
    )


def _torch_reference_fp32(h, w_padded, targets, *, vocab_size):
    """Independent plain-torch fp32 reference (matmul + log_softmax) for autograd checks."""

    z = h.float() @ w_padded.float().t()
    zhat = op.softcap_golden(z, multiplier=MULT, softcap=CAP)[:, :vocab_size]
    logp = torch.log_softmax(zhat, dim=-1).gather(1, targets[:, None]).squeeze(1)
    return logp


# --------------------------------------------------------------------------------------
# CPU-safe unit tests
# --------------------------------------------------------------------------------------


@pytest.mark.unit
def test_pad_lm_head_weight_appends_zero_rows():
    w = torch.randn(1900, 16).to(torch.bfloat16)
    wp = op.pad_lm_head_weight(w, padded_vocab=2048)
    assert wp.shape == (2048, 16) and wp.dtype == torch.bfloat16
    assert torch.equal(wp[:1900], w)
    assert torch.count_nonzero(wp[1900:]) == 0
    assert op.pad_lm_head_weight(wp, padded_vocab=2048) is wp  # already aligned: no copy
    with pytest.raises(ValueError):
        op.pad_lm_head_weight(w, padded_vocab=1024)  # smaller than V
    with pytest.raises(ValueError):
        op.pad_lm_head_weight(w, padded_vocab=1901)  # not a multiple of 8
    assert op.pad_lm_head_weight(torch.zeros(op.VOCAB_SIZE, 1, dtype=torch.bfloat16)).shape == (
        op.PADDED_VOCAB_SIZE,
        1,
    )


@pytest.mark.unit
def test_tile_geometry_real_shape():
    tile, tiles = op.tile_geometry(op.PADDED_VOCAB_SIZE, op.VOCAB_SIZE)
    assert tile == 25344 and len(tiles) == 8
    assert tiles[0] == (0, 25344)
    assert tiles[7] == (7 * 25344, 25344 - 704)
    assert sum(real for _, real in tiles) == op.VOCAB_SIZE


@pytest.mark.unit
def test_combine_tree_is_the_canonical_eight_leaf_tree():
    expr = op._combine_tree([str(i) for i in range(8)], lambda a, b: f"({a}+{b})")
    assert expr == "(((0+1)+(2+3))+((4+5)+(6+7)))"
    assert op._combine_tree(["a", "b", "c", "d"], lambda a, b: f"({a}+{b})") == "((a+b)+(c+d))"
    with pytest.raises(ValueError):
        op._combine_tree([1, 2, 3], lambda a, b: a + b)


@pytest.mark.unit
def test_combine_tile_stats_matches_logsumexp_and_handles_empty_tiles():
    g = torch.Generator().manual_seed(1)
    rows, tiles, width = 5, 8, 37
    zhat = torch.randn(rows, tiles, width, generator=g, dtype=torch.float64) * 10
    stats = []
    for t in range(tiles):
        block = zhat[:, t]
        m = block.max(dim=-1).values
        s = torch.exp(block - m[:, None]).sum(dim=-1)
        stats.append((m, s))
    m_all, logz = op.combine_tile_stats(stats)
    torch.testing.assert_close(m_all, zhat.amax(dim=(1, 2)))
    torch.testing.assert_close(logz, torch.logsumexp(zhat.reshape(rows, -1), dim=-1))
    # TP sub-trees are nodes of the same tree: rank merges then rank tree == flat tree, bitwise.
    for tp in (2, 4, 8):
        per = tiles // tp
        ranks = [op.merge_tile_stats(stats[r * per : (r + 1) * per]) for r in range(tp)]
        assert torch.equal(op.combine_tile_stats(ranks)[1], logz)
    # An empty tile (-inf, 0) is the identity and never produces NaN.
    empty = (
        torch.full((rows,), float("-inf"), dtype=torch.float64),
        torch.zeros(rows, dtype=torch.float64),
    )
    with_empty = stats[:7] + [empty]
    _, logz7 = op.combine_tile_stats(with_empty)
    assert torch.isfinite(logz7).all()
    torch.testing.assert_close(logz7, torch.logsumexp(zhat[:, :7].reshape(rows, -1), dim=-1))


@pytest.mark.unit
def test_golden_fp32_matches_fp64_and_rejects_pad_targets():
    h, w_padded, targets = _small_inputs(6, seed=2)
    logp32 = op.lm_head_softcap_logprob_fp32(h, w_padded, targets, vocab_size=SMALL_V)
    logp64 = op.lm_head_softcap_logprob_fp32(
        h, w_padded, targets, vocab_size=SMALL_V, compute_dtype=torch.float64
    )
    assert logp32.dtype == torch.float32 and logp32.shape == (6,)
    assert (logp32 <= 1e-6).all()
    torch.testing.assert_close(logp32, logp64, atol=1e-4, rtol=0.0)
    bad = targets.clone()
    bad[0] = SMALL_V  # first padded column
    with pytest.raises(ValueError, match="padded"):
        op.lm_head_softcap_logprob_fp32(h, w_padded, bad, vocab_size=SMALL_V)


@pytest.mark.unit
def test_trace_is_complete_and_stable():
    first, second = op.trace(), op.trace()
    assert first == second
    for key in ("op", "dtype_in", "dtype_acc", "dtype_out", "reduction_order", "tile_config"):
        assert key in first
    assert first["op"] == "lm_head_softcap_logprob"
    assert first["dtype_in"] == "bfloat16" and first["dtype_acc"] == "float32"
    assert first["dtype_out"] == "float32"
    assert first["tile_config"]["num_tiles"] == 8 and first["tile_config"]["tile_size"] == 25344
    assert first["tile_config"]["padded_vocab_size"] == 202752
    assert first["tile_config"]["autotune"] is False
    assert "((0+1)+(2+3))+((4+5)+(6+7))" in first["reduction_order"]["tile_tree"]
    assert first["constants"]["output_multiplier"] == pytest.approx(0.19611613513818404)
    assert first["constants"]["final_logit_softcapping"] == 20.0
    json.dumps(first)  # serialisable provenance
    # GEMM provenance (finding lm_head_softcap_logprob:3): real order + R6 launch constants.
    order = first["reduction_order"]["gemm"]
    assert "8 canonical chunks" in order and "block order" not in order
    assert "ragged_k=True" in order and "roundup(T, 256)" in order
    gemm = first["gemm"]
    assert gemm["op"] == "muse_glimmer.fast_det_gemm" and gemm["dtype_out"] == "float32"
    assert set(gemm["tile_config"]) == {"decode16", "decode64", "prefill"}
    for cfg in gemm["tile_config"].values():
        assert {"block_m", "block_n", "block_k", "num_warps", "num_stages"} <= set(cfg)
    legacy = op.trace(gemm=op.legacy_det_gemm_fp32)
    assert "explicit non-default backend" in legacy["reduction_order"]["gemm"]


@pytest.mark.unit
def test_default_gemm_backend_and_no_silent_fallback():
    """The default is fast_det_gemm; a broken det_gemm import propagates (finding
    lm_head_softcap_logprob:4) instead of silently selecting the legacy kernel."""

    assert op.gemm_backend_name() == "fast_det_gemm"
    assert op.resolve_gemm() is op.resolve_gemm(None)
    code = (
        "import sys; sys.modules['rl_engine.models.muse_glimmer.ops.det_gemm'] = None\n"
        "try:\n"
        "    from rl_engine.models.muse_glimmer.ops import lm_head_softcap_logprob as m\n"
        "    print('SILENT', m.gemm_backend_name())\n"
        "except ImportError as exc:\n"
        "    print('RAISED', type(exc).__name__)\n"
    )
    env = dict(os.environ, PYTHONPATH=str(REPO))
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=300
    )
    assert "RAISED" in out.stdout, out.stdout + out.stderr


# --------------------------------------------------------------------------------------
# GPU gates
# --------------------------------------------------------------------------------------


@requires_cuda
@pytest.mark.ws1
def test_softcap_kernel_is_bitwise_torch_tanh():
    """Contract O7: tl tanh chain must agree with torch fp32 tanh (or use one shared formula)."""

    from rl_engine.models.muse_glimmer.triton.softcap_logprob import launch_softcap

    g = torch.Generator(device="cuda").manual_seed(3)
    z = torch.cat(
        [
            torch.randn(1 << 22, device="cuda", generator=g) * 60.0,
            torch.randn(1 << 20, device="cuda", generator=g) * 1e-3,
            torch.tensor([0.0, -0.0, 1e-30, 1e4, -1e4, 300.0, -300.0], device="cuda"),
        ]
    )
    ours = launch_softcap(z, multiplier=MULT, softcap=CAP)
    golden = op.softcap_golden(z, multiplier=MULT, softcap=CAP)
    assert torch.equal(ours, golden)
    assert (ours.abs() <= CAP).all()


@requires_cuda
@pytest.mark.ws1
def test_near_certain_token_logprob_is_never_positive():
    """The tile maximum must contribute exp(0) == 1 exactly, so s_t >= 1 and logp <= 0.

    With FP contraction, pass 2 computed ``softcap * tanh(..) - m`` as one FMA, the max element's
    exp was 0.999999 and 44/2048 such rows came out 1-2 ulp above 0 (48/1024 generated tokens on the 30B).
    """

    rows, vocab = 512, op.VOCAB_SIZE
    tile = op.TILE_SIZE
    g = torch.Generator(device="cuda").manual_seed(0)
    targets = torch.randint(0, vocab, (rows,), device="cuda", generator=g)
    big = torch.rand(rows, device="cuda", generator=g) * 150.0 + 60.0
    stats = []
    for t in range((vocab + tile - 1) // tile):
        z = torch.randn(rows, tile, device="cuda", generator=g) * 3.0 - 60.0
        hit = (targets // tile) == t
        z[hit.nonzero().squeeze(1), (targets[hit] % tile)] = big[hit]
        stats.append(op.softcap_tile_stats(z, t, targets, multiplier=MULT, softcap=CAP, vocab_size=vocab))
    m_all = torch.stack([s[0] for s in stats]).max(0).values
    for m_t, s_t, _, _ in stats:
        assert (s_t[m_t == m_all] >= 1.0).all()
    _, log_z = op.combine_tile_stats([(s[0], s[1]) for s in stats])
    zhat_t = torch.stack([torch.where(s[3], s[2], torch.zeros_like(s[2])) for s in stats]).sum(0)
    assert (zhat_t - log_z <= 0).all()


@requires_cuda
@pytest.mark.ws1
def test_g1_forward_matches_fp64_golden_small():
    h, w_padded, targets = _small_inputs(37, seed=4, device="cuda")
    logp = _strict(h, w_padded, targets)
    assert logp.dtype == torch.float32 and logp.shape == (37,)
    assert (logp <= 1e-6).all()
    gold64 = op.lm_head_softcap_logprob_fp32(
        h, w_padded, targets, vocab_size=SMALL_V, compute_dtype=torch.float64
    )
    gold32 = op.lm_head_softcap_logprob_fp32(h, w_padded, targets, vocab_size=SMALL_V)
    err64 = (logp - gold64).abs().max().item()
    err32 = (logp - gold32).abs().max().item()
    print(f"\nG1 small: max|logp - fp64 golden| = {err64:.3e}, vs fp32 golden = {err32:.3e}")
    assert err64 <= _ws1_logprob_atol()
    assert err64 <= TIGHT_ATOL and err32 <= TIGHT_ATOL
    # The selected-token soft-capped logit is bitwise what the kernel streamed.
    full = op.softcap_full_logits(h, w_padded, multiplier=MULT, softcap=CAP)
    assert full.shape == (37, SMALL_VP)
    ref = _torch_reference_fp32(h, w_padded, targets, vocab_size=SMALL_V)
    assert (logp - ref).abs().max().item() <= TIGHT_ATOL


@requires_cuda
@pytest.mark.ws1
def test_g2_batch_invariance_rows_and_positions():
    h, w_padded, targets = _small_inputs(256, seed=5, device="cuda")
    full = _strict(h, w_padded, targets)
    for i in (0, 1, 63, 64, 128, 255):
        single = _strict(h[i : i + 1], w_padded, targets[i : i + 1])
        assert torch.equal(single, full[i : i + 1]), f"row {i} differs at T=1 vs T=256"
    perm = torch.randperm(256, generator=torch.Generator().manual_seed(6)).to("cuda")
    permuted = _strict(h[perm], w_padded, targets[perm])
    assert torch.equal(permuted, full[perm])
    small_batch = _strict(h[:16], w_padded, targets[:16])
    assert torch.equal(small_batch, full[:16])
    # repeat determinism
    assert torch.equal(_strict(h, w_padded, targets), full)


@requires_cuda
@pytest.mark.ws1
def test_tile_order_independence():
    h, w_padded, targets = _small_inputs(64, seed=7, device="cuda")
    canonical, logz = op._forward_tiles(
        h, w_padded, targets, multiplier=MULT, softcap=CAP, vocab_size=SMALL_V, gemm=None
    )
    rng = random.Random(8)
    for order in (list(range(7, -1, -1)), [3, 0, 6, 1, 7, 2, 5, 4], rng.sample(range(8), 8)):
        other, other_logz = op._forward_tiles(
            h, w_padded, targets, multiplier=MULT, softcap=CAP, vocab_size=SMALL_V, gemm=None,
            tile_order=order,
        )
        assert torch.equal(other, canonical), f"tile order {order} changed the bits"
        assert torch.equal(other_logz, logz)


@requires_cuda
@pytest.mark.ws2
def test_simulated_tp_equals_tp1():
    h, w_padded, targets = _small_inputs(48, seed=9, device="cuda")
    tp1 = _strict(h, w_padded, targets)
    for tp in (1, 2, 4, 8):
        sim = op.simulate_tp_logprob(h, w_padded, targets, tp=tp, vocab_size=SMALL_V)
        assert torch.equal(sim, tp1), f"TP{tp} differs from TP1"


@requires_cuda
@pytest.mark.ws1
def test_padded_columns_are_excluded_and_pad_targets_raise():
    h, w_padded, targets = _small_inputs(32, seed=10, device="cuda")
    clean = _strict(h, w_padded, targets)
    garbage = w_padded.clone()
    garbage[SMALL_V:] = (torch.randn(SMALL_VP - SMALL_V, SMALL_H, device="cuda") * 50).to(
        torch.bfloat16
    )
    assert torch.equal(_strict(h, garbage, targets), clean)
    bad = targets.clone()
    bad[3] = SMALL_V
    with pytest.raises(ValueError, match="padded"):
        _strict(h, w_padded, bad)
    bad[3] = SMALL_VP - 1
    with pytest.raises(ValueError, match="padded"):
        _strict(h, w_padded, bad)
    # Without padding (V == Vp) the op still works.
    h2, w2, t2 = _small_inputs(8, seed=11, device="cuda", vocab=SMALL_VP)
    logp2 = _strict(h2, w2, t2, vocab_size=SMALL_VP)
    ref2 = _torch_reference_fp32(h2, w2, t2, vocab_size=SMALL_VP)
    assert (logp2 - ref2).abs().max().item() <= TIGHT_ATOL


@requires_cuda
@pytest.mark.ws1
def test_g4_backward_matches_fp32_autograd_and_is_deterministic():
    h, w_padded, targets = _small_inputs(40, seed=12, device="cuda")
    dlogp = torch.randn(40, device="cuda", generator=torch.Generator(device="cuda").manual_seed(13))

    def run_strict():
        hh = h.clone().requires_grad_(True)
        ww = w_padded.clone().requires_grad_(True)
        logp = _strict(hh, ww, targets)
        logp.backward(dlogp)
        return logp.detach(), hh.grad.detach(), ww.grad.detach()

    logp_a, dh_a, dw_a = run_strict()
    logp_b, dh_b, dw_b = run_strict()
    assert torch.equal(logp_a, logp_b) and torch.equal(dh_a, dh_b) and torch.equal(dw_a, dw_b)
    assert dh_a.dtype == torch.bfloat16 and dw_a.dtype == torch.bfloat16
    assert dw_a.shape == w_padded.shape
    assert torch.count_nonzero(dw_a[SMALL_V:]) == 0, "padded rows must have zero gradient"

    hr = h.float().requires_grad_(True)
    wr = w_padded.float().requires_grad_(True)
    ref = _torch_reference_fp32(hr, wr, targets, vocab_size=SMALL_V)
    ref.backward(dlogp)
    for name, ours, theirs in (("dh", dh_a, hr.grad), ("dw", dw_a, wr.grad)):
        scale = theirs.abs().max().item()
        err = (ours.float() - theirs).abs().max().item()
        print(f"\nG4 small {name}: max|ours - fp32 autograd| = {err:.3e} (max|ref| = {scale:.3e})")
        torch.testing.assert_close(
            ours.float(), theirs, rtol=GRAD_RTOL, atol=GRAD_RTOL * scale
        )

    # dh is row-local: batch invariance of the gradient too (dW sums rows, so it is not).
    hh = h[:1].clone().requires_grad_(True)
    _strict(hh, w_padded, targets[:1]).backward(dlogp[:1])
    assert torch.equal(hh.grad, dh_a[:1])


def _count_scalar_reads(fn) -> int:
    """Device->host scalar reads (``aten::_local_scalar_dense``, i.e. ``bool()``/``item()``) that
    ``fn`` performs; each one is a stream synchronisation."""

    from torch.utils._python_dispatch import TorchDispatchMode

    class _Counter(TorchDispatchMode):
        count = 0

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            if "_local_scalar_dense" in str(func):
                type(self).count += 1
            return func(*args, **(kwargs or {}))

    with _Counter():
        fn()
    return _Counter.count


def _no_sync(fn) -> None:
    """Run ``fn`` with torch's sync debug mode set to error: any host sync raises."""

    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        fn()
    finally:
        torch.cuda.set_sync_debug_mode("default")


@requires_cuda
@pytest.mark.ws1
def test_host_syncs_and_device_side_target_rule():
    """One host sync per validated call (was two), none with validate_targets=False; an invalid
    target then yields NaN on the device instead of a silent -logZ (finding
    lm_head_softcap_logprob:1)."""

    h, w_padded, targets = _small_inputs(32, seed=40, device="cuda")
    with torch.no_grad():
        ref = _strict(h, w_padded, targets)  # warm-up / compile
        assert _count_scalar_reads(lambda: _strict(h, w_padded, targets)) == 1
        unchecked = []
        _no_sync(lambda: unchecked.append(_strict(h, w_padded, targets, validate_targets=False)))
        assert torch.equal(unchecked[0], ref)
        with pytest.raises(RuntimeError):  # the validated path does sync (control)
            _no_sync(lambda: _strict(h, w_padded, targets))
        bad = targets.clone()
        bad[[3, 7, 9]] = torch.tensor([SMALL_V, -100, SMALL_VP + 5], device="cuda")
        out = _strict(h, w_padded, bad, validate_targets=False)
        assert torch.isnan(out[[3, 7, 9]]).all()
        keep = torch.ones(32, dtype=torch.bool, device="cuda")
        keep[[3, 7, 9]] = False
        assert torch.equal(out[keep], ref[keep])
        tp = op.simulate_tp_logprob(h, w_padded, targets, tp=4, vocab_size=SMALL_V)
        assert torch.equal(tp, ref)


@requires_cuda
@pytest.mark.ws1
@pytest.mark.parametrize("tokens", [40, 255, 256, 300])
def test_dw_uses_declared_ragged_token_rule(tokens):
    """dW reduces over T with the fast_det_gemm ragged rule for EVERY T (finding
    lm_head_softcap_logprob:2; before: T % 256 != 0 silently switched to the legacy kernel):
    dW(T) == dW of the same tokens zero-padded to roundup(T, 256) with zero upstream gradient."""

    h, w_padded, targets = _small_inputs(tokens, seed=41, device="cuda")
    gen = torch.Generator(device="cuda").manual_seed(4)
    dlogp = torch.randn(tokens, device="cuda", generator=gen)
    hh = h.clone().requires_grad_(True)
    ww = w_padded.clone().requires_grad_(True)
    _strict(hh, ww, targets).backward(dlogp)
    t_pad = -(-tokens // 256) * 256
    h_pad = torch.zeros(t_pad, SMALL_H, device="cuda", dtype=torch.bfloat16)
    h_pad[:tokens] = h
    t_full = torch.zeros(t_pad, dtype=torch.int64, device="cuda")
    t_full[:tokens] = targets
    d_pad = torch.zeros(t_pad, device="cuda")
    d_pad[:tokens] = dlogp
    hp = h_pad.requires_grad_(True)
    wp = w_padded.clone().requires_grad_(True)
    _strict(hp, wp, t_full).backward(d_pad)
    assert torch.equal(ww.grad, wp.grad)
    assert torch.equal(hh.grad, hp.grad[:tokens])


@requires_cuda
@pytest.mark.ws1
@pytest.mark.parametrize("tokens", [256, 300])
def test_real_shape_backward_fast_path(tokens):
    """Real geometry backward through the fast dW path (finding lm_head_softcap_logprob:6):
    default backend, repeat torch.equal, allclose vs fp32 autograd, padded dW rows zero."""

    assert op.gemm_backend_name() == "fast_det_gemm"
    device = "cuda"
    g = torch.Generator(device=device).manual_seed(42)
    w = (torch.randn(op.VOCAB_SIZE, op.HIDDEN_SIZE, device=device, generator=g) * 0.02).to(
        torch.bfloat16
    )
    w_padded = op.pad_lm_head_weight(w)
    del w
    h = torch.randn(tokens, op.HIDDEN_SIZE, device=device, generator=g).to(torch.bfloat16)
    targets = torch.randint(0, op.VOCAB_SIZE, (tokens,), device=device, generator=g)
    dlogp = torch.randn(tokens, device=device, generator=g)

    def run():
        hh = h.clone().requires_grad_(True)
        ww = w_padded.clone().requires_grad_(True)
        op.lm_head_softcap_logprob(hh, ww, targets).backward(dlogp)
        return hh.grad, ww.grad

    dh_a, dw_a = run()
    dh_b, dw_b = run()
    assert torch.equal(dh_a, dh_b) and torch.equal(dw_a, dw_b)
    assert torch.count_nonzero(dw_a[op.VOCAB_SIZE :]) == 0
    del dh_b, dw_b
    hr = h.float().requires_grad_(True)
    wr = w_padded[: op.VOCAB_SIZE].float().requires_grad_(True)
    ref = _torch_reference_fp32(hr, wr, targets, vocab_size=op.VOCAB_SIZE)
    ref.backward(dlogp)
    for name, ours, theirs in (("dh", dh_a, hr.grad), ("dw", dw_a[: op.VOCAB_SIZE], wr.grad)):
        scale = theirs.abs().max().item()
        err = (ours.float() - theirs).abs().max().item()
        print(f"\nreal T={tokens} {name}: max|ours - fp32 autograd| = {err:.3e} "
              f"(max|ref| {scale:.3e})")
        torch.testing.assert_close(ours.float(), theirs, rtol=GRAD_RTOL, atol=GRAD_RTOL * scale)


@requires_cuda
@pytest.mark.ws1
def test_real_shape_determinism_and_forward_error():
    """V=202048 (padded 202752), H=6656: repeat determinism at T=64; fp64 error at T=8."""

    device = "cuda"
    g = torch.Generator(device=device).manual_seed(14)
    w = (torch.randn(op.VOCAB_SIZE, op.HIDDEN_SIZE, device=device, generator=g) * 0.02).to(
        torch.bfloat16
    )
    w_padded = op.pad_lm_head_weight(w)
    assert w_padded.shape == (op.PADDED_VOCAB_SIZE, op.HIDDEN_SIZE)
    h = torch.randn(64, op.HIDDEN_SIZE, device=device, generator=g).to(torch.bfloat16)
    targets = torch.randint(0, op.VOCAB_SIZE, (64,), device=device, generator=g)
    dlogp = torch.randn(64, device=device, generator=g)

    def run():
        hh = h.clone().requires_grad_(True)
        ww = w_padded.clone().requires_grad_(True)
        logp = op.lm_head_softcap_logprob(hh, ww, targets)
        logp.backward(dlogp)
        return logp.detach(), hh.grad, ww.grad

    logp_a, dh_a, dw_a = run()
    logp_b, dh_b, dw_b = run()
    assert torch.equal(logp_a, logp_b) and torch.equal(dh_a, dh_b) and torch.equal(dw_a, dw_b)
    assert torch.count_nonzero(dw_a[op.VOCAB_SIZE :]) == 0
    assert (logp_a <= 1e-6).all() and torch.isfinite(logp_a).all()
    # Batch invariance at the real shape.
    assert torch.equal(op.lm_head_softcap_logprob(h[5:6], w_padded, targets[5:6]), logp_a[5:6])
    del dw_a, dw_b
    # TP emulation at the real shape.
    assert torch.equal(op.simulate_tp_logprob(h[:4], w_padded, targets[:4], tp=4), logp_a[:4])

    gold64 = op.lm_head_softcap_logprob_fp32(
        h[:8], w_padded, targets[:8], compute_dtype=torch.float64
    )
    err = (logp_a[:8] - gold64).abs().max().item()
    print(f"\nG1 real shape (T=8): max|logp - fp64 golden| = {err:.3e}")
    assert err <= _ws1_logprob_atol() and err <= TIGHT_ATOL


@requires_cuda
@pytest.mark.ws1
@pytest.mark.skipif(not (CKPT / "REVISION").exists(), reason="Muse-Glimmer-30B checkpoint absent")
def test_real_weights_forward_shapes_and_memory():
    from safetensors import safe_open

    index = json.loads((CKPT / "model.safetensors.index.json").read_text())["weight_map"]
    with safe_open(CKPT / index["lm_head.weight"], framework="pt", device="cuda") as f:
        w = f.get_tensor("lm_head.weight")
    with safe_open(
        CKPT / index["model.language_model.norm.weight"], framework="pt", device="cuda"
    ) as f:
        norm_w = f.get_tensor("model.language_model.norm.weight")
    assert w.shape == (op.VOCAB_SIZE, op.HIDDEN_SIZE) and w.dtype == torch.bfloat16
    assert norm_w.shape == (op.HIDDEN_SIZE,)
    w_padded = op.pad_lm_head_weight(w)
    del w
    g = torch.Generator(device="cuda").manual_seed(15)
    x = torch.randn(16, op.HIDDEN_SIZE, device="cuda", generator=g, dtype=torch.float32)
    # final norm (O2): y = x_hat * w, eps=1e-5 -> bf16
    h = (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-5) * norm_w.float()).to(
        torch.bfloat16
    )
    targets = torch.randint(0, op.VOCAB_SIZE, (16,), device="cuda", generator=g)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    logp = op.lm_head_softcap_logprob(h, w_padded, targets)
    torch.cuda.synchronize()
    peak_extra = (torch.cuda.max_memory_allocated() - base) / 2**20
    print(f"\nreal weights: logp[:4]={logp[:4].tolist()} peak extra memory = {peak_extra:.0f} MiB")
    assert logp.shape == (16,) and logp.dtype == torch.float32
    assert torch.isfinite(logp).all() and (logp <= 1e-6).all()
    assert torch.equal(op.lm_head_softcap_logprob(h[3:4], w_padded, targets[3:4]), logp[3:4])
    # One fp32 tile (16 x 25344 x 4 B = 1.5 MiB) plus a transposed bf16 weight tile (322 MiB).
    assert peak_extra < 2048
