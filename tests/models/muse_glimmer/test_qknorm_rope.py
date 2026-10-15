# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Contract gates G1-G5 for O4 ``qknorm_scaled_rope`` (Muse-Glimmer-30B strict POC).

Tolerances: ``rl_engine/contracts/profiles/precision/ws1.json`` ``accuracy.default.elementwise``
(bf16: atol 0.02 / rtol 0.016; fp32: 1e-5 / 1e-5) and ``judgments.gradient_accuracy.elementwise``
(same numbers).  Parity gates use ``torch.equal``.  The HF comparison (double rounding) is
reported, not bitwise-gated.  GPU only.
"""

from __future__ import annotations

import json

import pytest
import torch

from rl_engine.models.muse_glimmer.ops import qknorm_rope as op
from rl_engine.models.muse_glimmer.ops.qknorm_rope import (
    QK_SCALE,
    RMS_NORM_EPS,
    ROPE_THETA,
    QKNormScaledRopeOp,
    qknorm_scaled_rope,
    qknorm_scaled_rope_backward_fp32,
    qknorm_scaled_rope_fp32,
    qknorm_scaled_rope_fp64,
    rope_table,
    rope_table_fp64,
    trace,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

DEV = torch.device("cuda")
HQ, HK, D = 32, 2, 128
T_REAL = 4096
# ws1.json accuracy.default.elementwise / gradient_accuracy.elementwise
TOL_BF16 = {"atol": 0.02, "rtol": 0.016}
TOL_FP32 = {"atol": 1e-5, "rtol": 1e-5}


@pytest.fixture(autouse=True)
def _strict_numerics():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    yield


def _make_qk(T: int, seed: int, scale: float = 1.0, hq: int = HQ, hk: int = HK):
    gen = torch.Generator(device=DEV).manual_seed(seed)
    q = (torch.randn(T, hq, D, device=DEV, generator=gen) * scale).to(torch.bfloat16)
    k = (torch.randn(T, hk, D, device=DEV, generator=gen) * scale).to(torch.bfloat16)
    return q, k


def _tables(positions: torch.Tensor):
    return rope_table(positions.to(DEV), device=DEV)


def _assert_close(got: torch.Tensor, ref: torch.Tensor, tol: dict, what: str) -> None:
    got = got.float()
    ref = ref.float()
    worst = ((got - ref).abs() / (tol["atol"] + tol["rtol"] * ref.abs())).max().item()
    assert torch.allclose(got, ref, **tol), (
        f"{what}: max |diff| = {(got - ref).abs().max().item():.3e}, "
        f"worst err/tol ratio = {worst:.3f}"
    )


# --------------------------------------------------------------------------------------
# rope_table - THE table definition
# --------------------------------------------------------------------------------------


class TestRopeTable:
    def test_shape_dtype_and_duplicated_halves(self):
        pos = torch.arange(37, device=DEV)
        cos, sin = _tables(pos)
        assert cos.shape == sin.shape == (37, D)
        assert cos.dtype == sin.dtype == torch.float32
        assert torch.equal(cos[:, :64], cos[:, 64:]) and torch.equal(sin[:, :64], sin[:, 64:])
        assert torch.equal(cos[0], torch.ones(D, device=DEV))
        assert torch.equal(sin[0], torch.zeros(D, device=DEV))

    def test_deterministic_and_int32_equals_int64(self):
        pos = torch.arange(T_REAL, device=DEV)
        c1, s1 = _tables(pos)
        c2, s2 = _tables(pos.to(torch.int32))
        c3, s3 = _tables(pos)
        assert torch.equal(c1, c2) and torch.equal(s1, s2)
        assert torch.equal(c1, c3) and torch.equal(s1, s3)

    def test_gather_consistency(self):
        """Rows for a sub-range / arbitrary positions equal the rows of the full table."""
        full_c, full_s = _tables(torch.arange(T_REAL, device=DEV))
        sub = torch.tensor([0, 1, 2047, 2048, 4095, 777], device=DEV)
        c, s = _tables(sub)
        assert torch.equal(c, full_c[sub]) and torch.equal(s, full_s[sub])
        c, s = _tables(torch.arange(100, 300, device=DEV))
        assert torch.equal(c, full_c[100:300]) and torch.equal(s, full_s[100:300])

    def test_matches_fp64_definition_within_fp32_angle_error(self):
        """fp32 angle p*inv_freq carries |err| <= ~p * 2^-24 rad; measured bound reported."""
        pos = torch.arange(T_REAL, device=DEV)
        c32, s32 = _tables(pos)
        c64, s64 = rope_table_fp64(pos, device=DEV)
        err = max((c32.double() - c64).abs().max().item(), (s32.double() - s64).abs().max().item())
        print(f"\n[rope_table] fp32 vs fp64 table max abs diff at T={T_REAL}: {err:.3e}")
        assert err < 4096 * 2.0**-24 * 1.5  # 3.7e-4; measured 2.8e-4
        c32s, s32s = _tables(pos[:64])
        c64s, s64s = rope_table_fp64(pos[:64], device=DEV)
        assert torch.allclose(c32s.double(), c64s, atol=1e-5, rtol=0)
        assert torch.allclose(s32s.double(), s64s, atol=1e-5, rtol=0)

    def test_matches_hf_rotary_embedding_fp32(self):
        """Same bits as MuseGlimmerTextRotaryEmbedding's fp32 cos/sin (before its bf16 cast),
        when HF's inv_freq is computed on the same device."""
        pos = torch.arange(T_REAL, device=DEV)
        exponent = torch.arange(0, D, 2, dtype=torch.float, device=DEV) / D
        inv_freq = 1.0 / (ROPE_THETA**exponent)
        inv_freq_expanded = inv_freq[None, :, None].float().expand(1, -1, 1)
        pos_expanded = pos[None, None, :].float()
        freqs = (inv_freq_expanded @ pos_expanded).transpose(1, 2)  # K=1 matmul: exact products
        emb = torch.cat((freqs, freqs), dim=-1)
        cos_hf, sin_hf = (emb.cos() * 1.0)[0], (emb.sin() * 1.0)[0]
        cos, sin = _tables(pos)
        assert torch.equal(cos, cos_hf) and torch.equal(sin, sin_hf)

    def test_rejects_bad_positions(self):
        with pytest.raises(TypeError):
            rope_table(torch.arange(4, device=DEV).float(), device=DEV)
        with pytest.raises(ValueError):
            rope_table(torch.zeros(2, 2, dtype=torch.long, device=DEV), device=DEV)


# --------------------------------------------------------------------------------------
# G1 forward correctness
# --------------------------------------------------------------------------------------


class TestG1Forward:
    @pytest.mark.parametrize("T,scale", [(1, 1.0), (5, 0.05), (64, 1.0), (T_REAL, 2.0)])
    def test_vs_fp32_golden(self, T, scale):
        q, k = _make_qk(T, seed=T, scale=scale)
        cos, sin = _tables(torch.arange(T, device=DEV) + 17)
        q_hat, k_hat = qknorm_scaled_rope(q, k, cos, sin)
        gq, gk = qknorm_scaled_rope_fp32(q, k, cos, sin)
        assert q_hat.dtype == k_hat.dtype == torch.bfloat16
        assert q_hat.shape == q.shape and k_hat.shape == k.shape
        assert gq.dtype == torch.float32
        _assert_close(q_hat, gq, TOL_BF16, "q_hat vs fp32 golden")
        _assert_close(k_hat, gk, TOL_BF16, "k_hat vs fp32 golden")

    @pytest.mark.parametrize("T", [7, T_REAL])
    def test_matches_fp32_golden_bitwise(self, T):
        """Stronger than the contract: kernel == golden.to(bf16) bitwise (same tree, no FMA).
        If this breaks after a Triton upgrade, G1 (tolerance) still decides the contract."""
        q, k = _make_qk(T, seed=11, scale=2.0)
        cos, sin = _tables(torch.arange(T, device=DEV))
        q_hat, k_hat = qknorm_scaled_rope(q, k, cos, sin)
        gq, gk = qknorm_scaled_rope_fp32(q, k, cos, sin)
        neq_q = (q_hat != gq.to(torch.bfloat16)).sum().item()
        neq_k = (k_hat != gk.to(torch.bfloat16)).sum().item()
        assert neq_q == 0 and neq_k == 0, f"bitwise mismatches: q {neq_q}, k {neq_k}"

    def test_vs_fp64_golden_with_fp64_table_real_shape(self):
        q, k = _make_qk(T_REAL, seed=3, scale=2.0)
        pos = torch.arange(T_REAL, device=DEV)
        cos, sin = _tables(pos)
        c64, s64 = rope_table_fp64(pos, device=DEV)
        q_hat, k_hat = qknorm_scaled_rope(q, k, cos, sin)
        gq, gk = qknorm_scaled_rope_fp64(q, k, c64, s64)
        _assert_close(q_hat, gq, TOL_BF16, "q_hat vs fp64 golden (fp64 table)")
        _assert_close(k_hat, gk, TOL_BF16, "k_hat vs fp64 golden (fp64 table)")

    def test_fp32_golden_vs_fp64_golden(self):
        """Arithmetic error of the fp32 golden (same fp32 table) is at fp32 level; with an fp64
        table the fp32 angle error dominates (~p * 2^-24 * |x_hat|)."""
        q, k = _make_qk(T_REAL, seed=5, scale=2.0)
        pos = torch.arange(T_REAL, device=DEV)
        cos, sin = _tables(pos)
        c64, s64 = rope_table_fp64(pos, device=DEV)
        gq32, gk32 = qknorm_scaled_rope_fp32(q, k, cos, sin)
        gq64s, gk64s = qknorm_scaled_rope_fp64(q, k, cos, sin)
        gq64, gk64 = qknorm_scaled_rope_fp64(q, k, c64, s64)
        _assert_close(gq32, gq64s.float(), TOL_FP32, "fp32 golden vs fp64 golden, fp32 table")
        _assert_close(gk32, gk64s.float(), TOL_FP32, "fp32 golden vs fp64 golden, fp32 table")
        err_q = (gq32.double() - gq64).abs().max().item()
        err_k = (gk32.double() - gk64).abs().max().item()
        print(f"\n[G1] fp32 vs fp64 golden (fp64 table) max abs: q {err_q:.3e} k {err_k:.3e}")
        # bound: |d angle| <= 4096 * 2^-24 (~2.4e-4) times max |q_hat| (<= ~6 sigma * 3.87)
        assert err_q < 1e-2 and err_k < 3e-3

    def test_global_layer_is_norm_and_scale_only(self):
        q, k = _make_qk(T_REAL, seed=7, scale=3.0)
        q_hat, k_hat = qknorm_scaled_rope(q, k, None, None)
        ones = torch.ones(T_REAL, D, device=DEV)
        zeros = torch.zeros(T_REAL, D, device=DEV)
        q_id, k_id = qknorm_scaled_rope(q, k, ones, zeros)
        assert torch.equal(q_hat, q_id) and torch.equal(k_hat, k_id)
        gq, gk = qknorm_scaled_rope_fp32(q, k, None, None)
        assert torch.equal(q_hat, gq.to(torch.bfloat16))
        assert torch.equal(k_hat, gk.to(torch.bfloat16))
        # weightless norm: rms(k_hat) ~ 1, rms(q_hat) ~ 3.87 (eps negligible at scale 3)
        rms_k = k_hat.float().pow(2).mean(-1).sqrt()
        rms_q = q_hat.float().pow(2).mean(-1).sqrt()
        assert torch.allclose(rms_k, torch.ones_like(rms_k), atol=5e-3)
        assert torch.allclose(rms_q, torch.full_like(rms_q, QK_SCALE), atol=2e-2)

    def test_only_q_is_scaled(self):
        """RFC row text says 'scaled Q/K norm'; the model scales Q only (k_scale=None)."""
        q, k = _make_qk(16, seed=9)
        q_hat, k_hat = qknorm_scaled_rope(q, k, None, None)
        q_hat1, k_hat1 = qknorm_scaled_rope(q, k, None, None, qk_scale=1.0)
        assert torch.equal(k_hat, k_hat1)
        assert not torch.equal(q_hat, q_hat1)
        _assert_close(q_hat, q_hat1.float() * QK_SCALE, TOL_BF16, "q scale")

    def test_op_object_and_defaults(self):
        q, k = _make_qk(8, seed=1)
        cos, sin = _tables(torch.arange(8, device=DEV))
        o = QKNormScaledRopeOp()
        assert o.eps == RMS_NORM_EPS and o.qk_scale == QK_SCALE
        a = o(q, k, cos, sin)
        b = qknorm_scaled_rope(q, k, cos, sin)
        assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
        g = o.forward_fp32(q, k, cos, sin)
        assert torch.equal(g[0], qknorm_scaled_rope_fp32(q, k, cos, sin)[0])

    def test_rejects_malformed_inputs(self):
        q, k = _make_qk(4, seed=2)
        cos, sin = _tables(torch.arange(4, device=DEV))
        with pytest.raises(ValueError):
            qknorm_scaled_rope(q, k, cos, None)
        with pytest.raises(ValueError):
            qknorm_scaled_rope(q, k, cos[:2], sin[:2])
        with pytest.raises(TypeError):
            qknorm_scaled_rope(q, k, cos.double(), sin.double())
        with pytest.raises(ValueError):
            qknorm_scaled_rope(q, k[:3], cos, sin)
        with pytest.raises(ValueError):
            qknorm_scaled_rope(q[..., :64].contiguous(), k[..., :64].contiguous(), None, None)


# --------------------------------------------------------------------------------------
# G2 batch invariance
# --------------------------------------------------------------------------------------


class TestG2BatchInvariance:
    def test_t1_vs_real_shape_rows(self):
        q, k = _make_qk(T_REAL, seed=21, scale=2.0)
        cos, sin = _tables(torch.arange(T_REAL, device=DEV))
        q_hat, k_hat = qknorm_scaled_rope(q, k, cos, sin)
        for t in (0, 1, 63, 64, 777, 2048, T_REAL - 1):
            sq, sk = qknorm_scaled_rope(q[t : t + 1], k[t : t + 1], cos[t : t + 1], sin[t : t + 1])
            assert torch.equal(sq[0], q_hat[t]) and torch.equal(sk[0], k_hat[t]), t

    @pytest.mark.parametrize("batch", [2, 4, 16])
    def test_same_row_at_every_batch_position(self, batch):
        q, k = _make_qk(1, seed=22, scale=0.5)
        pos = torch.tensor([1234], device=DEV)
        cos, sin = _tables(pos)
        ref_q, ref_k = qknorm_scaled_rope(q, k, cos, sin)
        for j in range(batch):
            oq, ok = _make_qk(batch, seed=100 + j, scale=4.0)
            opos = torch.randint(0, 131072, (batch,), device=DEV)
            oq[j], ok[j], opos[j] = q[0], k[0], pos[0]
            ocos, osin = _tables(opos)
            bq, bk = qknorm_scaled_rope(oq, ok, ocos, osin)
            assert torch.equal(bq[j], ref_q[0]) and torch.equal(bk[j], ref_k[0]), (batch, j)

    def test_global_path_batch_invariance(self):
        q, k = _make_qk(T_REAL, seed=23)
        q_hat, k_hat = qknorm_scaled_rope(q, k, None, None)
        for t in (0, 4095, 1000):
            sq, sk = qknorm_scaled_rope(q[t : t + 1], k[t : t + 1], None, None)
            assert torch.equal(sq[0], q_hat[t]) and torch.equal(sk[0], k_hat[t])

    def test_backward_batch_invariance(self):
        q, k = _make_qk(T_REAL, seed=24, scale=2.0)
        cos, sin = _tables(torch.arange(T_REAL, device=DEV))
        gen = torch.Generator(device=DEV).manual_seed(0)
        dyq = torch.randn(q.shape, device=DEV, generator=gen).to(torch.bfloat16)
        dyk = torch.randn(k.shape, device=DEV, generator=gen).to(torch.bfloat16)
        qg, kg = q.clone().requires_grad_(True), k.clone().requires_grad_(True)
        dq, dk = torch.autograd.grad(qknorm_scaled_rope(qg, kg, cos, sin), (qg, kg), (dyq, dyk))
        for t in (0, 777, T_REAL - 1):
            sq = q[t : t + 1].clone().requires_grad_(True)
            sk = k[t : t + 1].clone().requires_grad_(True)
            sdq, sdk = torch.autograd.grad(
                qknorm_scaled_rope(sq, sk, cos[t : t + 1], sin[t : t + 1]),
                (sq, sk),
                (dyq[t : t + 1], dyk[t : t + 1]),
            )
            assert torch.equal(sdq[0], dq[t]) and torch.equal(sdk[0], dk[t]), t


# --------------------------------------------------------------------------------------
# G3 position handling (sequence-split invariance via positions)
# --------------------------------------------------------------------------------------


class TestG3Positions:
    def test_one_shot_vs_chunked_vs_per_token(self):
        T = 97
        q, k = _make_qk(T, seed=31, scale=1.5)
        pos = torch.arange(T, device=DEV) + 1000
        cos, sin = _tables(pos)
        q_hat, k_hat = qknorm_scaled_rope(q, k, cos, sin)
        bounds = [0, 13, 50, 64, T]
        for a, b in zip(bounds[:-1], bounds[1:]):
            cq, ck = qknorm_scaled_rope(q[a:b], k[a:b], *_tables(pos[a:b]))
            assert torch.equal(cq, q_hat[a:b]) and torch.equal(ck, k_hat[a:b])
        for t in (0, 12, 13, 96):
            tq, tk = qknorm_scaled_rope(q[t : t + 1], k[t : t + 1], *_tables(pos[t : t + 1]))
            assert torch.equal(tq[0], q_hat[t]) and torch.equal(tk[0], k_hat[t])

    def test_packed_sequences_keep_logical_positions(self):
        qa, ka = _make_qk(5, seed=32)
        qb, kb = _make_qk(9, seed=33)
        pa = torch.tensor([0, 1, 2, 3, 4], device=DEV)
        pb = torch.arange(9, device=DEV) + 2040
        ra = qknorm_scaled_rope(qa, ka, *_tables(pa))
        rb = qknorm_scaled_rope(qb, kb, *_tables(pb))
        packed = qknorm_scaled_rope(
            torch.cat([qa, qb]), torch.cat([ka, kb]), *_tables(torch.cat([pa, pb]))
        )
        assert torch.equal(packed[0][:5], ra[0]) and torch.equal(packed[1][:5], ra[1])
        assert torch.equal(packed[0][5:], rb[0]) and torch.equal(packed[1][5:], rb[1])
        naive = qknorm_scaled_rope(
            torch.cat([qa, qb]), torch.cat([ka, kb]), *_tables(torch.arange(14, device=DEV))
        )
        assert not torch.equal(naive[0][5:], rb[0])

    def test_result_depends_only_on_absolute_position(self):
        """Same (q, k, p) inside different sequences at different indices -> identical bits;
        a different p -> different bits."""
        q, k = _make_qk(1, seed=34)
        p = 3000
        ref = qknorm_scaled_rope(q, k, *_tables(torch.tensor([p], device=DEV)))
        for T, idx, start in ((64, 5, p - 5), (4096, 4000, p - 4000), (3, 2, p - 2)):
            oq, ok = _make_qk(T, seed=200 + T, scale=2.0)
            oq[idx], ok[idx] = q[0], k[0]
            out = qknorm_scaled_rope(oq, ok, *_tables(torch.arange(T, device=DEV) + start))
            assert torch.equal(out[0][idx], ref[0][0]) and torch.equal(out[1][idx], ref[1][0])
        other = qknorm_scaled_rope(q, k, *_tables(torch.tensor([p + 1], device=DEV)))
        assert not torch.equal(other[0], ref[0])

    def test_global_layer_ignores_positions(self):
        q, k = _make_qk(8, seed=35)
        a = qknorm_scaled_rope(q, k, None, None)
        b = qknorm_scaled_rope(q[3:4], k[3:4], None, None)
        assert torch.equal(a[0][3], b[0][0]) and torch.equal(a[1][3], b[1][0])


# --------------------------------------------------------------------------------------
# G4 backward
# --------------------------------------------------------------------------------------


def _grads(q, k, cos, sin, dyq, dyk, fn=qknorm_scaled_rope):
    qg, kg = q.clone().requires_grad_(True), k.clone().requires_grad_(True)
    outs = fn(qg, kg, cos, sin)
    return torch.autograd.grad(outs, (qg, kg), (dyq, dyk))


class TestG4Backward:
    def _setup(self, T, seed, local=True, scale=2.0):
        q, k = _make_qk(T, seed=seed, scale=scale)
        cos, sin = _tables(torch.arange(T, device=DEV) + 5) if local else (None, None)
        gen = torch.Generator(device=DEV).manual_seed(seed + 1)
        dyq = torch.randn(q.shape, device=DEV, generator=gen).to(torch.bfloat16)
        dyk = torch.randn(k.shape, device=DEV, generator=gen).to(torch.bfloat16)
        return q, k, cos, sin, dyq, dyk

    @pytest.mark.parametrize("local", [True, False])
    def test_repeat_determinism_real_shape(self, local):
        q, k, cos, sin, dyq, dyk = self._setup(T_REAL, 41, local)
        dq1, dk1 = _grads(q, k, cos, sin, dyq, dyk)
        dq2, dk2 = _grads(q, k, cos, sin, dyq, dyk)
        assert dq1.dtype == dk1.dtype == torch.bfloat16
        assert torch.equal(dq1, dq2) and torch.equal(dk1, dk2)

    @pytest.mark.parametrize(
        "T,local,scale", [(3, True, 0.1), (64, False, 1.0), (T_REAL, True, 2.0)]
    )
    def test_vs_fp32_autograd_of_golden(self, T, local, scale):
        q, k, cos, sin, dyq, dyk = self._setup(T, 42 + T, local, scale)
        dq, dk = _grads(q, k, cos, sin, dyq, dyk)
        rq, rk = _grads(
            q.float(), k.float(), cos, sin, dyq.float(), dyk.float(), fn=qknorm_scaled_rope_fp32
        )
        _assert_close(dq, rq, TOL_BF16, "dq vs fp32 autograd")
        _assert_close(dk, rk, TOL_BF16, "dk vs fp32 autograd")

    @pytest.mark.parametrize("local", [True, False])
    def test_matches_handwritten_fp32_vjp_bitwise(self, local):
        q, k, cos, sin, dyq, dyk = self._setup(T_REAL, 43, local)
        dq, dk = _grads(q, k, cos, sin, dyq, dyk)
        hq, hk = qknorm_scaled_rope_backward_fp32(q, k, dyq, dyk, cos, sin)
        assert torch.equal(dq, hq.to(torch.bfloat16)) and torch.equal(dk, hk.to(torch.bfloat16))

    def test_handwritten_vjp_vs_autograd_fp32(self):
        q, k, cos, sin, dyq, dyk = self._setup(T_REAL, 44, True)
        hq, hk = qknorm_scaled_rope_backward_fp32(q, k, dyq, dyk, cos, sin)
        rq, rk = _grads(
            q.float(), k.float(), cos, sin, dyq.float(), dyk.float(), fn=qknorm_scaled_rope_fp32
        )
        _assert_close(hq, rq, TOL_FP32, "handwritten dq vs autograd")
        _assert_close(hk, rk, TOL_FP32, "handwritten dk vs autograd")

    def test_partial_grads(self):
        q, k, cos, sin, dyq, dyk = self._setup(8, 45, True)
        qg = q.clone().requires_grad_(True)
        q_hat, _k_hat = qknorm_scaled_rope(qg, k, cos, sin)
        (dq,) = torch.autograd.grad(q_hat, (qg,), (dyq,))
        full_dq, _ = _grads(q, k, cos, sin, dyq, dyk)
        assert torch.equal(dq, full_dq)
        assert k.grad is None


# --------------------------------------------------------------------------------------
# HF semantics (tolerance reference, double rounding quantified)
# --------------------------------------------------------------------------------------


def _hf_rmsnorm(x: torch.Tensor, eps: float) -> torch.Tensor:
    """MuseGlimmerRMSNorm(with_scale=False).forward: fp32 math, type_as(x)."""
    h = x.float()
    ms = h.pow(2).mean(-1, keepdim=True) + eps
    return (h * torch.pow(ms, -0.5)).type_as(x)


def _hf_path(q, k, positions, *, eps=RMS_NORM_EPS, scale=QK_SCALE, theta=ROPE_THETA):
    """HF MuseGlimmerTextAttention.forward semantics: norm -> bf16, *scale in bf16, bf16 RoPE
    with the fp32 table cast to bf16 (cos_sin_cast_dtype = bfloat16)."""
    qn = _hf_rmsnorm(q, eps) * scale  # bf16 * python float -> bf16 (second rounding)
    kn = _hf_rmsnorm(k, eps)
    inv_freq = 1.0 / (theta ** (torch.arange(0, D, 2, dtype=torch.float) / D))  # CPU, as HF init
    inv_freq = inv_freq.to(DEV)
    inv_freq_expanded = inv_freq[None, :, None].float().expand(1, -1, 1)
    pos_expanded = positions[None, None, :].float()
    freqs = (inv_freq_expanded @ pos_expanded).transpose(1, 2)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = (emb.cos() * 1.0).to(torch.bfloat16)[0].unsqueeze(1)  # (T, 1, D)
    sin = (emb.sin() * 1.0).to(torch.bfloat16)[0].unsqueeze(1)

    def rot(x):
        return torch.cat((-x[..., D // 2 :], x[..., : D // 2]), dim=-1)

    return (qn * cos) + (rot(qn) * sin), (kn * cos) + (rot(kn) * sin)


class TestHFSemantics:
    def test_hf_multi_rounding_vs_contract_single_rounding_real_shape(self):
        """HF rounds to bf16 after the norm, after *3.87, casts cos/sin to bf16 and rounds each
        of q*cos, rotate_half(q)*sin and their sum; the contract rounds once.  The two are NOT
        within the ws1 bf16 elementwise tolerance everywhere (measured: max |diff| 0.25 on
        q_hat, ~0.14% of elements beyond tolerance where the rotation cancels), so HF is a
        tolerance reference only.  Gate: the strict op is at least as close to the fp64 truth as
        HF (max and mean) and within TOL_BF16 of the truth; the HF distance is printed."""
        q, k = _make_qk(T_REAL, seed=51, scale=2.0)
        pos = torch.arange(T_REAL, device=DEV)
        cos, sin = _tables(pos)
        q_hat, k_hat = qknorm_scaled_rope(q, k, cos, sin)
        q_hf, k_hf = _hf_path(q, k, pos)
        c64, s64 = rope_table_fp64(pos, device=DEV)
        q64, k64 = qknorm_scaled_rope_fp64(q, k, c64, s64)
        lines = []
        for name, ours, hf, truth in (("q_hat", q_hat, q_hf, q64), ("k_hat", k_hat, k_hf, k64)):
            d = (ours.double() - hf.double()).abs()
            ratio = d / (TOL_BF16["atol"] + TOL_BF16["rtol"] * hf.double().abs())
            e_ours = (ours.double() - truth).abs()
            e_hf = (hf.double() - truth).abs()
            lines.append(
                f"[HF] {name}: strict-vs-HF max_abs={d.max().item():.4f} "
                f"mean_abs={d.mean().item():.3e} frac_neq={(d > 0).double().mean().item():.4f} "
                f"frac_beyond_ws1_bf16_tol={(ratio > 1).double().mean().item():.3e} "
                f"worst_err/tol={ratio.max().item():.2f} | err vs fp64 truth: "
                f"strict max={e_ours.max().item():.4e} mean={e_ours.mean().item():.3e} "
                f"rms={e_ours.pow(2).mean().sqrt().item():.3e}; "
                f"HF max={e_hf.max().item():.4e} mean={e_hf.mean().item():.3e} "
                f"rms={e_hf.pow(2).mean().sqrt().item():.3e}"
            )
            _assert_close(ours, truth.float(), TOL_BF16, f"{name}: strict vs fp64 truth")
            assert e_ours.max().item() <= e_hf.max().item()
            assert e_ours.mean().item() < e_hf.mean().item()
            assert (ratio > 1).double().mean().item() < 0.01  # HF deviation stays rare
        print("\n" + "\n".join(lines))


# --------------------------------------------------------------------------------------
# G5 trace
# --------------------------------------------------------------------------------------


class TestG5Trace:
    def test_trace_emitted_and_stable(self):
        t1, t2 = trace(), trace()
        assert t1 == t2
        for key in ("op", "dtype_in", "dtype_acc", "dtype_out", "reduction_order", "tile_config"):
            assert key in t1, key
        assert t1["op"] == "qknorm_scaled_rope" and t1["dtype_acc"] == "float32"
        assert t1["dtype_in"] == t1["dtype_out"] == "bfloat16"
        tc = t1["tile_config"]
        assert tc["BLOCK_HQ"] == 32 and tc["BLOCK_HK"] == 2 and tc["HEAD_DIM"] == 128
        assert tc["enable_fp_fusion"] is False and tc["num_warps"] == 4
        json.dumps(t1)  # serializable
        assert QKNormScaledRopeOp().trace() == t1
        assert op.trace(HQ=8, HK=1)["tile_config"]["BLOCK_HK"] == 1


# --------------------------------------------------------------------------------------
# Review follow-ups (findings qknorm_rope:1..5, see REPORT_qknorm_rope.md "Review follow-ups")
# --------------------------------------------------------------------------------------

N_LAYERS = 52
ROPE_POSITIONS = 131072  # max_position_embeddings
# sha256(cos.bytes || sin.bytes) of rope_table(arange(131072)) on CUDA, measured on A100 with
# torch 2.13.0+cu130 (vLLM env) AND torch 2.13.0+cu129 (Megatron env): identical.
CANONICAL_SHA256 = "088dbb8c93a55ce79d189181c46ac06fbdb61dab6c3914218bbe40ed3ea56d9c"


class TestF1RopeTableIdentity:
    """O4: 'table built once by a fixed fp32 torch-CUDA function, cached, identical object on
    both runtimes'.  A CPU build differs from the CUDA build in ~19% of the elements."""

    def test_rope_table_rejects_non_cuda_device(self):
        with pytest.raises(RuntimeError, match="torch-CUDA"):
            rope_table(torch.arange(8), device="cpu")
        with pytest.raises(RuntimeError, match="torch-CUDA"):
            rope_table(torch.arange(8, device=DEV), device=torch.device("cpu"))
        # the fp64 reference table is not the runtime table and may live anywhere
        c64, _ = rope_table_fp64(torch.arange(8), device="cpu")
        assert c64.dtype == torch.float64

    def test_canonical_table_is_cached_identical_object(self):
        c1, s1 = op.canonical_rope_table(DEV)
        c2, s2 = op.canonical_rope_table(DEV)
        assert c1 is c2 and s1 is s2
        assert c1.shape == s1.shape == (ROPE_POSITIONS, D) and c1.dtype == torch.float32
        assert op.ROPE_TABLE_POSITIONS == ROPE_POSITIONS
        c3, s3 = op.canonical_rope_table()  # default = current CUDA device
        assert c3 is c1 and s3 is s1
        full_c, full_s = rope_table(torch.arange(ROPE_POSITIONS, device=DEV), device=DEV)
        assert torch.equal(c1, full_c) and torch.equal(s1, full_s)

    def test_canonical_table_fingerprint_is_pinned(self):
        """Cross-runtime identity: both envs (cu130 vLLM, cu129 Megatron) must produce these
        bits; a torch/CUDA math-library change that alters cos/sin/pow is caught here."""
        assert op.CANONICAL_ROPE_TABLE_SHA256 == CANONICAL_SHA256
        assert op.rope_table_sha256(DEV) == CANONICAL_SHA256

    def test_rope_rows_gathers_the_canonical_table(self):
        pos = torch.tensor([0, 1, 2047, 2048, 4095, 777, ROPE_POSITIONS - 1], device=DEV)
        rc, rs = rope_table(pos, device=DEV)
        for p in (pos, pos.to(torch.int32), pos.cpu()):
            for check in (True, False):
                c, s = op.rope_rows(p, check_range=check)
                assert c.device.type == "cuda"
                assert torch.equal(c, rc) and torch.equal(s, rs)

    def test_rope_rows_rejects_out_of_range_positions(self):
        for bad in ([ROPE_POSITIONS], [-1], [3, ROPE_POSITIONS + 5]):
            with pytest.raises(ValueError, match="position"):
                op.rope_rows(torch.tensor(bad, device=DEV))
        with pytest.raises(TypeError):
            op.rope_rows(torch.tensor([1.0], device=DEV))

    def test_other_repo_table_builders_are_bitwise_identical_on_cuda(self):
        """Every table builder in the repo (reference golden, runtime fallback) must produce the
        canonical bits on CUDA, so whichever runtime builds the table, it is the same."""
        from rl_engine.models.muse_glimmer.reference import golden_ops
        from rl_engine.models.muse_glimmer.runtime import fallback_ops

        pos = torch.arange(ROPE_POSITIONS, device=DEV)
        cc, sc = op.canonical_rope_table(DEV)
        for fn in (golden_ops.rope_table, fallback_ops.rope_table):
            c, s = fn(pos, theta=ROPE_THETA, dim=D, device=DEV)
            assert torch.equal(c, cc) and torch.equal(s, sc), fn.__module__

    def test_trace_records_table_provenance(self):
        tab = trace()["rope_table"]
        assert tab["device_type"] == "cuda"
        assert tab["positions"] == ROPE_POSITIONS
        assert tab["sha256"] == CANONICAL_SHA256
        assert tab["builder"].endswith("qknorm_rope.rope_table")


class TestF2DtypeFailClosed:
    """R5 / O4: bf16 in, ONE rounding to bf16 at the end.  fp16/fp32 inputs used to be accepted
    and silently produced fp16/fp32 outputs (no bf16 rounding)."""

    @pytest.mark.parametrize("dt", [torch.float32, torch.float16])
    def test_rejects_non_bf16_inputs(self, dt):
        q, k = _make_qk(4, seed=2)
        cos, sin = _tables(torch.arange(4, device=DEV))
        for qq, kk in ((q.to(dt), k.to(dt)), (q.to(dt), k), (q, k.to(dt))):
            with pytest.raises(TypeError, match="bf16"):
                qknorm_scaled_rope(qq, kk, cos, sin)
            with pytest.raises(TypeError, match="bf16"):
                qknorm_scaled_rope(qq, kk, None, None)

    def test_backward_launcher_rejects_non_bf16_grads(self):
        from rl_engine.models.muse_glimmer.triton.qknorm_rope import qknorm_rope_backward_triton

        q, k = _make_qk(4, seed=3)
        cos, sin = _tables(torch.arange(4, device=DEV))
        with pytest.raises(TypeError, match="bf16"):
            qknorm_rope_backward_triton(
                q, k, q.float(), k.float(), cos, sin, eps=RMS_NORM_EPS, qk_scale=QK_SCALE
            )

    def test_bf16_outputs_and_grads(self):
        q, k = _make_qk(4, seed=4)
        cos, sin = _tables(torch.arange(4, device=DEV))
        dq, dk = _grads(q, k, cos, sin, torch.ones_like(q), torch.ones_like(k))
        assert dq.dtype == dk.dtype == torch.bfloat16


def _tp_shards():
    """(tp, rank, q-head slice, kv head) for TP2/4/8: the 32 Q heads split evenly, one KV head
    per rank (replicated at TP4/8); GQA: q head h reads kv head h // 16."""
    for t in (2, 4, 8):
        hq = HQ // t
        for r in range(t):
            yield t, r, slice(r * hq, (r + 1) * hq), (r * hq) // 16


class TestG2TensorParallelShards:
    """Contract parity axis 'TP1 vs TP2/4/8': a rank's head shard (16/8/4 Q heads, 1 KV head;
    BLOCK_HQ 16/8/4, BLOCK_HK 1) must give exactly the bits of the matching slice of the full
    32/2-head call, forward and backward, local and global."""

    @pytest.mark.parametrize("T", [1, 7, 256, T_REAL])
    @pytest.mark.parametrize("local", [True, False])
    def test_head_shards_equal_full_call_slices(self, T, local):
        q, k = _make_qk(T, seed=60 + T, scale=2.0)
        cos, sin = _tables(torch.arange(T, device=DEV) + 300) if local else (None, None)
        gen = torch.Generator(device=DEV).manual_seed(61)
        dyq = torch.randn(q.shape, device=DEV, generator=gen).to(torch.bfloat16)
        dyk = torch.randn(k.shape, device=DEV, generator=gen).to(torch.bfloat16)
        fq, fk = qknorm_scaled_rope(q, k, cos, sin)
        fdq, fdk = _grads(q, k, cos, sin, dyq, dyk)
        n = 0
        for t, r, qs, kv in _tp_shards():
            ks = slice(kv, kv + 1)
            q_r, k_r = q[:, qs].contiguous(), k[:, ks].contiguous()
            oq, ok = qknorm_scaled_rope(q_r, k_r, cos, sin)
            assert torch.equal(oq, fq[:, qs]), (t, r, "q_hat")
            assert torch.equal(ok, fk[:, ks]), (t, r, "k_hat")
            dq, dk = _grads(q_r, k_r, cos, sin, dyq[:, qs].contiguous(), dyk[:, ks].contiguous())
            assert torch.equal(dq, fdq[:, qs]), (t, r, "dq")
            assert torch.equal(dk, fdk[:, ks]), (t, r, "dk")
            n += 1
        assert n == 14

    def test_non_contiguous_shard_views(self):
        q, k = _make_qk(64, seed=62)
        cos, sin = _tables(torch.arange(64, device=DEV))
        fq, fk = qknorm_scaled_rope(q, k, cos, sin)
        oq, ok = qknorm_scaled_rope(q[:, 8:16], k[:, 1:2], cos, sin)  # strided views
        assert torch.equal(oq, fq[:, 8:16]) and torch.equal(ok, fk[:, 1:2])

    @pytest.mark.parametrize("tp,hq", [(1, 32), (2, 16), (4, 8), (8, 4)])
    def test_shard_tile_config_in_trace(self, tp, hq):
        hk = 2 if tp == 1 else 1
        tc = trace(HQ=hq, HK=hk)["tile_config"]
        assert tc["BLOCK_HQ"] == hq and tc["BLOCK_HK"] == hk and tc["num_warps"] == 4


class TestF4ExplicitLayerSelection:
    """Local (RoPE) vs global (NoPE) used to be implied by `cos is None` only."""

    def test_layer_schedule_matches_manifest(self):
        from pathlib import Path

        manifest = Path(op.__file__).resolve().parents[1] / "museglimmer_manifest.json"
        text_cfg = json.loads(manifest.read_text())["model_identity"]["config_fingerprint"][
            "text_config"
        ]
        types = text_cfg["layer_types"]
        assert len(types) == N_LAYERS
        got = [op.layer_is_local(i) for i in range(N_LAYERS)]
        assert got == [t == "sliding_attention" for t in types]
        assert sum(got) == 39 and got.count(False) == 13
        for bad in (-1, N_LAYERS):
            with pytest.raises(ValueError):
                op.layer_is_local(bad)

    def test_explicit_local_flag_fails_closed(self):
        q, k = _make_qk(8, seed=70)
        cos, sin = _tables(torch.arange(8, device=DEV))
        with pytest.raises(ValueError, match="local"):
            qknorm_scaled_rope(q, k, None, None, local=True)
        with pytest.raises(ValueError, match="local"):
            qknorm_scaled_rope(q, k, cos, sin, local=False)

    def test_explicit_flag_gives_legacy_bits(self):
        q, k = _make_qk(T_REAL, seed=71, scale=2.0)
        cos, sin = _tables(torch.arange(T_REAL, device=DEV))
        for c, s, local in ((cos, sin, True), (None, None, False)):
            a = qknorm_scaled_rope(q, k, c, s)
            b = qknorm_scaled_rope(q, k, c, s, local=local)
            assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
            o = QKNormScaledRopeOp().forward(q, k, c, s, local=local)
            assert torch.equal(a[0], o[0]) and torch.equal(a[1], o[1])

    def test_positions_api_contract_signature(self):
        """`qknorm_scaled_rope_at(q, k, pos, local=...)` (the contract's assembly signature)
        gathers the canonical table itself; layer selection is mandatory and explicit."""
        T = 300
        q, k = _make_qk(T, seed=72, scale=2.0)
        pos = torch.randint(0, ROPE_POSITIONS, (T,), device=DEV)
        ref_l = qknorm_scaled_rope(q, k, *rope_table(pos, device=DEV))
        ref_g = qknorm_scaled_rope(q, k, None, None)
        for kw, ref in (
            ({"local": True}, ref_l),
            ({"local": False}, ref_g),
            ({"layer_idx": 0}, ref_l),
            ({"layer_idx": 3}, ref_g),
            ({"layer_idx": 51}, ref_g),
            ({"layer_idx": 50}, ref_l),
        ):
            got = op.qknorm_scaled_rope_at(q, k, pos, **kw)
            assert torch.equal(got[0], ref[0]) and torch.equal(got[1], ref[1]), kw
        got = op.qknorm_scaled_rope_at(q, k, pos, local=True, check_range=False)
        assert torch.equal(got[0], ref_l[0]) and torch.equal(got[1], ref_l[1])
        # global layers ignore positions (may be None)
        got = op.qknorm_scaled_rope_at(q, k, None, local=False)
        assert torch.equal(got[0], ref_g[0])
        with pytest.raises(ValueError):
            op.qknorm_scaled_rope_at(q, k, pos)  # no layer selection given
        with pytest.raises(ValueError):
            op.qknorm_scaled_rope_at(q, k, pos, local=True, layer_idx=3)  # contradictory
        with pytest.raises(ValueError):
            op.qknorm_scaled_rope_at(q, k, None, local=True)
        with pytest.raises(ValueError):
            op.qknorm_scaled_rope_at(q, k, pos[:-1], local=True)

    def test_positions_api_backward_matches(self):
        T = 64
        q, k = _make_qk(T, seed=73, scale=2.0)
        pos = torch.arange(T, device=DEV) + 4000
        gen = torch.Generator(device=DEV).manual_seed(74)
        dyq = torch.randn(q.shape, device=DEV, generator=gen).to(torch.bfloat16)
        dyk = torch.randn(k.shape, device=DEV, generator=gen).to(torch.bfloat16)
        ref = _grads(q, k, *rope_table(pos, device=DEV), dyq, dyk)
        qg, kg = q.clone().requires_grad_(True), k.clone().requires_grad_(True)
        got = torch.autograd.grad(
            op.qknorm_scaled_rope_at(qg, kg, pos, local=True), (qg, kg), (dyq, dyk)
        )
        assert torch.equal(got[0], ref[0]) and torch.equal(got[1], ref[1])

    def test_trace_records_layer_selection(self):
        sel = trace()["layer_selection"]
        assert "local" in sel and "i % 4 == 3" in sel


class TestF5LargeTokenCounts:
    """Element offsets must not wrap at T*HQ*D >= 2^31 (T >= 524288 at HQ=32): with int32
    offsets the old kernel raised 'illegal memory access' or, with a different allocation
    layout, silently wrote rows >= 524288 into an unrelated tensor."""

    T_BIG = 524288 + 64

    def test_rows_beyond_int32_offsets_match_small_calls(self):
        free, _total = torch.cuda.mem_get_info()
        if free < 24 * 2**30:
            pytest.skip(f"needs ~20 GiB free GPU memory, have {free / 2**30:.1f} GiB")
        T = self.T_BIG
        gen = torch.Generator(device=DEV).manual_seed(80)
        q = torch.randn(T, HQ, D, device=DEV, generator=gen, dtype=torch.bfloat16)
        k = torch.randn(T, HK, D, device=DEV, generator=gen, dtype=torch.bfloat16)
        rows = torch.arange(T - 128, T, device=DEV)  # straddles row 524288
        for local in (False, True):
            if local:
                cos, sin = rope_table(torch.arange(T, device=DEV) % ROPE_POSITIONS, device=DEV)
                c_r, s_r = cos[rows], sin[rows]
            else:
                cos = sin = c_r = s_r = None
            oq, ok = qknorm_scaled_rope(q, k, cos, sin)
            rq, rk = qknorm_scaled_rope(q[rows], k[rows], c_r, s_r)
            assert torch.equal(oq[rows], rq) and torch.equal(ok[rows], rk), local
            del oq, ok
            if local:
                dyq = torch.ones_like(q)
                dyk = torch.ones_like(k)
                dq, dk = _grads(q, k, cos, sin, dyq, dyk)
                sdq, sdk = _grads(q[rows], k[rows], c_r, s_r, dyq[rows], dyk[rows])
                assert torch.equal(dq[rows], sdq) and torch.equal(dk[rows], sdk)
                del dq, dk, dyq, dyk
        torch.cuda.empty_cache()

    def test_trace_records_int64_offsets(self):
        assert trace()["tile_config"]["offsets"] == "int64"


class TestModelShapesStayBitwise:
    """Fail-closed checks must not break any shape the 30B model hits: T = 1..8192 (training /
    prefill) and decode batches 1..256, both layer kinds; kernel == golden -> bf16 bitwise."""

    TS = list(range(1, 18)) + [31, 63, 64, 65, 127, 255, 256, 257, 1000, 2047, 2048, 2049]
    TS += [4095, 4096, 4097, 8191, 8192]

    @pytest.mark.parametrize("local", [True, False])
    def test_token_counts_bitwise_vs_golden(self, local):
        for T in self.TS:
            q, k = _make_qk(T, seed=90 + T, scale=2.0)
            pos = torch.randint(0, ROPE_POSITIONS, (T,), device=DEV)
            cos, sin = rope_table(pos, device=DEV) if local else (None, None)
            q_hat, k_hat = qknorm_scaled_rope(q, k, cos, sin)
            gq, gk = qknorm_scaled_rope_fp32(q, k, cos, sin)
            assert torch.equal(q_hat, gq.to(torch.bfloat16)), (T, local)
            assert torch.equal(k_hat, gk.to(torch.bfloat16)), (T, local)

    @pytest.mark.parametrize("local", [True, False])
    def test_decode_batches_1_to_256(self, local):
        q, k = _make_qk(256, seed=95, scale=2.0)
        pos = torch.randint(0, ROPE_POSITIONS, (256,), device=DEV)
        cos, sin = rope_table(pos, device=DEV) if local else (None, None)
        ref_q, ref_k = qknorm_scaled_rope(q, k, cos, sin)
        for b in range(1, 257):
            c = cos[:b] if local else None
            s = sin[:b] if local else None
            oq, ok = qknorm_scaled_rope(q[:b], k[:b], c, s)
            assert torch.equal(oq, ref_q[:b]) and torch.equal(ok, ref_k[:b]), b

    @pytest.mark.parametrize("T", [1, 17, 2049, 8192])
    def test_backward_bitwise_vs_handwritten_vjp(self, T):
        q, k = _make_qk(T, seed=97, scale=2.0)
        cos, sin = rope_table(torch.arange(T, device=DEV), device=DEV)
        gen = torch.Generator(device=DEV).manual_seed(98)
        dyq = torch.randn(q.shape, device=DEV, generator=gen).to(torch.bfloat16)
        dyk = torch.randn(k.shape, device=DEV, generator=gen).to(torch.bfloat16)
        dq, dk = _grads(q, k, cos, sin, dyq, dyk)
        hq, hk = qknorm_scaled_rope_backward_fp32(q, k, dyq, dyk, cos, sin)
        assert torch.equal(dq, hq.to(torch.bfloat16)) and torch.equal(dk, hk.to(torch.bfloat16))
