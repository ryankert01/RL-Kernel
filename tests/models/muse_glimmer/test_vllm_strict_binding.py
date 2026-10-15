# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Unit tests for the Muse Glimmer vLLM strict binding (RFC #501 row ``vllm_runtime_adapter``).

CPU: fused-weight slice mapping, lm_head tile ownership, LoRA sharding/parsing, provenance writer.
GPU (skipped without CUDA): fallback-op correctness/repeatability/batch invariance, TP-simulated
row-parallel tree, LoRA zero no-op, paged == varlen == decode attention, strict-vs-fallback
agreement when the sibling ops are present.  The vLLM engine E2E test only runs with
``RL_KERNEL_MUSEGLIMMER_E2E=1`` and the checkpoint on disk.
"""

from __future__ import annotations

import json
import os

import pytest
import torch

from rl_engine.models.muse_glimmer.runtime import fallback_ops as fb
from rl_engine.models.muse_glimmer.runtime import vllm_strict as vs
from rl_engine.models.muse_glimmer.runtime.strict_ops import OP_NAMES, probe_ops, resolve_ops

CUDA = torch.cuda.is_available()
gpu = pytest.mark.skipif(not CUDA, reason="CUDA required")
DEVICE = torch.device("cuda" if CUDA else "cpu")


@pytest.fixture(autouse=True)
def _no_tf32():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(0)


def _bf(*shape, device=DEVICE):
    return torch.randn(*shape, device=device).to(torch.bfloat16)


# ------------------------------------------------------------------------------------------
# CPU: slice mapping, tiles, LoRA, provenance
# ------------------------------------------------------------------------------------------


class TestSliceMapping:
    def test_qkv_slices_tp1(self):
        hidden, d, nh, nkv = 32, 8, 4, 2
        wq, wk, wv = (
            _bf(nh * d, hidden, device="cpu"),
            _bf(nkv * d, hidden, device="cpu"),
            _bf(nkv * d, hidden, device="cpu"),
        )
        fused = torch.cat([wq, wk, wv])
        sq, sk, sv = vs.canonical_qkv_slices(fused, nh * d, nkv * d)
        assert torch.equal(sq, wq) and torch.equal(sk, wk) and torch.equal(sv, wv)
        with pytest.raises(ValueError):
            vs.canonical_qkv_slices(fused, nh * d, d)

    def test_qkv_slices_tp2_match_head_shards(self):
        hidden, d, nh, nkv, tp = 32, 8, 4, 2, 2
        wq, wk, wv = (
            _bf(nh * d, hidden, device="cpu"),
            _bf(nkv * d, hidden, device="cpu"),
            _bf(nkv * d, hidden, device="cpu"),
        )
        for rank in range(tp):
            q_rows = slice(rank * (nh // tp) * d, (rank + 1) * (nh // tp) * d)
            kv_rows = slice(rank * (nkv // tp) * d, (rank + 1) * (nkv // tp) * d)
            fused_rank = torch.cat([wq[q_rows], wk[kv_rows], wv[kv_rows]])  # vLLM per-rank layout
            sq, sk, sv = vs.canonical_qkv_slices(fused_rank, (nh // tp) * d, (nkv // tp) * d)
            assert torch.equal(sq, wq[q_rows])
            assert torch.equal(sk, wk[kv_rows])
            assert torch.equal(sv, wv[kv_rows])

    def test_gate_up_slices(self):
        w_gate, w_up = _bf(12, 6, device="cpu"), _bf(12, 6, device="cpu")
        g, u = vs.canonical_gate_up_slices(torch.cat([w_gate, w_up]))
        assert torch.equal(g, w_gate) and torch.equal(u, w_up)

    @pytest.mark.parametrize("tp", [1, 2, 4, 8])
    def test_lm_head_tiles_partition(self, tp):
        tiles = [vs.lm_head_tiles_for_rank(tp, r) for r in range(tp)]
        assert sorted(sum(tiles, [])) == list(range(8))
        assert all(len(t) == 8 // tp for t in tiles)
        assert vs.TILE_ROWS == 25344 and vs.PADDED_VOCAB == 8 * vs.TILE_ROWS

    @pytest.mark.parametrize("tp", [1, 2])
    def test_shard_lora_shapes(self, tp):
        r = 4
        for name, (k, n) in vs.LORA_SHAPES.items():
            a = torch.zeros(r, k, dtype=torch.bfloat16)
            b = torch.zeros(n, r, dtype=torch.bfloat16)
            for rank in range(tp):
                a_local, b_local = vs.shard_lora(name, a, b, tp_size=tp, tp_rank=rank)
                if name in ("o_proj", "mlp.down_proj"):
                    assert a_local.shape == (r, k // tp) and b_local.shape == (n, r)
                elif name in ("k_proj", "v_proj"):
                    assert a_local.shape == (r, k) and b_local.shape == (128 * max(1, 2 // tp), r)
                else:
                    assert a_local.shape == (r, k) and b_local.shape == (n // tp, r)
        with pytest.raises(ValueError):
            vs.shard_lora("q_proj", torch.zeros(r, 10), torch.zeros(4096, r), tp_size=1, tp_rank=0)

    def test_parse_lora_state_dict(self):
        sd = {
            "layers.3.q_proj.lora_A": torch.zeros(2, 6656),
            "layers.3.q_proj.lora_B": torch.zeros(4096, 2),
        }
        pairs = vs.parse_lora_state_dict(sd)
        assert set(pairs) == {(3, "q_proj")}
        with pytest.raises(KeyError):
            vs.parse_lora_state_dict({"layers.3.q_proj.lora_A": torch.zeros(2, 6656)})
        with pytest.raises(KeyError):
            vs.parse_lora_state_dict({"layers.3.foo.lora_A": torch.zeros(2, 6656)})


class TestProvenance:
    def test_persist_writes_readback_json(self, tmp_path, monkeypatch):
        monkeypatch.setenv(vs.READBACK_ENV, str(tmp_path))
        state = vs.StrictBindingState(ops=resolve_ops(force_fallback=True))
        state.role = "test"
        state.hooks.append("unit.hook")
        state.call_counts["attention_forward"] += 3
        path = state.persist()
        assert path is not None and path.exists()
        data = json.loads(path.read_text())
        for key in (
            "hooks",
            "backend_ids",
            "call_counts",
            "fallbacks",
            "lora_sha256",
            "tp",
            "canonical_tp",
            "kv_ownership",
            "sampling",
            "op_impls",
        ):
            assert key in data, key
        assert data["call_counts"]["attention_forward"] == 3
        assert data["canonical_tp"] == 8 and data["tp"] == 1
        assert set(data["fallbacks"]) == set(OP_NAMES)
        assert all(label.startswith("fallback:") for label in data["op_impls"].values())
        assert data["kv_ownership"]["vocab_tiles"] == [0, 8]

    def test_resolve_ops_labels_are_honest(self):
        ops = resolve_ops()
        for name in OP_NAMES:
            label = ops.labels[name]
            assert label.startswith("strict:") or label.startswith("fallback:")
            assert ops.is_strict(name) == label.startswith("strict:")


# ------------------------------------------------------------------------------------------
# GPU: fallback ops
# ------------------------------------------------------------------------------------------


@gpu
class TestFallbackGemm:
    def test_matches_fp64_repeatable_batch_invariant(self):
        a, w = _bf(8, 512), _bf(96, 512)
        y = fb.det_gemm(a, w, out_dtype=torch.float32)
        ref = (a.double() @ w.double().t()).float()
        assert torch.allclose(y, ref, atol=1e-3, rtol=1e-4)
        assert torch.equal(y, fb.det_gemm(a, w, out_dtype=torch.float32))
        single = fb.det_gemm(a[2:3], w, out_dtype=torch.float32)
        assert torch.equal(single[0], y[2]), "fallback GEMM must be batch invariant"
        assert fb.det_gemm(a, w).dtype == torch.bfloat16

    @pytest.mark.parametrize("tp", [2, 4, 8])
    def test_row_parallel_tree_matches_tp1(self, tp):
        a, w = _bf(5, 1024), _bf(64, 1024)
        full = fb.det_gemm(a, w, out_dtype=torch.float32)
        k_local = 1024 // tp
        partials = [
            fb.det_gemm_partial(
                a[:, r * k_local : (r + 1) * k_local],
                w[:, r * k_local : (r + 1) * k_local],
                fb.rank_chunk_ids(tp, r),
            )
            for r in range(tp)
        ]
        assert torch.equal(fb.combine_tree(partials), full)

    def test_fixed_sum_last(self):
        x = torch.randn(3, 6656, device=DEVICE)
        assert torch.allclose(fb.fixed_sum_last(x), x.sum(-1), atol=1e-2)
        assert torch.equal(fb.fixed_sum_last(x)[1], fb.fixed_sum_last(x[1:2])[0])


@gpu
class TestFallbackRowLocal:
    def test_norms_match_reference(self):
        x, w = _bf(4, 6656), _bf(6656)
        xf = x.float()
        inv = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-5)
        ref_c = (xf * inv * (w.float() + 1)).to(torch.bfloat16)
        ref_w = (xf * inv * w.float()).to(torch.bfloat16)
        assert torch.allclose(fb.centered_rmsnorm(x, w, eps=1e-5).float(), ref_c.float(), atol=2e-2)
        assert torch.allclose(fb.weighted_rmsnorm(x, w, eps=1e-5).float(), ref_w.float(), atol=2e-2)
        emb = fb.normalized_embedding(x, torch.tensor([3, 0], device=DEVICE), eps=1e-5)
        assert torch.allclose(
            emb[0].float(), (xf[3] * inv[3]).to(torch.bfloat16).float(), atol=2e-2
        )
        r, pre = fb.fused_post_norm_residual_pre_norm(x, x, w, 1e-8, w, 1e-5)
        post = fb.centered_rmsnorm(x, w, eps=1e-8)
        assert torch.equal(r, (x.float() + post.float()).to(torch.bfloat16))
        assert torch.equal(pre, fb.centered_rmsnorm(r, w, eps=1e-5))

    def test_qknorm_rope_matches_reference(self):
        q, k = _bf(6, 32, 128), _bf(6, 2, 128)
        pos = torch.arange(10, 16, device=DEVICE)
        cos, sin = fb.rope_table(pos, theta=500000.0, dim=128, device=DEVICE)
        assert cos.shape == (6, 128)
        qh, kh = fb.qknorm_scaled_rope(q, k, cos, sin, eps=1e-5, qk_scale=3.87)

        def ref(x, scale):
            xf = x.float()
            xf = xf * torch.pow(xf.pow(2).mean(-1, keepdim=True) + 1e-5, -0.5) * scale
            x1, x2 = xf[..., :64], xf[..., 64:]
            rot = torch.cat((-x2, x1), -1)
            return xf * cos[:, None] + rot * sin[:, None]

        assert torch.allclose(qh.float(), ref(q, 3.87), atol=5e-2, rtol=2e-2)
        assert torch.allclose(kh.float(), ref(k, 1.0), atol=2e-2, rtol=2e-2)
        q0, _ = fb.qknorm_scaled_rope(q[2:3], k[2:3], cos[2:3], sin[2:3], eps=1e-5, qk_scale=3.87)
        assert torch.equal(q0[0], qh[2])

    def test_swiglu_and_softcap_logp(self):
        g, u = _bf(4, 256), _bf(4, 256)
        ref = (torch.nn.functional.silu(g.float()) * u.float()).to(torch.bfloat16)
        assert torch.allclose(fb.swiglu(g, u).float(), ref.float(), atol=2e-2)
        h, w = _bf(5, 64), _bf(256, 64)
        vocab = 250
        targets = torch.randint(0, vocab, (5,), device=DEVICE)
        logp = fb.lm_head_softcap_logprob(
            h, w, targets, multiplier=0.19611613513818404, softcap=20.0, vocab_size=vocab
        )
        z = (h.double() @ w.double().t())[:, :vocab]
        zhat = 20.0 * torch.tanh(z * 0.19611613513818404 / 20.0)
        ref_logp = torch.log_softmax(zhat, -1).gather(1, targets[:, None].long()).squeeze(1)
        assert torch.allclose(logp.double(), ref_logp, atol=1e-4)
        assert torch.equal(
            logp[1:2],
            fb.lm_head_softcap_logprob(
                h[1:2],
                w,
                targets[1:2],
                multiplier=0.19611613513818404,
                softcap=20.0,
                vocab_size=vocab,
            ),
        )


@gpu
class TestFallbackAttention:
    @pytest.mark.parametrize("window", [None, 40])
    def test_paged_equals_varlen_equals_decode(self, window):
        nh, nkv, d, length = 4, 2, 32, 70
        q = _bf(length, nh, d)
        k = _bf(length, nkv, d)
        v = _bf(length, nkv, d)
        gate = _bf(length, nh * d)
        cu = torch.tensor([0, length], dtype=torch.int32, device=DEVICE)
        pos = torch.arange(length, device=DEVICE)
        y_var, o_var, lse_var = fb.gated_attention_fwd_varlen(
            q, k, v, gate, cu, cu, pos, window=window
        )
        # fp32 reference (same mask) for correctness.
        scores = (
            torch.einsum("qhd,khd->hqk", q.float(), k.float().repeat_interleave(nh // nkv, 1))
            * d**-0.5
        )
        qi, ki = pos[:, None], pos[None, :]
        visible = ki <= qi
        if window is not None:
            visible &= ki > qi - window
        scores = scores.masked_fill(~visible, float("-inf"))
        probs = torch.softmax(scores, -1)
        ref_o = torch.einsum("hqk,khd->qhd", probs, v.float().repeat_interleave(nh // nkv, 1))
        assert torch.allclose(o_var.float().view(length, nh, d), ref_o, atol=3e-2, rtol=3e-2)
        ref_y = (o_var.float() * torch.sigmoid(gate.float())).to(torch.bfloat16)
        assert torch.equal(y_var, ref_y)
        # Paged copy with shuffled pages (page = 16) and a null block 0 never referenced.
        page = 16
        npages = -(-length // page)
        perm = torch.randperm(npages, device=DEVICE) + 1
        k_cache = torch.zeros(npages + 1, page, nkv, d, dtype=torch.bfloat16, device=DEVICE)
        v_cache = torch.zeros_like(k_cache)
        padded_k = torch.zeros(npages * page, nkv, d, dtype=torch.bfloat16, device=DEVICE)
        padded_v = torch.zeros_like(padded_k)
        padded_k[:length], padded_v[:length] = k, v
        k_cache[perm] = padded_k.view(npages, page, nkv, d)
        v_cache[perm] = padded_v.view(npages, page, nkv, d)
        block_table = perm.to(torch.int32)[None, :]
        seq_lens = torch.tensor([length], dtype=torch.int32, device=DEVICE)
        y_pg, o_pg, lse_pg = fb.gated_attention_fwd_paged(
            q, gate, k_cache, v_cache, block_table, seq_lens, cu, window=window
        )
        assert (
            torch.equal(y_pg, y_var) and torch.equal(o_pg, o_var) and torch.equal(lse_pg, lse_var)
        )
        # Decode: one query at a time (chunked prefill of the tail) must reproduce the rows.
        for position in (0, 17, 63, 64, 69):
            q_dec = q[position : position + 1]
            seq_dec = torch.tensor([position + 1], dtype=torch.int32, device=DEVICE)
            cu_dec = torch.tensor([0, 1], dtype=torch.int32, device=DEVICE)
            y_dec, _, _ = fb.gated_attention_fwd_paged(
                q_dec,
                gate[position : position + 1],
                k_cache,
                v_cache,
                block_table,
                seq_dec,
                cu_dec,
                window=window,
            )
            assert torch.equal(y_dec[0], y_var[position]), f"decode row {position} differs"
        # Two requests in one batch: rows must equal the single-request rows (batch invariance).
        block_table2 = torch.cat([block_table, block_table])
        seq2 = torch.tensor([length, 40], dtype=torch.int32, device=DEVICE)
        cu2 = torch.tensor([0, length, length + 3], dtype=torch.int32, device=DEVICE)
        q2 = torch.cat([q, q[37:40]])
        g2 = torch.cat([gate, gate[37:40]])
        y2, _, _ = fb.gated_attention_fwd_paged(
            q2, g2, k_cache, v_cache, block_table2, seq2, cu2, window=window
        )
        assert torch.equal(y2[:length], y_var)
        assert torch.equal(y2[length:], y_var[37:40])


@gpu
class TestLoRA:
    def test_zero_b_is_bitwise_noop_and_nonzero_b_changes(self):
        state = vs.StrictBindingState(ops=resolve_ops(force_fallback=True))
        x = _bf(6, 512)
        w_col = _bf(256, 512)
        a = _bf(8, 512)
        b_zero = torch.zeros(256, 8, dtype=torch.bfloat16, device=DEVICE)
        base = vs.column_parallel_project(state, x, w_col)
        assert torch.equal(vs.column_parallel_project(state, x, w_col, (a, b_zero)), base)
        b_one = _bf(256, 8)
        assert not torch.equal(vs.column_parallel_project(state, x, w_col, (a, b_one)), base)
        w_row = _bf(128, 512)
        a_row = _bf(8, 512)
        b_row_zero = torch.zeros(128, 8, dtype=torch.bfloat16, device=DEVICE)
        base_row = vs.row_parallel_project(state, x, w_row)
        assert torch.equal(vs.row_parallel_project(state, x, w_row, (a_row, b_row_zero)), base_row)
        # Explicit formula check for the non-zero case (column parallel): the declared unmerged
        # rule y = bf16(acc32(W x) + sum_k B[:, k] z[k]) with z = bf16(acc32(A x)), one rounding.
        acc = fb.det_gemm(x, w_col, out_dtype=torch.float32)
        z = fb.det_gemm(x, a, out_dtype=torch.float32).to(torch.bfloat16)
        loop = torch.zeros_like(acc)
        for k in range(z.shape[1]):
            loop = loop + z[:, k : k + 1].float() * b_one[:, k].float().unsqueeze(0)
        expected = (acc + loop).to(torch.bfloat16)
        assert torch.equal(vs.column_parallel_project(state, x, w_col, (a, b_one)), expected)

    def test_lora_rule_matches_megatron_binding(self):
        """The rollout's strict LoRA forward equals the trainer's (megatron_lora) bit for bit."""

        from rl_engine.models.muse_glimmer.runtime import megatron_lora as ml
        from rl_engine.models.muse_glimmer.runtime.strict_kernels import TPComm

        state = vs.StrictBindingState(ops=probe_ops(resolve_ops(), DEVICE))
        comm = TPComm(None)
        for tokens in (1, 5, 37):
            x = _bf(tokens, 1024)
            w_col, a, b = _bf(512, 1024), _bf(16, 1024) * 0.03, _bf(512, 16) * 0.05
            y = vs.column_parallel_project(state, x, w_col, (a, b))
            y_meg = ml.StrictColumnLinear.apply(x, a.float(), b.float(), comm, 512, w_col)
            assert torch.equal(y, y_meg)
            w_row, a_row, b_row = _bf(256, 1024), _bf(16, 1024) * 0.03, _bf(256, 16) * 0.05
            y = vs.row_parallel_project(state, x, w_row, (a_row, b_row))
            y_meg = ml.StrictRowLinear.apply(x, w_row, a_row.float(), b_row.float(), comm, 1024)
            assert torch.equal(y, y_meg)


@gpu
class TestStrictSiblings:
    def test_probe_keeps_strict_ops_under_cuda_default_device(self):
        """vLLM builds the model under a CUDA default device and bf16 default dtype; the probe must
        not demote any strict op there."""

        previous = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        try:
            with torch.device(DEVICE):
                ops = probe_ops(resolve_ops(), DEVICE)
        finally:
            torch.set_default_dtype(previous)
        assert ops.demotions == [], ops.demotions
        assert ops.all_strict, ops.fallback_names

    def test_probe_and_strict_vs_fallback_agree(self):
        ops = probe_ops(resolve_ops(), DEVICE)
        strict_names = [n for n in OP_NAMES if ops.is_strict(n)]
        if not strict_names:
            pytest.skip("no sibling strict ops present in this worktree")
        fallback = resolve_ops(force_fallback=True)
        if ops.is_strict("det_gemm"):
            a, w = _bf(4, 512), _bf(256, 512)
            ys, yf = ops.det_gemm(a, w, out_dtype=torch.float32), fallback.det_gemm(
                a, w, out_dtype=torch.float32
            )
            assert torch.allclose(ys, yf, atol=1e-3, rtol=1e-4)
        if ops.is_strict("centered_rmsnorm"):
            x, w = _bf(4, 6656), _bf(6656)
            assert torch.allclose(
                ops.centered_rmsnorm(x, w, eps=1e-5).float(),
                fallback.centered_rmsnorm(x, w, eps=1e-5).float(),
                atol=2e-2,
            )
        if ops.is_strict("qknorm_scaled_rope"):
            q, k = _bf(4, 32, 128), _bf(4, 2, 128)
            pos = torch.arange(4, device=DEVICE)
            cos, sin = ops.rope_table(pos, theta=500000.0, dim=128, device=DEVICE)
            qs, ks = ops.qknorm_scaled_rope(q, k, cos, sin, eps=1e-5, qk_scale=3.87)
            qf, kf = fallback.qknorm_scaled_rope(q, k, cos, sin, eps=1e-5, qk_scale=3.87)
            assert torch.allclose(qs.float(), qf.float(), atol=5e-2, rtol=2e-2)
            assert torch.allclose(ks.float(), kf.float(), atol=2e-2, rtol=2e-2)
        if ops.is_strict("softcap_tile_stats"):
            z = torch.randn(3, 25344, device=DEVICE)
            targets = torch.tensor([1, 25344 + 7, 202047], device=DEVICE)
            for tile in (0, 1, 7):
                s_strict = ops.softcap_tile_stats(
                    z,
                    tile,
                    targets,
                    multiplier=0.19611613513818404,
                    softcap=20.0,
                    vocab_size=202048,
                )
                s_fb = fallback.softcap_tile_stats(
                    z,
                    tile,
                    targets,
                    multiplier=0.19611613513818404,
                    softcap=20.0,
                    vocab_size=202048,
                )
                assert torch.allclose(s_strict[0], s_fb[0], atol=1e-5)
                assert torch.allclose(s_strict[1], s_fb[1], rtol=1e-4)
                assert torch.equal(s_strict[3].bool(), s_fb[3].bool())


@gpu
class TestTPSimulation:
    def test_tp2_tp4_tp8_layer_and_logp_equal_tp1(self):
        """Binding arithmetic at simulated TP 2/4/8 (threads + in-process all-gather, real shapes,
        KV replicated at TP4/8, LoRA sharded) equals TP1 bit for bit."""

        from rl_engine.models.muse_glimmer.runtime.vllm_tp_sim import simulate

        report = simulate((2, 4, 8), tokens=19, device=DEVICE)
        bad = [
            (e["tp"], e["lora"], e.get("errors"), e.get("equal_to_tp1"))
            for e in report["results"]
            if not e["all_equal"]
        ]
        assert report["all_equal"], bad


# ------------------------------------------------------------------------------------------
# Wave B: per-rank heads, fail-closed engine features, strict sampling logits, LoRA hash
# recomputation, results labels
# ------------------------------------------------------------------------------------------


class TestPerRankHeads:
    @pytest.mark.parametrize("tp,expected", [(1, (32, 2)), (2, (16, 1)), (4, (8, 1)), (8, (4, 1))])
    def test_rank_head_counts(self, tp, expected):
        assert vs.rank_head_counts(tp) == expected
        with pytest.raises(ValueError):
            vs.rank_head_counts(3)

    @pytest.mark.parametrize("tp", [1, 2, 4, 8])
    def test_kv_ownership_matches_vllm_sharding(self, tp):
        for rank in range(tp):
            state = vs.StrictBindingState(ops=resolve_ops(force_fallback=True))
            state.tp_size, state.tp_rank = tp, rank
            own = state.kv_ownership()
            hq, hkv = vs.rank_head_counts(tp)
            assert own["q_heads"] == [rank * hq, (rank + 1) * hq]
            kv = (2 * rank) // tp if tp > 1 else 0  # vLLM QKVParallelLinear: head (2r)//t
            assert own["kv_heads_cached"] == [kv, kv + hkv] == own["kv_heads_projected"]
            assert own["kv_heads_replicated"] == (tp > 2)
            # every q head of this rank reads its cached kv head (GQA group of 16)
            assert all(h // 16 in range(kv, kv + hkv) for h in range(*own["q_heads"]))


class _NS:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _engine_config(**overrides):
    cache = _NS(
        enable_prefix_caching=False,
        kv_offloading_size=None,
        kv_sharing_fast_prefill=False,
        cache_dtype="auto",
    )
    parallel = _NS(
        decode_context_parallel_size=1, prefill_context_parallel_size=1, pipeline_parallel_size=1
    )
    cfg = _NS(
        cache_config=cache,
        parallel_config=parallel,
        model_config=_NS(logprobs_mode="raw_logprobs"),
        kv_transfer_config=None,
        speculative_config=None,
    )
    for path, value in overrides.items():
        target = cfg
        parts = path.split(".")
        for part in parts[:-1]:
            target = getattr(target, part)
        setattr(target, parts[-1], value)
    return cfg


class TestFailClosedEngineFeatures:
    """vllm_runtime:R2 — features that address KV pages in vLLM's native layout are rejected."""

    def test_clean_config_passes(self):
        verdicts = vs.check_strict_engine_config(_engine_config())
        assert verdicts["prefix_caching_off"] == "passed"
        assert all(v == "passed" for v in verdicts.values()), verdicts

    @pytest.mark.parametrize(
        "path,value",
        [
            ("cache_config.enable_prefix_caching", True),
            ("cache_config.kv_offloading_size", 4.0),
            ("kv_transfer_config", object()),
            ("speculative_config", object()),
            ("cache_config.kv_sharing_fast_prefill", True),
            ("parallel_config.decode_context_parallel_size", 2),
            ("parallel_config.prefill_context_parallel_size", 2),
            ("model_config.logprobs_mode", "processed_logprobs"),
        ],
    )
    def test_kv_page_features_raise(self, path, value):
        with pytest.raises(RuntimeError, match="fail closed"):
            vs.check_strict_engine_config(_engine_config(**{path: value}))

    def test_native_page_writers_raise(self):
        state = vs.StrictBindingState(ops=resolve_ops(force_fallback=True))
        guard = vs._forbid(state, "copy_kv_cache_blocks_inplace")
        with pytest.raises(RuntimeError, match="native layout"):
            guard(object(), object())
        assert state.call_counts["forbidden.copy_kv_cache_blocks_inplace"] == 1


class TestFailClosedSampling:
    """vllm_runtime:R6 — requests that would return native numbers raise."""

    def test_plain_request_passes(self):
        vs.check_sampling_request(
            num_logprobs=0, max_token_ids=0, top_k=None, top_p=None, logprobs_mode="raw_logprobs"
        )
        vs.check_prompt_logprobs_request(0, "raw_logprobs")

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"num_logprobs": 5},
            {"max_token_ids": 3},
            {"top_k": torch.tensor([5])},
            {"top_p": torch.tensor([0.9])},
            {"logprobs_mode": "processed_logprobs"},
        ],
    )
    def test_native_columns_raise(self, kwargs):
        base = dict(
            num_logprobs=0, max_token_ids=0, top_k=None, top_p=None, logprobs_mode="raw_logprobs"
        )
        base.update(kwargs)
        with pytest.raises(RuntimeError, match="fail closed"):
            vs.check_sampling_request(**base)

    @pytest.mark.parametrize("k", [1, 5, -1])
    def test_prompt_topk_raises(self, k):
        with pytest.raises(RuntimeError, match="fail closed"):
            vs.check_prompt_logprobs_request(k, "raw_logprobs")


def _logits_state(hidden: int = 512, seed: int = 0):
    state = vs.StrictBindingState(ops=probe_ops(resolve_ops(), DEVICE))
    gen = torch.Generator(device="cpu").manual_seed(seed)
    w = torch.zeros(vs.PADDED_VOCAB, hidden, dtype=torch.bfloat16, device=DEVICE)
    w[: vs.REAL_VOCAB] = (torch.randn(vs.REAL_VOCAB, hidden, generator=gen) * 0.5).to(
        DEVICE, torch.bfloat16
    )
    state.lm_head = _NS(weight=w)
    state.cfg.update(
        {"output_multiplier": 0.19611613513818404, "final_logit_softcapping": 20.0}
    )
    return state


@gpu
class TestStrictSamplingLogits:
    """vllm_runtime:R5 — token selection reads strict fp32 logits that agree with strict O7."""

    def test_logits_match_o7_and_are_batch_invariant(self):
        state = _logits_state()
        if not state.ops.all_strict:
            pytest.skip("strict sibling ops not importable")
        h = _bf(9, 512) * 4.0
        logits, z_tiles = vs.strict_logits(state, h)
        assert logits.shape == (9, vs.REAL_VOCAB) and logits.dtype == torch.float32
        assert len(z_tiles) == 8
        targets = torch.tensor([0, 5, 25343, 25344, 101000, 150000, 202047, 7, 9], device=DEVICE)
        logp = vs.strict_selected_logp(state, h, targets)
        # the sampler path (stashed tiles) and the prefill path give the same bits
        assert torch.equal(vs.logp_from_z_tiles(state, z_tiles, targets), logp)
        # the logit of the target is exactly the zhat O7 used (same kernel soft cap)
        picked = logits.gather(1, targets[:, None]).squeeze(1)
        m_lse = torch.logsumexp(logits.double(), dim=-1)
        assert torch.allclose(picked.double() - m_lse, logp.double(), atol=2e-5)
        # batch invariance: one row alone == the same row inside the batch
        for row in (0, 4, 8):
            alone, _ = vs.strict_logits(state, h[row : row + 1])
            assert torch.equal(alone[0], logits[row])
        # greedy choice from strict logits and its strict logp is <= 0 (O7 sign invariant)
        greedy = logits.argmax(dim=-1)
        assert (vs.strict_selected_logp(state, h, greedy) <= 0).all()

    def test_empty_batch(self):
        state = _logits_state()
        logits, tiles = vs.strict_logits(state, _bf(0, 512))
        assert logits.shape == (0, vs.REAL_VOCAB) and tiles == []


def _lora_sd(layer=0, rank=2, seed=0):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    sd = {}
    for name in ("q_proj", "o_proj"):
        k, n = vs.LORA_SHAPES[name]
        sd[f"layers.{layer}.{name}.lora_A"] = torch.randn(rank, k, generator=gen).to(
            torch.bfloat16
        )
        sd[f"layers.{layer}.{name}.lora_B"] = torch.randn(n, rank, generator=gen).to(
            torch.bfloat16
        )
    return sd


class TestLoRAHashRecomputed:
    """e2e_bridge:R4 — provenance()["lora_sha256"] is the hash every TP worker recomputed."""

    def _bound_state(self, monkeypatch):
        state = vs.StrictBindingState(ops=resolve_ops(force_fallback=True))
        state.model = torch.nn.Linear(2, 2)  # parameters() -> the device the slots go to
        monkeypatch.setattr(vs, "_STATE", state)
        monkeypatch.delenv(vs.READBACK_ENV, raising=False)
        return state

    def test_worker_reports_recomputed_not_declared(self, monkeypatch):
        from rl_engine.models.muse_glimmer.e2e.sync import lora_sha256

        state = self._bound_state(monkeypatch)
        sd = _lora_sd()
        digest = lora_sha256(sd)
        reply = vs.worker_load_lora(None, sd, digest)
        assert reply["recomputed_sha256"] == digest == state.lora_sha256
        assert state.lora_sha256_declared == digest
        prov = state.provenance()
        assert prov["lora_sha256"] == digest and "recomputed" in prov["lora_sha256_source"]

    def test_tampered_tensor_fails_and_keeps_previous_adapter(self, monkeypatch):
        from rl_engine.models.muse_glimmer.e2e.sync import lora_sha256

        state = self._bound_state(monkeypatch)
        sd = _lora_sd()
        digest = lora_sha256(sd)
        vs.worker_load_lora(None, sd, digest)
        tampered = dict(_lora_sd(seed=1))
        with pytest.raises(RuntimeError, match="fail closed"):
            vs.worker_load_lora(None, tampered, lora_sha256(sd))
        assert state.lora_sha256 == digest

    def test_clear(self, monkeypatch):
        state = self._bound_state(monkeypatch)
        reply = vs.worker_load_lora(None, {}, "")
        assert reply["recomputed_sha256"] is None and state.lora_sha256 is None
        with pytest.raises(RuntimeError):
            vs.worker_load_lora(None, {}, "ab" * 32)

    def test_engine_aggregation_raises_on_disagreement_or_echo(self):
        from rl_engine.models.muse_glimmer.e2e.sync import WeightSyncError
        from rl_engine.models.muse_glimmer.runtime.strict_rollout_engine import (
            aggregate_worker_lora_hashes,
        )

        a, b = "a" * 64, "b" * 64
        assert aggregate_worker_lora_hashes(
            [{"lora_sha256": a}, {"lora_sha256": a}], key="lora_sha256", driver_sha256=a
        ) == (a, [a, a])
        assert aggregate_worker_lora_hashes(
            [{"lora_sha256": None}], key="lora_sha256", driver_sha256=None
        ) == (None, [None])
        with pytest.raises(WeightSyncError, match="different"):
            aggregate_worker_lora_hashes(
                [{"lora_sha256": a}, {"lora_sha256": b}], key="lora_sha256", driver_sha256=a
            )
        with pytest.raises(WeightSyncError, match="driver"):
            aggregate_worker_lora_hashes(
                [{"lora_sha256": b}], key="lora_sha256", driver_sha256=a
            )
        with pytest.raises(WeightSyncError, match="failed"):
            aggregate_worker_lora_hashes(
                [{"error": "boom"}], key="lora_sha256", driver_sha256=a
            )
        with pytest.raises(WeightSyncError):
            aggregate_worker_lora_hashes([], key="lora_sha256", driver_sha256=a)


class TestNativeLoRARecomputed:
    """vllm_runtime:R10 — native mode recomputes the hash from vLLM's loaded LoRA tensors."""

    def _lora_model(self, sd, scaling=1.0):
        from vllm.lora.lora_weights import LoRALayerWeights, PackedLoRALayerWeights

        def layer(name, slot):
            a = sd[f"layers.0.{slot}.lora_A"].clone()
            b = sd[f"layers.0.{slot}.lora_B"].clone()
            return LoRALayerWeights(name, a.shape[0], a.shape[0], a, b, scaling=scaling)

        prefix = "language_model.model.layers.0."
        loras = {
            prefix + "self_attn.qkv_proj": PackedLoRALayerWeights.pack(
                [layer(prefix + "self_attn.q_proj", "q_proj"), None, None]
            ),
            prefix + "self_attn.o_proj": layer(prefix + "self_attn.o_proj", "o_proj"),
        }
        return _NS(loras=loras)

    def test_hash_from_vllm_tensors(self):
        pytest.importorskip("vllm")
        from rl_engine.models.muse_glimmer.e2e.sync import lora_sha256
        from rl_engine.models.muse_glimmer.runtime import vllm_worker_tools as tools

        sd = _lora_sd()
        model = self._lora_model(sd)
        manager = _NS(_adapter_manager=_NS(get_adapter=lambda i: model if i == 3 else None))
        worker = _NS(model_runner=_NS(lora_manager=manager))
        report = tools.worker_native_lora_report(worker, 3)
        assert report["recomputed_sha256"] == lora_sha256(sd)
        assert report["slots"] == 2 and report["slots_nonzero_b"] == 2
        with pytest.raises(RuntimeError, match="not loaded"):
            tools.worker_native_lora_report(worker, 4)

    def test_scaled_adapter_is_rejected(self):
        pytest.importorskip("vllm")
        from rl_engine.models.muse_glimmer.runtime import vllm_worker_tools as tools

        with pytest.raises(RuntimeError, match="scaling"):
            tools.canonical_state_from_lora_model(self._lora_model(_lora_sd(), scaling=2.0))


class TestWorkerPaths:
    """vllm_runtime:R11 — a worker importing rl_engine from another tree fails the engine."""

    def test_check_worker_paths(self):
        from rl_engine.models.muse_glimmer.runtime.strict_rollout_engine import check_worker_paths

        check_worker_paths("/a/rl_engine/__init__.py", ["/a/rl_engine/__init__.py"] * 2)
        with pytest.raises(RuntimeError, match="stale"):
            check_worker_paths("/a/rl_engine/__init__.py", ["/b/rl_engine/__init__.py"])
        with pytest.raises(RuntimeError):
            check_worker_paths(None, [None])


@gpu
class TestLoRARuleCheckIsIndependent:
    """vllm_runtime:R8 — the 'formula' of worker_check_lora_rule does not reuse the binding's
    helpers: a broken binding helper must make binding_equals_formula False."""

    def _mock_model(self, hidden=512, q_size=512, kv_size=128, ffn=1024):
        def lin(n, k):
            return _NS(weight=_bf(n, k) * 0.05)

        attn = _NS(
            qkv_proj=lin(q_size + 2 * kv_size, hidden),
            output_gate_proj=lin(q_size, hidden),
            o_proj=lin(hidden, q_size),
            q_size=q_size,
            kv_size=kv_size,
        )
        mlp = _NS(gate_up_proj=lin(2 * ffn, hidden), down_proj=lin(hidden, ffn))
        return _NS(model=_NS(layers=[_NS(self_attn=attn, mlp=mlp)]))

    def test_formula_detects_a_broken_binding_helper(self, monkeypatch):
        state = vs.StrictBindingState(ops=probe_ops(resolve_ops(), DEVICE))
        if not state.ops.all_strict:
            pytest.skip("strict sibling ops not importable")
        state.model = self._mock_model()
        state.lora = {
            0: {
                "q_proj": (_bf(16, 512) * 0.03, _bf(512, 16) * 0.05),
                "o_proj": (_bf(16, 512) * 0.03, _bf(512, 16) * 0.05),
            }
        }
        monkeypatch.setattr(vs, "_STATE", state)
        good = vs.worker_check_lora_rule(None, 0)
        assert good["all_equal"], good
        original = vs._padded_partial

        def broken(*args, **kwargs):
            out = original(*args, **kwargs)
            return out * (1.0 + 2.0**-10)

        monkeypatch.setattr(vs, "_padded_partial", broken)
        bad = vs.worker_check_lora_rule(None, 0)
        assert not bad["slots"]["q_proj"]["binding_equals_formula"]
        assert not bad["all_equal"]


class TestCollectLabels:
    """vllm_runtime:R7 — comparison labels come from the comparison key."""

    def test_labels(self):
        from rl_engine.models.muse_glimmer.runtime import vllm_collect_results as cr

        seq = [{"id": "s", "logprobs_hex": ["bf800000"]}]
        a = {"mode": "strict", "tp": 1, "sequences": seq, "lora_sha256": "x" * 64}
        b = {"mode": "strict", "tp": 1, "sequences": seq, "lora_sha256": None}
        out = cr.compare_xrt(a, b, a_label="strict lora tp1", b_label="strict tp1")
        assert (out["a"], out["b"]) == ("strict lora tp1", "strict tp1")
        assert cr.xrt_label(a) == "strict lora tp1" and cr.xrt_label(b) == "strict tp1"
        ref = {"runtime": "reference_chain", "mode": "strict", "tp": 1, "backend": "x mode=train"}
        assert cr.xrt_label(ref) == "reference chain train"


# ------------------------------------------------------------------------------------------
# E2E (opt-in)
# ------------------------------------------------------------------------------------------


E2E = pytest.mark.skipif(
    os.getenv("RL_KERNEL_MUSEGLIMMER_E2E", "") != "1" or not CUDA,
    reason="set RL_KERNEL_MUSEGLIMMER_E2E=1 with the checkpoint and a GPU lease",
)
_XRT = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "reports", "muse_glimmer_poc", "data",
    "xrt_tokens.json",
)


def _bits(values):
    import struct

    return [struct.pack(">f", float(v)).hex() for v in values]


@E2E
def test_strict_engine_protocol_with_real_lora(tmp_path):
    """Drive StrictRolloutEngine through e2e.engines.RolloutEngineLike with a real rank-16 LoRA."""

    import inspect

    from rl_engine.models.muse_glimmer.e2e.engines import RolloutEngineLike, coerce_samples
    from rl_engine.models.muse_glimmer.e2e.sync import (
        LoRAPackage,
        WeightSyncError,
        lora_sha256,
        transfer_lora,
        validate_lora_state_dict,
    )
    from rl_engine.models.muse_glimmer.runtime.run_vllm_strict import random_lora_state_dict
    from rl_engine.models.muse_glimmer.runtime.strict_rollout_engine import StrictRolloutEngine

    # Structural conformance: same method names, parameter names and kinds as the protocol.
    for name in ("generate", "logprobs", "load_lora", "provenance", "close"):
        want = inspect.signature(getattr(RolloutEngineLike, name))
        have = inspect.signature(getattr(StrictRolloutEngine, name))
        assert [(p.name, p.kind) for p in want.parameters.values()] == [
            (p.name, p.kind) for p in have.parameters.values()
        ], name

    with open(_XRT, encoding="utf-8") as fh:
        xrt = json.load(fh)
    prompts = [
        s["token_ids"][: s["completion_start"]] for s in xrt["sequences"][:2]
    ]
    model = os.path.expanduser(
        os.getenv("RL_KERNEL_MUSEGLIMMER_MODEL", "~/models/Muse-Glimmer-30B")
    )
    engine = StrictRolloutEngine(
        model, tensor_parallel_size=1, gpu_ids=[0], strict=True, max_model_len=1024, seed=3
    )
    try:
        assert engine.provenance()["lora_sha256"] is None
        state = random_lora_state_dict(seed=11, rank=16, b_std=0.01)
        summary = validate_lora_state_dict(state)
        assert summary["num_modules"] == 52 * 8 and summary["ranks"] == [16]
        package = LoRAPackage(state, lora_sha256(state), step=0)
        info = transfer_lora(package, engine, via_file=tmp_path / "lora.pt")
        prov = engine.provenance()
        assert info["identity"] and prov["lora_sha256"] == package.sha256
        assert prov["lora_sha256_per_worker"] == [package.sha256]
        assert prov["workers"][0]["lora_sha256_declared"] == package.sha256
        assert "recomputed" in prov["lora_sha256_source"]
        with pytest.raises(WeightSyncError):
            engine.load_lora(state, "0" * 64)
        assert engine.provenance()["lora_sha256"] == package.sha256

        groups = engine.generate(prompts, n=2, max_tokens=12, temperature=1.0)
        samples = coerce_samples(groups)
        assert len(samples) == len(prompts) and all(len(g) == 2 for g in samples)
        sequences, starts, decoded = [], [], []
        for prompt, group in zip(prompts, samples):
            for sample in group:
                assert len(sample.token_ids) == len(sample.logprobs) > 0
                for v in sample.logprobs:
                    # O7 sign invariant (restored after c88607f: the softcap kernels launch with
                    # enable_fp_fusion=False, so a near-certain token's logp can no longer be
                    # +1..2 ulp above 0; issue vllm_runtime:R9).
                    assert v <= 0.0 and float(torch.tensor(v, dtype=torch.float32)) == v
                sequences.append(prompt + sample.token_ids)
                starts.append(len(prompt))
                decoded.append(sample.logprobs)
        rescored = engine.logprobs(sequences, starts)
        assert [_bits(r) for r in rescored] == [_bits(d) for d in decoded]

        # A second sync (new adapter) replaces the first one on every rank.
        state2 = random_lora_state_dict(seed=12, rank=16, b_std=0.01)
        transfer_lora(LoRAPackage(state2, lora_sha256(state2), step=1), engine)
        assert engine.provenance()["lora_sha256"] == lora_sha256(state2)
        again = engine.logprobs(sequences, starts)
        assert [_bits(r) for r in again] != [_bits(r) for r in rescored]
    finally:
        engine.close()


@pytest.mark.skipif(
    os.getenv("RL_KERNEL_MUSEGLIMMER_E2E", "") != "1" or not CUDA,
    reason="set RL_KERNEL_MUSEGLIMMER_E2E=1 with the checkpoint and a GPU lease",
)
def test_strict_engine_decode_equals_prefill():
    from rl_engine.models.muse_glimmer.runtime.strict_rollout_engine import StrictRolloutEngine

    model = os.path.expanduser(
        os.getenv("RL_KERNEL_MUSEGLIMMER_MODEL", "~/models/Muse-Glimmer-30B")
    )
    with open(_XRT, encoding="utf-8") as fh:
        xrt = json.load(fh)
    long_seq = next(s for s in xrt["sequences"] if s["id"] == "gsm8k_long_window")
    long_prompt = long_seq["token_ids"][: long_seq["completion_start"]]
    assert len(long_prompt) > 2048  # decode positions are past the 2048-key window
    engine = StrictRolloutEngine(
        model, tensor_parallel_size=1, gpu_ids=[0], strict=True, max_model_len=2560, seed=0
    )
    try:
        prompts = [[200000, 791, 6864, 315, 9822, 374], long_prompt]
        for temperature in (0.0, 1.0):
            samples = engine.generate(prompts, n=1, max_tokens=24, temperature=temperature)
            sequences = [p + s[0].token_ids for p, s in zip(prompts, samples)]
            again = engine.logprobs(sequences, [len(p) for p in prompts])
            for got, sample in zip(again, samples):
                assert _bits(got) == _bits(sample[0].logprobs)
                assert all(v <= 0.0 for v in got)
        prov = engine.provenance()
        assert prov["workers"][0]["hooks"]
        assert prov["workers"][0]["call_counts"]["compute_logits.strict"] > 0
    finally:
        engine.close()
