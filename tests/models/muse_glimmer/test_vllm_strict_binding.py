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
        # Explicit formula check for the non-zero case (column parallel).
        z = fb.det_gemm(x, a)
        delta = fb.det_gemm(z, b_one, k_chunks=1)
        expected = (base.float() + delta.float()).to(torch.bfloat16)
        assert torch.equal(vs.column_parallel_project(state, x, w_col, (a, b_one)), expected)


@gpu
class TestStrictSiblings:
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


# ------------------------------------------------------------------------------------------
# E2E (opt-in)
# ------------------------------------------------------------------------------------------


@pytest.mark.skipif(
    os.getenv("RL_KERNEL_MUSEGLIMMER_E2E", "") != "1" or not CUDA,
    reason="set RL_KERNEL_MUSEGLIMMER_E2E=1 with the checkpoint and a GPU lease",
)
def test_strict_engine_decode_equals_prefill():
    from rl_engine.models.muse_glimmer.runtime.strict_rollout_engine import StrictRolloutEngine

    model = os.path.expanduser(
        os.getenv("RL_KERNEL_MUSEGLIMMER_MODEL", "~/models/Muse-Glimmer-30B")
    )
    engine = StrictRolloutEngine(
        model, tensor_parallel_size=1, gpu_ids=[0], strict=True, max_model_len=512, seed=0
    )
    try:
        prompts = [[200000, 791, 6864, 315, 9822, 374]]
        samples = engine.generate(prompts, n=1, max_tokens=4, temperature=0.0)
        tokens = samples[0][0].token_ids
        again = engine.logprobs([prompts[0] + tokens], [len(prompts[0])])
        assert again[0] == samples[0][0].logprobs
        prov = engine.provenance()
        assert prov["workers"][0]["hooks"]
    finally:
        engine.close()
