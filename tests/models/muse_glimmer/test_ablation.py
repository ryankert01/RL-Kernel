# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Tests for the Muse-Glimmer drift ablation harness (RFC #501 row ``museglimmer_ablation``).

CPU: metric arithmetic, padded-batch / chunk layouts, the shared token file, and the HF drivers
(alone / L-1 prefill / chunked / decode / padded right+left) on a tiny random
``MuseGlimmerTextModel`` in fp32 -- they must agree to fp32 rounding, which proves the row
bookkeeping (offsets, positions, sliding-window cache) before any real-weight run; factor context
managers must restore global state.  CUDA (tiny config, golden strict ops): the packed / padded /
paged-batch / continuous-batching strict runners must be byte-equal to the reference's own
single-sequence ``logprobs(mode="train")``.
"""

from __future__ import annotations

import math
import pathlib
import struct

import pytest
import torch

from rl_engine.models.muse_glimmer.ablation import ablation_hf as A

REPO = pathlib.Path(__file__).resolve().parents[3]
TOKENS = REPO / "reports/muse_glimmer_poc/data/xrt_tokens.json"


# --------------------------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------------------------
def test_compare_tokens_identical_is_zero():
    x = torch.tensor([-0.5, -1.25, -3.0])
    r = A.compare_tokens(x, x.clone(), torch.tensor([1, 2, 3]), torch.tensor([1, 2, 3]))
    assert r["n_bits_differ"] == 0 and r["max_abs_dlogp"] == 0.0 and r["mean_k3"] == 0.0
    assert r["argmax_flips"] == 0 and r["first_differing_token"] is None
    assert r["bits_differ"] == [0, 0, 0] and r["dlogp"] == [0.0, 0.0, 0.0]


def test_compare_tokens_counts_and_k3():
    b = torch.tensor([-0.5, -1.0, -2.0, 0.0])
    v = torch.tensor([-0.5, -1.1, -2.0, -0.0])  # -0.0 vs 0.0: equal values, different bits
    r = A.compare_tokens(b, v, torch.tensor([0, 0, 0, 0]), torch.tensor([0, 5, 0, 0]))
    assert r["n_bits_differ"] == 2 and r["bits_differ"] == [0, 1, 0, 1]
    assert r["first_differing_token"] == 1 and r["argmax_flips"] == 1
    d = float(torch.tensor(-1.1).double() - torch.tensor(-1.0).double())
    assert r["max_abs_dlogp"] == pytest.approx(abs(d))
    assert r["mean_k3"] == pytest.approx((math.exp(d) - d - 1) / 4)
    assert r["logp_hex"][1] == "%08x" % struct.unpack("<I", struct.pack("<f", -1.1))[0]


def test_aggregate_is_token_weighted():
    r1 = A.compare_tokens(torch.zeros(1), torch.full((1,), -1.0))
    r2 = A.compare_tokens(torch.zeros(3), torch.zeros(3))
    for r in (r1, r2):
        r["argmax_flips"] = 0
    agg = A.aggregate([r1, r2])
    assert agg["n_tokens"] == 4 and agg["n_bits_differ"] == 1
    assert agg["mean_abs_dlogp"] == pytest.approx(0.25)
    assert agg["sequences_with_any_diff"] == 1 and agg["argmax_flips"] == 0


# --------------------------------------------------------------------------------------------
# layouts and token file
# --------------------------------------------------------------------------------------------
def _seqs():
    return [A.SeqSpec("a", (5, 6, 7, 8, 9), 3), A.SeqSpec("b", (1, 2, 3), 1)]


def test_padded_batch_right_and_left():
    ids, mask, pos, off = A.padded_batch(_seqs(), "right", pad_id=0)
    assert off == [0, 0] and ids[1].tolist() == [1, 2, 3, 0, 0]
    assert mask[1].tolist() == [1, 1, 1, 0, 0] and pos[1].tolist() == [0, 1, 2, 3, 4]
    ids, mask, pos, off = A.padded_batch(_seqs(), "left", pad_id=0)
    assert off == [0, 2] and ids[1].tolist() == [0, 0, 1, 2, 3]
    assert mask[1].tolist() == [0, 0, 1, 1, 1] and pos[1].tolist() == [0, 0, 0, 1, 2]
    assert pos[0].tolist() == [0, 1, 2, 3, 4]


def test_chunk_plan_covers_range():
    assert A.chunk_plan(5, 2) == [(0, 2), (2, 4), (4, 5)]
    assert A.chunk_plan(64, 64) == [(0, 64)]
    with pytest.raises(ValueError):
        A.chunk_plan(0, 4)


def test_shared_token_file():
    seqs, meta = A.load_xrt_tokens(TOKENS)
    assert len(seqs) == 9 and meta["schema"] == "muse_glimmer_xrt_v1"
    long = [s for s in seqs if s.length > 2048]
    assert len(long) == 1 and long[0].completion_start == 2261 and long[0].length == 2439
    assert sum(s.n_completion for s in seqs) == 933
    assert all(len(s.targets) == s.n_completion for s in seqs)


def test_settings_registry_covers_required_factors():
    ids = {s.factor_id for s in A.HF_SETTINGS}
    assert {1, 2, 3, 4, 5, 6, 7, 9} <= ids  # 8 (second GPU) is a separate process
    names = [s.name for s in A.HF_SETTINGS]
    assert len(names) == len(set(names))
    for want in ("padded_batch9_right", "padded_batch9_left", "kv_cache_decode",
                 "chunked_prefill_64", "chunked_prefill_256", "chunked_prefill_1024", "sdpa",
                 "tf32_on", "softcap_fp32", "logits_fp32", "rope_elementwise"):
        assert want in names


# --------------------------------------------------------------------------------------------
# HF drivers on a tiny random model (CPU, fp32)
# --------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def tiny_hf():
    from rl_engine.models.muse_glimmer.reference.chain_harness import hf_text_model_from_weights
    from rl_engine.models.muse_glimmer.reference.weights import MuseGlimmerWeights, StrictConfig

    cfg = StrictConfig.tiny(num_hidden_layers=4, hidden_size=128, intermediate_size=256,
                            sliding_window=40)
    w = MuseGlimmerWeights.random(cfg, device="cpu", seed=3, layout="nk")
    model, lm_head = hf_text_model_from_weights(w, "cpu")
    model = model.float()
    A._set_rope_buffers(model, A.hf_rope_inv_freq(model.config))  # fp32 like HF __init__
    gen = torch.Generator().manual_seed(0)
    toks = torch.randint(0, cfg.vocab_size, (70,), generator=gen).tolist()
    seqs = [A.SeqSpec("long", tuple(toks), 50), A.SeqSpec("short", tuple(toks[:33]), 20)]
    return model, lm_head.float(), seqs


def test_hf_drivers_agree_in_fp32(tiny_hf):
    model, _lm, seqs = tiny_hf
    ref = [A.hidden_alone(model, s) for s in seqs]
    for s, r in zip(seqs, ref):
        assert r.shape[0] == s.n_completion
        for h in (A.hidden_prefill_minus_last(model, s), A.hidden_chunked(model, s, 16),
                  A.hidden_chunked(model, s, 7), A.hidden_decode(model, s)):
            assert h.shape == r.shape
            torch.testing.assert_close(h, r, atol=1e-4, rtol=1e-4)
    for side in ("right", "left"):
        rows = A.hidden_padded(model, seqs, side)
        for h, r in zip(rows, ref):
            torch.testing.assert_close(h, r, atol=1e-4, rtol=1e-4)


def test_score_all_and_head_modes(tiny_hf):
    model, lm, seqs = tiny_hf
    cfg = model.config
    base, stats = A.score_all(model, lm, cfg, seqs, driver="alone")
    assert stats["wall_ms"] > 0 and len(base) == 2
    for sc, s in zip(base, seqs):
        assert sc.logp.shape == (s.n_completion,) and bool((sc.logp <= 0).all())
    h = A.hidden_alone(model, seqs[0])
    zb = A.head_logits(h.bfloat16(), lm.bfloat16(), cfg, "bf16")
    zs = A.head_logits(h.bfloat16(), lm.bfloat16(), cfg, "softcap_fp32")
    zf = A.head_logits(h.bfloat16(), lm.bfloat16(), cfg, "logits_fp32")
    assert zb.dtype == zs.dtype == zf.dtype == torch.float32
    cap = float(cfg.final_logit_softcapping)
    assert float(zf.abs().max()) <= cap
    torch.testing.assert_close(zs, zf, atol=0.05, rtol=0)
    torch.testing.assert_close(zb, zf, atol=0.2, rtol=0)


def test_rope_elementwise_matches_matmul_on_cpu(tiny_hf):
    model, _lm, _seqs = tiny_hf
    x = torch.zeros(1, dtype=torch.float32)
    pos = torch.arange(0, 3000)[None]
    c0, s0 = model.rotary_emb(x, pos)
    with A.rope_elementwise():
        c1, s1 = model.rotary_emb(x, pos)
    # fp32 matmul with K = 1 is one rounded product, like the elementwise multiply.
    assert torch.equal(c0, c1) and torch.equal(s0, s1)
    c2, _ = model.rotary_emb(x, pos)
    assert torch.equal(c0, c2)  # restored


def test_contexts_restore_global_state(tiny_hf):
    from transformers.models.muse_glimmer import modeling_muse_glimmer as hf

    model, _lm, _seqs = tiny_hf
    tf = torch.backends.cuda.matmul.allow_tf32
    rr = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    impl = model.config._attn_implementation
    fwd = hf.MuseGlimmerTextRotaryEmbedding.forward
    inv = model.rotary_emb.inv_freq
    with A.enter_contexts(model, ("sdpa", "tf32_on", "rope_elementwise", "rope_inv_freq_cuda",
                                  "bf16_rr_off")):
        assert model.config._attn_implementation == "sdpa"
        assert torch.backends.cuda.matmul.allow_tf32 is True
        assert hf.MuseGlimmerTextRotaryEmbedding.forward is not fwd
    assert torch.backends.cuda.matmul.allow_tf32 == tf
    assert torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction == rr
    assert model.config._attn_implementation == impl
    assert hf.MuseGlimmerTextRotaryEmbedding.forward is fwd
    assert model.rotary_emb.inv_freq is inv


# --------------------------------------------------------------------------------------------
# strict batch runners (tiny config, golden ops with the strict Triton GEMM) -- CUDA only
# --------------------------------------------------------------------------------------------
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device (Triton GEMM)")
def test_strict_batch_runners_are_byte_equal_to_reference():
    from rl_engine.models.muse_glimmer.ablation import ablation_strict as S
    from rl_engine.models.muse_glimmer.reference.strict_model import MuseGlimmerStrictModel
    from rl_engine.models.muse_glimmer.reference.weights import MuseGlimmerWeights, StrictConfig

    cfg = StrictConfig.tiny()  # window 100
    w = MuseGlimmerWeights.random(cfg, device="cuda", seed=1, layout="both")
    model = MuseGlimmerStrictModel(w, "golden")
    gen = torch.Generator().manual_seed(5)
    toks = torch.randint(0, cfg.vocab_size, (160,), generator=gen).tolist()
    seqs = [A.SeqSpec("x", tuple(toks[:41]), 30), A.SeqSpec("y", tuple(toks), 120),
            A.SeqSpec("z", tuple(toks[7:90]), 61)]
    base = S.score_alone(model, seqs, "train")
    runs = {layout: S.score_packed_batch(model, seqs, layout, pad_id=1)
            for layout in S.STRICT_BATCH_LAYOUTS}
    runs["paged_prefill"] = S.score_paged_batch_prefill(model, seqs)
    runs["batched_decode"], steps = S.score_batched_decode(model, seqs)
    assert steps == max(s.n_completion for s in seqs) - 1
    runs["decode"] = S.score_alone(model, seqs, "decode")
    runs["chunked"] = S.score_alone(model, seqs, "chunked", 16)
    for name, var in runs.items():
        for b, v, s in zip(base, var, seqs):
            assert v.logp.shape == (s.n_completion,), name
            assert torch.equal(fp(b.logp), fp(v.logp)), (name, s.id)


def fp(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.int32)
