# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Tests for the Muse-Glimmer drift ablation harness (RFC #501 row ``museglimmer_ablation``).

CPU: metric arithmetic (incl. top-1 ties and the full-vocabulary KL), padded-batch / chunk
layouts, the shared token file, the settings registry's one-factor structure, the merge
(synthetic parts: null-control naming, baseline gating, derived one-factor records, env keys),
and the HF drivers (alone / L-1 prefill / chunked / decode / padded right+left) on a tiny random
``MuseGlimmerTextModel`` in fp32 -- they must agree to fp32 rounding, which proves the row
bookkeeping (offsets, positions, sliding-window cache) before any real-weight run; factor context
managers must restore global state.  CUDA: the ``rope_inv_freq_cuda`` factor really evaluates the
pow on the GPU; (tiny config, golden strict ops) the packed / padded / paged-batch /
continuous-batching strict runners must be byte-equal to the reference's own single-sequence
``logprobs(mode="train")``, ``packed_pass`` must pass (and a perturbed copy must fail) the
self-check against the reference's public paths, and ``ablation_strict`` may use public
reference names only.
"""

from __future__ import annotations

import ast
import math
import pathlib
import struct
import types

import pytest
import torch

from rl_engine.models.muse_glimmer.ablation import ablation_hf as A

REPO = pathlib.Path(__file__).resolve().parents[3]
TOKENS = REPO / "reports/muse_glimmer_poc/data/xrt_tokens.json"
ABLATION_DIR = REPO / "rl_engine/models/muse_glimmer/ablation"
NEEDS_CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")


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
    assert agg["mean_kl_full"] is None and agg["argmax_flips_unique"] is None


def test_argmax_flips_split_at_top1_ties():
    """A flip where either run's top-1 logit is tied is a tie-break, counted separately."""
    lp = torch.tensor([-1.0, -1.0, -1.0, -1.0])
    base_am, var_am = torch.tensor([3, 5, 7, 9]), torch.tensor([3, 6, 8, 10])
    base_gap = torch.tensor([0.5, 0.0, 0.25, 0.125])  # position 1: baseline tie
    var_gap = torch.tensor([0.5, 0.25, 0.0, 0.0625])  # position 2: variant tie
    r = A.compare_tokens(lp, lp.clone(), base_am, var_am, base_gap=base_gap, var_gap=var_gap)
    assert r["argmax_flips"] == 3 and r["argmax_flip"] == [0, 1, 1, 1]
    assert r["argmax_flips_at_ties"] == 2 and r["argmax_flips_unique"] == 1
    assert r["top1_tie"] == [0, 1, 1, 0] and r["n_top1_ties"] == 2
    assert r["var_argmax"] == [3, 6, 8, 10]
    agg = A.aggregate([r, r])
    assert (agg["argmax_flips"], agg["argmax_flips_unique"], agg["argmax_flips_at_ties"]) == \
        (6, 2, 4)


def test_kl_rows_exact_zero_and_value():
    gen = torch.Generator().manual_seed(0)
    a = torch.log_softmax(torch.randn(70, 33, generator=gen), -1)
    b = torch.log_softmax(torch.randn(70, 33, generator=gen), -1)
    assert torch.equal(A.kl_rows(a, a.clone()), torch.zeros(70, dtype=torch.float64))
    want = (a.double().exp() * (a.double() - b.double())).sum(-1)
    torch.testing.assert_close(A.kl_rows(a, b), want, rtol=0, atol=1e-12)
    r = A.compare_tokens(a[:, 0], b[:, 0], kl=A.kl_rows(a, b))
    assert r["mean_kl_full"] == pytest.approx(float(want.mean())) and len(r["kl_full"]) == 70


def test_hex_roundtrip_is_exact():
    x = torch.tensor([-0.0, 0.0, -1e-30, -33.5, float("nan"), -1.1])
    assert torch.equal(A.fp32_bits(A.hex_to_fp32(A.fp32_hex(x))), A.fp32_bits(x))


def _rec(setting: str, sid: str, logp, am, gap, base=None) -> dict:
    b = base if base is not None else torch.zeros(len(logp))
    r = A.compare_tokens(b, torch.tensor(logp), torch.zeros(len(am), dtype=torch.long),
                         torch.tensor(am), base_gap=torch.ones(len(am)),
                         var_gap=torch.tensor(gap))
    r.update({"engine": "hf", "setting": setting, "sequence": sid, "factor": "f",
              "factor_id": 3, "wall_ms": 1.0})
    return r


def test_derive_one_factor_records_from_stored_values():
    """Chunked vs the L-1 prefill (R2): computed from the two settings' exact hex values."""
    recs = [
        _rec("prefill_L_minus_1", "s0", [-1.0, -2.0, -3.0], [1, 2, 3], [0.5, 0.0, 0.5]),
        _rec("chunked_prefill_64", "s0", [-1.0, -2.5, -3.0], [1, 4, 5], [0.5, 0.5, 0.5]),
        _rec("prefill_L_minus_1", "s1", [-0.5], [7], [0.25]),
        _rec("chunked_prefill_64", "s1", [-0.5], [7], [0.25]),
    ]
    settings = [{"engine": "hf", "setting": "chunked_prefill_64", "factor": "chunked_prefill",
                 "factor_id": 3, "one_factor_ref": "prefill_L_minus_1",
                 "one_factor": "chunked_prefill"},
                {"engine": "hf", "setting": "prefill_L_minus_1", "factor": "prefill_length",
                 "factor_id": 3, "one_factor_ref": None}]
    d_recs, d_sets = A.derive_one_factor_records(recs, settings)
    assert [r["setting"] for r in d_recs] == ["chunked_prefill_64__vs__prefill_L_minus_1"] * 2
    s0 = d_recs[0]
    assert s0["baseline"] == "hf/prefill_L_minus_1" and s0["derived_from_stored_values"]
    assert s0["bits_differ"] == [0, 1, 0] and s0["dlogp"] == [0.0, -0.5, 0.0]
    assert s0["argmax_flip"] == [0, 1, 1]
    assert (s0["argmax_flips_at_ties"], s0["argmax_flips_unique"]) == (1, 1)
    assert s0["kl_full"] is None
    assert d_recs[1]["n_bits_differ"] == 0
    (summ,) = d_sets
    assert summ["n_tokens"] == 4 and summ["n_bits_differ"] == 1 and summ["n_factors"] == 1
    assert summ["factor"] == "chunked_prefill" and summ["reference_setting"] == "prefill_L_minus_1"


def test_token_char_spans_and_calc_flags():
    vocab = {0: "Half", 1: " is", 2: " <<", 3: "48", 4: "/2=24", 5: ">>", 6: "24", 7: "."}
    text, spans = A.token_char_spans(lambda ids: "".join(vocab[i] for i in ids), list(range(8)))
    assert text == "Half is <<48/2=24>>24."
    assert [text[a:b] for a, b in spans] == list(vocab.values())
    assert A.calc_annotation_flags(text, spans) == [0, 0, 1, 1, 1, 1, 0, 0]


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


_KNOWN_CONTEXTS = {"sdpa", "tf32_on", "rope_elementwise", "rope_inv_freq_cuda", "bf16_rr_off"}


def test_settings_registry_is_one_factor_consistent():
    """Every HF setting is runnable (known driver / head / contexts) and every multi-factor
    setting names a reference that differs from it in exactly ONE knob (R2: chunking is
    compared against the L-1 prefill, not against the L-token baseline)."""
    by = {s.name: s for s in A.HF_SETTINGS}
    assert len(by) == len(A.HF_SETTINGS)
    assert {1, 2, 3, 4, 5, 6, 7, 9, 10, 11} <= {s.factor_id for s in A.HF_SETTINGS}
    for s in A.HF_SETTINGS:
        assert s.driver in A.DRIVERS and s.head in A.HEAD_MODES, s.name
        assert set(s.contexts) <= _KNOWN_CONTEXTS, s.name
        assert (s.driver == "chunked") == (s.chunk is not None), s.name
        if s.one_factor_ref is None:
            continue
        assert s.n_factors >= 2 and s.one_factor, s.name
        ref = by[s.one_factor_ref]
        knobs = [(s.driver, s.chunk) != (ref.driver, ref.chunk), s.head != ref.head,
                 set(s.contexts) != set(ref.contexts)]
        assert sum(knobs) == 1, (s.name, ref.name)
        if knobs[2]:
            assert len(set(s.contexts) ^ set(ref.contexts)) == 1, s.name
        assert ref.n_factors == s.n_factors - 1, (s.name, ref.name)
    for c in (64, 256, 1024):
        assert by[f"chunked_prefill_{c}"].one_factor_ref == "prefill_L_minus_1"


class _RotNS(types.SimpleNamespace):
    pass


def _rope_model(device: str) -> types.SimpleNamespace:
    cfg = types.SimpleNamespace(rope_parameters={"rope_theta": 500000.0}, head_dim=128,
                                hidden_size=4096, num_attention_heads=32)
    inv = A.hf_rope_inv_freq(cfg).to(device)
    return types.SimpleNamespace(config=cfg, rotary_emb=_RotNS(inv_freq=inv,
                                                                original_inv_freq=inv.clone()))


def test_rope_inv_freq_formula_is_hf_expression_on_cpu():
    model = _rope_model("cpu")
    hf = A.hf_rope_inv_freq(model.config)
    assert hf.device.type == "cpu"
    assert torch.equal(A.fp32_bits(A.rope_inv_freq_formula(model.config, "cpu")),
                       A.fp32_bits(hf))
    with pytest.raises(ValueError, match="CUDA"):
        with A.rope_inv_freq_on(model, "cpu"):
            pass


@NEEDS_CUDA
def test_rope_inv_freq_cuda_context_evaluates_pow_on_the_gpu():
    """R1: the factor must install the table whose pow ran on the GPU, not a copy of HF's CPU
    table (the old harness copied the CPU result, so 'CPU vs CUDA' compared a table to itself)."""
    model = _rope_model("cuda")
    cpu_bits = A.fp32_bits(A.hf_rope_inv_freq(model.config))
    cuda = A.rope_inv_freq_formula(model.config, "cuda")
    assert cuda.device.type == "cuda"
    saved = model.rotary_emb.inv_freq
    with A.rope_inv_freq_on(model, "cuda"):
        got = model.rotary_emb.inv_freq
        assert got.device == saved.device
        assert torch.equal(A.fp32_bits(got), A.fp32_bits(cuda))
        assert torch.equal(A.fp32_bits(model.rotary_emb.original_inv_freq), A.fp32_bits(cuda))
    assert model.rotary_emb.inv_freq is saved
    # Recorded fact of this torch/GPU pair (4 of 64 entries on A100, torch 2.13): the CUDA pow
    # differs from the CPU pow, so the factor is a real one.
    assert int((A.fp32_bits(cuda) != cpu_bits).sum()) > 0


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
    for h, r in zip(A.hidden_batched_decode(model, seqs), ref):  # rollout-like combination
        assert h.shape == r.shape
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


def test_layer_trace_rows_and_diff(tiny_hf):
    model, _lm, seqs = tiny_hf
    base = A.layer_trace(model, seqs, 1, "alone")
    assert list(base) == ["embedding"] + [f"layer_{i}" for i in range(4)] + ["final_norm"]
    assert all(v.shape == (seqs[1].length - 1, 128) for v in base.values())
    same = A.layer_diff(base, A.layer_trace(model, seqs, 1, "alone"), seqs[1].completion_start)
    assert all(r["frac_elems_differ"] == 0.0 and r["first_row_with_diff"] is None for r in same)
    for driver, chunk in (("padded_left", None), ("chunked", 8), ("decode", None)):
        var = A.layer_trace(model, seqs, 1, driver, chunk)
        rows = A.layer_diff(base, var, seqs[1].completion_start)
        assert rows[0]["point"] == "embedding" and rows[0]["max_abs"] == 0.0  # row-local
        assert all(r["max_abs"] < 1e-3 for r in rows), driver


def test_layer_trace_sublayer_and_rope_points(tiny_hf):
    """R6: sub-layer points of layer 0 (execution order, before ``layer_0``) and the RoPE cos
    table, with the same row bookkeeping for every driver."""
    model, _lm, seqs = tiny_hf
    kw = {"rope": True, "sublayers": (0,)}
    base = A.layer_trace(model, seqs, 1, "alone", **kw)
    names = list(base)
    want = (["embedding", "rope_cos", "rope_sin"] + [f"layer_0.{p}" for p in A.SUBLAYER_POINTS]
            + ["layer_0"])
    assert names[: len(want)] == want and names[-1] == "final_norm"
    assert "layer_1.self_attn.q_proj" not in base
    assert [x for x in names if A.is_residual_point(x)] == \
        ["embedding"] + [f"layer_{i}" for i in range(4)] + ["final_norm"]
    n = seqs[1].length - 1
    cfg = model.config
    assert base["rope_cos"].shape == base["rope_sin"].shape == (n, cfg.head_dim)
    assert not torch.equal(base["rope_cos"], base["rope_sin"])
    width = cfg.num_attention_heads * cfg.head_dim
    assert base["layer_0.self_attn.q_proj"].shape == (n, width)
    assert base["layer_0.self_attn.o_proj.input"].shape == (n, width)  # gated attention output
    assert base["layer_0.mlp.gate_proj"].shape == (n, cfg.intermediate_size)
    for driver, chunk in (("padded_right", None), ("chunked", 8), ("decode", None)):
        var = A.layer_trace(model, seqs, 1, driver, chunk, **kw)
        assert list(var) == names, driver
        rows = A.layer_diff(base, var, seqs[1].completion_start)
        assert [r["point"] for r in rows[1:3]] == ["rope_cos", "rope_sin"], driver
        assert rows[1]["max_abs"] == 0.0 and rows[2]["max_abs"] == 0.0, driver
        assert all(r["max_abs"] < 1e-3 for r in rows), driver
    with pytest.raises(ValueError):
        A.LayerTap(model, sublayers=(99,))


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
    with A.enter_contexts(model, ("sdpa", "tf32_on", "rope_elementwise", "bf16_rr_off")):
        assert model.config._attn_implementation == "sdpa"
        assert torch.backends.cuda.matmul.allow_tf32 is True
        assert hf.MuseGlimmerTextRotaryEmbedding.forward is not fwd
    assert torch.backends.cuda.matmul.allow_tf32 == tf
    assert torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction == rr
    assert model.config._attn_implementation == impl
    assert hf.MuseGlimmerTextRotaryEmbedding.forward is fwd
    assert model.rotary_emb.inv_freq is inv
    # the GPU-pow factor fails closed on a CPU model instead of re-installing the CPU table
    with pytest.raises(ValueError, match="CUDA"):
        with A.enter_contexts(model, ("rope_inv_freq_cuda",)):
            pass
    assert model.rotary_emb.inv_freq is inv


# --------------------------------------------------------------------------------------------
# merge (synthetic parts, CPU)
# --------------------------------------------------------------------------------------------
def _fake_scores(seqs, seed: int, *, perturb: set[int] = frozenset()) -> list[A.SeqScore]:
    gen = torch.Generator().manual_seed(seed)
    out = []
    for i, s in enumerate(seqs):
        g = torch.Generator().manual_seed(1000 + i)  # same values for every seed ...
        lp = -torch.rand(s.n_completion, generator=g) * 5
        if i in perturb:  # ... except perturbed sequences
            lp = lp + torch.rand(s.n_completion, generator=gen) * 1e-3
        am = torch.arange(s.n_completion) % 7
        gap = torch.where(torch.arange(s.n_completion) % 5 == 0, 0.0, 0.25)
        out.append(A.SeqScore(lp, am, 1.0, top2_gap=gap))
    return out


def _fake_part(R, kind: str, gpu: str, seqs, base, settings, path: str, commit: str = "c0"):
    from rl_engine.models.muse_glimmer.ablation import ablation_hf as AH

    part = {"part": kind, "commit": commit, "worktree_dirty": False, "argv": ["x", kind],
            "env": {"gpu_lease": gpu}, "tokens": {"path": str(TOKENS)}, "started": "t0",
            "finished": "t1", "status": "complete", "errors": [], "_path": path,
            "baseline": {"name": "b", "wall_ms": 1.0, "peak_mem_gib": 1.0,
                         "per_sequence": AH.baseline_payload(seqs, base)},
            "settings": [], "records": []}
    engine = "strict" if kind == "strict" else "hf"
    for name, var in settings:
        st = {s.name: s for s in AH.HF_SETTINGS}.get(name)
        meta = st.summary() if (st is not None and engine == "hf") else None
        recs = AH.make_records(engine, name, "f", 3, "b", seqs, base, var)
        part["records"].extend(recs)
        part["settings"].append(R._setting_summary(engine, name, "f", 3, "d", recs, {}, meta))
    return part


def test_merge_synthetic_parts(tmp_path):
    """R8/R10: env keyed per part (two parts on one GPU do not collide), null-control naming,
    an extra HF part with a different baseline is not merged, derived one-factor records."""
    from rl_engine.models.muse_glimmer.ablation import run_ablation as R

    seqs, _ = A.load_xrt_tokens(TOKENS)
    base = _fake_scores(seqs, 0)
    l1 = _fake_scores(seqs, 1, perturb={0, 3})
    c64 = _fake_scores(seqs, 2, perturb={0, 3, 8})
    hf = _fake_part(R, "hf", "7", seqs, base,
                    [("prefill_L_minus_1", l1), ("chunked_prefill_64", c64),
                     ("repeat_immediate", _fake_scores(seqs, 3))], "p/hf.json")
    other_gpu = _fake_part(R, "hf_baseline_only", "3", seqs, _fake_scores(seqs, 4),
                           [("repeat_immediate", _fake_scores(seqs, 4))], "p/hf_b3.json")
    same_gpu = _fake_part(R, "hf_baseline_only", "7", seqs, _fake_scores(seqs, 5),
                          [("repeat_immediate", _fake_scores(seqs, 5))], "p/hf_b7.json")
    drifted = _fake_part(R, "hf", "7", seqs, _fake_scores(seqs, 6, perturb={1}),
                         [("sdpa", _fake_scores(seqs, 6))], "p/hf_extra.json")
    strict = _fake_part(R, "strict", "3", seqs, base, [("packed_batch9", base)], "p/st.json")
    strict2 = _fake_part(R, "strict", "7", seqs, base, [], "p/st7.json")
    parts = [hf, other_gpu, same_gpu, drifted, strict, strict2]
    pay = R.merge_parts(parts, str(TOKENS), tokenizer=None)
    assert len(pay["env"]) == len(parts)  # R8: no key collision
    assert all("env" in p for p in pay["parts"])
    names = {(s["engine"], s["setting"]) for s in pay["settings"]}
    assert ("hf", "second_gpu_3_vs_7") in names and ("strict", "second_gpu_7_vs_3") in names
    # two same-GPU controls (the drifted extra HF part first, then the baseline-only part)
    assert ("hf", "new_process_same_gpu_7") in names and ("hf", "new_process_same_gpu_7#2") in names
    assert ("hf", "sdpa") not in names  # its baseline differs: settings not merged
    assert pay["extra_hf_parts"][0]["baseline_bitwise_equal_to_main"] is False
    by = {(s["engine"], s["setting"]): s for s in pay["settings"]}
    assert by[("hf", "new_process_same_gpu_7")]["n_bits_differ"] > 0
    assert by[("hf", "new_process_same_gpu_7#2")]["n_bits_differ"] == 0
    assert by[("hf", "second_gpu_3_vs_7")]["n_bits_differ"] == 0
    d = by[("hf", "chunked_prefill_64__vs__prefill_L_minus_1")]
    want = sum(int((A.fp32_bits(a.logp) != A.fp32_bits(b.logp)).sum()) for a, b in zip(c64, l1))
    assert d["derived_from_stored_values"] and d["n_bits_differ"] == want and d["n_factors"] == 1
    assert by[("hf", "chunked_prefill_64")]["n_factors"] == 2
    assert pay["status"] == "partial" and pay["status_missing"] == ["hf_layers part"]
    assert all(s["n_factors"] == 0 for s in pay["settings"]
               if s["factor"] in ("repeat_null", "second_gpu_null"))
    assert pay["analysis"]["low_p_gold_tokens"]["status"].startswith("NOT RUN")
    assert pay["analysis"]["hf_baseline_top1_ties"]["n"] == sum(
        int((b.top2_gap == 0).sum()) for b in base)
    one = {s["setting"] for s in pay["analysis"]["one_factor_settings"]}
    assert "chunked_prefill_64__vs__prefill_L_minus_1" in one and "chunked_prefill_64" not in one
    assert pay["headline"]
    md = R.markdown_table(pay["settings"])
    assert "chunked_prefill_64__vs__prefill_L_minus_1" in md
    # cross-run repeat against an earlier merged file and an earlier part file
    prev = R.compare_previous(pay, pay, "self")
    assert prev["hf"]["tokens_differ"] == 0 and prev["strict"]["tokens_differ"] == 0
    prev = R.compare_previous(pay, drifted, "p/hf_extra.json")
    assert prev["hf"]["tokens_differ"] == prev["hf"]["per_sequence"]["gsm8k_test_1"] > 0
    assert "strict" not in prev


# --------------------------------------------------------------------------------------------
# strict batch runners (tiny config, golden ops with the strict Triton GEMM) -- CUDA only
# --------------------------------------------------------------------------------------------
def test_ablation_strict_uses_public_reference_names_only():
    """R7: no private reference import and no private attribute access (``x._name``)."""
    tree = ast.parse((ABLATION_DIR / "ablation_strict.py").read_text())
    bad = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
                "rl_engine.models.muse_glimmer.reference"):
            bad += [a.name for a in node.names if a.name.startswith("_")]
        if isinstance(node, ast.Attribute) and node.attr.startswith("_") \
                and not node.attr.startswith("__"):
            bad.append(node.attr)
    assert bad == []


@pytest.fixture(scope="module")
def tiny_strict():
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device (Triton GEMM)")
    from rl_engine.models.muse_glimmer.reference.strict_model import MuseGlimmerStrictModel
    from rl_engine.models.muse_glimmer.reference.weights import MuseGlimmerWeights, StrictConfig

    cfg = StrictConfig.tiny()  # window 100
    w = MuseGlimmerWeights.random(cfg, device="cuda", seed=1, layout="both")
    model = MuseGlimmerStrictModel(w, "golden")
    gen = torch.Generator().manual_seed(5)
    toks = torch.randint(0, cfg.vocab_size, (160,), generator=gen).tolist()
    seqs = [A.SeqSpec("x", tuple(toks[:41]), 30), A.SeqSpec("y", tuple(toks), 120),
            A.SeqSpec("z", tuple(toks[7:90]), 61)]
    return model, seqs


def test_strict_batch_runners_are_byte_equal_to_reference(tiny_strict):
    from rl_engine.models.muse_glimmer.ablation import ablation_strict as S

    model, seqs = tiny_strict
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


@pytest.mark.parametrize("mixed", [False, True])
def test_strict_continuous_batching_staggered_copies(tiny_strict, mixed):
    """R10: copies admitted while others run and finish (batch grows, shrinks, rows move);
    every copy of every sequence is byte-equal to the training forward."""
    from rl_engine.models.muse_glimmer.ablation import ablation_strict as S

    model, seqs = tiny_strict
    base = S.score_alone(model, seqs, "train")
    run = S.score_continuous_batching(model, seqs, copies=4, stagger=3, mixed=mixed)
    assert len(run.per_copy) == 4 and run.max_decode_rows == 12
    assert run.prefill_seqs_per_step[0] == 3 and run.prefill_seqs_per_step[3] == 3
    assert S.copies_bits_differ(base, run) == [[0] * 4] * 3


def test_packed_pass_selfcheck_passes_and_catches_a_fork(tiny_strict, monkeypatch):
    """R7: packed_pass is byte-equal to the reference's public paths, and a re-composition that
    drifts from the reference (here: the o_proj output scaled by one bf16 ulp-ish factor) is
    detected and refused."""
    from rl_engine.models.muse_glimmer.ablation import ablation_strict as S

    model, seqs = tiny_strict
    res = S.selfcheck_packed_pass(model, seqs)
    assert res["all_equal"] and all(res[k]["equal"] for k in (
        "varlen_vs_forward_train", "paged_vs_prefill", "paged_multi_vs_decode_batch"))
    orig = S.row_parallel_gemm
    monkeypatch.setattr(S, "row_parallel_gemm", lambda g, a, w: orig(g, a, w) * 1.0078125)
    with pytest.raises(RuntimeError, match="not byte-equal"):
        S.selfcheck_packed_pass(model, seqs)
    bad = S.selfcheck_packed_pass(model, seqs, raise_on_mismatch=False)
    assert not bad["all_equal"] and bad["varlen_vs_forward_train"]["n_rows_differ"] > 0


def fp(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.int32)
