# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Tiny-config gates for the Muse Glimmer strict reference (RFC #501 full_model_chain).

4 layers with the real [L, L, L, G] schedule, hidden 256, 8 heads / 2 kv heads, head dim 32,
FFN 512, vocab 1024 (and 1000 to exercise lm_head padding), window 100 (not a multiple of the
64-key block). Gates: train == prefill == chunked == decode == per-token (``torch.equal``);
backward of the training forward deterministic across runs; HF-equivalent math at bf16 tolerance.
"""

from __future__ import annotations

import pytest
import torch

from rl_engine.models.muse_glimmer.operator_trace import MuseGlimmerSpec
from rl_engine.models.muse_glimmer.reference.backend import resolve_ops
from rl_engine.models.muse_glimmer.reference.chain_harness import (
    PathSpec,
    compare_paths,
    compare_strict_to_hf_outputs,
    default_path_specs,
    hf_text_logits,
    hf_text_model_from_weights,
    localize_first_drift,
    strict_reference_outputs,
)
from rl_engine.models.muse_glimmer.reference.golden_ops import (
    GoldenGemm,
    attention_core,
    tree_sum_last,
)
from rl_engine.models.muse_glimmer.reference.strict_model import (
    MuseGlimmerStrictModel,
    nanmean_loss,
)
from rl_engine.models.muse_glimmer.reference.weights import MuseGlimmerWeights, StrictConfig

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
DEVICE = torch.device("cuda")


def _tokens(n: int, vocab: int, seed: int) -> torch.Tensor:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randint(0, vocab, (n,), generator=gen)


@pytest.fixture(scope="module")
def tiny_model():
    cfg = StrictConfig.tiny()
    weights = MuseGlimmerWeights.random(cfg, device=DEVICE, seed=1, layout="both")
    return MuseGlimmerStrictModel(weights, "golden")


# -------------------------------------------------------------------------------------------
# config / naming
# -------------------------------------------------------------------------------------------
def test_official_config_node_names_match_operator_trace():
    spec = MuseGlimmerSpec.from_manifest()
    cfg = StrictConfig.from_spec(spec)
    assert cfg.node_names() == spec.node_names()
    for name in cfg.node_names():
        assert cfg.node_kind(name) == spec.node_kind(name)
    assert cfg.padded_vocab_size == 202752 and cfg.sliding_window == 2048


def test_tiny_config_has_real_schedule_and_padding():
    cfg = StrictConfig.tiny()
    assert list(cfg.layer_types) == ["sliding_attention"] * 3 + ["full_attention"]
    assert cfg.layer_rope_theta() == [500000.0, 500000.0, 500000.0, 0.0]
    assert StrictConfig.tiny(vocab_size=1000).padded_vocab_size == 1024


# -------------------------------------------------------------------------------------------
# building blocks
# -------------------------------------------------------------------------------------------
def test_tree_sum_is_row_independent_and_exact_on_integers():
    x = torch.randint(-8, 8, (37, 6656), device=DEVICE).float()
    full = tree_sum_last(x)
    for i in (0, 5, 36):
        assert torch.equal(tree_sum_last(x[i : i + 1])[0], full[i])
    assert torch.equal(full, x.sum(-1))  # small integers: every order is exact


def test_golden_gemm_is_batch_invariant_and_row_parallel_tree_matches_reference():
    gemm = GoldenGemm("triton_fp32")
    a = torch.randn(96, 512, device=DEVICE).to(torch.bfloat16)
    w = torch.randn(192, 512, device=DEVICE).to(torch.bfloat16)
    full = gemm(a, w, out_dtype=torch.float32)
    one = gemm(a[7:8], w, out_dtype=torch.float32)
    assert torch.equal(full[7:8], one)
    assert torch.equal(gemm.row_parallel(a, w)[3:4], gemm.row_parallel(a[3:4], w))
    ref = (a.float() @ w.float().t())
    assert torch.allclose(full, ref, atol=1e-2, rtol=1e-3)
    assert torch.allclose(gemm.row_parallel(a, w).float(), ref, atol=0.25, rtol=2e-2)


def test_attention_core_split_invariance_with_window_inside_blocks():
    torch.manual_seed(0)
    nq, hq, hkv, d, window = 230, 8, 2, 32, 100
    q = torch.randn(nq, hq, d, device=DEVICE).to(torch.bfloat16)
    k = torch.randn(nq, hkv, d, device=DEVICE).to(torch.bfloat16)
    v = torch.randn(nq, hkv, d, device=DEVICE).to(torch.bfloat16)
    pos = torch.arange(nq, device=DEVICE)
    full, lse_full = attention_core(q, k, v, pos, window=window)
    # chunked queries over the same keys (all keys exist) must match row for row
    pieces = []
    for s, e in ((0, 1), (1, 64), (64, 130), (130, 131), (131, 230)):
        o, _ = attention_core(q[s:e], k[:e], v[:e], pos[s:e], window=window, k_len=e)
        pieces.append(o)
    assert torch.equal(torch.cat(pieces), full)
    # a gathered block range (paged form) with k_pos0 > 0: position 200 first sees key 101,
    # i.e. block 1 (keys 64..127), so the gather starts at absolute key 64
    o_last, lse_last = attention_core(
        q[200:230], k[64:230], v[64:230], pos[200:230], window=window, k_pos0=64, k_len=230
    )
    assert torch.equal(o_last, full[200:230]) and torch.equal(lse_last, lse_full[200:230])
    # global (no window): decode row equals the one-shot row
    g_full, _ = attention_core(q, k, v, pos, window=None)
    g_one, _ = attention_core(q[229:230], k, v, pos[229:230], window=None)
    assert torch.equal(g_one, g_full[229:230])
    # tolerance against a plain fp32 reference
    kf = k.float().repeat_interleave(hq // hkv, dim=1)
    vf = v.float().repeat_interleave(hq // hkv, dim=1)
    s = torch.einsum("qhd,khd->hqk", q.float(), kf) * d**-0.5
    i, j = torch.meshgrid(pos, pos, indexing="ij")
    s = s.masked_fill(~((j <= i) & (j > i - window))[None], float("-inf"))
    ref = torch.einsum("hqk,khd->qhd", torch.softmax(s, -1), vf)
    assert torch.allclose(full.float(), ref, atol=2e-2, rtol=2e-2)


# -------------------------------------------------------------------------------------------
# the chain gate
# -------------------------------------------------------------------------------------------
@pytest.mark.parametrize("vocab", [1024, 1000])
@pytest.mark.parametrize("prompt_len,completion_len", [(150, 20), (3, 5)])
def test_train_prefill_chunked_decode_are_byte_equal(vocab, prompt_len, completion_len):
    cfg = StrictConfig.tiny(vocab_size=vocab)
    weights = MuseGlimmerWeights.random(cfg, device=DEVICE, seed=vocab, layout="kn")
    model = MuseGlimmerStrictModel(weights, "golden")
    ids = _tokens(prompt_len + completion_len, vocab, seed=7)
    specs = default_path_specs(prompt_len, completion_len, include_per_token=True,
                               window=cfg.sliding_window)
    specs.append(PathSpec("chunked:ones_then_rest", "chunked",
                          (1, 1, 1, prompt_len + completion_len - 4)))
    report = compare_paths(model, ids[:prompt_len], ids[prompt_len:], specs=specs)
    assert report.all_equal, report.markdown()
    assert all(torch.isfinite(r.logp).all() for r in report.results)
    assert {r.spec.mode for r in report.results} >= {"train", "prefill", "chunked", "decode",
                                                      "per_token"}


def test_window_eviction_happens_and_keeps_bits(tiny_model):
    cfg = tiny_model.cfg
    ids = _tokens(300, cfg.vocab_size, seed=3)
    # one-shot vs 64-chunks: with window 100 and 300 tokens, local layers evict pages
    one = tiny_model.logprobs(ids[:280], ids[280:], mode="prefill")
    session = tiny_model.new_session(300)
    scored, targets = ids[:299], ids[1:300]
    chunked = tiny_model.prefill(session, scored, targets=targets, chunk_sizes=[64] * 4 + [43])
    assert session.cache.evictions > 0
    assert session.cache.pages_in_use(3) > session.cache.pages_in_use(0)  # global keeps all
    assert torch.equal(chunked[279:], one)


def test_first_drift_localizer_reports_no_drift_on_equal_paths(tiny_model):
    ids = _tokens(70, tiny_model.cfg.vocab_size, seed=11)
    drift = localize_first_drift(tiny_model, ids[:60], ids[60:],
                                 PathSpec("decode", "decode"))
    assert drift.node is None and drift.rows_differing == 0
    assert drift.nodes_checked >= 4 * 12


def test_first_drift_localizer_catches_an_injected_difference(tiny_model):
    ids = _tokens(40, tiny_model.cfg.vocab_size, seed=12)
    ops = tiny_model.ops
    original = ops.attention_paged

    def perturbed(*args, **kwargs):
        y, o, lse = original(*args, **kwargs)
        return y, o + torch.ones_like(o) * 1e-2, lse  # only the ungated o is perturbed

    ops.attention_paged = perturbed
    try:
        drift = localize_first_drift(tiny_model, ids[:30], ids[30:],
                                     PathSpec("prefill", "prefill"))
    finally:
        ops.attention_paged = original
    assert drift.node == "layers.0.attn" and drift.kind == "attention_local"
    assert drift.first_row == 0


# -------------------------------------------------------------------------------------------
# backward (tiny only: a full-model backward does not fit one GPU)
# -------------------------------------------------------------------------------------------
def test_backward_is_deterministic_across_runs_and_finite():
    cfg = StrictConfig.tiny()
    ids = _tokens(96, cfg.vocab_size, seed=5).to(DEVICE)
    targets = torch.cat([ids[1:], torch.full((1,), -1, device=DEVICE)])

    def run(seed_weights: int = 2) -> tuple[float, dict[str, torch.Tensor]]:
        weights = MuseGlimmerWeights.random(cfg, device=DEVICE, seed=seed_weights, layout="nk",
                                            requires_grad=True)
        model = MuseGlimmerStrictModel(weights, "golden")
        out = model.forward_train(ids, targets)
        loss = nanmean_loss(out["logp"])
        loss.backward()
        grads = {k: v.grad.detach().clone() for k, v in weights.parameters().items()
                 if v.grad is not None}
        return float(loss.item()), grads

    loss_a, grads_a = run()
    loss_b, grads_b = run()
    assert loss_a == loss_b and torch.isfinite(torch.tensor(loss_a))
    assert set(grads_a) == set(grads_b)
    assert len(grads_a) == 3 + 4 * 12  # every text parameter received a gradient
    for name in grads_a:
        assert torch.isfinite(grads_a[name]).all(), name
        assert torch.equal(grads_a[name], grads_b[name]), f"nondeterministic grad: {name}"
    # repeated tokens exercise the sorted-segment embedding backward
    emb = grads_a["model.language_model.embed_tokens.weight"]
    assert emb.abs().sum() > 0


# -------------------------------------------------------------------------------------------
# HF-equivalent math at bf16 tolerance (NOT bitwise)
# -------------------------------------------------------------------------------------------
def test_matches_transformers_tiny_model_at_tolerance():
    pytest.importorskip("transformers")
    cfg = StrictConfig.tiny()
    weights = MuseGlimmerWeights.random(cfg, device=DEVICE, seed=9, layout="both")
    model = MuseGlimmerStrictModel(weights, "golden")
    ids = _tokens(230, cfg.vocab_size, seed=21)
    ref = strict_reference_outputs(model, ids)
    hf_model, lm_head = hf_text_model_from_weights(weights, DEVICE)
    hf_logits, hf_hidden = hf_text_logits(hf_model, lm_head, ids, cfg)
    report = compare_strict_to_hf_outputs(ref, hf_logits, hf_hidden, completion_start=200)
    # Tolerance anchor, NOT bitwise: HF rounds cos/sin, the qk scale, scores, the gate and silu
    # to bf16 where the contract keeps fp32, so the residual stream drifts by bf16 noise per
    # layer. Observed (seeds 9/10, 230 tokens): embedding ≤ 1 bf16 ulp, max|Δlogp| ≈ 0.025,
    # max|Δlogit| ≈ 0.055 on logits with std 0.064 (random weights give near-flat logits, so
    # argmax agreement (~0.84) is not informative here; the real-weights run is the anchor).
    assert report["per_layer"][0]["layer"] == "embedding"
    assert report["per_layer"][0]["max_abs"] <= 1e-2, report["per_layer"][0]
    assert report["logits_all_positions"]["max_abs"] <= 0.15, report["logits_all_positions"]
    assert report["logp_all"]["max_abs"] <= 0.1, report["logp_all"]
    assert report["logp_all"]["mean_abs"] <= 0.02, report["logp_all"]
    assert 0.0 <= report["argmax_agreement_all"] <= 1.0
    assert len(report["per_layer"]) == cfg.num_hidden_layers + 1  # embedding, L0..L2, final


def test_triton_backend_requires_sibling_ops_or_allow_partial():
    from rl_engine.models.muse_glimmer.reference.backend import SiblingOpsMissing

    try:
        ops = resolve_ops("triton")
    except SiblingOpsMissing:
        ops = resolve_ops("triton", allow_partial=True)
        assert ops.golden_nodes()  # substitutions are recorded, never silent
    assert ops.backend == "triton"


# -------------------------------------------------------------------------------------------
# batched decode (contract G2 at the model level), profiling regions, xrt scoring
# -------------------------------------------------------------------------------------------
def _fill_batch_cache(model, prompts, completions):
    from rl_engine.models.muse_glimmer.reference.strict_model import Session

    total = max(int(p.numel() + c.numel()) for p, c in zip(prompts, completions))
    cache = model.new_batch_cache(total + 2, len(prompts))
    first = []
    for b, (p, c) in enumerate(zip(prompts, completions)):
        tgt = torch.cat([p[1:], c[:1]])
        first.append(model.prefill(Session(cache=cache, seq=b), p, targets=tgt)[-1:].cpu())
    return cache, first


def test_decode_batch_matches_single_sequence_decode_bitwise(tiny_model):
    vocab = tiny_model.cfg.vocab_size
    lens = [30, 57, 130, 3]  # 130 > window 100: local layers evict pages
    prompts = [_tokens(n, vocab, seed=40 + i) for i, n in enumerate(lens)]
    completions = [_tokens(4, vocab, seed=60 + i) for i in range(len(lens))]
    refs = [tiny_model.logprobs(p, c, mode="decode").cpu() for p, c in zip(prompts, completions)]
    cache, first = _fill_batch_cache(tiny_model, prompts, completions)
    got = [[f] for f in first]
    seqs = list(range(len(lens)))
    for j in range(3):
        toks = [int(c[j]) for c in completions]
        tgts = [int(c[j + 1]) for c in completions]
        # replay the step at batch 1 and in a permuted sub-batch, then commit it at full batch
        single = [tiny_model.decode_batch(cache, [b], [toks[b]], [tgts[b]], commit=False).cpu()
                  for b in seqs]
        perm = [2, 0, 3]
        sub = tiny_model.decode_batch(cache, perm, [toks[b] for b in perm],
                                      [tgts[b] for b in perm], commit=False).cpu()
        full = tiny_model.decode_batch(cache, seqs, toks, tgts, commit=True).cpu()
        assert torch.equal(full, torch.cat(single))
        assert torch.equal(sub, full[perm])
        for b in seqs:
            got[b].append(full[b : b + 1])
    for b in seqs:
        assert torch.equal(torch.cat(got[b]), refs[b]), f"sequence {b}"
    assert [cache.seq_len(b) for b in seqs] == [n + 3 for n in lens]
    assert cache.evictions > 0


def test_region_timer_regions_nest_bound_overhead_and_keep_bits(tiny_model):
    from rl_engine.models.muse_glimmer.reference.profiling import (
        CLASSES,
        RegionTimer,
        combine_repeats,
        summarize_regions,
        timed_call,
    )

    ids = _tokens(150, tiny_model.cfg.vocab_size, seed=31).to(DEVICE)
    targets = torch.cat([ids[1:], torch.full((1,), -1, device=DEVICE)])

    def call():
        with torch.no_grad():
            return tiny_model.forward_train(ids, targets)["logp"]

    plain = call()
    clean_ms = min(timed_call(call)[1] for _ in range(3))
    timer = RegionTimer()
    tiny_model.set_timer(timer)
    try:
        with torch.no_grad(), timer.region("forward", "model"):
            timed = tiny_model.forward_train(ids, targets)["logp"]
    finally:
        tiny_model.set_timer(None)
    assert torch.equal(plain.view(torch.int32), timed.view(torch.int32))  # bitwise, NaN incl.
    nl = tiny_model.cfg.num_hidden_layers
    rows = timer.resolve()
    # regions nest on the GPU timeline: no region is shorter than its children together
    # (exclusive >= 0 up to the CUDA-event resolution summed over ~50 children)
    worst = min(rows, key=lambda r: r["exclusive_ms"])
    assert worst["exclusive_ms"] >= -0.05, worst
    summary = summarize_regions(rows, nl)
    # the instrumented call measures the same work: bounded overhead vs the clean call
    assert 0.5 * clean_ms <= summary["total_ms"] <= 5.0 * clean_ms + 5.0, (
        summary["total_ms"], clean_ms)
    # every layer's own regions are charged to that layer index
    for cls in ("qkvg_gemm", "qknorm_rope", "attention", "o_gemm", "ffn_gemm"):
        assert all(v > 0 for v in summary["per_layer_ms"][cls]), cls
    combined = combine_repeats([summary], nl)
    by_kind = summary["by_kind_ms"]
    for cls in ("qkvg_gemm", "qknorm_rope", "attention", "o_gemm", "norm", "ffn_gemm"):
        assert by_kind["local"][cls] > 0 and by_kind["global"][cls] > 0, cls
    assert by_kind["model"]["lm_head_gemm"] > 0 and by_kind["model"]["embedding"] > 0
    assert set(combined["by_class_ms"]) == set(CLASSES)
    assert all(len(v) == nl for v in summary["per_layer_ms"].values())


def test_kernel_attribution_charges_gpu_time_to_regions(tiny_model):
    from rl_engine.models.muse_glimmer.reference.profiling import (
        RegionTimer,
        attributed_kernel_profile,
    )

    ids = _tokens(150, tiny_model.cfg.vocab_size, seed=32).to(DEVICE)
    targets = torch.cat([ids[1:], torch.full((1,), -1, device=DEVICE)])

    def call():
        with torch.no_grad():
            return tiny_model.forward_train(ids, targets)["logp"]

    from rl_engine.models.muse_glimmer.reference.profiling import timed_call

    call()
    wall_ms = min(timed_call(call)[1] for _ in range(3))
    prof = attributed_kernel_profile(call, RegionTimer(), tiny_model.set_timer,
                                     num_layers=tiny_model.cfg.num_hidden_layers)
    assert tiny_model._timer is None
    assert prof["gpu_busy_ms"] > 0 and prof["kernel_launches"] > 0
    assert prof["launch_events_matched"] >= 0.9 * prof["kernel_launches"], prof
    # almost every kernel is charged to a region (the forward runs inside the root region)
    assert prof["unattributed_ms"] <= 0.1 * prof["gpu_busy_ms"], prof["unattributed_ms"]
    # one stream: kernel time cannot exceed the wall time of the same call (no double count)
    assert prof["gpu_busy_ms"] <= 1.25 * wall_ms + 0.5, (prof["gpu_busy_ms"], wall_ms)
    # attention kernels of every layer are charged to that layer index
    assert all(v > 0 for v in prof["per_layer_gpu_ms"]["attention"])
    gpu = prof["by_kind_class_gpu_ms"]
    assert gpu["local"]["attention"] > 0 and gpu["global"]["qkvg_gemm"] > 0
    assert gpu["model"]["lm_head_gemm"] > 0


def test_xrt_scoring_writes_hex_of_train_logp_and_rollout_forms_agree(tiny_model, tmp_path):
    import json

    from rl_engine.models.muse_glimmer.reference.xrt_scoring import (
        compare_xrt,
        hex_to_fp32,
        score_token_file,
    )

    vocab = tiny_model.cfg.vocab_size
    seqs = []
    for i, (n, cs) in enumerate([(40, 25), (140, 110)]):
        seqs.append({"id": f"s{i}", "token_ids": _tokens(n, vocab, seed=90 + i).tolist(),
                     "completion_start": cs})
    path = tmp_path / "tokens.json"
    path.write_text(json.dumps({"schema": "muse_glimmer_xrt_v1", "sequences": seqs}))
    modes = ("prefill", "decode", "per_token", "chunked:prompt_boundary", "chunked:20/19",
             "chunked:window_boundary_100")
    out = score_token_file(tiny_model, path, check_modes=modes)
    assert out["schema"] == "muse_glimmer_xrt_logprobs_v1" and out["tp"] == 1
    for s, rec in zip(seqs, out["sequences"]):
        ids = torch.tensor(s["token_ids"])
        cs = s["completion_start"]
        ref = tiny_model.logprobs(ids[:cs], ids[cs:], mode="train").float().cpu()
        assert len(rec["logprobs_hex"]) == len(s["token_ids"]) - cs
        assert all(len(h) == 8 for h in rec["logprobs_hex"])
        assert torch.equal(hex_to_fp32(rec["logprobs_hex"]), ref)
    first, second = (r["modes"] for r in out["checks"]["sequences"])
    assert "skipped" in first["chunked:window_boundary_100"]  # 39 scored tokens < 101
    assert "skipped" in second["chunked:20/19"]
    assert second["chunked:window_boundary_100"]["byte_equal"]
    assert out["checks"]["all_byte_equal_to_train"], out["checks"]
    assert out["checks"]["n_checks"] == 5 + 5 and out["checks"]["n_skipped"] == 2
    # decode times the prompt prefill and every decode step separately
    dec = second["decode"]
    assert dec["decode_steps"] == 30 - 1 == len(dec["decode_step_seconds"])
    assert dec["prompt_prefill_seconds"] > 0 and dec["decode_step_context"][0] == 111
    # compare_xrt counts bit patterns: one flipped mantissa bit -> exactly one mismatch
    import copy

    other = copy.deepcopy(out)
    h = other["sequences"][1]["logprobs_hex"][3]
    flipped = f"{int(h, 16) ^ 1:08x}"
    other["sequences"][1]["logprobs_hex"][3] = flipped
    diff = compare_xrt(out, other)
    assert diff["n_tokens"] == 15 + 30 and diff["equal_tokens"] == 15 + 30 - 1
    row = diff["sequences"][1]
    expect = abs(float(hex_to_fp32([h])[0]) - float(hex_to_fp32([flipped])[0]))
    assert row["equal_tokens"] == 29 and row["max_abs_diff"] == expect > 0
    assert diff["sequences"][0]["equal_tokens"] == 15


def test_xrt_scoring_compares_bit_patterns_and_rejects_vacuous_checks(tmp_path):
    """-0.0 vs +0.0 is a mismatch (bit patterns, not float ==); no check run is not a pass."""
    import json
    from types import SimpleNamespace

    from rl_engine.models.muse_glimmer.reference.xrt_scoring import score_token_file

    class _Stub:
        device = DEVICE
        ops = SimpleNamespace(backend="stub", provenance={})

        def logprobs(self, prompt, completion, *, mode, chunk_sizes=None, timings=None):
            out = torch.zeros(int(completion.numel()))  # +0.0
            if mode == "prefill":
                out[0] = -0.0  # same float value, different bits
            if timings is not None:
                timings.update(prompt_prefill_seconds=0.0, prompt_len=int(prompt.numel()),
                               decode_step_seconds=[0.0] * (int(completion.numel()) - 1),
                               decode_context_start=int(prompt.numel()))
            return out

    path = tmp_path / "tokens.json"
    path.write_text(json.dumps({"schema": "muse_glimmer_xrt_v1", "sequences": [
        {"id": "a", "token_ids": list(range(12)), "completion_start": 8}]}))
    out = score_token_file(_Stub(), path, check_modes=("prefill", "decode"))
    modes = out["checks"]["sequences"][0]["modes"]
    assert modes["prefill"]["byte_equal"] is False and modes["prefill"]["first_mismatch"] == 0
    assert modes["prefill"]["max_abs_diff"] == 0.0  # float-equal, bit-different
    assert modes["decode"]["byte_equal"] is True
    assert out["checks"]["all_byte_equal_to_train"] is False
    none = score_token_file(_Stub(), path, check_modes=())
    assert none["checks"]["n_checks"] == 0 and none["checks"]["all_byte_equal_to_train"] is False
    skipped = score_token_file(_Stub(), path, check_modes=("chunked:window_boundary_100",))
    assert skipped["checks"]["n_skipped"] == 1
    assert skipped["checks"]["all_byte_equal_to_train"] is False


def test_decode_batch_and_kv_plans_reject_bad_sequence_ids(tiny_model):
    """Duplicate ids used to give a silently wrong logp (two rows wrote the same cache slot);
    a negative id aliased the last row and then tripped a device-side assert. Fail closed
    BEFORE any K/V is written."""
    from rl_engine.models.muse_glimmer.reference.strict_model import KVPlan

    vocab = tiny_model.cfg.vocab_size
    prompts = [_tokens(20, vocab, seed=150), _tokens(9, vocab, seed=151)]
    completions = [_tokens(3, vocab, seed=152), _tokens(3, vocab, seed=153)]
    cache, _ = _fill_batch_cache(tiny_model, prompts, completions)
    before = [t.clone() for t in (cache.k, cache.v, cache.block_table, cache.seq_lens)]
    for seqs in ([1, 1], [0, 1, 0], [-1], [2], [0, -2]):
        with pytest.raises(ValueError, match="sequence ids"):
            tiny_model.decode_batch(cache, seqs, [5] * len(seqs), [6] * len(seqs), commit=True)
    after = (cache.k, cache.v, cache.block_table, cache.seq_lens)
    assert all(torch.equal(a, b) for a, b in zip(before, after))  # nothing was written
    for plan in (
        KVPlan(cache, [0, 0], [20, 20], [1, 1]),  # duplicate
        KVPlan(cache, [0], [3], [1]),  # start is not the committed length (20)
        KVPlan(cache, [0, 1], [20], [1, 1]),  # ragged lists
        KVPlan(cache, [0], [20], [0]),  # empty entry
        KVPlan(cache, [5], [0], [1]),  # out of range
    ):
        with pytest.raises(ValueError):
            plan.validate()
    # a valid step still works and matches the single-sequence decode path
    ref = tiny_model.logprobs(prompts[1], completions[1], mode="decode").cpu()
    step = tiny_model.decode_batch(cache, [1], [int(completions[1][0])],
                                   [int(completions[1][1])], commit=False).cpu()
    assert torch.equal(step.view(torch.int32), ref[1:2].view(torch.int32))
