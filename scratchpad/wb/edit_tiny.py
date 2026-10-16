import pathlib

p = pathlib.Path("tests/models/muse_glimmer/test_strict_model_tiny.py")
s = p.read_text()


def rep(old, new):
    global s
    assert s.count(old) == 1, old[:80]
    s = s.replace(old, new)


# ---------------------------------------------------------------- R8: region timer identities
rep('''def test_region_timer_classes_sum_to_root_and_keep_bits(tiny_model):
    from rl_engine.models.muse_glimmer.reference.profiling import (
        CLASSES,
        RegionTimer,
        combine_repeats,
        summarize_regions,
    )

    ids = _tokens(150, tiny_model.cfg.vocab_size, seed=31).to(DEVICE)
    targets = torch.cat([ids[1:], torch.full((1,), -1, device=DEVICE)])
    with torch.no_grad():
        plain = tiny_model.forward_train(ids, targets)["logp"]
    timer = RegionTimer()
    tiny_model.set_timer(timer)
    try:
        with torch.no_grad(), timer.region("forward", "model"):
            timed = tiny_model.forward_train(ids, targets)["logp"]
    finally:
        tiny_model.set_timer(None)
    assert torch.equal(plain.view(torch.int32), timed.view(torch.int32))  # bitwise, NaN incl.
    nl = tiny_model.cfg.num_hidden_layers
    summary = summarize_regions(timer.resolve(), nl)
    combined = combine_repeats([summary], nl)
    assert abs(combined["sum_of_classes_ms"] - summary["total_ms"]) <= 1e-3 * summary["total_ms"]
    by_kind = summary["by_kind_ms"]''', '''def test_region_timer_regions_nest_bound_overhead_and_keep_bits(tiny_model):
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
    by_kind = summary["by_kind_ms"]''')

# ---------------------------------------------------------------- R8: kernel attribution identity
rep('''    call()
    prof = attributed_kernel_profile(call, RegionTimer(), tiny_model.set_timer,
                                     num_layers=tiny_model.cfg.num_hidden_layers)
    assert tiny_model._timer is None
    assert prof["gpu_busy_ms"] > 0 and prof["kernel_launches"] > 0
    assert prof["launch_events_matched"] >= 0.9 * prof["kernel_launches"], prof
    attributed = sum(prof["by_class_gpu_ms"].values())
    assert abs(attributed + prof["unattributed_ms"] - prof["gpu_busy_ms"]) < 1e-3 + 1e-6 * 1e3
    gpu = prof["by_kind_class_gpu_ms"]''', '''    from rl_engine.models.muse_glimmer.reference.profiling import timed_call

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
    gpu = prof["by_kind_class_gpu_ms"]''')

# ---------------------------------------------------------------- R8: compare_xrt self identity
rep('''    assert second["chunked:window_boundary_100"]["byte_equal"]
    assert out["checks"]["all_byte_equal_to_train"], out["checks"]
    diff = compare_xrt(out, out)
    assert diff["equal_tokens"] == diff["n_tokens"] == 15 + 30''', '''    assert second["chunked:window_boundary_100"]["byte_equal"]
    assert out["checks"]["all_byte_equal_to_train"], out["checks"]
    assert out["checks"]["n_checks"] == 6 + 5 and out["checks"]["n_skipped"] == 1
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
    assert torch.equal(step.view(torch.int32), ref[1:2].view(torch.int32))''')
p.write_text(s)
print("ok")
