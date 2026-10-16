from pathlib import Path

p = Path("/home/ubuntu/RL-Kernel/.claude/worktrees/wf_65f5032c-579-2/tests/models/muse_glimmer/test_vllm_strict_binding.py")
s = p.read_text()

# R9: restore the O7 sign invariant (c88607f launches the softcap kernels without FP fusion).
old = '''                for v in sample.logprobs:
                    # <= a few ulp above 0: the strict O7 tile kernel can return +1..2 ulp for a
                    # near-certain argmax (its pass-2 exp(zhat - m) of the max element is not
                    # exactly 1; reported to the lm_head row).
                    assert v <= 1e-5 and float(torch.tensor(v, dtype=torch.float32)) == v'''
new = '''                for v in sample.logprobs:
                    # O7 sign invariant (restored after c88607f: the softcap kernels launch with
                    # enable_fp_fusion=False, so a near-certain token's logp can no longer be
                    # +1..2 ulp above 0; issue vllm_runtime:R9).
                    assert v <= 0.0 and float(torch.tensor(v, dtype=torch.float32)) == v'''
assert old in s
s = s.replace(old, new, 1)

# provenance now reports the TP workers' recomputed hashes
old = '''        info = transfer_lora(package, engine, via_file=tmp_path / "lora.pt")
        assert info["identity"] and engine.provenance()["lora_sha256"] == package.sha256'''
new = '''        info = transfer_lora(package, engine, via_file=tmp_path / "lora.pt")
        prov = engine.provenance()
        assert info["identity"] and prov["lora_sha256"] == package.sha256
        assert prov["lora_sha256_per_worker"] == [package.sha256]
        assert prov["workers"][0]["lora_sha256_declared"] == package.sha256
        assert "recomputed" in prov["lora_sha256_source"]'''
assert old in s
s = s.replace(old, new, 1)

old = '''# ------------------------------------------------------------------------------------------
# E2E (opt-in)
# ------------------------------------------------------------------------------------------
'''
new = '''# ------------------------------------------------------------------------------------------
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
'''
assert old in s
s = s.replace(old, new, 1)
p.write_text(s)
print("ok")
