import pathlib

p = pathlib.Path("tests/models/muse_glimmer/test_qknorm_rope.py")
s = p.read_text()


def rep(old, new):
    global s
    assert s.count(old) == 1, old[:80]
    s = s.replace(old, new)


rep('''from __future__ import annotations

import json

import pytest
import torch
''', '''from __future__ import annotations

import json
import re
import warnings

import pytest
import torch
''')

# canonical table vs the FORMULA (rope_table now returns canonical rows: comparing the two
# would be a tautology)
rep('''        c3, s3 = op.canonical_rope_table()  # default = current CUDA device
        assert c3 is c1 and s3 is s1
        full_c, full_s = rope_table(torch.arange(ROPE_POSITIONS, device=DEV), device=DEV)
        assert torch.equal(c1, full_c) and torch.equal(s1, full_s)''', '''        c3, s3 = op.canonical_rope_table()  # default = current CUDA device
        assert c3 is c1 and s3 is s1
        # the cached table IS the O4 formula evaluated directly on arange(131072)
        full_c, full_s = op._rope_table_in(
            torch.arange(ROPE_POSITIONS, device=DEV), theta=ROPE_THETA, dim=D, device=DEV,
            dtype=torch.float32,
        )
        assert torch.equal(c1, full_c) and torch.equal(s1, full_s)
        # rope_table returns COPIES of the canonical rows: writing to them is harmless
        rc, _ = rope_table(torch.arange(5, device=DEV), device=DEV)
        assert rc.data_ptr() != c1.data_ptr()
        rc.add_(1.0)
        assert op.canonical_rope_table(DEV)[0] is c1 and torch.equal(c1[:5], full_c[:5])
        # a direct formula evaluation at arbitrary positions equals the gathered rows
        pos = torch.tensor([3, 2047, 2048, 99999, ROPE_POSITIONS - 1], device=DEV)
        dc, ds = op._rope_table_in(pos, theta=ROPE_THETA, dim=D, device=DEV, dtype=torch.float32)
        gc_, gs = rope_table(pos, device=DEV)
        assert torch.equal(dc, gc_) and torch.equal(ds, gs)''')

rep('''        from rl_engine.models.muse_glimmer.reference import golden_ops
        from rl_engine.models.muse_glimmer.runtime import fallback_ops

        pos = torch.arange(ROPE_POSITIONS, device=DEV)
        cc, sc = op.canonical_rope_table(DEV)
        for fn in (golden_ops.rope_table, fallback_ops.rope_table):
            c, s = fn(pos, theta=ROPE_THETA, dim=D, device=DEV)
            assert torch.equal(c, cc) and torch.equal(s, sc), fn.__module__''', '''        from rl_engine.models.muse_glimmer.reference import golden_ops
        from rl_engine.models.muse_glimmer.runtime import fallback_ops, strict_kernels

        pos = torch.arange(ROPE_POSITIONS, device=DEV)
        cc, sc = op.canonical_rope_table(DEV)
        # golden_ops (reference), fallback_ops (vLLM binding fallback), strict_kernels (Megatron
        # binding fallback): all 4 builders of the repo
        for fn in (golden_ops.rope_table, fallback_ops.rope_table, strict_kernels._fb_rope_table):
            c, s = fn(pos, theta=ROPE_THETA, dim=D, device=DEV)
            assert torch.equal(c, cc) and torch.equal(s, sc), fn.__module__
        # the reference golden builder delegates to the canonical table and fails closed on CPU
        with pytest.raises(RuntimeError, match="torch-CUDA"):
            golden_ops.rope_table(torch.arange(4), theta=ROPE_THETA, dim=D, device="cpu")''')

rep('''        tab = trace()["rope_table"]
        assert tab["device_type"] == "cuda"
        assert tab["positions"] == ROPE_POSITIONS
        assert tab["sha256"] == CANONICAL_SHA256
        assert tab["builder"].endswith("qknorm_rope.rope_table")''', '''        tab = trace()["rope_table"]
        assert tab["device_type"] == "cuda"
        assert tab["positions"] == ROPE_POSITIONS
        assert tab["sha256_pinned"] == CANONICAL_SHA256
        assert "runtime_provenance" in tab["sha256_built"]
        assert tab["builder"].endswith("qknorm_rope.rope_table")''')

# R7: int64 offsets checked on the compiled kernel, not on a string
rep('''    def test_trace_records_int64_offsets(self):
        assert trace()["tile_config"]["offsets"] == "int64"''', '''    def test_trace_records_int64_offsets(self):
        assert trace()["tile_config"]["offsets"] == "int64"

    @pytest.mark.parametrize("bwd", [False, True])
    @pytest.mark.parametrize("hq,hk", [(32, 2), (4, 1)])
    def test_compiled_kernels_use_int64_row_offsets(self, bwd, hq, hk):
        """The behavioural guard without a 20 GiB allocation: in the kernels' TTIR the program
        id is sign-extended to i64 and the head-row pointer offsets of the Q and K tiles are
        i64 tensors (with int32 offsets, i.e. the old `t = tl.program_id(0)`, there is no
        extension and every offset is i32)."""
        from rl_engine.models.muse_glimmer.triton import qknorm_rope as tk

        q, k = _make_qk(2, seed=81, hq=hq, hk=hk)
        cos, sin = _tables(torch.arange(2, device=DEV))
        out_q, out_k = torch.empty_like(q), torch.empty_like(k)
        bq, bk = tk.triton.next_power_of_2(hq), tk.triton.next_power_of_2(hk)
        kw = dict(HQ=hq, HK=hk, BLOCK_HQ=bq, BLOCK_HK=bk, HALF=D // 2, HAS_ROPE=True,
                  num_warps=tk.NUM_WARPS, num_stages=tk.NUM_STAGES, enable_fp_fusion=False)
        if bwd:
            compiled = tk._qknorm_rope_bwd_kernel[(2,)](
                q, k, q, k, cos, sin, out_q, out_k, QK_SCALE, RMS_NORM_EPS, **kw)
        else:
            compiled = tk._qknorm_rope_fwd_kernel[(2,)](
                q, k, cos, sin, out_q, out_k, QK_SCALE, RMS_NORM_EPS, **kw)
        ttir = compiled.asm["ttir"]
        pids = re.findall(r"(%\\w+) = tt.get_program_id x : i32", ttir)
        assert pids, "no program id in the TTIR"
        for pid in pids:
            assert re.search(re.escape(pid) + r" : i32 to i64", ttir), f"{pid} not extended"
        adds = [ln for ln in ttir.splitlines() if "tt.addptr" in ln]
        for rows in (bq, bk):
            tile = f"tensor<{rows}x{D // 2}xi64>"
            assert any(tile in ln for ln in adds), f"no int64 row offsets for the {rows}-row tile"
        assert any("!tt.ptr<f32>, i64" in ln for ln in adds), "table row base is not int64"''')

s += '''

# --------------------------------------------------------------------------------------
# Wave B review follow-ups (kernels_rope:R1..R11, see REPORT_qknorm_rope.md)
# --------------------------------------------------------------------------------------


@pytest.fixture
def _layer_kind_state(monkeypatch):
    """Isolate the process-wide layer-kind switch, warning latch and counters."""
    monkeypatch.setattr(op, "_REQUIRE_LAYER_KIND", None)
    monkeypatch.setattr(op, "_WARNED_IMPLICIT", False)
    monkeypatch.setattr(op, "_LAYER_KIND_CALLS", {k: 0 for k in op._LAYER_KIND_CALLS})
    monkeypatch.delenv(op.LAYER_KIND_ENV, raising=False)
    yield


@pytest.fixture
def _private_tables(monkeypatch):
    """Tests that corrupt a canonical table work on copies of the module caches."""
    monkeypatch.setattr(op, "_CANONICAL_TABLES", dict(op._CANONICAL_TABLES))
    monkeypatch.setattr(op, "_CANONICAL_VERSIONS", dict(op._CANONICAL_VERSIONS))
    monkeypatch.setattr(op, "_CANONICAL_SHA", dict(op._CANONICAL_SHA))
    yield


class TestWaveBLayerKind:
    """R1: the legacy implicit form (local=None) is counted, warns once and can be made
    mandatory per process (env var for the E2E harness, or the switch)."""

    def test_implicit_form_is_counted_and_warned_once(self, _layer_kind_state):
        q, k = _make_qk(8, seed=200)
        cos, sin = _tables(torch.arange(8, device=DEV))
        with pytest.warns(RuntimeWarning, match="local=None"):
            a = qknorm_scaled_rope(q, k, cos, sin)
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # warned once per process
            qknorm_scaled_rope(q, k, None, None)
            b = qknorm_scaled_rope(q, k, cos, sin, local=True)
            qknorm_scaled_rope(q, k, None, None, local=False)
        assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
        calls = op.runtime_provenance(compute_digest=False)["layer_kind_calls"]
        assert calls == {"explicit_local": 1, "explicit_global": 1, "implicit_local": 1,
                         "implicit_global": 1}

    def test_env_var_makes_the_layer_kind_mandatory(self, _layer_kind_state, monkeypatch):
        q, k = _make_qk(8, seed=201)
        cos, sin = _tables(torch.arange(8, device=DEV))
        monkeypatch.setenv(op.LAYER_KIND_ENV, "1")
        assert op.layer_kind_required()
        for c, s_ in ((cos, sin), (None, None)):
            with pytest.raises(ValueError, match="explicit"):
                qknorm_scaled_rope(q, k, c, s_)
            with pytest.raises(ValueError, match="explicit"):
                QKNormScaledRopeOp()(q, k, c, s_)
        # explicit calls and the positions API are unaffected
        qknorm_scaled_rope(q, k, cos, sin, local=True)
        qknorm_scaled_rope(q, k, None, None, local=False)
        op.qknorm_scaled_rope_at(q, k, torch.arange(8, device=DEV), layer_idx=0)
        assert op.runtime_provenance(compute_digest=False)["layer_kind_required"] is True
        # the programmatic switch overrides the environment, None defers to it again
        op.require_explicit_layer_kind(False)
        qknorm_scaled_rope(q, k, None, None)
        op.require_explicit_layer_kind(None)
        with pytest.raises(ValueError):
            qknorm_scaled_rope(q, k, None, None)
        monkeypatch.setenv(op.LAYER_KIND_ENV, "0")
        assert not op.layer_kind_required()

    def test_switch_without_env(self, _layer_kind_state):
        q, k = _make_qk(4, seed=202)
        op.require_explicit_layer_kind(True)
        try:
            with pytest.raises(ValueError, match="explicit"):
                qknorm_scaled_rope(q, k, None, None)
        finally:
            op.require_explicit_layer_kind(None)


class TestWaveBTableIntegrity:
    """R3/R4/R6/R11: the shared canonical table is version-checked, verify=True re-hashes,
    the first build may not happen during CUDA-graph capture, and the runtime provenance
    reports the digest of the table this process actually built."""

    def test_verify_raises_on_a_digest_mismatch(self, monkeypatch):
        op.canonical_rope_table(DEV, verify=True)
        monkeypatch.setattr(op, "CANONICAL_ROPE_TABLE_SHA256", "0" * 64)
        with pytest.raises(RuntimeError, match="sha256"):
            op.canonical_rope_table(DEV, verify=True)

    def test_in_place_change_is_detected_without_a_sync(self, _private_tables):
        c, s_ = op.canonical_rope_table(DEV, dim=64)  # a private key: never the model's table
        op.rope_rows(torch.arange(4, device=DEV), dim=64)
        c.mul_(1 + 2**-20)  # bumps the version counter
        with pytest.raises(RuntimeError, match="modified in place"):
            op.canonical_rope_table(DEV, dim=64)
        with pytest.raises(RuntimeError, match="modified in place"):
            op.rope_rows(torch.arange(4, device=DEV), dim=64)
        with pytest.raises(RuntimeError, match="modified in place"):
            rope_table(torch.arange(4, device=DEV), dim=64, device=DEV)
        with pytest.raises(RuntimeError, match="modified in place"):
            op.rope_table_sha256(DEV, dim=64)

    def test_verify_rehashes_instead_of_reading_the_memo(self):
        """A write that bypasses the version counter (.data) is still caught by verify=True,
        and the logged digest follows the table's actual bits."""
        c, _ = op.canonical_rope_table(DEV, verify=True)
        assert op.rope_table_sha256(DEV) == CANONICAL_SHA256  # memoised
        old = c.data[7, 3].clone()
        try:
            c.data[7, 3] = old + 1.0  # no version bump: the cheap check cannot see it
            op.canonical_rope_table(DEV)  # passes (documented limit of the cheap check)
            with pytest.raises(RuntimeError, match="sha256"):
                op.canonical_rope_table(DEV, verify=True)
            assert op.rope_table_sha256(DEV) != CANONICAL_SHA256  # memo was refreshed
        finally:
            c.data[7, 3] = old
        op.canonical_rope_table(DEV, verify=True)
        assert op.rope_table_sha256(DEV) == CANONICAL_SHA256

    def test_first_build_inside_cuda_graph_capture_raises(self, _private_tables):
        key_dim = 32  # not built yet in this test's private cache view
        key = (torch.cuda.current_device(), ROPE_THETA, key_dim)
        op._CANONICAL_TABLES.pop(key, None)
        op._CANONICAL_VERSIONS.pop(key, None)
        pos = torch.arange(8, device=DEV)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            with pytest.raises(RuntimeError, match="before CUDA-graph capture"):
                with torch.cuda.graph(g):
                    op.rope_rows(pos, dim=key_dim)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        assert key not in op._CANONICAL_TABLES  # no half-built table was cached
        # built eagerly first, the same call can be captured and replayed
        c_ref, s_ref = op.rope_rows(pos, dim=key_dim)
        g2 = torch.cuda.CUDAGraph()
        with torch.cuda.stream(stream):
            with torch.cuda.graph(g2):
                c_g, s_g = op.rope_rows(pos, dim=key_dim)
        torch.cuda.current_stream().wait_stream(stream)
        g2.replay()
        torch.cuda.synchronize()
        assert torch.equal(c_g, c_ref) and torch.equal(s_g, s_ref)

    def test_runtime_provenance_reports_the_built_digest(self):
        op.canonical_rope_table(DEV)
        prov = op.runtime_provenance()
        dev = f"cuda:{torch.cuda.current_device()}"
        assert prov["rope_table_sha256_built"][dev] == CANONICAL_SHA256
        assert prov["rope_table_equals_pinned"][dev] is True
        assert prov["rope_table_sha256_pinned"] == CANONICAL_SHA256
        assert set(prov["layer_kind_calls"]) == {
            "explicit_local", "explicit_global", "implicit_local", "implicit_global"}
        json.dumps(prov)
        assert op.runtime_provenance(compute_digest=False)["rope_table_sha256_built"][dev] is None
        # trace() stays static (G5) and points at runtime_provenance for measured fields
        assert trace() == trace() and trace()["scope"].startswith("describes the op")

    def test_canonical_table_built_under_inference_mode_is_a_normal_tensor(
        self, _private_tables
    ):
        key = (torch.cuda.current_device(), 10000.0, D)
        op._CANONICAL_TABLES.pop(key, None)
        op._CANONICAL_VERSIONS.pop(key, None)
        with torch.inference_mode():
            c, _ = op.canonical_rope_table(DEV, theta=10000.0)
        assert not c.is_inference()
        op.canonical_rope_table(DEV, theta=10000.0)  # version counters are readable
'''
p.write_text(s)
print("ok")
