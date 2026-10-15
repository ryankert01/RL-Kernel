# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""WS2 ``cp_global_attention``: full-causal NoPE layers on the allgather CP layout (RFC #501).

CP2 and CP4 must reproduce the CP1 bits of the same strict core (``torch.equal`` on y, o and
lse, and on dq/dk/dv/dgate), at lengths 2048/4096/8192; the zigzag/striped layouts fail closed.
"""

from __future__ import annotations

import pytest
import torch

from rl_engine.models.muse_glimmer.runtime import cp_attention as cpa
from tests.models.muse_glimmer.distributed import _cp_harness as harness

pytestmark = [pytest.mark.ws2]

# ws1.json "attention" bf16 tolerance (correctness oracle only; parity gates are bitwise).
_BF16_ATOL = 5.0e-2
_BF16_RTOL = 2.0e-2


# ------------------------------------------------------------------ CPU-safe contract checks
@pytest.mark.unit
@pytest.mark.parametrize("layout", ["zigzag", "striped", "ring", "interleaved", "ALLGATHER", ""])
def test_non_allgather_layout_fails_closed(layout):
    with pytest.raises(cpa.CPLayoutError, match="not supported"):
        cpa.CPAttentionPlan(2, 0, layout=layout)


@pytest.mark.unit
def test_plan_chunks_are_contiguous_and_canonical():
    assert cpa.CPAttentionPlan(4, 1).local_token_range(8192) == (2048, 4096)
    assert cpa.CPAttentionPlan(2, 1).local_token_range(2048) == (1024, 2048)
    assert cpa.CPAttentionPlan(1, 0).canonical_chunk_ids() == tuple(range(8))
    assert cpa.CPAttentionPlan(2, 1).canonical_chunk_ids() == (4, 5, 6, 7)
    assert cpa.CPAttentionPlan(4, 2).canonical_chunk_ids() == (4, 5)
    assert cpa.CPAttentionPlan(8, 7).canonical_chunk_ids() == (7,)
    with pytest.raises(ValueError):
        cpa.CPAttentionPlan(3, 0)
    with pytest.raises(ValueError):
        cpa.CPAttentionPlan(2, 2)
    with pytest.raises(ValueError):
        cpa.CPAttentionPlan(4, 0).local_token_range(1002)


@pytest.mark.unit
def test_permuted_positions_fail_closed_before_any_kernel():
    total = 1024
    q = torch.zeros(total, 32, 128, dtype=torch.bfloat16)
    k = torch.zeros(total, 2, 128, dtype=torch.bfloat16)
    gate = torch.zeros(total, 4096, dtype=torch.bfloat16)
    blocks = [(0, 256), (768, 1024), (256, 512), (512, 768)]
    zigzag = torch.cat([torch.arange(lo, hi) for lo, hi in blocks]).to(torch.int32)
    with pytest.raises(cpa.CPLayoutError, match="contiguous allgather chunk"):
        cpa.cp_gated_attention_forward(
            q, k, k, gate, zigzag, cpa.CPAttentionPlan(1, 0), None, window=None
        )
    # Rank 1 of CP2 handed rank 0's chunk: rejected before the collective is touched.
    half = total // 2
    with pytest.raises(cpa.CPLayoutError, match="contiguous allgather chunk"):
        cpa.cp_gated_attention_forward(
            q[:half], k[:half], k[:half], gate[:half],
            torch.arange(0, half, dtype=torch.int32),
            cpa.CPAttentionPlan(2, 1), None, window=None,
        )


@pytest.mark.unit
def test_trace_reports_layout_and_kernel_provenance():
    record = cpa.trace(cpa.CPAttentionPlan(2, 1))
    assert record["cp_layout"] == "allgather"
    assert "zigzag" in record["cp_rejected_layouts"]
    assert record["cp_local_canonical_chunks"] == [4, 5, 6, 7]
    assert "no (out,lse) merge" in record["cp_forward"]
    assert isinstance(record["forward_is_fallback"], bool)
    for key in ("dtype_in", "dtype_acc", "dtype_out", "reduction_order", "tile_config"):
        assert key in record


# ------------------------------------------------------------------ single GPU
@pytest.mark.skipif(harness.visible_gpus() < 1, reason="requires one CUDA GPU")
def test_rows_alone_reproduce_full_run_bits():
    """The O5 property that makes CP trivial: a row's bits do not depend on its launch-mates."""
    harness.disable_tf32()
    device = torch.device("cuda", 0)
    total = 2048
    inp = harness.make_global_inputs(total, seed=1, device=device)
    y, o, lse = cpa.gated_attention_rows(
        inp["q"], inp["k"], inp["v"], inp["gate"], inp["positions"], window=None
    )
    for start in (total // 2, total // 2 + 192, 64 * 7 + 5):
        rows = slice(start, total)
        y_r, o_r, lse_r = cpa.gated_attention_rows(
            inp["q"][rows], inp["k"], inp["v"], inp["gate"][rows], inp["positions"][rows],
            window=None,
        )
        assert torch.equal(y_r, y[rows]), start
        assert torch.equal(o_r, o[rows]), start
        assert torch.equal(lse_r, lse[rows]), start


@pytest.mark.skipif(harness.visible_gpus() < 1, reason="requires one CUDA GPU")
def test_cp1_forward_matches_naive_fp32_within_bf16_tolerance():
    harness.disable_tf32()
    device = torch.device("cuda", 0)
    inp = harness.make_global_inputs(2048, seed=2, device=device)
    y, o, lse = cpa.cp_gated_attention_forward(
        inp["q"], inp["k"], inp["v"], inp["gate"], inp["positions"],
        cpa.CPAttentionPlan(1, 0), None, window=None, return_aux=True,
    )
    want_y, want_o, want_lse = harness.naive_reference(
        inp["q"], inp["k"], inp["v"], inp["gate"], inp["positions"], window=None
    )
    torch.testing.assert_close(y.float(), want_y, atol=_BF16_ATOL, rtol=_BF16_RTOL)
    torch.testing.assert_close(o.float(), want_o, atol=_BF16_ATOL, rtol=_BF16_RTOL)
    torch.testing.assert_close(lse, want_lse, atol=1.0e-4, rtol=1.0e-4)


# ------------------------------------------------------------------ multi GPU
_needs_own_processes = pytest.mark.skipif(
    harness.EXTERNAL_WORLD_SIZE != 1,
    reason="this test owns its worker processes; run pytest directly, not under torchrun",
)


@_needs_own_processes
@pytest.mark.parametrize("cp", [2, 4])
def test_cp_global_forward_is_bitwise_against_cp1(cp, tmp_path):
    if harness.visible_gpus() < cp:
        pytest.skip(f"requires {cp} CUDA GPUs")
    records = harness.spawn_and_collect(
        cp, harness.forward_parity_worker, tmp_path, harness.LENGTHS, None
    )
    harness.assert_all_equal(records)
    for record in records:
        if record["rank"] > 0:
            assert all(record["teeth_relative_positions_differ"].values()), record["rank"]


@_needs_own_processes
@pytest.mark.parametrize("cp", [2, 4])
def test_cp_global_backward_is_bitwise_against_cp1(cp, tmp_path):
    if harness.visible_gpus() < cp:
        pytest.skip(f"requires {cp} CUDA GPUs")
    records = harness.spawn_and_collect(
        cp, harness.backward_parity_worker, tmp_path, harness.LENGTHS, None
    )
    harness.assert_all_equal(records)
