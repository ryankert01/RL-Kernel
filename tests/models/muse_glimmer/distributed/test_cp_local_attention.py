# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""WS2 ``cp_local_attention``: sliding-window layers (W=2048 on ABSOLUTE positions) under CP.

At every tested length the window crosses a chunk boundary for CP2 and CP4 (chunk <= W or the
chunk start is inside another rank's keys), so each rank reads keys owned by its neighbours
through the all-gathered K/V. CP2/CP4 must reproduce the CP1 bits.
"""

from __future__ import annotations

import pytest
import torch

from rl_engine.models.muse_glimmer.runtime import cp_attention as cpa
from tests.models.muse_glimmer.distributed import _cp_harness as harness

pytestmark = [pytest.mark.ws2]

_BF16_ATOL = 5.0e-2
_BF16_RTOL = 2.0e-2


@pytest.mark.skipif(harness.visible_gpus() < 1, reason="requires one CUDA GPU")
def test_window_is_evaluated_on_absolute_positions():
    """Rank 1's rows alone, with the full K/V: absolute positions match the reference, a
    CP-oblivious relative numbering (pos - chunk_start) does not."""
    harness.disable_tf32()
    device = torch.device("cuda", 0)
    total = 4096
    inp = harness.make_global_inputs(total, seed=5, device=device)
    rows = slice(total // 2, total)
    y, _, _ = cpa.gated_attention_rows(
        inp["q"][rows], inp["k"], inp["v"], inp["gate"][rows], inp["positions"][rows],
        window=harness.WINDOW,
    )
    want_y, _, _ = harness.naive_reference(
        inp["q"], inp["k"], inp["v"], inp["gate"], inp["positions"], window=harness.WINDOW
    )
    torch.testing.assert_close(y.float(), want_y[rows], atol=_BF16_ATOL, rtol=_BF16_RTOL)
    relative = inp["positions"][rows] - total // 2
    y_rel, _, _ = cpa.gated_attention_rows(
        inp["q"][rows], inp["k"], inp["v"], inp["gate"][rows], relative, window=harness.WINDOW
    )
    assert (y_rel.float() - want_y[rows]).abs().max().item() > 0.1


@pytest.mark.skipif(harness.visible_gpus() < 1, reason="requires one CUDA GPU")
def test_rows_alone_reproduce_full_run_bits_with_window():
    harness.disable_tf32()
    device = torch.device("cuda", 0)
    total = 4096
    inp = harness.make_global_inputs(total, seed=6, device=device)
    y, o, lse = cpa.gated_attention_rows(
        inp["q"], inp["k"], inp["v"], inp["gate"], inp["positions"], window=harness.WINDOW
    )
    for start in (1024, 2048, 2048 + 192, 3072):
        rows = slice(start, total)
        y_r, o_r, lse_r = cpa.gated_attention_rows(
            inp["q"][rows], inp["k"], inp["v"], inp["gate"][rows], inp["positions"][rows],
            window=harness.WINDOW,
        )
        assert torch.equal(y_r, y[rows]), start
        assert torch.equal(o_r, o[rows]), start
        assert torch.equal(lse_r, lse[rows]), start


_needs_own_processes = pytest.mark.skipif(
    harness.EXTERNAL_WORLD_SIZE != 1,
    reason="this test owns its worker processes; run pytest directly, not under torchrun",
)


@_needs_own_processes
@pytest.mark.parametrize("cp", [2, 4])
def test_cp_local_forward_is_bitwise_against_cp1(cp, tmp_path):
    if harness.visible_gpus() < cp:
        pytest.skip(f"requires {cp} CUDA GPUs")
    records = harness.spawn_and_collect(
        cp, harness.forward_parity_worker, tmp_path, harness.LENGTHS, harness.WINDOW
    )
    harness.assert_all_equal(records)
    for record in records:
        if record["rank"] > 0:
            assert all(record["teeth_relative_positions_differ"].values()), record["rank"]


@_needs_own_processes
@pytest.mark.parametrize("cp", [2, 4])
def test_cp_local_backward_is_bitwise_against_cp1(cp, tmp_path):
    if harness.visible_gpus() < cp:
        pytest.skip(f"requires {cp} CUDA GPUs")
    records = harness.spawn_and_collect(
        cp, harness.backward_parity_worker, tmp_path, harness.LENGTHS, harness.WINDOW
    )
    harness.assert_all_equal(records)
