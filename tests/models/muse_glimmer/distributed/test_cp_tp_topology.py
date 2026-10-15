# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""ONE TP x CP topology: TP2 (16 Q heads + its KV head per rank) x CP2 on 4 GPUs.

Attention itself needs no TP collective (each TP rank owns the KV head its heads read), so the
only communication is the CP all-gather of K/V inside each TP shard's CP group. The result must
be bitwise the single-GPU (TP1 x CP1) result restricted to the rank's heads, channels and rows.
"""

from __future__ import annotations

import pytest

from rl_engine.models.muse_glimmer.runtime import cp_attention as cpa
from tests.models.muse_glimmer.distributed import _cp_harness as harness

pytestmark = [pytest.mark.ws2]

_TP = 2
_CP = 2


@pytest.mark.unit
def test_topology_mapping_is_tp_fast_cp_slow():
    ranks = [cpa.TPCPTopology.from_global_rank(r, tp_world=_TP, cp_world=_CP) for r in range(4)]
    assert [(t.tp_rank, t.cp_rank) for t in ranks] == [(0, 0), (1, 0), (0, 1), (1, 1)]
    assert ranks[1].cp_group_ranks() == (1, 3)
    assert ranks[2].cp_group_ranks() == (0, 2)
    assert ranks[1].q_head_range() == (16, 32)
    assert ranks[1].kv_head_range() == (1, 2)
    assert ranks[1].gate_channel_range() == (2048, 4096)
    assert ranks[3].plan == cpa.CPAttentionPlan(2, 1)
    with pytest.raises(ValueError, match="tp_world"):
        cpa.TPCPTopology(tp_world=4, tp_rank=0, cp_world=2, cp_rank=0)


@pytest.mark.skipif(
    harness.EXTERNAL_WORLD_SIZE != 1,
    reason="this test owns its worker processes; run pytest directly, not under torchrun",
)
def test_tp2_cp2_forward_is_bitwise_against_single_gpu(tmp_path):
    world = _TP * _CP
    if harness.visible_gpus() < world:
        pytest.skip(f"requires {world} CUDA GPUs")
    records = harness.spawn_and_collect(
        world,
        harness.tpcp_parity_worker,
        tmp_path,
        (4096, 8192),
        (None, harness.WINDOW),
        _TP,
        _CP,
    )
    harness.assert_all_equal(records)
    assert {tuple(r["topology"]["q_head_range"]) for r in records} == {(0, 16), (16, 32)}
