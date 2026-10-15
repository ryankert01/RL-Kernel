# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""``kv_head_replication`` forward half (RFC #501 WS2): replicated-KV attention == TP1.

Real head geometry (32 Q heads, 2 KV heads, head_dim 128, gate width 4096). The single-device
proof runs every TP in {1, 2, 4, 8} by slicing (the forward needs no communication); the NCCL
tests additionally place one rank per GPU and gather the rank outputs through the deterministic
collective to rebuild the TP1 tensor.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from rl_engine.models.muse_glimmer.runtime import kv_replication as kvr
from rl_engine.models.muse_glimmer.runtime.tp_layout import official_layout
from tests.models.muse_glimmer.distributed.spawn_utils import (
    disable_tf32,
    requires_gpus,
    run_spawned,
    sha256_of,
)

pytestmark = [
    pytest.mark.ws2,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU"),
]

HEADS, KV_HEADS, HEAD_DIM = 32, 2, 128
SEED = 20261015
CASES = {
    # name: (sequence lengths, window)
    "global_two_sequences": ((160, 352), None),
    "local_small_window": ((640,), 256),
    "local_real_window": ((2176,), kvr.SLIDING_WINDOW),
}


def _inputs(case: str, device: torch.device, seed: int = SEED):
    lengths, window = CASES[case]
    total = sum(lengths)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.randn(total, HEADS, HEAD_DIM, generator=gen).to(device, torch.bfloat16)
    k = torch.randn(total, KV_HEADS, HEAD_DIM, generator=gen).to(device, torch.bfloat16)
    v = torch.randn(total, KV_HEADS, HEAD_DIM, generator=gen).to(device, torch.bfloat16)
    gate = torch.randn(total, HEADS * HEAD_DIM, generator=gen).to(device, torch.bfloat16)
    cu_seqlens = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], device=device)
    positions = torch.cat([torch.arange(n, device=device) for n in lengths])
    return q, k, v, gate, cu_seqlens, positions, window


def _sdpa_reference(q, k, v, gate, cu_seqlens, positions, window):
    outs = []
    scale = HEAD_DIM**-0.5
    for seq in range(cu_seqlens.numel() - 1):
        a0, a1 = int(cu_seqlens[seq]), int(cu_seqlens[seq + 1])
        qq = q[a0:a1].float().permute(1, 0, 2)
        kk = k[a0:a1].float().repeat_interleave(HEADS // KV_HEADS, dim=1).permute(1, 0, 2)
        vv = v[a0:a1].float().repeat_interleave(HEADS // KV_HEADS, dim=1).permute(1, 0, 2)
        scores = (qq @ kk.transpose(1, 2)) * scale
        pos = positions[a0:a1]
        visible = pos[None, :, None] >= pos[None, None, :]
        if window is not None:
            visible = visible & (pos[None, :, None] - pos[None, None, :] < window)
        scores = scores.masked_fill(~visible, float("-inf"))
        out = torch.softmax(scores, dim=-1) @ vv
        outs.append(out.permute(1, 0, 2).reshape(a1 - a0, HEADS * HEAD_DIM))
    return torch.cat(outs) * torch.sigmoid(gate.float())


class TestOwnershipMap:
    def test_declared_map_matches_rfc_rule(self):
        # RFC #501 (audited): rank r -> KV head floor(2r/TP); identity at TP2; 2x / 4x replication.
        assert official_layout(1).kv_heads(0) == (0, 1)
        for tp in (2, 4, 8):
            layout = official_layout(tp)
            assert layout.kv_replication == max(1, tp // 2)
            for rank in range(tp):
                assert layout.kv_head(rank) == (2 * rank) // tp
                assert layout.kv_heads(rank) == (layout.q_heads(rank).start // 16,)
        with pytest.raises(Exception):
            official_layout(1).kv_head(0)

    def test_ownership_map_payload(self):
        payload = official_layout(8).kv_ownership_map()
        assert payload["kv_replication"] == 4 and payload["gqa_group"] == 16
        assert [entry["kv_heads"] for entry in payload["ranks"]] == [[0]] * 4 + [[1]] * 4
        assert [entry["q_heads"] for entry in payload["ranks"]][:2] == [[0, 4], [4, 8]]

    def test_trace_is_stable(self):
        layout = official_layout(4)
        assert kvr.trace(layout) == kvr.trace(layout)
        assert "4-leaf sub-tree" in kvr.trace(layout)["backward_rule"]


@requires_gpus(1)
class TestSingleDeviceProof:
    @pytest.fixture(autouse=True)
    def _tf32_off(self):
        disable_tf32()

    @pytest.mark.parametrize("case", list(CASES))
    def test_every_tp_equals_tp1_bitwise(self, case):
        q, k, v, gate, cu, pos, window = _inputs(case, torch.device("cuda"))
        report = kvr.prove_kv_replication(q, k, v, gate, cu, pos, window=window)
        for tp, entry in report["per_tp"].items():
            assert entry["equal"], f"{case}: TP{tp} rank outputs differ from TP1: {entry}"
            assert entry["concat_equal"], f"{case}: TP{tp} concat differs from TP1"
        assert set(report["per_tp"]) == {1, 2, 4, 8}

    def test_repeat_is_bitwise(self):
        q, k, v, gate, cu, pos, window = _inputs("global_two_sequences", torch.device("cuda"))
        first = kvr.gated_attention_fwd(q, k, v, gate, cu, pos, window=window)
        second = kvr.gated_attention_fwd(q, k, v, gate, cu, pos, window=window)
        assert first.dtype == torch.bfloat16 and torch.equal(first, second)

    @pytest.mark.parametrize("case", ["global_two_sequences", "local_small_window"])
    def test_matches_sdpa_reference_within_tolerance(self, case):
        q, k, v, gate, cu, pos, window = _inputs(case, torch.device("cuda"))
        strict = kvr.gated_attention_fwd(q, k, v, gate, cu, pos, window=window).float()
        reference = _sdpa_reference(q, k, v, gate, cu, pos, window)
        # ws1.json attention/bfloat16: atol 0.05, rtol 0.02.
        assert torch.allclose(strict, reference, atol=0.05, rtol=0.02)

    def test_replicas_hold_identical_bits(self):
        q, k, v, gate, cu, pos, _ = _inputs("global_two_sequences", torch.device("cuda"))
        layout = official_layout(8)
        for kv_head, ranks in ((0, range(0, 4)), (1, range(4, 8))):
            copies = [kvr.rank_inputs(q, k, v, gate, layout, r)[1] for r in ranks]
            assert all(torch.equal(copies[0], c) for c in copies[1:])
            assert torch.equal(copies[0][:, 0], k[:, kv_head])

    def test_shape_and_dtype_guards(self):
        q, k, v, gate, cu, pos, _ = _inputs("global_two_sequences", torch.device("cuda"))
        with pytest.raises(TypeError):
            kvr.gated_attention_fwd(q.float(), k, v, gate, cu, pos, window=None)
        with pytest.raises(ValueError):
            kvr.gated_attention_fwd(q, k, v, gate[:, :128], cu, pos, window=None)


# --------------------------------------------------------------------------- #
# NCCL ranks
# --------------------------------------------------------------------------- #


def _kv_worker(rank: int, world_size: int, case: str) -> dict[str, Any]:
    import torch.distributed as dist

    from rl_engine.models.muse_glimmer.runtime.tp_canonical_tree import make_collective

    device = torch.device("cuda", rank)
    layout = official_layout(world_size)
    layout.fail_closed_if_undeclared(world_size=dist.get_world_size())
    collective = make_collective(dist.group.WORLD, device)
    q, k, v, gate, cu, pos, window = _inputs(case, device)
    y_rank = kvr.rank_kv_replicated_attention(q, k, v, gate, cu, pos, layout, rank, window=window)
    y_rank_again = kvr.rank_kv_replicated_attention(
        q, k, v, gate, cu, pos, layout, rank, window=window
    )
    reference = kvr.gated_attention_fwd(q, k, v, gate, cu, pos, window=window)
    c0, c1 = layout.q_columns(rank)
    # all_gather concatenates along dim 0: gather the transposed shard to rebuild (T, 4096).
    gathered = collective.all_gather(y_rank.t().contiguous()).t()
    result = {
        "rank_equals_tp1_slice": bool(torch.equal(y_rank, reference[:, c0:c1])),
        "rerun_equal": bool(torch.equal(y_rank, y_rank_again)),
        "gathered_equals_tp1": bool(torch.equal(gathered, reference)),
        "sha": sha256_of(gathered),
        "tp1_sha": sha256_of(reference),
        "kv_head": layout.kv_heads(rank),
        "trace": kvr.trace(layout),
    }
    if hasattr(collective, "close"):
        collective.close()
    return result


def _assert_kv_results(results: list[dict[str, Any]]) -> None:
    for result in results:
        rank = result["rank"]
        assert result["rank_equals_tp1_slice"], f"rank {rank}: output differs from TP1 slice"
        assert result["rerun_equal"], f"rank {rank}: bits changed between identical runs"
        assert result["gathered_equals_tp1"], f"rank {rank}: gathered tensor differs from TP1"
        assert result["sha"] == result["tp1_sha"]
    assert len({result["sha"] for result in results}) == 1


class TestCrossTP:
    @requires_gpus(2)
    def test_tp2_bitwise_equal_to_tp1(self):
        _assert_kv_results(run_spawned(2, _kv_worker, "global_two_sequences"))

    @requires_gpus(4)
    def test_tp4_bitwise_equal_to_tp1(self):
        _assert_kv_results(run_spawned(4, _kv_worker, "global_two_sequences"))

    @requires_gpus(4)
    def test_tp2_equals_tp4(self):
        tp2 = run_spawned(2, _kv_worker, "local_small_window")
        tp4 = run_spawned(4, _kv_worker, "local_small_window")
        assert tp2[0]["sha"] == tp4[0]["sha"] == tp2[0]["tp1_sha"]

    @requires_gpus(8)
    def test_tp8_bitwise_equal_to_tp1(self):
        _assert_kv_results(run_spawned(8, _kv_worker, "global_two_sequences"))
