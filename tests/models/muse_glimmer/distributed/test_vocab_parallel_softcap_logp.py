# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""``vocab_parallel_softcap_logp`` (RFC #501 WS2): soft-capped logprob on 8 canonical vocab tiles.

Real shapes: T=64 tokens, H=6656, V=202048 padded to 202752 (8 tiles of 25344), random LM head.
Targets cover every tile-position class (first column of tile 0, the last column of every tile,
the last real column 202047); a target in the 704-row pad region must raise on every rank.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from rl_engine.models.muse_glimmer.runtime import vocab_parallel_softcap_logp as vp
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

TOKENS = 64
HIDDEN = 6656
VOCAB = 202048
PADDED = 202752
TILE = 25344
SEED = 20261015
# One target per tile-position class, then random real-vocabulary ids.
SPECIAL_TARGETS = [0, TILE - 1, TILE, 2 * TILE - 1, 4 * TILE - 1, 7 * TILE - 1, 7 * TILE, VOCAB - 1]


def _inputs(device: torch.device, seed: int = SEED):
    """Identical on every rank: h and targets seeded on CPU, the 2.7 GB LM head on the GPU."""

    gen = torch.Generator(device="cpu").manual_seed(seed)
    h = torch.randn(TOKENS, HIDDEN, generator=gen).to(device=device, dtype=torch.bfloat16)
    targets = torch.randint(0, VOCAB, (TOKENS,), generator=gen)
    targets[: len(SPECIAL_TARGETS)] = torch.tensor(SPECIAL_TARGETS)
    cuda_gen = torch.Generator(device=device).manual_seed(seed + 1)
    w_lm = torch.randn(VOCAB, HIDDEN, device=device, dtype=torch.bfloat16, generator=cuda_gen)
    w_pad = vp.pad_lm_head_weight(w_lm)
    del w_lm
    return h, w_pad, targets.to(device)


@pytest.fixture(scope="module")
def inputs():
    disable_tf32()
    return _inputs(torch.device("cuda"))


def _tile_logits(h, w_pad):
    return [vp.tile_logits_fp32(h, w_pad[t * TILE : (t + 1) * TILE]) for t in range(8)]


def _simulated_tp(h, w_pad, targets, tp: int) -> torch.Tensor:
    layout = official_layout(tp)
    stacks: list[list[torch.Tensor]] = [[], [], [], []]
    for rank in range(tp):
        local = []
        for tile in layout.vocab_tiles(rank):
            z = vp.tile_logits_fp32(h, w_pad[tile * TILE : (tile + 1) * TILE])
            local.append(
                vp.softcap_tile_stats(
                    z,
                    tile,
                    targets,
                    multiplier=vp.OUTPUT_MULTIPLIER,
                    softcap=vp.FINAL_LOGIT_SOFTCAP,
                    vocab_size=VOCAB,
                )
            )
        for index in range(4):
            stacks[index].append(torch.stack([s[index] for s in local]))
    m_all, s_all, tl_all, _own_all = (torch.cat(x) for x in stacks)
    _, log_z = vp.combine_tile_stats((m_all, s_all))
    rows = torch.arange(TOKENS, device=h.device)
    return tl_all[targets // TILE, rows] - log_z


@requires_gpus(1)
class TestSingleDevice:
    def test_padding_adds_704_zero_rows(self):
        w = torch.randn(VOCAB, 8, device="cuda").to(torch.bfloat16)
        padded = vp.pad_lm_head_weight(w)
        assert padded.shape == (PADDED, 8)
        assert torch.equal(padded[:VOCAB], w) and bool((padded[VOCAB:] == 0).all())

    def test_targets_cover_every_tile(self, inputs):
        _, _, targets = inputs
        assert set((targets[: len(SPECIAL_TARGETS)] // TILE).tolist()) >= {0, 1, 3, 6, 7}
        assert set((targets // TILE).tolist()) == set(range(8))

    def test_tp1_matches_fp64_reference(self, inputs):
        h, w_pad, targets = inputs
        logp = vp.vocab_parallel_softcap_logp(h, w_pad, targets, official_layout(1))
        reference = vp.vocab_softcap_logp_reference_fp64(
            _tile_logits(h, w_pad), targets, vocab_size=VOCAB
        )
        assert logp.dtype == torch.float32 and logp.shape == (TOKENS,)
        # ws1.json logprob/float32: atol 1e-5, rtol 0 (same fp32 logits feed both sides).
        assert torch.allclose(logp.double(), reference, atol=1e-5, rtol=0.0)
        assert bool((logp < 0).all()) and bool(torch.isfinite(logp).all())

    def test_tp1_repeat_is_bitwise(self, inputs):
        h, w_pad, targets = inputs
        layout = official_layout(1)
        first = vp.vocab_parallel_softcap_logp(h, w_pad, targets, layout)
        second = vp.vocab_parallel_softcap_logp(h, w_pad, targets, layout)
        assert torch.equal(first, second)

    @pytest.mark.parametrize("tp", [2, 4, 8])
    def test_simulated_tp_matches_tp1_bitwise(self, inputs, tp):
        h, w_pad, targets = inputs
        reference = vp.vocab_parallel_softcap_logp(h, w_pad, targets, official_layout(1))
        assert torch.equal(_simulated_tp(h, w_pad, targets, tp), reference)

    @pytest.mark.parametrize("bad", [VOCAB, PADDED - 1, PADDED, -1])
    def test_pad_region_or_invalid_target_raises(self, inputs, bad):
        h, w_pad, targets = inputs
        poisoned = targets.clone()
        poisoned[3] = bad
        with pytest.raises(ValueError, match="real vocabulary"):
            vp.vocab_parallel_softcap_logp(h, w_pad, poisoned, official_layout(1))

    def test_padded_logits_are_excluded_bitwise(self, inputs):
        h, w_pad, targets = inputs
        layout = official_layout(1)
        reference = vp.vocab_parallel_softcap_logp(h, w_pad, targets, layout)
        poisoned = w_pad.clone()
        poisoned[VOCAB:] = 64.0  # huge padded logits must not touch M_t, S_t or logp bits
        assert torch.equal(vp.vocab_parallel_softcap_logp(h, poisoned, targets, layout), reference)

    def test_tile_stats_are_batch_invariant(self, inputs):
        h, w_pad, targets = inputs
        z = vp.tile_logits_fp32(h, w_pad[7 * TILE : 8 * TILE])
        kwargs = dict(
            multiplier=vp.OUTPUT_MULTIPLIER, softcap=vp.FINAL_LOGIT_SOFTCAP, vocab_size=VOCAB
        )
        full = vp.softcap_tile_stats(z, 7, targets, **kwargs)
        row = vp.softcap_tile_stats(z[7:8].contiguous(), 7, targets[7:8], **kwargs)
        for stat_full, stat_row in zip(full, row, strict=True):
            assert torch.equal(stat_full[7:8], stat_row)
        assert bool(full[3][7] == 1.0)  # target 202047 lives in tile 7

    def test_softcap_follows_hf_order_in_fp32(self):
        z = torch.linspace(-400.0, 400.0, 1001, device="cuda")
        capped = vp.softcap_fp32(z, multiplier=vp.OUTPUT_MULTIPLIER, softcap=vp.FINAL_LOGIT_SOFTCAP)
        mult = torch.tensor(vp.OUTPUT_MULTIPLIER, device="cuda")
        cap = torch.tensor(vp.FINAL_LOGIT_SOFTCAP, device="cuda")
        hf = cap * ((z * mult) / cap).tanh()  # mul m -> div T -> tanh -> mul T, IEEE division
        assert torch.equal(capped, hf) and bool((capped.abs() <= vp.FINAL_LOGIT_SOFTCAP).all())
        assert torch.allclose(
            capped, ((z * vp.OUTPUT_MULTIPLIER) / vp.FINAL_LOGIT_SOFTCAP).tanh() * 20.0, atol=1e-5
        )

    def test_wrong_local_tile_count_fails_closed(self, inputs):
        h, w_pad, targets = inputs
        with pytest.raises(ValueError, match="owns 8 tiles"):
            vp.vocab_parallel_softcap_logp(h, w_pad[:TILE], targets, official_layout(1))

    def test_trace_is_stable(self):
        layout = official_layout(2)
        assert vp.trace(layout) == vp.trace(layout)
        assert vp.trace(layout)["tile_to_rank"] == {0: 0, 1: 0, 2: 0, 3: 0, 4: 1, 5: 1, 6: 1, 7: 1}


# --------------------------------------------------------------------------- #
# NCCL ranks
# --------------------------------------------------------------------------- #


def _vocab_worker(rank: int, world_size: int) -> dict[str, Any]:
    import torch.distributed as dist

    from rl_engine.models.muse_glimmer.runtime.tp_canonical_tree import make_collective

    device = torch.device("cuda", rank)
    layout = official_layout(world_size)
    layout.fail_closed_if_undeclared(world_size=dist.get_world_size())
    collective = make_collective(dist.group.WORLD, device)
    h, w_pad, targets = _inputs(device)
    v0, v1 = layout.vocab_range(rank)
    w_local = w_pad[v0:v1].contiguous()

    logp = vp.vocab_parallel_softcap_logp(h, w_local, targets, layout, collective)
    rerun = vp.vocab_parallel_softcap_logp(h, w_local, targets, layout, collective)
    reference = vp.vocab_parallel_softcap_logp(h, w_pad, targets, official_layout(1))
    poisoned = targets.clone()
    poisoned[0] = VOCAB
    try:
        vp.vocab_parallel_softcap_logp(h, w_local, poisoned, layout, collective)
        pad_raised = False
    except ValueError:
        pad_raised = True
    result = {
        "tp_equals_tp1": bool(torch.equal(logp, reference)),
        "rerun_equal": bool(torch.equal(logp, rerun)),
        "pad_target_raised": pad_raised,
        "sha": sha256_of(logp),
        "tp1_sha": sha256_of(reference),
        "trace": vp.trace(layout, collective),
    }
    if hasattr(collective, "close"):
        collective.close()
    return result


def _assert_vocab_results(results: list[dict[str, Any]]) -> None:
    for result in results:
        rank = result["rank"]
        assert result["tp_equals_tp1"], f"rank {rank}: logp bits differ from TP1"
        assert result["rerun_equal"], f"rank {rank}: logp bits changed between identical runs"
        assert result["pad_target_raised"], f"rank {rank}: padded target did not raise"
    assert len({result["sha"] for result in results}) == 1


class TestCrossTP:
    @requires_gpus(2)
    def test_tp2_bitwise_equal_to_tp1(self):
        _assert_vocab_results(run_spawned(2, _vocab_worker))

    @requires_gpus(4)
    def test_tp4_bitwise_equal_to_tp1(self):
        _assert_vocab_results(run_spawned(4, _vocab_worker))

    @requires_gpus(4)
    def test_tp2_equals_tp4(self):
        tp2 = run_spawned(2, _vocab_worker)
        tp4 = run_spawned(4, _vocab_worker)
        assert tp2[0]["sha"] == tp4[0]["sha"] == tp2[0]["tp1_sha"]

    @requires_gpus(8)
    def test_tp8_bitwise_equal_to_tp1(self):
        _assert_vocab_results(run_spawned(8, _vocab_worker))
