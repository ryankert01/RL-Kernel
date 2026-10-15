# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""``tp_canonical_tree`` (RFC #501 WS2): row-parallel o_proj / down on the canonical 8-leaf tree.

Real Muse-Glimmer shapes (M=256; o_proj K=4096 N=6656; down K=19968 N=6656), random weights.
Single-device tests simulate TP by folding per-rank sub-tree roots; the NCCL tests spawn one
process per GPU and go through the deterministic collective (TP2 / TP4 always, TP8 opt-in).
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from rl_engine.models.muse_glimmer.runtime import tp_canonical_tree as tct
from rl_engine.models.muse_glimmer.runtime.tp_layout import TPLayout, TPLayoutError, official_layout
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

M_ROWS = 256
SHAPES = {"o_proj": (4096, 6656), "down": (19968, 6656)}  # (K, N)
SEED = 20261015


def _inputs(shape_name: str, device: torch.device, seed: int = SEED):
    k_size, n_size = SHAPES[shape_name]
    gen = torch.Generator(device="cpu").manual_seed(seed)
    a = torch.randn(M_ROWS, k_size, generator=gen).to(device=device, dtype=torch.bfloat16)
    w = torch.randn(n_size, k_size, generator=gen).to(device=device, dtype=torch.bfloat16)
    return a, w


def _simulated_tp(a, w, tp: int) -> torch.Tensor:
    layout = official_layout(tp)
    width = a.size(1) // layout.canonical_tp
    roots = []
    for rank in range(tp):
        k0, k1 = layout.k_range(rank, width)
        roots.append(
            tct.subtree_partial_fp32(a[:, k0:k1], w[:, k0:k1], layout.canonical_chunks(rank))
        )
    return tct.combine_tree(torch.stack(roots)).to(torch.bfloat16)


# --------------------------------------------------------------------------- #
# Layout
# --------------------------------------------------------------------------- #


class TestLayout:
    @pytest.mark.parametrize("tp", [1, 2, 4, 8])
    def test_chunk_and_head_ownership(self, tp):
        layout = official_layout(tp)
        assert layout.chunks_per_rank == 8 // tp
        for rank in range(tp):
            chunks = layout.canonical_chunks(rank)
            assert list(chunks) == list(range(rank * 8 // tp, (rank + 1) * 8 // tp))
            assert layout.o_proj_k_range(rank) == (chunks.start * 512, chunks.stop * 512)
            assert layout.down_k_range(rank) == (chunks.start * 2496, chunks.stop * 2496)
            assert layout.vocab_range(rank) == (chunks.start * 25344, chunks.stop * 25344)
            for chunk in chunks:
                assert layout.chunk_owner(chunk) == rank
            # o_proj K-chunks line up with the owned Q heads (4 heads = 512 columns per chunk)
            assert layout.o_proj_k_range(rank) == layout.q_columns(rank)

    def test_unsupported_tp_and_undeclared_layout_fail_closed(self):
        with pytest.raises(TPLayoutError):
            TPLayout(tp=3)
        with pytest.raises(TPLayoutError):
            TPLayout(tp=16)
        layout = official_layout(2)
        with pytest.raises(TPLayoutError, match="world_size"):
            layout.fail_closed_if_undeclared(world_size=4)
        with pytest.raises(TPLayoutError, match="replicated"):
            layout.fail_closed_if_undeclared({"kv_heads_per_canonical_rank": "sharded"})
        small = TPLayout(tp=2, num_q_heads=8, head_dim=16, o_proj_chunk=16)
        assert not small.is_official()
        with pytest.raises(TPLayoutError, match="official"):
            small.fail_closed_if_undeclared()

    def test_trace_is_stable(self):
        layout = official_layout(4)
        first, second = tct.trace(layout), tct.trace(layout)
        assert first == second
        assert first["reduction_order"].endswith("((0+1)+(2+3))+((4+5)+(6+7))")
        assert first["chunk_to_rank"] == {0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 2, 6: 3, 7: 3}
        assert first["dtype_partial"] == "float32" and first["dtype_out"] == "bfloat16"


# --------------------------------------------------------------------------- #
# Single device
# --------------------------------------------------------------------------- #


@requires_gpus(1)
class TestSingleDevice:
    @pytest.fixture(autouse=True)
    def _tf32_off(self):
        disable_tf32()

    @pytest.mark.parametrize("shape_name", list(SHAPES))
    def test_tp1_repeat_is_bitwise(self, shape_name):
        a, w = _inputs(shape_name, torch.device("cuda"))
        first = tct.row_parallel_tree_gemm_tp1(a, w)
        second = tct.row_parallel_tree_gemm_tp1(a, w)
        assert first.dtype == torch.bfloat16 and first.shape == (M_ROWS, SHAPES[shape_name][1])
        assert torch.equal(first, second)

    @pytest.mark.parametrize("shape_name", list(SHAPES))
    @pytest.mark.parametrize("tp", [2, 4, 8])
    def test_simulated_tp_matches_tp1_bitwise(self, shape_name, tp):
        a, w = _inputs(shape_name, torch.device("cuda"))
        assert torch.equal(_simulated_tp(a, w, tp), tct.row_parallel_tree_gemm_tp1(a, w))

    @pytest.mark.parametrize("shape_name", list(SHAPES))
    def test_left_fold_over_eight_chunks_is_not_the_tree(self, shape_name):
        """Negative control: the same fp32 partials folded left-to-right differ in bf16."""

        a, w = _inputs(shape_name, torch.device("cuda"))
        layout = official_layout(8)
        width = a.size(1) // 8
        roots = [
            tct.subtree_partial_fp32(
                a[:, c * width : (c + 1) * width], w[:, c * width : (c + 1) * width], [c]
            )
            for c in range(8)
        ]
        folded = roots[0]
        for root in roots[1:]:
            folded = folded + root
        reference = tct.row_parallel_tree_gemm_tp1(a, w, layout)
        mismatches = int((folded.to(torch.bfloat16) != reference).sum())
        assert mismatches > 0, "left fold unexpectedly matched the canonical tree"

    def test_tp1_is_batch_invariant(self):
        a, w = _inputs("o_proj", torch.device("cuda"))
        full = tct.row_parallel_tree_gemm_tp1(a, w)
        alone = tct.row_parallel_tree_gemm_tp1(a[5:6].contiguous(), w)
        assert torch.equal(alone[0], full[5])

    def test_tree_bf16_is_close_to_fp32_reference(self):
        a, w = _inputs("o_proj", torch.device("cuda"))
        strict = tct.row_parallel_tree_gemm_tp1(a, w).float()
        reference = a.float() @ w.float().t()  # cuBLAS fp32 (TF32 off) as a tolerance oracle
        # ws1.json reduction/bfloat16 tolerance (atol 0.05, rtol 0.02).
        assert torch.allclose(strict, reference, atol=0.05, rtol=0.02)

    def test_fallback_strided_launch_matches_contiguous_transpose(self):
        from rl_engine.backends.shared.triton.gemm.det_gemm import _triton_gemm_fp32

        a, w = _inputs("o_proj", torch.device("cuda"))
        chunk_a, chunk_w = a[:, :512].contiguous(), w[:, :512].contiguous()
        direct = tct._fallback_chunk_partial_fp32(chunk_a, chunk_w)
        copied = _triton_gemm_fp32(chunk_a, chunk_w.t().contiguous())
        assert direct.dtype == torch.float32 and torch.equal(direct, copied)

    def test_undeclared_chunk_width_fails_closed(self):
        a = torch.randn(8, 1024, device="cuda").to(torch.bfloat16)
        w = torch.randn(64, 1024, device="cuda").to(torch.bfloat16)
        with pytest.raises(TPLayoutError, match="not a declared row-parallel shard"):
            tct.row_parallel_tree_gemm(a, w, official_layout(1))
        out = tct.row_parallel_tree_gemm(a, w, official_layout(1), chunk_width=128)
        assert out.shape == (8, 64)
        with pytest.raises(TPLayoutError, match="requires a deterministic collective"):
            tct.row_parallel_tree_gemm(a[:, :512], w[:, :512], official_layout(2))

    def test_combine_tree_requires_power_of_two(self):
        with pytest.raises(ValueError):
            tct._fallback_combine_tree([torch.zeros(2)] * 3)
        with pytest.raises(ValueError):
            tct.subtree_partial_fp32(
                torch.zeros(4, 1024, dtype=torch.bfloat16, device="cuda"),
                torch.zeros(4, 1024, dtype=torch.bfloat16, device="cuda"),
                [1, 2],
            )


# --------------------------------------------------------------------------- #
# NCCL ranks
# --------------------------------------------------------------------------- #


def _tree_worker(rank: int, world_size: int, shape_name: str) -> dict[str, Any]:
    import torch.distributed as dist

    device = torch.device("cuda", rank)
    layout = official_layout(world_size)
    layout.fail_closed_if_undeclared(world_size=dist.get_world_size())
    collective = tct.make_collective(dist.group.WORLD, device)
    a, w = _inputs(shape_name, device)
    width = a.size(1) // layout.canonical_tp
    k0, k1 = layout.k_range(rank, width)
    a_local, w_local = a[:, k0:k1].contiguous(), w[:, k0:k1].contiguous()

    out = tct.row_parallel_tree_gemm(a_local, w_local, layout, collective)
    rerun = tct.row_parallel_tree_gemm(a_local, w_local, layout, collective)
    reference = tct.row_parallel_tree_gemm_tp1(a, w, layout)
    local_root = tct.subtree_partial_fp32(a_local, w_local, layout.canonical_chunks(rank))
    control = tct.nccl_all_reduce_control(local_root)
    timings = tct.time_rank_reduction(local_root, collective, reps=10)
    result = {
        "shape": shape_name,
        "tp_equals_tp1": bool(torch.equal(out, reference)),
        "rerun_equal": bool(torch.equal(out, rerun)),
        "control_equal": bool(torch.equal(control, reference)),
        "control_mismatches": int((control != reference).sum()),
        "numel": int(reference.numel()),
        "sha": sha256_of(out),
        "tp1_sha": sha256_of(reference),
        "timings": timings,
        "trace": tct.trace(layout, collective),
    }
    if hasattr(collective, "close"):
        collective.close()
    return result


def _assert_tree_results(results: list[dict[str, Any]]) -> None:
    for result in results:
        rank = result["rank"]
        assert result["tp_equals_tp1"], f"rank {rank}: TP result differs from TP1 bits"
        assert result["rerun_equal"], f"rank {rank}: bits changed between identical runs"
        assert result["sha"] == result["tp1_sha"]
    assert len({result["sha"] for result in results}) == 1, "ranks disagree on the output"


class TestCrossTP:
    @requires_gpus(2)
    @pytest.mark.parametrize("shape_name", list(SHAPES))
    def test_tp2_bitwise_equal_to_tp1(self, shape_name):
        _assert_tree_results(run_spawned(2, _tree_worker, shape_name))

    @requires_gpus(4)
    @pytest.mark.parametrize("shape_name", list(SHAPES))
    def test_tp4_bitwise_equal_to_tp1(self, shape_name):
        _assert_tree_results(run_spawned(4, _tree_worker, shape_name))

    @requires_gpus(4)
    @pytest.mark.parametrize("shape_name", list(SHAPES))
    def test_tp2_equals_tp4(self, shape_name):
        tp2 = run_spawned(2, _tree_worker, shape_name)
        tp4 = run_spawned(4, _tree_worker, shape_name)
        assert tp2[0]["sha"] == tp4[0]["sha"] == tp2[0]["tp1_sha"]

    @requires_gpus(8)
    @pytest.mark.parametrize("shape_name", list(SHAPES))
    def test_tp8_bitwise_equal_to_tp1(self, shape_name):
        _assert_tree_results(run_spawned(8, _tree_worker, shape_name))

    @requires_gpus(2)
    def test_nccl_all_reduce_control_tp2_is_equal_by_commutativity(self):
        """Two operands: ``a + b == b + a`` exactly, so NCCL cannot differ at TP2."""

        for result in run_spawned(2, _tree_worker, "o_proj"):
            assert result["control_equal"], "NCCL all_reduce over 2 ranks must equal a + b"

    @requires_gpus(4)
    def test_nccl_all_reduce_control_tp4_is_not_the_tree(self):
        """Four operands: NCCL's ring/tree schedule is not ((0+1)+(2+3)); expect mismatches."""

        results = run_spawned(4, _tree_worker, "down")
        if all(result["control_equal"] for result in results):
            pytest.xfail(
                "NCCL all_reduce accidentally matched the canonical tree on this run "
                "(order-dependent; see REPORT_ws2_tp.md)"
            )
        assert any(result["control_mismatches"] > 0 for result in results)
