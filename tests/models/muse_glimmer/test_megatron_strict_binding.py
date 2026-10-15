# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Unit tests for the Megatron Bridge strict binding (fused-weight slicing, strict LoRA linears
with simulated TP, LM-head tile composition, LoRA export). No Bridge model is needed; the
30B runs live in ``run_megatron_strict.py``.
"""

from __future__ import annotations

import os

import pytest
import torch

from rl_engine.models.muse_glimmer.runtime import strict_kernels as sk
from rl_engine.models.muse_glimmer.runtime.megatron_lora import (
    LayerLoRA,
    ProjectionShape,
    StrictColumnLinear,
    StrictRowLinear,
    export_lora,
    import_lora,
    lora_sha256,
)
from rl_engine.models.muse_glimmer.runtime.megatron_strict import (
    GatherHeads,
    SliceLocalHeads,
    StrictLMHeadLogprob,
    fused_qkv_blocks,
    fused_qkv_to_hf,
)

CUDA = torch.cuda.is_available()
requires_cuda = pytest.mark.skipif(not CUDA, reason="CUDA required")


@pytest.fixture(autouse=True)
def _deterministic():
    sk.set_deterministic_torch()
    torch.manual_seed(0)


# --------------------------------------------------------------------------------------
# Fused linear_qkv layout (Bridge MuseGlimmerQKVGMapping + MCore merge_qkv_weights)
# --------------------------------------------------------------------------------------


def _bridge_merge(q, k, v, gate, heads, kv_heads, head_dim, hidden):
    """Re-implementation of Bridge's combine_query_and_gate + MCore merge_qkv_weights."""
    hpg = heads // kv_heads
    qg = torch.cat((q.view(heads, head_dim, hidden), gate.view(heads, head_dim, hidden)), dim=1)
    qg = qg.reshape(-1, hidden).view(heads, 2 * head_dim, hidden)
    q_r, z_r = torch.chunk(qg, 2, dim=1)
    k_r = k.view(kv_heads, head_dim, hidden)
    v_r = v.view(kv_heads, head_dim, hidden)
    parts = []
    for g in range(kv_heads):
        parts.extend([q_r[g * hpg : (g + 1) * hpg], z_r[g * hpg : (g + 1) * hpg], k_r[g : g + 1], v_r[g : g + 1]])
    return torch.cat(parts, dim=0).reshape(-1, hidden)


@pytest.mark.parametrize("heads,kv_heads,head_dim,hidden", [(4, 2, 8, 16), (32, 2, 128, 64)])
def test_fused_qkv_slicing_matches_bridge_layout(heads, kv_heads, head_dim, hidden):
    q = torch.randn(heads * head_dim, hidden).to(torch.bfloat16)
    k = torch.randn(kv_heads * head_dim, hidden).to(torch.bfloat16)
    v = torch.randn(kv_heads * head_dim, hidden).to(torch.bfloat16)
    gate = torch.randn(heads * head_dim, hidden).to(torch.bfloat16)
    fused = _bridge_merge(q, k, v, gate, heads, kv_heads, head_dim, hidden)
    assert fused.shape == (heads * head_dim * 2 + kv_heads * head_dim * 2, hidden)
    parts = fused_qkv_to_hf(fused, heads, kv_heads, head_dim)
    assert torch.equal(parts["q_proj"], q)
    assert torch.equal(parts["k_proj"], k)
    assert torch.equal(parts["v_proj"], v)
    assert torch.equal(parts["gate_proj"], gate)
    # TP2 shard = one group: the same slicing with groups_local=1 on half the rows.
    half = fused.shape[0] // kv_heads
    for r in range(kv_heads):
        qb, gb, kb, vb = fused_qkv_blocks(fused[r * half : (r + 1) * half], heads // kv_heads, head_dim, 1)
        hpg = heads // kv_heads
        assert torch.equal(qb[0], q[r * hpg * head_dim : (r + 1) * hpg * head_dim])
        assert torch.equal(kb[0], k[r * head_dim : (r + 1) * head_dim])
        assert torch.equal(vb[0], v[r * head_dim : (r + 1) * head_dim])
        assert torch.equal(gb[0], gate[r * hpg * head_dim : (r + 1) * hpg * head_dim])
    for blk in fused_qkv_blocks(fused, heads // kv_heads, head_dim, kv_heads):
        assert all(b.data_ptr() >= fused.data_ptr() for b in blk)  # views, no copies


# --------------------------------------------------------------------------------------
# Simulated TP: record/replay all-gather
# --------------------------------------------------------------------------------------


class RecordReplayComm:
    """Single-process TP emulation by fixed-point iteration over all-gather calls.

    Every pass runs each rank's function once. All-gather call ``idx`` returns this rank's live
    input in its own slot and the peers' inputs recorded at the same call index in the PREVIOUS
    pass (the rank's own value as a placeholder in the first pass), and records the live input
    for the next pass. A value gathered at call ``idx`` may depend on values gathered at
    earlier calls (e.g. the LM-head backward partials depend on the forward's gathered tile
    statistics through ``logZ``), so one record pass + one replay pass is NOT enough: after
    ``d + 1`` passes every gathered value of dependency depth ``d`` is exact. ``_simulate``
    iterates until two consecutive passes record bitwise-identical values.
    """

    def __init__(self, world: int, rank: int):
        self.world, self.rank = world, rank
        self.backend = "record_replay"
        self.prev: dict[int, dict[int, torch.Tensor]] = {}
        self.book: dict[int, dict[int, torch.Tensor]] = {}
        self.counter = 0

    def start_pass(self, prev: dict[int, dict[int, torch.Tensor]]) -> None:
        self.prev, self.book, self.counter = prev, {}, 0

    def all_gather(self, x):
        idx = self.counter
        self.counter += 1
        self.book.setdefault(idx, {})[self.rank] = x.detach().clone()
        peers = self.prev.get(idx, {})
        return [x if r == self.rank else peers.get(r, x).clone() for r in range(self.world)]

    def all_gather_tree(self, p):
        return sk.gemm_backend().combine(self.all_gather(p)) if self.world > 1 else p


def _books_equal(a: dict, b: dict) -> bool:
    if a.keys() != b.keys():
        return False
    for idx in a:
        if a[idx].keys() != b[idx].keys():
            return False
        if not all(torch.equal(a[idx][r], b[idx][r]) for r in a[idx]):
            return False
    return True


def _simulate(world, fn, max_passes: int = 6):
    """Iterate ``fn(comm)`` over all ranks until the gathered values reach a fixed point."""
    comms = [RecordReplayComm(world, r) for r in range(world)]
    prev: dict = {}
    for _ in range(max_passes):
        for c in comms:
            c.start_pass(prev)
        outs = [fn(c) for c in comms]
        book: dict = {}
        for c in comms:
            for idx, per_rank in c.book.items():
                book.setdefault(idx, {}).update(per_rank)
        if world == 1 or _books_equal(book, prev):
            return outs
        prev = book
    raise AssertionError(f"TP simulation did not converge in {max_passes} passes")


# --------------------------------------------------------------------------------------
# Strict column / row linears with LoRA
# --------------------------------------------------------------------------------------


@requires_cuda
def test_column_linear_lora_zero_is_bitwise_noop_and_repeatable():
    m, k, n, r = 37, 512, 256, 16
    x = torch.randn(m, k, device="cuda").to(torch.bfloat16)
    w = (torch.randn(n, k, device="cuda") * 0.05).to(torch.bfloat16)
    a = torch.randn(r, k, device="cuda") * 0.02
    b = torch.zeros(n, r, device="cuda")
    comm = sk.TPComm(None)
    y_plain = StrictColumnLinear.apply(x, None, None, comm, n, w)
    y_lora = StrictColumnLinear.apply(x, a, b, comm, n, w)
    assert torch.equal(y_plain, y_lora)
    ref = (x.double() @ w.double().t()).float()
    assert torch.allclose(y_plain.float(), ref, atol=5e-2, rtol=2e-2)
    # two blocks vs one block: identical bits
    y_blocks = StrictColumnLinear.apply(x, None, None, comm, n, w[: n // 2], w[n // 2 :])
    assert torch.equal(y_blocks, y_plain)
    # repeat determinism of forward and LoRA gradients
    b2 = (torch.randn(n, r, device="cuda") * 0.02).requires_grad_(True)
    a2 = a.clone().requires_grad_(True)
    xg = x.clone().requires_grad_(True)
    grads = []
    gy = torch.randn(m, n, device="cuda").to(torch.bfloat16)
    for _ in range(2):
        y = StrictColumnLinear.apply(xg, a2, b2, comm, n, w)
        g = torch.autograd.grad((y.float() * gy.float()).sum(), (xg, a2, b2))
        grads.append(g)
    for g1, g2 in zip(grads[0], grads[1], strict=True):
        assert torch.equal(g1, g2)
    # gradient correctness vs fp64 autograd of the unrounded function
    xd, ad, bd, wd = (t.detach().double().requires_grad_(True) for t in (xg, a2, b2, w))
    yd = xd @ wd.t() + (xd @ ad.t()) @ bd.t()
    gy_d = gy.double()
    gd = torch.autograd.grad((yd * gy_d).sum(), (xd, ad, bd))
    for got, want, tol in zip(grads[0], gd, (0.12, 0.05, 0.05), strict=True):
        assert torch.allclose(got.float(), want.float(), atol=tol, rtol=5e-2), (got - want).abs().max()


@requires_cuda
def test_column_linear_tp2_matches_tp1_bitwise():
    m, k, n, r = 64, 512, 512, 16
    x = torch.randn(m, k, device="cuda").to(torch.bfloat16)
    w = (torch.randn(n, k, device="cuda") * 0.05).to(torch.bfloat16)
    a = torch.randn(r, k, device="cuda") * 0.02
    b = torch.randn(n, r, device="cuda") * 0.02
    gy = torch.randn(m, n, device="cuda").to(torch.bfloat16)

    def run(comm):
        rank, world = comm.rank, comm.world
        n_l = n // world
        xg = x.clone().requires_grad_(True)
        ag = a.clone().requires_grad_(True)
        bg = b[rank * n_l : (rank + 1) * n_l].clone().requires_grad_(True)
        y = StrictColumnLinear.apply(xg, ag, bg, comm, n, w[rank * n_l : (rank + 1) * n_l])
        dx, da, db = torch.autograd.grad(y, (xg, ag, bg), gy[:, rank * n_l : (rank + 1) * n_l])
        return y, dx, da, db

    (y1, dx1, da1, db1) = _simulate(1, run)[0]
    outs = _simulate(2, run)
    assert torch.equal(torch.cat([o[0] for o in outs], dim=1), y1)
    for o in outs:
        assert torch.equal(o[1], dx1)
        assert torch.equal(o[2], da1)
    assert torch.equal(torch.cat([o[3] for o in outs], dim=0), db1)


@requires_cuda
def test_row_linear_tp2_matches_tp1_bitwise_and_lora_noop():
    m, k, n, r = 64, 1024, 256, 16
    x = torch.randn(m, k, device="cuda").to(torch.bfloat16)
    w = (torch.randn(n, k, device="cuda") * 0.05).to(torch.bfloat16)
    a = torch.randn(r, k, device="cuda") * 0.02
    b = torch.randn(n, r, device="cuda") * 0.02
    gy = torch.randn(m, n, device="cuda").to(torch.bfloat16)
    comm1 = sk.TPComm(None)
    y_plain = StrictRowLinear.apply(x, w, None, None, comm1, k)
    y_zero = StrictRowLinear.apply(x, w, a, torch.zeros_like(b), comm1, k)
    assert torch.equal(y_plain, y_zero)
    assert torch.allclose(y_plain.float(), (x.double() @ w.double().t()).float(), atol=5e-2, rtol=2e-2)

    def run(comm):
        rank, world = comm.rank, comm.world
        k_l = k // world
        xg = x[:, rank * k_l : (rank + 1) * k_l].clone().requires_grad_(True)
        ag = a[:, rank * k_l : (rank + 1) * k_l].clone().requires_grad_(True)
        bg = b.clone().requires_grad_(True)
        y = StrictRowLinear.apply(xg, w[:, rank * k_l : (rank + 1) * k_l], ag, bg, comm, k)
        dx, da, db = torch.autograd.grad(y, (xg, ag, bg), gy)
        return y, dx, da, db

    (y1, dx1, da1, db1) = _simulate(1, run)[0]
    outs = _simulate(2, run)
    for o in outs:
        assert torch.equal(o[0], y1)
        assert torch.equal(o[3], db1)
    assert torch.equal(torch.cat([o[1] for o in outs], dim=1), dx1)
    assert torch.equal(torch.cat([o[2] for o in outs], dim=1), da1)


# --------------------------------------------------------------------------------------
# LM head: canonical tiles, TP composition, reference
# --------------------------------------------------------------------------------------


@requires_cuda
def test_lm_head_tiles_tp2_equals_tp1_and_reference():
    vocab, padded, hidden, m = 2000, 2048, 256, 24
    tile_rows = padded // 8
    mult, cap = 0.19611613513818404, 20.0
    h = torch.randn(m, hidden, device="cuda").to(torch.bfloat16)
    w = (torch.randn(vocab, hidden, device="cuda") * 0.05).to(torch.bfloat16)
    w_pad = torch.nn.functional.pad(w, (0, 0, 0, padded - vocab))
    targets = torch.randint(0, vocab, (m,), device="cuda")

    def run(comm):
        ids = sk.local_chunk_ids(comm.rank, comm.world)
        tiles = w_pad[ids[0] * tile_rows : (ids[-1] + 1) * tile_rows].contiguous()
        hg = h.clone().requires_grad_(True)
        lp = StrictLMHeadLogprob.apply(hg, targets, tiles, tuple(ids), comm, vocab, tile_rows, mult, cap)
        (dh,) = torch.autograd.grad(lp.sum(), (hg,))
        return lp, dh

    lp1, dh1 = _simulate(1, run)[0]
    for lp2, dh2 in _simulate(2, run):
        assert torch.equal(lp2, lp1)
        assert torch.equal(dh2, dh1)
    z = (h.float() @ w.float().t()) * mult
    zh = cap * torch.tanh(z / cap)
    ref = torch.log_softmax(zh, dim=-1).gather(1, targets[:, None]).squeeze(1)
    assert torch.allclose(lp1, ref, atol=2e-2, rtol=1e-2), (lp1 - ref).abs().max()
    try:
        from rl_engine.models.muse_glimmer.ops.lm_head_softcap_logprob import lm_head_softcap_logprob
    except ImportError:
        return
    sib = lm_head_softcap_logprob(h, w_pad, targets, multiplier=mult, softcap=cap, vocab_size=vocab)
    assert torch.equal(sib, lp1), "binding LM-head composition must equal the sibling op at TP1"


@requires_cuda
def test_gather_and_slice_heads_roundtrip_grads():
    t, n = 8, 64
    y = torch.randn(t, n, device="cuda").to(torch.bfloat16)

    def run(comm):
        n_l = n // comm.world
        x_local = y[:, comm.rank * n_l : (comm.rank + 1) * n_l].clone().requires_grad_(True)
        full = GatherHeads.apply(x_local, comm)
        local = SliceLocalHeads.apply(full * 2, comm)
        (g,) = torch.autograd.grad((local.float() ** 2).sum(), (x_local,))
        return full, local, g

    f1, l1, g1 = _simulate(1, run)[0]
    outs = _simulate(2, run)
    for f, _l, _g in outs:
        assert torch.equal(f, y)
    assert torch.equal(torch.cat([o[1] for o in outs], dim=1), l1)
    assert torch.equal(torch.cat([o[2] for o in outs], dim=1), g1)


# --------------------------------------------------------------------------------------
# LoRA container: TP-invariant init, export/import, hash
# --------------------------------------------------------------------------------------


def test_lora_init_is_tp_invariant_and_export_roundtrip():
    shapes = (ProjectionShape("q_proj", 64, 32, "column"), ProjectionShape("o_proj", 32, 64, "row"))
    full = LayerLoRA(3, rank=4, shapes=shapes, tp_rank=0, tp_world=1, seed=7, device="cpu")
    r0 = LayerLoRA(3, rank=4, shapes=shapes, tp_rank=0, tp_world=2, seed=7, device="cpu")
    r1 = LayerLoRA(3, rank=4, shapes=shapes, tp_rank=1, tp_world=2, seed=7, device="cpu")
    a_full, b_full = full.pair("o_proj")
    assert torch.equal(r0.pair("o_proj")[0], a_full[:, :32])
    assert torch.equal(r1.pair("o_proj")[0], a_full[:, 32:])
    assert torch.equal(r0.pair("q_proj")[0], full.pair("q_proj")[0])
    assert b_full.abs().sum() == 0
    with torch.no_grad():
        for p in full.parameters():
            p.add_(torch.randn_like(p))
    state, sha = export_lora([full], sk.TPComm(None))
    assert set(state) == {"layers.3.q_proj.lora_A", "layers.3.q_proj.lora_B", "layers.3.o_proj.lora_A", "layers.3.o_proj.lora_B"}
    assert state["layers.3.q_proj.lora_A"].shape == (4, 32) and state["layers.3.o_proj.lora_B"].shape == (32, 4)
    assert sha == lora_sha256(state) and len(sha) == 64
    other = LayerLoRA(3, rank=4, shapes=shapes, tp_rank=0, tp_world=1, seed=99, device="cpu")
    import_lora([other], state, sha)
    for s in shapes:
        for got, want in zip(other.pair(s.name), full.pair(s.name), strict=True):
            assert torch.equal(got.to(torch.bfloat16), want.to(torch.bfloat16))
    with pytest.raises(ValueError):
        import_lora([other], state, "0" * 64)


def test_small_k_fp32_matches_sequential_reference():
    z = torch.randn(5, 16).to(torch.bfloat16)
    b = torch.randn(7, 16).to(torch.bfloat16)
    out = sk.small_k_fp32(z, b)
    ref = torch.zeros(5, 7)
    for k in range(16):
        ref = ref + z[:, k : k + 1].float() * b[:, k].float()[None, :]
    assert torch.equal(out, ref)
    assert torch.equal(sk.combine_tree_torch([torch.ones(2)] * 8), torch.full((2,), 8.0))


def test_lora_export_matches_e2e_sync_schema_and_hash():
    from rl_engine.models.muse_glimmer.e2e import sync
    from rl_engine.models.muse_glimmer.runtime.megatron_lora import (
        PROJECTIONS,
        muse_glimmer_projection_shapes,
    )

    shapes = muse_glimmer_projection_shapes(hidden=64, q_width=32, kv_width=16, ffn=48)
    layers = [
        LayerLoRA(i, rank=4, shapes=shapes, tp_rank=0, tp_world=1, seed=3, device="cpu", b_init_std=0.1)
        for i in range(2)
    ]
    state, sha = export_lora(layers, sk.TPComm(None))
    summary = sync.validate_lora_state_dict(state)
    assert summary["num_modules"] == 2 * len(PROJECTIONS) and summary["ranks"] == [4]
    assert sha == sync.lora_sha256(state) == lora_sha256(state)
    assert state["layers.1.mlp.down_proj.lora_A"].shape == (4, 48)
    assert state["layers.0.q_proj.lora_B"].shape == (32, 4)


def test_lora_b_init_and_reset_are_tp_invariant():
    shapes = (ProjectionShape("q_proj", 64, 32, "column"), ProjectionShape("o_proj", 32, 64, "row"))
    full = LayerLoRA(1, rank=4, shapes=shapes, tp_rank=0, tp_world=1, seed=5, device="cpu", b_init_std=0.5)
    shards = [
        LayerLoRA(1, rank=4, shapes=shapes, tp_rank=r, tp_world=2, seed=5, device="cpu", b_init_std=0.5)
        for r in range(2)
    ]
    assert full.pair("q_proj")[1].abs().sum() > 0
    assert torch.equal(torch.cat([s.pair("q_proj")[1] for s in shards], 0), full.pair("q_proj")[1])
    assert all(torch.equal(s.pair("o_proj")[1], full.pair("o_proj")[1]) for s in shards)
    for lora in (full, *shards):
        lora.reset_b(0.25, 9)
    assert torch.equal(torch.cat([s.pair("q_proj")[1] for s in shards], 0), full.pair("q_proj")[1])
    full.reset_b(0.0, 9)
    assert full.pair("o_proj")[1].abs().sum() == 0


def test_micro_batches_are_order_preserving_and_budgeted():
    from rl_engine.models.muse_glimmer.runtime.strict_trainer import micro_batches

    assert micro_batches([5, 5, 20, 3, 3], 8) == [[0], [1], [2], [3, 4]]
    assert micro_batches([3, 3, 2], 8) == [[0, 1, 2]]
    lens = [191, 146, 251, 146, 273, 274, 202, 294, 2439]
    groups = micro_batches(lens, 1024)
    assert [i for g in groups for i in g] == list(range(9))
    assert all(sum(lens[i] for i in g) <= 1024 or len(g) == 1 for g in groups)


@requires_cuda
def test_composed_strict_attention_matches_sibling_kernels():
    ops = sk.resolve_ops()
    if not ops.sources["gated_attention"].strict:
        pytest.skip("strict attention kernels unavailable")
    from rl_engine.models.muse_glimmer.ops.gated_attention import strict_gated_attention_fwd_varlen
    from rl_engine.models.muse_glimmer.triton.gated_attention_bwd import strict_gated_attention_bwd

    lens = [70, 130]
    t = sum(lens)
    cu = torch.tensor([0, 70, 200], dtype=torch.int32, device="cuda")
    pos = torch.cat([torch.arange(n) for n in lens]).to(device="cuda", dtype=torch.int32)
    q = torch.randn(t, 32, 128, device="cuda").to(torch.bfloat16).requires_grad_(True)
    k = torch.randn(t, 2, 128, device="cuda").to(torch.bfloat16).requires_grad_(True)
    v = torch.randn(t, 2, 128, device="cuda").to(torch.bfloat16).requires_grad_(True)
    g = torch.randn(t, 4096, device="cuda").to(torch.bfloat16).requires_grad_(True)
    dy = torch.randn(t, 4096, device="cuda").to(torch.bfloat16)
    for window in (None, 64):
        y = ops.gated_attention(q, k, v, g, cu, cu, pos, window=window)
        grads = torch.autograd.grad(y, (q, k, v, g), dy)
        y_ref, o_ref, lse_ref = strict_gated_attention_fwd_varlen(
            q.detach(), k.detach(), v.detach(), g.detach(), cu, cu, pos, window=window
        )
        assert torch.equal(y, y_ref)
        ref = strict_gated_attention_bwd(
            q.detach(), k.detach(), v.detach(), g.detach(), o_ref, lse_ref, dy, cu, cu, pos,
            window=window,
        )
        for got, want in zip(grads, ref, strict=True):
            assert torch.equal(got, want)


# --------------------------------------------------------------------------------------
# Real NCCL TP2 (2 GPUs): the simulated parity above, on real collectives
# --------------------------------------------------------------------------------------


def _nccl_tp2_worker(rank: int, port: int, out_dir: str) -> None:
    import torch.distributed as dist

    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=2, init_method=f"tcp://127.0.0.1:{port}")
    sk.set_deterministic_torch()
    comm1, comm2 = sk.TPComm(None), sk.TPComm(dist.group.WORLD)
    gen = torch.Generator().manual_seed(0)

    def rnd(*shape, scale=1.0):
        return (torch.randn(*shape, generator=gen) * scale).to(device="cuda", dtype=torch.bfloat16)

    res: dict[str, bool] = {}
    # LM head (canonical tiles, rank owns 4 of 8)
    vocab, padded, hidden, m = 2000, 2048, 256, 24
    rows = padded // 8
    h, w = rnd(m, hidden), rnd(vocab, hidden, scale=0.05)
    w_pad = torch.nn.functional.pad(w, (0, 0, 0, padded - vocab))
    targets = torch.randint(0, vocab, (m,), generator=gen).cuda()

    def lm(comm):
        ids = sk.local_chunk_ids(comm.rank, comm.world)
        tiles = w_pad[ids[0] * rows : (ids[-1] + 1) * rows].contiguous()
        hg = h.clone().requires_grad_(True)
        lp = StrictLMHeadLogprob.apply(
            hg, targets, tiles, tuple(ids), comm, vocab, rows, 0.19611613513818404, 20.0
        )
        (dh,) = torch.autograd.grad(lp.sum(), (hg,))
        return lp, dh

    lp1, dh1 = lm(comm1)
    lp2, dh2 = lm(comm2)
    res["lm_logp"] = torch.equal(lp1, lp2)
    res["lm_dh"] = torch.equal(dh1, dh2)
    # column linear with LoRA
    mm, k, n, r = 64, 512, 512, 16
    x, wc = rnd(mm, k), rnd(n, k, scale=0.05)
    a = torch.randn(r, k, generator=gen).cuda() * 0.02
    b = torch.randn(n, r, generator=gen).cuda() * 0.02
    gy = rnd(mm, n)

    def col(comm):
        n_l = n // comm.world
        sl = slice(comm.rank * n_l, (comm.rank + 1) * n_l)
        xg = x.clone().requires_grad_(True)
        ag = a.clone().requires_grad_(True)
        bg = b[sl].clone().requires_grad_(True)
        y = StrictColumnLinear.apply(xg, ag, bg, comm, n, wc[sl])
        return (y, *torch.autograd.grad(y, (xg, ag, bg), gy[:, sl]))

    c1, c2 = col(comm1), col(comm2)
    sl = slice(rank * n // 2, (rank + 1) * n // 2)
    res["col_y"] = torch.equal(c1[0][:, sl], c2[0])
    res["col_dx"] = torch.equal(c1[1], c2[1])
    res["col_da"] = torch.equal(c1[2], c2[2])
    res["col_db"] = torch.equal(c1[3][sl], c2[3])
    # row linear with LoRA
    k2 = 1024
    xr, wr = rnd(mm, k2), rnd(n, k2, scale=0.05)
    ar = torch.randn(r, k2, generator=gen).cuda() * 0.02

    def row(comm):
        k_l = k2 // comm.world
        s = slice(comm.rank * k_l, (comm.rank + 1) * k_l)
        xg = xr[:, s].clone().requires_grad_(True)
        ag = ar[:, s].clone().requires_grad_(True)
        bg = b.clone().requires_grad_(True)
        y = StrictRowLinear.apply(xg, wr[:, s], ag, bg, comm, k2)
        return (y, *torch.autograd.grad(y, (xg, ag, bg), gy))

    r1, r2 = row(comm1), row(comm2)
    s = slice(rank * k2 // 2, (rank + 1) * k2 // 2)
    res["row_y"] = torch.equal(r1[0], r2[0])
    res["row_dx"] = torch.equal(r1[1][:, s], r2[1])
    res["row_da"] = torch.equal(r1[2][:, s], r2[2])
    res["row_db"] = torch.equal(r1[3], r2[3])
    torch.save(res, f"{out_dir}/rank{rank}.pt")
    dist.destroy_process_group()


@pytest.mark.skipif(not CUDA or torch.cuda.device_count() < 2, reason="needs 2 GPUs")
def test_nccl_tp2_equals_tp1_lm_head_column_row(tmp_path):
    import socket

    import torch.multiprocessing as mp

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    mp.spawn(_nccl_tp2_worker, args=(port, str(tmp_path)), nprocs=2, join=True)
    for rank in range(2):
        res = torch.load(tmp_path / f"rank{rank}.pt")
        assert all(res.values()), {k: v for k, v in res.items() if not v}


# --------------------------------------------------------------------------------------
# TrainerLike protocol on the real 30B checkpoint (opt-in: RL_KERNEL_MUSEGLIMMER_E2E=1)
# --------------------------------------------------------------------------------------


@pytest.mark.skipif(
    os.environ.get("RL_KERNEL_MUSEGLIMMER_E2E") != "1", reason="set RL_KERNEL_MUSEGLIMMER_E2E=1"
)
def test_strict_trainer_protocol_real_checkpoint():
    import json
    from pathlib import Path

    from rl_engine.models.muse_glimmer.e2e import sync
    from rl_engine.models.muse_glimmer.e2e.engines import TrainerLike
    from rl_engine.models.muse_glimmer.runtime.strict_trainer import StrictTrainer

    tp = int(os.environ.get("MUSEGLIMMER_TRAIN_TP", "1"))
    ckpt = os.path.expanduser(os.environ.get("MUSEGLIMMER_CHECKPOINT", "~/models/Muse-Glimmer-30B"))
    xrt = Path(__file__).resolve().parents[3] / "reports" / "muse_glimmer_poc" / "data" / "xrt_tokens.json"
    data = json.loads(xrt.read_text())["sequences"][:3]
    seqs = [s["token_ids"] for s in data]
    starts = [s["completion_start"] for s in data]
    for name in ("token_logprobs", "step", "export_lora", "provenance", "close"):
        assert callable(getattr(StrictTrainer, name)) and hasattr(TrainerLike, name)
    trainer = StrictTrainer(ckpt, tensor_parallel_size=tp, gpu_ids=list(range(tp)), lora_rank=16,
                            lr=1e-4, strict=True, seed=1234)
    try:
        pkg0 = sync.export_lora_verified(trainer, step=0)  # hash + key schema
        assert trainer.provenance()["lora_sha256"] == pkg0.sha256
        lps = trainer.token_logprobs(seqs, starts)
        assert len(lps) == len(seqs)
        for lp, seq, st in zip(lps, seqs, starts, strict=True):
            assert lp.dtype == torch.float32 and lp.shape == (len(seq) - st,) and lp.requires_grad
            assert bool(torch.isfinite(lp).all()) and bool((lp <= 0).all())
        gen = torch.Generator().manual_seed(7)
        g = [torch.randn(lp.shape[0], generator=gen) for lp in lps]
        loss = sum((lp * gi.to(lp.device)).sum() for lp, gi in zip(lps, g, strict=True))
        info = trainer.step(loss)  # the E2E bridge's surrogate loss sum_i (lp_i * g_i).sum()
        assert info["recompute_logp_mismatches"] == 0
        assert info["recompute_logp_checked"] == sum(lp.numel() for lp in lps)
        assert info["grad_norm"] > 0 and info["step"] == 1
        pkg1 = sync.export_lora_verified(trainer, step=1)
        assert pkg1.sha256 != pkg0.sha256
        assert trainer.provenance()["lora_sha256"] == pkg1.sha256
        # any scalar loss: only the first sequence, non-linear in logp
        lps2 = trainer.token_logprobs(seqs, starts)
        info2 = trainer.step(-(lps2[0].exp().mean()))
        assert info2["recompute_logp_mismatches"] == 0 and info2["step"] == 2
    finally:
        trainer.close()
