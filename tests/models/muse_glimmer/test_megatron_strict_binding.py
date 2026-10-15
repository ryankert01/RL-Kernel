# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Unit tests for the Megatron Bridge strict binding (fused-weight slicing, strict LoRA linears
with simulated TP, LM-head tile composition, LoRA export). No Bridge model is needed; the
30B runs live in ``run_megatron_strict.py``.
"""

from __future__ import annotations

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
    """Single-process TP emulation: pass 1 records each rank's all-gather inputs, pass 2
    replays with the peers' recorded inputs (every gathered value is rank-local)."""

    def __init__(self, world: int, rank: int, book: dict):
        self.world, self.rank, self.book = world, rank, book
        self.backend = "record_replay"
        self.mode = "record"
        self.counter = 0

    def all_gather(self, x):
        idx = self.counter
        self.counter += 1
        if self.mode == "record":
            self.book.setdefault(idx, {})[self.rank] = x.detach().clone()
            return [x.detach().clone() for _ in range(self.world)]
        return [self.book[idx][r] for r in range(self.world)]

    def all_gather_tree(self, p):
        return sk.gemm_backend().combine(self.all_gather(p)) if self.world > 1 else p


def _simulate(world, fn):
    """Run ``fn(comm)`` per rank in record mode, then replay; return per-rank replay results."""
    book: dict = {}
    comms = [RecordReplayComm(world, r, book) for r in range(world)]
    for c in comms:
        fn(c)
    outs = []
    for c in comms:
        c.mode = "replay"
        c.counter = 0
        outs.append(fn(c))
    return outs


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
    for _ in range(2):
        y = StrictColumnLinear.apply(xg, a2, b2, comm, n, w)
        gy = torch.randn_like(y)
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
    vocab, padded, hidden, m = 1000, 1024, 256, 24
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
