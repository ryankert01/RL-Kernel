# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Unit tests for the Megatron Bridge strict binding (fused-weight slicing, strict LoRA linears
with simulated TP, per-rank attention heads, LM-head tile composition, LoRA export, fail-closed
op resolution, the packing-invariant LoRA gradient rule, the recompute check). No Bridge model
is needed; the 30B runs live in ``run_megatron_strict.py``.
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
from rl_engine.models.muse_glimmer.runtime.megatron_attention import (
    GatherKVHead,
    attention_core,
    kv_replica_ranks,
    rank_gated_attention,
    rank_head_geometry,
)
from rl_engine.models.muse_glimmer.runtime.megatron_strict import (
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
def test_column_linear_tp_matches_tp1_bitwise():
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
    for world in (2, 4, 8):
        outs = _simulate(world, run)
        assert torch.equal(torch.cat([o[0] for o in outs], dim=1), y1)
        for o in outs:
            assert torch.equal(o[1], dx1)
            assert torch.equal(o[2], da1)
        assert torch.equal(torch.cat([o[3] for o in outs], dim=0), db1)


@requires_cuda
def test_row_linear_tp_matches_tp1_bitwise_and_lora_noop():
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
    for world in (2, 4, 8):
        outs = _simulate(world, run)
        for o in outs:
            assert torch.equal(o[0], y1)
            assert torch.equal(o[3], db1)
        assert torch.equal(torch.cat([o[1] for o in outs], dim=1), dx1)
        assert torch.equal(torch.cat([o[2] for o in outs], dim=1), da1)


# --------------------------------------------------------------------------------------
# LM head: canonical tiles, TP composition, reference
# --------------------------------------------------------------------------------------


@requires_cuda
def test_lm_head_tiles_tp_equals_tp1_and_reference():
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
    for world in (2, 4, 8):
        for lp2, dh2 in _simulate(world, run):
            assert torch.equal(lp2, lp1), world
            assert torch.equal(dh2, dh1), world
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


def test_rank_head_geometry_matches_the_contract_layout():
    assert rank_head_geometry(0, 1) == {
        "q_heads": 32, "kv_heads": 2, "q_head_start": 0, "kv_head": 0, "replicas": 1,
    }
    assert [rank_head_geometry(r, 2)["kv_head"] for r in range(2)] == [0, 1]
    assert [rank_head_geometry(r, 4)["kv_head"] for r in range(4)] == [0, 0, 1, 1]
    assert [rank_head_geometry(r, 8)["kv_head"] for r in range(8)] == [0, 0, 0, 0, 1, 1, 1, 1]
    assert [rank_head_geometry(r, 8)["q_head_start"] for r in range(8)] == list(range(0, 32, 4))
    assert kv_replica_ranks(3, 4) == [2, 3] and kv_replica_ranks(5, 8) == [4, 5, 6, 7]
    assert kv_replica_ranks(1, 2) == [1]


@requires_cuda
@pytest.mark.parametrize("world", [1, 2, 4, 8])
def test_gather_kv_head_forward_and_backward_without_double_counting(world):
    """Forward: the rank's whole KV head. Backward: only the rank's own k/v columns, so the
    concatenation over ranks of the returned gradients equals the full head gradients (each
    projection row's gradient comes from exactly one rank)."""
    t = 8
    k_full = torch.randn(t, 256, device="cuda").to(torch.bfloat16)
    g_heads = torch.randn(t, 256, device="cuda").to(torch.bfloat16)  # d loss / d (both heads)

    def run(comm):
        n_l = 256 // comm.world
        x_local = k_full[:, comm.rank * n_l : (comm.rank + 1) * n_l].clone().requires_grad_(True)
        head = GatherKVHead.apply(x_local, comm)
        geo = rank_head_geometry(comm.rank, comm.world)
        cols = slice(geo["kv_head"] * 128, (geo["kv_head"] + geo["kv_heads"]) * 128)
        (g,) = torch.autograd.grad(head, (x_local,), g_heads[:, cols])
        return head, g, cols

    for head, g, cols in _simulate(world, run):
        assert torch.equal(head, k_full[:, cols])
    outs = _simulate(world, run)
    assert torch.equal(torch.cat([o[1] for o in outs], dim=1), g_heads)


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
    assert len(sha) == 64 and sha == lora_sha256(state)
    assert state["layers.1.mlp.down_proj.lora_A"].shape == (4, 48)
    assert state["layers.0.q_proj.lora_B"].shape == (32, 4)


def test_lora_hash_matches_the_pinned_muse_glimmer_lora_sha256_v1_stream():
    """A fixed state hashes to a pinned digest, recomputed here from the documented byte stream
    (sorted keys; ``name|dtype|shape\\n`` then the little-endian bf16 bytes) without calling the
    e2e implementation, so a schema drift on either side fails."""
    import hashlib

    state = {
        "layers.0.q_proj.lora_A": (torch.arange(32, dtype=torch.float32).reshape(4, 8) / 16 - 1).to(
            torch.bfloat16
        ),
        "layers.0.q_proj.lora_B": (torch.arange(24, dtype=torch.float32).reshape(6, 4) / -32).to(
            torch.bfloat16
        ),
    }
    pinned = "a4ec80654da4322c144aebc0bae4084fcd4f90df945d334c36d6398caf4a2758"
    digest = hashlib.sha256()
    for name in sorted(state):
        t = state[name]
        digest.update(f"{name}|{t.dtype}|{','.join(str(d) for d in t.shape)}\n".encode())
        digest.update(t.contiguous().view(torch.int16).numpy().astype("<i2").tobytes())
    assert digest.hexdigest() == pinned
    assert lora_sha256(state) == pinned


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


def _attn_inputs(lens, *, seed=0, hq=32, hkv=2):
    gen = torch.Generator().manual_seed(seed)
    t = sum(lens)

    def rnd(*shape, scale=1.0):
        return (torch.randn(*shape, generator=gen) * scale).to(device="cuda", dtype=torch.bfloat16)

    cu = torch.tensor([0, *torch.cumsum(torch.tensor(lens), 0).tolist()], dtype=torch.int32,
                      device="cuda")
    pos = torch.cat([torch.arange(n) for n in lens]).to(device="cuda", dtype=torch.int32)
    return {
        "q": rnd(t, hq, 128).requires_grad_(True),
        "k": rnd(t, hkv, 128).requires_grad_(True),
        "v": rnd(t, hkv, 128).requires_grad_(True),
        "g": rnd(t, hq * 128).requires_grad_(True),
        "dy": rnd(t, hq * 128),
        "cu": cu,
        "pos": pos,
    }


def _normalized_err(got, ref):
    scale = float(ref.abs().max())
    return float((got.double() - ref.double()).abs().max()) / max(scale, 1e-30)


@requires_cuda
@pytest.mark.parametrize("window", [None, 64], ids=["global", "local"])
def test_rank_attention_tp1_equals_sibling_autograd_and_fp64_golden(window):
    """At TP1 the binding's per-rank attention (sibling fwd + preprocess + partials + subtree)
    equals the sibling autograd op bit for bit, and its gradients are within the attention row's
    normalized bound (max|err|/max|ref| <= 1e-2) of the FP64 autograd golden."""
    ops = sk.resolve_ops()
    from rl_engine.models.muse_glimmer.ops.attention_golden_bwd import reference_backward
    from rl_engine.models.muse_glimmer.ops.gated_attention_autograd import gated_attention

    inp = _attn_inputs([70, 130], seed=3)
    leaves = (inp["q"], inp["k"], inp["v"], inp["g"])
    comm = sk.TPComm(None)
    y = rank_gated_attention(*leaves, inp["cu"], inp["cu"], inp["pos"], window=window, comm=comm,
                             kernels=ops.attention)
    grads = torch.autograd.grad(y, leaves, inp["dy"])
    y_sib = gated_attention(*leaves, inp["cu"], inp["cu"], inp["pos"], window)
    grads_sib = torch.autograd.grad(y_sib, leaves, inp["dy"])
    assert torch.equal(y, y_sib)
    for got, want in zip(grads, grads_sib, strict=True):
        assert torch.equal(got, want)
    ref = reference_backward(*(t.detach() for t in leaves), inp["cu"], inp["cu"], inp["pos"],
                             inp["dy"], window=window)
    for name, got, want in zip(("dq", "dk", "dv", "dgate"), grads, ref, strict=True):
        assert _normalized_err(got, want) <= 1e-2, name


def _attention_block_case(device, seed=11):
    hidden, r = 512, 4
    lens = [96, 160]
    t = sum(lens)
    gen = torch.Generator().manual_seed(seed)

    def rnd(*shape, scale=1.0):
        return (torch.randn(*shape, generator=gen) * scale).to(device=device, dtype=torch.bfloat16)

    case = {"t": t, "hidden": hidden, "x": rnd(t, hidden)}
    fused = rnd(2 * (2 * 16 + 2) * 128, hidden, scale=0.05)  # the FUSED Bridge linear_qkv
    case["hf"] = fused_qkv_to_hf(fused, 32, 2, 128)  # un-fused exactly as the binding does
    case["w_o"] = rnd(hidden, 4096, scale=0.02)
    case["gy"] = rnd(t, hidden)
    case["cu"] = torch.tensor([0, lens[0], t], dtype=torch.int32, device=device)
    case["pos"] = torch.cat([torch.arange(n) for n in lens]).to(device=device, dtype=torch.int32)
    case["lora"] = {
        name: (
            (torch.randn(r, k_in, generator=gen) * 0.05).to(device),
            (torch.randn(n_out, r, generator=gen) * 0.05).to(device),
        )
        for name, n_out, k_in in (
            ("q_proj", 4096, hidden), ("gate_proj", 4096, hidden), ("k_proj", 256, hidden),
            ("v_proj", 256, hidden), ("o_proj", hidden, 4096),
        )
    }
    return case


def _attention_block(comm, case, ops, layer):
    """The binding's attention half on one rank: four column projections (canonical row shards,
    LoRA), ``attention_core`` (per-rank heads, KV head gathered at TP4/8, replica fold), the
    row-parallel ``o_proj`` (LoRA). Returns the output and every gradient."""
    w, rank = comm.world, comm.rank
    t = case["t"]

    def rows(tensor):
        n_l = tensor.shape[0] // w
        return tensor[rank * n_l : (rank + 1) * n_l]

    device = case["x"].device
    if layer == "local":
        cos, sin = ops.rope_rows(case["pos"].long(), device=device)
        window = 64
    else:
        cos = sin = None
        window = None
    xg = case["x"].clone().requires_grad_(True)
    params, outs = {}, {}
    for name in ("q_proj", "gate_proj", "k_proj", "v_proj"):
        a, b = case["lora"][name]
        params[name] = (a.clone().requires_grad_(True), rows(b).clone().requires_grad_(True))
        n_total = case["hf"][name].shape[0]
        outs[name] = StrictColumnLinear.apply(
            xg, *params[name], comm, n_total, rows(case["hf"][name]).contiguous(),
            segments=((0, 96), (96, t)),
        )
    y_local = attention_core(
        outs["q_proj"], outs["k_proj"], outs["v_proj"], outs["gate_proj"], ops=ops, comm=comm,
        cos=cos, sin=sin, cu_seqlens=case["cu"], positions=case["pos"], window=window,
        rms_eps=1e-5, qk_scale=3.87,
    )
    k_l = 4096 // w
    cols = slice(rank * k_l, (rank + 1) * k_l)
    a_o, b_o = case["lora"]["o_proj"]
    params["o_proj"] = (a_o[:, cols].clone().requires_grad_(True), b_o.clone().requires_grad_(True))
    o = StrictRowLinear.apply(y_local, case["w_o"][:, cols].contiguous(), *params["o_proj"], comm,
                              4096, segments=((0, 96), (96, t)))
    (dx,) = torch.autograd.grad((o.float() * case["gy"].float()).sum(), (xg,))
    named = {"dx": dx}
    for name, (a, b) in params.items():  # segment mode: the op accumulated into .grad
        named[f"{name}.A"], named[f"{name}.B"] = a.grad, b.grad
    return o, named


def _check_block_parity(o1, g1, outs):
    for o_w, g_w in outs:
        assert torch.equal(o_w, o1)
        assert torch.equal(g_w["dx"], g1["dx"])
    for name in ("q_proj", "gate_proj", "k_proj", "v_proj"):
        for _o, g_w in outs:
            assert torch.equal(g_w[f"{name}.A"], g1[f"{name}.A"]), name  # replicated A
        assert torch.equal(torch.cat([g[f"{name}.B"] for _, g in outs], 0), g1[f"{name}.B"]), name
    assert torch.equal(torch.cat([g["o_proj.A"] for _, g in outs], 1), g1["o_proj.A"])
    for _o, g_w in outs:
        assert torch.equal(g_w["o_proj.B"], g1["o_proj.B"])


@requires_cuda
@pytest.mark.parametrize("layer", ["local", "global"])
@pytest.mark.parametrize("world", [2, 4, 8])
def test_attention_block_per_rank_heads_tp_equals_tp1(world, layer):
    """TP2: 16 Q / 1 KV head per rank; TP4: 8 / 1 (KV head replicated on 2 ranks); TP8: 4 / 1
    (4 replicas). Output, dX and every TP-gathered LoRA gradient equal TP1 bitwise."""
    ops = sk.resolve_ops()
    case = _attention_block_case("cuda")
    o1, g1 = _simulate(1, lambda c: _attention_block(c, case, ops, layer))[0]
    outs = _simulate(world, lambda c: _attention_block(c, case, ops, layer), max_passes=12)
    _check_block_parity(o1, g1, outs)


# --------------------------------------------------------------------------------------
# Real NCCL TP2 / TP4: the simulated parity above, on real collectives
# --------------------------------------------------------------------------------------


def _nccl_tp_worker(rank: int, world: int, port: int, out_dir: str) -> None:
    import torch.distributed as dist

    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl", rank=rank, world_size=world, init_method=f"tcp://127.0.0.1:{port}",
        device_id=torch.device("cuda", rank),
    )
    sk.set_deterministic_torch()
    comm1, comm_w = sk.TPComm(None), sk.TPComm(dist.group.WORLD)
    gen = torch.Generator().manual_seed(0)

    def rnd(*shape, scale=1.0):
        return (torch.randn(*shape, generator=gen) * scale).to(device="cuda", dtype=torch.bfloat16)

    res: dict[str, bool] = {}
    # LM head (canonical tiles, rank owns 8/world of 8)
    vocab, padded, hidden, m = 2000, 2048, 256, 24
    tile_rows = padded // 8
    h, w = rnd(m, hidden), rnd(vocab, hidden, scale=0.05)
    w_pad = torch.nn.functional.pad(w, (0, 0, 0, padded - vocab))
    targets = torch.randint(0, vocab, (m,), generator=gen).cuda()

    def lm(comm):
        ids = sk.local_chunk_ids(comm.rank, comm.world)
        tiles = w_pad[ids[0] * tile_rows : (ids[-1] + 1) * tile_rows].contiguous()
        hg = h.clone().requires_grad_(True)
        lp = StrictLMHeadLogprob.apply(
            hg, targets, tiles, tuple(ids), comm, vocab, tile_rows, 0.19611613513818404, 20.0
        )
        (dh,) = torch.autograd.grad(lp.sum(), (hg,))
        return lp, dh

    lp1, dh1 = lm(comm1)
    lp2, dh2 = lm(comm_w)
    res["lm_logp"] = torch.equal(lp1, lp2)
    res["lm_dh"] = torch.equal(dh1, dh2)
    # column linear with LoRA
    mm, k, n, r = 64, 512, 512, 16
    x, wc = rnd(mm, k), rnd(n, k, scale=0.05)
    a = torch.randn(r, k, generator=gen).cuda() * 0.02
    b = torch.randn(n, r, generator=gen).cuda() * 0.02
    gy = rnd(mm, n)
    n_l = n // world
    sl = slice(rank * n_l, (rank + 1) * n_l)

    def col(comm):
        nl = n // comm.world
        s = slice(comm.rank * nl, (comm.rank + 1) * nl)
        xg = x.clone().requires_grad_(True)
        ag = a.clone().requires_grad_(True)
        bg = b[s].clone().requires_grad_(True)
        y = StrictColumnLinear.apply(xg, ag, bg, comm, n, wc[s])
        return (y, *torch.autograd.grad(y, (xg, ag, bg), gy[:, s]))

    c1, c2 = col(comm1), col(comm_w)
    res["col_y"] = torch.equal(c1[0][:, sl], c2[0])
    res["col_dx"] = torch.equal(c1[1], c2[1])
    res["col_da"] = torch.equal(c1[2], c2[2])
    res["col_db"] = torch.equal(c1[3][sl], c2[3])
    # row linear with LoRA
    k2 = 1024
    xr, wr = rnd(mm, k2), rnd(n, k2, scale=0.05)
    ar = torch.randn(r, k2, generator=gen).cuda() * 0.02
    k_l = k2 // world
    s = slice(rank * k_l, (rank + 1) * k_l)

    def row(comm):
        kl = k2 // comm.world
        ss = slice(comm.rank * kl, (comm.rank + 1) * kl)
        xg = xr[:, ss].clone().requires_grad_(True)
        ag = ar[:, ss].clone().requires_grad_(True)
        bg = b.clone().requires_grad_(True)
        y = StrictRowLinear.apply(xg, wr[:, ss], ag, bg, comm, k2)
        return (y, *torch.autograd.grad(y, (xg, ag, bg), gy))

    r1, r2 = row(comm1), row(comm_w)
    res["row_y"] = torch.equal(r1[0], r2[0])
    res["row_dx"] = torch.equal(r1[1][:, s], r2[1])
    res["row_da"] = torch.equal(r1[2][:, s], r2[2])
    res["row_db"] = torch.equal(r1[3], r2[3])
    # attention block with per-rank heads (TP4: replicated KV head, replica tree on NCCL)
    ops = sk.resolve_ops()
    case = _attention_block_case(torch.device("cuda", rank))
    for layer in ("local", "global"):
        o1, g1 = _attention_block(comm1, case, ops, layer)
        o2, g2 = _attention_block(comm_w, case, ops, layer)
        res[f"attn_{layer}_o"] = torch.equal(o1, o2)
        res[f"attn_{layer}_dx"] = torch.equal(g1["dx"], g2["dx"])
        for name in ("q_proj", "gate_proj", "k_proj", "v_proj"):
            nl = g1[f"{name}.B"].shape[0] // world
            res[f"attn_{layer}_{name}_A"] = torch.equal(g1[f"{name}.A"], g2[f"{name}.A"])
            res[f"attn_{layer}_{name}_B"] = torch.equal(
                g1[f"{name}.B"][rank * nl : (rank + 1) * nl], g2[f"{name}.B"]
            )
        kl = 4096 // world
        res[f"attn_{layer}_o_proj_A"] = torch.equal(
            g1["o_proj.A"][:, rank * kl : (rank + 1) * kl], g2["o_proj.A"]
        )
        res[f"attn_{layer}_o_proj_B"] = torch.equal(g1["o_proj.B"], g2["o_proj.B"])
    torch.save(res, f"{out_dir}/rank{rank}.pt")
    dist.destroy_process_group()


@pytest.mark.parametrize("world", [2, 4])
def test_nccl_tp_equals_tp1_lm_head_column_row_attention(world, tmp_path):
    if not CUDA or torch.cuda.device_count() < world:
        pytest.skip(f"needs {world} GPUs")
    import socket

    import torch.multiprocessing as mp

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    mp.spawn(_nccl_tp_worker, args=(world, port, str(tmp_path)), nprocs=world, join=True)
    for rank in range(world):
        res = torch.load(tmp_path / f"rank{rank}.pt")
        assert all(res.values()), {k: v for k, v in res.items() if not v}


# --------------------------------------------------------------------------------------
# Fail-closed resolution (review megatron_runtime:R1)
# --------------------------------------------------------------------------------------


@pytest.fixture
def fresh_resolution():
    sk.reset_resolution()
    yield
    sk.reset_resolution()


def test_resolve_ops_fails_closed_when_a_strict_module_is_missing(monkeypatch, fresh_resolution):
    real = sk._optional_module

    def fake(path):
        if path.endswith("ops.norms"):
            sk._IMPORT_ERRORS[path] = "ImportError: simulated broken sibling"
            return None
        return real(path)

    monkeypatch.setattr(sk, "_optional_module", fake)
    with pytest.raises(sk.StrictOpsUnavailable, match="centered_rmsnorm.*simulated broken"):
        sk.resolve_ops()
    prov = sk.resolve_ops(allow_fallback=True).provenance()  # tests only: labelled fallbacks
    assert prov["all_strict"] is False
    assert prov["centered_rmsnorm"].startswith("fallback:")
    from rl_engine.models.muse_glimmer.runtime.megatron_strict import (
        install_museglimmer_megatron_strict,
    )

    # the binding (and so StrictTrainer(strict=True)) refuses before touching the model
    with pytest.raises(sk.StrictOpsUnavailable):
        install_museglimmer_megatron_strict(object(), hf_path=None)


def test_strict_mode_env_values_are_validated(monkeypatch, fresh_resolution):
    monkeypatch.setenv("MUSE_STRICT_ATTN", "off")
    with pytest.raises(ValueError, match="MUSE_STRICT_ATTN"):
        sk.resolve_ops()
    sk.reset_resolution()
    monkeypatch.setenv("MUSE_STRICT_ATTN", "fallback")  # explicit fallback is still refused
    with pytest.raises(sk.StrictOpsUnavailable, match="gated_attention"):
        sk.resolve_ops()
    sk.reset_resolution()
    monkeypatch.setenv("MUSE_STRICT_ATTN", "auto")
    monkeypatch.setenv("MUSE_STRICT_GEMM", "bogus")
    with pytest.raises(ValueError, match="MUSE_STRICT_GEMM"):
        sk.resolve_ops()
    sk.reset_resolution()
    monkeypatch.setenv("MUSE_STRICT_GEMM", "cublas")
    with pytest.raises(sk.StrictOpsUnavailable, match="gemm"):
        sk.resolve_ops()


def test_tree_accumulator_equals_canonical_combine_tree():
    from rl_engine.models.muse_glimmer.ops.canonical_tree import combine_tree

    gen = torch.Generator().manual_seed(1)
    parts = [torch.randn(5, 7, generator=gen) * 10.0**i for i in range(8)]
    for n in (1, 2, 4, 8):
        acc = sk.TreeAccumulator()
        for p in parts[:n]:
            acc.push(p)
        assert torch.equal(acc.result(), combine_tree(parts[:n]))
    acc = sk.TreeAccumulator()
    for p in parts[:3]:
        acc.push(p)
    with pytest.raises(ValueError):
        acc.result()


# --------------------------------------------------------------------------------------
# LoRA token-axis gradients: the det_gemm dW rule (kernels_gemm:R1) and packing
# invariance (megatron_runtime:R2); binding dX vs ops autograd (kernels_gemm:R2)
# --------------------------------------------------------------------------------------


@requires_cuda
@pytest.mark.parametrize("t", [1, 200, 300, 513])
def test_lora_token_grad_uses_the_det_gemm_dw_rule(t):
    """T_pad = roundup(T, 256), 8 chunks of T_pad/8 (not the old 512 rule): bitwise the det_gemm
    dW kernel call, and one rounding of it is exactly ops fast_det_gemm's autograd dW."""
    from rl_engine.models.muse_glimmer.ops.det_gemm import fast_det_gemm
    from rl_engine.models.muse_glimmer.runtime.megatron_lora import _token_grad_gemm
    from rl_engine.models.muse_glimmer.triton.det_gemm_tree import tree_gemm

    gen = torch.Generator().manual_seed(t)
    lhs = torch.randn(t, 48, generator=gen).to(device="cuda", dtype=torch.bfloat16)
    rhs = torch.randn(t, 512, generator=gen).to(device="cuda", dtype=torch.bfloat16)
    got = _token_grad_gemm(lhs, rhs)
    assert got.dtype == torch.float32 and got.shape == (48, 512)
    assert torch.equal(got, tree_gemm(lhs.t(), rhs, out_dtype=torch.float32, allow_pad=True))
    w = (torch.randn(48, 512, generator=gen) * 0.05).to(device="cuda", dtype=torch.bfloat16)
    w.requires_grad_(True)
    fast_det_gemm(rhs, w).backward(lhs)  # dW[n, k] = sum_t lhs[t, n] * rhs[t, k]
    assert torch.equal(w.grad, got.to(torch.bfloat16))
    ref = (lhs.double().t() @ rhs.double()).float()
    assert torch.allclose(got, ref, atol=1e-3, rtol=1e-4)


@requires_cuda
@pytest.mark.parametrize("kind", ["column", "row"])
def test_lora_grads_are_packing_invariant(kind):
    """The same three sequences packed as one, two or three micro-batches give bitwise the same
    LoRA gradients (per-sequence dW partials, left fold in sample order); before the fix the
    token reduction spanned the whole micro-batch, so the bits depended on the packing."""
    lens = [70, 200, 330]
    offs = [0, 70, 270, 600]
    k, n, r = 1024, 256, 16
    gen = torch.Generator().manual_seed(5)
    x = torch.randn(600, k, generator=gen).to(device="cuda", dtype=torch.bfloat16)
    w = (torch.randn(n, k, generator=gen) * 0.05).to(device="cuda", dtype=torch.bfloat16)
    a0 = (torch.randn(r, k, generator=gen) * 0.02).cuda()
    b0 = (torch.randn(n, r, generator=gen) * 0.02).cuda()
    gy = torch.randn(600, n, generator=gen).to(device="cuda", dtype=torch.bfloat16)
    comm = sk.TPComm(None)

    def run(packing):
        a = a0.clone().requires_grad_(True)
        b = b0.clone().requires_grad_(True)
        for group in packing:
            rows = torch.cat([torch.arange(offs[i], offs[i + 1]) for i in group]).cuda()
            segs, pos = [], 0
            for i in group:
                segs.append((pos, pos + lens[i]))
                pos += lens[i]
            xs = x.index_select(0, rows)
            if kind == "column":
                y = StrictColumnLinear.apply(xs, a, b, comm, n, w, segments=segs)
            else:
                y = StrictRowLinear.apply(xs, w, a, b, comm, k, segments=segs)
            (y.float() * gy.index_select(0, rows).float()).sum().backward()
        return a.grad.clone(), b.grad.clone()

    ref = run([[0, 1, 2]])
    for packing in ([[0, 1], [2]], [[0], [1, 2]], [[0], [1], [2]]):
        got = run(packing)
        assert torch.equal(got[0], ref[0]) and torch.equal(got[1], ref[1]), packing
    singles = [run([[i]]) for i in range(3)]
    fold_a = (singles[0][0] + singles[1][0]) + singles[2][0]
    fold_b = (singles[0][1] + singles[1][1]) + singles[2][1]
    assert torch.equal(ref[0], fold_a) and torch.equal(ref[1], fold_b)


@requires_cuda
@pytest.mark.parametrize("m", [1, 64, 300])
@pytest.mark.parametrize(
    "name,n,k", [("k_proj", 256, 6656), ("q_proj", 4096, 6656), ("o_proj", 6656, 4096)]
)
def test_binding_dx_equals_ops_fast_det_gemm_autograd(m, name, n, k):
    """The binding's own StrictColumnLinear / StrictRowLinear input gradient (single-chunk
    leaves + canonical tree, simulated TP 1/2/4/8 with a list-returning gather) equals the ops
    ``fast_det_gemm`` autograd dA bitwise (no LoRA)."""
    from rl_engine.models.muse_glimmer.ops.det_gemm import fast_det_gemm

    gen = torch.Generator().manual_seed(m + n)
    x = torch.randn(m, k, generator=gen).to(device="cuda", dtype=torch.bfloat16)
    w = (torch.randn(n, k, generator=gen) * 0.02).to(device="cuda", dtype=torch.bfloat16)
    dy = torch.randn(m, n, generator=gen).to(device="cuda", dtype=torch.bfloat16)
    xr = x.clone().requires_grad_(True)
    fast_det_gemm(xr, w).backward(dy)
    want = xr.grad
    for world in (1, 2, 4, 8):
        if name == "o_proj":  # row-parallel: K split, dX_local = the rank's columns of dA

            def run(comm):
                kl = k // comm.world
                cols = slice(comm.rank * kl, (comm.rank + 1) * kl)
                xg = x[:, cols].clone().requires_grad_(True)
                y = StrictRowLinear.apply(xg, w[:, cols], None, None, comm, k)
                return torch.autograd.grad(y, (xg,), dy)[0]

            outs = _simulate(world, run)
            assert torch.equal(torch.cat(outs, dim=1), want), (name, m, world)
        else:  # column-parallel: N split, dX on every rank = the full dA

            def run(comm):
                nl = n // comm.world
                rows = slice(comm.rank * nl, (comm.rank + 1) * nl)
                xg = x.clone().requires_grad_(True)
                y = StrictColumnLinear.apply(xg, None, None, comm, n, w[rows])
                return torch.autograd.grad(y, (xg,), dy[:, rows])[0]

            for got in _simulate(world, run):
                assert torch.equal(got, want), (name, m, world)


# --------------------------------------------------------------------------------------
# StrictTrainer recompute check (review megatron_runtime:R3) and vision memory (R7)
# --------------------------------------------------------------------------------------


def test_recompute_mismatch_fails_closed_in_strict_mode():
    from rl_engine.models.muse_glimmer.runtime.strict_trainer import (
        StrictRecomputeMismatch,
        _RankState,
    )

    lora = torch.nn.Linear(2, 1, bias=False)
    state = _RankState.__new__(_RankState)  # no model: only the backward protocol is exercised
    state.device = torch.device("cpu")
    state.binding = None
    state.native_lora = [lora]
    stored = torch.tensor([-1.0, -2.0])

    def pending():
        return {"token_ids": [[1, 2, 3]], "starts": [1], "groups": [[0]], "values": [stored]}

    def recompute_differs(_ids, _starts, _groups, *, grad):
        leaf = torch.tensor([-1.0, -2.5], requires_grad=grad)
        return [leaf * 1.0]

    state._values = recompute_differs
    for strict in (True, False):
        state.strict = strict
        state._pending = pending()
        lora.weight.grad = torch.ones_like(lora.weight)  # an earlier micro-batch's gradient
        if strict:
            with pytest.raises(StrictRecomputeMismatch, match="1 of 2"):
                state.cmd_backward(torch.ones(2))
            assert lora.weight.grad is None  # the whole step's gradients are discarded
        else:  # native mode makes no bitwise promise: counted, not raised
            info = state.cmd_backward(torch.ones(2))
            assert info["recompute_logp_mismatches"] == 1
            assert lora.weight.grad is not None


@requires_cuda
def test_drop_vision_modules_releases_memory_held_by_setattr_cycles():
    """Bridge's ``hook_hf_module_setattr_for_tp_grad_sync`` leaves a bound ``__setattr__`` in
    each vision submodule (a reference cycle), so ``delattr`` alone freed nothing (review R7)."""
    import gc
    import types

    from rl_engine.models.muse_glimmer.runtime.megatron_strict import drop_vision_modules

    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.vision_tower = torch.nn.Sequential(
                torch.nn.Linear(1024, 1024, device="cuda"),
                torch.nn.Linear(1024, 1024, device="cuda"),
            )
            self.decoder = torch.nn.Linear(4, 4, device="cuda")

    model = Toy()
    for sub in model.vision_tower.modules():
        original = sub.__setattr__

        def wrapped(self, name, value, _orig=original):
            _orig(name, value)

        sub.__setattr__ = types.MethodType(wrapped, sub)
    gc.collect()
    before = torch.cuda.memory_allocated()
    info = drop_vision_modules(model)
    expected = 2 * (1024 * 1024 + 1024) * 4
    assert info["dropped"] == ["vision_tower"] and not hasattr(model, "vision_tower")
    assert abs(info["vision_param_gib"] * 2**30 - expected) < 1
    assert before - torch.cuda.memory_allocated() >= expected
    assert info["allocated_drop_gib"] * 2**30 >= expected


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
    strict = os.environ.get("MUSEGLIMMER_TRAIN_STRICT", "1") != "0"
    ckpt = os.path.expanduser(os.environ.get("MUSEGLIMMER_CHECKPOINT", "~/models/Muse-Glimmer-30B"))
    xrt = Path(__file__).resolve().parents[3] / "reports" / "muse_glimmer_poc" / "data" / "xrt_tokens.json"
    data = json.loads(xrt.read_text())["sequences"][:3]
    seqs = [s["token_ids"] for s in data]
    starts = [s["completion_start"] for s in data]
    for name in ("token_logprobs", "step", "export_lora", "provenance", "close"):
        assert callable(getattr(StrictTrainer, name)) and hasattr(TrainerLike, name)
    trainer = StrictTrainer(ckpt, tensor_parallel_size=tp, gpu_ids=list(range(tp)), lora_rank=16,
                            lr=1e-4, strict=strict, seed=1234)
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
        print("step", info)
        if strict:  # the native path makes no bitwise promise (recorded, not asserted)
            assert info["recompute_logp_mismatches"] == 0
        assert info["recompute_logp_checked"] == sum(lp.numel() for lp in lps)
        assert info["grad_norm"] > 0 and info["step"] == 1
        pkg1 = sync.export_lora_verified(trainer, step=1)
        assert pkg1.sha256 != pkg0.sha256
        assert trainer.provenance()["lora_sha256"] == pkg1.sha256
        # any scalar loss: only the first sequence, non-linear in logp
        lps2 = trainer.token_logprobs(seqs, starts)
        info2 = trainer.step(-(lps2[0].exp().mean()))
        print("step2", info2)
        assert info2["step"] == 2 and (not strict or info2["recompute_logp_mismatches"] == 0)
    finally:
        trainer.close()
