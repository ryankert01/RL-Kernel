from pathlib import Path
import sys

p = Path(sys.argv[1])
s = p.read_text()
blocks = Path(sys.argv[2]).read_text().split("# ===SPLIT===\n")
new_attention_tests, new_nccl, new_misc = blocks


def rep(old, new, count=1):
    global s
    assert s.count(old) == count, (old[:90], s.count(old))
    s = s.replace(old, new)


rep('''"""Unit tests for the Megatron Bridge strict binding (fused-weight slicing, strict LoRA linears
with simulated TP, LM-head tile composition, LoRA export). No Bridge model is needed; the
30B runs live in ``run_megatron_strict.py``.
"""''', '''"""Unit tests for the Megatron Bridge strict binding (fused-weight slicing, strict LoRA linears
with simulated TP, per-rank attention heads, LM-head tile composition, LoRA export, fail-closed
op resolution, the packing-invariant LoRA gradient rule, the recompute check). No Bridge model
is needed; the 30B runs live in ``run_megatron_strict.py``.
"""''')

rep('''from rl_engine.models.muse_glimmer.runtime.megatron_strict import (
    GatherHeads,
    SliceLocalHeads,
    StrictLMHeadLogprob,
    fused_qkv_blocks,
    fused_qkv_to_hf,
)''', '''from rl_engine.models.muse_glimmer.runtime.megatron_attention import (
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
)''')

# replace the head gather test
start = s.index("@requires_cuda\ndef test_gather_and_slice_heads_roundtrip_grads():")
end = s.index("# --------------------------------------------------------------------------------------\n# LoRA container")
s = s[:start] + '''def test_rank_head_geometry_matches_the_contract_layout():
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


''' + s[end:]

# non-tautological hash test
rep('''    state, sha = export_lora(layers, sk.TPComm(None))
    summary = sync.validate_lora_state_dict(state)
    assert summary["num_modules"] == 2 * len(PROJECTIONS) and summary["ranks"] == [4]
    assert sha == sync.lora_sha256(state) == lora_sha256(state)
    assert state["layers.1.mlp.down_proj.lora_A"].shape == (4, 48)
    assert state["layers.0.q_proj.lora_B"].shape == (32, 4)
''', '''    state, sha = export_lora(layers, sk.TPComm(None))
    summary = sync.validate_lora_state_dict(state)
    assert summary["num_modules"] == 2 * len(PROJECTIONS) and summary["ranks"] == [4]
    assert len(sha) == 64 and sha == lora_sha256(state)
    assert state["layers.1.mlp.down_proj.lora_A"].shape == (4, 48)
    assert state["layers.0.q_proj.lora_B"].shape == (32, 4)


def test_lora_hash_matches_the_pinned_muse_glimmer_lora_sha256_v1_stream():
    """A fixed state hashes to a pinned digest, recomputed here from the documented byte stream
    (sorted keys; ``name|dtype|shape\\\\n`` then the little-endian bf16 bytes) without calling the
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
        digest.update(f"{name}|{t.dtype}|{','.join(str(d) for d in t.shape)}\\n".encode())
        digest.update(t.contiguous().view(torch.int16).numpy().astype("<i2").tobytes())
    assert digest.hexdigest() == pinned
    assert lora_sha256(state) == pinned
''')

# attention tests: replace composed + block tests
start = s.index("@requires_cuda\ndef test_composed_strict_attention_matches_sibling_kernels():")
end = s.index("# --------------------------------------------------------------------------------------\n# Real NCCL TP2 (2 GPUs)")
s = s[:start] + new_attention_tests + s[end:]

# NCCL section: replace whole tail up to TrainerLike section
start = s.index("# --------------------------------------------------------------------------------------\n# Real NCCL TP2 (2 GPUs)")
end = s.index("# --------------------------------------------------------------------------------------\n# TrainerLike protocol")
s = s[:start] + new_nccl + new_misc + s[end:]
p.write_text(s)
print("ok")
