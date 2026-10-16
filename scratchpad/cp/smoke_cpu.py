import sys

sys.path.insert(0, "/home/ubuntu/RL-Kernel/.claude/worktrees/agent-a215e705260dd6ffe")
import torch  # noqa: E402

from rl_engine.models.muse_glimmer.runtime import cp_attention as cpa  # noqa: E402

print("forward_impl:", cpa.FORWARD_IMPL)
print("backward_impl:", cpa.BACKWARD_IMPL)
print("tree:", cpa.TREE_IMPL)
p = cpa.CPAttentionPlan(4, 1)
print(
    p.local_token_range(8192),
    p.canonical_chunk_ids(),
    cpa.CPAttentionPlan(2, 1).canonical_chunk_ids(),
    cpa.CPAttentionPlan(1, 0).canonical_chunk_ids(),
)
for layout in ("zigzag", "striped", "ring", "allgather_v2", ""):
    try:
        cpa.CPAttentionPlan(2, 0, layout=layout)
        print("NO RAISE", layout)
    except cpa.CPLayoutError as e:
        print("raised:", layout, "->", str(e)[:70])
try:
    p.validate_local_positions(torch.arange(2048, 4096, dtype=torch.int32), 8192)
    print("ok contiguous")
    p.validate_local_positions(torch.arange(0, 2048, dtype=torch.int32), 8192)
except cpa.CPLayoutError as e:
    print("raised wrong chunk:", str(e)[:60])
zig = torch.cat(
    [torch.arange(0, 256), torch.arange(768, 1024), torch.arange(256, 512), torch.arange(512, 768)]
)
try:
    cpa.CPAttentionPlan(1, 0).validate_local_positions(zig, 1024)
except cpa.CPLayoutError as e:
    print("raised zigzag perm:", str(e)[:60])
t = cpa.TPCPTopology.from_global_rank(3, tp_world=2, cp_world=2)
print(t, t.cp_group_ranks(), t.q_head_range(), t.kv_head_range(), t.gate_channel_range())
print(cpa.trace(p)["cp_local_canonical_chunks"])
