"""1-GPU smoke of the sibling Triton backward through the CP adapter (fallback forward)."""
import sys
import time

sys.path.insert(0, "/home/ubuntu/RL-Kernel/.claude/worktrees/agent-a215e705260dd6ffe")
import torch  # noqa: E402

from rl_engine.models.muse_glimmer.runtime import cp_attention as cpa  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
dev = torch.device("cuda", 0)
print("BACKWARD_IMPL:", cpa.BACKWARD_IMPL)
assert cpa._strict_bwd_partials is not None

T, window = 2048, 2048
g = torch.Generator(device="cpu").manual_seed(1)
q = torch.randn(T, 32, 128, generator=g).to(dev, torch.bfloat16)
k = torch.randn(T, 2, 128, generator=g).to(dev, torch.bfloat16)
v = torch.randn(T, 2, 128, generator=g).to(dev, torch.bfloat16)
gate = torch.randn(T, 4096, generator=g).to(dev, torch.bfloat16)
dy = torch.randn(T, 4096, generator=g).to(dev, torch.bfloat16)
pos = torch.arange(T, device=dev, dtype=torch.int32)

plan1 = cpa.CPAttentionPlan(1, 0)
y, o, lse = cpa.cp_gated_attention_forward(q, k, v, gate, pos, plan1, None, window=window, return_aux=True)
torch.cuda.synchronize()
t0 = time.perf_counter()
dq, dk, dv, dg = cpa.cp_gated_attention_backward(q, k, v, gate, o, lse, dy, pos, plan1, None, window=window)
torch.cuda.synchronize()
print(f"CP1 canonical backward (Triton partials) T={T}: {1e3 * (time.perf_counter() - t0):.0f} ms")
dq2, dk2, dv2, dg2 = cpa.cp_gated_attention_backward(q, k, v, gate, o, lse, dy, pos, plan1, None, window=window)
print("repeat equal:", torch.equal(dq, dq2), torch.equal(dk, dk2), torch.equal(dv, dv2), torch.equal(dg, dg2))

# Against the torch fallback partials path (same canonical tree), tolerance only.
cu = torch.tensor([0, T], dtype=torch.int32, device=dev)
fq, fkp, fvp, fg = cpa.fallback_gated_attention_bwd_partials(q, k, v, gate, o, lse, dy, cu, cu, pos, window=window)
fk = cpa.combine_tree(list(fkp.unbind(0))).to(torch.bfloat16)
fv = cpa.combine_tree(list(fvp.unbind(0))).to(torch.bfloat16)


def rel(a, b):
    return ((a.float() - b.float()).abs().max() / b.float().abs().max()).item()


print(f"triton-vs-torch-fallback rel: dq={rel(dq, fq):.2e} dk={rel(dk, fk):.2e} dv={rel(dv, fv):.2e} dgate={rel(dg, fg):.2e}")

# Emulated CP2 in-process: rank chunks via the plan, roots combined by combine_tree (no NCCL).
chunk = T // 8
roots_k = []
for c in range(8):
    rows = slice(c * chunk, (c + 1) * chunk)
    _, dkp_c, _, _ = cpa._kernel_backward_partials(q[rows], k, v, gate[rows], o[rows], lse[rows], dy[rows], pos[rows], window=window)
    roots_k.append(cpa.combine_tree(list(dkp_c.unbind(0))))
cp1 = cpa.combine_tree(roots_k)
cp2 = cpa.combine_tree([cpa.combine_tree(roots_k[:4]), cpa.combine_tree(roots_k[4:])])
cp4 = cpa.combine_tree([cpa.combine_tree(roots_k[i : i + 2]) for i in range(0, 8, 2)])
print("canonical tree with Triton partials CP1==CP2:", torch.equal(cp1, cp2), "CP1==CP4:", torch.equal(cp1, cp4))
print("tree root == cp_gated_attention_backward dk (bf16):", torch.equal(cp1.to(torch.bfloat16), dk))
# One-shot sequential (kernel over all queries) vs canonical chunk tree: expected to differ in last bits.
_, dkp_full, _, _ = cpa._kernel_backward_partials(q, k, v, gate, o, lse, dy, pos, window=window)
seq = cpa.combine_tree(list(dkp_full.unbind(0)))
print("one-shot sequential == chunk tree:", torch.equal(seq, cp1), "rel diff", rel(seq, cp1))
