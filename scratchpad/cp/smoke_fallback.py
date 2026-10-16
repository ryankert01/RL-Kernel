"""1-GPU smoke: fallback kernel vs naive fp32 reference; determinism; sequence-split invariance."""
import sys
import time

sys.path.insert(0, "/home/ubuntu/RL-Kernel/.claude/worktrees/agent-a215e705260dd6ffe")
import torch  # noqa: E402

from rl_engine.models.muse_glimmer.runtime import cp_attention as cpa  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
dev = torch.device("cuda", 0)


def inputs(T, seed=0, hq=32, hkv=2):
    g = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.randn(T, hq, 128, generator=g).to(dev, torch.bfloat16)
    k = torch.randn(T, hkv, 128, generator=g).to(dev, torch.bfloat16)
    v = torch.randn(T, hkv, 128, generator=g).to(dev, torch.bfloat16)
    gate = torch.randn(T, hq * 128, generator=g).to(dev, torch.bfloat16)
    return q, k, v, gate


def naive(q, k, v, gate, pos, window):
    T, hq, d = q.shape
    hkv = k.shape[1]
    G = hq // hkv
    kf = k.float().repeat_interleave(G, dim=1)
    vf = v.float().repeat_interleave(G, dim=1)
    s = torch.einsum("qhd,khd->hqk", q.float(), kf) * d**-0.5
    kp = torch.arange(k.shape[0], device=q.device)
    keep = kp[None, :] <= pos[:, None]
    if window is not None:
        keep &= kp[None, :] > pos[:, None] - window
    s = s.masked_fill(~keep[None], float("-inf"))
    p = torch.softmax(s, -1)
    o = torch.einsum("hqk,khd->qhd", p, vf)
    y = o.to(torch.bfloat16).float() * torch.sigmoid(gate.float().view(T, hq, d))
    return y.reshape(T, hq * d), o.reshape(T, hq * d), torch.logsumexp(s, -1).t()


for T, window in ((2048, None), (2048, 2048), (4096, 2048), (512, 128)):
    q, k, v, gate = inputs(T)
    pos = torch.arange(T, device=dev, dtype=torch.int32)
    cu_q = torch.tensor([0, T], dtype=torch.int32, device=dev)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    y, o, lse = cpa.fallback_gated_attention_fwd(q, k, v, gate, cu_q, cu_q, pos, window=window)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    y2, o2, lse2 = cpa.fallback_gated_attention_fwd(q, k, v, gate, cu_q, cu_q, pos, window=window)
    yn, on, lsen = naive(q, k, v, gate, pos, window)
    err_y = (y.float() - yn).abs().max().item()
    err_o = (o.float() - on).abs().max().item()
    err_l = (lse - lsen).abs().max().item()
    half = T // 2
    cu_h = torch.tensor([0, half], dtype=torch.int32, device=dev)
    yh, oh, lseh = cpa.fallback_gated_attention_fwd(
        q[half:], k, v, gate[half:], cu_h, cu_q, pos[half:], window=window
    )
    off = half + 192
    cu_o = torch.tensor([0, T - off], dtype=torch.int32, device=dev)
    yo, oo, lseo = cpa.fallback_gated_attention_fwd(
        q[off:], k, v, gate[off:], cu_o, cu_q, pos[off:], window=window
    )
    rep = torch.equal(y, y2) and torch.equal(lse, lse2)
    sp_half = torch.equal(yh, y[half:]) and torch.equal(oh, o[half:]) and torch.equal(lseh, lse[half:])
    sp_off = torch.equal(yo, y[off:]) and torch.equal(lseo, lse[off:])
    print(
        f"T={T} window={window}: fwd {dt*1e3:.0f} ms | max|dy|={err_y:.3e} max|do|={err_o:.3e} "
        f"max|dlse|={err_l:.3e} | repeat={rep} | split[T/2:]={sp_half} | split[T/2+192:]={sp_off}"
    )

T, window = 1024, 512
q, k, v, gate = inputs(T, seed=3)
pos = torch.arange(T, device=dev, dtype=torch.int32)
cu = torch.tensor([0, T], dtype=torch.int32, device=dev)
y, o, lse = cpa.fallback_gated_attention_fwd(q, k, v, gate, cu, cu, pos, window=window)
dy = torch.randn(T, 4096, generator=torch.Generator().manual_seed(9)).to(dev, torch.bfloat16)
torch.cuda.synchronize()
t0 = time.perf_counter()
dq, dkp, dvp, dg = cpa.fallback_gated_attention_bwd_partials(
    q, k, v, gate, o, lse, dy, cu, cu, pos, window=window
)
torch.cuda.synchronize()
print(
    f"bwd T={T}: {1e3*(time.perf_counter()-t0):.0f} ms; dkp shape {tuple(dkp.shape)} "
    "nonzero per group/kv:",
    (dkp.abs().sum(dim=(1, 3)) > 0).int().tolist(),
)
dq2, dkp2, dvp2, dg2 = cpa.fallback_gated_attention_bwd_partials(
    q, k, v, gate, o, lse, dy, cu, cu, pos, window=window
)
print(
    "bwd repeat equal:",
    torch.equal(dq, dq2),
    torch.equal(dkp, dkp2),
    torch.equal(dvp, dvp2),
    torch.equal(dg, dg2),
)
dk = cpa.combine_tree(list(dkp.unbind(0)))
dv = cpa.combine_tree(list(dvp.unbind(0)))
qr = q.float().requires_grad_()
kr = k.float().requires_grad_()
vr = v.float().requires_grad_()
gr = gate.float().requires_grad_()
G = 16
kf = kr.repeat_interleave(G, 1)
vf = vr.repeat_interleave(G, 1)
s = torch.einsum("qhd,khd->hqk", qr, kf) * 128**-0.5
kp = torch.arange(T, device=dev)
keep = (kp[None] <= pos[:, None]) & (kp[None] > pos[:, None] - window)
s = s.masked_fill(~keep[None], float("-inf"))
p = torch.softmax(s, -1)
oref = torch.einsum("hqk,khd->qhd", p, vf).reshape(T, 4096)
yref = oref * torch.sigmoid(gr)
yref.backward(dy.float())


def rel(a, b):
    return ((a.float() - b).abs().max() / b.abs().max()).item()


print(
    f"bwd vs naive autograd: rel dq={rel(dq, qr.grad.view(T, 32, 128)):.3e} "
    f"dk={rel(dk, kr.grad):.3e} dv={rel(dv, vr.grad):.3e} dgate={rel(dg, gr.grad):.3e}"
)
# Canonical-chunk structure at CP1 vs "CP2" emulated in-process: chunk partials and tree.
chunk = T // 8
roots = []
for c in range(8):
    rows = slice(c * chunk, (c + 1) * chunk)
    cu_c = torch.tensor([0, chunk], dtype=torch.int32, device=dev)
    _, dkp_c, _, _ = cpa.fallback_gated_attention_bwd_partials(
        q[rows], k, v, gate[rows], o[rows], lse[rows], dy[rows], cu_c, cu, pos[rows], window=window
    )
    roots.append(cpa.combine_tree(list(dkp_c.unbind(0))))
cp1 = cpa.combine_tree(roots)
cp2 = cpa.combine_tree([cpa.combine_tree(roots[:4]), cpa.combine_tree(roots[4:])])
cp4 = cpa.combine_tree([cpa.combine_tree(roots[i : i + 2]) for i in range(0, 8, 2)])
print("canonical tree CP1==CP2:", torch.equal(cp1, cp2), "CP1==CP4:", torch.equal(cp1, cp4))
print("chunked-tree vs one-shot sequential dk rel diff:", rel(cp1, dk))
