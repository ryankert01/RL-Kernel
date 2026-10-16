"""Which bf16 cast dominates the dK error? Torch emulation of the FA2-style strict backward."""
import sys

import torch

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
sys.path.insert(0, "/tmp/claude-1000/-home-ubuntu-RL-Kernel--claude-worktrees-swift-oak-hsv7/80655dc5-1eae-4992-a93b-f6f87e8ab1d8/scratchpad")
from bwd_smoke import make  # noqa: E402

from rl_engine.models.muse_glimmer.ops import attention_golden_bwd as gold  # noqa: E402

scale = 128 ** -0.5


def emulate(q, k, v, gate, dy, o_bf, lse, window, *, cast_ds, cast_do, d_from_bf16_o, cast_p):
    # single sequence, all heads
    L = q.shape[0]
    kv_idx = (torch.arange(32) // 16).to(q.device)
    qf = q.float(); kf = k.float().index_select(1, kv_idx); vf = v.float().index_select(1, kv_idx)
    g = gate.float(); sg = torch.sigmoid(g)
    do = (dy.float() * sg)
    if cast_do:
        do = do.bfloat16().float()
    do = do.view(L, 32, 128)
    s = torch.einsum("ihd,jhd->hij", qf, kf) * scale
    pos = torch.arange(L, device=q.device)
    vis = pos[None, :] <= pos[:, None]
    if window is not None:
        vis &= (pos[:, None] - window) < pos[None, :]
    s = torch.where(vis[None], s, float("-inf"))
    p = torch.exp(s - lse.t()[:, :, None])  # (h, i, j)
    if d_from_bf16_o:
        D = (do * o_bf.float().view(L, 32, 128)).sum(-1).t()  # (h, i)
    else:
        o_exact = torch.einsum("hij,jhd->hid", p, vf)
        D = (do.permute(1, 0, 2) * o_exact).sum(-1)
    dP = torch.einsum("ihd,jhd->hij", do, vf)
    dS = p * (dP - D[:, :, None])
    if cast_ds:
        dS = dS.bfloat16().float()
    pp = p.bfloat16().float() if cast_p else p
    dV_h = torch.einsum("hij,ihd->hjd", pp, do)  # per head
    dK_h = torch.einsum("hij,ihd->hjd", dS, qf) * scale
    dQ = torch.einsum("hij,jhd->ihd", dS, kf) * scale
    dK = dK_h.view(2, 16, L, 128).sum(1).permute(1, 0, 2)
    dV = dV_h.view(2, 16, L, 128).sum(1).permute(1, 0, 2)
    return dQ, dK, dV


def main():
    L = int(sys.argv[1]) if len(sys.argv) > 1 else 512
    window = None
    q, k, v, gate, dy, cu_q, cu_k, pos = make([L])
    y, o, lse = gold.golden_forward_fixed_order(q, k, v, gate, cu_q, cu_k, pos, window=window)
    rq, rk, rv, rg = gold.reference_backward(q, k, v, gate, cu_q, cu_k, pos, dy, window=window)
    configs = {
        "all casts (FA2 default)": dict(cast_ds=True, cast_do=True, d_from_bf16_o=True, cast_p=True),
        "no dS cast": dict(cast_ds=False, cast_do=True, d_from_bf16_o=True, cast_p=True),
        "no do cast": dict(cast_ds=True, cast_do=False, d_from_bf16_o=True, cast_p=True),
        "D from exact o": dict(cast_ds=True, cast_do=True, d_from_bf16_o=False, cast_p=True),
        "no p cast": dict(cast_ds=True, cast_do=True, d_from_bf16_o=True, cast_p=False),
        "no casts at all": dict(cast_ds=False, cast_do=False, d_from_bf16_o=False, cast_p=False),
    }
    print(f"L={L} max|dq|={rq.abs().max():.2f} max|dk|={rk.abs().max():.2f} max|dv|={rv.abs().max():.2f}")
    for name, cfg in configs.items():
        dQ, dK, dV = emulate(q, k, v, gate, dy, o, lse, window, **cfg)
        eq = (dQ.double() - rq).abs().max().item()
        ek = (dK.double() - rk).abs().max().item()
        ev = (dV.double() - rv).abs().max().item()
        # also after bf16 output rounding
        ekb = (dK.bfloat16().double() - rk).abs().max().item()
        print(f"{name:28s} dq err {eq:.3e}  dk err {ek:.3e} (bf16 out {ekb:.3e})  dv err {ev:.3e}")
    # scale of dy effect
    for sc in (0.1, 0.01):
        rq2, rk2, rv2, _ = gold.reference_backward(q, k, v, gate, cu_q, cu_k, pos, dy * sc, window=window)
        dQ, dK, dV = emulate(q, k, v, gate, (dy.float() * sc).bfloat16(), o, lse, window, **configs["all casts (FA2 default)"])
        print(f"dy*{sc}: max|dk|={rk2.abs().max():.3f} dk err {(dK.double()-rk2).abs().max():.3e}  dv err {(dV.double()-rv2).abs().max():.3e}")


main()
