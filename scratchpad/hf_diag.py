import sys

sys.path.insert(0, "/home/ubuntu/RL-Kernel/.claude/worktrees/agent-a1476a61a3b149700")
import torch  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
from rl_engine.models.muse_glimmer.ops import qknorm_rope as m  # noqa: E402
from tests.models.muse_glimmer.test_qknorm_rope import _hf_path, _make_qk  # noqa: E402

DEV = torch.device("cuda")
T = 4096
for scale in (0.05, 1.0, 2.0, 20.0):
    q, k = _make_qk(T, seed=51, scale=scale)
    pos = torch.arange(T, device=DEV)
    cos, sin = m.rope_table(pos, device=DEV)
    c64, s64 = m.rope_table_fp64(pos, device=DEV)
    oq, ok = m.qknorm_scaled_rope(q, k, cos, sin)
    hq, hk = _hf_path(q, k, pos)
    tq, tk = m.qknorm_scaled_rope_fp64(q, k, c64, s64)
    # HF with fp32 cos/sin but bf16 q (isolate table-cast effect)
    print(f"=== input scale {scale} ===")
    for name, ours, hf, truth in (("q", oq, hq, tq), ("k", ok, hk, tk)):
        d = (ours.double() - hf.double()).abs()
        tol = 0.02 + 0.016 * hf.double().abs()
        ratio = d / tol
        eo = (ours.double() - truth).abs()
        eh = (hf.double() - truth).abs()
        i = ratio.argmax()
        j = d.argmax()
        print(
            f"{name}: max_abs={d.max():.4f} mean_abs={d.mean():.3e} frac_neq={(d>0).double().mean():.4f} "
            f"frac_out_of_tol={(ratio>1).double().mean():.3e} n_out={(ratio>1).sum().item()} worst_ratio={ratio.max():.2f}"
        )
        print(
            f"   worst-ratio elem: ours={ours.flatten()[i].item():.4f} hf={hf.flatten()[i].item():.4f} truth={truth.flatten()[i].item():.4f}"
        )
        print(
            f"   max-abs elem: ours={ours.flatten()[j].item():.4f} hf={hf.flatten()[j].item():.4f} truth={truth.flatten()[j].item():.4f}"
        )
        print(
            f"   err vs fp64 truth: ours max={eo.max():.4e} mean={eo.mean():.4e} rms={eo.pow(2).mean().sqrt():.4e} | "
            f"HF max={eh.max():.4e} mean={eh.mean():.4e} rms={eh.pow(2).mean().sqrt():.4e}"
        )
        # relative to the row rms (|x_hat| scale)
        print(f"   ours max err / scale-of-output: {eo.max()/truth.abs().max():.3e}; HF: {eh.max()/truth.abs().max():.3e}")
