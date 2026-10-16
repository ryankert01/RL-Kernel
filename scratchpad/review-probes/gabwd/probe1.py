import inspect
import torch

from rl_engine.models.muse_glimmer.ops import attention_golden_bwd as gold
from rl_engine.models.muse_glimmer.ops import gated_attention as fwd_op
from rl_engine.models.muse_glimmer.ops import gated_attention_autograd as ag
from rl_engine.models.muse_glimmer.triton import gated_attention_bwd as bwd

print("ag.trace forward_is_fallback:", ag.trace()["forward_is_fallback"])
print("ag.FORWARD_IMPL:", ag.FORWARD_IMPL)
print("fwd wrapper sig:", inspect.signature(fwd_op.strict_gated_attention_fwd_varlen))
print("bwd trace combine_tree_impl:", bwd.trace()["combine_tree_impl"])


def make_inputs(lens, *, seed=0, pos0=None, k_lens=None):
    g = torch.Generator(device="cpu").manual_seed(seed)
    tq = sum(lens)
    k_lens = list(lens) if k_lens is None else list(k_lens)
    tk = sum(k_lens)
    q = torch.randn(tq, 32, 128, generator=g)
    q = q * torch.rsqrt(q.pow(2).mean(-1, keepdim=True) + 1e-5) * 3.87
    k = torch.randn(tk, 2, 128, generator=g)
    k = k * torch.rsqrt(k.pow(2).mean(-1, keepdim=True) + 1e-5)
    v = torch.randn(tk, 2, 128, generator=g)
    gate = torch.randn(tq, 4096, generator=g)
    dy = torch.randn(tq, 4096, generator=g)
    cu_q = torch.tensor([0] + torch.tensor(lens).cumsum(0).tolist(), dtype=torch.int32)
    cu_k = torch.tensor([0] + torch.tensor(k_lens).cumsum(0).tolist(), dtype=torch.int32)
    pos0 = [0] * len(lens) if pos0 is None else pos0
    pos = torch.cat([torch.arange(p, p + n, dtype=torch.int32) for p, n in zip(pos0, lens)])
    dev = torch.device("cuda")
    return dict(
        q=q.to(dev, torch.bfloat16), k=k.to(dev, torch.bfloat16), v=v.to(dev, torch.bfloat16),
        gate=gate.to(dev, torch.bfloat16), dy=dy.to(dev, torch.bfloat16),
        cu_q=cu_q.to(dev), cu_k=cu_k.to(dev), pos=pos.to(dev),
    )


def args(i):
    return (i["q"], i["k"], i["v"], i["gate"], i["cu_q"], i["cu_k"], i["pos"])


for window in (None, 2048):
    for lens in ([300, 64], [2100]):
        inp = make_inputs(lens, seed=9)
        y_g, o_g, lse_g = gold.golden_forward_fixed_order(*args(inp), window=window)
        try:
            out = fwd_op.strict_gated_attention_fwd_varlen(*args(inp), window=window)
        except Exception as e:  # noqa: BLE001
            print("strict fwd call with autograd's positional convention FAILED:", repr(e)[:300])
            break
        if isinstance(out, tuple):
            print("strict fwd returns tuple of", [(tuple(t.shape), t.dtype) for t in out])
            y_s, o_s, lse_s = out[:3]
        else:
            print("strict fwd returns", type(out))
            break
        print(f"[window={window} lens={lens}] y eq {torch.equal(y_s, y_g)} o eq {torch.equal(o_s, o_g)} "
              f"lse eq {torch.equal(lse_s, lse_g)} max|lse d|={(lse_s - lse_g).abs().max().item():.3e} "
              f"n_o_diff={(o_s != o_g).sum().item()} n_lse_diff={(lse_s != lse_g).sum().item()}")
        g_g = bwd.strict_gated_attention_bwd(inp["q"], inp["k"], inp["v"], inp["gate"], o_g, lse_g, inp["dy"],
                                             inp["cu_q"], inp["cu_k"], inp["pos"], window=window)
        g_s = bwd.strict_gated_attention_bwd(inp["q"], inp["k"], inp["v"], inp["gate"], o_s, lse_s, inp["dy"],
                                             inp["cu_q"], inp["cu_k"], inp["pos"], window=window)
        for n, a, b in zip(("dq", "dk", "dv", "dgate"), g_g, g_s):
            print(f"   grads(golden o/lse) vs grads(strict o/lse): {n} equal={torch.equal(a, b)} "
                  f"max|d|={(a.float() - b.float()).abs().max().item():.3e} n_diff={(a != b).sum().item()}")

# Query-chunk dependence of the backward: one-shot 0..299 vs chunk queries 200..299 (keys 0..299)
print("\n=== query chunk probe ===")
for window in (None, 2048):
    full = make_inputs([300], seed=9)
    dev = full["q"].device
    chunk = dict(
        q=full["q"][200:].contiguous(), k=full["k"], v=full["v"], gate=full["gate"][200:].contiguous(),
        dy=full["dy"][200:].contiguous(), cu_q=torch.tensor([0, 100], dtype=torch.int32, device=dev),
        cu_k=torch.tensor([0, 300], dtype=torch.int32, device=dev),
        pos=torch.arange(200, 300, dtype=torch.int32, device=dev),
    )
    head = dict(
        q=full["q"][:200].contiguous(), k=full["k"], v=full["v"], gate=full["gate"][:200].contiguous(),
        dy=full["dy"][:200].contiguous(), cu_q=torch.tensor([0, 200], dtype=torch.int32, device=dev),
        cu_k=torch.tensor([0, 300], dtype=torch.int32, device=dev),
        pos=torch.arange(0, 200, dtype=torch.int32, device=dev),
    )
    _, o_f, lse_f = gold.golden_forward_fixed_order(*args(full), window=window)
    _, o_c, lse_c = gold.golden_forward_fixed_order(*args(chunk), window=window)
    _, o_h, lse_h = gold.golden_forward_fixed_order(*args(head), window=window)
    print(f"[window={window}] golden fwd chunk o/lse rows equal: {torch.equal(o_f[200:], o_c)} {torch.equal(lse_f[200:], lse_c)}")
    gf = bwd.strict_gated_attention_bwd(full["q"], full["k"], full["v"], full["gate"], o_f, lse_f, full["dy"],
                                        full["cu_q"], full["cu_k"], full["pos"], window=window)
    gc = bwd.strict_gated_attention_bwd(chunk["q"], chunk["k"], chunk["v"], chunk["gate"], o_c, lse_c, chunk["dy"],
                                        chunk["cu_q"], chunk["cu_k"], chunk["pos"], window=window)
    gh = bwd.strict_gated_attention_bwd(head["q"], head["k"], head["v"], head["gate"], o_h, lse_h, head["dy"],
                                        head["cu_q"], head["cu_k"], head["pos"], window=window)
    print(f"   dq rows 200..299 one-shot vs chunk: equal={torch.equal(gf[0][200:], gc[0])} "
          f"max|d|={(gf[0][200:].float() - gc[0].float()).abs().max().item():.3e}")
    print(f"   dgate rows one-shot vs chunk: equal={torch.equal(gf[3][200:], gc[3])}")
    # dk/dv of the one-shot vs (head chunk + tail chunk) summed in fp32 partials
    _dg, do_f, dl_f = bwd.gated_attention_bwd_preprocess(full["dy"], o_f, full["gate"])
    _, dkp_f, dvp_f = bwd.strict_gated_attention_bwd_partials(full["q"], full["k"], full["v"], do_f, lse_f, dl_f,
                                                              full["cu_q"], full["cu_k"], full["pos"], window=window)
    _dg, do_c, dl_c = bwd.gated_attention_bwd_preprocess(chunk["dy"], o_c, chunk["gate"])
    _, dkp_c, dvp_c = bwd.strict_gated_attention_bwd_partials(chunk["q"], chunk["k"], chunk["v"], do_c, lse_c, dl_c,
                                                              chunk["cu_q"], chunk["cu_k"], chunk["pos"], window=window)
    _dg, do_h, dl_h = bwd.gated_attention_bwd_preprocess(head["dy"], o_h, head["gate"])
    _, dkp_h, dvp_h = bwd.strict_gated_attention_bwd_partials(head["q"], head["k"], head["v"], do_h, lse_h, dl_h,
                                                              head["cu_q"], head["cu_k"], head["pos"], window=window)
    print(f"   dk fp32 partials one-shot vs head+tail chunks: equal={torch.equal(dkp_f, dkp_h + dkp_c)} "
          f"max|d|={(dkp_f - (dkp_h + dkp_c)).abs().max().item():.3e} (max|dk|={dkp_f.abs().max().item():.3e})")
    print(f"   dv fp32 partials one-shot vs head+tail chunks: equal={torch.equal(dvp_f, dvp_h + dvp_c)} "
          f"max|d|={(dvp_f - (dvp_h + dvp_c)).abs().max().item():.3e}")
