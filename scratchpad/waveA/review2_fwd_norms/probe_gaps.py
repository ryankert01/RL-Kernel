"""Test-gap probes (review2): window off-by-one detectability, paged multi-seq TOKEN tiles,
norms mismatch fraction vs golden."""
import torch
from rl_engine.contracts.numerical import load_contract, resolve_tolerance
from rl_engine.models.muse_glimmer.ops import gated_attention as ga
from rl_engine.models.muse_glimmer.ops import norms as N
from rl_engine.models.muse_glimmer.ops.paged_kv import PagedKVCache

torch.backends.cuda.matmul.allow_tf32 = False
DEV = torch.device("cuda")
HQ, HKV, D, W = 32, 2, 128, 2048
TOL = resolve_tolerance(load_contract(), judgment="forward_accuracy", op_class="attention",
                        dtype=torch.bfloat16)


def inputs(n, seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    q = (torch.randn(n, HQ, D, device=DEV, generator=g) * 3.87).bfloat16()
    k = torch.randn(n, HKV, D, device=DEV, generator=g).bfloat16()
    v = torch.randn(n, HKV, D, device=DEV, generator=g).bfloat16()
    gate = (torch.randn(n, HQ * D, device=DEV, generator=g) * 2).bfloat16()
    return q, k, v, gate


def i32(x):
    return torch.tensor(list(x), dtype=torch.int32, device=DEV)


def run(name, fn):
    try:
        print(f"[{name}] {fn()}")
    except Exception as e:  # noqa: BLE001
        print(f"[{name}] raised {type(e).__name__}: {str(e)[:200]}")


def off_by_one_detectable():
    """Would test_g1 (n=2300, seed 11+n) catch a kernel whose window were 2047 or 2049?

    Compare the REAL kernel (W=2048) against goldens computed with W +- 1: if allclose passes,
    the G1 gate cannot distinguish an off-by-one window.
    """
    n = 2300
    q, k, v, gate = inputs(n, 11 + n)
    pos = torch.arange(n, dtype=torch.int32, device=DEV)
    y, o, lse = ga.strict_gated_attention_fwd_varlen(q, k, v, gate, i32([0, n]), i32([0, n]), pos,
                                                      window=W)
    res = {}
    for wg in (2047, 2048, 2049):
        yg, og, lg = ga.gated_attention_forward_fp32(q, k, v, gate, pos, wg)
        ok_y = torch.allclose(y.float(), yg.float(), atol=TOL.atol, rtol=TOL.rtol)
        ok_o = torch.allclose(o.float(), og.float(), atol=TOL.atol, rtol=TOL.rtol)
        el = (lse - lg).abs().max().item()
        res[wg] = dict(y_allclose=ok_y, o_allclose=ok_o, lse_err=f"{el:.2e}",
                       lse_gate_pass=el < 1e-4,
                       max_dy=f"{(y.float() - yg.float()).abs().max().item():.3e}")
    return res


def paged_multiseq_token_tiles():
    """Mixed paged batch (two prefill chunks + decodes) in one call vs each sequence alone."""
    lens = [300, 2100, 50, 2049]
    newq = [100, 40, 1, 1]  # chunk sizes -> max_seqlen_q=100 -> TOKEN tiles for all
    seqs = [inputs(n, 900 + i) for i, n in enumerate(lens)]
    refs = []
    for (q, k, v, gate), n in zip(seqs, lens):
        y, o, lse = ga.strict_gated_attention_fwd_varlen(
            q, k, v, gate, i32([0, n]), i32([0, n]), torch.arange(n, dtype=torch.int32,
                                                                  device=DEV), window=W)
        refs.append((y, o, lse))
    pages = max(lens) // 16 + 2
    cache = PagedKVCache(num_pages=pages * len(lens), num_seqs=len(lens), max_pages_per_seq=pages,
                         device=DEV, poison_freed=True)
    for i, (q, k, v, gate) in enumerate(seqs):
        cache.append(i, k, v)
        cache.evict_window(i, W, query_pos=lens[i] - newq[i])
    qs = torch.cat([s[0][-m:] for s, m in zip(seqs, newq)])
    gs = torch.cat([s[3][-m:] for s, m in zip(seqs, newq)])
    cu = i32([0, *torch.tensor(newq).cumsum(0).tolist()])
    y, o, lse = ga.strict_gated_attention_fwd_paged(qs, gs, cache.k_cache, cache.v_cache,
                                                    cache.block_table, cache.seq_lens, cu,
                                                    window=W)
    off = 0
    out = []
    for i, m in enumerate(newq):
        ok = all(torch.equal(a[off:off + m], b[-m:]) for a, b in zip((y, o, lse), refs[i]))
        out.append(ok)
        off += m
    return f"per-seq bitwise equal to one-shot varlen (TOKEN tiles, mixed batch): {out}"


def norms_mismatch_fraction():
    H = N.HIDDEN
    g = torch.Generator(device=DEV).manual_seed(10 + 4096)
    x = (torch.randn(4096, H, device=DEV, generator=g) * 2.0).bfloat16()
    g2 = torch.Generator(device=DEV).manual_seed(11)
    w = (torch.randn(H, device=DEV, generator=g2) * 0.1).bfloat16()
    res = {}
    for eps in (N.RMS_NORM_EPS, N.POST_NORM_EPS):
        y = N.centered_rmsnorm(x, w, eps=eps)
        yg = N.centered_rmsnorm_fp32(x, w, eps=eps).bfloat16()
        res[f"centered eps={eps}"] = f"{(y != yg).float().mean().item():.2e}"
    return f"strict vs golden.to(bf16) mismatch fraction: {res}"


if __name__ == "__main__":
    import sys
    for name in sys.argv[1:]:
        run(name, globals()[name])
