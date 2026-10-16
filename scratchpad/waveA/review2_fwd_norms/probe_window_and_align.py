"""Window-boundary semantics (independent of the golden) + pointer-alignment variants (review2)."""
import torch
from rl_engine.models.muse_glimmer.ops import gated_attention as ga
from rl_engine.models.muse_glimmer.ops import norms as N
from rl_engine.models.muse_glimmer.ops.paged_kv import PagedKVCache

torch.backends.cuda.matmul.allow_tf32 = False
DEV = torch.device("cuda")
HQ, HKV, D, W = 32, 2, 128, 2048


def i32(x):
    return torch.tensor(list(x), dtype=torch.int32, device=DEV)


def run(name, fn):
    try:
        print(f"[{name}] {fn()}")
    except Exception as e:  # noqa: BLE001
        print(f"[{name}] raised {type(e).__name__}: {str(e)[:200]}")


def window_semantics():
    """q = 0 -> uniform softmax over the visible keys; v channels flag individual keys."""
    n = 2112
    out = []
    for p in (2046, 2047, 2048, 2049, 2050, 2111):
        q = torch.zeros(n, HQ, D, device=DEV, dtype=torch.bfloat16)
        k = torch.randn(n, HKV, D, device=DEV).bfloat16()
        v = torch.zeros(n, HKV, D, device=DEV, dtype=torch.bfloat16)
        if p - W >= 0:
            v[p - W, :, 0] = 1.0  # just outside the window
        v[p - W + 1 if p - W + 1 >= 0 else 0, :, 1] = 1.0  # first key inside (or key 0)
        v[p, :, 2] = 1.0  # self
        if p + 1 < n:
            v[p + 1, :, 3] = 1.0  # future
        gate = torch.full((n, HQ * D), 30.0, device=DEV, dtype=torch.bfloat16)  # sigmoid ~ 1
        cu = i32([0, n])
        pos = torch.arange(n, dtype=torch.int32, device=DEV)
        res = {}
        for mode in ("token", "head"):
            y, o, lse = ga.strict_gated_attention_fwd_varlen(
                q, k, v, gate, cu, cu, pos, window=W, tile_mode=mode)
            row = o[p].view(HQ, D)[0, :4].float()
            cnt = round(1.0 / row[2].item()) if row[2].item() > 0 else None
            res[mode] = (row[0].item(), cnt, row[3].item(), round(lse[p, 0].item(), 4))
        # paged decode of the same row
        cache = PagedKVCache(num_pages=n // 16 + 4, num_seqs=1, max_pages_per_seq=n // 16 + 1,
                             device=DEV)
        cache.append(0, k[: p + 1], v[: p + 1])
        cache.evict_window(0, W)
        y, o, lse = ga.strict_gated_attention_fwd_paged(
            q[p : p + 1].contiguous(), gate[p : p + 1].contiguous(), cache.k_cache,
            cache.v_cache, cache.block_table, cache.seq_lens, i32([0, 1]), window=W)
        row = o[0].view(HQ, D)[0, :4].float()
        res["paged"] = (row[0].item(), round(1.0 / row[2].item()), row[3].item(),
                        round(lse[0, 0].item(), 4))
        import math
        expect = (0.0, min(p + 1, W), 0.0, round(math.log(min(p + 1, W)), 4))
        out.append((p, expect, res))
    lines = []
    for p, expect, res in out:
        ok = all(r[0] == 0.0 and r[1] == expect[1] and r[2] == 0.0 for r in res.values())
        lines.append(f"p={p}: expect visible={expect[1]} outside=0 future=0 -> "
                     f"{'OK' if ok else 'MISMATCH'} {res}")
    return "\n  " + "\n  ".join(lines)


def dense_fp64_reference():
    """Kernel vs an independent dense fp64 softmax (no blocking, no p rounding) at 2047..2049."""
    n = 2300
    g = torch.Generator(device=DEV).manual_seed(77)
    q = (torch.randn(n, HQ, D, device=DEV, generator=g) * 3.87).bfloat16()
    k = torch.randn(n, HKV, D, device=DEV, generator=g).bfloat16()
    v = torch.randn(n, HKV, D, device=DEV, generator=g).bfloat16()
    gate = (torch.randn(n, HQ * D, device=DEV, generator=g) * 2).bfloat16()
    cu = i32([0, n])
    pos = torch.arange(n, dtype=torch.int32, device=DEV)
    out = []
    for window in (None, W):
        y, o, lse = ga.strict_gated_attention_fwd_varlen(q, k, v, gate, cu, cu, pos,
                                                          window=window)
        rows = torch.tensor([0, 63, 64, 2046, 2047, 2048, 2049, 2050, 2299], device=DEV)
        qd = q[rows].double()  # (r, 32, 128)
        kd = k.double()[:, torch.arange(HQ, device=DEV) // 16]  # (n, 32, 128)
        vd = v.double()[:, torch.arange(HQ, device=DEV) // 16]
        s = torch.einsum("rhd,nhd->rhn", qd, kd) * (128 ** -0.5)
        kp = torch.arange(n, device=DEV)
        keep = kp[None, :] <= rows[:, None]
        if window is not None:
            keep &= kp[None, :] > rows[:, None] - window
        s = s.masked_fill(~keep[:, None, :], float("-inf"))
        lse64 = torch.logsumexp(s, dim=-1)
        pr = torch.softmax(s, dim=-1)
        od = torch.einsum("rhn,nhd->rhd", pr, vd).reshape(len(rows), -1)
        yd = od * torch.sigmoid(gate[rows].double())
        out.append((window, f"max|o-o64|={(o[rows].double() - od).abs().max().item():.3e}",
                    f"max|y-y64|={(y[rows].double() - yd).abs().max().item():.3e}",
                    f"max|lse-lse64|={(lse[rows].double() - lse64).abs().max().item():.3e}"))
    return out


def norms_alignment():
    """Same logical rows through an aligned tensor vs a contiguous view at a 2-byte offset."""
    H = 6656
    T = 64
    g = torch.Generator(device=DEV).manual_seed(5)
    x = (torch.randn(T, H, device=DEV, generator=g) * 2).bfloat16()
    w = (torch.randn(H, device=DEV, generator=g) * 0.1).bfloat16()
    buf = torch.empty(T * H + 8, device=DEV, dtype=torch.bfloat16)
    res = {}
    for off in (1, 2, 4, 8):
        xv = buf[off : off + T * H].view(T, H)
        xv.copy_(x)
        assert xv.is_contiguous() and torch.equal(xv, x)
        a = N.centered_rmsnorm(x, w, eps=1e-5)
        b = N.centered_rmsnorm(xv, w, eps=1e-5)
        rows_diff = int((~torch.eq(a, b).all(dim=1)).sum())
        res[f"x offset {off} elems ({off * 2} B)"] = rows_diff
    # fused op: branch misaligned
    r = (torch.randn(T, H, device=DEV, generator=g) * 3).bfloat16()
    bv = buf[1 : 1 + T * H].view(T, H)
    bv.copy_(x)
    f1 = N.fused_post_norm_residual_pre_norm(x, r, w, 1e-8, w, 1e-5)
    f2 = N.fused_post_norm_residual_pre_norm(bv, r, w, 1e-8, w, 1e-5)
    res["fused branch offset 1 elem: rnew rows differing"] = int(
        (~torch.eq(f1[0], f2[0]).all(dim=1)).sum())
    # weight misaligned
    wb = torch.empty(H + 8, device=DEV, dtype=torch.bfloat16)
    wv = wb[1 : 1 + H]
    wv.copy_(w)
    c = N.centered_rmsnorm(x, wv, eps=1e-5)
    res["w offset 1 elem: rows differing"] = int((~torch.eq(a, c).all(dim=1)).sum())
    # embedding: table view misaligned is not possible to construct cheaply at V=202048; use small V
    V = 1000
    tab = (torch.randn(V, H, device=DEV, generator=g) * 0.02).bfloat16()
    tb = torch.empty(V * H + 8, device=DEV, dtype=torch.bfloat16)
    tv = tb[1 : 1 + V * H].view(V, H)
    tv.copy_(tab)
    ids = torch.randint(0, V, (64,), device=DEV, generator=g)
    e1 = N.normalized_embedding(tab, ids)
    e2 = N.normalized_embedding(tv, ids)
    res["embedding table offset 1 elem: rows differing"] = int(
        (~torch.eq(e1, e2).all(dim=1)).sum())
    return res


def attention_alignment():
    """q/gate/k/v as contiguous views at an odd element offset vs aligned copies."""
    n = 300
    g = torch.Generator(device=DEV).manual_seed(9)
    q = (torch.randn(n, HQ, D, device=DEV, generator=g) * 3.87).bfloat16()
    k = torch.randn(n, HKV, D, device=DEV, generator=g).bfloat16()
    v = torch.randn(n, HKV, D, device=DEV, generator=g).bfloat16()
    gate = (torch.randn(n, HQ * D, device=DEV, generator=g) * 2).bfloat16()

    def mis(t):
        b = torch.empty(t.numel() + 8, device=DEV, dtype=t.dtype)
        tv = b[1 : 1 + t.numel()].view(t.shape)
        tv.copy_(t)
        return tv

    cu = i32([0, n])
    pos = torch.arange(n, dtype=torch.int32, device=DEV)
    res = {}
    for window in (None, W):
        a = ga.strict_gated_attention_fwd_varlen(q, k, v, gate, cu, cu, pos, window=window)
        b = ga.strict_gated_attention_fwd_varlen(mis(q), mis(k), mis(v), mis(gate), cu, cu, pos,
                                                 window=window)
        res[str(window)] = all(torch.equal(x, y) for x, y in zip(a, b))
    return f"misaligned views give identical (y, o, lse): {res}"


def embedding_oob():
    tab = torch.randn(100, 6656, device=DEV).bfloat16()
    ids = torch.tensor([3, 100], device=DEV)
    y = N.normalized_embedding(tab, ids)
    torch.cuda.synchronize()
    return f"OOB id accepted; row1 all zero: {bool((y[1] == 0).all())}"


if __name__ == "__main__":
    import sys
    for name in sys.argv[1:]:
        run(name, globals()[name])
