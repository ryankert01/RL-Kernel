import sys

p = sys.argv[1]
s = open(p).read()
start = s.index('# --------------------------------------------------------------------------------- parity')
end = s.index('def _diff_into(')
new_parity = '''# --------------------------------------------------------------------------------- parity
def _bits(t: torch.Tensor) -> torch.Tensor:
    """Integer view of a float tensor: mismatches are counted on bit patterns (-0.0 != +0.0)."""
    if not t.is_floating_point():
        return t
    return t.contiguous().view({2: torch.int16, 4: torch.int32, 8: torch.int64}[t.element_size()])


def _mm(a: torch.Tensor, b: torch.Tensor) -> int:
    """Bit-pattern mismatch count (-1 if shape/dtype differ)."""
    if a.shape != b.shape or a.dtype != b.dtype:
        return -1
    return int((_bits(a) != _bits(b)).sum().item())


def _packed_inputs(lens: list[int], seed: int, hq: int = HQ, hkv: int = HKV):
    seqs = [_inputs(n, seed=seed + i, hq=hq, hkv=hkv) for i, n in enumerate(lens)]
    inp = {k: torch.cat([s[k] for s in seqs]).contiguous() for k in ("q", "k", "v", "gate", "dy")}
    cu = _i32([0, *torch.tensor(lens).cumsum(0).tolist()])
    pos = torch.cat([torch.arange(n, dtype=torch.int32, device=DEV) for n in lens])
    return inp, cu, pos


def _tp_rank_slices(tp: int, r: int):
    """Rank r of TP t: Q heads [32r/t, 32(r+1)/t), gate/y/o columns, its KV head, its groups."""
    hq = HQ // tp
    heads = slice(r * hq, (r + 1) * hq)
    cols = slice(r * hq * D, (r + 1) * hq * D)
    kvh = (r * hq) // (HQ // HKV)
    groups = slice(r * (8 // tp), (r + 1) * (8 // tp))
    return hq, heads, cols, kvh, groups


def _tp_parity_varlen(case: str, lens: list[int], seed: int, window) -> list[dict[str, Any]]:
    """Forward (y, o, lse) and backward (dq, dgate, dK/dV group partials, tree) per TP rank."""
    from rl_engine.models.muse_glimmer.runtime.tp_canonical_tree import combine_tree as tp_tree

    inp, cu, pos = _packed_inputs(lens, seed)
    n = inp["q"].shape[0]
    y, o, lse = _fwd_varlen(inp, cu, cu, pos, window)
    dgate, do, delta = bwd.gated_attention_bwd_preprocess(inp["dy"], o, inp["gate"])
    dq, dkp, dvp = bwd.strict_gated_attention_bwd_partials(
        inp["q"], inp["k"], inp["v"], do, lse, delta, cu, cu, pos, window=window
    )
    dk, dv = bwd.combine_kv_partials(dkp), bwd.combine_kv_partials(dvp)
    out = []
    for tp in (2, 4, 8):
        rec: dict[str, Any] = {"case": case, "tp": tp, "window": window, "Tq": n, "ranks": []}
        roots_k, roots_v = [], []
        for r in range(tp):
            hq, heads, cols, kvh, groups = _tp_rank_slices(tp, r)
            sh = {"q": inp["q"][:, heads].contiguous(),
                  "k": inp["k"][:, kvh : kvh + 1].contiguous(),
                  "v": inp["v"][:, kvh : kvh + 1].contiguous(),
                  "gate": inp["gate"][:, cols].contiguous(),
                  "dy": inp["dy"][:, cols].contiguous()}
            heads_kw = {"num_q_heads": hq, "num_kv_heads": 1}
            y_r, o_r, l_r = _fwd_varlen(sh, cu, cu, pos, window, **heads_kw)
            dg_r, do_r, de_r = bwd.gated_attention_bwd_preprocess(
                sh["dy"], o_r, sh["gate"], num_q_heads=hq
            )
            dq_r, dkp_r, dvp_r = bwd.strict_gated_attention_bwd_partials(
                sh["q"], sh["k"], sh["v"], do_r, l_r, de_r, cu, cu, pos, window=window, **heads_kw
            )
            rank_rec = {
                "rank": r, "q_heads": [heads.start, heads.stop], "kv_head": kvh,
                "y": _mm(y_r, y[:, cols]), "o": _mm(o_r, o[:, cols]),
                "lse": _mm(l_r, lse[:, heads]), "dq": _mm(dq_r, dq[:, heads]),
                "dgate": _mm(dg_r, dgate[:, cols]), "delta": _mm(de_r, delta[:, heads]),
                "dk_group_partials": _mm(dkp_r[:, :, 0], dkp[groups, :, kvh]),
                "dv_group_partials": _mm(dvp_r[:, :, 0], dvp[groups, :, kvh]),
                "numel_y": y_r.numel(), "numel_dq": dq_r.numel(),
                "numel_partials": dkp_r.numel(),
            }
            if tp == 2:  # a TP2 rank owns its whole KV head: the fused entry is final
                f = bwd.strict_gated_attention_bwd(
                    sh["q"], sh["k"], sh["v"], sh["gate"], o_r, l_r, sh["dy"], cu, cu, pos,
                    window=window, **heads_kw,
                )
                rank_rec.update(
                    fused_dq=_mm(f[0], dq[:, heads]), fused_dk=_mm(f[1][:, 0], dk[:, kvh]),
                    fused_dv=_mm(f[2][:, 0], dv[:, kvh]), fused_dgate=_mm(f[3], dgate[:, cols]),
                )
            rec["ranks"].append(rank_rec)
            for roots, part in ((roots_k, dkp_r), (roots_v, dvp_r)):
                root = bwd.kv_partials_subtree(part)  # (Tk, 1, D) fp32: this rank's tree node
                full = torch.zeros((n, HKV, D), dtype=torch.float32, device=DEV)
                full[:, kvh] = root[:, 0]
                roots.append(full)
        # replicated-KV rule: every rank's sub-tree root (zeros at the other KV head) folded in
        # the canonical TP tree over the ranks, rounded once == TP1 dk/dv
        rec["dk_tp_tree_vs_tp1"] = _mm(tp_tree(roots_k).to(torch.bfloat16), dk)
        rec["dv_tp_tree_vs_tp1"] = _mm(tp_tree(roots_v).to(torch.bfloat16), dv)
        rec["numel_dk"] = dk.numel()
        out.append(rec)
    return out


def _tp_parity_paged_decode(bsz: int, ctx: int, window) -> list[dict[str, Any]]:
    """HEAD-tile forward (paged decode, BLOCK_M=16 rows, group 16/8/4) per TP rank."""
    seqs = [_inputs(ctx, seed=85_000 + 31 * i + ctx) for i in range(bsz)]
    k_all = torch.cat([s["k"] for s in seqs])
    v_all = torch.cat([s["v"] for s in seqs])
    kc, vc, table, seq_lens = _paged_from_contiguous(k_all, v_all, bsz, ctx)
    q = torch.cat([s["q"][-1:] for s in seqs]).contiguous()
    gate = torch.cat([s["gate"][-1:] for s in seqs]).contiguous()
    cu_q = _i32(range(bsz + 1))
    y, o, lse = ga.strict_gated_attention_fwd_paged(
        q, gate, kc, vc, table, seq_lens, cu_q, window=window
    )
    out = []
    for tp in (2, 4, 8):
        rec: dict[str, Any] = {"case": f"paged_decode/B={bsz}/ctx={ctx}", "tile": "head",
                               "tp": tp, "window": window, "ranks": []}
        for r in range(tp):
            hq, heads, cols, kvh, _groups = _tp_rank_slices(tp, r)
            y_r, o_r, l_r = ga.strict_gated_attention_fwd_paged(
                q[:, heads].contiguous(), gate[:, cols].contiguous(),
                kc[:, :, kvh : kvh + 1].contiguous(), vc[:, :, kvh : kvh + 1].contiguous(),
                table, seq_lens, cu_q, window=window, num_q_heads=hq, num_kv_heads=1,
            )
            rec["ranks"].append({"rank": r, "y": _mm(y_r, y[:, cols]), "o": _mm(o_r, o[:, cols]),
                                 "lse": _mm(l_r, lse[:, heads]), "numel_y": y_r.numel()})
        out.append(rec)
    return out


def run_parity(cp_lengths=(8, 300, 2048, 2100, 4096, 8192)) -> dict[str, Any]:
    """Bit-pattern mismatch counts (expected all 0) for TP head shards and CP, one GPU.

    TP t in {2, 4, 8}: rank r runs the strict forward/backward on its head shard (32/t Q heads,
    one KV head, replicated when t > 2) and is compared with the TP1 call's slices (y, o, lse,
    dq, dgate, the dK/dV group partials); the ranks' sub-tree roots folded in
    runtime/tp_canonical_tree.combine_tree must equal TP1 dk/dv. Forward also on the HEAD tile
    (paged decode). CP: the runtime/cp_attention backward at CP1 (the plan the trainer runs) and
    simulated CP2/4/8 ranks (each rank's kernel call on its contiguous rows against the full
    K/V; rank roots folded in the canonical tree; all_gather is a pure copy) vs the one-shot
    kernel. The real two-GPU CP2 run is runtime/check_cp_parity.py.
    """
    from rl_engine.models.muse_glimmer.ops.canonical_tree import combine_tree
    from rl_engine.models.muse_glimmer.runtime import cp_attention as cpa

    tp_records: list[dict[str, Any]] = []
    cp_records: list[dict[str, Any]] = []
    for window in (None, W):
        tp_records += _tp_parity_varlen("T=2100", [2100], 80_000, window)
        tp_records += _tp_parity_varlen("packed[1500,600,333]", [1500, 600, 333], 81_000, window)
        tp_records += _tp_parity_varlen("T=8192", [8192], 82_000, window)
        tp_records += _tp_parity_paged_decode(8, 2049, window)
        for n in cp_lengths:
            inp = _inputs(n, seed=90_000 + n)
            cu = _i32([0, n])
            pos = torch.arange(n, dtype=torch.int32, device=DEV)
            _y, o, lse = _fwd_varlen(inp, cu, cu, pos, window)
            one = bwd.strict_gated_attention_bwd(
                inp["q"], inp["k"], inp["v"], inp["gate"], o, lse, inp["dy"], cu, cu, pos,
                window=window,
            )
            cp1 = cpa.cp_gated_attention_backward(
                inp["q"], inp["k"], inp["v"], inp["gate"], o, lse, inp["dy"], pos,
                cpa.CPAttentionPlan(1, 0), None, window=window,
            )
            cp_records.append({
                "cp": 1, "path": "runtime/cp_attention.cp_gated_attention_backward", "T": n,
                "window": window, "dq": _mm(cp1[0], one[0]), "dk": _mm(cp1[1], one[1]),
                "dv": _mm(cp1[2], one[2]), "dgate": _mm(cp1[3], one[3]),
                "numel_dq": one[0].numel(), "numel_dk": one[1].numel(),
            })
            for cp in (2, 4, 8):
                if n % cp:
                    continue
                span = n // cp
                rk, rv, dqs, dgs = [], [], [], []
                for i in range(cp):
                    rows = slice(i * span, (i + 1) * span)
                    dg_i, do_i, de_i = bwd.gated_attention_bwd_preprocess(
                        inp["dy"][rows].contiguous(), o[rows].contiguous(),
                        inp["gate"][rows].contiguous(),
                    )
                    dq_i, dkp_i, dvp_i = bwd.strict_gated_attention_bwd_partials(
                        inp["q"][rows].contiguous(), inp["k"], inp["v"], do_i,
                        lse[rows].contiguous(), de_i, _i32([0, span]), cu,
                        pos[rows].contiguous(), window=window,
                    )
                    rk.append(dkp_i)
                    rv.append(dvp_i)
                    dqs.append(dq_i)
                    dgs.append(dg_i)
                got_k = bwd.combine_kv_partials(combine_tree(rk))
                got_v = bwd.combine_kv_partials(combine_tree(rv))
                cp_records.append({
                    "cp": cp, "path": "simulated ranks (kernel calls + canonical rank tree)",
                    "T": n, "window": window, "rows_per_rank": span,
                    "dq": _mm(torch.cat(dqs), one[0]), "dk": _mm(got_k, one[1]),
                    "dv": _mm(got_v, one[2]), "dgate": _mm(torch.cat(dgs), one[3]),
                    "numel_dq": one[0].numel(), "numel_dk": one[1].numel(),
                })
    return {"tp": tp_records, "cp": cp_records}


'''
s = s[:start] + new_parity + s[end:]
old = '''def run_bench(iters: int, quick: bool) -> dict[str, Any]:
    lengths = (512, 1024, 2048, 4096, 8192) if not quick else (1024, 4096)
    out: dict[str, Any] = {"prefill_forward": [], "decode_forward": [], "backward": [],
                           "tp_shard_forward": [], "tp_shard_backward": []}
    for window in (None, W):
        for n in lengths:
            out["prefill_forward"].append(bench_prefill(n, window, iters))
            out["backward"].append(bench_backward(n, window, max(5, iters // 2)))
        for bsz in ((1, 16, 64, 256) if not quick else (1, 64)):
            out["decode_forward"].append(bench_decode(bsz, 4096, window, iters))
    if _supports_heads():'''
new = '''SECTIONS = ("prefill", "decode", "backward", "tp")


def run_bench(iters: int, quick: bool, sections=SECTIONS) -> dict[str, Any]:
    lengths = (512, 1024, 2048, 4096, 8192) if not quick else (1024, 4096)
    out: dict[str, Any] = {"prefill_forward": [], "decode_forward": [], "backward": [],
                           "tp_shard_forward": [], "tp_shard_backward": [],
                           "sections": list(sections), "iters": iters}
    for window in (None, W):
        for n in lengths:
            if "prefill" in sections:
                out["prefill_forward"].append(bench_prefill(n, window, iters))
            if "backward" in sections:
                out["backward"].append(bench_backward(n, window, max(5, iters // 2)))
        if "decode" in sections:
            for bsz in ((1, 16, 64, 256) if not quick else (1, 64)):
                out["decode_forward"].append(bench_decode(bsz, 4096, window, iters))
    if "tp" in sections and _supports_heads():'''
assert old in s
s = s.replace(old, new)
old = '''    ap.add_argument("--old-module", default=None)'''
new = '''    ap.add_argument("--old-module", default=None)
    ap.add_argument("--sections", nargs="+", choices=SECTIONS, default=list(SECTIONS),
                    help="bench mode: which timing series to run")'''
assert old in s
s = s.replace(old, new)
old = '''        payload = run_bench(args.iters, args.quick)'''
new = '''        payload = run_bench(args.iters, args.quick, tuple(args.sections))'''
assert old in s
s = s.replace(old, new)
open(p, "w").write(s)
print("patched")
