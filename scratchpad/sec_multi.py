def sec_multi() -> str:
    k = load("kernels_attention.json")
    out = [fig(D.fig_cp(), "CP 下 K/V 先 all-gather（T=8192、CP2 时 0.23 ms，约占前向 4%），query 按 rank 切。")]
    if not k:
        out.append(pending("注意力的头数参数化与 CP 反向规则"))
    else:
        par = k.get("parity", {})
        tm = par.get("tp_matrix", [])
        cols = [("y", "y"), ("o", "o"), ("lse", "lse"), ("dq", "dq"), ("dgate", "dgate"), ("dK", "dk"), ("dV", "dv")]
        rows, vals = [], []
        for r in tm:
            mm = r.get("mismatches_summed_over_ranks", {})
            lab = f'{r["case"]} TP{r["tp"]} {"局部" if r.get("window") else "全局"}'
            row = []
            for _, c in cols:
                if c in ("dk", "dv"):
                    row.append(r.get(f"{c}_tp_tree_vs_tp1", r.get(f"{c}_replica_tree_vs_tp1")))
                else:
                    row.append(mm.get(c))
            rows.append(lab)
            vals.append(row)
        if rows:
            out.append(fig(heatmap(rows, [c for c, _ in cols], vals, title="每个 rank 只算自己的头：与 TP1 对应切片相比，bit 不同的元素数", cell=22, cell_w=52, col_rotate=False),
                           "TP2 每个 rank 16 个 Q 头 / 1 个 KV 头，TP4、TP8 是 8/1、4/1（KV 头复制）。前向和反向全部 0；复制的 KV 头的 dK/dV 由各副本的 fp32 部分和按规范树合并，也和 TP1 一样。所以 TP 下不需要复制 KV、也不需要 all-gather 头。"))
        v1 = {2048: 1557, 4096: 5540, 8192: 20440}
        out.append(fig(hbars([(f"T={t}", [n, 0]) for t, n in v1.items()], [("旧内核（按 query block 升序累加）", "s-bad"), ("新内核（8 个规范 query 块 + 树）", "s-ok")],
                             width=760, bar_h=14, value_fmt=lambda v: f"{v:.0f}", title="单卡一次性反向 vs CP1 路径：dK 里 bit 不同的元素数"),
                       "旧规则在 CP 下做不到；新规则下单卡、CP1、模拟 CP2/4/8、真实双卡 CP2 全部逐 bit 相同。前向 66/66 个摘要不变。"))
        ba = k.get("benchmarks", {}).get("backward_before_after", [])
        if ba:
            def med(x):
                x = sorted(x)
                return x[len(x) // 2]
            out.append(fig(hbars([(f'T={r["T"]} {"局部" if r.get("window") else "全局"}', [med(r["v1_ms"]), med(r["v2_ms"])]) for r in ba],
                                 [("旧反向", "s-p7"), ("新反向", "s-strict")], width=760, bar_h=12, unit=" ms", title="代价：反向耗时（ms，交替计时 7 轮的中位数）"),
                           "T ≥ 1024 时新反向慢 5–14%，T=512 时慢 55%（瞬时显存 2.5 倍）。"))
        sb = k.get("benchmarks", {}).get("tp_shard_backward", [])
        sf = k.get("benchmarks", {}).get("tp_shard_forward", [])
        if sb:
            rows = []
            for r in sb:
                f = next((x for x in sf if x.get("tp") == r.get("tp") and x.get("window") == r.get("window")), None)
                rows.append((f'TP{r["tp"]} {"局部" if r.get("window") else "全局"}', [f["strict_ms"] if f else None, r["strict_ms"]]))
            out.append(fig(hbars(rows, [("前向", "s-p4"), ("反向", "s-strict")], width=760, bar_h=12, unit=" ms",
                                 title="T=4096 时一个 rank 的注意力耗时（ms）"), "TP4/TP8 的反向不含跨 rank 的合并。"))
    return f"""
<section id="multi">
  <p class="eyebrow">WS2 · 多卡</p>
  <h2>TP 和 CP：切卡不改变加法顺序</h2>
  {''.join(out)}
  <div class="callout warn"><b>修正的契约缺陷</b>：契约 O5 原来的反向规则“dK/dV 按 query block 升序累加”不是 CP 不变的（每个 rank 只有自己的 query 块）。现在 O5 改成“整条序列 8 个规范 query 块，块间按 O3 树合并；每个 KV 头只在读它的 4 个 Q 头组之间合并”，单卡内核和 CP 路径共用同一条规则。</div>
</section>"""
