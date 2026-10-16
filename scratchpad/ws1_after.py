VERDICT_ZH = {"confirmed_fixed": "确认并修复", "confirmed_not_fixed": "确认，未改（已声明）", "refuted": "被驳回", "not_checked": "未核查"}
VERDICT_CLS = {"confirmed_fixed": "s-ok", "confirmed_not_fixed": "s-pend", "refuted": "s-p7", "not_checked": "s-bad"}


def ws1_review_block() -> str:
    kg, kr, ka = load("kernels_gemm.json"), load("kernels_rope.json"), load("kernels_attention.json")
    out = []
    rows = []
    for name, d in (("GEMM / FFN / LM head", kg), ("QK-norm + RoPE", kr), ("注意力反向", ka)):
        if not d:
            continue
        fs = d.get("findings") or d.get("finding_verdicts") or []
        cnt = {k: 0 for k in VERDICT_ZH}
        for f in fs:
            v = f.get("verdict")
            if v in cnt:
                cnt[v] += 1
        rows.append((f"{name}（{len(fs)} 条）", [cnt[k] for k in VERDICT_ZH]))
    if rows:
        out.append(fig(stacked(rows, [(VERDICT_ZH[k], VERDICT_CLS[k]) for k in VERDICT_ZH], title="上一轮对抗 review 留下的未验证 findings：这一轮逐条复现后的结论", width=860, bar_h=18, unit="条"),
                       "每条 finding 先写探针复现，确认后修复并补一个“修复前失败、修复后通过”的测试。"))
    if kg:
        p1 = kg.get("det_gemm_1_iters_probe", {})
        bf, af = p1.get("before", {}).get("forward_sweep", []), p1.get("after", {}).get("forward_sweep", [])
        if bf and af:
            rows = [(f"K={b['K']} M={b['M']}", [100 * b["mismatch"] / b["numel"], 100 * a["mismatch"] / a["numel"]]) for b, a in zip(bf, af)]
            out.append(fig(hbars(rows, [("修复前", "s-bad"), ("修复后", "s-ok")], width=860, bar_h=12, xmax=100, value_fmt=lambda v: f"{v:.1f}%",
                                 title="det_gemm:1：输出与“声明的 8 叶树”不一致的比例"),
                           "K 很小时每个 K 块只循环一次，Triton 编译器把树的加法折进了 MMA 的累加链（7 个加法只剩 3 个），约一半输出偏离契约。修复：强制每块至少两次迭代 + static_assert + 检查编译后的 TTGIR。30B 前向的 101 个形状哈希全部不变，只有反向里 53/126 个形状改了 bit。"))
        cp = kg.get("det_gemm_2_tp_probe", {}).get("column_parallel", [])
        if cp:
            rows = [(f"{r['shape']} TP{r['tp']}", [100 * r["naive_mismatch"] / r["numel"], 100 * sum(r["canonical_mismatch_per_rank"]) / r["numel"]]) for r in cp]
            out.append(fig(hbars(rows, [("rank 本地反向 + bf16 后再 all-reduce", "s-bad"), ("fp32 子树部分和 + 规范树", "s-ok")], width=860, bar_h=11, xmax=100,
                                 value_fmt=lambda v: f"{v:.1f}%", title="det_gemm:2：列并行 GEMM 的 dA，与 TP1 不同的元素比例"),
                           "原来的 autograd 反向在每个 rank 上先舍入到 bf16 再相加，TP>1 时 37–41% 的元素和 TP1 不同；改成每个 rank 输出 fp32 部分和、用规范树合并后只舍入一次，TP2/4/8 全部 0。"))
    if kr:
        t = kr.get("experiments", {}).get("table_cpu_vs_cuda_per_bucket")
        if t:
            xs = [b / 1024 for b in t["bucket_start"]]
            out.append(fig(lines(xs, [("cos", "s-p0", [100 * v for v in t["cos_mismatch_frac"]]), ("sin", "s-p1", [100 * v for v in t.get("sin_mismatch_frac", t["cos_mismatch_frac"])])],
                                 title="RoPE 表在 CPU 上构造 vs 在 GPU 上构造：bit 不同的元素比例", xlabel="位置（K token）", ymin=0, width=860, height=240,
                                 dots=False, y_fmt=lambda v: f"{v:.0f}%"),
                           "同一个公式、同样 fp32，CPU 和 CUDA 的 sin/cos 实现不同，约 20% 的表项 bit 不同。所以契约 O4 要求两边用同一个函数在 CUDA 上构造一次、缓存同一个对象；这一行现在会校验表的来源。"))
    return "".join(out)
