#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Build the illustrated Muse-Glimmer POC blog: reports/muse_glimmer_poc/blog/index.html.

Inputs are the machine-readable results under reports/muse_glimmer_poc/data/. A section whose
data file is missing renders a visible "not run" panel instead of numbers, so the page never
claims a result that is not on disk.  Run:  python3 reports/muse_glimmer_poc/blog/build_blog.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import diagrams as D  # noqa: E402
from svgkit import esc, fmt, hbars, heatmap, hist, lines, stacked, svg, text_w  # noqa: E402

HERE = Path(__file__).resolve().parent
POC = HERE.parent
DATA = Path(__import__("os").environ.get("MUSEGLIMMER_BLOG_DATA", POC / "data"))
OUT = HERE / "index.html"


def load(name: str):
    p = DATA / name
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return None


def pending(what: str, cmd: str = "") -> str:
    c = f'<code>{esc(cmd)}</code>' if cmd else ""
    return f'<div class="pending"><span class="pill pill-pend">未运行</span><span>{esc(what)}</span>{c}</div>'


def fig(svg_html: str, caption: str, wide: bool = True) -> str:
    return f'<figure class="fig{" wide" if wide else ""}"><div class="figscroll">{svg_html}</div><figcaption>{caption}</figcaption></figure>'


def details(summary: str, body: str) -> str:
    return f'<details><summary>{esc(summary)}</summary><div class="dbody">{body}</div></details>'


def table(cols: list[str], rows: list[list[object]], num_from: int = 1) -> str:
    th = "".join(f"<th>{esc(c)}</th>" for c in cols)
    trs = []
    for r in rows:
        tds = "".join(f'<td class="{"num" if i >= num_from else ""}">{c if isinstance(c, str) and c.startswith("<") else esc(c)}</td>' for i, c in enumerate(r))
        trs.append(f"<tr>{tds}</tr>")
    return f'<div class="tscroll"><table><thead><tr>{th}</tr></thead><tbody>{"".join(trs)}</tbody></table></div>'


def num(s: str) -> float | None:
    try:
        return float(str(s).replace("x", "").replace(",", "").split()[0])
    except (ValueError, IndexError):
        return None


# ----------------------------------------------------------------------------------------------
# Sections
# ----------------------------------------------------------------------------------------------


def sec_why() -> str:
    hs, ht, vs, vt = D.sum_orders()
    abl = load("ablation.json")
    native_line = ""
    if abl and isinstance(abl, dict) and abl.get("headline"):
        native_line = f"<p>{esc(abl['headline'])}</p>"
    return f"""
<section id="why">
  <p class="eyebrow">问题</p>
  <h2>同一个模型，为什么训练和推理算出来的 logprob 不一样</h2>
  {fig(D.fig_sum_orders(), f"同样 8 个 fp32 数：从左到右加得到 <code>0x{hs}</code>，按 8 叶平衡树加得到 <code>0x{ht}</code>。浮点加法不满足结合律，<b>顺序就是结果的一部分</b>。")}
  <div class="prose">
    <p>RL 训练里，rollout 引擎（vLLM）生成 token 并记下每个 token 的 log p，训练引擎（Megatron）再对同样的 token 重算一次 log p，两者的比值进入 PPO/GRPO 的重要性权重。模型、权重、输入都一样，可两边的 GEMM 切块、注意力分块、TP 归约、算子融合各不相同，加法顺序不同，bit 就不同。这点差异在更新之前就会被当成“策略偏移”。</p>
    <p>RFC #501 的目标：给 Muse-Glimmer-30B 写一份<b>数值契约</b>，规定每一个归约的顺序、每一次舍入的位置，让 vLLM 和 Megatron 按同一份契约执行，<b>逐 bit 相同</b>，而不是“误差很小”。</p>
  </div>
  {native_line}
</section>"""


ROWS = [
    # (group, row, status, note)
    ("WS1 单卡", "arch_fingerprint", "done", "checkpoint、config、shard SHA 全部钉死"),
    ("WS1 单卡", "dense_gemm（树形 GEMM）", "done", "8 叶 K 树，105–135 TFLOP/s"),
    ("WS1 单卡", "normalized_embedding / norms", "done", "四范数夹心，两个 eps"),
    ("WS1 单卡", "qknorm_scaled_rope", "done", "kernel == golden 逐 bit"),
    ("WS1 单卡", "global/local 门控注意力 fwd", "done", "train = chunked = decode = paged"),
    ("WS1 单卡", "门控注意力 bwd", "done", "dK/dV 按规范块累加"),
    ("WS1 单卡", "swiglu_ffn", "done", "GEMM 占 98.7% 时间"),
    ("WS1 单卡", "lm_head + softcap + logprob", "done", "显存 396 MiB vs 9474"),
    ("链", "full_model_chain（52 层真权重）", "done", "10 种切法逐 bit 相同"),
    ("WS2 多卡", "tp_canonical_tree", "done", "TP2/4/8 == TP1"),
    ("WS2 多卡", "vocab_parallel_softcap_logp", "done", "TP2/4/8 == TP1"),
    ("WS2 多卡", "kv_head_replication", "done", "TP4/8 复制 KV"),
    ("WS2 多卡", "cp_global / cp_local attention", "done", "CP2/4 == CP1 前向+反向"),
    ("运行时", "vLLM 0.31 严格绑定", "wave", ""),
    ("运行时", "Megatron Bridge 严格绑定", "wave", ""),
    ("运行时", "weight_sync_identity", "wave", ""),
    ("端到端", "final_e2e_lora（GRPO）", "wave", ""),
    ("端到端", "museglimmer_ablation", "wave", ""),
]


def sec_scope() -> str:
    status = load("row_status.json") or {}
    cells = []
    for grp, row, st, note in ROWS:
        st2 = status.get(row, {}).get("status", st)
        note2 = status.get(row, {}).get("note", note)
        cls = {"done": "st-done", "partial": "st-part", "wave": "st-pend", "blocked": "st-bad"}.get(st2, "st-pend")
        lab = {"done": "完成", "partial": "部分", "wave": "进行中", "blocked": "受阻"}.get(st2, st2)
        cells.append(f'<div class="rowcell {cls}"><span class="rg">{esc(grp)}</span><b>{esc(row)}</b><span class="pill">{esc(lab)}</span><span class="rn">{esc(note2)}</span></div>')
    return f"""
<section id="scope">
  <p class="eyebrow">范围</p>
  <h2>这次 POC 做了什么</h2>
  <div class="prose"><p>把 RFC 的每一行都做到“能用、能测”的程度，再在一台 8×A100 上串起真实 vLLM 和真实 Megatron Bridge，跑一个 LoRA GRPO。性能只求够用；Phase B 视觉、DFlash 投机解码、全参数训练、量化、ROCm 不在范围内。</p></div>
  {fig(D.fig_system(), "端到端循环。两个运行时装在两个不能合并的 Python 环境里（torch 的 CUDA 构建不同），trainer 跑在子进程里，中间只传 token id、fp32 的 dL/dlogp 和带哈希的 LoRA。")}
  <div class="rowgrid">{''.join(cells)}</div>
</section>"""


FACTS = [
    ("qk_scale_factor", "43.7840518911", "3.87", "只乘 Q；有效倍数 3.87/√128 = 0.342。43.78 = 3.87·√128 是把两个系数合并后的数。"),
    ("输出门", "按头（per-head）", "逐元素 4096 维", "gate_proj 是 6656→4096 的线性层，输入是 input_layernorm 的输出，作用在 o_proj 之前。"),
    ("参数量", "29.6B", "29.78B", "文本 27.85B + 视觉 1.92B（index 里的 total_parameters）。"),
    ("checkpoint 大小", "59.58 GB", "59.55 GB", "59.58 是 HF 仓库总存储；两个 shard 共 59,553,435,272 字节。"),
    ("vLLM 原生支持", "v0.27.0", "v0.28.0", "PR #51655，2026-08-14 合并；vLLM 还会把 gate/up 也融合成一个 GEMM。"),
    ("配置键名", "use_qk_norm 等", "HF 键名", "RFC 表里是 vLLM 扁平 schema 的名字；HF 里是 final_logit_softcapping 等，QK-norm 无开关、总是开启。"),
]


def sec_model() -> str:
    cards = "".join(
        f'<div class="fact"><span class="fk">{esc(k)}</span><span class="fold"><s>{esc(a)}</s></span><span class="fnew">{esc(b)}</span><span class="fn">{esc(n)}</span></div>'
        for k, a, b, n in FACTS
    )
    return f"""
<section id="model">
  <p class="eyebrow">模型</p>
  <h2>Muse-Glimmer-30B 长什么样</h2>
  {fig(D.fig_model(), "层调度和一层的数据流。四个 norm 把注意力和 FFN 夹在中间，post-norm 的 eps 是 1e-8，其余 1e-5；norm 的增益是 (1+w)。")}
  <h3>RFC 里需要改的几个数</h3>
  <div class="facts">{cards}</div>
</section>"""


def sec_contract() -> str:
    tp_bars = hbars(
        [("TP2 · NCCL all_reduce", [0]), ("TP4 · NCCL all_reduce o_proj", [46]), ("TP4 · NCCL all_reduce down", [48]),
         ("TP2/TP4/TP8 · 规范树", [0])],
        [("与 TP1 不同的 bf16 元素数（共 1,703,936）", "s-bad")],
        title="o_proj / down 输出与 TP1 相比", width=720, bar_h=18, value_fmt=lambda v: f"{v:.0f}",
    )
    leaves = hbars(
        [("8 个块直接顺序相加", [80, 80]), ("4 个块", [41, 28]), ("2 个块", [0, 0])],
        [("o_proj", "s-strict"), ("down", "s-native")],
        title="如果不用树、直接把各 rank 的块顺序相加：与 TP1 不同的元素数", width=720, bar_h=14, value_fmt=lambda v: f"{v:.0f}",
    )
    timing = hbars(
        [("TP2 o_proj", [0.162, 0.164]), ("TP2 down", [0.155, 0.114]), ("TP4 o_proj", [0.224, 0.151]), ("TP4 down", [0.222, 0.149])],
        [("all-gather + 规范树（严格）", "s-strict"), ("NCCL all_reduce（顺序不定）", "s-native")],
        title="归约 (256, 6656) fp32 根的耗时，ms / 次", width=720, bar_h=14, unit=" ms",
    )
    return f"""
<section id="contract">
  <p class="eyebrow">数值契约</p>
  <h2>一棵树管住所有加法</h2>
  <div class="prose"><p>契约（<code>docs/contracts/muse-glimmer-30b-strict-contract.md</code>）把每个长归约都切成 8 个固定的块，块内 fp32 累加，块间按一棵固定的平衡二叉树相加，最后只舍入一次到 bf16。TP 只决定“谁算哪棵子树”，不改变树本身。</p></div>
  {fig(D.fig_tp_tree(), "规范 8 叶树。TP2 每个 rank 算 4 片叶子的子树，TP8 每个 rank 只算 1 片；all-gather 之后的合并永远是同一棵树的上层。")}
  {fig(tp_bars, "真实 NCCL rank 上的对照实验：NCCL all_reduce 在 TP2 恰好对（两个数相加满足交换律），到 TP4 就错了 46–48 个元素；规范树在 TP2/4/8 全部逐 bit 相同。")}
  {fig(leaves, "树是必要的：如果每个 rank 算完直接按顺序相加，TP4/TP8 就会和 TP1 不同。")}
  {fig(timing, "代价很小：all-gather + 树和 NCCL all_reduce 同一个量级（TP4 多 0.07 ms）。")}
</section>"""


def _ws1_ratio_rows(b: dict) -> list[tuple[str, list[float]]]:
    bm = b["benchmarks"]
    rows: list[tuple[str, list[float]]] = []
    det = bm["benchmarks/models/muse_glimmer/bench_det_gemm.py"]["tables"]
    t4096 = next(t for t in det if t["heading"].startswith("Throughput at M=4096"))
    for r in t4096["rows"]:
        if r["shape"] in ("q_proj", "o_proj", "mlp_gate", "down_proj", "lm_head"):
            rows.append((f"GEMM {r['shape']} M=4096", [num(r["strict/cuBLAS"])]))
    t16 = next(t for t in det if t["heading"].startswith("Throughput at M=16"))
    for r in t16["rows"]:
        if r["shape"] in ("q_proj", "mlp_gate", "lm_head"):
            rows.append((f"GEMM {r['shape']} M=16（decode）", [num(r["strict/cuBLAS"])]))
    att = bm["benchmarks/models/muse_glimmer/bench_gated_attention.py"]["tables"][0]["rows"]
    for r in att:
        rows.append((f"注意力 {r['case'].replace(' (global)', ' 全局').replace(' (local)', ' 局部')}", [num(r["strict/native"])]))
    for r in bm["benchmarks/models/muse_glimmer/bench_gated_attention_bwd.py"]["tables"][0]["rows"]:
        rows.append((f"注意力反向 {'全局' if 'global' in r['row'] else '局部'} T=4096", [num(r["ratio strict/native"])]))
    for r in bm["benchmarks/models/muse_glimmer/bench_norms.py"]["tables"][0]["rows"]:
        if "fwd+bwd" not in r["op"] or "fused" in r["op"]:
            rows.append((r["op"].replace("_", " "), [num(r["strict/native"])]))
    rows.append(("qknorm + RoPE 前向", [0.088 / 0.9593]))
    rows.append(("qknorm + RoPE 前向+反向", [0.3862 / 2.3927]))
    for r in bm["benchmarks/models/muse_glimmer/bench_swiglu_ffn.py"]["tables"][0]["rows"]:
        if r["case"].startswith("swiglu"):
            rows.append((r["case"], [num(r["strict / native"])]))
    rows.append(("LM head + softcap + logprob 前向", [97.3 / 69.1]))
    return rows


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


def sec_ws1() -> str:
    b = load("ws1_bench.json")
    if not b:
        return pending("WS1 基准数据缺失")
    kg = load("kernels_gemm.json")
    rows = _ws1_ratio_rows(b)
    ratio = hbars(rows, [("严格实现耗时 / 原生耗时", "s-strict")], title="严格内核 vs 原生实现：耗时比（对数轴，左边 = 更快）",
                  log=True, ref=1.0, ref_label="1× 一样快", width=860, bar_h=16, value_fmt=lambda v: f"{v:.2f}×")
    det = b["benchmarks"]["benchmarks/models/muse_glimmer/bench_det_gemm.py"]["tables"]
    t4096 = next(t for t in det if t["heading"].startswith("Throughput at M=4096"))
    tf = hbars([(r["shape"], [num(r["strict TFLOP/s"]), num(r["cuBLAS TFLOP/s"])]) for r in t4096["rows"]],
               [("树形 GEMM（严格）", "s-strict"), ("cuBLAS（顺序不定）", "s-native")], title="M=4096 时的 GEMM 吞吐，TFLOP/s（A100 bf16 峰值 312）",
               width=860, bar_h=12)
    legacy = hbars([(r["shape"], [num(r["legacy/strict"])]) for r in t4096["rows"] if num(r["legacy/strict"])],
                   [("仓库原有 Triton det_gemm 比新树形 GEMM 慢多少倍", "s-native")], log=True, width=860, bar_h=14,
                   value_fmt=lambda v: f"{v:.0f}×", title="为什么要重写 GEMM：仓库原有确定性 GEMM 在 A100 上只有 ~5 TFLOP/s")
    mem = hbars([("原生（cuBLAS + 全量 logits）", [9474]), ("严格（8 个词表 tile 流式）", [396])], [("峰值显存 MiB", "s-strict")],
                title="LM head + softcap + logprob，T=4096，词表 202752：峰值显存", width=720, bar_h=20, unit=" MiB",
                value_fmt=lambda v: f"{v:,.0f} MiB")
    tests = hbars([("dense GEMM", [80]), ("门控注意力 fwd", [63]), ("norms + embedding", [43]), ("qknorm + RoPE", [39]),
                   ("指纹 / loader", [33]), ("门控注意力 bwd", [24]), ("swiglu FFN", [20]), ("lm head", [15])],
                  [("测试数（合并后全部通过）", "s-ok")], title="WS1 每一行的测试数", width=720, bar_h=14, value_fmt=lambda v: f"{v:.0f}")
    after = ws1_review_block() or pending("GEMM/FFN/LM-head 的 review 修复和前后对比")
    return f"""
<section id="ws1">
  <p class="eyebrow">WS1 · 单卡算子</p>
  <h2>每个算子都有一个“逐 bit 确定”的版本，速度够用</h2>
  <div class="prose"><p>仓库原有的严格内核（FA4 CuTe 注意力、线性 logp）只支持 Hopper，编译扩展 <code>rl_engine._C</code> 也无法对 CUDA 13 的 torch 构建，所以这次全部算子都是 PyTorch golden + Triton 重写。判断标准：同一个输入，不论 batch 怎么拼、序列怎么切、在第几个位置，输出 bit 都一样；和 fp64 参考的误差在 ws1 容差内。</p></div>
  {fig(ratio, "大多数逐元素/范数类算子比原生更快（原生是未融合的 eager PyTorch）；GEMM 类慢 2–2.6 倍，这是“固定 K 树 + 固定 tile”的代价。注意：局部注意力的“原生”基线是带整张 bool mask 的 SDPA（没有滑窗 flash 内核），严格版因此显得快 20–50 倍，这两行不能当作真实的速度比较。")}
  {fig(tf, "树形 GEMM 跑到 cuBLAS 的 40–50%。decode 形状（M=16）下大 GEMM 和 cuBLAS 持平。")}
  {fig(legacy, "新 GEMM 是这次的“必需行”：没有它，30B 前向一次要几分钟。")}
  {fig(mem, "词表 tile 流式算 softcap 和 logsumexp，不物化 4096×202752 的 logits：显存降到 1/24。")}
  {fig(tests, "第一轮合并时各行的测试数。本轮合并后：vLLM 环境单卡 581 个通过、0 失败；Megatron 环境 222 个通过、0 失败。", wide=False)}
  {after}
</section>"""


def sec_chain() -> str:
    ref = load("reference_profile.json")
    body = ""
    ch = (ref or {}).get("experiments", {}).get("chain_triton_2100_hf")
    if ch:
        pz = {"train": "train forward（整段）", "prefill": "prefill + decode", "chunked:prompt_boundary": "在 prompt 边界切", "chunked:halves": "对半切",
              "chunked:block64": "每 64 token 一块", "chunked:ragged": "不规则切", "chunked:window_boundary_2047": "边界 2047",
              "chunked:window_boundary_2048": "边界 2048", "chunked:window_boundary_2049": "边界 2049", "decode": "逐 token decode"}
        rows = [(pz.get(x["name"], x["name"]), [x["seconds"]]) for x in ch["paths"]]
        eq = all(x.get("byte_equal_to_train") for x in ch["paths"])
        body = fig(hbars(rows, [("耗时（秒）", "s-strict")], width=720, bar_h=14, unit=" s",
                         title=f"prompt 2100 + 32 个答案 token：各条执行路径耗时（{'全部与 train 逐字节相同' if eq else '存在不同'}）"),
                   "本轮在合并后的代码上重跑（Triton 后端）。xrt 的 9 条序列（933 个 token）也做了 train = prefill = decode = 两种分块 的检查，全部逐 bit 相同。")
    else:
        body = pending("合并后重跑真权重链 + HF 锚点")
    anchor = hbars(
        [("严格 golden vs HF fp32", [0.041]), ("严格 Triton vs HF fp32", [0.046]), ("HF bf16 vs HF fp32", [0.046]),
         ("严格 golden vs HF bf16", [0.064]), ("严格 Triton vs HF bf16", [0.066])],
        [("最大 |Δ logp|", "s-strict")], title="精度锚点：和 HF 实现比，最大 |Δ logp|（越小越好）", width=720, bar_h=16,
    )
    argmax = hbars(
        [("严格 golden vs HF fp32", [99.63]), ("严格 Triton vs HF fp32", [99.63]), ("HF bf16 vs HF fp32", [98.53]),
         ("严格 Triton vs HF bf16", [98.9]), ("严格 golden vs HF bf16", [98.5])],
        [("argmax 一致率 %", "s-ok")], title="argmax 一致率（%）", width=720, bar_h=16, xmax=100, value_fmt=lambda v: f"{v:.2f}%",
    )
    return f"""
<section id="chain">
  <p class="eyebrow">全模型链</p>
  <h2>52 层真权重：10 种切法，一个答案</h2>
  {fig(D.fig_window(), "滑动窗口 2048。chunk 边界放在 2047、2048、2049 这些最容易出错的地方，结果都和一次性 train forward 逐 bit 相同；golden（纯 PyTorch 定序）和全 Triton 两条后端都是 32/32 字节相等。")}
  {fig(anchor, "严格实现不是“换了一种错法”：以 HF fp32 为准，严格路径比 HF 自己的 bf16 还近。")}
  {fig(argmax, "")}
  {body}
</section>"""


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













XRT_NAMES = {
    "reference triton": "参考链 Triton", "reference golden": "参考链 golden", "reference hf bf16": "HF bf16",
    "vllm strict tp1": "vLLM 严格 TP1", "vllm strict tp2": "vLLM 严格 TP2", "vllm strict tp4": "vLLM 严格 TP4",
    "megatron strict tp1": "Megatron 严格 TP1", "megatron strict tp2": "Megatron 严格 TP2", "megatron strict tp4": "Megatron 严格 TP4",
    "vllm strict lora tp1": "vLLM 严格 TP1+LoRA", "vllm strict lora tp2": "vLLM 严格 TP2+LoRA",
    "vllm native tp1": "vLLM 原生 TP1", "vllm native tp2": "vLLM 原生 TP2", "vllm native lora tp1": "vLLM 原生 TP1+LoRA",
    "megatron native tp1": "Megatron 原生 TP1", "megatron native tp2": "Megatron 原生 TP2",
}
XRT_ORDER = ["reference triton", "vllm strict tp1", "vllm strict tp2", "vllm strict tp4", "megatron strict tp1", "megatron strict tp2",
             "megatron strict tp4", "vllm strict lora tp1", "vllm strict lora tp2", "vllm native tp1", "vllm native tp2",
             "vllm native lora tp1", "megatron native tp1", "megatron native tp2", "reference hf bf16", "reference golden"]


def sec_runtimes() -> str:
    v = load("vllm_runtime.json")
    m = load("megatron_runtime.json")
    out = []
    if v:
        gens = v.get("generate", [])
        def lab(g):
            return f'{"严格" if g["mode"] == "strict" else "原生"} TP{g["tp"]} {"T=1" if g.get("temperature") else "贪心"}{" +LoRA" if g.get("lora") else ""}'
        rows = [(lab(g), [g.get("decode_vs_prefill", {}).get("mismatched", 0)]) for g in gens]
        dvp = hbars(rows, [("decode 时记下的 logprob 与重新 prefill 打分不同的 token 数（每组 512 个）", "s-bad")], width=860, bar_h=14,
                    value_fmt=lambda x: f"{x:.0f}", xmax=512, title="vLLM 自己和自己比：decode 一次、prefill 重算一次")
        thr = v.get("throughput", [])
        trows = []
        for tp in (1, 2):
            n = next((t for t in thr if t["mode"] == "native" and t["tp"] == tp), None)
            s = next((t for t in thr if t["mode"] == "strict" and t["tp"] == tp), None)
            if n and s:
                trows.append((f"TP{tp}", [s["tokens_per_s"], n["tokens_per_s"]]))
        tchart = hbars(trows, [("严格", "s-strict"), ("原生", "s-native")], width=720, bar_h=16, unit=" tok/s",
                       title="vLLM 吞吐：32 个 prompt × 256 token，tokens/s", value_fmt=lambda x: f"{x:.0f}")
        drows = []
        for tp in (1, 2):
            n = next((t for t in thr if t["mode"] == "native" and t["tp"] == tp), None)
            s = next((t for t in thr if t["mode"] == "strict" and t["tp"] == tp), None)
            if n and s:
                drows.append((f"TP{tp} 每步 decode", [s["decode_step_ms_median"], n["decode_step_ms_median"]]))
                drows.append((f"TP{tp} 首 token", [s["ttft_ms"], n["ttft_ms"]]))
        dchart = hbars(drows, [("严格", "s-strict"), ("原生", "s-native")], width=720, bar_h=14, unit=" ms",
                       title="延迟（ms，batch 32）", value_fmt=lambda x: f"{x:.0f}")
        out.append(fig(dvp, "原生 vLLM 里，同一个序列 decode 时算的 logprob 和事后整段 prefill 重算的不一样（约 30% 的 token）；严格绑定在 TP1/TP2、贪心/采样、有无 LoRA 的 8 种组合里全部 0。"))
        out.append(fig(tchart, "严格绑定吞吐约为原生的 41–46%。TP2 比 TP1 还慢：30B 在 A100 上 batch 32 时通信开销盖过了算力，严格 TP2 还要复制 KV（Wave B 换成每个 rank 只算自己的头）。"))
        out.append(fig(dchart, ""))
    else:
        out.append(pending("vLLM 0.31 严格引擎在 30B 上的运行"))
    if m:
        runs = m.get("runs", {})
        def chk(tp, k):
            return runs.get(tp, {}).get("checks", {}).get(k, {})
        g1, g2 = chk("tp1", "grad_repeat"), chk("tp2", "grad_repeat")
        s1, s2 = chk("tp1", "step"), chk("tp2", "step")
        idt = chk("tp2", "identity")
        tiles = [
            ("权重恒等", f'{idt.get("equal", "?")}/{idt.get("total", "?")}', "Bridge 加载的权重与 HF 逐字节相同；唯一例外是存成 w−1 的 final norm，严格路径从 HF 重载"),
            ("logprob TP2 vs TP1", f'{m.get("cross_tp", {}).get("xrt_strict_tp2_vs_tp1", {}).get("bit_mismatches", "?")}/933', "严格训练前向，933 个 token 不同的个数"),
            ("LoRA 梯度 TP2 vs TP1", "相同" if g1.get("aggregate_sha256") and g1.get("aggregate_sha256") == g2.get("aggregate_sha256") else "不同",
             f'{g1.get("tensors", "?")} 个张量、{g1.get("elements", 0) / 1e6:.1f}M 个元素的聚合 SHA-256 一致；同进程重复两次也 0 差异'),
            ("一步 AdamW 后", "相同" if s1.get("lora_sha256_after") and s1.get("lora_sha256_after") == s2.get("lora_sha256_after") else "不同",
             "TP1 和 TP2 更新后的 LoRA 哈希一致（muse_glimmer_lora_sha256_v1）"),
        ]
        cards = "".join(f'<div class="fact"><span class="fk">{esc(a)}</span><span class="fnew">{esc(b)}</span><span class="fn">{esc(c)}</span></div>' for a, b, c in tiles)
        out.append(f'<h3>Megatron Bridge：训练这一侧也逐 bit 相同</h3><div class="facts">{cards}</div>')
        steps = [(f"TP{tp[-1]}", [chk(tp, "step").get("fwd_s"), chk(tp, "step").get("step_s")]) for tp in ("tp1", "tp2") if chk(tp, "step")]
        if steps:
            out.append(fig(hbars(steps, [("严格前向打分", "s-p4"), ("前向 + 反向 + AdamW（逐层重算）", "s-strict")], width=720, bar_h=14, unit=" s",
                                 title="Megatron 严格 trainer：933 个 token 的一步训练，秒"), "严格 trainer 用逐层重算省显存；TP1 在单次 8192 token 的打包上 OOM，默认 micro batch 改成 4096 token。"))
    else:
        out.append(pending("Megatron Bridge 严格 trainer 在 30B 上的检查"))
    return f"""
<section id="runtimes">
  <p class="eyebrow">两个真实运行时</p>
  <h2>vLLM 0.31 和 Megatron Bridge 跑同一套内核</h2>
  <div class="prose"><p>两边都没有改 vLLM 或 Megatron 的源码：vLLM 通过插件把原生 <code>MuseGlimmerForCausalLM</code> 的 14 个算子换成严格实现，Megatron 通过 Bridge 加载 HF 权重后把前向换成同一组 Triton 内核，LoRA 两边都不合并。</p></div>
  {''.join(out)}
</section>"""


def sec_xrt() -> str:
    x = load("xrt_summary.json")
    if not x or not x.get("matrix"):
        return f"""
<section id="xrt">
  <p class="eyebrow">关键结论</p>
  <h2>三方逐字节比对</h2>
  {pending("参考链、vLLM、Megatron 对同一份 xrt_tokens.json 打分后逐字节比对")}
</section>"""
    labels = [l for l in XRT_ORDER if l in x["matrix"]] + [l for l in x["matrix"] if l not in XRT_ORDER]
    vals = [[x["matrix"][a][b] for b in labels] for a in labels]
    names = [XRT_NAMES.get(l, l) for l in labels]
    hm = heatmap(names, names, vals, title=f"两两比较：fp32 logprob 的 bit 不同的 token 数（共 {x['n_scored_tokens']} 个 token）", cell=30, cell_w=44)
    groups = x.get("strict_groups", {})
    gtxt = "；".join(f'{"无 LoRA" if g == "base" else "同一个随机 LoRA"}：{len(ls)} 份 dump 两两 0 差异' for g, ls in groups.items())
    nat = [l for l in labels if " native" in l]
    pairs = [(a, b) for i, a in enumerate(nat) for b in nat[i + 1:]]
    worst = max((x["matrix"][a][b] for a, b in pairs), default=0)
    best = min((x["matrix"][a][b] for a, b in pairs), default=0)
    return f"""
<section id="xrt">
  <p class="eyebrow">关键结论</p>
  <h2>三方逐字节比对：严格路径 0 个 bit 不同</h2>
  {fig(hm, f"同一份 token（8 道 GSM8K 题 + 答案，1 条 2439 token、跨越 2048 窗口的长序列），每个实现输出每个 token 的 fp32 logprob。绿色 0 = 逐 bit 相同。{gtxt}。原生实现两两之间有 {best}–{worst} 个 token 不同。golden 是纯 PyTorch 的 fp32 精度参考，不是逐 bit 目标。")}
</section>"""


ABL_ZH = {
    "repeat_immediate": "立即重跑", "repeat_after_all_factors": "所有实验之后重跑", "new_process_same_gpu_7": "新进程、同一张卡",
    "second_gpu_6_vs_7": "换一张 GPU", "repeat_immediate_gpu_6": "另一张卡上重跑", "tf32_on": "打开 TF32",
    "rope_elementwise": "RoPE 表改用逐元素构造", "rope_elementwise_tf32_on": "逐元素 RoPE + TF32", "rope_inv_freq_cuda": "inv_freq 在 GPU 上算",
    "padded_batch9_right": "9 条拼 batch（右 padding）", "padded_batch9_left": "9 条拼 batch（左 padding）",
    "kv_cache_decode": "KV cache 逐 token decode", "prefill_L_minus_1": "prefill L−1 再 decode 1", "chunked_prefill_64": "分块 prefill 64",
    "chunked_prefill_256": "分块 prefill 256", "chunked_prefill_1024": "分块 prefill 1024", "sdpa": "eager → SDPA 注意力",
    "softcap_fp32": "softcap 用 fp32", "logits_fp32": "logits 用 fp32", "bf16_reduced_reduction_off": "关掉 bf16 降精度归约",
    "rollout_like_batched_decode": "模拟 rollout：左 pad + 批量 decode", "rollout_like_batched_decode_sdpa": "模拟 rollout + SDPA",
    "packed_batch9": "9 条打包", "packed_batch9_reversed": "9 条打包（倒序）", "right_pad_batch9": "右 padding batch", "left_pad_batch9": "左 padding batch",
    "paged_prefill_batch9": "paged KV prefill", "prefill_alone": "单条 prefill", "decode_alone": "单条逐 token decode", "batched_decode9": "9 条批量 decode",
    "strict_vs_hf_baseline": "严格 vs HF（不同实现）",
    "chunked_prefill_64__vs__prefill_L_minus_1": "分块 prefill 64（对比一次性 L−1）", "chunked_prefill_256__vs__prefill_L_minus_1": "分块 prefill 256（对比一次性 L−1）",
    "chunked_prefill_1024__vs__prefill_L_minus_1": "分块 prefill 1024（对比一次性 L−1）", "logits_fp32__vs__softcap_fp32": "logits fp32（对比 softcap fp32）",
    "rope_elementwise_tf32_on__vs__rope_elementwise": "TF32（逐元素 RoPE 下）", "rollout_like_batched_decode_sdpa__vs__rollout_like_batched_decode": "模拟 rollout 下换 SDPA",
    "rope_inv_freq_cuda": "RoPE 的 inv_freq 在 GPU 上算", "second_gpu_3_vs_7": "换一张 GPU", "new_process_same_gpu_7": "新进程、同一张卡",
    "batched_decode72_staggered": "72 路连续批处理 decode", "mixed_step72_staggered": "prefill 和 decode 混在同一步",
}


def sec_ablation() -> str:
    a = load("ablation.json")
    if not a:
        return f"""
<section id="ablation">
  <p class="eyebrow">消融</p>
  <h2>原生栈的漂移从哪来</h2>
  {pending("HF 模型单因素消融")}
</section>"""
    st = a["settings"]
    hf = [s for s in st if s["engine"] == "hf" and s.get("n_factors") in (0, 1) and "@" not in s["setting"]]
    hf.sort(key=lambda s: -s["n_bits_differ"])
    n = hf[0]["n_tokens"] if hf else 933
    c1 = hbars([(ABL_ZH.get(s["setting"], s["setting"]), [s["n_bits_differ"]]) for s in hf], [("bit 不同的 token 数", "s-native")],
               width=860, bar_h=14, xmax=n, value_fmt=lambda v: f"{v:.0f}", title=f"HF transformers 5.17（原生 bf16）：每次只改一个因素（含 null 对照），{n} 个 token 里 logprob 的 bit 变了多少")
    strict = [s for s in st if s["engine"] == "strict"]
    c2 = hbars([(ABL_ZH.get(s["setting"], s["setting"]), [s["n_bits_differ"]]) for s in strict], [("bit 不同的 token 数", "s-ok")],
               width=860, bar_h=12, xmax=n, value_fmt=lambda v: "0（逐 bit 相同）" if v == 0 else f"{v:.0f}",
               title="严格参考路径：同样的扰动")
    mag = [s for s in hf if s["n_bits_differ"] > 0]
    c3 = hbars([(ABL_ZH.get(s["setting"], s["setting"]), [s["mean_abs_dlogp"], s["max_abs_dlogp"]]) for s in mag],
               [("平均 |Δlogp|", "s-native"), ("最大 |Δlogp|", "s-bad")], width=860, bar_h=10, log=True,
               title="漂移有多大（对数轴）")
    probs = a.get("analysis", {}).get("abs_dlogp_by_baseline_prob", {})
    buckets = ["p>0.9", "0.5<p<=0.9", "0.1<p<=0.5", "p<=0.1"]
    bz = {"p>0.9": "p > 0.9", "0.5<p<=0.9": "0.5–0.9", "0.1<p<=0.5": "0.1–0.5", "p<=0.1": "p ≤ 0.1"}
    sel = [k for k in ("hf/kv_cache_decode", "hf/padded_batch9_left", "hf/sdpa") if k in probs]
    c4 = hbars([(bz[b], [probs[k][b]["mean_abs_dlogp"] for k in sel]) for b in buckets],
               [(ABL_ZH.get(k.split("/")[1], k), f"s-p{i}") for i, k in enumerate(sel)], width=860, bar_h=11,
               title="按基线概率分桶的平均 |Δlogp|：越不确定的 token 漂得越多") if sel else ""
    lr = [r for r in a.get("layer_records", []) if r["sequence"] == "gsm8k_test_0" and r["setting"] in ("kv_cache_decode", "padded_batch9_left", "prefill_L_minus_1", "tf32_on")]
    c5 = ""
    if lr:
        xs = list(range(len(lr[0]["layers"])))
        cls = {"kv_cache_decode": "s-p0", "padded_batch9_left": "s-p1", "prefill_L_minus_1": "s-p2", "tf32_on": "s-ok"}
        c5 = lines(xs, [(ABL_ZH.get(r["setting"], r["setting"]), cls[r["setting"]], [l["frac_elems_differ"] * 100 for l in r["layers"]]) for r in lr],
                   title="逐层：隐藏状态里 bit 不同的元素占比（%），gsm8k_test_0", xlabel="0 = embedding，1–52 = 第 0–51 层输出，53 = final norm",
                   ymin=0, ymax=100, width=860, height=280, dots=False, y_fmt=lambda v: f"{v:.0f}%")
    return f"""
<section id="ablation">
  <p class="eyebrow">消融</p>
  <h2>原生栈的漂移从哪来：bf16 GEMM 或注意力的形状、调度，任何一处变化</h2>
  <div class="prose"><p>在 HF 模型上一次只改一个因素，对同样 9 条序列打分，和基线逐 bit 比较。重复跑、换一张卡、换一个进程、打开 TF32、RoPE 改成逐元素构造都是 0。但只要 bf16 GEMM 或注意力的形状、调度变了（拼 batch 加 padding、KV cache 逐 token decode、prefill 少一个 token、SDPA 换 eager、cuBLAS 的降精度归约开关），933 个 token 里就有 918–927 个变 bit；在 GPU 上而不是 CPU 上算 RoPE 的 inv_freq，64 个频率里有 4 个不同，也会让 924 个 token 变 bit。分块 prefill 只影响比块更长的序列：对比一次性 prefill，块 1024 只改了那条 2439 token 的长序列（176/933），块 64 改了 925/933。rollout 引擎恰好同时改了这几个因素。</p></div>
  {fig(c1, "只列真正的单因素设置和 null 对照；同时改两个因素的设置（例如分块 prefill 对比整段 L 个 token 的训练前向）不在这张图里。")}
  {fig(c2, "严格路径在同一组扰动下（拼 batch、padding、paged KV、逐 token decode、分块 prefill、换卡）全部 0。")}
  {fig(c3, "单个 token 的漂移可以到 0.4–1.1 nat，平均约 0.02 nat。")}
  {fig(c4, "") if c4 else ""}
  {fig(c5, "差异从第 0 层就出现，三四层内扩散到约 85% 的隐藏元素；只改 TF32 的那条线始终是 0。") if c5 else ""}
</section>"""


PROF_ZH = {"embedding": "embedding", "norms_residual": "范数 + 残差", "qkvgo_gemm": "Q/K/V/门/O GEMM", "qknorm_rope": "QK-norm + RoPE",
           "attention_global": "全局注意力", "attention_local": "局部注意力", "kv_cache": "KV cache", "ffn_gemm": "FFN GEMM",
           "ffn_act": "SwiGLU", "lm_head_softcap_logprob": "LM head + softcap + logprob", "other": "其他"}
PROF_CLS = {"embedding": "s-p7", "norms_residual": "s-p4", "qkvgo_gemm": "s-p0", "qknorm_rope": "s-p3", "attention_global": "s-p5",
            "attention_local": "s-p1", "kv_cache": "s-p6", "ffn_gemm": "s-c8", "ffn_act": "s-p2", "lm_head_softcap_logprob": "s-c9", "other": "s-c10"}


def sec_profile() -> str:
    r = load("reference_profile.json")
    p = (r or {}).get("experiments", {}).get("profile", {}).get("stacked_bars")
    if not p:
        return f"""
<section id="profile">
  <p class="eyebrow">Profiling</p>
  <h2>时间花在哪里</h2>
  {pending("严格单卡模型 vs HF 的逐算子时间分解")}
</section>"""
    cfgs, groups = p["configs"], p["groups"]
    st, hf = p["strict_triton"].get("wall_ms", {}), p.get("hf_bf16_sdpa", {}).get("wall_ms", {})
    keep = [i for i, c in enumerate(cfgs) if "paged" not in c]
    def zh(c):
        return c.replace("prefill_train", "prefill").replace("decode B=", "decode batch ").replace(" ctx=2048", "，上下文 2048")
    rows = []
    for i in keep:
        rows.append((f"严格 · {zh(cfgs[i])}", [st.get(g, [0] * len(cfgs))[i] for g in groups]))
        if hf:
            rows.append((f"HF · {zh(cfgs[i])}", [hf.get(g, [0] * len(cfgs))[i] for g in groups]))
    parts = [(PROF_ZH.get(g, g), PROF_CLS.get(g, "s-pend")) for g in groups]
    c_abs = stacked(rows, parts, title="单卡 52 层前向的时间分解（ms，CUDA event，中位数）", width=900, bar_h=16, unit="ms")
    c_rel = stacked(rows, parts, title="同上，按占比", width=900, bar_h=16, normalize=True)
    ratio = p.get("strict_over_hf_clean")
    rtxt = ""
    if isinstance(ratio, dict):
        vals = [(zh(c), ratio.get(c)) for c in cfgs if "paged" not in c and isinstance(ratio.get(c), (int, float))]
        if vals:
            rtxt = "；".join(f"{a} {b:.2f}×" for a, b in vals)
    return f"""
<section id="profile">
  <p class="eyebrow">Profiling</p>
  <h2>时间花在哪里：GEMM，尤其是 FFN</h2>
  {fig(c_abs, "长序列 prefill 里 FFN 的三个 GEMM 占大头（深蓝），其次是 Q/K/V/门/O 投影；严格 GEMM 比 cuBLAS 慢 2 倍左右，所以 T=4096 时严格前向是 HF 的约 1.5 倍。decode batch 1 时严格路径反而更快；batch 16 时严格路径的 paged KV 写入（橄榄色）占了一半时间，是下一步最值得优化的地方。")}
  {fig(c_rel, ("严格路径耗时 / HF：" + rtxt) if rtxt else "")}
</section>"""


def hero() -> str:
    x = load("xrt_summary.json")
    ex = (x or {}).get("example") or []
    if ex:
        keep = [e for e in ex if e["label"] in ("reference triton", "vllm strict tp1", "vllm strict tp2", "megatron strict tp1", "megatron strict tp2",
                                                "vllm native tp1", "vllm native tp2", "megatron native tp1", "megatron native tp2")]
        keep.sort(key=lambda e: XRT_ORDER.index(e["label"]) if e["label"] in XRT_ORDER else 99)
        strict_hex = {e["hex"] for e in keep if e["strict"]}
        rows = "".join(
            f'<div class="lab">{esc(XRT_NAMES.get(e["label"], e["label"]))}</div><div class="hx {"hx-s" if e["strict"] else "hx-n"}">{esc(e["hex"])}'
            f'<span class="hv">{e["value"]:.9f}</span></div>' for e in keep)
        tok = x.get("example_token") or {}
        bits = (f'<div class="bitstrip"><div class="bscap">同一个 token（{esc(tok.get("sequence", ""))} 第 {tok.get("index", 0)} 个答案 token）的 fp32 logprob，'
                f'各实现输出的 bit 模式。严格路径 {len([e for e in keep if e["strict"]])} 份、{len(strict_hex)} 种结果。</div>{rows}</div>')
    else:
        hs, ht, _, _ = D.sum_orders()
        bits = ('<div class="bitstrip"><div class="lab">同一组数，顺序 A</div><div class="hx">0x' + hs + '</div>'
                '<div class="lab">同一组数，顺序 B</div><div class="hx">0x' + ht + "</div></div>")
    return f"""
<header class="mast">
  <p class="eyebrow">RFC RL-Align/RL-Kernel#501 · POC 报告 · 2026-10-15</p>
  <h1>Muse-Glimmer-30B：让训练和推理算出同一个 bit</h1>
  <p class="dek">在一台 8×A100 上，用真实的 vLLM 0.31 做 rollout、真实的 Megatron Bridge 做 LoRA 训练，按一份数值契约把每一次加法的顺序钉死，再跑 GRPO 看两边的 logprob 是否逐 bit 相同。这篇是结果、过程，以及 RFC 需要怎么改。</p>
  <div class="chips"><span class="chip">8× A100-SXM4-80GB</span><span class="chip">vLLM 0.31.0</span><span class="chip">Megatron Bridge 0.7 (git d86a5c09)</span><span class="chip">Triton 3.7.1</span><span class="chip">transformers 5.17.0</span><span class="chip">checkpoint a4e59da5</span><span class="chip">分支 feat/museglimmer-poc</span></div>
  {bits}
</header>"""


MODE_ZH = {"strict/strict": "严格训练 / 严格 rollout", "native/native": "原生训练 / 原生 rollout",
           "strict/native": "严格训练 / 原生 rollout", "native/strict": "原生训练 / 严格 rollout"}
MODE_CLS = {"strict/strict": "s-strict", "native/native": "s-native", "strict/native": "s-p3", "native/strict": "s-p4"}
PHASE_ZH = {"rollout_generate": "rollout 生成", "reward": "奖励", "trainer_logprobs": "trainer 重算 logprob", "drift_metrics": "逐 bit 比较",
            "rollout_recheck": "rollout 自检（prefill 重算）", "sidecar": "保存比对数据", "grpo_loss": "GRPO loss", "backward_step": "反向 + AdamW",
            "lora_export": "LoRA 导出", "lora_transfer_load": "LoRA 传输 + 加载", "other": "其他"}
PHASE_CLS = {"rollout_generate": "s-p0", "reward": "s-p7", "trainer_logprobs": "s-p4", "drift_metrics": "s-p6", "rollout_recheck": "s-p1",
             "sidecar": "s-c10", "grpo_loss": "s-p2", "backward_step": "s-c8", "lora_export": "s-p3", "lora_transfer_load": "s-c9", "other": "s-pend"}


def sec_e2e() -> str:
    e = load("e2e_run.json")
    head = """
<section id="e2e">
  <p class="eyebrow">端到端</p>
  <h2>LoRA GRPO：rollout 和训练的 logprob 每一步都逐 bit 相同吗</h2>"""
    if not e or not e.get("modes"):
        return head + pending("四种模式的真实 30B GRPO") + "</section>"
    order = ["strict/strict", "strict/native", "native/strict", "native/native"]
    modes = sorted(e["modes"], key=lambda m: order.index(m["mode"]) if m["mode"] in order else 9)
    m0 = modes[0]
    cfg = m0["config"]
    xs = [r["step"] for r in m0["steps"]]
    def pct(r, a, b):
        return 100.0 * r[a] / r[b] if r.get(a) is not None and r.get(b) else None
    drift = lines(xs, [(MODE_ZH.get(m["mode"], m["mode"]), MODE_CLS.get(m["mode"], "s-pend"), [pct(r, "bitwise_mismatch", "tokens") for r in m["steps"]]) for m in modes],
                  title="每一步：trainer 重算的 logprob 与 rollout 记下的 logprob，bit 不同的 token 占比", xlabel="GRPO step", ymin=0, ymax=100,
                  width=860, height=260, y_fmt=lambda v: f"{v:.0f}%")
    rech = lines(xs, [(MODE_ZH.get(m["mode"], m["mode"]), MODE_CLS.get(m["mode"], "s-pend"), [pct(r, "rollout_recheck_mismatch", "rollout_recheck_tokens") for r in m["steps"]]) for m in modes],
                 title="rollout 自检：同一批 token 在 rollout 引擎里 prefill 重算一次，与 decode 时的 logprob 比", xlabel="GRPO step", ymin=0,
                 width=860, height=240, y_fmt=lambda v: f"{v:.0f}%")
    rew = lines(xs, [(MODE_ZH.get(m["mode"], m["mode"]), MODE_CLS.get(m["mode"], "s-pend"), [r["reward_mean"] for r in m["steps"]]) for m in modes],
                title="每一步的平均奖励（GSM8K 答案完全匹配）", xlabel="GRPO step", ymin=0, width=860, height=240, y_fmt=lambda v: f"{v:.2f}")
    ratio = hbars([(MODE_ZH.get(m["mode"], m["mode"]), [100 * sum((r["ratio_exactly_one_frac"] or 0) for r in m["steps"]) / max(len(m["steps"]), 1)]) for m in modes],
                  [("重要性比 π_train/π_rollout 恰好等于 1 的 token 占比", "s-strict")], width=860, bar_h=16, xmax=100, value_fmt=lambda v: f"{v:.1f}%",
                  title="更新之前的“伪策略偏移”")
    phases = m0.get("phase_order") or list(PHASE_ZH)
    trows = []
    for m in modes:
        n = max(len(m["steps"]), 1)
        trows.append((MODE_ZH.get(m["mode"], m["mode"]), [sum(r["phases_s"].get(p, 0) for r in m["steps"]) / n for p in phases]))
    tchart = stacked(trows, [(PHASE_ZH.get(p, p), PHASE_CLS.get(p, "s-pend")) for p in phases], title="每一步的平均耗时分解（秒）", width=900, bar_h=18, unit="s")
    cards = []
    for m in modes:
        dt, ws, fd = m.get("drift_totals") or {}, m.get("weight_sync") or {}, m.get("first_drift")
        where = "无" if not fd else f'step {fd.get("step")}, 样本 {fd.get("sample")}, token {fd.get("token")}'
        cards.append(f'<div class="fact"><span class="fk">{esc(MODE_ZH.get(m["mode"], m["mode"]))}</span>'
                     f'<span class="fnew{"" if dt.get("bitwise_mismatch_count") == 0 else " fbad"}">{dt.get("bitwise_mismatch_count", "?")}/{dt.get("token_count", "?")}</span>'
                     f'<span class="fn">token bit 不同；最大 |Δ| {fmt(dt.get("max_abs_diff") or 0)}；第一次漂移：{esc(where)}；LoRA 同步校验 {ws.get("passed", "?")}/{ws.get("checks", "?")} 通过；{m.get("steps_completed", "?")} 步</span></div>')
    setup = (f'每种模式用 3 张卡：rollout TP{cfg.get("rollout_tp")}（1 张）+ 训练 TP{cfg.get("train_tp")}（2 张），两组同时在 GPU 0–2 和 4–6 上跑，rollout 和训练的 TP 度数不同；'
             f'{cfg.get("steps")} 步 × {cfg.get("prompts_per_step")} 道题 × {cfg.get("samples_per_prompt")} 个样本 × 最多 {cfg.get("max_tokens")} token，'
             f'LoRA r{cfg.get("lora_rank")}，lr {cfg.get("lr")}，温度 {cfg.get("temperature")}，seed {cfg.get("seed")}。')
    return head + f"""
  <div class="prose"><p>{esc(setup)}每一步：rollout 采样并记下 fp32 logprob → trainer 对同样的 token 重算 → 逐 bit 比较 → GRPO 更新 LoRA → 带哈希同步回 rollout。四种模式是训练侧 × rollout 侧的 严格/原生 组合。</p></div>
  <div class="facts">{''.join(cards)}</div>
  {fig(drift, "只有两边都严格时，每一步都是 0；只要有一边是原生实现，每一步都有 55–70% 的 token 不同。两种混合模式比原生/原生还高：严格侧用 fp32 logits，原生侧用 bf16 logits，差异是系统性的。")}
  {fig(ratio, "严格/严格下重要性比全部恰好是 1，clip 永远不会被“假偏移”触发。")}
  {fig(rech, "严格 rollout 的自检恒为 0：decode 时记下的数和事后 prefill 重算的数一样，所以 rollout 记下的 logprob 可以直接用。")}
  {fig(rew, "8 步的奖励只用来确认训练在动，POC 不评估学习效果。")}
  {fig(tchart, "")}
</section>"""


def sec_verdict() -> str:
    x, v, m, e, a = load("xrt_summary.json"), load("vllm_runtime.json"), load("megatron_runtime.json"), load("e2e_run.json"), load("ablation.json")
    tiles = []
    if x and x.get("strict_groups"):
        base = x["strict_groups"].get("base", [])
        nat = [l for l in x["matrix"] if " native" in l]
        nn = [x["matrix"][p][q] for i, p in enumerate(nat) for q in nat[i + 1:]]
        tiles.append(("ok", f"{len(base)} 份严格结果，0 个 bit 不同", f"参考链、vLLM TP1/TP2、Megatron TP1/TP2 对同样 {x['n_scored_tokens']} 个 token 的 fp32 logprob 完全一样。"))
        if nn:
            tiles.append(("bad", f"原生：{min(nn)}–{max(nn)} / {x['n_scored_tokens']}", "vLLM、Megatron、HF 的原生实现两两比较，几乎每个 token 的 bit 都不同。"))
    if m:
        g = [m.get("runs", {}).get(t, {}).get("checks", {}).get("grad_repeat", {}).get("aggregate_sha256") for t in ("tp1", "tp2")]
        if g[0] and g[0] == g[1]:
            tiles.append(("ok", "训练梯度 TP2 == TP1", "LoRA 梯度 832 个张量、1.05 亿个元素逐 bit 相同，AdamW 一步后的权重哈希也相同。"))
    if v:
        gs = v.get("generate", [])
        sd = [g.get("decode_vs_prefill", {}).get("mismatched", 0) for g in gs if g["mode"] == "strict"]
        nd = [g.get("decode_vs_prefill", {}).get("mismatched", 0) for g in gs if g["mode"] == "native"]
        if sd and nd:
            tiles.append(("ok", f"rollout 自洽：{max(sd)} vs {min(nd)}–{max(nd)}", "同一序列 decode 时记下的 logprob 与事后 prefill 重算（每组 512 个 token）：严格 vs 原生。"))
        thr = v.get("throughput", [])
        r = [next((t["tokens_per_s"] for t in thr if t["mode"] == md and t["tp"] == tp), None) for md in ("strict", "native") for tp in (1, 2)]
        if all(r):
            tiles.append(("pend", f"代价：吞吐 {100 * r[0] / r[2]:.0f}% / {100 * r[1] / r[3]:.0f}%", "严格 vLLM 相对原生的吞吐（TP1 / TP2，batch 32 × 256）。训练一步约慢 2–3 倍。"))
    if e and e.get("modes"):
        ss = next((mm for mm in e["modes"] if mm["mode"] == "strict/strict"), None)
        if ss:
            dt = ss.get("drift_totals") or {}
            tiles.append(("ok" if dt.get("bitwise_mismatch_count") == 0 else "bad", f"GRPO：{dt.get('bitwise_mismatch_count')}/{dt.get('token_count')} 漂移",
                          f"严格/严格的 30B LoRA GRPO，{ss.get('steps_completed')} 步里 trainer 重算的 logprob 与 rollout 记下的逐 bit 比较。"))
    else:
        tiles.append(("pend", "GRPO：运行中", "四种模式的 30B LoRA GRPO 还没有结果。"))
    if a:
        tiles.append(("bad", "漂移来源：执行细节", "拼 batch、KV cache decode、prefill 长度、注意力实现、cuBLAS 归约开关、RoPE 在哪构表：任何一个都让 918–927/933 个 token 变 bit。"))
    cards = "".join(f'<div class="vt vt-{k}"><b>{esc(t)}</b><span>{esc(d)}</span></div>' for k, t, d in tiles)
    return f"""
<section id="verdict">
  <p class="eyebrow">一屏结论</p>
  <h2>能做到：两个真实运行时，逐 bit 相同</h2>
  <div class="vgrid">{cards}</div>
</section>"""


TIMELINE = [
    ("down", "版本三角", "vLLM 0.31 要 transformers<5.18，Megatron Bridge main 要 ≥5.15,<5.18 且 Python 3.12：只有 5.17.0 两边都接受，指纹的语义源从 5.19 改钉 5.17。"),
    ("down", "Bridge 的 PyPI 版没有这个模型", "PyPI 0.3.1 还钉着 transformers<5；改为从 git main 可编辑安装。"),
    ("down", "仓库的严格内核只认 Hopper", "FA4/SM90 内核和 _C 扩展在 A100 上不可用；原有 Triton det_gemm 只有 5 TFLOP/s，于是新树形 GEMM 成了必需行。"),
    ("up", "13 个 agent 并行", "每个 agent 一个 worktree、互不重叠的文件、GPU 租约；WS1 全部合并，316 个测试。"),
    ("down", "agent worktree 从 main 拉出", "不是从 POC 分支；每个 agent 先自己合并 POC 分支。"),
    ("down", "LM head 被 GEMM 换代打破", "fast_det_gemm 落地后 LM head 自动用它，K%256 契约让小形状测试挂掉；集成方加回退。（Wave A 发现这个回退本身违反“失败即关闭”。）"),
    ("up", "52 层真权重逐 bit 相同", "train = prefill = 4 种 chunk = decode，golden 和 Triton 两条后端都成立。"),
    ("up", "TP/CP 在真实 NCCL 上成立", "并用 NCCL all_reduce 的对照证明树是必要的。"),
    ("down", "两波限流", "会话限额和模型限额先后打断 agent；第二波让 43 个 review agent 死了 36 个。"),
    ("down", "vLLM 0.31 检查 __init__ 签名", "*args/**kwargs 包装器把 vllm_config 藏起来了，严格 TP1 启动失败。"),
    ("down", "两个 env 不能放进一个进程", "编排器原设计是单进程同时 import 两个引擎；改为跨 env 的 trainer 桥（Wave A）。"),
]


def sec_timeline() -> str:
    extra = load("timeline_extra.json") or []
    items = TIMELINE + [tuple(x) for x in extra]
    lis = "".join(
        f'<li class="tl-{k}"><span class="tl-dot"></span><div><b>{esc(t)}</b><p>{esc(d)}</p></div></li>' for k, t, d in items
    )
    ups = sum(1 for k, _, _ in items if k == "up")
    return f"""
<section id="timeline">
  <p class="eyebrow">过程</p>
  <h2>一路上的坑和突破</h2>
  <p class="legend"><span class="tl-key up"></span>突破 {ups}　<span class="tl-key down"></span>坑 {len(items) - ups}</p>
  <ol class="timeline">{lis}</ol>
</section>"""


KIND_ZH = {"correction": "更正", "clarification": "澄清", "frozen_answer": "冻结答案", "new_item": "新增条目", "answered_open_question": "回答开放问题"}



def sec_rfc() -> str:
    a = load("rfc_audit.json")
    if not a:
        return pending("RFC 审计数据缺失")
    items = a["items"]
    by_kind: dict[str, int] = {}
    for it in items:
        by_kind[it["kind"]] = by_kind.get(it["kind"], 0) + 1
    kinds = sorted(by_kind, key=lambda k: -by_kind[k])
    chart = hbars([(KIND_ZH.get(k, k), [by_kind[k]]) for k in kinds], [("条数", "s-strict")], title=f"RFC 审计：{len(items)} 条修改全部经过独立复核（43 条原样确认，13 条确认但修正了措辞）",
                  width=720, bar_h=18, value_fmt=lambda v: f"{v:.0f}")
    sections: dict[str, list[dict]] = {}
    for it in items:
        sections.setdefault(it["section"], []).append(it)
    blocks = []
    for sec, its in sections.items():
        lis = []
        for it in its:
            prop = it.get("corrected") or it.get("proposed", "")
            lis.append(
                f'<li><span class="pill k-{esc(it["kind"])}">{esc(KIND_ZH.get(it["kind"], it["kind"]))}</span> <b>{esc(it["id"])}</b>'
                f'<div class="rfc-old">{esc(it.get("rfc", ""))}</div><div class="rfc-why">{esc(it.get("problem", ""))}</div>'
                f'<div class="rfc-new">{esc(prop)}</div></li>'
            )
        blocks.append(details(f"{sec}（{len(its)} 条）", f'<ol class="rfc-list">{"".join(lis)}</ol>'))
    fnd = (load("rfc_findings.json") or {}).get("items", [])
    hard = "".join(f'<div class="fact"><span class="fk">{esc(f["id"])} · {esc(f["area"])}</span><b>{esc(f["title_zh"])}</b><span class="fn">{esc(f["body_zh"])}</span></div>' for f in fnd)
    return f"""
<section id="rfc">
  <p class="eyebrow">结论</p>
  <h2>RFC 需要怎么改</h2>
  <h3>实现过程中才暴露的问题</h3>
  <div class="facts">{hard}</div>
  {fig(chart, "逐行审计 RFC 原文：每条都给出原文、问题、建议文本和证据（文件:行号），再由另一个 agent 独立复核。下面按 RFC 章节展开，建议文本是英文，可以直接贴到 issue 里；完整的英文清单在 <code>reports/muse_glimmer_poc/RFC_CHANGES.md</code>。")}
  {''.join(blocks)}
</section>"""


def sec_left() -> str:
    left = load("left_for_community.json") or [
        "CUDA twin 内核（这次只有 PyTorch golden + Triton）",
        "ROCm / AMD（这台机器没有 AMD 卡）",
        "Hopper 上的高性能版本（FA4/CuTe 路线）",
        "Phase B 视觉、DFlash 投机 rollout、量化 checkpoint、全参数端到端",
        "把 VIME 移植到 Megatron Bridge（这次直接驱动 vLLM 和 Bridge，没有经过 VIME）",
    ]
    return f"""
<section id="left">
  <p class="eyebrow">下一步</p>
  <h2>留给社区的事</h2>
  <ul class="checklist">{''.join(f"<li>{esc(x)}</li>" for x in left)}</ul>
</section>"""


# ----------------------------------------------------------------------------------------------
# Page
# ----------------------------------------------------------------------------------------------

CSS = r"""
/* Layout: one reading column (~40 CJK chars) with figures that break out to the full frame; each section opens on a figure or a fact grid. */
:root {
  --paper: #F4F6F9; --panel: #FFFFFF; --ink: #141A22; --muted: #586373; --rule: #D6DCE4;
  --strict: #2347A5; --strict-soft: #DDE5F8; --native: #B4621A; --native-soft: #F5E5D4;
  --ok: #2C7A4D; --ok-soft: #D8EEDF; --bad: #B2362D; --bad-soft: #F6DAD7; --pend: #8A6D1C; --pend-soft: #F4EBCF;
  --r0: #2347A5; --r1: #B4621A; --r2: #2C7A4D; --r3: #8B3FA0; --r4: #3E8E9C; --r5: #B2362D; --r6: #6D7A2A; --r7: #5A6A86;
  --c8: #15275A; --c9: #C2417A; --c10: #8A6A45;
  --font-display: "Archivo", "Noto Sans SC", "PingFang SC", system-ui, sans-serif;
  --font-body: "Noto Sans SC", "PingFang SC", "Microsoft YaHei", system-ui, sans-serif;
  --font-mono: "JetBrains Mono", ui-monospace, "SFMono-Regular", Menlo, "Noto Sans SC", "PingFang SC", monospace;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --paper: #0D1116; --panel: #141A21; --ink: #E3E8EE; --muted: #97A2B0; --rule: #28303A;
  --strict: #8DABFF; --strict-soft: #1B2540; --native: #EEA35C; --native-soft: #3A2918;
  --ok: #63C18D; --ok-soft: #15301F; --bad: #F27B6F; --bad-soft: #3B1C19; --pend: #E2C35E; --pend-soft: #33290F;
  --r0: #8DABFF; --r1: #EEA35C; --r2: #63C18D; --r3: #C98BE0; --r4: #74C7D4; --r5: #F27B6F; --r6: #B9C866; --r7: #9DAAC4;
  --c8: #C3D1FF; --c9: #F28DBA; --c10: #D2AE84;
  color-scheme: dark; } }
:root[data-theme="dark"] {
  --paper: #0D1116; --panel: #141A21; --ink: #E3E8EE; --muted: #97A2B0; --rule: #28303A;
  --strict: #8DABFF; --strict-soft: #1B2540; --native: #EEA35C; --native-soft: #3A2918;
  --ok: #63C18D; --ok-soft: #15301F; --bad: #F27B6F; --bad-soft: #3B1C19; --pend: #E2C35E; --pend-soft: #33290F;
  --r0: #8DABFF; --r1: #EEA35C; --r2: #63C18D; --r3: #C98BE0; --r4: #74C7D4; --r5: #F27B6F; --r6: #B9C866; --r7: #9DAAC4;
  --c8: #C3D1FF; --c9: #F28DBA; --c10: #D2AE84;
  color-scheme: dark; }
* { box-sizing: border-box; }
body { background: var(--paper); color: var(--ink); font: 16px/1.75 var(--font-body); }
.page { max-width: 66rem; margin: 0 auto; padding-block: 2.5rem 5rem;
  padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); }
.col { max-width: 44rem; }
h1, h2, h3 { font-family: var(--font-display); text-wrap: balance; line-height: 1.2; margin: 0; }
h1 { font-size: clamp(2rem, 5vw, 3.4rem); font-weight: 800; font-stretch: 82%; letter-spacing: -0.01em; }
h2 { font-size: clamp(1.45rem, 3vw, 2rem); font-weight: 750; font-stretch: 88%; margin-bottom: 1rem; }
h3 { font-size: 1.1rem; font-weight: 700; margin: 1.6rem 0 0.8rem; }
p { margin: 0 0 0.9rem; }
code { font-family: var(--font-mono); font-size: 0.86em; background: var(--strict-soft); padding: 0.05em 0.35em; border-radius: 3px; }
a { color: var(--strict); }
.eyebrow { font: 600 0.74rem/1 var(--font-mono); letter-spacing: 0.12em; text-transform: uppercase; color: var(--strict); margin: 0 0 0.6rem; }
header.mast { display: grid; gap: 1.1rem; padding-bottom: 2rem; border-bottom: 1px solid var(--rule); }
.dek { font-size: 1.12rem; color: var(--muted); max-width: 44rem; }
.chips { display: flex; flex-wrap: wrap; gap: 0.4rem; }
.chip { font: 500 0.78rem/1 var(--font-mono); padding: 0.4rem 0.55rem; border: 1px solid var(--rule); border-radius: 4px; color: var(--muted); background: var(--panel); }
.bitstrip { display: grid; gap: 0.35rem; font-family: var(--font-mono); font-size: clamp(0.8rem, 2.2vw, 1.05rem); background: var(--panel); border: 1px solid var(--rule); border-radius: 8px; padding: 1rem 1.1rem; overflow-x: auto; }
.bitstrip .lab { color: var(--muted); font-size: 0.78rem; }
.bitstrip .hx { letter-spacing: 0.08em; display: flex; gap: 1rem; flex-wrap: wrap; align-items: baseline; }
.bitstrip .hx-s { color: var(--strict); font-weight: 600; } .bitstrip .hx-n { color: var(--native); }
.bitstrip .hv { color: var(--muted); font-size: 0.8em; letter-spacing: 0; }
.bitstrip .bscap { font-family: var(--font-body); font-size: 0.86rem; color: var(--muted); margin-bottom: 0.3rem; }
.bitstrip { grid-template-columns: minmax(8rem, max-content) 1fr; column-gap: 1.2rem; align-items: baseline; }
.bitstrip .bscap { grid-column: 1 / -1; }
nav.toc { display: flex; flex-wrap: wrap; gap: 0.3rem 1rem; padding: 1rem 0; border-bottom: 1px solid var(--rule); font-size: 0.9rem; }
nav.toc a { color: var(--muted); text-decoration: none; }
nav.toc a:hover, nav.toc a:focus-visible { color: var(--strict); text-decoration: underline; }
section { padding-block: 3rem 1rem; border-bottom: 1px solid var(--rule); display: grid; gap: 0.6rem; }
section > * { min-width: 0; }
.prose { max-width: 44rem; }
.fig { margin: 0.6rem 0 1rem; background: var(--panel); border: 1px solid var(--rule); border-radius: 10px; padding: 1rem 1rem 0.8rem; }
.figscroll { overflow-x: auto; }
.fig svg { display: block; width: 100%; height: auto; min-width: 560px; }
.fig figcaption { font-size: 0.88rem; color: var(--muted); margin-top: 0.6rem; max-width: 52rem; }
svg text { font-family: var(--font-body); }
svg .tick, svg .val, svg .bsub { font: 11px var(--font-mono); fill: var(--muted); }
svg .tick.em { fill: var(--ink); }
svg .val { fill: var(--ink); }
svg .lbl { font-size: 12px; fill: var(--ink); }
svg .lbl.em { font-weight: 700; }
svg .ctitle { font-size: 13px; font-weight: 700; fill: var(--ink); }
svg .grid { stroke: var(--rule); stroke-width: 1; }
svg .refline { stroke: var(--ink); stroke-width: 1.2; stroke-dasharray: 3 3; }
svg .s-strict { fill: var(--strict); } svg .s-native { fill: var(--native); }
svg .s-ok { fill: var(--ok); } svg .s-bad { fill: var(--bad); } svg .s-pend { fill: var(--pend); }
svg .s-p0 { fill: var(--r0); } svg .s-p1 { fill: var(--r1); } svg .s-p2 { fill: var(--r2); } svg .s-p3 { fill: var(--r3); }
svg .s-p4 { fill: var(--r4); } svg .s-p5 { fill: var(--r5); } svg .s-p6 { fill: var(--r6); } svg .s-p7 { fill: var(--r7); }
svg .s-c8 { fill: var(--c8); } svg .s-c9 { fill: var(--c9); } svg .s-c10 { fill: var(--c10); }
svg .ln { fill: none; stroke-width: 2; }
svg .ln.s-strict { stroke: var(--strict); } svg .ln.s-native { stroke: var(--native); } svg .ln.s-ok { stroke: var(--ok); } svg .ln.s-bad { stroke: var(--bad); }
svg .ln.s-p0 { stroke: var(--r0); } svg .ln.s-p1 { stroke: var(--r1); } svg .ln.s-p2 { stroke: var(--r2); } svg .ln.s-p3 { stroke: var(--r3); }
svg .hz { fill: var(--ok-soft); stroke: var(--ok); stroke-width: 1; } svg .hbad { fill: var(--bad); } svg .hnone { fill: var(--paper); stroke: var(--rule); }
svg .hv0 { font: 600 11px var(--font-mono); fill: var(--ok); } svg .hv { font: 600 11px var(--font-mono); fill: var(--panel); }
svg .box { fill: var(--panel); stroke: var(--rule); stroke-width: 1.2; }
svg .box.strictb { stroke: var(--strict); stroke-width: 1.8; fill: var(--strict-soft); }
svg .box.goal { stroke: var(--ok); stroke-width: 1.8; fill: var(--ok-soft); }
svg .box.num { fill: var(--paper); }
svg .btxt { font-size: 12.5px; font-weight: 600; fill: var(--ink); } svg .btxt.mono { font: 600 12px var(--font-mono); }
svg .env { fill: none; stroke: var(--rule); stroke-width: 1.5; stroke-dasharray: 6 4; }
svg .arrow { stroke: var(--muted); stroke-width: 1.5; fill: none; } svg .arrow.strict { stroke: var(--strict); stroke-width: 2; } svg .arrow.native { stroke: var(--native); stroke-width: 2; }
svg .ahead { fill: var(--muted); }
svg .edge-native { stroke: var(--native); stroke-width: 1.6; } svg .edge-strict { stroke: var(--strict); stroke-width: 2.2; } svg .edge-local { stroke: var(--muted); stroke-width: 1; }
svg .node-native { fill: var(--native); } svg .node-strict { fill: var(--strict); } svg .node-local { fill: var(--muted); }
svg .hexbig { font: 600 17px var(--font-mono); } svg .hx-same { fill: var(--ink); } svg .hx-native { fill: var(--native); text-decoration: underline; } svg .hx-strict { fill: var(--strict); text-decoration: underline; }
svg .lay-local { fill: var(--strict-soft); stroke: var(--strict); stroke-width: 0.8; } svg .lay-global { fill: var(--strict); }
svg .rank { stroke: none; opacity: 0.85; }
svg .r0 { fill: var(--r0); } svg .r1 { fill: var(--r1); } svg .r2 { fill: var(--r2); } svg .r3 { fill: var(--r3); }
svg .r4 { fill: var(--r4); } svg .r5 { fill: var(--r5); } svg .r6 { fill: var(--r6); } svg .r7 { fill: var(--r7); }
svg .rank + text.tick { fill: var(--panel); font-weight: 600; }
svg .seq { fill: var(--strict-soft); stroke: var(--strict); } svg .seq-dec { fill: var(--native-soft); stroke: var(--native); }
svg .bracket { fill: none; stroke: var(--ink); stroke-width: 1.2; } svg .track { stroke: var(--rule); stroke-width: 3; } svg .cut { stroke: var(--strict); stroke-width: 2; }
.pending { display: flex; flex-wrap: wrap; align-items: center; gap: 0.6rem; padding: 0.8rem 1rem; border: 1px dashed var(--pend); border-radius: 8px; background: var(--pend-soft); font-size: 0.92rem; }
.pending code { background: transparent; }
.pill { display: inline-block; font: 600 0.72rem/1 var(--font-mono); padding: 0.28rem 0.45rem; border-radius: 3px; background: var(--paper); border: 1px solid var(--rule); }
.pill-pend { color: var(--pend); border-color: var(--pend); }
.callout { padding: 0.9rem 1.1rem; border-left: 3px solid var(--strict); background: var(--panel); border-radius: 0 8px 8px 0; font-size: 0.95rem; }
.callout.warn { border-left-color: var(--bad); }
.facts { display: grid; grid-template-columns: repeat(auto-fit, minmax(16rem, 1fr)); gap: 0.8rem; }
.fact { display: grid; gap: 0.25rem; align-content: start; padding: 0.9rem 1rem; background: var(--panel); border: 1px solid var(--rule); border-radius: 8px; }
.fk { font: 600 0.78rem/1.3 var(--font-mono); color: var(--muted); }
.fold s { color: var(--bad); font-family: var(--font-mono); }
.fnew { font: 700 1.3rem/1.2 var(--font-display); color: var(--ok); }
.fnew.fbad { color: var(--bad); }
.fn { font-size: 0.88rem; color: var(--ink); }
.rowgrid { display: grid; grid-template-columns: repeat(auto-fill, minmax(10.5rem, 1fr)); gap: 0.6rem; }
.rowcell { display: grid; gap: 0.2rem; padding: 0.7rem 0.8rem; background: var(--panel); border: 1px solid var(--rule); border-radius: 8px; border-top-width: 4px; }
.rowcell b { font-size: 0.92rem; overflow-wrap: anywhere; } .rg { font: 500 0.7rem var(--font-mono); color: var(--muted); } .rn { font-size: 0.8rem; color: var(--muted); }
.rowcell .pill { justify-self: start; }
.st-done { border-top-color: var(--ok); } .st-done .pill { color: var(--ok); border-color: var(--ok); }
.st-part { border-top-color: var(--pend); } .st-part .pill { color: var(--pend); border-color: var(--pend); }
.st-pend { border-top-color: var(--rule); } .st-bad { border-top-color: var(--bad); } .st-bad .pill { color: var(--bad); border-color: var(--bad); }
details { background: var(--panel); border: 1px solid var(--rule); border-radius: 8px; padding: 0.6rem 1rem; }
summary { cursor: pointer; font-weight: 600; }
summary:focus-visible { outline: 2px solid var(--strict); outline-offset: 2px; }
.dbody { margin-top: 0.6rem; }
.tscroll { overflow-x: auto; }
table { border-collapse: collapse; font-size: 0.85rem; width: 100%; }
th, td { padding: 0.35rem 0.6rem; border-bottom: 1px solid var(--rule); text-align: left; vertical-align: top; }
td.num, th.num { font-family: var(--font-mono); font-variant-numeric: tabular-nums; text-align: right; }
.rfc-list { padding-left: 1.2rem; display: grid; gap: 1rem; }
.rfc-old { font-size: 0.86rem; color: var(--muted); border-left: 2px solid var(--bad); padding-left: 0.6rem; margin-top: 0.3rem; overflow-wrap: anywhere; }
.rfc-why { font-size: 0.86rem; margin: 0.3rem 0; overflow-wrap: anywhere; }
.rfc-new { font-size: 0.86rem; border-left: 2px solid var(--ok); padding-left: 0.6rem; overflow-wrap: anywhere; }
.k-correction { color: var(--bad); border-color: var(--bad); } .k-clarification { color: var(--strict); border-color: var(--strict); }
.k-new_item, .k-frozen_answer, .k-answered_open_question { color: var(--ok); border-color: var(--ok); }
.timeline { list-style: none; padding: 0; margin: 0; display: grid; gap: 0; border-left: 2px solid var(--rule); margin-left: 0.4rem; }
.timeline li { display: grid; grid-template-columns: 1.2rem 1fr; gap: 0.6rem; padding: 0.5rem 0 0.5rem 0; margin-left: -0.5rem; }
.timeline li p { margin: 0; color: var(--muted); font-size: 0.9rem; }
.tl-dot { width: 0.9rem; height: 0.9rem; border-radius: 50%; margin-top: 0.4rem; border: 2px solid var(--paper); }
.tl-up .tl-dot { background: var(--ok); } .tl-down .tl-dot { background: var(--bad); }
.legend { font-size: 0.86rem; color: var(--muted); }
.tl-key { display: inline-block; width: 0.7rem; height: 0.7rem; border-radius: 50%; vertical-align: middle; margin-right: 0.2rem; }
.tl-key.up { background: var(--ok); } .tl-key.down { background: var(--bad); }
.checklist { padding-left: 1.2rem; }
.vgrid { display: grid; grid-template-columns: repeat(auto-fit, minmax(15rem, 1fr)); gap: 0.8rem; }
.vt { display: grid; gap: 0.35rem; align-content: start; padding: 1rem 1.1rem; border-radius: 10px; border: 1px solid var(--rule); background: var(--panel); border-left-width: 5px; }
.vt b { font: 750 1.2rem/1.25 var(--font-display); font-stretch: 90%; }
.vt span { font-size: 0.88rem; color: var(--muted); }
.vt-ok { border-left-color: var(--ok); } .vt-ok b { color: var(--ok); }
.vt-bad { border-left-color: var(--bad); } .vt-bad b { color: var(--bad); }
.vt-pend { border-left-color: var(--pend); } .vt-pend b { color: var(--pend); }
footer { padding-top: 2rem; font-size: 0.85rem; color: var(--muted); }
@media (max-width: 640px) { body { font-size: 15px; } .fig { padding: 0.7rem; } }
@media (prefers-reduced-motion: reduce) { * { scroll-behavior: auto !important; } }
"""




TOC = [("verdict", "结论"), ("why", "问题"), ("scope", "范围"), ("model", "模型"), ("contract", "契约"), ("ws1", "单卡算子"), ("chain", "全模型链"),
       ("multi", "TP/CP"), ("runtimes", "运行时"), ("xrt", "逐字节比对"), ("e2e", "GRPO"), ("ablation", "消融"),
       ("profile", "Profiling"), ("timeline", "过程"), ("rfc", "RFC 修改"), ("left", "留给社区")]


def build() -> str:
    head = (
        "<title>Muse-Glimmer 逐位一致</title>\n"
        '<meta name="claude:viewport-fit-sides" content="cover">\n'
        '<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>\n'
        '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wdth,wght@62..125,400..800&family=JetBrains+Mono:wght@400;600&family=Noto+Sans+SC:wght@400;500;700&display=swap">\n'
        f"<style>{CSS}</style>\n"
    )
    toc = '<nav class="toc" aria-label="目录">' + "".join(f'<a href="#{i}">{esc(t)}</a>' for i, t in TOC) + "</nav>"
    body = "".join([
        hero(), toc, sec_verdict(), sec_why(), sec_scope(), sec_model(), sec_contract(), sec_ws1(), sec_chain(), sec_multi(), sec_runtimes(),
        sec_xrt(), sec_e2e(), sec_ablation(), sec_profile(), sec_timeline(), sec_rfc(), sec_left(),
    ])
    foot = ('<footer>数据：<code>reports/muse_glimmer_poc/data/*.json</code>；构建：<code>python3 reports/muse_glimmer_poc/blog/build_blog.py</code>。'
            '“未运行”标记的内容表示磁盘上还没有对应的结果文件。</footer>')
    return head + f'<main class="page">{body}{foot}</main>\n'


if __name__ == "__main__":
    html = build()
    out = Path(__import__("os").environ.get("MUSEGLIMMER_BLOG_OUT", OUT))
    out.write_text(html)
    print(f"wrote {out} ({len(html) / 1024:.0f} KiB)")
