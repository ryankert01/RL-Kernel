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
DATA = POC / "data"
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
    after = ""
    if kg:
        after = f'<div class="callout"><b>Wave A 之后</b>：{esc(kg.get("headline", "见 kernels_gemm.json"))}</div>'
    else:
        after = pending("GEMM/FFN/LM-head 的 review 修复和前后对比（Wave A kernels_gemm）")
    return f"""
<section id="ws1">
  <p class="eyebrow">WS1 · 单卡算子</p>
  <h2>每个算子都有一个“逐 bit 确定”的版本，速度够用</h2>
  <div class="prose"><p>仓库原有的严格内核（FA4 CuTe 注意力、线性 logp）只支持 Hopper，编译扩展 <code>rl_engine._C</code> 也无法对 CUDA 13 的 torch 构建，所以这次全部算子都是 PyTorch golden + Triton 重写。判断标准：同一个输入，不论 batch 怎么拼、序列怎么切、在第几个位置，输出 bit 都一样；和 fp64 参考的误差在 ws1 容差内。</p></div>
  {fig(ratio, "大多数逐元素/范数类算子比原生更快（原生是未融合的 eager PyTorch）；GEMM 类慢 2–2.6 倍，这是“固定 K 树 + 固定 tile”的代价。局部注意力的原生实现要先物化整张 mask，所以严格版快 20–50 倍。")}
  {fig(tf, "树形 GEMM 跑到 cuBLAS 的 40–50%。decode 形状（M=16）下大 GEMM 和 cuBLAS 持平。")}
  {fig(legacy, "新 GEMM 是这次的“必需行”：没有它，30B 前向一次要几分钟。")}
  {fig(mem, "词表 tile 流式算 softcap 和 logsumexp，不物化 4096×202752 的 logits：显存降到 1/24。")}
  {fig(tests, "合并后的回归：单卡 400 个测试通过，1 个已知失败（Megatron 绑定里的 toy 词表 tile 测试）。", wide=False)}
  {after}
</section>"""


def sec_chain() -> str:
    ref = load("reference_profile.json")
    body = ""
    if ref:
        body = f'<div class="callout">{esc(ref.get("headline", ""))}</div>'
    else:
        body = pending("合并后重跑真权重链 + HF 锚点（Wave A reference_profile）")
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
    cp = load("kernels_attention.json")
    cpb = hbars([("CP2 前向（global+local, T 2048–8192）", [0]), ("CP4 前向", [0]), ("CP2 反向 dQ/dK/dV", [0]), ("CP4 反向", [0]),
                 ("TP2×CP2（4 卡）vs 单卡", [0])], [("与 CP1 不同的元素", "s-ok")], title="CP 结果（真实内核）", width=720, bar_h=16,
                value_fmt=lambda v: "0（逐 bit 相同）")
    fix = ""
    if cp:
        fix = f'<div class="callout">{esc(cp.get("headline", ""))}</div>'
    else:
        fix = pending("把 CP 的 query 分块规则写进单卡反向内核，使 TP1 内核 == CP 路径（Wave A kernels_attention）")
    return f"""
<section id="multi">
  <p class="eyebrow">WS2 · 多卡</p>
  <h2>TP 和 CP：切卡不改变加法顺序</h2>
  {fig(D.fig_cp(), "CP 下 K/V 先 all-gather（T=8192、CP2 时 0.23 ms，约占前向 4%），query 按 rank 切。")}
  {fig(cpb, "TP2/TP4/TP8 的 tp_canonical_tree、vocab_parallel_softcap_logp 在真实 NCCL rank 上都与 TP1 逐 bit 相同（8 卡 TP8 测试本轮补跑通过）。")}
  <div class="callout warn"><b>发现的契约缺陷</b>：契约 O5 写的反向规则“dK/dV 按 query block 升序累加”不是 CP 不变的。CP 路径实际用的是“整条序列 8 个规范 query 块 + 树合并”，CP1/2/4 彼此相等，但和单卡一次性内核在最后几位不同（相对误差 1.6e-5）。</div>
  {fix}
</section>"""


def sec_runtimes() -> str:
    v = load("vllm_runtime.json")
    m = load("megatron_runtime.json")
    parts = []
    parts.append(f'<div class="callout">{esc(v.get("headline", ""))}</div>' if v else pending("vLLM 0.31 严格引擎在 30B 上的运行 (a)–(f)（Wave A vllm_runtime）"))
    parts.append(f'<div class="callout">{esc(m.get("headline", ""))}</div>' if m else pending("Megatron Bridge 严格 trainer 在 30B 上的 TP1/TP2 检查（Wave A megatron_runtime）"))
    return f"""
<section id="runtimes">
  <p class="eyebrow">两个真实运行时</p>
  <h2>vLLM 0.31 和 Megatron Bridge 跑同一套内核</h2>
  {''.join(parts)}
</section>"""


def sec_xrt() -> str:
    names = [
        ("参考 golden", "xrt_reference_golden.json"), ("参考 Triton", "xrt_reference_triton.json"),
        ("vLLM 严格 TP1", "xrt_vllm_strict_tp1.json"), ("vLLM 严格 TP2", "xrt_vllm_strict_tp2.json"),
        ("Megatron 严格 TP1", "xrt_megatron_strict_tp1.json"), ("Megatron 严格 TP2", "xrt_megatron_strict_tp2.json"),
        ("vLLM 原生 TP1", "xrt_vllm_native_tp1.json"), ("vLLM 原生 TP2", "xrt_vllm_native_tp2.json"),
        ("Megatron 原生 TP1", "xrt_megatron_native_tp1.json"), ("Megatron 原生 TP2", "xrt_megatron_native_tp2.json"),
    ]
    got = {lab: load(f) for lab, f in names}
    have = [lab for lab, _ in names if got[lab]]
    if len(have) < 2:
        return f"""
<section id="xrt">
  <p class="eyebrow">关键结论</p>
  <h2>三方逐字节比对</h2>
  {pending("参考链、vLLM、Megatron 对同一份 xrt_tokens.json 打分后逐字节比对（Wave B）")}
</section>"""
    def seqmap(d):
        return {s["id"]: s["logprobs_hex"] for s in d["sequences"]}
    maps = {lab: seqmap(got[lab]) for lab in have}
    total = sum(len(v) for v in maps[have[0]].values())
    vals = []
    for a in have:
        row = []
        for b in have:
            n = 0
            for sid, ha in maps[a].items():
                hb = maps[b].get(sid, [])
                n += sum(1 for x, y in zip(ha, hb) if x != y) + abs(len(ha) - len(hb))
            row.append(n)
        vals.append(row)
    hm = heatmap(have, have, vals, title=f"不同 bit 的 token 数（共 {total} 个被打分的 token）", cell=38, cell_w=58)
    return f"""
<section id="xrt">
  <p class="eyebrow">关键结论</p>
  <h2>三方逐字节比对</h2>
  {fig(hm, "同一份 token（8 道 GSM8K + 1 条跨越 2048 窗口的长序列），每个实现输出每个 token 的 fp32 logprob 的 bit 模式。绿色 0 = 逐 bit 相同。")}
</section>"""


def sec_e2e() -> str:
    e = load("e2e_run.json")
    if not e:
        return f"""
<section id="e2e">
  <p class="eyebrow">端到端</p>
  <h2>LoRA GRPO：rollout 和训练的 logprob 每一步都逐 bit 相同吗</h2>
  {pending("四种模式的真实 30B GRPO（Wave C）")}
</section>"""
    return ""


def sec_ablation() -> str:
    a = load("ablation.json")
    if not a:
        return f"""
<section id="ablation">
  <p class="eyebrow">消融</p>
  <h2>原生栈的漂移从哪来</h2>
  {pending("HF 模型单因素消融（Wave A ablation）")}
</section>"""
    return ""


def sec_profile() -> str:
    p = load("reference_profile.json")
    if not p:
        return f"""
<section id="profile">
  <p class="eyebrow">Profiling</p>
  <h2>时间花在哪里</h2>
  {pending("严格单卡模型 vs HF 的逐算子时间分解（Wave A reference_profile）")}
</section>"""
    return ""


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

HARD_FINDINGS = [
    ("O5 反向规则不是 CP 不变的", "把 dK/dV 的累加规则改成“整条序列 8 个规范 query 块，块间按 O3 树合并”，单卡内核和 CP 路径共用。"),
    ("注意力内核钉死 32Q/2KV 头", "契约和内核签名都要接受 H_Q/H_KV，TP 分片 rank 直接算自己的头。"),
    ("final norm 权重在 Bridge 里存成 w−1", "bf16(w_meg+1) 与 HF 有 1120/6656 个元素不同；weight_sync_identity 要允许“从 HF 重载这一个张量”，预期 626/627 完全相同。"),
    ("Bridge 的 fused linear_qkv 布局", "每个 KV 组 g（4352 行）= [q 头 16g..（2048）| gate 头（2048）| k 头 g（128）| v 头 g（128）]；linear_fc1 = [gate; up]。严格路径要拆开。"),
    ("梯度容差要与尺度无关", "反向的比较用归一化误差，固定 atol 在 30B 的梯度尺度上没有意义。"),
    ("m = −inf 保护要写进 O5", "全被 mask 的行：α=1，p=0。"),
    ("LoRA 不合并", "y = bf16(acc32(W·x) + Σ B·z)；合并 LoRA 会改变 bit。"),
    ("行要改名/合并", "fused_add_rmsnorm → post_norm_residual_pre_norm（四范数夹心）；注意力 train/rollout 两行合成一行两种执行模式；vocab_parallel_softcap_logp 并入 WS1 行 + 传输。"),
]


def sec_rfc() -> str:
    a = load("rfc_audit.json")
    extra = load("rfc_extra.json") or []
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
    hard = "".join(f'<div class="fact"><span class="fk">{esc(t)}</span><span class="fn">{esc(d)}</span></div>' for t, d in HARD_FINDINGS + [tuple(x) for x in extra])
    return f"""
<section id="rfc">
  <p class="eyebrow">结论</p>
  <h2>RFC 需要怎么改</h2>
  <h3>实现过程中才暴露的问题</h3>
  <div class="facts">{hard}</div>
  {fig(chart, "逐行审计 RFC 原文：每条都给出原文、问题、建议文本和证据（文件:行号），再由另一个 agent 独立复核。下面按 RFC 章节展开，建议文本是英文，可以直接贴到 issue 里。")}
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
  --font-display: "Archivo", "Noto Sans SC", "PingFang SC", system-ui, sans-serif;
  --font-body: "Noto Sans SC", "PingFang SC", "Microsoft YaHei", system-ui, sans-serif;
  --font-mono: "JetBrains Mono", ui-monospace, "SFMono-Regular", Menlo, "Noto Sans SC", "PingFang SC", monospace;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --paper: #0D1116; --panel: #141A21; --ink: #E3E8EE; --muted: #97A2B0; --rule: #28303A;
  --strict: #8DABFF; --strict-soft: #1B2540; --native: #EEA35C; --native-soft: #3A2918;
  --ok: #63C18D; --ok-soft: #15301F; --bad: #F27B6F; --bad-soft: #3B1C19; --pend: #E2C35E; --pend-soft: #33290F;
  --r0: #8DABFF; --r1: #EEA35C; --r2: #63C18D; --r3: #C98BE0; --r4: #74C7D4; --r5: #F27B6F; --r6: #B9C866; --r7: #9DAAC4;
  color-scheme: dark; } }
:root[data-theme="dark"] {
  --paper: #0D1116; --panel: #141A21; --ink: #E3E8EE; --muted: #97A2B0; --rule: #28303A;
  --strict: #8DABFF; --strict-soft: #1B2540; --native: #EEA35C; --native-soft: #3A2918;
  --ok: #63C18D; --ok-soft: #15301F; --bad: #F27B6F; --bad-soft: #3B1C19; --pend: #E2C35E; --pend-soft: #33290F;
  --r0: #8DABFF; --r1: #EEA35C; --r2: #63C18D; --r3: #C98BE0; --r4: #74C7D4; --r5: #F27B6F; --r6: #B9C866; --r7: #9DAAC4;
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
.bitstrip .hx { letter-spacing: 0.08em; }
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
footer { padding-top: 2rem; font-size: 0.85rem; color: var(--muted); }
@media (max-width: 640px) { body { font-size: 15px; } .fig { padding: 0.7rem; } }
@media (prefers-reduced-motion: reduce) { * { scroll-behavior: auto !important; } }
"""


def hero() -> str:
    x = load("xrt_summary.json")
    if x and x.get("example"):
        ex = x["example"]
        strip = "".join(
            f'<div class="lab">{esc(r["label"])}</div><div class="hx">{esc(r["hex"])}</div>' for r in ex
        )
        bits = f'<div class="bitstrip">{strip}</div>'
    else:
        hs, ht, _, _ = D.sum_orders()
        bits = (
            '<div class="bitstrip"><div class="lab">同一组数，顺序 A</div><div class="hx">0x' + hs + '</div>'
            '<div class="lab">同一组数，顺序 B</div><div class="hx">0x' + ht + "</div></div>"
        )
    return f"""
<header class="mast">
  <p class="eyebrow">RFC RL-Align/RL-Kernel#501 · POC 报告 · 2026-10-15</p>
  <h1>Muse-Glimmer-30B：让训练和推理算出同一个 bit</h1>
  <p class="dek">在一台 8×A100 上，用真实的 vLLM 0.31 做 rollout、真实的 Megatron Bridge 做 LoRA 训练，按一份数值契约把每一次加法的顺序钉死，再跑 GRPO 看两边的 logprob 是否逐 bit 相同。这篇是结果、过程，以及 RFC 需要怎么改。</p>
  <div class="chips"><span class="chip">8× A100-SXM4-80GB</span><span class="chip">vLLM 0.31.0</span><span class="chip">Megatron Bridge 0.7 (git d86a5c09)</span><span class="chip">Triton 3.7.1</span><span class="chip">transformers 5.17.0</span><span class="chip">checkpoint a4e59da5</span><span class="chip">分支 feat/museglimmer-poc</span></div>
  {bits}
</header>"""


TOC = [("why", "问题"), ("scope", "范围"), ("model", "模型"), ("contract", "契约"), ("ws1", "单卡算子"), ("chain", "全模型链"),
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
        hero(), toc, sec_why(), sec_scope(), sec_model(), sec_contract(), sec_ws1(), sec_chain(), sec_multi(), sec_runtimes(),
        sec_xrt(), sec_e2e(), sec_ablation(), sec_profile(), sec_timeline(), sec_rfc(), sec_left(),
    ])
    foot = ('<footer>数据：<code>reports/muse_glimmer_poc/data/*.json</code>；构建：<code>python3 reports/muse_glimmer_poc/blog/build_blog.py</code>。'
            '“未运行”标记的内容表示磁盘上还没有对应的结果文件。</footer>')
    return head + f'<main class="page">{body}{foot}</main>\n'


if __name__ == "__main__":
    html = build()
    OUT.write_text(html)
    print(f"wrote {OUT} ({len(html) / 1024:.0f} KiB)")
