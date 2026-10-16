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
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import diagrams as D  # noqa: E402
from audit_fixes import AUDIT_FIXES, section_title  # noqa: E402
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
    ("WS1 单卡", "dense_gemm（树形 GEMM）", "done", "8 叶 K 树；M=4096 大形状约 cuBLAS 的 1/2"),
    ("WS1 单卡", "normalized_embedding / norms", "done", "四范数夹心，两个 eps"),
    ("WS1 单卡", "qknorm_scaled_rope", "done", "kernel == golden 逐 bit"),
    ("WS1 单卡", "global/local 门控注意力 fwd", "done", "train = chunked = decode = paged"),
    ("WS1 单卡", "门控注意力 bwd", "done", "dK/dV 按规范块累加"),
    ("WS1 单卡", "swiglu_ffn", "done", "三个 GEMM 占 FFN 内核时间约 99%"),
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


def _pct(s: object) -> float | None:
    try:
        return float(str(s).strip().rstrip("%"))
    except ValueError:
        return None


def gemm_m4096() -> dict:
    """Large M=4096 GEMMs (N >= 4096): strict TFLOP/s in the WS1 bench and in the wave B re-measure (kernels_gemm.json)."""
    out = {"tflops": [], "frac_ws1": [], "frac_kg": []}
    b = load("ws1_bench.json")
    if b:
        det = b["benchmarks"]["benchmarks/models/muse_glimmer/bench_det_gemm.py"]["tables"]
        for r in next(t for t in det if t["heading"].startswith("Throughput at M=4096"))["rows"]:
            out["frac_ws1"].append(1 / num(r["strict/cuBLAS"]))
            if int(r["N"]) >= 4096:
                out["tflops"].append(num(r["strict TFLOP/s"]))
    kg = load("kernels_gemm.json")
    for r in ((kg or {}).get("benchmarks") or {}).get("gemm_forward") or []:
        if r["M"] == 4096:
            out["frac_kg"].append(r["cublas_ms_after"] / r["strict_ms_after"])
            if r["N"] >= 4096:
                out["tflops"].append(r["strict_tflops_after"])
    return out


def row_notes() -> dict[str, str]:
    """Row-card notes computed from the data files (they override row_status.json and the ROWS defaults)."""
    notes: dict[str, str] = {}
    g = gemm_m4096()
    if g["tflops"]:
        notes["dense_gemm（树形 GEMM）"] = f'8 叶 K 树；M=4096 大形状 {min(g["tflops"]):.0f}–{max(g["tflops"]):.0f} TFLOP/s（约 cuBLAS 的 1/2）'
    b = load("ws1_bench.json")
    if b:
        sw = b["benchmarks"]["benchmarks/models/muse_glimmer/bench_swiglu_ffn.py"]["tables"][0]["rows"]
        share = next((_pct(r["strict / native"]) for r in sw if r["case"].startswith("GEMM share")), None)
        if share:
            notes["swiglu_ffn"] = f"三个 GEMM 占 FFN 内核时间约 {share:.0f}%（SwiGLU 内核 {100 - round(share)}%）"
    fr = load("final_regression.json") or {}
    cps = [int(k[2:]) for k, d in (fr.get("cp_parity_torchrun") or {}).items()
           if d.get("records") and all(r[m] == 0 for r in d["records"] for m in r if m.endswith("mismatch"))]
    if cps:
        notes["cp_global / cp_local attention"] = f'CP{"/".join(str(c) for c in sorted({2, *cps}))} == CP1 前向+反向'
    x, v = load("xrt_summary.json"), load("vllm_runtime.json")
    if x and v:
        vb = sorted((x["files"][l]["tp"], l) for l in x["strict_groups"].get("base", []) if x["files"][l]["runtime"].startswith("vllm"))
        ref_eq = all(x["matrix"][l]["reference triton"] == 0 for _, l in vb)
        dvp = all(g_.get("decode_vs_prefill", {}).get("mismatched") == 0 for g_ in v.get("generate", []) if g_["mode"] == "strict")
        n = x["n_scored_tokens"]
        notes["vLLM 0.31 严格绑定"] = (f'{n if ref_eq else "?"}/{n} == 参考链；{" == ".join(f"TP{t}" for t, _ in vb)}'
                                  f'{"；decode == prefill" if dvp else ""}')
    m = load("megatron_runtime.json")
    if m:
        runs = m.get("runs", {})
        tps = [t for t in ("tp1", "tp2", "tp4", "tp8") if t in runs]
        def same(k, f):
            vals = [runs[t].get("checks", {}).get(k, {}).get(f) for t in tps]
            return None not in vals and len(set(vals)) == 1
        if same("grad_repeat", "aggregate_sha256") and same("step", "lora_sha256_after"):
            notes["Megatron Bridge 严格绑定"] = f'logprob、LoRA 梯度、AdamW 一步 {" == ".join(t.upper() for t in tps)}'
    a = load("ablation.json")
    if a:
        hf1 = [s for s in a["settings"] if s["engine"] == "hf" and s.get("n_factors") == 1]
        st = [s for s in a["settings"] if s["engine"] == "strict"]
        mt = re.search(r"0 of (\d+) scored tokens", a.get("headline", ""))
        tot = f'0/{mt.group(1)}' if mt and not any(s["n_bits_differ"] for s in st) else f'{sum(s["n_bits_differ"] for s in st)} 个不同'
        notes["museglimmer_ablation"] = (f'{len(hf1)} 个 HF 单因素比较（{sum(1 for s in hf1 if s["n_bits_differ"])} 个变 bit）'
                                        f'+ {len(st)} 个严格设置（{tot}）')
    return notes


def e2e_layout() -> tuple[str, str]:
    """Rollout / train TP of the four-mode GRPO run (the unlabelled runs), for the system diagram."""
    e = load("e2e_run.json") or {}
    base = [m for m in e.get("modes", []) if not m.get("label")]
    cfg = (base[0].get("config") or {}) if base else {}
    return f'TP{cfg.get("rollout_tp", "?")}', f'TP{cfg.get("train_tp", "?")}'


def sec_scope() -> str:
    status = load("row_status.json") or {}
    computed = row_notes()
    cells = []
    for grp, row, st, note in ROWS:
        st2 = status.get(row, {}).get("status", st)
        note2 = computed.get(row) or status.get(row, {}).get("note", note)
        cls = {"done": "st-done", "partial": "st-part", "wave": "st-pend", "blocked": "st-bad"}.get(st2, "st-pend")
        lab = {"done": "完成", "partial": "部分", "wave": "进行中", "blocked": "受阻"}.get(st2, st2)
        cells.append(f'<div class="rowcell {cls}"><span class="rg">{esc(grp)}</span><b>{esc(row)}</b><span class="pill">{esc(lab)}</span><span class="rn">{esc(note2)}</span></div>')
    rtp, ttp = e2e_layout()
    return f"""
<section id="scope">
  <p class="eyebrow">范围</p>
  <h2>这次 POC 做了什么</h2>
  <div class="prose"><p>把 RFC 的每一行都做到“能用、能测”的程度，再在一台 8×A100 上串起真实 vLLM 和真实 Megatron Bridge，跑一个 LoRA GRPO。性能只求够用；Phase B 视觉、DFlash 投机解码、全参数训练、量化、ROCm 不在范围内。</p></div>
  {fig(D.fig_system(rollout_tp=rtp, train_tp=ttp), "端到端循环（四种模式的 GRPO 布局）。两个运行时装在两个不能合并的 Python 环境里（torch 的 CUDA 构建不同），trainer 跑在另一个 env 的服务进程里，中间只传 token id、trainer 的 fp32 logprob、fp32 的 dL/dlogp 和带哈希的 LoRA。")}
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
    # REPORT_ws2_tp.md: two runs of the NCCL control (labelled fallback kernels, then the sibling kernels)
    nccl_runs = {"TP2 · NCCL all_reduce": [0, 0], "TP4 · NCCL all_reduce o_proj": [31, 46], "TP4 · NCCL all_reduce down": [50, 48],
                 "TP2/TP4/TP8 · 规范树": [0, 0]}
    tp4 = [v for k, vs in nccl_runs.items() if k.startswith("TP4") for v in vs]
    tp_bars = hbars(
        list(nccl_runs.items()),
        [("第一次运行（标注的回退内核）", "s-p7"), ("第二次运行（正式内核）", "s-bad")],
        title="o_proj / down 输出与 TP1 相比：不同的 bf16 元素数（共 1,703,936）", width=720, bar_h=14, value_fmt=lambda v: f"{v:.0f}",
    )
    leaves = hbars(
        [("8 个块直接顺序相加", [80, 80]), ("4 个块", [41, 28]), ("2 个块", [0, 0])],
        [("o_proj", "s-strict"), ("down", "s-native")],
        title="如果不用树、直接把各 rank 的块顺序相加：与 TP1 不同的元素数", width=720, bar_h=14, value_fmt=lambda v: f"{v:.0f}",
    )
    # REPORT_ws2_tp.md, rank 0, sibling run: (strict all-gather + tree, NCCL fp32 all_reduce) ms per call
    tdata = [("TP2 o_proj", [0.162, 0.164]), ("TP2 down", [0.155, 0.114]), ("TP4 o_proj", [0.224, 0.151]), ("TP4 down", [0.222, 0.149])]
    trat = [a / b for _, (a, b) in tdata]
    tp4d = max(a - b for k, (a, b) in tdata if k.startswith("TP4"))
    timing = hbars(
        tdata,
        [("all-gather + 规范树（严格）", "s-strict"), ("NCCL all_reduce（顺序不定）", "s-native")],
        title="归约 (256, 6656) fp32 根的耗时，ms / 次", width=720, bar_h=14, unit=" ms",
    )
    return f"""
<section id="contract">
  <p class="eyebrow">数值契约</p>
  <h2>一棵树管住跨 rank 的加法</h2>
  <div class="prose"><p>契约（<code>docs/contracts/muse-glimmer-30b-strict-contract.md</code>）给每个归约都定死顺序：GEMM 的 K 维、词表和 dK/dV 的 query 维切成 8 个固定的块，块内 fp32 累加，块间按一棵固定的平衡二叉树相加，再在声明的位置舍入一次（GEMM 输出舍入到 bf16；logits 保持 fp32）；注意力按 64 个 key 一块、升序做 online softmax；norm 是逐行固定顺序的归约。TP 只决定“谁算哪棵子树”，不改变树本身。</p></div>
  {fig(D.fig_tp_tree(), "规范 8 叶树。TP2 每个 rank 算 4 片叶子的子树，TP8 每个 rank 只算 1 片；all-gather 之后的合并永远是同一棵树的上层。")}
  {fig(tp_bars, f"真实 NCCL rank 上的对照实验，跑了两次：NCCL all_reduce 在 TP2 恰好对（两个数相加满足交换律），到 TP4 就错了 {min(tp4)}–{max(tp4)} 个元素，两次运行的结果也不同；规范树在 TP2/4/8 全部逐 bit 相同。")}
  {fig(leaves, "树是必要的：如果每个 rank 算完直接按顺序相加，TP4/TP8 就会和 TP1 不同。")}
  {fig(timing, f"代价：在 M=256 时，all-gather + 树是同尺寸 fp32 NCCL all_reduce 的 {min(trat):.1f}–{max(trat):.1f} 倍（TP4 多 {tp4d:.2f} ms；rank 0 单次采样）；原生运行时实际做的是 bf16 all_reduce，字节数只有一半。")}
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
    # attention: wave B paired runs (kernels_attention.json); local layers only against vLLM FA2's sliding-window
    # kernel (the wave A SDPA dense-mask baseline of the local rows is withdrawn, ratio = null in the data)
    wb = (((load("kernels_attention.json") or {}).get("benchmarks_wave_b") or {}).get("run_1_all_sections") or {}).get("data") or {}
    def att(sec, **kw):
        return next((r for r in wb.get(sec, []) if all(r.get(k) == v for k, v in kw.items())), None)
    pg, pl = att("prefill_forward", T=4096, window=None), att("prefill_forward", T=4096, window=2048)
    dg, bg = att("decode_forward", B=64, window=None), att("backward", T=4096, window=None)
    if pg:
        rows += [("注意力 prefill T=4096 全局（vs SDPA）", [pg["ratio"]]), ("注意力 prefill T=4096 全局（vs FA2）", [pg["flash_ratio"]])]
    if pl:
        rows.append(("注意力 prefill T=4096 局部（vs FA2 滑窗）", [pl["flash_ratio"]]))
    if dg:
        rows += [(f'注意力 decode B=64 ctx={dg["ctx"]} 全局（vs SDPA）', [dg["ratio"]]), (f'注意力 decode B=64 ctx={dg["ctx"]} 全局（vs FA2）', [dg["flash_ratio"]])]
    if bg:
        rows.append(("注意力反向 全局 T=4096（vs SDPA）", [bg["ratio"]]))
    for r in bm["benchmarks/models/muse_glimmer/bench_norms.py"]["tables"][0]["rows"]:
        if "fwd+bwd" not in r["op"] or "fused" in r["op"]:
            rows.append((r["op"].replace("_", " "), [num(r["strict/native"])]))
    rows.append(("qknorm + RoPE 前向", [0.088 / 0.9593]))
    rows.append(("qknorm + RoPE 前向+反向", [0.3862 / 2.3927]))
    for r in bm["benchmarks/models/muse_glimmer/bench_swiglu_ffn.py"]["tables"][0]["rows"]:
        if r["case"] in ("swiglu (T x F) fwd", "swiglu_ffn fwd (T tokens)"):  # the two "share" rows are time shares, not ratios
            rows.append((r["case"], [num(r["strict / native"])]))
    rows.append(("LM head + softcap + logprob 前向", [97.3 / 69.1]))
    return rows


VERDICT_ZH = {"confirmed_fixed": "确认并修复", "test_added": "覆盖缺口，已补测试", "partially_fixed": "部分修复（未关闭）",
              "confirmed_not_fixed": "确认，未改（已声明）", "refuted": "被驳回", "not_checked": "未核查"}
VERDICT_CLS = {"confirmed_fixed": "s-ok", "test_added": "s-p4", "partially_fixed": "s-p3", "confirmed_not_fixed": "s-pend",
               "refuted": "s-p7", "not_checked": "s-bad"}


def verdict_key(v: object) -> str:
    """Map a finding verdict (an enum key or free text such as 'partially_fixed: ...') onto a VERDICT_ZH key."""
    v = str(v or "")
    if v in VERDICT_ZH:
        return v
    if v.startswith("partially_fixed") or v.startswith("fixed (partially"):
        return "partially_fixed"
    if v.startswith("coverage_gap") or "test_added" in v:
        return "test_added"
    if v.startswith("fixed"):
        return "confirmed_fixed"
    return "not_checked"


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
            cnt[verdict_key(f.get("verdict"))] += 1
        rows.append((f"{name}（{len(fs)} 条）", cnt))
    rope_note = ""
    rc = next((c for n, c in rows if n.startswith("QK-norm")), None)
    if rc:
        rope_note = ("QK-norm + RoPE 的 " + str(sum(rc.values())) + " 条：" +
                     "；".join(f"{VERDICT_ZH[k]} {rc[k]} 条" for k in VERDICT_ZH if rc[k]) +
                     "（其中一条修复在 wave B 才完成：所有严格构表路径读同一张规范表；部分修复的一条要等调用方显式传 local= 才关闭）。")
    if rows:
        used = [k for k in VERDICT_ZH if any(c[k] for _, c in rows)]
        out.append(fig(stacked([(n, [c[k] for k in used]) for n, c in rows], [(VERDICT_ZH[k], VERDICT_CLS[k]) for k in used],
                               title="上一轮对抗 review 留下的未验证 findings：这一轮逐条复现后的结论", width=860, bar_h=18, unit="条"),
                       "每条 finding 先写探针复现，确认后修复并补一个“修复前失败、修复后通过”的测试。" + rope_note))
    if kg:
        p1 = kg.get("det_gemm_1_iters_probe", {})
        bf, af = p1.get("before", {}).get("forward_sweep", []), p1.get("after", {}).get("forward_sweep", [])
        if bf and af:
            rows = [(f"K={b['K']} M={b['M']}", [100 * b["mismatch"] / b["numel"], 100 * a["mismatch"] / a["numel"]]) for b, a in zip(bf, af)]
            bad = [r[1][0] for r in rows if r[1][0] > 0]
            ks = sorted({b["K"] for b in bf if b["mismatch"]})
            out.append(fig(hbars(rows, [("修复前", "s-bad"), ("修复后", "s-ok")], width=860, bar_h=12, xmax=100, value_fmt=lambda v: f"{v:.1f}%",
                                 title="det_gemm:1：输出与“声明的 8 叶树”不一致的比例"),
                           f"K 很小时每个 K 块只循环一次，Triton 编译器把树的加法折进了 MMA 的累加链（7 个加法只剩 3 个），K={'/'.join(map(str, ks))} 时 {min(bad):.0f}–{max(bad):.0f}% 的输出偏离契约。修复：强制每块至少两次迭代 + static_assert + 检查编译后的 TTGIR。30B 前向的 101 个形状哈希全部不变，只有反向里 53/126 个形状改了 bit。"))
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
            cf = t["cos_mismatch_frac"]
            fr = cf + list(t.get("sin_mismatch_frac", []))
            out.append(fig(lines(xs, [("cos", "s-p0", [100 * v for v in t["cos_mismatch_frac"]]), ("sin", "s-p1", [100 * v for v in t.get("sin_mismatch_frac", t["cos_mismatch_frac"])])],
                                 title="RoPE 表在 CPU 上构造 vs 在 GPU 上构造：bit 不同的元素比例", xlabel="位置（K token）", ymin=0, width=860, height=240,
                                 dots=False, y_fmt=lambda v: f"{v:.0f}%"),
                           f"同一个公式、同样 fp32，CPU 和 CUDA 的 sin/cos 实现不同，约 {100 * min(fr):.0f}–{100 * max(fr):.0f}%（平均约 {100 * sum(cf) / len(cf):.0f}%）的表项 bit 不同，位置越大越多。所以契约 O4 要求两边用同一个函数在 CUDA 上构造一次、缓存同一个对象；这一行现在会校验表的来源。"))
    return "".join(out)


def regression_text() -> str:
    """The final regression on the merged code (data/final_regression.json), or the per-wave counts when it is missing."""
    fr = load("final_regression.json")
    if not fr or not fr.get("suites"):
        return ("wave A 合并后：vLLM 环境 581 通过 / 7 跳过，Megatron 环境 222 通过 / 1 跳过；wave B 各分支的单卡全套为 653–675 通过 / 8 跳过"
                "（最终合并代码上的全套回归没有记录）。")
    zh = {"single-GPU suite, vLLM env": "vLLM 环境单卡全套", "Megatron env (binding, e2e bridge/loop, det_gemm, fingerprint)": "Megatron 环境",
          "distributed on 8 GPUs (TP2/4/8, CP2/4, TP2xCP2, KV replication)": "8 卡分布式（TP2/4/8、CP2/4、TP2×CP2、KV 复制）"}
    parts = [f'{zh.get(x["name"], x["name"])}{"" if zh.get(x["name"], "").endswith("）") else " "}{x["passed"]} 通过 / {x["skipped"]} 跳过 / {x["failed"]} 失败'
             for x in fr["suites"]]
    cp = fr.get("cp_parity_torchrun") or {}
    cps = [k.upper() for k, v in cp.items() if v.get("records") and all(r[m] == 0 for r in v["records"] for m in r if m.endswith("mismatch"))]
    tail = f"；torchrun 真多进程的 {'/'.join(cps)} 前向 + 反向与单卡逐 bit 相同" if cps else ""
    return "最终合并代码上的回归：" + "；".join(parts) + tail + "。"


def sec_ws1() -> str:
    b = load("ws1_bench.json")
    if not b:
        return pending("WS1 基准数据缺失")
    rows = _ws1_ratio_rows(b)
    gemm4096 = [v[0] for k, v in rows if k.startswith("GEMM") and "M=4096" in k]
    ratio = hbars(rows, [("严格实现耗时 / 原生耗时", "s-strict")], title="严格内核 vs 原生实现：耗时比（对数轴，左边 = 更快）",
                  log=True, ref=1.0, ref_label="1× 一样快", width=860, bar_h=16, value_fmt=lambda v: f"{v:.2f}×")
    bm = b["benchmarks"]
    det = bm["benchmarks/models/muse_glimmer/bench_det_gemm.py"]["tables"]
    t4096 = next(t for t in det if t["heading"].startswith("Throughput at M=4096"))
    t16 = next(t for t in det if t["heading"].startswith("Throughput at M=16"))
    frac = [1 / num(r["strict/cuBLAS"]) for r in t4096["rows"]]
    big16 = [1 / num(r["strict/cuBLAS"]) for r in t16["rows"] if r["shape"] in ("mlp_gate", "up_proj", "down_proj", "lm_head")]
    kgf = gemm_m4096()["frac_kg"]
    tf = hbars([(r["shape"], [num(r["strict TFLOP/s"]), num(r["cuBLAS TFLOP/s"])]) for r in t4096["rows"]],
               [("树形 GEMM（严格）", "s-strict"), ("cuBLAS（顺序不定）", "s-native")], title="M=4096 时的 GEMM 吞吐，TFLOP/s（A100 bf16 峰值 312；WS1 基准）",
               width=860, bar_h=12)
    def mnk(r):
        return 2 * 4096 * int(r["N"]) * int(r["K"])
    leg = {r["shape"]: mnk(r) / (num(r["legacy TritonDetGemmOp ms"]) * 1e-3) / 1e12 for r in t4096["rows"] if num(r["legacy TritonDetGemmOp ms"])}
    leg_ffn = [v for k, v in leg.items() if k in ("mlp_gate", "up_proj", "down_proj")]
    legacy = hbars([(r["shape"], [num(r["legacy/strict"])]) for r in t4096["rows"] if num(r["legacy/strict"])],
                   [("仓库原有 Triton det_gemm 比新树形 GEMM 慢多少倍", "s-native")], log=True, width=860, bar_h=14,
                   value_fmt=lambda v: f"{v:.0f}×",
                   title=f"为什么要重写 GEMM：仓库原有确定性 GEMM 在 A100 上只有 {min(leg.values()):.0f}–{max(leg.values()):.0f} TFLOP/s"
                         f"（FFN 形状约 {min(leg_ffn):.0f}–{max(leg_ffn):.0f}）")
    lm = bm["benchmarks/models/muse_glimmer/bench_lm_head_softcap_logprob.py"]
    mv = re.search(r"V=(\d+) \(padded (\d+)", lm.get("raw", ""))
    vocab, vpad = (mv.group(1), mv.group(2)) if mv else ("202048", "202752")
    lr = lm["tables"][0]["rows"][0]
    nat_mib, str_mib = num(lr["native peak MiB"]), num(lr["strict peak MiB"])
    mem = hbars([("原生（cuBLAS + 全量 logits）", [nat_mib]), ("严格（8 个词表 tile 流式）", [str_mib])], [("峰值显存 MiB", "s-strict")],
                title=f"LM head + softcap + logprob，T=4096，词表 {vocab}（严格路径补齐到 {vpad}）：峰值显存", width=720, bar_h=20, unit=" MiB",
                value_fmt=lambda v: f"{v:,.0f} MiB")
    tests = hbars([("dense GEMM", [80]), ("门控注意力 fwd", [63]), ("norms + embedding", [43]), ("qknorm + RoPE", [39]),
                   ("指纹 / loader", [33]), ("门控注意力 bwd", [24]), ("swiglu FFN", [20]), ("lm head", [15])],
                  [("测试数（合并后全部通过）", "s-ok")], title="WS1 每一行的测试数", width=720, bar_h=14, value_fmt=lambda v: f"{v:.0f}")
    after = ws1_review_block() or pending("GEMM/FFN/LM-head 的 review 修复和前后对比")
    kg_txt = f"（修复 det_gemm:1 之后在 wave B 重测是 {100 * min(kgf):.0f}–{100 * max(kgf):.0f}%）" if kgf else ""
    return f"""
<section id="ws1">
  <p class="eyebrow">WS1 · 单卡算子</p>
  <h2>每个算子都有一个“逐 bit 确定”的版本，速度够用</h2>
  <div class="prose"><p>仓库原有的严格内核（FA4 CuTe 注意力、线性 logp）只支持 Hopper，编译扩展 <code>rl_engine._C</code> 在这套环境里没有构建，所以这次全部算子都是 PyTorch golden + Triton 重写。判断标准：同一个输入，不论 batch 怎么拼、序列怎么切、在第几个位置，输出 bit 都一样；和 fp64 参考的误差在 ws1 容差内（门控注意力反向的梯度容差按量级缩放，已在报告中声明）。</p></div>
  {fig(ratio, f"大多数逐元素/范数类算子比原生更快（原生是未融合的 eager PyTorch）；GEMM 在 M=4096 时慢 {min(gemm4096):.1f}–{max(gemm4096):.1f} 倍，这是“固定 K 树 + 固定 tile”的代价。注意力前向几行是 wave B 的交替计时（5 轮中位数）；反向那行是单次序列的均值，两次运行分别是 2.54× 和 2.81×。全局层对比 SDPA 和 vLLM FA2，局部层只用 vLLM FA2 的滑窗内核作基线；旧的 SDPA 整张 mask 基线已撤回。")}
  {fig(tf, f"树形 GEMM 跑到 cuBLAS 的 {100 * min(frac):.0f}–{100 * max(frac):.0f}%{kg_txt}。decode 形状（M=16）下 FFN 和 LM head 的大 GEMM 是 cuBLAS 的 {100 * min(big16):.0f}–{100 * max(big16):.0f}%，基本持平。")}
  {fig(legacy, "新 GEMM 是这次的“必需行”：没有它，30B 前向一次要几分钟。")}
  {fig(mem, f"词表 tile 流式算 softcap 和 logsumexp，不物化 4096×{vocab} 的 logits：显存降到 1/{nat_mib / str_mib:.0f}。")}
  {fig(tests, "第一轮合并时各行的测试数。" + regression_text(), wide=False)}
  {after}
</section>"""


CHUNK_ZH = {"chunked:prompt_boundary": "prompt 边界", "chunked:block64": "64 对齐", "chunked:ragged": "不规则",
            "chunked:window_boundary_2047": "W−1", "chunked:window_boundary_2048": "W", "chunked:window_boundary_2049": "W+1",
            "chunked:halves": "对半"}


def sec_chain() -> str:
    ref = load("reference_profile.json")
    ex = (ref or {}).get("experiments", {})
    body = ""
    ch = ex.get("chain_triton_2100_hf")
    if ch:
        pz = {"train": "train forward（整段）", "prefill": "prefill + decode", "chunked:prompt_boundary": "在 prompt 边界切", "chunked:halves": "对半切",
              "chunked:block64": "每 64 token 一块", "chunked:ragged": "不规则切", "chunked:window_boundary_2047": "边界 2047",
              "chunked:window_boundary_2048": "边界 2048", "chunked:window_boundary_2049": "边界 2049", "decode": "逐 token decode"}
        rows = [(pz.get(x["name"], x["name"]), [x["seconds"]]) for x in ch["paths"]]
        eq = all(x.get("byte_equal_to_train") for x in ch["paths"])
        xs = ex.get("xrt_scoring", {})
        def modes(k):
            return "、".join(CHUNK_ZH.get(m, m) for m in xs.get(k, {}).get("check_modes", []) if m.startswith("chunked:"))
        xeq = all(xs.get(k, {}).get("all_byte_equal_to_train") for k in ("triton", "golden") if k in xs)
        xtxt = (f"xrt 的 9 条序列（{xs.get('triton', {}).get('n_tokens', '?')} 个 token）也做了 train = prefill = decode = 多种分块"
                f"（Triton：{modes('triton')}；golden：{modes('golden')}）的检查，{'全部逐 bit 相同' if xeq else '存在不同'}。") if xs else ""
        body = fig(hbars(rows, [("耗时（秒）", "s-strict")], width=720, bar_h=14, unit=" s",
                         title=f"prompt 2100 + 32 个答案 token：各条执行路径耗时（{'全部与 train 逐字节相同' if eq else '存在不同'}）"),
                   "本轮在合并后的代码上重跑（Triton 后端）。" + xtxt)
    else:
        body = pending("合并后重跑真权重链 + HF 锚点")
    # wave A (REPORT_full_model_chain.md, original hand-back at 09eef9d, before the O7 fix): one 512+32 sequence,
    # max |dlogp| over the 32 completion tokens, argmax over all 544 positions; HF fp32 was measured only then
    wa_dl = [("严格 golden vs HF fp32（wave A）", 0.041), ("严格 Triton vs HF fp32（wave A）", 0.046), ("HF bf16 vs HF fp32（wave A）", 0.046),
             ("严格 golden vs HF bf16（wave A）", 0.064), ("严格 Triton vs HF bf16（wave A）", 0.066)]
    wa_am = [("严格 golden vs HF fp32（wave A）", 99.63), ("严格 Triton vs HF fp32（wave A）", 99.63), ("HF bf16 vs HF fp32（wave A）", 98.53),
             ("严格 golden vs HF bf16（wave A）", 98.5), ("严格 Triton vs HF bf16（wave A）", 98.9)]
    wb_dl, wb_am = [], []
    for k, tag in (("chain_triton_512_hf", "512+32"), ("chain_triton_2100_hf", "2100+32")):
        an = ex.get(k, {}).get("hf_anchor") or {}
        if an.get("logp_completion"):
            lab = f"严格 Triton vs HF bf16（wave B，{tag}）"
            wb_dl.append((lab, an["logp_completion"]["max_abs"]))
            wb_am.append((lab, 100 * an["argmax_agreement_all"]))
    anchor = hbars([(a, [v]) for a, v in wa_dl + wb_dl], [("最大 |Δ logp|", "s-strict")],
                   title="精度锚点：和 HF 实现比，答案 token 上的最大 |Δ logp|（越小越好）", width=720, bar_h=16)
    argmax = hbars([(a, [v]) for a, v in wa_am + wb_am], [("argmax 一致率 %", "s-ok")], title="argmax 一致率（%，全部位置）",
                   width=720, bar_h=16, xmax=100, value_fmt=lambda v: f"{v:.2f}%")
    return f"""
<section id="chain">
  <p class="eyebrow">全模型链</p>
  <h2>52 层真权重：10 种切法，一个答案</h2>
  {fig(D.fig_window(), "滑动窗口 2048。chunk 边界放在 2047、2048、2049 这些最容易出错的地方，结果都和一次性 train forward 逐 bit 相同；golden（fp32 定序参考：PyTorch 算子 + 仓库原有的 fp32 Triton GEMM）和全 Triton 两条后端都是 32/32 字节相等。")}
  {fig(anchor, "严格实现不是“换了一种错法”。wave A 的行：单条序列 512+32，|Δlogp| 取 32 个答案 token 的最大值，argmax 取全部 544 个位置，在 O7 修复前测得（HF fp32 锚点只在那时测过）；wave B 在合并后的代码上只重测了 HF bf16 锚点，并加了一条 2100+32 的长序列。以 HF fp32 为准，严格路径的误差不比 HF 自己的 bf16 大：golden 更近，Triton 的最大误差持平（0.046），均值（0.016 vs 0.019）和 argmax 更好。")}
  {fig(argmax, "")}
  {body}
</section>"""


def sec_multi() -> str:
    k = load("kernels_attention.json")
    # REPORT_ws2_cp.md "Wave A update": AG of K+V 0.280 / 0.343 ms vs a 5.1-6.3 / 3.0-3.2 ms rank forward (CP2, T=8192)
    out = [fig(D.fig_cp(), "CP 下 K/V 先 all-gather（T=8192、CP2 时全局层 0.28 ms，约占该 rank 前向的 5%；局部层 0.34 ms，约 11%），query 按 rank 切。")]
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
                           "TP2 每个 rank 16 个 Q 头 / 1 个 KV 头，TP4、TP8 是 8/1、4/1（KV 头复制）。前向和反向全部 0。所以 TP 下每个 rank 只算自己的 Q 头，不需要把 Q 头 all-gather 回来；TP4/TP8 时 KV 头按副本复制，各副本的 fp32 dK/dV 部分和按 rank 顺序 all-gather 后用同一棵树合并一次，也和 TP1 一样。"))
        fr = (load("final_regression.json") or {}).get("cp_parity_torchrun") or {}
        cpr = [k.upper() for k, d in fr.items() if d.get("records") and all(r[m] == 0 for r in d["records"] for m in r if m.endswith("mismatch"))]
        cp_real = f"，最终回归里 torchrun 真多进程的 {'/'.join(cpr)} 也一样" if cpr else ""
        v1 = {2048: 1557, 4096: 5540, 8192: 20440}
        out.append(fig(hbars([(f"T={t}", [n, 0]) for t, n in v1.items()], [("旧内核（按 query block 升序累加）", "s-bad"), ("新内核（8 个规范 query 块 + 树）", "s-ok")],
                             width=760, bar_h=14, value_fmt=lambda v: f"{v:.0f}", title="单卡一次性反向 vs CP1 路径：dK 里 bit 不同的元素数"),
                       "旧规则在 CP 下做不到；新规则下单卡、CP1、模拟 CP2/4/8、真实双卡 CP2 全部逐 bit 相同" + cp_real + "。前向 66/66 个摘要不变。"))
        ba = k.get("benchmarks", {}).get("backward_before_after", [])
        if ba:
            def med(x):
                x = sorted(x)
                return x[len(x) // 2]
            r512 = [med(r["v2_ms"]) / med(r["v1_ms"]) for r in ba if r["T"] == 512]
            rbig = [med(r["v2_ms"]) / med(r["v1_ms"]) for r in ba if r["T"] >= 1024]
            out.append(fig(hbars([(f'T={r["T"]} {"局部" if r.get("window") else "全局"}', [med(r["v1_ms"]), med(r["v2_ms"])]) for r in ba],
                                 [("旧反向", "s-p7"), ("新反向", "s-strict")], width=760, bar_h=12, unit=" ms", title="代价：反向耗时（ms，交替计时 7 轮的中位数）"),
                           f"T ≥ 1024 时新反向慢 {100 * (min(rbig) - 1):.0f}–{100 * (max(rbig) - 1):.0f}%，T=512 时慢 {100 * (min(r512) - 1):.0f}–{100 * (max(r512) - 1):.0f}%（瞬时显存 2.5 倍）。"))
        wb = k.get("benchmarks_wave_b") or {}
        d1 = (wb.get("run_1_all_sections") or {}).get("data") or {}
        d2 = (wb.get("run_2_backward_tp") or {}).get("data") or {}
        def shard(d, sec, tp, w):
            return next((x["strict_ms"] for x in d.get(sec, []) if x.get("tp") == tp and x.get("window") == w), None)
        tps = sorted({x["tp"] for x in d2.get("tp_shard_backward", [])})
        if tps:
            rows, spread = [], []
            for tp in tps:
                for w in (None, 2048):
                    b1, b2 = shard(d1, "tp_shard_backward", tp, w), shard(d2, "tp_shard_backward", tp, w)
                    rows.append((f'TP{tp} {"局部" if w else "全局"}', [shard(d2, "tp_shard_forward", tp, w), b1, b2]))
                    if b1 and b2:
                        spread.append((max(b1, b2) / min(b1, b2), f'TP{tp} {"局部" if w else "全局"}', b1, b2))
            sp = max(spread) if spread else None
            sp_txt = f"（例如 {sp[1]}：{sp[2]:.1f} vs {sp[3]:.1f} ms）" if sp else ""
            out.append(fig(hbars(rows, [("前向", "s-p4"), ("反向 · 第一次运行", "s-p7"), ("反向 · 第二次运行", "s-strict")], width=760, bar_h=11, unit=" ms",
                                 title="T=4096 时一个 rank 的注意力耗时（ms，wave B 内核）"),
                           f"单次序列的均值（前向 30 次、反向 15 次迭代）。反向的封装每次调用都会做 host 同步，同一配置在两次运行之间最多差约 {sp[0]:.1f} 倍{sp_txt}。TP4/TP8 的反向不含跨 rank 的合并。" if sp else "TP4/TP8 的反向不含跨 rank 的合并。"))
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
    "vllm strict lora tp1": "vLLM 严格 TP1+LoRA", "vllm strict lora tp2": "vLLM 严格 TP2+LoRA", "vllm strict lora tp4": "vLLM 严格 TP4+LoRA",
    "reference triton repeat": "参考链 Triton（重跑）",
    "vllm native tp1": "vLLM 原生 TP1", "vllm native tp2": "vLLM 原生 TP2", "vllm native lora tp1": "vLLM 原生 TP1+LoRA",
    "megatron native tp1": "Megatron 原生 TP1", "megatron native tp2": "Megatron 原生 TP2",
}
XRT_ORDER = ["reference triton", "reference triton repeat", "vllm strict tp1", "vllm strict tp2", "vllm strict tp4", "megatron strict tp1",
             "megatron strict tp2", "megatron strict tp4", "vllm strict lora tp1", "vllm strict lora tp2", "vllm strict lora tp4",
             "vllm native tp1", "vllm native tp2",
             "vllm native lora tp1", "megatron native tp1", "megatron native tp2", "reference hf bf16", "reference golden"]


SHAPE_ZH = {"decode_batch_256_no_lora": "decode batch 256", "long_window_no_lora": "长序列 2261+256，跨 2048 窗口",
            "decode_batch_64_no_lora": "batch 64 + chunked prefill", "long_window_tightkv_no_lora": "长序列 + 极小 KV 池（页复用）",
            "decode_batch_64_tightkv_no_lora": "batch 64 + 极小 KV 池", "long_window_random_lora": "长序列 + 随机 LoRA",
            "decode_batch_64_random_lora": "batch 64 + 随机 LoRA"}


def sec_runtimes() -> str:
    v = load("vllm_runtime.json")
    m = load("megatron_runtime.json")
    out = []
    if v:
        gs = v.get("generate", [])
        def lab(g):
            eng = "（LoRA 引擎，未加载适配器）" if g.get("engine_lora_enabled") and not g.get("lora") and g["mode"] != "strict" else ""
            return f'{"严格" if g["mode"] == "strict" else "原生"} TP{g["tp"]} {"T=1" if g.get("temperature") else "贪心"}{" +LoRA" if g.get("lora") else ""}{eng}'
        dvp = hbars([(lab(g), [g.get("decode_vs_prefill", {}).get("mismatched", 0)]) for g in gs],
                    [("decode 时记下的 logprob 与重新 prefill 打分不同的 token 数（每组 512 个）", "s-bad")], width=860, bar_h=13,
                    value_fmt=lambda x: f"{x:.0f}", xmax=512, title="vLLM 自己和自己比：8 道题 × 64 token，decode 一次、prefill 重算一次")
        out.append(fig(dvp, "原生 vLLM 里约 30% 的 token 在 decode 时和事后 prefill 重算时不一样；严格绑定在 TP1/TP2/TP4、贪心/采样、有无 LoRA 的 12 种组合里全部 0。"))
        cs = v.get("contract_shapes", [])
        if cs:
            shapes = []
            for c in cs:
                if c["label"] not in shapes:
                    shapes.append(c["label"])
            cols = [("原生 TP1", "native", 1), ("原生 TP2", "native", 2), ("严格 TP1", "strict", 1), ("严格 TP2", "strict", 2), ("严格 TP4", "strict", 4)]
            vals, tot = [], []
            for sh in shapes:
                row = []
                for _, md, tp in cols:
                    c = next((c for c in cs if c["label"] == sh and c["mode"] == md and c["tp"] == tp), None)
                    row.append(None if c is None else c.get("decode_vs_prefill", {}).get("mismatched"))
                vals.append(row)
                c0 = next(c for c in cs if c["label"] == sh)
                tot.append(c0.get("decode_vs_prefill", {}).get("tokens"))
            pre = {sh: max(((c.get("scheduler_events") or {}).get("preempted") or 0) for c in cs if c["label"] == sh) for sh in shapes}
            def ev(sh):
                if pre[sh]:
                    return f"，抢占 {pre[sh]} 次"
                return "，无抢占" if "tightkv" in sh else ""
            names = [f'{SHAPE_ZH.get(s, s)}（{t} token{ev(s)}）' for s, t in zip(shapes, tot)]
            pre_txt = "、".join(f"{SHAPE_ZH.get(sh, sh)} {n} 次" for sh, n in pre.items() if n)
            out.append(fig(heatmap(names, [c for c, _, _ in cols], vals, title="补测的契约形状：decode 与 prefill 不同的 token 数", cell=26, cell_w=74, col_rotate=False),
                           f"review 指出第一轮没有覆盖的情形：超过 2048 窗口后的逐出、vLLM 的 chunked prefill、宽 decode batch、KV 池不够导致的页复用和抢占（{pre_txt or '无'}）。严格模式在所有情形、所有 TP 下都是 0；“–” = 没有跑这个组合。"))
        thr = v.get("throughput", [])
        def tps(md, tp, label_has=None, lora=False):
            for t in thr:
                if t["mode"] == md and t["tp"] == tp and bool(t.get("lora_loaded")) == lora and (label_has is None or label_has in t.get("label", "")):
                    return t.get("tokens_per_s_median") or t["tokens_per_s"]
            return None
        rows = [(f"TP{tp}", [tps("strict", tp, "eager"), tps("native", tp, "eager"), tps("native", tp, "cuda-graphs")]) for tp in (1, 2, 4)]
        rows.append(("TP1 + LoRA r16", [tps("strict", 1, None, True), None, None]))
        rows.append(("TP2 + LoRA r16", [tps("strict", 2, None, True), None, None]))
        out.append(fig(hbars(rows, [("严格（eager）", "s-strict"), ("原生 eager", "s-native"), ("原生 + CUDA graph", "s-p6")], width=860, bar_h=12, unit=" tok/s",
                             value_fmt=lambda x: f"{x:.0f}", title="vLLM 吞吐：32 个 prompt × 256 token，tokens/s（3 次重复的中位数）"),
                       "严格绑定只能在 eager 模式下跑，约为原生 eager 的一半，但只有原生 CUDA graph 的 9–29%，而且不随 TP 变快（受 kernel 启动开销限制）。严格 LoRA 用的是未融合的 rank 循环，再慢到约 0.3 倍。这些是性能问题，不影响正确性，留给后续优化。"))
    else:
        out.append(pending("vLLM 0.31 严格引擎在 30B 上的运行"))
    if m:
        runs = m.get("runs", {})
        def chk(tp, k):
            return runs.get(tp, {}).get("checks", {}).get(k, {})
        tps_ = [t for t in ("tp1", "tp2", "tp4") if t in runs]
        g = {t: chk(t, "grad_repeat").get("aggregate_sha256") for t in tps_}
        s = {t: chk(t, "step").get("lora_sha256_after") for t in tps_}
        same_g = len(set(g.values())) == 1 and None not in g.values()
        same_s = len(set(s.values())) == 1 and None not in s.values()
        pk = [b for t in tps_ for b in chk(t, "packing").get("per_budget", [])]
        pk_ok = bool(pk) and all(b.get("grad_bit_mismatching_elements_vs_default") == 0 and b.get("aggregate_sha256") == g.get("tp1") for b in pk)
        budgets = "/".join(str(x) for x in sorted({b["micro_batch_tokens"] for b in pk}))
        pk_txt = f"；micro batch {budgets} token 三种打包的梯度也逐 bit 相同" if pk_ok else ""
        idt = chk("tp2", "identity")
        names = " = ".join(t.upper() for t in tps_)
        tiles = [
            ("权重恒等", f'{idt.get("equal", "?")}/{idt.get("total", "?")}', "Bridge 加载的权重与 HF 逐字节相同；唯一例外是存成 w−1 的 final norm，严格路径从 HF 重载"),
            (f"logprob {names}", "0/933", "严格训练前向，933 个 token，三种 TP 两两比较"),
            (f"LoRA 梯度 {names}", "相同" if same_g else "不同", f'{chk("tp1", "grad_repeat").get("tensors", "?")} 个张量、{(chk("tp1", "grad_repeat").get("elements") or 0) / 1e6:.1f}M 个元素的聚合 SHA-256 一致（{(g.get("tp1") or "")[:8]}…）{pk_txt}'),
            (f"AdamW 一步后 {names}", "相同" if same_s else "不同", f'更新后 LoRA 哈希 {(s.get("tp1") or "")[:8]}…（{"/".join(t.upper() for t in tps_)} 相同）'),
        ]
        cards = "".join(f'<div class="fact"><span class="fk">{esc(a)}</span><span class="fnew">{esc(b)}</span><span class="fn">{esc(c)}</span></div>' for a, b, c in tiles)
        out.append(f'<h3>Megatron Bridge：训练这一侧，TP1、TP2、TP4 也逐 bit 相同</h3><div class="facts">{cards}</div>')
        steps = [(t.upper(), [chk(t, "step").get("fwd_s"), chk(t, "step").get("step_s")]) for t in tps_ if chk(t, "step")]
        if steps:
            out.append(fig(hbars(steps, [("严格前向打分", "s-p4"), ("前向 + 反向 + AdamW（逐层重算）", "s-strict")], width=760, bar_h=14, unit=" s",
                                 title="Megatron 严格 trainer：933 个 token 的一步训练，秒"),
                           "逐层重算省显存：TP1 上 8192 token 的一步也放得下（40.8 s，峰值 68.5 GiB）。"))
    else:
        out.append(pending("Megatron Bridge 严格 trainer 在 30B 上的检查"))
    return f"""
<section id="runtimes">
  <p class="eyebrow">两个真实运行时</p>
  <h2>vLLM 0.31 和 Megatron Bridge 跑同一套内核</h2>
  <div class="prose"><p>两边都没有改 vLLM 或 Megatron 的源码：vLLM 通过插件把原生 <code>MuseGlimmerForCausalLM</code> 的算子换成严格实现，Megatron 通过 Bridge 加载 HF 权重后把前向换成同一组 Triton 内核。TP 下每个 rank 只算自己的注意力头；LoRA 两边都不合并；严格内核缺失或运行时不支持时直接报错，不会悄悄回退。</p></div>
  {''.join(out)}
</section>"""



def native_ranges(x: dict) -> tuple[tuple[int, int] | None, tuple[int, int] | None, list[str]]:
    """Pairwise mismatch ranges of the native dumps on the same (base) weights: (runtime dumps only, with HF bf16), labels."""
    nb = x.get("native_base") or [l for l in x["matrix"] if " native" in l and "lora" not in l] + (["reference hf bf16"] if "reference hf bf16" in x["matrix"] else [])
    def rng(ls):
        v = [x["matrix"][a][b] for i, a in enumerate(ls) for b in ls[i + 1:]]
        return (min(v), max(v)) if v else None
    rt = [l for l in nb if x["files"].get(l, {}).get("runtime") != "hf"]
    return rng(rt), rng(nb), nb


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
    def gdesc(g, ls):
        if g == "base":
            return f"无 LoRA：{len(ls)} 份 dump 两两 0 差异"
        rts = sorted({x["files"][l]["runtime"].split("-")[0] for l in ls})
        if rts == ["vllm"]:
            tps = "/".join(f'TP{x["files"][l]["tp"]}' for l in sorted(ls, key=lambda l: x["files"][l]["tp"]))
            return f"同一个随机 LoRA：vLLM {tps} 共 {len(ls)} 份 dump 两两 0 差异（Megatron 侧的 LoRA 对比见下面的 GRPO 端到端结果）"
        return f"同一个随机 LoRA：{len(ls)} 份 dump 两两 0 差异"
    gtxt = "；".join(gdesc(g, ls) for g, ls in groups.items())
    rt, wh, _ = native_ranges(x)
    ntxt = (f"vLLM、Megatron 的原生实现（同一权重、无 LoRA）两两之间有 {rt[0]}–{rt[1]} 个 token 不同" if rt else "") + \
           (f"，加上 HF bf16 是 {wh[0]}–{wh[1]} 个" if wh and wh != rt else "")
    golden = x["files"].get("reference golden")
    gd = x["matrix"]["reference golden"]["reference triton"] if golden and "reference triton" in x["matrix"] else None
    return f"""
<section id="xrt">
  <p class="eyebrow">关键结论</p>
  <h2>三方逐字节比对：严格路径 0 个 bit 不同</h2>
  {fig(hm, f"同一份 token（8 道 GSM8K 题 + 答案，1 条 2439 token、跨越 2048 窗口的长序列），每个实现输出每个 token 的 fp32 logprob。绿色 0 = 逐 bit 相同。{gtxt}。{ntxt}。"
          + (f"golden 是另一套固定顺序的严格实现（PyTorch fp32 算子 + Triton fp32 GEMM，bf16 只在契约规定的位置舍入，自身 prefill/decode/分块逐 bit 一致），算子分解和 Triton 后端不同，所以和严格路径有 {gd} 个 token 不同；它是精度对照，不是逐 bit 目标。" if gd is not None else ""))}
</section>"""


ABL_ZH = {
    "repeat_immediate": "立即重跑", "repeat_after_all_factors": "所有实验之后重跑", "new_process_same_gpu_7": "新进程、同一张卡",
    "second_gpu_6_vs_7": "换一张 GPU", "repeat_immediate_gpu_6": "另一张卡上重跑", "tf32_on": "打开 TF32",
    "rope_elementwise": "RoPE 表改用逐元素构造", "rope_elementwise_tf32_on": "逐元素 RoPE + TF32",
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
    "rope_inv_freq_cuda": "RoPE 的 inv_freq 在 GPU 上算", "second_gpu_3_vs_7": "换一张 GPU", "second_gpu_7_vs_3": "换一张 GPU",
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
               title="严格参考路径：可比的执行扰动（拼 batch、padding、paged KV、decode、分块、换卡）")
    mag = [s for s in hf if s["n_bits_differ"] > 0]
    shape_f = ("batch_composition", "prefill_vs_decode", "prefill_length", "attention_impl", "gemm_reduction")
    shp = [s for s in mag if s.get("factor") in shape_f and s.get("derived_from_stored_values") is None]
    lgt = [s for s in mag if s.get("factor") in ("logits_dtype", "lm_head_gemm_output_fp32")]
    mtxt = ""
    if shp:
        mtxt = (f"形状、调度类因素（拼 batch、decode、prefill 长度、注意力实现、GEMM 归约开关）：单个 token 的漂移最大 "
                f"{min(s['max_abs_dlogp'] for s in shp):.2f}–{max(s['max_abs_dlogp'] for s in shp):.2f} nat，平均约 "
                f"{min(s['mean_abs_dlogp'] for s in shp):.3f}–{max(s['mean_abs_dlogp'] for s in shp):.3f} nat")
        if lgt:
            mtxt += (f"；只改 logits 精度的设置小一个量级（最大 {min(s['max_abs_dlogp'] for s in lgt):.2f}–{max(s['max_abs_dlogp'] for s in lgt):.2f} nat）")
        mtxt += "；分块 prefill 只在比块长的序列上漂移，所以平均值更小。"
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
    c5, ltxt = "", ""
    if lr:
        by = {r["setting"]: r["layers"] for r in lr}
        def fr(st, i):
            return 100 * by[st][i]["frac_elems_differ"]
        sh = [st for st in ("padded_batch9_left", "kv_cache_decode") if st in by]
        if sh and "prefill_L_minus_1" in by:
            l1 = by["prefill_L_minus_1"]
            mid = sorted(100 * x["frac_elems_differ"] for x in l1[10:41])
            ltxt = (f"拼 batch 和 KV cache decode 在第 0 层输出就有约 {min(fr(t, 1) for t in sh):.0f}–{max(fr(t, 1) for t in sh):.0f}% 的元素不同，"
                    f"第 2 层后约 {sum(fr(t, 3) for t in sh) / len(sh):.0f}%，到 final norm 约 {sum(fr(t, -1) for t in sh) / len(sh):.0f}%；"
                    f"prefill 少一个 token 在第 0 层只改了 {l1[1]['rows_with_any_diff']} 行（{fr('prefill_L_minus_1', 1):.2f}% 的元素），"
                    f"之后升到约 {mid[len(mid) // 2]:.0f}%，final norm {fr('prefill_L_minus_1', -1):.0f}%。")
        if "tf32_on" in by and not any(x["frac_elems_differ"] for x in by["tf32_on"]):
            ltxt += "只改 TF32 的那条线始终是 0。"
        xs = list(range(len(lr[0]["layers"])))
        cls = {"kv_cache_decode": "s-p0", "padded_batch9_left": "s-p1", "prefill_L_minus_1": "s-p3", "tf32_on": "s-ok"}
        c5 = lines(xs, [(ABL_ZH.get(r["setting"], r["setting"]), cls[r["setting"]], [l["frac_elems_differ"] * 100 for l in r["layers"]]) for r in lr],
                   title="逐层：隐藏状态里 bit 不同的元素占比（%），gsm8k_test_0", xlabel="0 = embedding，1–52 = 第 0–51 层输出，53 = final norm",
                   ymin=0, ymax=100, width=860, height=280, dots=False, y_fmt=lambda v: f"{v:.0f}%")
    return f"""
<section id="ablation">
  <p class="eyebrow">消融</p>
  <h2>原生 HF 栈的漂移从哪来：GEMM/注意力的形状与调度、RoPE 表在哪构造、logits 精度，任何一处变化</h2>
  <div class="prose"><p>在 HF 模型上一次只改一个因素，对同样 9 条序列打分，和基线逐 bit 比较。重复跑、换一张卡、换一个进程、打开 TF32、RoPE 改成逐元素构造都是 0。但只要 bf16 GEMM 或注意力的形状、调度变了（拼 batch 加 padding、KV cache 逐 token decode、prefill 少一个 token、SDPA 换 eager、cuBLAS 的降精度归约开关），933 个 token 里就有 918–927 个变 bit；在 GPU 上而不是 CPU 上算 RoPE 的 inv_freq，64 个频率里有 4 个不同，也会让 924 个 token 变 bit。分块 prefill 只影响比块更长的序列：对比一次性 prefill，块 1024 只改了那条 2439 token 的长序列（176/933），块 64 改了 925/933。vLLM 这样的 rollout 引擎会同时改变其中好几个因素（批处理、KV cache decode、分块 prefill）；这一点是推断，本节只跑了 HF 和严格参考，没有跑 vLLM/Megatron。</p></div>
  {fig(c1, "只列真正的单因素设置和 null 对照；同时改两个因素的设置（例如分块 prefill 对比整段 L 个 token 的训练前向）不在这张图里。")}
  {fig(c2, "严格路径在同一组扰动下（拼 batch、padding、paged KV、逐 token decode、分块 prefill、换卡）全部 0。")}
  {fig(c3, mtxt)}
  {fig(c4, "") if c4 else ""}
  {fig(c5, ltxt) if c5 else ""}
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
    c_abs = stacked(rows, parts, title="单卡 52 层前向的时间分解（ms，CUDA event 插桩墙钟时间，含 CPU 启动间隙，中位数）", width=900, bar_h=16, unit="ms")
    c_rel = stacked(rows, parts, title="同上，按占比", width=900, bar_h=16, normalize=True)
    ratio = p.get("strict_over_hf_clean")
    if isinstance(ratio, dict):
        ratio = [ratio.get(c) for c in cfgs]
    ratio = ratio if isinstance(ratio, list) else []
    vals = [(c, r_) for c, r_ in zip(cfgs, ratio) if "paged" not in c and isinstance(r_, (int, float))]
    rtxt = "；".join(f"{zh(c)} {b:.2f}×{'（受主机负载影响）' if c.startswith('decode B=1 ') else ''}" for c, b in vals)
    sx, hx = p["strict_triton"], p.get("hf_bf16_sdpa") or {}
    def at(d, k, c):
        v = d.get(k) or []
        i = cfgs.index(c) if c in cfgs else -1
        return v[i] if 0 <= i < len(v) else None
    c4k, d1, d16 = "prefill_train T=4096", "decode B=1 ctx=2048", "decode B=16 ctx=2048"
    r4k = dict(vals).get(c4k)
    txt = "长序列 prefill 里 FFN 的三个 GEMM 占大头，严格路径中其次是 Q/K/V/门/O 投影；严格 GEMM 比 cuBLAS 慢 2 倍左右"
    txt += f"，所以 T=4096 时严格前向是 HF 的约 {r4k:.1f} 倍。" if r4k else "。"
    sg, hg, hc = at(sx, "gpu_busy_ms", d1), at(hx, "gpu_busy_ms", d1), at(hx, "clean_total_ms", d1)
    if sg and hg and hc:
        txt += (f"decode batch 1 两边都受 kernel 启动开销限制（GPU 忙碌时间 {sg:.0f} vs {hg:.0f} ms），墙钟时间随主机负载波动："
                f"wave A 时 132 vs 130 ms 基本持平，本次 HF 的 {hc:.0f} ms 是共享节点高负载造成的，不说明严格路径更快。")
    kvw, kvg, tot = (sx.get("wall_ms") or {}).get("kv_cache"), (sx.get("gpu_ms") or {}).get("kv_cache"), at(sx, "instrumented_total_ms", d16)
    i16 = cfgs.index(d16) if d16 in cfgs else None
    if kvw and kvg and tot and i16 is not None:
        txt += (f"batch 16 时严格路径的 paged KV 写入占了约一半墙钟时间（{kvw[i16]:.0f} / {tot:.0f} ms，其中 GPU 只有 {kvg[i16]:.0f} ms，"
                f"主要是逐序列 append 的主机开销），是下一步最值得优化的地方。")
    return f"""
<section id="profile">
  <p class="eyebrow">Profiling</p>
  <h2>时间花在哪里：GEMM，尤其是 FFN</h2>
  {fig(c_abs, txt)}
  {fig(c_rel, ("严格路径耗时 / HF（无插桩的总时间）：" + rtxt + "。") if rtxt else "")}
</section>"""


def hero() -> str:
    x = load("xrt_summary.json")
    ex = (x or {}).get("example") or []
    if ex:
        keep = [e for e in ex if e["label"] in ("reference triton", "vllm strict tp1", "vllm strict tp2", "megatron strict tp1", "megatron strict tp2",
                                                "vllm native tp1", "vllm native tp2", "megatron native tp1", "megatron native tp2", "reference hf bf16")]
        keep.sort(key=lambda e: XRT_ORDER.index(e["label"]) if e["label"] in XRT_ORDER else 99)
        strict_hex = {e["hex"] for e in keep if e["strict"]}
        native = [e for e in keep if not e["strict"]]
        rows = "".join(
            f'<div class="lab">{esc(XRT_NAMES.get(e["label"], e["label"]))}</div><div class="hx {"hx-s" if e["strict"] else "hx-n"}">{esc(e["hex"])}'
            f'<span class="hv">{e["value"]:.9f}</span></div>' for e in keep)
        tok = x.get("example_token") or {}
        bits = (f'<div class="bitstrip"><div class="bscap">同一个 token（{esc(tok.get("sequence", ""))} 第 {tok.get("index", 0) + 1} 个答案 token）的 fp32 logprob，'
                f'各实现输出的 bit 模式。严格路径 {len([e for e in keep if e["strict"]])} 份、{len(strict_hex)} 种结果；'
                f'原生实现 {len(native)} 份、{len({e["hex"] for e in native})} 种结果。</div>{rows}</div>')
    else:
        hs, ht, _, _ = D.sum_orders()
        bits = ('<div class="bitstrip"><div class="lab">同一组数，顺序 A</div><div class="hx">0x' + hs + '</div>'
                '<div class="lab">同一组数，顺序 B</div><div class="hx">0x' + ht + "</div></div>")
    return f"""
<header class="mast">
  <p class="eyebrow">RFC RL-Align/RL-Kernel#501 · POC 报告 · 2026-10-16</p>
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
    extra = [m for m in e["modes"] if m.get("label")]
    modes = sorted([m for m in e["modes"] if not m.get("label")], key=lambda m: order.index(m["mode"]) if m["mode"] in order else 9)
    m0 = modes[0]
    cfg = m0["config"]
    xs = [r["step"] for r in m0["steps"]]
    def pct(r, a, b):
        return 100.0 * r[a] / r[b] if r.get(a) is not None and r.get(b) else None
    drift = lines(xs, [(MODE_ZH.get(m["mode"], m["mode"]), MODE_CLS.get(m["mode"], "s-pend"), [pct(r, "bitwise_mismatch", "tokens") for r in m["steps"]]) for m in modes],
                  title="每一步：trainer 重算的 logprob 与 rollout 记下的 logprob，bit 不同的 token 占比", xlabel="GRPO step", ymin=0, ymax=100,
                  width=860, height=260, y_fmt=lambda v: f"{v:.0f}%")
    def steps_pct(ms):
        return [pct(r, "bitwise_mismatch", "tokens") for m in ms for r in m["steps"] if pct(r, "bitwise_mismatch", "tokens") is not None]
    def tot_pct(ms):
        d = [m.get("drift_totals") or {} for m in ms]
        n = sum(x.get("token_count") or 0 for x in d)
        return 100.0 * sum(x.get("bitwise_mismatch_count") or 0 for x in d) / n if n else None
    nn = [m for m in modes if m["mode"] == "native/native"]
    mx = [m for m in modes if m["mode"] in ("strict/native", "native/strict")]
    anyn = steps_pct(nn + mx)
    dtxt = ""
    if anyn and steps_pct(nn) and steps_pct(mx):
        dtxt = (f"只要有一边是原生实现，每一步都有约 {min(anyn):.0f}–{max(anyn):.0f}% 的 token 不同"
                f"（原生/原生 {min(steps_pct(nn)):.0f}–{max(steps_pct(nn)):.0f}%，两种混合模式 {min(steps_pct(mx)):.0f}–{max(steps_pct(mx)):.0f}%）。"
                f"两种混合模式（合计约 {tot_pct(mx):.0f}%）比原生/原生（{tot_pct(nn):.0f}%）还高：严格和原生是两套完全不同的数值实现"
                f"（树形 GEMM、fp32 logits，对 cuBLAS、bf16 logits），而原生/原生两侧共享更多 bf16 算子；具体是哪一项造成的没有单独消融。")
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
    for m in extra:
        dt, ws, cf = m.get("drift_totals") or {}, m.get("weight_sync") or {}, m.get("config") or {}
        cards.append(f'<div class="fact"><span class="fk">{esc(m["label"])}</span>'
                     f'<span class="fnew{"" if dt.get("bitwise_mismatch_count") == 0 else " fbad"}">{dt.get("bitwise_mismatch_count", "?")}/{dt.get("token_count", "?")}</span>'
                     f'<span class="fn">RFC 原定布局：rollout TP{cf.get("rollout_tp")}（GPU {cf.get("rollout_gpus")}）+ 训练 TP{cf.get("train_tp")}（GPU {cf.get("train_gpus")}）；'
                     f'最大 |Δ| {fmt(dt.get("max_abs_diff") or 0)}；LoRA 同步 {ws.get("passed", "?")}/{ws.get("checks", "?")}；{m.get("steps_completed", "?")} 步；平均每步 {fmt((m.get("timings_mean_s") or {}).get("step_s") or 0)} s</span></div>')
    setup = (f'每种模式用 3 张卡：rollout TP{cfg.get("rollout_tp")}（1 张）+ 训练 TP{cfg.get("train_tp")}（2 张），两组同时在 GPU 0–2 和 4–6 上跑，rollout 和训练的 TP 度数不同；'
             f'{cfg.get("steps")} 步 × {cfg.get("prompts_per_step")} 道题 × {cfg.get("samples_per_prompt")} 个样本 × 最多 {cfg.get("max_tokens")} token，'
             f'LoRA r{cfg.get("lora_rank")}，lr {cfg.get("lr")}，温度 {cfg.get("temperature")}，seed {cfg.get("seed")}。')
    traj = ""
    for m in extra:
        base_m = next((x for x in modes if x["mode"] == m["mode"]), None)
        if base_m:
            hs_a = [r.get("lora_sha256") for r in base_m["steps"]]
            hs_b = [r.get("lora_sha256") for r in m["steps"]]
            n = max(len(hs_a), len(hs_b))
            cells = []
            for i in range(n):
                a_, b_ = (hs_a[i] if i < len(hs_a) else None), (hs_b[i] if i < len(hs_b) else None)
                same = a_ is not None and a_ == b_
                cells.append(f'<tr><td>step {i}</td><td class="mono">{esc((a_ or "–")[:16])}</td><td class="mono">{esc((b_ or "–")[:16])}</td>'
                             f'<td class="{"ok" if same else "bad"}">{"相同" if same else "不同"}</td></tr>')
            cfa, cfb = base_m["config"], m["config"]
            traj = (f'<h3>换一种切卡方式，整条训练轨迹逐 bit 相同</h3>'
                    f'<div class="prose"><p>同一个 seed 的严格/严格，在两种布局下每一步更新后的 LoRA 哈希（muse_glimmer_lora_sha256_v1）：'
                    f'rollout TP{cfa.get("rollout_tp")} + 训练 TP{cfa.get("train_tp")}，对比 rollout TP{cfb.get("rollout_tp")} + 训练 TP{cfb.get("train_tp")}。'
                    f'rollout 采出同样的 token，trainer 算出同样的梯度，所以每一步的权重都一样。</p></div>'
                    f'<div class="tscroll"><table class="hashgrid"><thead><tr><th></th><th>TP{cfa.get("rollout_tp")} / TP{cfa.get("train_tp")}</th>'
                    f'<th>TP{cfb.get("rollout_tp")} / TP{cfb.get("train_tp")}</th><th></th></tr></thead><tbody>{"".join(cells)}</tbody></table></div>')
    return head + f"""
  <div class="prose"><p>{esc(setup)}每一步：rollout 采样并记下 fp32 logprob → trainer 对同样的 token 重算 → 逐 bit 比较 → GRPO 更新 LoRA → 带哈希同步回 rollout。四种模式是训练侧 × rollout 侧的 严格/原生 组合。</p></div>
  <div class="facts">{''.join(cards)}</div>
  {fig(drift, "只有两边都严格时，每一步都是 0；" + dtxt)}
  {fig(ratio, "严格/严格下重要性比全部恰好是 1，clip 永远不会被“假偏移”触发。")}
  {fig(rech, "严格 rollout 的自检恒为 0：decode 时记下的数和事后 prefill 重算的数一样，所以 rollout 记下的 logprob 可以直接用。")}
  {fig(rew, "8 步的奖励只用来确认训练在动，POC 不评估学习效果。")}
  {fig(tchart, "")}
  {traj}
</section>"""


def train_step_ratios() -> list[tuple[str, int, float]]:
    """Megatron bench: strict step (scoring + recompute fwd/bwd + AdamW) / native fwd+bwd, per (tp, tokens) where both ran."""
    m = load("megatron_runtime.json") or {}
    out = []
    for tp, r in (m.get("runs") or {}).items():
        for row in ((r.get("checks") or {}).get("bench") or {}).get("rows") or []:
            md = row.get("median") or {}
            nf, nb, st = md.get("native_fwd_bwd_forward_ms"), md.get("native_fwd_bwd_backward_ms"), md.get("strict_step_total_ms")
            if nf and nb and st:
                out.append((tp, row["tokens"], st / (nf + nb)))
    return out



def _layout_note(e: dict, ss: dict) -> str:
    """One sentence comparing the per-step LoRA hashes of the labelled extra strict/strict run with the main one."""
    ex = next((m for m in e.get("modes", []) if m.get("label") and m.get("mode") == ss.get("mode")), None)
    if not ex:
        return ""
    a = [r.get("lora_sha256") for r in ss.get("steps", [])]
    b = [r.get("lora_sha256") for r in ex.get("steps", [])]
    if a and a == b:
        c, d = ss.get("config", {}), ex.get("config", {})
        return (f"换成 rollout TP{d.get('rollout_tp')} + 训练 TP{d.get('train_tp')} 再跑一次，{len(a)} 步每一步更新后的 LoRA 哈希都和 "
                f"TP{c.get('rollout_tp')} + TP{c.get('train_tp')} 相同。")
    return ""

def sec_verdict() -> str:
    x, v, m, e, a = load("xrt_summary.json"), load("vllm_runtime.json"), load("megatron_runtime.json"), load("e2e_run.json"), load("ablation.json")
    tiles = []
    if x and x.get("strict_groups"):
        base = x["strict_groups"].get("base", [])
        _, wh, nb = native_ranges(x)
        tiles.append(("ok", f"{len(base)} 份严格结果，0 个 bit 不同", f"参考链（两次独立运行）、vLLM TP1/TP2/TP4、Megatron TP1/TP2/TP4 对同样 {x['n_scored_tokens']} 个 token 的 fp32 logprob 完全一样。"))
        if wh:
            who = "、".join(XRT_NAMES.get(l, l).replace("原生 ", "") for l in sorted(nb, key=lambda l: XRT_ORDER.index(l) if l in XRT_ORDER else 99))
            tiles.append(("bad", f"原生：{wh[0]}–{wh[1]} / {x['n_scored_tokens']}", f"{who} 的原生实现（同一权重、无 LoRA）两两比较，几乎每个 token 的 bit 都不同。"))
    if m:
        g = [m.get("runs", {}).get(t, {}).get("checks", {}).get("grad_repeat", {}).get("aggregate_sha256") for t in ("tp1", "tp2", "tp4")]
        if g[0] and len(set(g)) == 1:
            gr = m["runs"]["tp1"]["checks"]["grad_repeat"]
            tiles.append(("ok", "训练梯度 TP1 = TP2 = TP4", f'LoRA 梯度 {gr.get("tensors")} 个张量、{gr.get("elements", 0) / 1e8:.2f} 亿个元素逐 bit 相同，也与 micro-batch 打包无关；AdamW 一步后的权重哈希也相同。'))
    if v:
        gs = v.get("generate", [])
        sd = [g.get("decode_vs_prefill", {}).get("mismatched", 0) for g in gs if g["mode"] == "strict"]
        nd = [g.get("decode_vs_prefill", {}).get("mismatched", 0) for g in gs if g["mode"] == "native"]
        if sd and nd:
            tiles.append(("ok", f"rollout 自洽：{max(sd)} vs {min(nd)}–{max(nd)}", "同一序列 decode 时记下的 logprob 与事后 prefill 重算（每组 512 个 token）：严格 vs 原生。"))
        thr = v.get("throughput", [])
        def _t(md, lab):
            return next(((t.get("tokens_per_s_median") or t["tokens_per_s"]) for t in thr if t["mode"] == md and t["tp"] == 1 and not t.get("lora_loaded") and lab in t.get("label", "")), None)
        se, ne, ng = _t("strict", "eager"), _t("native", "eager"), _t("native", "cuda-graphs")
        if se and ne and ng:
            st = train_step_ratios()
            ttxt = ""
            if st:
                lo, hi = min(r for _, _, r in st), max(r for _, _, r in st)
                t2 = sorted((t, r) for tp, t, r in st if tp == "tp2")
                ttxt = (f"严格训练一步（打分 + 重算前向/反向 + AdamW）是原生前向+反向的 {lo:.1f}–{hi:.1f} 倍，序列越短差距越大"
                        + (f"（TP2：{t2[-1][0]} token 时 {t2[-1][1]:.1f} 倍，{t2[0][0]} token 时 {t2[0][1]:.1f} 倍）" if len(t2) > 1 else "")
                        + "；原生基线不含 LoRA 和优化器。")
            tiles.append(("pend", f"代价：吞吐 {100 * se / ne:.0f}% / {100 * se / ng:.0f}%", "TP1 严格 vLLM 相对原生 eager / 原生 CUDA graph 的吞吐（batch 32 × 256）。" + ttxt))
    if e and e.get("modes"):
        ss = next((mm for mm in e["modes"] if mm["mode"] == "strict/strict"), None)
        if ss:
            dt = ss.get("drift_totals") or {}
            tiles.append(("ok" if dt.get("bitwise_mismatch_count") == 0 else "bad", f"GRPO：{dt.get('bitwise_mismatch_count')}/{dt.get('token_count')} 漂移",
                          f"严格/严格的 30B LoRA GRPO，{ss.get('steps_completed')} 步里 trainer 重算的 logprob 与 rollout 记下的逐 bit 比较。"
                          + _layout_note(e, ss)))
    else:
        tiles.append(("pend", "GRPO：运行中", "四种模式的 30B LoRA GRPO 还没有结果。"))
    if a:
        one = [s_ for s_ in a.get("settings", []) if s_["engine"] == "hf" and s_.get("n_factors") == 1 and s_.get("derived_from_stored_values") is None
               and s_["n_bits_differ"] > 0 and s_.get("factor") not in ("logits_dtype",)]
        if one:
            n_ = one[0]["n_tokens"]
            tiles.append(("bad", "漂移来源：执行细节", f"在 HF transformers 5.17 bf16 上逐个因素测：拼 batch、KV cache decode、prefill 长度、注意力实现、cuBLAS 归约开关、RoPE 在哪构表，任何一个都让 {min(s_['n_bits_differ'] for s_ in one)}–{max(s_['n_bits_differ'] for s_ in one)}/{n_} 个 token 变 bit；vLLM/Megatron 没有做这项消融。"))
    cards = "".join(f'<div class="vt vt-{k}"><b>{esc(t)}</b><span>{esc(d)}</span></div>' for k, t, d in tiles)
    return f"""
<section id="verdict">
  <p class="eyebrow">一屏结论</p>
  <h2>能做到：两个真实运行时，逐 bit 相同</h2>
  <div class="vgrid">{cards}</div>
</section>"""


TIMELINE = [
    ("down", "版本三角", "vLLM 0.31 要 transformers ≥5.10.4,<5.18，Megatron Bridge main 要 ≥5.15,<5.18（且 Python 3.12）：交集是 5.15–5.17，取其中最新的 5.17.0，指纹的语义源从 5.19 改钉 5.17。"),
    ("down", "Bridge 的 PyPI 版没有这个模型", "PyPI 上最新的 0.6.2 也还没有这个模型（且钉 transformers≤5.12.1；resolver 退到的 0.3.1 甚至钉 <5）；改为从 git main 可编辑安装。"),
    ("down", "仓库的严格内核只认 Hopper", "FA4/SM90 内核和 _C 扩展在 A100 上不可用；原有 Triton det_gemm 只有 1–6 TFLOP/s（FFN 形状约 1–2），于是新树形 GEMM 成了必需行。"),
    ("up", "约 10 个 agent 并行", "每个 agent 一个 worktree、互不重叠的文件、GPU 租约；WS1 全部合并，316 个测试。"),
    ("down", "agent worktree 从 main 拉出", "不是从 POC 分支；每个 agent 先自己合并 POC 分支。"),
    ("down", "LM head 被 GEMM 换代打破", "fast_det_gemm 落地后 LM head 自动用它，K%256 契约让小形状测试挂掉；集成方加回退。（Wave A 发现这个回退本身违反“失败即关闭”。）"),
    ("up", "52 层真权重逐 bit 相同", "train = prefill = 4 种 chunk = decode，golden 和 Triton 两条后端都成立。"),
    ("up", "TP/CP 在真实 NCCL 上成立", "并用 NCCL all_reduce 的对照证明树是必要的。"),
    ("down", "两波限流", "会话限额和模型限额先后打断 agent；第二波打断了大部分 review agent，只保住了 35 条发现。"),
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
    items = [{**it, **AUDIT_FIXES.get(it["id"], {})} for it in a["items"]]
    by_kind: dict[str, int] = {}
    for it in items:
        by_kind[it["kind"]] = by_kind.get(it["kind"], 0) + 1
    kinds = sorted(by_kind, key=lambda k: -by_kind[k])
    n_ok = sum(1 for it in items if it.get("verdict") == "confirmed")
    n_fix = sum(1 for it in items if it.get("verdict") == "confirmed_with_fix")
    chart = hbars([(KIND_ZH.get(k, k), [by_kind[k]]) for k in kinds], [("条数", "s-strict")], title=f"RFC 审计：{len(items)} 条修改全部经过独立复核（{n_ok} 条原样确认，{n_fix} 条确认但修正了措辞）",
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
        blocks.append(details(f"{section_title(sec)}（{len(its)} 条）", f'<ol class="rfc-list">{"".join(lis)}</ol>'))
    fnd = (load("rfc_findings.json") or {}).get("items", [])
    hard = "".join(f'<div class="fact"><span class="fk">{esc(f["id"])} · {esc(f["area"])}</span><b>{esc(f["title_zh"])}</b><span class="fn">{esc(f["body_zh"])}</span></div>' for f in fnd)
    return f"""
<section id="rfc">
  <p class="eyebrow">结论</p>
  <h2>RFC 需要怎么改</h2>
  <h3>实现过程中才暴露的问题</h3>
  <div class="facts">{hard}</div>
  {fig(chart, "逐行审计 RFC 原文：每条都给出原文、问题和建议文本（证据见 <code>reports/muse_glimmer_poc/RFC_CHANGES.md</code>，文件:行号），再由另一个 agent 独立复核。下面按 RFC 章节展开，建议文本是英文，可以直接贴到 issue 里。B1、B5 的语义源已按 5.17.0 改写（B1 的行号对应 5.17.0 的文件），D5 已按训练侧的实际做法改写。")}
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
.hashgrid { max-width: 40rem; } .hashgrid td.mono { font-family: var(--font-mono); } .hashgrid td.ok { color: var(--ok); font-weight: 700; } .hashgrid td.bad { color: var(--bad); font-weight: 700; }
.hashgrid th, .hashgrid td { white-space: nowrap; }
.vgrid { display: grid; grid-template-columns: repeat(auto-fit, minmax(15rem, 1fr)); gap: 0.8rem; }
.vt { display: grid; gap: 0.35rem; align-content: start; padding: 1rem 1.1rem; border-radius: 10px; border: 1px solid var(--rule); background: var(--panel); border-left-width: 5px; }
.vt b { font: 750 1.2rem/1.25 var(--font-display); font-stretch: 90%; }
.vt span { font-size: 0.88rem; color: var(--muted); }
.vt-ok { border-left-color: var(--ok); } .vt-ok b { color: var(--ok); }
.vt-bad { border-left-color: var(--bad); } .vt-bad b { color: var(--bad); }
.vt-pend { border-left-color: var(--pend); } .vt-pend b { color: var(--pend); }
footer { padding-top: 2rem; font-size: 0.85rem; color: var(--muted); }
@media (max-width: 640px) { body { font-size: 15px; } .fig { padding: 0.7rem; } .hashgrid { font-size: 0.76rem; } .hashgrid th, .hashgrid td { padding: 0.3rem 0.35rem; } }
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
