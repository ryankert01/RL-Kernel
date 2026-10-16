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
    hf = [s for s in st if s["engine"] == "hf"]
    hf.sort(key=lambda s: -s["n_bits_differ"])
    n = hf[0]["n_tokens"] if hf else 933
    c1 = hbars([(ABL_ZH.get(s["setting"], s["setting"]), [s["n_bits_differ"]]) for s in hf], [("bit 不同的 token 数", "s-native")],
               width=860, bar_h=14, xmax=n, value_fmt=lambda v: f"{v:.0f}", title=f"HF transformers 5.17（原生 bf16）：每次只改一个因素，{n} 个 token 里 logprob 的 bit 变了多少")
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
  <h2>原生栈的漂移从哪来：几乎任何一个执行细节</h2>
  <div class="prose"><p>在 HF 模型上一次只改一个因素，对同样 9 条序列打分，和基线逐 bit 比较。TF32、RoPE 表怎么构造、换一张卡、换一个进程都不影响结果（null 对照全是 0）；但 batch 怎么拼、有没有 KV cache、prefill 怎么切、注意力用哪个实现、logits 用什么精度，<b>任何一个</b>都会让 98–99% 的 token 变 bit。rollout 引擎恰好同时改了这几个因素。</p></div>
  {fig(c1, "")}
  {fig(c2, "严格路径在同一组扰动下（拼 batch、padding、paged KV、逐 token decode、分块 prefill、换卡）全部 0。")}
  {fig(c3, "单个 token 的漂移可以到 0.4–1.1 nat，平均约 0.02 nat。")}
  {fig(c4, "") if c4 else ""}
  {fig(c5, "差异从第 0 层就出现，三四层内扩散到约 85% 的隐藏元素；只改 TF32 的那条线始终是 0。") if c5 else ""}
</section>"""


PROF_ZH = {"embedding": "embedding", "norms_residual": "范数 + 残差", "qkvgo_gemm": "Q/K/V/门/O GEMM", "qknorm_rope": "QK-norm + RoPE",
           "attention_global": "全局注意力", "attention_local": "局部注意力", "kv_cache": "KV cache", "ffn_gemm": "FFN GEMM",
           "ffn_act": "SwiGLU", "lm_head_softcap_logprob": "LM head + softcap + logprob", "other": "其他"}
PROF_CLS = {"embedding": "s-p7", "norms_residual": "s-p4", "qkvgo_gemm": "s-p0", "qknorm_rope": "s-p3", "attention_global": "s-p5",
            "attention_local": "s-p1", "kv_cache": "s-p6", "ffn_gemm": "s-strict", "ffn_act": "s-p2", "lm_head_softcap_logprob": "s-native", "other": "s-pend"}


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
  {fig(c_abs, "长序列 prefill 里 FFN 的三个 GEMM 占大头，其次是 Q/K/V/门/O 投影；decode 时 KV cache 和逐元素算子的比例上升。")}
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
