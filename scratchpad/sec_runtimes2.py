SHAPE_ZH = {"decode_batch_256_no_lora": "decode batch 256", "long_window_no_lora": "长序列 2261+256，跨 2048 窗口",
            "decode_batch_64_no_lora": "batch 64 + chunked prefill", "long_window_tightkv_no_lora": "长序列 + 极小 KV 池（抢占）",
            "decode_batch_64_tightkv_no_lora": "batch 64 + 极小 KV 池", "long_window_random_lora": "长序列 + 随机 LoRA",
            "decode_batch_64_random_lora": "batch 64 + 随机 LoRA"}


def sec_runtimes() -> str:
    v = load("vllm_runtime.json")
    m = load("megatron_runtime.json")
    out = []
    if v:
        gs = v.get("generate", [])
        def lab(g):
            return f'{"严格" if g["mode"] == "strict" else "原生"} TP{g["tp"]} {"T=1" if g.get("temperature") else "贪心"}{" +LoRA" if g.get("lora") else ""}'
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
            names = [f'{SHAPE_ZH.get(s, s)}（{t} token）' for s, t in zip(shapes, tot)]
            out.append(fig(heatmap(names, [c for c, _, _ in cols], vals, title="补测的契约形状：decode 与 prefill 不同的 token 数", cell=26, cell_w=74, col_rotate=False),
                           "review 指出第一轮没有覆盖的情形：超过 2048 窗口后的逐出、vLLM 的 chunked prefill、宽 decode batch、KV 池不够导致的抢占（13 次）。严格模式在所有情形、所有 TP 下都是 0；“–” = 没有跑这个组合。"))
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
        idt = chk("tp2", "identity")
        names = " = ".join(t.upper() for t in tps_)
        tiles = [
            ("权重恒等", f'{idt.get("equal", "?")}/{idt.get("total", "?")}', "Bridge 加载的权重与 HF 逐字节相同；唯一例外是存成 w−1 的 final norm，严格路径从 HF 重载"),
            (f"logprob {names}", "0/933", "严格训练前向，933 个 token，三种 TP 两两比较"),
            (f"LoRA 梯度 {names}", "相同" if same_g else "不同", f'{chk("tp1", "grad_repeat").get("tensors", "?")} 个张量、{(chk("tp1", "grad_repeat").get("elements") or 0) / 1e6:.1f}M 个元素的聚合 SHA-256 一致（{(g.get("tp1") or "")[:8]}…）'),
            (f"AdamW 一步后 {names}", "相同" if same_s else "不同", f'更新后 LoRA 哈希 {(s.get("tp1") or "")[:8]}…；micro batch 512/4096/8192 三种打包也一样'),
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
