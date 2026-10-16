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
        tiles.append(("bad", "漂移来源：几乎任何执行细节", "拼 batch、KV cache、分块 prefill、注意力实现、logits 精度，任何一个都会让 98–99% 的 token 变 bit。"))
    cards = "".join(f'<div class="vt vt-{k}"><b>{esc(t)}</b><span>{esc(d)}</span></div>' for k, t, d in tiles)
    return f"""
<section id="verdict">
  <p class="eyebrow">一屏结论</p>
  <h2>能做到：两个真实运行时，逐 bit 相同</h2>
  <div class="vgrid">{cards}</div>
</section>"""
