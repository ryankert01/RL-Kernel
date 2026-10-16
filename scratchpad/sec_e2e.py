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
                     f'<span class="fnew">{dt.get("bitwise_mismatch_count", "?")}/{dt.get("token_count", "?")}</span>'
                     f'<span class="fn">token bit 不同；最大 |Δ| {fmt(dt.get("max_abs_diff") or 0)}；第一次漂移：{esc(where)}；LoRA 同步校验 {ws.get("passed", "?")}/{ws.get("checks", "?")} 通过；{m.get("steps_completed", "?")} 步</span></div>')
    setup = (f'rollout TP{cfg.get("rollout_tp")}（GPU {cfg.get("rollout_gpus")}）+ 训练 TP{cfg.get("train_tp")}（GPU {cfg.get("train_gpus")}），'
             f'{cfg.get("steps")} 步 × {cfg.get("prompts_per_step")} 道题 × {cfg.get("samples_per_prompt")} 个样本 × 最多 {cfg.get("max_tokens")} token，'
             f'LoRA r{cfg.get("lora_rank")}，lr {cfg.get("lr")}，温度 {cfg.get("temperature")}，seed {cfg.get("seed")}。')
    return head + f"""
  <div class="prose"><p>{esc(setup)}每一步：rollout 采样并记下 fp32 logprob → trainer 对同样的 token 重算 → 逐 bit 比较 → GRPO 更新 LoRA → 带哈希同步回 rollout。四种模式是训练侧 × rollout 侧的 严格/原生 组合。</p></div>
  <div class="facts">{''.join(cards)}</div>
  {fig(drift, "只有两边都严格时，每一步都是 0；只要有一边是原生实现，几乎每个 token 都不同。")}
  {fig(ratio, "严格/严格下重要性比全部恰好是 1，clip 永远不会被“假偏移”触发。")}
  {fig(rech, "严格 rollout 的自检恒为 0：decode 时记下的数和事后 prefill 重算的数一样，所以 rollout 记下的 logprob 可以直接用。")}
  {fig(rew, "8 步的奖励只用来确认训练在动，POC 不评估学习效果。")}
  {fig(tchart, "")}
</section>"""
