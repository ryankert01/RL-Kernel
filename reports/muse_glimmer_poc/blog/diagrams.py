# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Hand-drawn explanatory diagrams for the Muse-Glimmer POC blog (inline SVG, theme classes)."""

from __future__ import annotations

import struct

from svgkit import esc, svg


def _arrow_defs(fid: str) -> str:
    return (
        f'<defs><marker id="ah-{fid}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
        f'orient="auto-start-reverse"><path class="ahead" d="M0,0 L10,5 L0,10 z"/></marker></defs>'
    )


def _arr(fid: str, x1, y1, x2, y2, cls: str = "arrow") -> str:
    return f'<line class="{cls}" x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" marker-end="url(#ah-{fid})"/>'


def _box(x, y, w, h, label: str, cls: str = "box", tcls: str = "btxt", sub: str = "") -> str:
    out = f'<rect class="{cls}" x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="6"/>'
    if sub:
        out += f'<text class="{tcls}" x="{x + w / 2:.1f}" y="{y + h / 2 - 3:.1f}" text-anchor="middle">{esc(label)}</text>'
        out += f'<text class="bsub" x="{x + w / 2:.1f}" y="{y + h / 2 + 12:.1f}" text-anchor="middle">{esc(sub)}</text>'
    else:
        out += f'<text class="{tcls}" x="{x + w / 2:.1f}" y="{y + h / 2 + 4:.1f}" text-anchor="middle">{esc(label)}</text>'
    return out


def f32(x: float) -> float:
    return struct.unpack("<f", struct.pack("<f", x))[0]


def f32hex(x: float) -> str:
    return struct.pack(">f", x).hex()


SUM_VALUES = [0.7031, -1.9375, 3.1e-4, 1234.5, -1233.9, 0.1, 2.2e-5, -0.47]


def sum_orders() -> tuple[str, str, float, float]:
    vals = [f32(v) for v in SUM_VALUES]
    seq = 0.0
    for v in vals:
        seq = f32(seq + v)
    level = vals
    while len(level) > 1:
        level = [f32(level[i] + level[i + 1]) for i in range(0, len(level), 2)]
    return f32hex(seq), f32hex(level[0]), seq, level[0]


def _hex_spans(a: str, b: str, x: float, y: float, cls_same: str, cls_diff: str, anchor: str = "start") -> str:
    parts = []
    for ca, cb in zip(a, b):
        parts.append(f'<tspan class="{cls_same if ca == cb else cls_diff}">{ca}</tspan>')
    return f'<text class="hexbig" x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}">0x{"".join(parts)}</text>'


def fig_sum_orders() -> str:
    """Same 8 fp32 numbers, two summation orders, two different bit patterns."""
    fid = "sum"
    hs, ht, vs, vt = sum_orders()
    W, H = 860, 330
    out = [_arrow_defs(fid)]
    n = len(SUM_VALUES)
    cw, gap, x0 = 64, 8, 10
    xs = [x0 + i * (cw + gap) for i in range(n)]
    # values row (middle)
    ym = 112
    for i, v in enumerate(SUM_VALUES):
        out.append(_box(xs[i], ym, cw, 30, f"{v:g}", cls="box num", tcls="btxt mono"))
    # top: left-to-right chain
    yt = 52
    out.append('<text class="lbl em" x="0" y="22">从左到右累加（一个 for 循环）</text>')
    prev = None
    for i in range(n):
        cx = xs[i] + cw / 2
        out.append(f'<circle class="node-native" cx="{cx:.1f}" cy="{yt:.1f}" r="7"/>')
        out.append(f'<line class="edge-native" x1="{cx:.1f}" y1="{ym:.1f}" x2="{cx:.1f}" y2="{yt + 7:.1f}"/>')
        if prev is not None:
            out.append(f'<line class="edge-native" x1="{prev + 7:.1f}" y1="{yt:.1f}" x2="{cx - 7:.1f}" y2="{yt:.1f}"/>')
        prev = cx
    out.append(_arr(fid, prev + 8, yt, prev + 34, yt, "arrow native"))
    out.append(_hex_spans(hs, ht, prev + 38, yt - 10, "hx-same", "hx-native"))
    out.append(f'<text class="bsub" x="{prev + 38:.1f}" y="{yt + 12:.1f}">= {vs:.7f}</text>')
    # bottom: balanced 8-leaf tree
    out.append(f'<text class="lbl em" x="0" y="{H - 10}">8 叶平衡树（契约规定的顺序）</text>')
    level = [xs[i] + cw / 2 for i in range(n)]
    y = ym + 30
    dy = 34
    while len(level) > 1:
        nxt = []
        for k in range(0, len(level), 2):
            cx = (level[k] + level[k + 1]) / 2
            ny = y + dy
            out.append(f'<line class="edge-strict" x1="{level[k]:.1f}" y1="{y + (0 if y == ym + 30 else 7):.1f}" x2="{cx:.1f}" y2="{ny - 7:.1f}"/>')
            out.append(f'<line class="edge-strict" x1="{level[k + 1]:.1f}" y1="{y + (0 if y == ym + 30 else 7):.1f}" x2="{cx:.1f}" y2="{ny - 7:.1f}"/>')
            out.append(f'<circle class="node-strict" cx="{cx:.1f}" cy="{ny:.1f}" r="7"/>')
            nxt.append(cx)
        level, y = nxt, y + dy
    rx = level[0]
    out.append(_arr(fid, rx + 8, y, xs[-1] + cw + 34, y, "arrow strict"))
    out.append(_hex_spans(ht, hs, xs[-1] + cw + 38, y - 10, "hx-same", "hx-strict"))
    out.append(f'<text class="bsub" x="{xs[-1] + cw + 38:.1f}" y="{y + 12:.1f}">= {vt:.7f}</text>')
    return svg(W, H, "".join(out), "同样 8 个 fp32 数，两种累加顺序，结果的 bit 不同", cls="diagram")


def fig_model() -> str:
    """52-layer schedule strip + one decoder layer's dataflow."""
    fid = "model"
    W, H = 1000, 300
    out = [_arrow_defs(fid)]
    out.append('<text class="lbl em" x="0" y="16">52 层：每 4 层里 3 层滑动窗口（RoPE），第 4 层全局（NoPE）</text>')
    cw = (W - 20) / 52
    for i in range(52):
        g = i % 4 == 3
        out.append(f'<rect class="{"lay-global" if g else "lay-local"}" x="{i * cw:.1f}" y="28" width="{cw - 2:.1f}" height="26" rx="2"><title>layer {i}: {"global" if g else "local"}</title></rect>')
        if i % 4 == 3:
            out.append(f'<text class="tick" x="{i * cw + cw / 2:.1f}" y="68" text-anchor="middle">{i}</text>')
    out.append('<rect class="lay-local" x="0" y="80" width="12" height="12" rx="2"/><text class="tick" x="18" y="90">local：只看最近 2048 个 token，RoPE θ=500000（39 层）</text>')
    out.append('<rect class="lay-global" x="440" y="80" width="12" height="12" rx="2"/><text class="tick" x="458" y="90">global：causal 全序列，无位置编码（13 层）</text>')
    # layer dataflow
    y = 120
    bh = 46
    steps = [
        ("input_layernorm", "(1+w)·rms，eps 1e-5", "box"),
        ("q/k/v/gate 投影", "四个独立 GEMM", "box strictb"),
        ("QK-norm + RoPE", "只有 Q 再乘 3.87", "box strictb"),
        ("门控注意力", "σ(gate) ⊙ attn", "box strictb"),
        ("o_proj", "TP 行并行（K 维 8 块树）", "box strictb"),
        ("post_attn norm", "eps 1e-8 → 残差加", "box"),
    ]
    steps2 = [
        ("pre_ffn norm", "eps 1e-5", "box"),
        ("SwiGLU FFN", "6656→19968→6656", "box strictb"),
        ("post_ffn norm", "eps 1e-8 → 残差加", "box"),
        ("52 层之后", "final norm: w·rms", "box"),
        ("LM head + softcap", "×0.196 ÷20 tanh ×20", "box strictb"),
        ("fp32 log p", "两边都要这个数", "box goal"),
    ]
    bw = (W - 5 * 18) / 6
    for row, items in enumerate((steps, steps2)):
        yy = y + row * (bh + 54)
        for k, (a, b, c) in enumerate(items):
            x = k * (bw + 18)
            out.append(_box(x, yy, bw, bh, a, cls=c, sub=b))
            if k < len(items) - 1:
                out.append(_arr(fid, x + bw + 2, yy + bh / 2, x + bw + 16, yy + bh / 2))
    # wrap arrow from row1 end to row2 start
    x_end = 5 * (bw + 18) + bw / 2
    out.append(f'<path class="arrow" d="M{x_end:.1f},{y + bh:.1f} V{y + bh + 26:.1f} H{bw / 2:.1f} V{y + bh + 52:.1f}" marker-end="url(#ah-{fid})" fill="none"/>')
    out.append(f'<text class="tick" x="0" y="{y + 2 * bh + 54 + 30}">蓝框 = GEMM、注意力、FFN、LM head 这些主要的严格 Triton 内核（定序、定 tile、fp32 累加）；白框的 norm 也有各自的严格内核（逐行固定顺序归约）。</text>')
    return svg(W, H, "".join(out), "Muse-Glimmer-30B 的层结构和一层的数据流", cls="diagram")


def fig_tp_tree() -> str:
    """The canonical 8-leaf tree and how TP1/2/4/8 own sub-trees of it."""
    fid = "tp"
    W, H = 900, 300
    out = [_arrow_defs(fid)]
    rows = [(1, "TP1"), (2, "TP2"), (4, "TP4"), (8, "TP8")]
    col_w = W / 4
    for c, (tp, name) in enumerate(rows):
        ox = c * col_w + 12
        lw = (col_w - 30) / 8
        ly = 230
        out.append(f'<text class="lbl em" x="{ox + (col_w - 30) / 2:.1f}" y="18" text-anchor="middle">{name}</text>')
        leaves = [ox + i * lw + lw / 2 for i in range(8)]
        per = 8 // tp
        for i, lx in enumerate(leaves):
            r = i // per
            out.append(f'<rect class="rank r{r % 8}" x="{lx - lw / 2 + 1:.1f}" y="{ly}" width="{lw - 2:.1f}" height="22" rx="3"/>')
            out.append(f'<text class="tick" x="{lx:.1f}" y="{ly + 15}" text-anchor="middle">{i}</text>')
        level, y = leaves, ly
        depth = 0
        while len(level) > 1:
            nxt = []
            span = 2 ** (depth + 1)
            for k in range(0, len(level), 2):
                cx = (level[k] + level[k + 1]) / 2
                ny = y - 44
                cross = span > per  # this add happens after the all-gather
                cls = "edge-strict" if cross else "edge-local"
                out.append(f'<line class="{cls}" x1="{level[k]:.1f}" y1="{y - (0 if depth == 0 else 6):.1f}" x2="{cx:.1f}" y2="{ny + 6:.1f}"/>')
                out.append(f'<line class="{cls}" x1="{level[k + 1]:.1f}" y1="{y - (0 if depth == 0 else 6):.1f}" x2="{cx:.1f}" y2="{ny + 6:.1f}"/>')
                out.append(f'<circle class="{"node-strict" if cross else "node-local"}" cx="{cx:.1f}" cy="{ny:.1f}" r="6"/>')
                nxt.append(cx)
            level, y, depth = nxt, ny, depth + 1
        out.append(f'<text class="bsub" x="{level[0]:.1f}" y="{y - 12:.1f}" text-anchor="middle">一次 bf16 舍入</text>')
    out.append(f'<text class="tick" x="0" y="{H - 14}">叶子 = K 维（或词表）的 8 个固定块，颜色 = 哪个 rank 算它。细线：rank 内 fp32 子树；粗线：all-gather 之后所有 rank 用同一棵树合并。</text>')
    out.append(f'<text class="tick" x="0" y="{H}">TP 怎么切，只决定谁算哪棵子树；加法顺序永远是同一棵树，所以 TP1/2/4/8 的结果逐 bit 相同。</text>')
    return svg(W, H + 6, "".join(out), "规范 8 叶归约树与 TP 切分", cls="diagram")


def fig_window() -> str:
    """Prompt of 2100 tokens + 32 decode steps; window 2048; chunk boundaries that were tested."""
    fid = "win"
    W, H = 900, 250
    out = [_arrow_defs(fid)]
    total = 2132
    x0, x1 = 10, W - 190
    def X(t: float) -> float:
        return x0 + t / total * (x1 - x0)
    y = 60
    out.append(f'<rect class="seq" x="{X(0):.1f}" y="{y}" width="{X(2100) - X(0):.1f}" height="22" rx="3"/>')
    out.append(f'<rect class="seq-dec" x="{X(2100):.1f}" y="{y}" width="{X(total) - X(2100):.1f}" height="22" rx="3"/>')
    out.append(f'<text class="tick" x="{X(1050):.1f}" y="{y + 15}" text-anchor="middle">prompt 2100 token</text>')
    out.append(f'<text class="tick" x="{X(2116):.1f}" y="{y - 6}" text-anchor="middle">decode 32</text>')
    # window bracket for the last query
    q = 2131
    out.append(f'<path class="bracket" d="M{X(q - 2047):.1f},{y + 30} V{y + 38} H{X(q):.1f} V{y + 30}"/>')
    out.append(f'<text class="tick" x="{X(q - 1024):.1f}" y="{y + 52}" text-anchor="middle">local 层：最后一个 query 只看这 2048 个 key（更早的被逐出）</text>')
    out.append('<text class="lbl em" x="0" y="20">同一串 token，10 种执行切法，每个 token 的 fp32 logprob 逐 bit 相同</text>')
    cuts = [
        ("train forward", [], "一次全序列"),
        ("prefill + decode", [2100], "最常见的推理"),
        ("对半切", [1066], ""),
        ("64 对齐", list(range(64, 2100, 64)), ""),
        ("不规则", [7, 300, 1201, 1999], ""),
        ("边界 2047", [2047], "窗口边界前 1"),
        ("边界 2048", [2048], "正好窗口"),
        ("边界 2049", [2049], "窗口边界后 1"),
    ]
    yy = y + 76
    for k, (name, bs, note) in enumerate(cuts):
        yk = yy + k * 17
        out.append(f'<text class="tick" x="{x0 - 6}" y="{yk + 9}" text-anchor="end"></text>')
        out.append(f'<line class="track" x1="{X(0):.1f}" y1="{yk + 5}" x2="{X(2100):.1f}" y2="{yk + 5}"/>')
        for b in bs:
            out.append(f'<line class="cut" x1="{X(b):.1f}" y1="{yk}" x2="{X(b):.1f}" y2="{yk + 10}"/>')
        out.append(f'<text class="tick" x="{X(2100) + 6:.1f}" y="{yk + 9}">{esc(name)}{(" · " + esc(note)) if note else ""}</text>')
    return svg(W, yy + len(cuts) * 17 + 8, "".join(out), "滑动窗口与各种 chunk 切法", cls="diagram")


def fig_system(rollout_tp: str = "TP1", train_tp: str = "TP2") -> str:
    """The POC's end-to-end loop with the cross-env bridge."""
    fid = "sys"
    W, H = 900, 330
    out = [_arrow_defs(fid)]
    # env frames
    out.append('<rect class="env" x="0" y="26" width="430" height="290" rx="10"/>')
    out.append('<text class="lbl em" x="14" y="48">~/envs/vllm · torch 2.13 cu130 · vLLM 0.31</text>')
    out.append('<rect class="env" x="470" y="26" width="430" height="290" rx="10"/>')
    out.append('<text class="lbl em" x="484" y="48">~/envs/megatron · torch 2.13 cu129 · Megatron Bridge</text>')
    out.append(_box(20, 70, 190, 56, "GSM8K 题目", sub="chat template → token id"))
    out.append(_box(230, 70, 180, 56, f"vLLM 严格 rollout", cls="box strictb", sub=f"{rollout_tp} · 采样 + fp32 logprob"))
    out.append(_box(20, 160, 390, 56, "GRPO 编排器（loop.py）", sub="奖励 → 组内优势 → token 级 loss；逐 token 比 bit"))
    out.append(_box(20, 246, 190, 52, "RemoteTrainer 客户端", sub="loss.backward() 得 dL/dlogp"))
    out.append(_box(490, 160, 390, 56, "Megatron 严格 trainer", cls="box strictb", sub=f"{train_tp} · LoRA r16 · 同一套 Triton 内核"))
    out.append(_box(620, 246, 260, 52, "trainer server（单进程启动）", sub=f"StrictTrainer 自己拉起 {train_tp} worker"))
    out.append(_box(490, 70, 390, 56, "LoRA 导出 + sha256", sub="muse_glimmer_lora_sha256_v1"))
    out.append(_arr(fid, 210, 98, 228, 98))
    out.append(_arr(fid, 320, 126, 320, 158))
    out.append(_arr(fid, 115, 216, 115, 244))
    out.append(_arr(fid, 212, 272, 618, 272, "arrow strict"))
    out.append('<text class="bsub" x="450" y="264" text-anchor="middle">token id + dL/dlogp（fp32，逐 bit）</text>')
    out.append(_arr(fid, 785, 244, 785, 218))
    out.append(_arr(fid, 685, 158, 685, 128))
    out.append(_arr(fid, 488, 98, 412, 98, "arrow strict"))
    out.append('<text class="bsub" x="450" y="88" text-anchor="middle">LoRA 权重</text>')
    out.append('<text class="bsub" x="450" y="118" text-anchor="middle">哈希校验</text>')
    out.append(_arr(fid, 488, 190, 412, 190, "arrow"))
    out.append('<text class="bsub" x="450" y="182" text-anchor="middle">logprob</text>')
    return svg(W, H, "".join(out), "端到端 GRPO 循环和跨环境桥", cls="diagram")


def fig_cp() -> str:
    """CP: K/V all-gathered, dK/dV partials per canonical query chunk combined in the tree."""
    W, H = 900, 210
    out = [_arrow_defs("cp")]
    x0, x1 = 30, W - 20
    T = 8
    cw = (x1 - x0) / T
    out.append('<text class="lbl em" x="0" y="16">CP 反向：把整条序列的 query 切成 8 个规范块，每块对 dK/dV 的贡献单独算，再按同一棵树合并</text>')
    for cp, y in ((1, 40), (2, 92), (4, 144)):
        out.append(f'<text class="lbl" x="{x0 - 8}" y="{y + 17}" text-anchor="end">CP{cp}</text>')
        per = T // cp
        for i in range(T):
            r = i // per
            out.append(f'<rect class="rank r{r * (8 // cp) % 8}" x="{x0 + i * cw + 1:.1f}" y="{y}" width="{cw - 2:.1f}" height="26" rx="3"/>')
            out.append(f'<text class="tick" x="{x0 + i * cw + cw / 2:.1f}" y="{y + 17}" text-anchor="middle">q 块 {i}{"（rank " + str(r) + "）" if i % per == 0 else ""}</text>')
    out.append(f'<text class="tick" x="{x0}" y="{H - 8}">每个 rank 只拥有自己的 query 块；dK/dV 的 8 份部分和不论在哪个 rank 上算，都在同一棵 8 叶树里相加，所以 CP1 = CP2 = CP4。</text>')
    return svg(W, H, "".join(out), "CP 下 dK/dV 的规范 query 分块", cls="diagram")
