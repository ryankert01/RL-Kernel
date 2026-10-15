# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Tiny dependency-free SVG chart kit for the Muse-Glimmer POC blog.

Every chart is drawn to one explicit scale, colors come from CSS classes that map to the page's
theme tokens (so both light and dark themes work), and labels only name values the chart reaches.
"""

from __future__ import annotations

import html
import math
from typing import Iterable, Sequence

FONT_PX = 12


def esc(s: object) -> str:
    return html.escape(str(s), quote=True)


def fmt(v: float, digits: int = 3) -> str:
    if v is None:
        return "n/a"
    if v == 0:
        return "0"
    if float(v).is_integer() and abs(v) < 1e7:
        return f"{int(v):,}"
    a = abs(v)
    if a >= 1000:
        return f"{v:,.0f}"
    if a >= 100:
        return f"{v:.0f}"
    if a >= 10:
        return f"{v:.1f}"
    if a >= 1:
        return f"{v:.2f}"
    if a >= 0.01:
        return f"{v:.3f}"
    return f"{v:.1e}"


def text_w(s: str, px: float = FONT_PX) -> float:
    """Rough rendered width: CJK glyphs ~1em, latin ~0.6em."""
    w = 0.0
    for ch in str(s):
        w += px if ord(ch) > 0x2E80 else px * 0.6
    return w


def svg(w: float, h: float, body: str, label: str, cls: str = "chart") -> str:
    return (
        f'<svg class="{cls}" viewBox="0 0 {w:.0f} {h:.0f}" role="img" aria-label="{esc(label)}" '
        f'preserveAspectRatio="xMidYMid meet">{body}</svg>'
    )


def nice_ticks(lo: float, hi: float, n: int = 5) -> list[float]:
    if hi <= lo:
        hi = lo + 1
    raw = (hi - lo) / max(n, 1)
    mag = 10 ** math.floor(math.log10(raw))
    step = min((m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw), default=10 * mag)
    start = math.floor(lo / step) * step
    ticks, t = [], start
    while t <= hi + 1e-9 * step:
        if t >= lo - 1e-9 * step:
            ticks.append(round(t, 10))
        t += step
    return ticks


def hbars(
    rows: Sequence[tuple[str, Sequence[float | None]]],
    series: Sequence[tuple[str, str]],
    *,
    unit: str = "",
    title: str = "",
    log: bool = False,
    ref: float | None = None,
    ref_label: str = "",
    width: int = 720,
    bar_h: int = 14,
    value_fmt=fmt,
    xmax: float | None = None,
) -> str:
    """Horizontal grouped bars. rows: (label, [value per series]); series: (name, css class)."""
    label_w = max(text_w(r[0]) for r in rows) + 14
    left, right, top = label_w, 64, 30 if title or len(series) > 1 else 10
    group_h = bar_h * len(series) + 10
    h = top + group_h * len(rows) + 30
    vals = [v for _, vs in rows for v in vs if v is not None]
    vmax = xmax or max(vals + ([ref] if ref else []))
    if log:
        vmin = min([v for v in vals if v > 0] + ([ref] if ref else []))
        lo, hi = math.floor(math.log10(vmin)), math.ceil(math.log10(vmax))
        if hi == lo:
            hi += 1
        def X(v: float) -> float:
            return left + (math.log10(max(v, 10 ** lo)) - lo) / (hi - lo) * (width - left - right)
        ticks = [10 ** e * m for e in range(lo, hi + 1) for m in (1, 2, 5) if 10 ** lo <= 10 ** e * m <= 10 ** hi]
        x0 = X(10 ** lo)
    else:
        ticks = nice_ticks(0, vmax, 5)
        top_t = ticks[-1] if ticks[-1] >= vmax else vmax
        def X(v: float) -> float:
            return left + v / top_t * (width - left - right)
        x0 = X(0)
    out = []
    plot_b = top + group_h * len(rows)
    for t in ticks:
        x = X(t)
        out.append(f'<line class="grid" x1="{x:.1f}" y1="{top - 4}" x2="{x:.1f}" y2="{plot_b}"/>')
        out.append(f'<text class="tick" x="{x:.1f}" y="{plot_b + 14}" text-anchor="middle">{esc(fmt(t))}{esc(unit) if t == ticks[-1] else ""}</text>')
    if ref is not None:
        x = X(ref)
        out.append(f'<line class="refline" x1="{x:.1f}" y1="{top - 8}" x2="{x:.1f}" y2="{plot_b + 2}"/>')
        if ref_label:
            out.append(f'<text class="tick em" x="{x + 4:.1f}" y="{top - 10}">{esc(ref_label)}</text>')
    for i, (lab, vs) in enumerate(rows):
        y0 = top + i * group_h + 4
        out.append(f'<text class="lbl" x="{left - 8}" y="{y0 + group_h / 2 - 2:.1f}" text-anchor="end" dominant-baseline="middle">{esc(lab)}</text>')
        for j, v in enumerate(vs):
            y = y0 + j * bar_h
            if v is None:
                out.append(f'<text class="tick" x="{x0 + 4:.1f}" y="{y + bar_h - 3}">未运行</text>')
                continue
            if log and ref is not None:
                xa, xb = sorted((X(ref), X(v)))
            else:
                xa, xb = x0, X(v)
            out.append(f'<rect class="{series[j][1]}" x="{xa:.1f}" y="{y:.1f}" width="{max(xb - xa, 1.5):.1f}" height="{bar_h - 3}" rx="1.5"/>')
            tx = max(xa, xb) + 4
            out.append(f'<text class="val" x="{tx:.1f}" y="{y + bar_h - 4}">{esc(value_fmt(v))}</text>')
    if title:
        out.append(f'<text class="ctitle" x="0" y="14">{esc(title)}</text>')
    if len(series) > 1:
        lx = width - right
        for name, c in reversed(series):
            lx -= text_w(name, 11) + 26
            out.append(f'<rect class="{c}" x="{lx:.1f}" y="5" width="10" height="10" rx="1.5"/><text class="tick" x="{lx + 14:.1f}" y="14">{esc(name)}</text>')
    return svg(width, h, "".join(out), title or "bar chart")


def lines(
    xs: Sequence[float],
    series: Sequence[tuple[str, str, Sequence[float | None]]],
    *,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    width: int = 720,
    height: int = 260,
    ymin: float | None = None,
    ymax: float | None = None,
    y_fmt=fmt,
    dots: bool = True,
) -> str:
    left, right, top, bottom = 56, 16, 28, 36
    vals = [v for _, _, ys in series for v in ys if v is not None]
    lo = ymin if ymin is not None else min(vals + [0])
    hi = ymax if ymax is not None else max(vals + [lo + 1e-9])
    ticks = nice_ticks(lo, hi, 4)
    lo, hi = min(lo, ticks[0]), max(hi, ticks[-1])
    xlo, xhi = min(xs), max(xs) if max(xs) > min(xs) else min(xs) + 1
    def X(x: float) -> float:
        return left + (x - xlo) / (xhi - xlo) * (width - left - right)
    def Y(y: float) -> float:
        return top + (hi - y) / (hi - lo or 1) * (height - top - bottom)
    out = []
    for t in ticks:
        out.append(f'<line class="grid" x1="{left}" y1="{Y(t):.1f}" x2="{width - right}" y2="{Y(t):.1f}"/>')
        out.append(f'<text class="tick" x="{left - 6}" y="{Y(t) + 4:.1f}" text-anchor="end">{esc(y_fmt(t))}</text>')
    for x in xs:
        out.append(f'<text class="tick" x="{X(x):.1f}" y="{height - bottom + 16}" text-anchor="middle">{esc(fmt(x))}</text>')
    for name, c, ys in series:
        pts = [(X(x), Y(y)) for x, y in zip(xs, ys) if y is not None]
        if len(pts) > 1:
            d = " ".join(f"{'M' if k == 0 else 'L'}{px:.1f},{py:.1f}" for k, (px, py) in enumerate(pts))
            out.append(f'<path class="ln {c}" d="{d}"/>')
        if dots:
            for px, py in pts:
                out.append(f'<circle class="{c}" cx="{px:.1f}" cy="{py:.1f}" r="2.6"/>')
    if xlabel:
        out.append(f'<text class="tick" x="{(left + width - right) / 2:.1f}" y="{height - 4}" text-anchor="middle">{esc(xlabel)}</text>')
    if ylabel:
        out.append(f'<text class="tick" x="{left - 6}" y="{top - 10}" text-anchor="end">{esc(ylabel)}</text>')
    if title:
        out.append(f'<text class="ctitle" x="0" y="14">{esc(title)}</text>')
    lx = width - right
    for name, c, _ in reversed(series):
        lx -= text_w(name, 11) + 26
        out.append(f'<rect class="{c}" x="{lx:.1f}" y="5" width="10" height="10" rx="1.5"/><text class="tick" x="{lx + 14:.1f}" y="14">{esc(name)}</text>')
    return svg(width, height, "".join(out), title or "line chart")


def heatmap(
    row_labels: Sequence[str],
    col_labels: Sequence[str],
    values: Sequence[Sequence[float | None]],
    *,
    title: str = "",
    cell: int = 34,
    cell_w: int | None = None,
    vmax: float | None = None,
    zero_class: str = "hz",
    value_fmt=fmt,
    col_rotate: bool = True,
) -> str:
    """values[i][j] >= 0; 0 renders in zero_class (the 'byte-equal' colour); None = not run."""
    cw = cell_w or cell
    left = max(text_w(r) for r in row_labels) + 12
    col_h = (max(text_w(c, 11) for c in col_labels) * (0.72 if col_rotate else 0) + 18) if col_labels else 0
    top = (22 if title else 4) + col_h
    w = left + cw * len(col_labels) + 8
    h = top + cell * len(row_labels) + 8
    vm = vmax or max([v for row in values for v in row if v] + [1])
    out = []
    for j, c in enumerate(col_labels):
        x = left + j * cw + cw / 2
        if col_rotate:
            out.append(f'<text class="tick" transform="translate({x:.1f},{top - 6}) rotate(-40)">{esc(c)}</text>')
        else:
            out.append(f'<text class="tick" x="{x:.1f}" y="{top - 6}" text-anchor="middle">{esc(c)}</text>')
    for i, r in enumerate(row_labels):
        y = top + i * cell
        out.append(f'<text class="lbl" x="{left - 8}" y="{y + cell / 2 + 4:.1f}" text-anchor="end">{esc(r)}</text>')
        for j, v in enumerate(values[i]):
            x = left + j * cw
            if v is None:
                out.append(f'<rect class="hnone" x="{x + 1}" y="{y + 1}" width="{cw - 2}" height="{cell - 2}" rx="2"/>')
                out.append(f'<text class="tick" x="{x + cw / 2:.1f}" y="{y + cell / 2 + 4:.1f}" text-anchor="middle">–</text>')
                continue
            if v == 0:
                out.append(f'<rect class="{zero_class}" x="{x + 1}" y="{y + 1}" width="{cw - 2}" height="{cell - 2}" rx="2"/>')
                out.append(f'<text class="hv0" x="{x + cw / 2:.1f}" y="{y + cell / 2 + 4:.1f}" text-anchor="middle">0</text>')
            else:
                a = 0.18 + 0.82 * min(1.0, math.log1p(v) / math.log1p(vm))
                out.append(f'<rect class="hbad" style="opacity:{a:.2f}" x="{x + 1}" y="{y + 1}" width="{cw - 2}" height="{cell - 2}" rx="2"/>')
                out.append(f'<text class="hv" x="{x + cw / 2:.1f}" y="{y + cell / 2 + 4:.1f}" text-anchor="middle">{esc(value_fmt(v))}</text>')
    if title:
        out.append(f'<text class="ctitle" x="0" y="14">{esc(title)}</text>')
    return svg(w, h, "".join(out), title or "heatmap")


def stacked(
    rows: Sequence[tuple[str, Sequence[float]]],
    parts: Sequence[tuple[str, str]],
    *,
    title: str = "",
    unit: str = "ms",
    width: int = 720,
    bar_h: int = 20,
    normalize: bool = False,
) -> str:
    label_w = max(text_w(r[0]) for r in rows) + 14
    left, right = label_w, 70
    legend_rows = 1 + int(sum(text_w(n, 11) + 26 for n, _ in parts) // (width - 20))
    top = 22 + 16 * legend_rows
    h = top + (bar_h + 10) * len(rows) + 26
    totals = [sum(v for v in vs if v) for _, vs in rows]
    vmax = 1.0 if normalize else max(totals)
    ticks = [0, 0.25, 0.5, 0.75, 1.0] if normalize else nice_ticks(0, vmax, 5)
    top_t = 1.0 if normalize else max(ticks[-1], vmax)
    def X(v: float) -> float:
        return left + v / top_t * (width - left - right)
    out = []
    plot_b = top + (bar_h + 10) * len(rows)
    for t in ticks:
        out.append(f'<line class="grid" x1="{X(t):.1f}" y1="{top - 4}" x2="{X(t):.1f}" y2="{plot_b}"/>')
        lab = f"{t * 100:.0f}%" if normalize else fmt(t) + (unit if t == ticks[-1] else "")
        out.append(f'<text class="tick" x="{X(t):.1f}" y="{plot_b + 14}" text-anchor="middle">{esc(lab)}</text>')
    for i, (lab, vs) in enumerate(rows):
        y = top + i * (bar_h + 10)
        out.append(f'<text class="lbl" x="{left - 8}" y="{y + bar_h / 2 + 4:.1f}" text-anchor="end">{esc(lab)}</text>')
        acc = 0.0
        tot = totals[i] or 1
        for (name, c), v in zip(parts, vs):
            if not v:
                continue
            v2 = v / tot if normalize else v
            out.append(f'<rect class="{c}" x="{X(acc):.1f}" y="{y}" width="{max(X(acc + v2) - X(acc), 0.8):.1f}" height="{bar_h}"><title>{esc(name)}: {esc(fmt(v))} {esc(unit)}</title></rect>')
            acc += v2
        out.append(f'<text class="val" x="{X(acc) + 4:.1f}" y="{y + bar_h / 2 + 4:.1f}">{esc(fmt(totals[i]))} {esc(unit)}</text>')
    if title:
        out.append(f'<text class="ctitle" x="0" y="14">{esc(title)}</text>')
    lx, ly = 0.0, 22 + 12
    for name, c in parts:
        wdt = text_w(name, 11) + 26
        if lx + wdt > width:
            lx, ly = 0.0, ly + 16
        out.append(f'<rect class="{c}" x="{lx:.1f}" y="{ly - 9}" width="10" height="10" rx="1.5"/><text class="tick" x="{lx + 14:.1f}" y="{ly}">{esc(name)}</text>')
        lx += wdt
    return svg(width, h, "".join(out), title or "stacked bars")


def hist(
    values: Iterable[float],
    *,
    bins: Sequence[float],
    title: str = "",
    cls: str = "s-strict",
    width: int = 360,
    height: int = 180,
    xlabel: str = "",
) -> str:
    counts = [0] * (len(bins) - 1)
    for v in values:
        for k in range(len(bins) - 1):
            if bins[k] <= v < bins[k + 1] or (k == len(bins) - 2 and v == bins[-1]):
                counts[k] += 1
                break
    left, right, top, bottom = 40, 8, 24, 32
    cmax = max(counts + [1])
    bw = (width - left - right) / len(counts)
    out = []
    for t in nice_ticks(0, cmax, 3):
        y = top + (1 - t / max(nice_ticks(0, cmax, 3)[-1], cmax)) * (height - top - bottom)
        out.append(f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}"/><text class="tick" x="{left - 4}" y="{y + 4:.1f}" text-anchor="end">{t:.0f}</text>')
    ytop = max(nice_ticks(0, cmax, 3)[-1], cmax)
    for k, c in enumerate(counts):
        hgt = c / ytop * (height - top - bottom)
        out.append(f'<rect class="{cls}" x="{left + k * bw + 1:.1f}" y="{height - bottom - hgt:.1f}" width="{bw - 2:.1f}" height="{hgt:.1f}"/>')
    for k in range(0, len(bins), max(1, len(bins) // 5)):
        out.append(f'<text class="tick" x="{left + k * bw:.1f}" y="{height - bottom + 14}" text-anchor="middle">{esc(fmt(bins[k]))}</text>')
    if xlabel:
        out.append(f'<text class="tick" x="{(left + width) / 2:.1f}" y="{height - 4}" text-anchor="middle">{esc(xlabel)}</text>')
    if title:
        out.append(f'<text class="ctitle" x="0" y="14">{esc(title)}</text>')
    return svg(width, height, "".join(out), title or "histogram")
