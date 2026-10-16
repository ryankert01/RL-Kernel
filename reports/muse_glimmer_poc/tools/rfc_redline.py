#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Render a word-level redline of RFC #501 (original vs revised) as one HTML page.

Usage: rfc_redline.py [--out reports/muse_glimmer_poc/rfc/redline.html]
Reads reports/muse_glimmer_poc/rfc/RFC_501_original.md and RFC_501_revised.md.
"""

from __future__ import annotations

import argparse
import difflib
import html
import json
import re
from pathlib import Path

RFC = Path(__file__).resolve().parents[1] / "rfc"


def tokens(line: str) -> list[str]:
    """Words, whitespace, and whole `code` / **bold** spans (never split by the diff)."""
    return re.findall(r"`[^`]*`|\*\*[^*]+\*\*|\s+|[^\s`*]+|[`*]", line)


def inline_md(s: str) -> str:
    """Escape, then render `code` and **bold** only (enough for a redline)."""
    s = html.escape(s)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", s)
    return s


def word_diff(a: str, b: str) -> str:
    ta, tb = tokens(a), tokens(b)
    out = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, ta, tb, autojunk=False).get_opcodes():
        if op == "equal":
            out.append(inline_md("".join(ta[i1:i2])))
        if op in ("delete", "replace"):
            out.append(f'<del>{inline_md("".join(ta[i1:i2]))}</del>')
        if op in ("insert", "replace"):
            out.append(f'<ins>{inline_md("".join(tb[j1:j2]))}</ins>')
    return "".join(out)


def line_class(line: str) -> str:
    if line.startswith("#"):
        return "h"
    if line.startswith("|"):
        return "t"
    if line.startswith("<img"):
        return "img"
    return "p"


def redline(orig: list[str], rev: list[str]) -> tuple[str, dict]:
    rows, stats = [], {"changed_lines": 0, "added_lines": 0, "removed_lines": 0, "unchanged_lines": 0}
    sm = difflib.SequenceMatcher(None, orig, rev, autojunk=False)
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            for l in orig[i1:i2]:
                stats["unchanged_lines"] += 1 if l.strip() else 0
                if l.strip():
                    rows.append(f'<div class="ln {line_class(l)}">{inline_md(l) if not l.startswith("<img") else "<span class=muted>（图）</span>"}</div>')
            continue
        a, b = orig[i1:i2], rev[j1:j2]
        n = min(len(a), len(b))
        for k in range(n):
            if a[k] == b[k]:
                rows.append(f'<div class="ln {line_class(b[k])}">{inline_md(b[k])}</div>')
            else:
                stats["changed_lines"] += 1
                rows.append(f'<div class="ln chg {line_class(b[k])}"><span class="mk" aria-hidden="true">改</span>{word_diff(a[k], b[k])}</div>')
        for l in a[n:]:
            if l.strip():
                stats["removed_lines"] += 1
                rows.append(f'<div class="ln rm {line_class(l)}"><span class="mk" aria-hidden="true">删</span><del>{inline_md(l)}</del></div>')
        for l in b[n:]:
            if l.strip():
                stats["added_lines"] += 1
                rows.append(f'<div class="ln add {line_class(l)}"><span class="mk" aria-hidden="true">增</span><ins>{inline_md(l)}</ins></div>')
    return "".join(rows), stats


CSS = r"""
/* Layout: one reading column for a legal-style redline; the page's job is to show every change at a glance. */
:root {
  --paper: #F6F7F9; --panel: #FFFFFF; --ink: #161B22; --muted: #5C6675; --rule: #D9DEE5;
  --del: #B4352C; --del-bg: #FBE3E0; --ins: #1F6F45; --ins-bg: #DDF2E5; --accent: #2347A5;
  --font-body: "Noto Sans SC", "IBM Plex Sans", system-ui, sans-serif;
  --font-mono: "JetBrains Mono", ui-monospace, Menlo, "Noto Sans SC", monospace;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --paper: #0E1217; --panel: #151B22; --ink: #E3E8EE; --muted: #98A3B1; --rule: #29313B;
  --del: #F2867B; --del-bg: #3A1C1A; --ins: #6FD39A; --ins-bg: #14301F; --accent: #8DABFF; color-scheme: dark; } }
:root[data-theme="dark"] {
  --paper: #0E1217; --panel: #151B22; --ink: #E3E8EE; --muted: #98A3B1; --rule: #29313B;
  --del: #F2867B; --del-bg: #3A1C1A; --ins: #6FD39A; --ins-bg: #14301F; --accent: #8DABFF; color-scheme: dark; }
* { box-sizing: border-box; }
body { background: var(--paper); color: var(--ink); font: 15px/1.7 var(--font-body); }
.page { max-width: 62rem; margin: 0 auto; padding-block: 2rem 4rem;
  padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); }
h1 { font-size: clamp(1.5rem, 3.5vw, 2.1rem); margin: 0 0 .4rem; text-wrap: balance; }
.sub { color: var(--muted); margin: 0 0 1.2rem; max-width: 46rem; }
.stats { display: flex; flex-wrap: wrap; gap: .5rem; margin-bottom: 1rem; }
.stat { font: 600 .8rem/1 var(--font-mono); padding: .45rem .6rem; border: 1px solid var(--rule); border-radius: 4px; background: var(--panel); }
.stat.d { color: var(--del); } .stat.i { color: var(--ins); }
.tabs { display: flex; gap: .4rem; margin: 1rem 0; }
.tabs button { font: 600 .9rem var(--font-body); padding: .45rem .9rem; border-radius: 6px; border: 1px solid var(--rule); background: var(--panel); color: var(--ink); cursor: pointer; }
.tabs button[aria-pressed="true"] { border-color: var(--accent); color: var(--accent); }
.tabs button:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.doc { background: var(--panel); border: 1px solid var(--rule); border-radius: 10px; padding: 1.2rem 1.4rem; }
.ln { padding: .15rem 0 .15rem 2rem; position: relative; overflow-wrap: anywhere; }
.ln.h { font-weight: 700; margin-top: 1rem; font-size: 1.05rem; }
.ln.t { font-family: var(--font-mono); font-size: .8rem; line-height: 1.6; border-bottom: 1px dashed var(--rule); }
.ln.chg, .ln.add, .ln.rm { background: color-mix(in srgb, var(--accent) 6%, transparent); border-left: 3px solid var(--accent); }
.mk { position: absolute; left: .35rem; top: .2rem; font: 700 .68rem var(--font-body); color: var(--accent); }
del { color: var(--del); background: var(--del-bg); text-decoration: line-through; text-decoration-thickness: 1px; }
ins { color: var(--ins); background: var(--ins-bg); text-decoration: none; }
code { font-family: var(--font-mono); font-size: .88em; }
.muted { color: var(--muted); }
#clean { white-space: normal; }
#clean table { border-collapse: collapse; width: 100%; font-size: .85rem; display: block; overflow-x: auto; }
#clean th, #clean td { border: 1px solid var(--rule); padding: .3rem .5rem; vertical-align: top; }
#clean img { display: none; }
.legend { font-size: .85rem; color: var(--muted); margin: .4rem 0 0; }
@media (max-width: 640px) { .doc { padding: .8rem; } .ln { padding-left: 1.6rem; } }
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(RFC / "redline.html"))
    a = ap.parse_args()
    orig = (RFC / "RFC_501_original.md").read_text().splitlines()
    rev_text = (RFC / "RFC_501_revised.md").read_text()
    rev = rev_text.splitlines()
    body, st = redline(orig, rev)
    no, nr = len("\n".join(orig)), len(rev_text)
    head = (
        "<title>RFC #501 修订对照</title>\n"
        '<meta name="claude:viewport-fit-sides" content="cover">\n'
        '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Noto+Sans+SC:wght@400;600;700&display=swap">\n'
        f"<style>{CSS}</style>\n"
    )
    page = f"""{head}<main class="page">
<h1>RFC #501 修订对照</h1>
<p class="sub">以 #501 当前原文为底，只改必须改的地方：POC 实测和逐行审计发现的错误、已经能回答的开放问题、实现中才暴露的契约缺口。删除标红，新增标绿；“改”=行内修改，“增”=新增行。冻结语义的细节交给 Foundation PR（#558）的 <code>OFFICIAL_FROZEN_SEMANTICS</code>，RFC 里只写结论。</p>
<div class="stats"><span class="stat">修改 {st['changed_lines']} 行</span><span class="stat i">新增 {st['added_lines']} 行</span><span class="stat d">删除 {st['removed_lines']} 行</span><span class="stat">未改 {st['unchanged_lines']} 行</span><span class="stat">长度 {no:,} → {nr:,} 字符（{100 * (nr - no) / no:+.0f}%）</span></div>
<div class="tabs" role="group" aria-label="视图"><button id="b-red" aria-pressed="true">对照（红删绿增）</button><button id="b-clean" aria-pressed="false">修订后全文</button></div>
<div class="doc" id="red">{body}</div>
<div class="doc" id="clean" hidden></div>
<p class="legend">源文件：<code>reports/muse_glimmer_poc/rfc/RFC_501_original.md</code> → <code>RFC_501_revised.md</code>（分支 feat/museglimmer-poc）。</p>
</main>
<script src="https://cdnjs.cloudflare.com/ajax/libs/marked/12.0.2/marked.min.js"></script>
<script>
const REV = {json.dumps(rev_text)};
const red = document.getElementById('red'), clean = document.getElementById('clean');
const bR = document.getElementById('b-red'), bC = document.getElementById('b-clean');
function show(c) {{
  red.hidden = c; clean.hidden = !c; bR.setAttribute('aria-pressed', String(!c)); bC.setAttribute('aria-pressed', String(c));
  if (c && !clean.dataset.done) {{
    try {{ clean.innerHTML = window.marked ? marked.parse(REV) : '<pre>' + REV.replace(/[&<>]/g, s => ({{'&':'&amp;','<':'&lt;','>':'&gt;'}})[s]) + '</pre>'; }}
    catch (e) {{ clean.textContent = REV; }}
    clean.dataset.done = '1';
  }}
}}
bR.addEventListener('click', () => show(false)); bC.addEventListener('click', () => show(true));
</script>
"""
    Path(a.out).write_text(page)
    print(f"wrote {a.out}: {st}; {no} -> {nr} chars")


if __name__ == "__main__":
    main()
