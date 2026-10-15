# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Render ``REPORT_vllm_runtime_adapter.md`` from ``data/vllm_runtime.json`` (the results JSON).

Every number in the report is read from the results JSON (which is itself copied from run files),
so the Markdown file and the data cannot disagree::

    python -m rl_engine.models.muse_glimmer.runtime.vllm_report \
        reports/muse_glimmer_poc/data/vllm_runtime.json \
        rl_engine/models/muse_glimmer/runtime/REPORT_vllm_runtime_adapter.md

The rendered text is also stored in the JSON under ``report_markdown``.  Static prose (design
notes, RFC feedback, integration requests) lives in ``vllm_runtime.json`` -> ``report_static``
so that it is versioned with the data.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

NR = "NOT RUN"


def _frac(check: Any) -> str:
    if not isinstance(check, dict):
        return NR
    return f"{check['mismatched']} / {check['tokens']}"


def _info(check: Any) -> str:
    if not isinstance(check, dict) or "nonzero_tokens" not in check:
        return ""
    return (
        f" ({check['nonzero_tokens']} non-zero, {check.get('tokens_lp_below_1e-3')} below -1e-3)"
    )


def _gen(results: dict, mode: str, tp: int, temperature: float, lora: bool) -> Any:
    for g in results.get("generate") or []:
        if not isinstance(g, dict):
            continue
        if (
            g["mode"] == mode
            and g["tp"] == tp
            and float(g["temperature"]) == float(temperature)
            and (g["lora"] is not None) == lora
            and (not g.get("engine_lora_enabled") or lora)
        ):
            return g.get("decode_vs_prefill")
    return None


def _contract(results: dict, mode: str, tp: int, label: str) -> Any:
    for c in results.get("contract_shapes") or []:
        if isinstance(c, dict) and c["mode"] == mode and c["tp"] == tp and c["label"] == label:
            return c
    return None


def _cmp(results: dict, name: str) -> Any:
    return ((results.get("xrt") or {}).get("comparisons") or {}).get(name, NR)


def _cell_cmp(results: dict, name: str) -> str:
    c = _cmp(results, name)
    if not isinstance(c, dict):
        return NR
    extra = f", max {c['max_abs_diff']:.3g}" if c["mismatched"] else ""
    return f"{c['mismatched']} / {c['tokens']}{extra}"


def _tp_cells(results: dict, mode: str, fn) -> str:
    cells = []
    for tp in (1, 2, 4):
        value = fn(mode, tp)
        if value is not None:
            cells.append(f"TP{tp} {value}")
    return "; ".join(cells) if cells else NR


def headline(results: dict) -> list[str]:
    rows = [
        "| check (30B, real weights) | native vLLM | strict binding |",
        "|---|---|---|",
    ]

    def gen_cell(mode: str, temperature: float, lora: bool) -> str:
        def one(m: str, tp: int):
            c = _gen(results, m, tp, temperature, lora)
            return f"{_frac(c)}{_info(c)}" if isinstance(c, dict) else None

        return _tp_cells(results, mode, one)

    for temperature in (0.0, 1.0):
        rows.append(
            f"| decode logprobs == one-shot prefill re-score, 8 GSM8K prompts x 64 tokens, "
            f"T={temperature:g} (mismatched / compared) | {gen_cell('native', temperature, False)}"
            f" | {gen_cell('strict', temperature, False)} |"
        )
    for label, what in (
        (
            "long_window_no_lora",
            "long window: 2261-token prompt + 256 decode tokens (positions 2261-2516 > 2048, "
            "sliding-window eviction), chunked prefill (max_num_batched_tokens=512), 1 greedy + "
            "4 seeded T=1 copies",
        ),
        (
            "decode_batch_64_no_lora",
            "decode batch 64 (64 short prompts, T=1, 64 tokens, ignore_eos), chunked prefill 512",
        ),
        (
            "decode_batch_256_no_lora",
            "decode batch 256 (256 short prompts, T=1, 64 tokens, ignore_eos), default budget",
        ),
        ("long_window_random_lora", "long window as above with the rank-16 random LoRA"),
        ("decode_batch_64_random_lora", "decode batch 64 as above with the rank-16 random LoRA"),
        (
            "long_window_tightkv_no_lora",
            "long window, 1 greedy + 8 T=1 copies, in a 1600-block KV pool (freed pages reused, "
            "preemption + recompute)",
        ),
        (
            "decode_batch_64_tightkv_no_lora",
            "decode batch 64 in the same 1600-block KV pool",
        ),
    ):

        def one(m: str, tp: int, label=label):
            c = _contract(results, m, tp, label)
            if c is None:
                return None
            dvp = c["decode_vs_prefill"]
            width = c["engine_steps"].get("max_decode_batch")
            text = f"{_frac(dvp)}{_info(dvp)}, max decode batch {width}"
            events = c.get("scheduler_events")
            if events and (events.get("preempted") or "tightkv" in label):
                text += f", preempted {events['preempted']} / resumed {events['resumed']}"
            return text

        rows.append(f"| {what} | {_tp_cells(results, 'native', one)} | "
                    f"{_tp_cells(results, 'strict', one)} |")
    rows.append(
        "| xrt scoring (9 sequences, 933 tokens, one crossing the 2048 window) TP2 vs TP1 | "
        f"{_cell_cmp(results, 'native_tp2_vs_native_tp1')} | "
        f"{_cell_cmp(results, 'strict_tp2_vs_strict_tp1')} |"
    )
    rows.append(
        f"| same, TP4 vs TP1 (real 4-GPU engine) | {NR} | "
        f"{_cell_cmp(results, 'strict_tp4_vs_strict_tp1')} |"
    )
    refs = [k for k in (results.get("xrt") or {}).get("comparisons", {}) if "reference_chain" in k]
    for name in sorted(refs):
        if name.startswith("strict"):
            rows.append(f"| {name.replace('_', ' ')} | | {_cell_cmp(results, name)} |")
    rows.append(
        f"| strict vs native xrt scores TP1 | | {_cell_cmp(results, 'strict_tp1_vs_native_tp1')} |"
    )
    for temperature in (0.0, 1.0):
        rows.append(
            f"| rank-16 random LoRA on all 416 slots: decode == prefill, T={temperature:g} | "
            f"{gen_cell('native', temperature, True)} | {gen_cell('strict', temperature, True)} |"
        )
    rows.append(
        f"| LoRA xrt scores TP2 vs TP1 / TP4 vs TP1 | | "
        f"{_cell_cmp(results, 'strict_lora_tp2_vs_strict_lora_tp1')} / "
        f"{_cell_cmp(results, 'strict_lora_tp4_vs_strict_lora_tp1')} |"
    )
    zero = results.get("lora_zero")
    if isinstance(zero, list) and zero:
        cell = "; ".join(f"TP{z['tp']} {_frac(z['vs_no_lora'])}" for z in zero)
    else:
        cell = NR
    rows.append(f"| zero-B LoRA changes any score | | {cell} |")
    rules = []
    for lr in results.get("lora_random") or []:
        if isinstance(lr, dict) and lr.get("unmerged_rule"):
            slots = [s for u in lr["unmerged_rule"] for s in u["slots"].values()]
            rules.append(
                f"binding == independent formula {sum(s['binding_equals_formula'] for s in slots)}"
                f" / {len(slots)}, == trainer megatron_lora "
                f"{sum(s['binding_equals_megatron'] for s in slots)} / {len(slots)}"
            )
    rows.append(
        f"| unmerged LoRA rule on real weights (layers 0, 3, 51 x 8 slots, TP1) | | "
        f"{'; '.join(rules) or NR} |"
    )
    hashes = []
    for lr in results.get("lora_random") or []:
        if isinstance(lr, dict):
            hashes.append(
                f"{lr['mode']} TP{lr['tp']}: {'yes' if lr.get('hash_identity') else 'NO'} "
                f"({len(lr.get('worker_hashes') or lr.get('native_lora_reports') or [])} workers)"
            )
    rows.append(
        "| `provenance()[\"lora_sha256\"]` = hash recomputed by every TP worker on what it "
        f"loaded == exported hash | {'; '.join(h for h in hashes if h.startswith('native')) or NR}"
        f" | {'; '.join(h for h in hashes if h.startswith('strict')) or NR} |"
    )
    inv = results.get("token_selection_invariance") or {}
    for key, what in (
        ("across_tp", "generated tokens + logprob bits, TP2/TP4 vs TP1 (same prompts, seeds)"),
        ("across_batch", "token choice, request i in a 64-wide vs a 256-wide batch (same seed)"),
        (
            "across_kv_pressure",
            "token choice, tiny KV pool (reuse, preemption) vs normal pool (same seed)",
        ),
    ):
        entries = inv.get(key)
        if not isinstance(entries, list):
            rows.append(f"| {what} | {NR} | {NR} |")
            continue
        cells = {"native": [], "strict": []}
        for e in entries:
            mode = "strict" if " strict" in f" {e['what']}" else "native"
            cells[mode].append(
                f"{e['what'].split(':')[0]}: {e['identical_tokens_and_logprob_bits']} / "
                f"{e['sequences']}"
            )
        rows.append(
            f"| {what} (sequences identical in tokens and bits) | "
            f"{'; '.join(cells['native']) or NR} | {'; '.join(cells['strict']) or NR} |"
        )
    return rows


def throughput(results: dict) -> list[str]:
    bench = results.get("throughput")
    if not isinstance(bench, list) or not bench:
        return [f"Throughput: {NR}."]
    rows = [
        "| config | tokens/s median (min-max, n) | TTFT whole batch (ms) | decode step median "
        "(ms) | peak alloc / rank (GiB) | KV cache / rank (GiB, pages) | host load 1m |",
        "|---|---|---|---|---|---|---|",
    ]
    for b in bench:
        mem = b.get("memory_per_rank") or [{}]
        loads = [r["host_loadavg_1m_before"] for r in b.get("repeat_runs", [])]
        rows.append(
            f"| {b['label']} | {b['tokens_per_s']:.1f} ({min(b['tokens_per_s_repeats']):.1f}-"
            f"{max(b['tokens_per_s_repeats']):.1f}, n={len(b['tokens_per_s_repeats'])}) | "
            f"{b['ttft_ms']:.0f} | {b['decode_step_ms_median']:.1f} | "
            f"{b['peak_allocated_gib_max_rank']:.1f} | {mem[0].get('kv_cache_gib', 0):.1f}, "
            f"{mem[0].get('num_gpu_blocks')} | {min(loads or [0]):.0f}-{max(loads or [0]):.0f} |"
        )
    by_label = {b["label"]: b["tokens_per_s"] for b in bench}
    ratios = []
    for tp in (1, 2, 4):
        strict = by_label.get(f"strict tp{tp} eager")
        for base in ("eager", "cuda-graphs"):
            native = by_label.get(f"native tp{tp} {base}")
            if strict and native:
                ratios.append(f"TP{tp} strict / native {base} = {strict / native:.2f}x")
        lora = by_label.get(f"strict tp{tp} eager +LoRA r16")
        if strict and lora:
            ratios.append(f"TP{tp} strict +LoRA / strict = {lora / strict:.2f}x")
    if ratios:
        rows.append("")
        rows.append("Ratios of the medians: " + "; ".join(ratios) + ".")
    return rows


def render(results: dict) -> str:
    static = results.get("report_static") or {}
    lines: list[str] = []
    lines.append("# REPORT — `vllm_runtime_adapter` (RFC #501, Muse-Glimmer-30B strict POC)")
    lines.append("")
    lines.append(f"**STATUS: {results.get('status_line', results.get('status', 'unknown'))}**")
    lines.append("")
    commits = results.get("commits_measured") or []
    env = results.get("env") or {}
    lines.append(
        f"Code commit of the binding: `{results.get('commit')}`; runs measured at "
        f"{', '.join('`' + c[:10] + '`' for c in commits)} (dirty-tree runs: "
        f"{results.get('dirty_runs') or 'none'}). Real 30B checkpoint (revision a4e59da5), "
        f"{env.get('gpu')}, vLLM {env.get('vllm')}, torch {env.get('torch')}, Triton "
        f"{env.get('triton')}. Machine-readable results with raw per-token / per-step series: "
        "`reports/muse_glimmer_poc/data/vllm_runtime.json` (this file is rendered from it by "
        "`runtime/vllm_report.py`); logprob dumps (`muse_glimmer_xrt_logprobs_v1`): "
        "`reports/muse_glimmer_poc/data/xrt_vllm_{strict,native}[_lora]_tp{1,2,4}.json`."
    )
    lines.append("")
    lines.append("## Headline results")
    lines.append("")
    lines.extend(headline(results))
    lines.append("")
    lines.append(
        "\"non-zero\" counts the compared tokens whose logprob is not exactly 0.0 (a token with "
        "logprob 0.0 compares trivially); \"below -1e-3\" counts the informative ones."
    )
    lines.append("")
    lines.append("## Throughput")
    lines.append("")
    lines.extend(throughput(results))
    lines.append("")
    for key in ("throughput_notes", "profile_notes"):
        if static.get(key):
            lines.append(static[key])
            lines.append("")
    for section in (
        "what_changed",
        "design_notes",
        "contract_deviations",
        "tests_text",
        "how_to_reproduce",
        "rfc_feedback",
        "integration_requests",
        "open_issues",
    ):
        text = static.get(section)
        if text:
            lines.append(text.rstrip())
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    data_path, report_path = Path(args[0]), Path(args[1])
    results = json.loads(data_path.read_text(encoding="utf-8"))
    text = render(results)
    results["report_markdown"] = text
    results["report_markdown_note"] = (
        f"rendered by rl_engine.models.muse_glimmer.runtime.vllm_report into {report_path}"
    )
    data_path.write_text(json.dumps(results, indent=1), encoding="utf-8")
    report_path.write_text(text, encoding="utf-8")
    print(text[:3000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
