# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""CLI: native vs strict Muse-Glimmer-30B rollout through vLLM 0.31 with bit-pattern logprob dumps.

One engine is built per invocation and the requested experiments run on it in order (the 60 GB
load dominates otherwise).  Run under a GPU lease (the lease sets CUDA_VISIBLE_DEVICES)::

    python -m rl_engine.models.muse_glimmer.runtime.run_vllm_strict --tp 1 --native \
        --prompt-file reports/muse_glimmer_poc/data/xrt_tokens.json --temperatures 0,1 \
        --prefill-check --out native_tp1.json
    python -m rl_engine.models.muse_glimmer.runtime.run_vllm_strict --tp 1 --strict \
        --prompt-file reports/muse_glimmer_poc/data/xrt_tokens.json --temperatures 0,1 \
        --prefill-check --lora-zero-check --lora-random-check \
        --xrt-score reports/muse_glimmer_poc/data/xrt_tokens.json \
        --xrt-out reports/muse_glimmer_poc/data/xrt_vllm_strict_tp1.json --out strict_tp1.json

Experiments (all optional, in this order):

* generate (``--prompt-file``; for a ``muse_glimmer_xrt_v1`` file the prompts are
  ``token_ids[:completion_start]`` of the sequences whose id starts with ``--xrt-filter``), once per
  ``--temperatures`` entry; ``--prefill-check`` re-scores prompt+completion in one prefill and
  compares the fp32 bit patterns token by token.
* ``--lora-zero-check``: rank-r LoRA with random A and B = 0 on every slot of every layer; the
  prefill scores must be bit-identical to the no-LoRA ones.
* ``--lora-random-check``: non-zero random rank-r LoRA (seeded, :func:`random_lora_state_dict`),
  hashed with ``e2e.sync.lora_sha256``; checks ``provenance()["lora_sha256"]``, the unmerged rule
  on real layer weights (``worker_check_lora_rule``), decode == prefill under LoRA, and that the
  LoRA changes the scores.  ``--xrt-lora-out`` also scores the xrt file under this LoRA.
* ``--xrt-score FILE --xrt-out OUT``: ``muse_glimmer_xrt_logprobs_v1`` dump of the xrt file.
* ``--long-window``: the contract's long shape: the ``gsm8k_long_window`` prompt of the xrt file
  (2261 tokens, already past the 2048 window) decoded for ``--long-window-tokens`` (default 256,
  ignore_eos) tokens, one greedy + ``--long-window-copies`` seeded T=1 copies in one batch, so
  every decode position is > 2048 and the sliding-window pages before ``p - 2048`` are evicted
  (vLLM null block 0 -> strict -1).  Chunked prefill splits the prompt when
  ``--max-num-batched-tokens`` is small.  Decode == one-shot prefill re-score, bit by bit.
* ``--decode-batches 64,256``: that many short prompts (the xrt GSM8K prompts repeated, distinct
  seeds) decoded together for ``--decode-batch-tokens`` tokens at T=1 with ignore_eos; the
  worker step timer records the scheduled requests per step (to show the decode batch width);
  decode == prefill re-score bit by bit.
* ``--bench``: batch ``--bench-batch`` x ``--bench-tokens`` (ignore_eos) throughput repeated
  ``--bench-repeats`` times (median + spread), TTFT, per-step decode ms (synchronized worker step
  timer), peak memory and KV-cache size; ``--bench-lora`` repeats it with the random LoRA loaded.
* ``--profile``: torch.profiler top ops of one prefill step and one decode step (batch
  ``--bench-batch``).

Every run file records the commit, ``git status --porcelain`` (``git_dirty``) and the driver's
``rl_engine`` path; the engine refuses workers that import ``rl_engine`` from elsewhere.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

DEFAULT_MODEL = os.path.expanduser("~/models/Muse-Glimmer-30B")
DEFAULT_PROMPTS = [
    "The capital of France is",
    "Write one sentence about the ocean.",
    "Q: What is 17 * 23?\nA:",
    "Once upon a time, in a small village by the sea,",
]
XRT_SCHEMA = "muse_glimmer_xrt_v1"
XRT_OUT_SCHEMA = "muse_glimmer_xrt_logprobs_v1"
RUN_SCHEMA = "muse_glimmer_vllm_run_v1"


def fp32_hex(value: float) -> str:
    return struct.pack(">f", float(value)).hex()


def hex_rows(rows: list[list[float]]) -> list[list[str]]:
    return [[fp32_hex(v) for v in row] for row in rows]


def _git_head() -> str | None:
    try:
        root = Path(__file__).resolve().parents[4]
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:  # noqa: BLE001
        return None


def _git_status() -> dict[str, Any]:
    """``git status --porcelain`` of the tree this module runs from (dirty-tree flag)."""

    try:
        root = Path(__file__).resolve().parents[4]
        lines = subprocess.check_output(
            ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).splitlines()
        return {"git_dirty": bool(lines), "git_status_porcelain": lines, "tree": str(root)}
    except Exception as exc:  # noqa: BLE001
        return {"git_dirty": None, "git_status_error": f"{type(exc).__name__}: {exc}"}


def _env_record() -> dict[str, Any]:
    import torch

    record: dict[str, Any] = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    for name in ("triton", "vllm", "transformers"):
        try:
            module = __import__(name)
            record[name] = str(getattr(module, "__version__", "unknown"))
        except Exception:  # noqa: BLE001
            record[name] = None
    return record


# ----------------------------------------------------------------------------------------------
# Inputs
# ----------------------------------------------------------------------------------------------


def load_prompts(
    path: str | None, tokenizer: Any, chat_template: bool, xrt_filter: str
) -> tuple[list[str], list[list[int]]]:
    """Return (ids, prompt token lists) from a list file or a ``muse_glimmer_xrt_v1`` file."""

    if path:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if isinstance(data, dict) and data.get("schema") == XRT_SCHEMA:
            ids, prompts = [], []
            for seq in data["sequences"]:
                if not str(seq["id"]).startswith(xrt_filter):
                    continue
                ids.append(str(seq["id"]))
                prompts.append([int(t) for t in seq["token_ids"][: int(seq["completion_start"])]])
            if not prompts:
                raise ValueError(f"no xrt sequences match --xrt-filter {xrt_filter!r}")
            return ids, prompts
        if isinstance(data, dict):
            data = data.get("prompts") or data.get("prompt_token_ids")
        if not isinstance(data, list):
            raise ValueError("prompt file must hold a JSON list of strings or token-id lists")
    else:
        data = list(DEFAULT_PROMPTS)
    prompts: list[list[int]] = []
    for item in data:
        if isinstance(item, str):
            if chat_template:
                ids = tokenizer.apply_chat_template(
                    [{"role": "user", "content": item}], add_generation_prompt=True, tokenize=True
                )
                if hasattr(ids, "input_ids"):
                    ids = ids["input_ids"]
            else:
                ids = tokenizer(item, add_special_tokens=True)["input_ids"]
            prompts.append([int(t) for t in ids])
        else:
            prompts.append([int(t) for t in item])
    return [f"prompt_{i}" for i in range(len(prompts))], prompts


def random_lora_state_dict(
    *, seed: int, rank: int = 16, b_std: float = 0.01, zero_b: bool = False, layers: int = 52
) -> dict[str, Any]:
    """Seeded bf16 LoRA on every slot named by ``e2e.sync`` (``layers.{i}.{proj}.lora_{A,B}``).

    A ~ U(-1/sqrt(K), 1/sqrt(K)) (kaiming-uniform bound, as the Megatron binding's init);
    B ~ N(0, b_std^2) (or exactly 0 with ``zero_b``).  Drawn on CPU with one torch.Generator in
    sorted key order, so the tensors (and the hash) depend only on (seed, rank, b_std, zero_b).
    """

    import torch

    from rl_engine.models.muse_glimmer.e2e.sync import LORA_PROJECTIONS
    from rl_engine.models.muse_glimmer.runtime.vllm_strict import LORA_SHAPES

    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    state: dict[str, torch.Tensor] = {}
    keys = []
    for layer in range(int(layers)):
        for proj in LORA_PROJECTIONS:
            keys.append((f"layers.{layer}.{proj}", proj))
    for prefix, proj in sorted(keys):
        k_in, n_out = LORA_SHAPES[proj]
        bound = 1.0 / math.sqrt(k_in)
        a = (torch.rand(rank, k_in, generator=gen) * 2.0 - 1.0) * bound
        if zero_b:
            b = torch.zeros(n_out, rank)
        else:
            b = torch.randn(n_out, rank, generator=gen) * float(b_std)
        state[f"{prefix}.lora_A"] = a.to(torch.bfloat16).contiguous()
        state[f"{prefix}.lora_B"] = b.to(torch.bfloat16).contiguous()
    return state


# ----------------------------------------------------------------------------------------------
# Comparisons
# ----------------------------------------------------------------------------------------------


def compare_rows(a_rows: list[list[float]], b_rows: list[list[float]]) -> dict[str, Any]:
    """Bitwise + numeric comparison of two ragged fp32 logprob tables.

    Besides the mismatch count it reports how informative the compared tokens are (a token whose
    logprob is exactly 0.0 compares trivially): ``nonzero_tokens`` (a != 0.0),
    ``tokens_lp_below_1e-3`` (a < -1e-3), ``tokens_lp_below_0.1`` and ``positive_tokens`` (a > 0,
    which the O7 sign invariant forbids).
    """

    per_seq = []
    total = mismatched = 0
    nonzero = below_1e3 = below_01 = positive = 0
    max_abs = 0.0
    abs_series: list[list[float]] = []
    first: Any = None
    for i, (ra, rb) in enumerate(zip(a_rows, b_rows, strict=True)):
        if len(ra) != len(rb):
            raise ValueError(f"row {i}: {len(ra)} vs {len(rb)} entries")
        diffs = [abs(x - y) for x, y in zip(ra, rb)]
        bits = [fp32_hex(x) != fp32_hex(y) for x, y in zip(ra, rb)]
        count = sum(bits)
        if count and first is None:
            first = {"row": i, "token": bits.index(True)}
        per_seq.append({"tokens": len(ra), "mismatched": count, "max_abs": max(diffs or [0.0])})
        total += len(ra)
        mismatched += count
        nonzero += sum(1 for x in ra if x != 0.0)
        below_1e3 += sum(1 for x in ra if x < -1e-3)
        below_01 += sum(1 for x in ra if x < -0.1)
        positive += sum(1 for x in ra if x > 0.0)
        max_abs = max(max_abs, max(diffs or [0.0]))
        abs_series.append(diffs)
    return {
        "tokens": total,
        "mismatched": mismatched,
        "bitwise_equal": mismatched == 0,
        "nonzero_tokens": nonzero,
        "tokens_lp_below_1e-3": below_1e3,
        "tokens_lp_below_0.1": below_01,
        "positive_tokens": positive,
        "max_abs_diff": max_abs,
        "first_mismatch": first,
        "per_sequence": per_seq,
        "abs_diff_per_token": abs_series,
    }


def _score_check(engine: Any, prompts: list[list[int]], samples: list[list[Any]]) -> Any:
    flat_prompts = [p for p, group in zip(prompts, samples, strict=True) for _ in group]
    flat_tokens = [s.token_ids for group in samples for s in group]
    flat_logprobs = [s.logprobs for group in samples for s in group]
    sequences = [p + c for p, c in zip(flat_prompts, flat_tokens, strict=True)]
    starts = [len(p) for p in flat_prompts]
    t0 = time.perf_counter()
    prefill = engine.logprobs(sequences, starts)
    seconds = time.perf_counter() - t0
    check = compare_rows(flat_logprobs, prefill)
    check["prefill_seconds"] = seconds
    check["prefill_logprobs"] = prefill
    check["prefill_logprobs_hex"] = hex_rows(prefill)
    return check, sequences, starts, prefill


def run_generate(
    engine: Any,
    ids: list[str],
    prompts: list[list[int]],
    *,
    temperature: float,
    max_tokens: int,
    n: int,
    prefill_check: bool,
    tokenizer: Any,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    samples = engine.generate(prompts, n=n, max_tokens=max_tokens, temperature=temperature)
    seconds = time.perf_counter() - t0
    flat_tokens = [s.token_ids for group in samples for s in group]
    flat_logprobs = [s.logprobs for group in samples for s in group]
    generated = sum(len(t) for t in flat_tokens)
    record: dict[str, Any] = {
        "temperature": temperature,
        "n": n,
        "max_tokens": max_tokens,
        "ids": [i for i, group in zip(ids, samples, strict=True) for _ in group],
        "prompt_lens": [len(p) for p, group in zip(prompts, samples, strict=True) for _ in group],
        "token_ids": flat_tokens,
        "logprobs": flat_logprobs,
        "logprobs_hex": hex_rows(flat_logprobs),
        "finish_reasons": [s.finish_reason for group in samples for s in group],
        "texts": [tokenizer.decode(t) for t in flat_tokens],
        "generate_ms": seconds * 1e3,
        "generated_tokens": generated,
        "tokens_per_s": generated / seconds if seconds else None,
    }
    if prefill_check:
        check, _seqs, _starts, _pre = _score_check(engine, prompts, samples)
        record["decode_vs_prefill"] = check
    return record, samples


def _raw_generate(
    engine: Any,
    prompts: list[list[int]],
    *,
    temperatures: list[float],
    seeds: list[int],
    max_tokens: int,
    ignore_eos: bool,
) -> tuple[list[list[int]], list[list[float]], list[str], float, list[dict[str, Any]]]:
    """One ``llm.generate`` call (one engine batch) with per-request temperature and seed.

    The selected-token logprobs are the engine's (strict O7 in strict mode, vLLM raw logprobs in
    native mode).  Returns tokens, logprobs, finish reasons, seconds and the rank-0 step series.
    """

    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    from rl_engine.models.muse_glimmer.runtime import vllm_worker_tools as tools

    llm = engine.llm
    params = [
        SamplingParams(
            max_tokens=int(max_tokens),
            temperature=float(t),
            seed=int(seed),
            ignore_eos=bool(ignore_eos),
            logprobs=0,
            detokenize=False,
            skip_special_tokens=False,
        )
        for t, seed in zip(temperatures, seeds, strict=True)
    ]
    llm.collective_rpc(tools.worker_step_timer, kwargs={"action": "start"})
    t0 = time.perf_counter()
    try:
        outs = llm.generate(
            [TokensPrompt(prompt_token_ids=p) for p in prompts],
            params,
            use_tqdm=False,
            lora_request=engine._lora_request(),
        )
    finally:
        steps = llm.collective_rpc(tools.worker_step_timer, kwargs={"action": "stop"})
    seconds = time.perf_counter() - t0
    tokens, logprobs, finish = [], [], []
    for out in outs:
        completion = out.outputs[0]
        ids = [int(t) for t in completion.token_ids]
        entries = completion.logprobs or []
        if len(entries) != len(ids):
            raise RuntimeError(f"{len(entries)} logprob entries for {len(ids)} tokens")
        tokens.append(ids)
        logprobs.append(
            [
                float(torch_fp32(entry[tok].logprob))
                for tok, entry in zip(ids, entries, strict=True)
            ]
        )
        finish.append(str(completion.finish_reason))
    return tokens, logprobs, finish, seconds, (steps[0] if steps else [])


def torch_fp32(value: float) -> float:
    return struct.unpack(">f", struct.pack(">f", float(value)))[0]


def _step_summary(steps: list[dict[str, Any]]) -> dict[str, Any]:
    decode = [s for s in steps if s["tokens"] == s["reqs"]]
    mixed = [s for s in steps if s["tokens"] != s["reqs"]]
    return {
        "steps": len(steps),
        "decode_only_steps": len(decode),
        "steps_with_prefill_tokens": len(mixed),
        "max_scheduled_tokens": max((s["tokens"] for s in steps), default=0),
        "max_decode_batch": max((s["reqs"] for s in decode), default=0),
        "decode_batch_histogram": {
            str(k): sum(1 for s in decode if s["reqs"] == k)
            for k in sorted({s["reqs"] for s in decode})
        },
        "series": steps,
    }


def run_batch_check(
    engine: Any,
    label: str,
    prompts: list[list[int]],
    *,
    temperatures: list[float],
    seeds: list[int],
    max_tokens: int,
    ignore_eos: bool,
    ids: list[str],
) -> dict[str, Any]:
    """One engine batch + decode == prefill re-score check (contract shapes)."""

    tokens, logprobs, finish, seconds, steps = _raw_generate(
        engine,
        prompts,
        temperatures=temperatures,
        seeds=seeds,
        max_tokens=max_tokens,
        ignore_eos=ignore_eos,
    )
    sequences = [p + t for p, t in zip(prompts, tokens, strict=True)]
    starts = [len(p) for p in prompts]
    t0 = time.perf_counter()
    prefill = engine.logprobs(sequences, starts)
    prefill_s = time.perf_counter() - t0
    check = compare_rows(logprobs, prefill)
    positions = [
        (len(p), len(p) + len(t) - 1) for p, t in zip(prompts, tokens, strict=True) if t
    ]
    return {
        "label": label,
        "requests": len(prompts),
        "ids": ids,
        "temperatures": temperatures,
        "seeds": seeds,
        "max_tokens": max_tokens,
        "ignore_eos": ignore_eos,
        "prompt_lens": starts,
        "decode_positions_min_max": [
            min((a for a, _ in positions), default=None),
            max((b for _, b in positions), default=None),
        ],
        "generated_tokens": sum(len(t) for t in tokens),
        "generate_s": seconds,
        "prefill_rescore_s": prefill_s,
        "finish_reasons": finish,
        "token_ids": tokens,
        "logprobs": logprobs,
        "prefill_logprobs": prefill,
        "decode_vs_prefill": check,
        "engine_steps": _step_summary(steps),
    }


def xrt_dump(
    engine: Any, xrt_path: str, *, strict: bool, tp: int, backend: str, extra: dict[str, Any]
) -> tuple[dict[str, Any], list[list[float]]]:
    data = json.loads(Path(xrt_path).read_text(encoding="utf-8"))
    if data.get("schema") != XRT_SCHEMA:
        raise ValueError(f"{xrt_path} is not {XRT_SCHEMA}")
    seqs = data["sequences"]
    t0 = time.perf_counter()
    rows = engine.logprobs(
        [[int(t) for t in s["token_ids"]] for s in seqs], [int(s["completion_start"]) for s in seqs]
    )
    seconds = time.perf_counter() - t0
    import vllm

    out = {
        "schema": XRT_OUT_SCHEMA,
        "runtime": f"vllm-{vllm.__version__}",
        "mode": "strict" if strict else "native",
        "tp": tp,
        "backend": backend,
        "sequences": [
            {"id": s["id"], "logprobs_hex": [fp32_hex(v) for v in row]}
            for s, row in zip(seqs, rows, strict=True)
        ],
        "source": {
            "file": str(xrt_path),
            "checkpoint_revision": data.get("checkpoint_revision"),
            "tokenizer_json_sha256": data.get("tokenizer_json_sha256"),
        },
        "score_ms": seconds * 1e3,
        **extra,
    }
    return out, rows


# ----------------------------------------------------------------------------------------------
# Throughput / memory / profiler
# ----------------------------------------------------------------------------------------------


def _bench_prompts(prompts: list[list[int]], batch: int) -> list[list[int]]:
    return [prompts[i % len(prompts)] for i in range(batch)]


def run_bench(
    engine: Any, prompts: list[list[int]], *, batch: int, tokens: int, repeats: int = 3
) -> dict[str, Any]:
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    from rl_engine.models.muse_glimmer.runtime import vllm_worker_tools as tools

    llm = engine.llm
    batch_prompts = [TokensPrompt(prompt_token_ids=p) for p in _bench_prompts(prompts, batch)]

    def params(max_tokens: int) -> list[Any]:
        return [
            SamplingParams(
                max_tokens=max_tokens,
                temperature=1.0,
                seed=1000 + i,
                ignore_eos=True,
                logprobs=0,
                detokenize=False,
            )
            for i in range(batch)
        ]

    def timed(max_tokens: int) -> tuple[float, int]:
        t0 = time.perf_counter()
        outs = llm.generate(
            batch_prompts, params(max_tokens), use_tqdm=False, lora_request=engine._lora_request()
        )
        seconds = time.perf_counter() - t0
        return seconds, sum(len(o.outputs[0].token_ids) for o in outs)

    load_before = os.getloadavg()
    timed(2)  # warm-up (Triton compile caches, allocator)
    llm.collective_rpc(tools.worker_memory, kwargs={"reset_peak": True})
    ttft_runs = [timed(1)[0] for _ in range(3)]
    repeat_runs = []
    for _ in range(max(1, int(repeats))):
        load = os.getloadavg()[0]
        seconds, generated = timed(tokens)
        repeat_runs.append(
            {
                "wall_s": seconds,
                "generated_tokens": generated,
                "tokens_per_s": generated / seconds if seconds else None,
                "host_loadavg_1m_before": load,
                "host_loadavg_1m_after": os.getloadavg()[0],
            }
        )
    rates = sorted(r["tokens_per_s"] for r in repeat_runs)
    median_rate = rates[len(rates) // 2]
    full_s, full_tokens = repeat_runs[0]["wall_s"], repeat_runs[0]["generated_tokens"]
    llm.collective_rpc(tools.worker_step_timer, kwargs={"action": "start"})
    timed_s, _ = timed(tokens)
    steps = llm.collective_rpc(tools.worker_step_timer, kwargs={"action": "stop"})
    memory = llm.collective_rpc(tools.worker_memory)
    rank0 = steps[0] if steps else []
    decode_steps = [s for s in rank0 if s["tokens"] == s["reqs"]]
    prefill_steps = [s for s in rank0 if s["tokens"] != s["reqs"]]
    decode_ms = sorted(s["ms"] for s in decode_steps)
    prompt_tokens = sum(len(p) for p in _bench_prompts(prompts, batch))
    return {
        "batch": batch,
        "max_tokens": tokens,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": full_tokens,
        "wall_s": full_s,
        "tokens_per_s": median_rate,
        "tokens_per_s_definition": f"median over {len(repeat_runs)} repeats (generated / wall)",
        "tokens_per_s_repeats": [r["tokens_per_s"] for r in repeat_runs],
        "tokens_per_s_min_max": [rates[0], rates[-1]],
        "tokens_per_s_spread_pct": (rates[-1] - rates[0]) / median_rate * 100.0,
        "repeat_runs": repeat_runs,
        "lora_sha256": engine._lora_sha256,
        "ttft_ms_runs": [t * 1e3 for t in ttft_runs],
        "ttft_ms": min(ttft_runs) * 1e3,
        "ttft_definition": "wall time of generate(max_tokens=1) for the whole batch (min of 3)",
        "decode_ms_per_step_from_wall": (
            (sorted(r["wall_s"] for r in repeat_runs)[len(repeat_runs) // 2] - min(ttft_runs))
            / max(1, tokens - 1)
            * 1e3
        ),
        "step_timer_wall_s": timed_s,
        "step_series_rank0": rank0,
        "decode_step_ms_median": decode_ms[len(decode_ms) // 2] if decode_ms else None,
        "decode_step_ms_p90": decode_ms[int(len(decode_ms) * 0.9)] if decode_ms else None,
        "prefill_steps_rank0": prefill_steps,
        "memory_per_rank": memory,
        # Eager decode is CPU-launch bound: other processes on the host change these numbers.
        "host_loadavg_1m_before_after": [load_before[0], os.getloadavg()[0]],
    }


def run_profile(engine: Any, prompts: list[list[int]], *, batch: int) -> dict[str, Any]:
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    from rl_engine.models.muse_glimmer.runtime import vllm_worker_tools as tools

    llm = engine.llm
    batch_prompts = [TokensPrompt(prompt_token_ids=p) for p in _bench_prompts(prompts, batch)]

    def params(max_tokens: int) -> list[Any]:
        return [
            SamplingParams(
                max_tokens=max_tokens,
                temperature=1.0,
                seed=7 + i,
                ignore_eos=True,
                logprobs=0,
                detokenize=False,
            )
            for i in range(batch)
        ]

    out: dict[str, Any] = {}
    prompt_tokens = sum(len(p) for p in _bench_prompts(prompts, batch))
    for label, max_tokens in (("prefill", 1), ("decode", 4)):
        llm.collective_rpc(
            tools.worker_profile,
            kwargs={
                "action": "arm",
                "kind": label,
                "min_tokens": prompt_tokens // 2,
                "reqs": batch,
                "top": 25,
            },
        )
        llm.generate(batch_prompts, params(max_tokens), use_tqdm=False)
        results = llm.collective_rpc(tools.worker_profile, kwargs={"action": "result"})
        out[label] = results[0] if results else None
    return out


# ----------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--tp", type=int, default=1)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--strict", action="store_true", default=True)
    mode.add_argument("--native", action="store_true")
    parser.add_argument("--prompt-file", default=None)
    parser.add_argument("--xrt-filter", default="gsm8k_test_")
    parser.add_argument("--chat-template", action="store_true")
    parser.add_argument("--no-generate", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperatures", default="0")
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--max-num-batched-tokens", type=int, default=None)
    parser.add_argument(
        "--cuda-graphs",
        action="store_true",
        help="native only: enforce_eager=False (CUDA-graph baseline)",
    )
    parser.add_argument("--long-window", action="store_true")
    parser.add_argument("--long-window-id", default="gsm8k_long_window")
    parser.add_argument("--long-window-copies", type=int, default=4)
    parser.add_argument("--long-window-tokens", type=int, default=256)
    parser.add_argument("--decode-batches", default="")
    parser.add_argument("--decode-batch-tokens", type=int, default=64)
    parser.add_argument("--contract-lora", action="store_true",
                        help="also run --long-window/--decode-batches with the random LoRA")
    parser.add_argument("--gpu-util", type=float, default=0.90)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prefill-check", action="store_true")
    parser.add_argument("--lora-zero-check", action="store_true")
    parser.add_argument("--lora-random-check", action="store_true")
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-seed", type=int, default=20261015)
    parser.add_argument("--lora-b-std", type=float, default=0.01)
    parser.add_argument("--lora-rule-layers", default="0,3,51")
    parser.add_argument("--xrt-score", default=None)
    parser.add_argument("--xrt-out", default=None)
    parser.add_argument("--xrt-lora-out", default=None)
    parser.add_argument("--bench", action="store_true")
    parser.add_argument("--bench-batch", type=int, default=32)
    parser.add_argument("--bench-tokens", type=int, default=256)
    parser.add_argument("--bench-repeats", type=int, default=3)
    parser.add_argument("--bench-lora", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--native-lora", action="store_true", help="native: enable vLLM LoRA")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    strict = not args.native

    from rl_engine.models.muse_glimmer.runtime.strict_rollout_engine import StrictRolloutEngine

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    gpu_ids = list(range(len(visible.split(",")))) if visible else list(range(args.tp))

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    result: dict[str, Any] = {
        "schema": RUN_SCHEMA,
        "mode": "strict" if strict else "native",
        "tp": args.tp,
        "commit": _git_head(),
        "command": " ".join(
            [
                os.path.basename(sys.executable),
                "-m",
                __spec__.name if __spec__ else "run_vllm_strict",
                *(argv if argv is not None else sys.argv[1:]),
            ]
        ),
        "env": _env_record(),
        **_git_status(),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    if strict and args.cuda_graphs:
        parser.error("--cuda-graphs is a native baseline; the strict binding requires eager mode")
    out_path = Path(args.out)

    def flush() -> None:
        out_path.write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")

    t_load = time.perf_counter()
    engine = StrictRolloutEngine(
        args.model,
        tensor_parallel_size=args.tp,
        gpu_ids=gpu_ids,
        strict=strict,
        max_model_len=args.max_model_len,
        seed=args.seed,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_util,
        max_lora_rank=args.lora_rank,
        enable_native_lora=args.native_lora,
        enforce_eager=not args.cuda_graphs,
        max_num_batched_tokens=args.max_num_batched_tokens,
    )
    result["engine_load_s"] = time.perf_counter() - t_load
    result["engine_kwargs"] = {k: str(v) for k, v in engine.engine_kwargs.items()}
    result["rl_engine_file"] = engine.rl_engine_file
    backend = (
        "rl_engine strict binding (Triton strict ops, per-rank attention heads, strict logits)"
        if strict
        else "vLLM native (FlashAttention, cuBLAS, NCCL all-reduce"
        + (", CUDA graphs)" if args.cuda_graphs else ", eager)")
    )
    try:
        ids, prompts = load_prompts(
            args.prompt_file, tokenizer, args.chat_template, args.xrt_filter
        )
        result["prompt_ids"] = ids
        result["prompt_token_ids"] = prompts
        temperatures = [float(t) for t in args.temperatures.split(",") if t.strip()]
        result["generate"] = []
        greedy_samples = None
        if not args.no_generate:
            for temperature in temperatures:
                record, samples = run_generate(
                    engine,
                    ids,
                    prompts,
                    temperature=temperature,
                    max_tokens=args.max_tokens,
                    n=args.n,
                    prefill_check=args.prefill_check,
                    tokenizer=tokenizer,
                )
                result["generate"].append(record)
                if greedy_samples is None:
                    greedy_samples = samples
                flush()
        base_scores = None
        if greedy_samples is not None and (args.lora_zero_check or args.lora_random_check):
            _c, seqs, starts, base_scores = _score_check(engine, prompts, greedy_samples)
        if args.lora_zero_check and strict and base_scores is not None:
            from rl_engine.models.muse_glimmer.e2e.sync import lora_sha256

            state = random_lora_state_dict(seed=args.lora_seed, rank=args.lora_rank, zero_b=True)
            digest = lora_sha256(state)
            engine.load_lora(state, digest)
            with_lora = engine.logprobs(seqs, starts)
            prov_hash = engine.provenance()["lora_sha256"]
            result["lora_zero"] = {
                "rank": args.lora_rank,
                "slots": len(state) // 2,
                "sha256": digest,
                "provenance_lora_sha256": prov_hash,
                "hash_identity": prov_hash == digest,
                "vs_no_lora": compare_rows(base_scores, with_lora),
            }
            engine.clear_lora()
            flush()
        if args.lora_random_check and (strict or args.native_lora) and base_scores is not None:
            from rl_engine.models.muse_glimmer.e2e.sync import lora_sha256
            from rl_engine.models.muse_glimmer.runtime.vllm_strict import worker_check_lora_rule

            state = random_lora_state_dict(
                seed=args.lora_seed, rank=args.lora_rank, b_std=args.lora_b_std
            )
            digest = lora_sha256(state)
            t0 = time.perf_counter()
            engine.load_lora(state, digest)
            load_ms = (time.perf_counter() - t0) * 1e3
            prov = engine.provenance()
            record = {
                "rank": args.lora_rank,
                "seed": args.lora_seed,
                "b_std": args.lora_b_std,
                "recipe": "run_vllm_strict.random_lora_state_dict(seed, rank, b_std)",
                "slots": len(state) // 2,
                "elements": int(sum(t.numel() for t in state.values())),
                "sha256": digest,
                "provenance_lora_sha256": prov["lora_sha256"],
                "hash_identity": prov["lora_sha256"] == digest,
                "worker_hashes": [w.get("lora_sha256") for w in prov["workers"]],
                "load_ms": load_ms,
                "lora_replies": prov["lora_replies"],
            }
            with_lora = engine.logprobs(seqs, starts)
            record["scores_vs_no_lora"] = compare_rows(base_scores, with_lora)
            if args.tp == 1 and strict:
                record["unmerged_rule"] = [
                    engine.llm.collective_rpc(worker_check_lora_rule, args=(int(layer),))[0]
                    for layer in args.lora_rule_layers.split(",")
                ]
            gens = []
            for temperature in temperatures:
                gen, _s = run_generate(
                    engine,
                    ids,
                    prompts,
                    temperature=temperature,
                    max_tokens=args.max_tokens,
                    n=args.n,
                    prefill_check=True,
                    tokenizer=tokenizer,
                )
                gens.append(gen)
            record["generate"] = gens
            result["lora_random"] = record
            flush()
            if args.xrt_score and args.xrt_lora_out:
                dump, _rows = xrt_dump(
                    engine,
                    args.xrt_score,
                    strict=strict,
                    tp=args.tp,
                    backend=backend,
                    extra={"commit": result["commit"], "lora_sha256": digest,
                           "lora_recipe": record["recipe"], "lora_seed": args.lora_seed,
                           "lora_rank": args.lora_rank, "lora_b_std": args.lora_b_std},
                )
                Path(args.xrt_lora_out).write_text(json.dumps(dump, indent=1), encoding="utf-8")
                result["xrt_lora"] = {"out": args.xrt_lora_out, "score_ms": dump["score_ms"]}
            engine.clear_lora()
            flush()
        if args.xrt_score:
            dump, rows = xrt_dump(
                engine,
                args.xrt_score,
                strict=strict,
                tp=args.tp,
                backend=backend,
                extra={"commit": result["commit"], "lora_sha256": None},
            )
            if args.xrt_out:
                Path(args.xrt_out).write_text(json.dumps(dump, indent=1), encoding="utf-8")
            result["xrt"] = {
                "out": args.xrt_out,
                "score_ms": dump["score_ms"],
                "logprobs": rows,
                "ids": [s["id"] for s in dump["sequences"]],
            }
            flush()
        if args.long_window or args.decode_batches:
            from rl_engine.models.muse_glimmer.e2e.sync import lora_sha256

            lora_states: list[tuple[str, Any]] = [("no_lora", None)]
            if args.contract_lora and (strict or args.native_lora):
                lora_states.append(
                    (
                        "random_lora",
                        random_lora_state_dict(
                            seed=args.lora_seed, rank=args.lora_rank, b_std=args.lora_b_std
                        ),
                    )
                )
            result["contract_shapes"] = []
            for lora_label, lora_state in lora_states:
                if lora_state is not None:
                    engine.load_lora(lora_state, lora_sha256(lora_state))
                if args.long_window:
                    long_ids, long_prompts = load_prompts(
                        args.xrt_score or args.prompt_file, tokenizer, False, args.long_window_id
                    )
                    copies = 1 + int(args.long_window_copies)
                    rec = run_batch_check(
                        engine,
                        f"long_window_{lora_label}",
                        [long_prompts[0]] * copies,
                        temperatures=[0.0] + [1.0] * (copies - 1),
                        seeds=[4242 + i for i in range(copies)],
                        max_tokens=args.long_window_tokens,
                        ignore_eos=True,
                        ids=[f"{long_ids[0]}#{i}" for i in range(copies)],
                    )
                    rec["lora_sha256"] = engine._lora_sha256
                    result["contract_shapes"].append(rec)
                    flush()
                for width in [int(x) for x in args.decode_batches.split(",") if x.strip()]:
                    batch_prompts = [prompts[i % len(prompts)] for i in range(width)]
                    rec = run_batch_check(
                        engine,
                        f"decode_batch_{width}_{lora_label}",
                        batch_prompts,
                        temperatures=[1.0] * width,
                        seeds=[9000 + i for i in range(width)],
                        max_tokens=args.decode_batch_tokens,
                        ignore_eos=True,
                        ids=[f"{ids[i % len(ids)]}#{i}" for i in range(width)],
                    )
                    rec["lora_sha256"] = engine._lora_sha256
                    result["contract_shapes"].append(rec)
                    flush()
                if lora_state is not None:
                    engine.clear_lora()
        if args.bench:
            result["bench"] = run_bench(
                engine,
                prompts,
                batch=args.bench_batch,
                tokens=args.bench_tokens,
                repeats=args.bench_repeats,
            )
            flush()
        if args.bench_lora and (strict or args.native_lora):
            from rl_engine.models.muse_glimmer.e2e.sync import lora_sha256

            state = random_lora_state_dict(
                seed=args.lora_seed, rank=args.lora_rank, b_std=args.lora_b_std
            )
            engine.load_lora(state, lora_sha256(state))
            result["bench_lora"] = run_bench(
                engine,
                prompts,
                batch=args.bench_batch,
                tokens=args.bench_tokens,
                repeats=args.bench_repeats,
            )
            engine.clear_lora()
            flush()
        if args.profile:
            result["profile"] = run_profile(engine, prompts, batch=args.bench_batch)
            flush()
        result["meta"] = engine.provenance()
        result["status"] = "completed"
    except Exception as exc:  # noqa: BLE001 - keep partial results on disk
        import traceback

        result["status"] = "failed"
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
        raise
    finally:
        flush()
        engine.close()
    summary = {
        "mode": result["mode"],
        "tp": result["tp"],
        "status": result.get("status"),
        "generate": [
            {
                "temperature": g["temperature"],
                "tokens": g["generated_tokens"],
                "tokens_per_s": g["tokens_per_s"],
                "decode_vs_prefill_mismatched": (g.get("decode_vs_prefill") or {}).get(
                    "mismatched"
                ),
            }
            for g in result.get("generate", [])
        ],
        "lora_zero_equal": (result.get("lora_zero") or {}).get("vs_no_lora", {}).get(
            "bitwise_equal"
        ),
        "lora_random_identity": (result.get("lora_random") or {}).get("hash_identity"),
        "contract_shapes": [
            {
                "label": c["label"],
                "requests": c["requests"],
                "tokens": c["decode_vs_prefill"]["tokens"],
                "mismatched": c["decode_vs_prefill"]["mismatched"],
                "max_decode_batch": c["engine_steps"]["max_decode_batch"],
            }
            for c in result.get("contract_shapes", [])
        ],
        "bench_tokens_per_s": (result.get("bench") or {}).get("tokens_per_s"),
    }
    print(json.dumps(summary, indent=2), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
