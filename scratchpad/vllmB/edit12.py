from pathlib import Path

p = Path("/home/ubuntu/RL-Kernel/.claude/worktrees/wf_65f5032c-579-2/rl_engine/models/muse_glimmer/runtime/run_vllm_strict.py")
s = p.read_text()

old = '''* ``--bench``: batch ``--bench-batch`` x ``--bench-tokens`` (ignore_eos) throughput, TTFT, per-step
  decode ms (synchronized worker step timer), peak memory and KV-cache size.
* ``--profile``: torch.profiler top ops of one prefill step and one decode step (batch
  ``--bench-batch``).
"""'''
new = '''* ``--long-window``: the contract's long shape: the ``gsm8k_long_window`` prompt of the xrt file
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
"""'''
assert old in s
s = s.replace(old, new, 1)

old = '''def _git_head() -> str | None:
    try:
        root = Path(__file__).resolve().parents[4]
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:  # noqa: BLE001
        return None
'''
new = '''def _git_head() -> str | None:
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
'''
assert old in s
s = s.replace(old, new, 1)

old = '''def compare_rows(a_rows: list[list[float]], b_rows: list[list[float]]) -> dict[str, Any]:
    """Bitwise + numeric comparison of two ragged fp32 logprob tables."""

    per_seq = []
    total = mismatched = 0
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
        max_abs = max(max_abs, max(diffs or [0.0]))
        abs_series.append(diffs)
    return {
        "tokens": total,
        "mismatched": mismatched,
        "bitwise_equal": mismatched == 0,
        "max_abs_diff": max_abs,
        "first_mismatch": first,
        "per_sequence": per_seq,
        "abs_diff_per_token": abs_series,
    }
'''
new = '''def compare_rows(a_rows: list[list[float]], b_rows: list[list[float]]) -> dict[str, Any]:
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
'''
assert old in s
s = s.replace(old, new, 1)

old = '''def xrt_dump(
    engine: Any, xrt_path: str, *, strict: bool, tp: int, backend: str, extra: dict[str, Any]
) -> tuple[dict[str, Any], list[list[float]]]:'''
new = '''def _raw_generate(
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
) -> tuple[dict[str, Any], list[list[float]]]:'''
assert old in s
s = s.replace(old, new, 1)

# prefill logprobs as floats too (R7)
old = '''    check = compare_rows(flat_logprobs, prefill)
    check["prefill_seconds"] = seconds
    check["prefill_logprobs_hex"] = hex_rows(prefill)
    return check, sequences, starts, prefill'''
new = '''    check = compare_rows(flat_logprobs, prefill)
    check["prefill_seconds"] = seconds
    check["prefill_logprobs"] = prefill
    check["prefill_logprobs_hex"] = hex_rows(prefill)
    return check, sequences, starts, prefill'''
assert old in s
s = s.replace(old, new, 1)

# bench repeats
old = '''    load_before = os.getloadavg()
    timed(2)  # warm-up (Triton compile caches, allocator)
    llm.collective_rpc(tools.worker_memory, kwargs={"reset_peak": True})
    ttft_runs = [timed(1)[0] for _ in range(3)]
    full_s, full_tokens = timed(tokens)
    llm.collective_rpc(tools.worker_step_timer, kwargs={"action": "start"})'''
new = '''    load_before = os.getloadavg()
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
    llm.collective_rpc(tools.worker_step_timer, kwargs={"action": "start"})'''
assert old in s
s = s.replace(old, new, 1)
old = '''def run_bench(engine: Any, prompts: list[list[int]], *, batch: int, tokens: int) -> dict[str, Any]:'''
new = '''def run_bench(
    engine: Any, prompts: list[list[int]], *, batch: int, tokens: int, repeats: int = 3
) -> dict[str, Any]:'''
assert old in s
s = s.replace(old, new, 1)
old = '''        outs = llm.generate(batch_prompts, params(max_tokens), use_tqdm=False)
        seconds = time.perf_counter() - t0
        return seconds, sum(len(o.outputs[0].token_ids) for o in outs)
'''
new = '''        outs = llm.generate(
            batch_prompts, params(max_tokens), use_tqdm=False, lora_request=engine._lora_request()
        )
        seconds = time.perf_counter() - t0
        return seconds, sum(len(o.outputs[0].token_ids) for o in outs)
'''
assert old in s
s = s.replace(old, new, 1)
old = '''        "generated_tokens": full_tokens,
        "wall_s": full_s,
        "tokens_per_s": full_tokens / full_s if full_s else None,'''
new = '''        "generated_tokens": full_tokens,
        "wall_s": full_s,
        "tokens_per_s": median_rate,
        "tokens_per_s_definition": f"median over {len(repeat_runs)} repeats (generated / wall)",
        "tokens_per_s_repeats": [r["tokens_per_s"] for r in repeat_runs],
        "tokens_per_s_min_max": [rates[0], rates[-1]],
        "tokens_per_s_spread_pct": (rates[-1] - rates[0]) / median_rate * 100.0,
        "repeat_runs": repeat_runs,
        "lora_sha256": engine._lora_sha256,'''
assert old in s
s = s.replace(old, new, 1)
old = '''        "decode_ms_per_step_from_wall": (full_s - min(ttft_runs)) / max(1, tokens - 1) * 1e3,'''
new = '''        "decode_ms_per_step_from_wall": (
            (sorted(r["wall_s"] for r in repeat_runs)[len(repeat_runs) // 2] - min(ttft_runs))
            / max(1, tokens - 1)
            * 1e3
        ),'''
assert old in s
s = s.replace(old, new, 1)

# CLI
old = '''    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-seqs", type=int, default=64)'''
new = '''    parser.add_argument("--max-model-len", type=int, default=4096)
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
                        help="also run --long-window/--decode-batches with the random LoRA")'''
assert old in s
s = s.replace(old, new, 1)
old = '''    parser.add_argument("--bench-tokens", type=int, default=256)
    parser.add_argument("--profile", action="store_true")'''
new = '''    parser.add_argument("--bench-tokens", type=int, default=256)
    parser.add_argument("--bench-repeats", type=int, default=3)
    parser.add_argument("--bench-lora", action="store_true")
    parser.add_argument("--profile", action="store_true")'''
assert old in s
s = s.replace(old, new, 1)
old = '''        "env": _env_record(),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }'''
new = '''        "env": _env_record(),
        **_git_status(),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    if strict and args.cuda_graphs:
        parser.error("--cuda-graphs is a native baseline; the strict binding requires eager mode")'''
assert old in s
s = s.replace(old, new, 1)
old = '''        max_lora_rank=args.lora_rank,
        enable_native_lora=args.native_lora,
    )
    result["engine_load_s"] = time.perf_counter() - t_load
    backend = (
        "rl_engine strict binding (Triton strict ops, KV-replicated attention at TP>1)"
        if strict
        else "vLLM native (FlashAttention, cuBLAS, NCCL all-reduce)"
    )'''
new = '''        max_lora_rank=args.lora_rank,
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
    )'''
assert old in s
s = s.replace(old, new, 1)

# contract-shape experiments + bench lora, inserted before --bench
old = '''        if args.bench:
            result["bench"] = run_bench(
                engine, prompts, batch=args.bench_batch, tokens=args.bench_tokens
            )
            flush()'''
new = '''        if args.long_window or args.decode_batches:
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
            flush()'''
assert old in s
s = s.replace(old, new, 1)

old = '''        "lora_random_identity": (result.get("lora_random") or {}).get("hash_identity"),
    }'''
new = '''        "lora_random_identity": (result.get("lora_random") or {}).get("hash_identity"),
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
    }'''
assert old in s
s = s.replace(old, new, 1)
p.write_text(s)
print("ok")
