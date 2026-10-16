from pathlib import Path

root = Path("/home/ubuntu/RL-Kernel/.claude/worktrees/wf_65f5032c-579-2/rl_engine/models/muse_glimmer/runtime")
p = root / "vllm_worker_tools.py"
s = p.read_text()
old = '''def _scheduled(scheduler_output: Any) -> tuple[int, int]:
    tokens = int(getattr(scheduler_output, "total_num_scheduled_tokens", 0) or 0)
    per_req = getattr(scheduler_output, "num_scheduled_tokens", None) or {}
    return tokens, len(per_req)
'''
new = '''def _scheduled(scheduler_output: Any) -> tuple[int, int]:
    tokens = int(getattr(scheduler_output, "total_num_scheduled_tokens", 0) or 0)
    per_req = getattr(scheduler_output, "num_scheduled_tokens", None) or {}
    return tokens, len(per_req)


_SCHED_ATTR = "_rlk_museglimmer_sched_counters"


def _count_scheduler_events(worker: Any, scheduler_output: Any) -> tuple[int, int]:
    """Preempted / resumed request counts of one scheduler output (accumulated on the worker)."""

    preempted = len(getattr(scheduler_output, "preempted_req_ids", None) or ())
    cached = getattr(scheduler_output, "scheduled_cached_reqs", None)
    resumed = len(getattr(cached, "resumed_req_ids", None) or ())
    counters = getattr(worker, _SCHED_ATTR, None)
    if counters is None:
        counters = {"preempted": 0, "resumed": 0}
        setattr(worker, _SCHED_ATTR, counters)
    counters["preempted"] += preempted
    counters["resumed"] += resumed
    return preempted, resumed
'''
assert old in s
s = s.replace(old, new, 1)
old = '''def worker_step_timer(worker: Any, action: str = "start") -> list[dict[str, Any]] | None:
    """``action='start'`` installs the per-step timer; ``'stop'`` removes it and returns steps."""

    if action == "start":'''
new = '''def worker_step_timer(worker: Any, action: str = "start") -> Any:
    """``action='start'`` installs the per-step timer; ``'stop'`` removes it and returns steps;
    ``'counters'`` returns the preempted / resumed request totals seen while timing."""

    if action == "counters":
        return dict(getattr(worker, _SCHED_ATTR, None) or {"preempted": 0, "resumed": 0})
    if action == "start":'''
assert old in s
s = s.replace(old, new, 1)
old = '''        def execute_model(scheduler_output: Any, *args: Any, **kwargs: Any) -> Any:
            tokens, reqs = _scheduled(scheduler_output)
            if tokens == 0:
                return original_execute(scheduler_output, *args, **kwargs)
            torch.cuda.synchronize()
            start = time.perf_counter()
            out = original_execute(scheduler_output, *args, **kwargs)
            if out is not None:
                torch.cuda.synchronize()
                record["steps"].append(
                    {
                        "tokens": tokens,
                        "reqs": reqs,
                        "ms": (time.perf_counter() - start) * 1e3,
                        "sampled_in_execute": True,
                    }
                )
            else:
                record["open"] = (start, tokens, reqs)
            return out

        def sample_tokens(*args: Any, **kwargs: Any) -> Any:
            out = original_sample(*args, **kwargs)
            pending = record["open"]
            if pending is not None:
                torch.cuda.synchronize()
                start, tokens, reqs = pending
                record["steps"].append(
                    {"tokens": tokens, "reqs": reqs, "ms": (time.perf_counter() - start) * 1e3}
                )
                record["open"] = None
            return out
'''
new = '''        def execute_model(scheduler_output: Any, *args: Any, **kwargs: Any) -> Any:
            tokens, reqs = _scheduled(scheduler_output)
            preempted, resumed = _count_scheduler_events(worker, scheduler_output)
            if tokens == 0:
                return original_execute(scheduler_output, *args, **kwargs)
            torch.cuda.synchronize()
            start = time.perf_counter()
            out = original_execute(scheduler_output, *args, **kwargs)
            if out is not None:
                torch.cuda.synchronize()
                record["steps"].append(
                    {
                        "tokens": tokens,
                        "reqs": reqs,
                        "ms": (time.perf_counter() - start) * 1e3,
                        "preempted": preempted,
                        "resumed": resumed,
                        "sampled_in_execute": True,
                    }
                )
            else:
                record["open"] = (start, tokens, reqs, preempted, resumed)
            return out

        def sample_tokens(*args: Any, **kwargs: Any) -> Any:
            out = original_sample(*args, **kwargs)
            pending = record["open"]
            if pending is not None:
                torch.cuda.synchronize()
                start, tokens, reqs, preempted, resumed = pending
                record["steps"].append(
                    {
                        "tokens": tokens,
                        "reqs": reqs,
                        "ms": (time.perf_counter() - start) * 1e3,
                        "preempted": preempted,
                        "resumed": resumed,
                    }
                )
                record["open"] = None
            return out
'''
assert old in s
s = s.replace(old, new, 1)
p.write_text(s)

p = root / "run_vllm_strict.py"
s = p.read_text()
old = '''    llm.collective_rpc(tools.worker_step_timer, kwargs={"action": "start"})
    t0 = time.perf_counter()
    try:
        outs = llm.generate('''
new = '''    counters_before = llm.collective_rpc(tools.worker_step_timer, kwargs={"action": "counters"})
    llm.collective_rpc(tools.worker_step_timer, kwargs={"action": "start"})
    t0 = time.perf_counter()
    try:
        outs = llm.generate('''
assert old in s
s = s.replace(old, new, 1)
old = '''    finally:
        steps = llm.collective_rpc(tools.worker_step_timer, kwargs={"action": "stop"})
    seconds = time.perf_counter() - t0'''
new = '''    finally:
        steps = llm.collective_rpc(tools.worker_step_timer, kwargs={"action": "stop"})
    seconds = time.perf_counter() - t0
    counters_after = llm.collective_rpc(tools.worker_step_timer, kwargs={"action": "counters"})
    events = {
        key: int(counters_after[0][key]) - int(counters_before[0][key])
        for key in ("preempted", "resumed")
    }'''
assert old in s
s = s.replace(old, new, 1)
old = '''        finish.append(str(completion.finish_reason))
    return tokens, logprobs, finish, seconds, (steps[0] if steps else [])'''
new = '''        finish.append(str(completion.finish_reason))
    rank0 = list(steps[0]) if steps else []
    return tokens, logprobs, finish, seconds, rank0, events'''
assert old in s
s = s.replace(old, new, 1)
old = ''') -> tuple[list[list[int]], list[list[float]], list[str], float, list[dict[str, Any]]]:'''
new = ''') -> tuple[
    list[list[int]], list[list[float]], list[str], float, list[dict[str, Any]], dict[str, int]
]:'''
assert old in s
s = s.replace(old, new, 1)
old = '''    tokens, logprobs, finish, seconds, steps = _raw_generate('''
new = '''    tokens, logprobs, finish, seconds, steps, events = _raw_generate('''
assert old in s
s = s.replace(old, new, 1)
old = '''        "engine_steps": _step_summary(steps),
    }'''
new = '''        "engine_steps": _step_summary(steps),
        "scheduler_events": events,
    }'''
assert old in s
s = s.replace(old, new, 1)
old = '''    parser.add_argument("--max-num-batched-tokens", type=int, default=None)'''
new = '''    parser.add_argument("--max-num-batched-tokens", type=int, default=None)
    parser.add_argument(
        "--num-gpu-blocks-override",
        type=int,
        default=None,
        help="tiny KV pool: forces page reuse and preemption/recompute",
    )
    parser.add_argument("--contract-tag", default="", help="suffix for contract-shape labels")'''
assert old in s
s = s.replace(old, new, 1)
old = '''        max_num_batched_tokens=args.max_num_batched_tokens,
    )'''
new = '''        max_num_batched_tokens=args.max_num_batched_tokens,
        num_gpu_blocks_override=args.num_gpu_blocks_override,
    )'''
assert old in s
s = s.replace(old, new, 1)
s = s.replace('''                        f"long_window_{lora_label}",''', '''                        f"long_window{args.contract_tag}_{lora_label}",''', 1)
s = s.replace('''                        f"decode_batch_{width}_{lora_label}",''', '''                        f"decode_batch_{width}{args.contract_tag}_{lora_label}",''', 1)
p.write_text(s)

p = root / "strict_rollout_engine.py"
s = p.read_text()
old = '''        enforce_eager: bool = True,
        max_num_batched_tokens: int | None = None,
    ) -> None:'''
new = '''        enforce_eager: bool = True,
        max_num_batched_tokens: int | None = None,
        num_gpu_blocks_override: int | None = None,
    ) -> None:'''
assert old in s
s = s.replace(old, new, 1)
old = '''        if max_num_batched_tokens is not None:
            kwargs["max_num_batched_tokens"] = int(max_num_batched_tokens)'''
new = '''        if max_num_batched_tokens is not None:
            kwargs["max_num_batched_tokens"] = int(max_num_batched_tokens)
        if num_gpu_blocks_override is not None:
            kwargs["num_gpu_blocks_override"] = int(num_gpu_blocks_override)'''
assert old in s
s = s.replace(old, new, 1)
p.write_text(s)
print("ok")
