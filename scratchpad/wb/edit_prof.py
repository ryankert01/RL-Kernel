import pathlib

p = pathlib.Path("rl_engine/models/muse_glimmer/reference/profiling.py")
s = p.read_text()


def rep(old, new, count=1):
    global s
    assert s.count(old) == count, (old[:90], s.count(old))
    s = s.replace(old, new)


# attributed_kernel_profile: optional prepare() before the profiled call
rep('''    num_layers: int,
    root_cls: str = _ROOT,
    top: int = 12,
) -> dict[str, Any]:
    """One extra call under torch.profiler''', '''    num_layers: int,
    root_cls: str = _ROOT,
    top: int = 12,
    prepare: Callable[[], Any] | None = None,
) -> dict[str, Any]:
    """One extra call under torch.profiler''')
rep('''    from torch.profiler import ProfilerActivity, profile

    torch.cuda.synchronize()
    install(timer)''', '''    from torch.profiler import ProfilerActivity, profile

    if prepare is not None:
        prepare()  # per-call state (e.g. a fresh KV cache) outside the profiled region
    torch.cuda.synchronize()
    install(timer)''')

rep('''def measure(
    fn: Callable[[], Any],
    timer: RegionTimer,
    install: Callable[[RegionTimer | None], None],
    *,
    num_layers: int,
    warmup: int,
    repeats: int,
    root_cls: str = _ROOT,
    profiler_check: bool = False,
) -> dict[str, Any]:
    """Warm up, then ``repeats`` instrumented calls (regions) and ``repeats`` clean calls."""
    dev = torch.device("cuda")
    install(None)
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    mem_before = torch.cuda.memory_allocated(dev)
    torch.cuda.reset_peak_memory_stats(dev)
    clean = []
    for _ in range(repeats):
        _, ms = timed_call(fn)
        clean.append(ms)
    peak = torch.cuda.max_memory_allocated(dev)
    summaries = []
    install(timer)
    try:
        for _ in range(repeats):
            timer.reset()
            idx = timer.begin(root_cls, "model", None)
            fn()
            timer.end(idx)
            summaries.append(summarize_regions(timer.resolve(), num_layers))
    finally:
        install(None)
    out = combine_repeats(summaries, num_layers)
    out["clean_total_ms_repeats"] = clean
    out["clean_total_ms_median"] = _median(clean)
    out["instrumentation_overhead_frac"] = (
        out["total_ms_median"] / out["clean_total_ms_median"] - 1.0
        if out["clean_total_ms_median"] > 0 else float("nan")
    )
    out["mem_allocated_before_gib"] = _gib(mem_before)
    out["peak_allocated_gib"] = _gib(peak)
    out["activation_peak_gib"] = _gib(peak - mem_before)
    if profiler_check:
        try:
            out["kernel_profile"] = attributed_kernel_profile(
                fn, timer, install, num_layers=num_layers, root_cls=root_cls
            )
        except Exception as exc:  # the cross-check must never sink the measurement
            timer.annotate = False
            install(None)
            timer._stack.clear()
            timer._recs = []
            out["kernel_profile"] = {"error": f"{type(exc).__name__}: {exc}"}
    return out''', '''def measure(
    fn: Callable[[], Any],
    timer: RegionTimer,
    install: Callable[[RegionTimer | None], None],
    *,
    num_layers: int,
    warmup: int,
    repeats: int,
    root_cls: str = _ROOT,
    profiler_check: bool = False,
    profiler_repeats: int = 1,
    prepare: Callable[[], Any] | None = None,
    release: Callable[[], Any] | None = None,
) -> dict[str, Any]:
    """Warm up, then ``repeats`` clean calls and ``repeats`` instrumented calls (regions), then
    ``profiler_repeats`` torch.profiler calls (the first keeps every per-class/per-layer field
    in ``kernel_profile``; all of them are summarised in ``kernel_profile_repeats``).

    ``prepare()`` runs before EVERY call, outside the timed region (per-call state such as a
    fresh paged KV cache); ``release()`` drops that state after the warm-up, so
    ``mem_allocated_before_gib`` holds no per-call state and ``activation_peak_gib`` includes
    exactly one call's state (``prepared_state_gib``, measured around the first clean
    ``prepare()``) plus its activations."""
    dev = torch.device("cuda")
    install(None)

    def prep() -> None:
        if prepare is not None:
            prepare()

    for _ in range(warmup):
        prep()
        fn()
    if release is not None:
        release()
    torch.cuda.synchronize()
    mem_before = torch.cuda.memory_allocated(dev)
    torch.cuda.reset_peak_memory_stats(dev)
    clean = []
    prepared_bytes = None
    for _ in range(repeats):
        if prepare is not None:
            before = torch.cuda.memory_allocated(dev)
            prep()
            if prepared_bytes is None:
                prepared_bytes = torch.cuda.memory_allocated(dev) - before
        _, ms = timed_call(fn)
        clean.append(ms)
    peak = torch.cuda.max_memory_allocated(dev)
    summaries = []
    install(timer)
    try:
        for _ in range(repeats):
            prep()
            timer.reset()
            idx = timer.begin(root_cls, "model", None)
            fn()
            timer.end(idx)
            summaries.append(summarize_regions(timer.resolve(), num_layers))
    finally:
        install(None)
    out = combine_repeats(summaries, num_layers)
    out["clean_total_ms_repeats"] = clean
    out["clean_total_ms_median"] = _median(clean)
    out["instrumentation_overhead_frac"] = (
        out["total_ms_median"] / out["clean_total_ms_median"] - 1.0
        if out["clean_total_ms_median"] > 0 else float("nan")
    )
    out["mem_allocated_before_gib"] = _gib(mem_before)
    out["peak_allocated_gib"] = _gib(peak)
    out["activation_peak_gib"] = _gib(peak - mem_before)
    out["prepared_state_gib"] = None if prepared_bytes is None else _gib(prepared_bytes)
    if profiler_check:
        reps = []
        for r in range(max(1, int(profiler_repeats))):
            try:
                kp = attributed_kernel_profile(
                    fn, timer, install, num_layers=num_layers, root_cls=root_cls,
                    prepare=prepare,
                )
            except Exception as exc:  # the cross-check must never sink the measurement
                timer.annotate = False
                install(None)
                timer._stack.clear()
                timer._recs = []
                kp = {"error": f"{type(exc).__name__}: {exc}"}
            if r == 0:
                out["kernel_profile"] = kp
            reps.append(kp)
        ok = [k for k in reps if "error" not in k]
        out["kernel_profile_repeats"] = {
            "n": len(reps),
            "n_ok": len(ok),
            "gpu_busy_ms": [k["gpu_busy_ms"] for k in ok],
            "by_kind_class_gpu_ms": [k["by_kind_class_gpu_ms"] for k in ok],
            "gpu_busy_ms_median": _median([k["gpu_busy_ms"] for k in ok]),
        }
    if release is not None:
        release()
    return out''')

rep('''        if paged_prefill:
            # fresh sessions are allocated up front so the cache zero-fill is not timed
            sessions = [model.new_session(t)
                        for _ in range(warmup + 2 * repeats + int(profiler_check))]

            def paged_call(ids=ids, targets=targets, sessions=sessions):
                return model.prefill(sessions.pop(), ids, targets=targets)

            log(f"[strict] paged prefill T={t}")
            res = measure(paged_call, timer, model.set_timer, num_layers=nl, warmup=warmup,
                          repeats=repeats, profiler_check=profiler_check)
            del sessions
            runs.append({"phase": "prefill_paged", "T": t, "batch": 1, "context": 0,
                         "tokens": t, **res})''', '''        if paged_prefill:
            # one fresh session per call, allocated by prepare() outside the timed region (the
            # cache zero-fill is not timed) and only one alive at a time (peak = one cache)
            holder: dict[str, Any] = {}

            def prepare(holder=holder, t=t):
                holder.pop("session", None)
                holder["session"] = model.new_session(t)

            def paged_call(ids=ids, targets=targets, holder=holder):
                return model.prefill(holder["session"], ids, targets=targets)

            log(f"[strict] paged prefill T={t}")
            res = measure(paged_call, timer, model.set_timer, num_layers=nl, warmup=warmup,
                          repeats=repeats, profiler_check=profiler_check,
                          profiler_repeats=profiler_repeats, prepare=prepare,
                          release=holder.clear)
            runs.append({"phase": "prefill_paged", "T": t, "batch": 1, "context": 0,
                         "tokens": t, **res})''')

rep('''        res = measure(train_call, timer, model.set_timer, num_layers=nl, warmup=warmup,
                      repeats=repeats, profiler_check=profiler_check)''', '''        res = measure(train_call, timer, model.set_timer, num_layers=nl, warmup=warmup,
                      repeats=repeats, profiler_check=profiler_check,
                      profiler_repeats=profiler_repeats)''')
rep('''            res = measure(decode_call, timer, model.set_timer, num_layers=nl, warmup=warmup,
                          repeats=decode_repeats, profiler_check=profiler_check)''', '''            res = measure(decode_call, timer, model.set_timer, num_layers=nl, warmup=warmup,
                          repeats=decode_repeats, profiler_check=profiler_check,
                          profiler_repeats=profiler_repeats)''')
# profile_strict / profile_hf signature: profiler_repeats
rep('''    paged_prefill: bool = True,
    batch_invariance: bool = True,
    profiler_check: bool = True,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:''', '''    paged_prefill: bool = True,
    batch_invariance: bool = True,
    profiler_check: bool = True,
    profiler_repeats: int = 3,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:''')
rep('''    decode_repeats: int = 5,
    profiler_check: bool = True,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """The same configurations on HF bf16''', '''    decode_repeats: int = 5,
    profiler_check: bool = True,
    profiler_repeats: int = 3,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """The same configurations on HF bf16''')
rep('''            res = measure(call, timer, inst.install, num_layers=nl, warmup=warmup,
                          repeats=repeats, profiler_check=profiler_check)''', '''            res = measure(call, timer, inst.install, num_layers=nl, warmup=warmup,
                          repeats=repeats, profiler_check=profiler_check,
                          profiler_repeats=profiler_repeats)''')
rep('''            res = measure(step, timer, inst.install, num_layers=nl, warmup=warmup,
                          repeats=decode_repeats, profiler_check=profiler_check)''', '''            res = measure(step, timer, inst.install, num_layers=nl, warmup=warmup,
                          repeats=decode_repeats, profiler_check=profiler_check,
                          profiler_repeats=profiler_repeats)''')

# R6: bit-pattern compare in decode_batch_invariance
rep('''    sub = model.decode_batch(cache, perm, toks[perm], tgts[perm], commit=False).cpu()
    eq_full = int(torch.eq(full, single).sum())
    eq_perm = int(torch.eq(sub, single[perm]).sum())''', '''    sub = model.decode_batch(cache, perm, toks[perm], tgts[perm], commit=False).cpu()
    # bit patterns, not float ==: -0.0 vs +0.0 or NaN payloads count as different
    eq_full = int(torch.eq(full.view(torch.int32), single.view(torch.int32)).sum())
    eq_perm = int(torch.eq(sub.view(torch.int32), single[perm].view(torch.int32)).sum())''')
rep('''        "permutation": perm,
        "all_equal": eq_full == bmax and eq_perm == len(perm),''', '''        "permutation": perm,
        "compare": "fp32 bit patterns (view(int32))",
        "all_equal": eq_full == bmax and eq_perm == len(perm),''')
p.write_text(s)
print("ok")
