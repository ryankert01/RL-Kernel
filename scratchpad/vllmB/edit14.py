from pathlib import Path

root = Path("/home/ubuntu/RL-Kernel/.claude/worktrees/wf_65f5032c-579-2/rl_engine/models/muse_glimmer/runtime")
p = root / "vllm_attention_backend.py"
s = p.read_text()
old = '''PAGE_SIZE = 16
CALL_COUNTS: Counter[str] = Counter()
_OPS: Any = None
'''
new = '''PAGE_SIZE = 16
CALL_COUNTS: Counter[str] = Counter()
_OPS: Any = None
# Opt-in evidence counter (measurement mode, a few extra GPU ops per local-layer call): how many
# block-table entries inside a sequence's page range are vLLM's null block 0, i.e. sliding-window
# pages vLLM has already evicted when the strict op reads the table.  Read with
# :func:`null_page_counts` (one host sync).
COUNT_NULL_PAGES_ENV = "RL_KERNEL_MUSEGLIMMER_COUNT_NULL_PAGES"
_NULL_PAGE_ACC: dict[str, Any] = {}


def _count_null_pages_enabled() -> bool:
    import os

    return os.getenv(COUNT_NULL_PAGES_ENV, "").strip() in ("1", "true", "yes")


def null_page_counts() -> dict[str, int]:
    """Accumulated in-range null-block entries seen by local / global layers (opt-in)."""

    return {key: int(acc.item()) for key, acc in _NULL_PAGE_ACC.items()}


def _accumulate_null_pages(
    key: str, block_table: torch.Tensor, seq_lens: torch.Tensor, num_reqs: int
) -> None:
    table = block_table[:num_reqs]
    pages = (seq_lens[:num_reqs].to(torch.int64) + PAGE_SIZE - 1) // PAGE_SIZE
    cols = torch.arange(table.shape[1], device=table.device)
    hits = ((table == 0) & (cols[None, :] < pages[:, None])).sum()
    acc = _NULL_PAGE_ACC.get(key)
    if acc is None:
        acc = torch.zeros((), dtype=torch.int64, device=table.device)
        _NULL_PAGE_ACC[key] = acc
    acc.add_(hits)
'''
assert old in s
s = s.replace(old, new, 1)
old = '''            block_table = strict_block_table(attn_metadata.block_table[:num_reqs])
            window = None if self.sliding_window[0] < 0 else int(self.sliding_window[0]) + 1'''
new = '''            block_table = strict_block_table(attn_metadata.block_table[:num_reqs])
            window = None if self.sliding_window[0] < 0 else int(self.sliding_window[0]) + 1
            if _count_null_pages_enabled():
                _accumulate_null_pages(
                    "local" if window is not None else "global",
                    attn_metadata.block_table,
                    seq_lens,
                    num_reqs,
                )'''
assert old in s
s = s.replace(old, new, 1)
s = s.replace('''__all__ = [
    "BACKEND_CLASS_PATH",
    "CALL_COUNTS",''', '''__all__ = [
    "BACKEND_CLASS_PATH",
    "CALL_COUNTS",
    "COUNT_NULL_PAGES_ENV",''', 1)
s = s.replace('''    "bind_ops",
    "register",''', '''    "bind_ops",
    "null_page_counts",
    "register",''', 1)
p.write_text(s)

p = root / "vllm_strict.py"
s = p.read_text()
old = '''        counts.update({k: int(v) for k, v in attn_backend.CALL_COUNTS.items()})'''
new = '''        counts.update({k: int(v) for k, v in attn_backend.CALL_COUNTS.items()})
        counts.update(
            {
                f"attention.{k}_null_pages_in_range": v
                for k, v in attn_backend.null_page_counts().items()
            }
        )'''
assert old in s
s = s.replace(old, new, 1)
p.write_text(s)

p = root / "run_vllm_strict.py"
s = p.read_text()
old = '''    parser.add_argument("--contract-lora", action="store_true",
                        help="also run --long-window/--decode-batches with the random LoRA")'''
new = '''    parser.add_argument("--contract-lora", action="store_true",
                        help="also run --long-window/--decode-batches with the random LoRA")
    parser.add_argument(
        "--count-null-pages",
        action="store_true",
        help="strict: count evicted (null) pages the attention reads (measurement mode)",
    )'''
assert old in s
s = s.replace(old, new, 1)
old = '''    t_load = time.perf_counter()
    engine = StrictRolloutEngine('''
new = '''    if args.count_null_pages:
        from rl_engine.models.muse_glimmer.runtime.vllm_attention_backend import (
            COUNT_NULL_PAGES_ENV,
        )

        os.environ[COUNT_NULL_PAGES_ENV] = "1"
        result["count_null_pages"] = True
    t_load = time.perf_counter()
    engine = StrictRolloutEngine('''
assert old in s
s = s.replace(old, new, 1)
# per contract record: null-page counter delta from worker provenance
old = '''    rec = run_batch_check(
                        engine,
                        f"long_window_{lora_label}",'''
new = '''    null_before = _null_pages(engine)
                    rec = run_batch_check(
                        engine,
                        f"long_window_{lora_label}",'''
assert old in s
s = s.replace(old, new, 1)
s = s.replace('''    null_before = _null_pages(engine)
                    rec = run_batch_check(''', '''null_before = _null_pages(engine)
                    rec = run_batch_check(''', 1)
old = '''                        ids=[f"{long_ids[0]}#{i}" for i in range(copies)],
                    )
                    rec["lora_sha256"] = engine._lora_sha256'''
new = '''                        ids=[f"{long_ids[0]}#{i}" for i in range(copies)],
                    )
                    rec["lora_sha256"] = engine._lora_sha256
                    rec["null_pages_read"] = _null_pages_delta(null_before, _null_pages(engine))'''
assert old in s
s = s.replace(old, new, 1)
old = '''def run_batch_check('''
new = '''def _null_pages(engine: Any) -> dict[str, int] | None:
    """Rank-0 in-range null-page counters of the strict backend (None unless enabled)."""

    if not engine.strict or not os.getenv("RL_KERNEL_MUSEGLIMMER_COUNT_NULL_PAGES"):
        return None
    workers = engine.provenance().get("workers") or [{}]
    counts = workers[0].get("call_counts") or {}
    return {k: int(v) for k, v in counts.items() if k.endswith("_null_pages_in_range")}


def _null_pages_delta(before: Any, after: Any) -> Any:
    if before is None or after is None:
        return "not measured (run with --count-null-pages)"
    return {
        k: int(after.get(k, 0)) - int(before.get(k, 0)) for k in sorted(set(before) | set(after))
    } | {
        "definition": "block-table entries inside a sequence's page range that are vLLM's null "
        "block 0 (evicted sliding-window pages) when the strict op read them, summed over "
        "layer calls on rank 0"
    }


def run_batch_check('''
assert old in s
s = s.replace(old, new, 1)
p.write_text(s)
print("ok")
