from pathlib import Path

root = Path("/home/ubuntu/RL-Kernel/.claude/worktrees/wf_65f5032c-579-2/rl_engine/models/muse_glimmer/runtime")
p = root / "vllm_collect_results.py"
s = p.read_text()
old = '''def _check(check: dict | None) -> Any:
    if not check:
        return "NOT RUN"
    return {k: check.get(k) for k in _CHECK_KEYS if k in check}
'''
new = '''def _check(check: dict | None) -> Any:
    if not check:
        return "NOT RUN"
    out = {k: check.get(k) for k in _CHECK_KEYS if k in check}
    if out.get("mismatched") == 0 and "prefill_logprobs" in out:
        # Bitwise equal to the decode logprobs stored next to this record: do not store twice.
        out["prefill_logprobs"] = "bitwise identical to the record's logprobs"
    return out
'''
assert old in s
s = s.replace(old, new, 1)
old = '''    Path(args.out).write_text(json.dumps(result, indent=1), encoding="utf-8")'''
new = '''    # Compact: the raw per-token series dominate the size; indent would put each float on a line.
    Path(args.out).write_text(json.dumps(result, separators=(",", ":")), encoding="utf-8")'''
assert old in s
s = s.replace(old, new, 1)
p.write_text(s)

p = root / "vllm_report.py"
s = p.read_text()
old = '''    data_path.write_text(json.dumps(results, indent=1), encoding="utf-8")'''
new = '''    data_path.write_text(json.dumps(results, separators=(",", ":")), encoding="utf-8")'''
assert old in s
s = s.replace(old, new, 1)
p.write_text(s)
print("ok")
