from pathlib import Path

p = Path("/home/ubuntu/RL-Kernel/.claude/worktrees/wf_65f5032c-579-2/rl_engine/models/muse_glimmer/runtime/vllm_collect_results.py")
s = p.read_text()
old = '''def _runs_in(directory: Path | None) -> list[dict]:'''
new = '''def _code_dirty(run: dict) -> Any:
    """True if a run's tree had modified files outside ``reports/`` (data-only edits are not
    code changes: the runs themselves rewrite the xrt dumps)."""

    if "git_dirty_code" in run:
        return run["git_dirty_code"]
    lines = run.get("git_status_porcelain")
    if lines is None:
        return run.get("git_dirty")
    return any(not line[3:].startswith("reports/") for line in lines)


def _runs_in(directory: Path | None) -> list[dict]:'''
assert old in s
s = s.replace(old, new, 1)
old = '''        "dirty_runs": [r["_file"] for r in runs if r.get("git_dirty_code", r.get("git_dirty"))],
        "dirty_runs_data_only": [
            {"run": r["_file"], "modified": r.get("git_status_porcelain")}
            for r in runs
            if r.get("git_dirty") and not r.get("git_dirty_code", True)
        ],'''
new = '''        "dirty_runs": [r["_file"] for r in runs if _code_dirty(r)],
        "dirty_runs_data_only": [
            {"run": r["_file"], "modified": r.get("git_status_porcelain")}
            for r in runs
            if r.get("git_dirty") and not _code_dirty(r)
        ],'''
assert old in s
s = s.replace(old, new, 1)
s = s.replace('''                "git_dirty_code": r.get("git_dirty_code"),''', '''                "git_dirty_code": _code_dirty(r),''', 1)
p.write_text(s)
print("ok")
