from pathlib import Path

p = Path("/home/ubuntu/RL-Kernel/.claude/worktrees/wf_65f5032c-579-2/rl_engine/models/muse_glimmer/runtime/vllm_collect_results.py")
s = p.read_text()
old = '''def compare_xrt(a: dict, b: dict) -> dict[str, Any]:
    """Token-level bitwise comparison of two muse_glimmer_xrt_logprobs_v1 files."""
'''
new = '''def xrt_label(doc: dict) -> str:
    """Human label of an xrt dump: runtime/mode, LoRA marker, TP (reference chain: its mode)."""

    if str(doc.get("runtime", "")).startswith("reference"):
        mode = str(doc.get("backend", "")).split("mode=")[-1] if "mode=" in str(
            doc.get("backend", "")
        ) else str(doc.get("mode", ""))
        return f"reference chain {mode}".strip()
    lora = " lora" if doc.get("lora_sha256") else ""
    return f"{doc['mode']}{lora} tp{doc['tp']}"


def compare_xrt(
    a: dict, b: dict, *, a_label: str | None = None, b_label: str | None = None
) -> dict[str, Any]:
    """Token-level bitwise comparison of two muse_glimmer_xrt_logprobs_v1 files."""
'''
assert old in s
s = s.replace(old, new, 1)
old = '''    return {
        "a": f"{a['mode']} tp{a['tp']}",
        "mean_abs_diff": sum_abs / total if total else 0.0,
        "b": f"{b['mode']} tp{b['tp']}",'''
new = '''    return {
        "a": a_label or xrt_label(a),
        "b": b_label or xrt_label(b),
        "mean_abs_diff": sum_abs / total if total else 0.0,'''
assert old in s
s = s.replace(old, new, 1)
p.write_text(s)
print("ok")
