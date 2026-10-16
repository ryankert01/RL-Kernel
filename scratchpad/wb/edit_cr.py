import pathlib

p = pathlib.Path("rl_engine/models/muse_glimmer/reference/collect_results.py")
s = p.read_text()


def rep(old, new):
    global s
    assert s.count(old) == 1, old[:80]
    s = s.replace(old, new)


rep('''import argparse
import json
import pathlib
import shutil
import sys
''', '''import argparse
import json
import pathlib
import shutil
import subprocess
import sys
''')
rep('''DATA = REPO_ROOT / "reports" / "muse_glimmer_poc" / "data"
''', '''DATA = REPO_ROOT / "reports" / "muse_glimmer_poc" / "data"
# The O7 fix (softcap kernels without FMA contraction) changed strict Triton logprob bits; a
# measurement of the triton backend taken before it is stale for bitwise comparisons.
O7_FIX_COMMIT = "c88607f"


def includes_commit(sha: str | None, ancestor: str = O7_FIX_COMMIT) -> bool | None:
    """True if ``ancestor`` is an ancestor of (or equal to) ``sha``; None if unknown."""
    if not sha:
        return None
    proc = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "merge-base", "--is-ancestor", ancestor, sha],
        capture_output=True, text=True, check=False,
    )
    return {0: True, 1: False}.get(proc.returncode)


def provenance(payload: dict[str, Any] | None) -> dict[str, Any]:
    """The commit a run file says it measured (captured at run start) and whether it includes
    the O7 fix."""
    if payload is None:
        return {"commit": None, "includes_o7_fix": None}
    c = payload.get("commit") or {}
    sha = c.get("sha") if isinstance(c, dict) else c
    return {"commit": c, "includes_o7_fix": includes_commit(sha)}
''')
rep('''    train = chain["results"][0]
    rec: dict[str, Any] = {
        "status": "PASS" if chain["all_equal"] else "FAIL",''', '''    train = chain["results"][0]
    rec: dict[str, Any] = {
        "status": "PASS" if chain["all_equal"] else "FAIL",
        **provenance(payload),''')
rep('''        out[name] = {
            "status": "PASS" if x["checks"]["all_byte_equal_to_train"] else "FAIL",
            "commit": x.get("commit"),''', '''        out[name] = {
            "status": "PASS" if x["checks"]["all_byte_equal_to_train"] else "FAIL",
            **provenance(x),
            "n_checks": x["checks"].get("n_checks"),''')
rep('''            r = dict(r)
            r["by_group_ms"], r["by_group_frac"] = regroup(r["by_kind_class_ms"])
            kp = r.get("kernel_profile") or {}
            if "by_kind_class_gpu_ms" in kp:
                r["by_group_gpu_ms"], r["by_group_gpu_frac"] = regroup(kp["by_kind_class_gpu_ms"])''', '''            r = dict(r)
            r["by_group_ms"], r["by_group_frac"] = regroup(r["by_kind_class_ms"])
            kp = dict(r.get("kernel_profile") or {})
            if "by_kind_class_gpu_ms" in kp:
                r["by_group_gpu_ms"], r["by_group_gpu_frac"] = regroup(kp["by_kind_class_gpu_ms"])
                # ONE grouping in the output: the nested field is replaced by the regrouped
                # values (a profile recorded before a group existed would disagree otherwise)
                kp["by_group_gpu_ms"] = r["by_group_gpu_ms"]
                kp["groups"] = list(GROUPS)
                r["kernel_profile"] = kp
                reps = r.get("kernel_profile_repeats") or {}
                if reps.get("by_kind_class_gpu_ms"):
                    groups_per_rep = [regroup(k)[0] for k in reps["by_kind_class_gpu_ms"]]
                    r["kernel_profile_repeats"] = {
                        **{k: v for k, v in reps.items() if k != "by_kind_class_gpu_ms"},
                        "by_group_gpu_ms": {g: [x[g] for x in groups_per_rep] for g in GROUPS},
                    }''')
rep('''    result = {
        "row": "full_model_chain follow-up: reference scoring of xrt_tokens.json + single-GPU "
               "profiling",
        "status": args.status,
        "commit": source_commit(),''', '''    chains = {
        "chain_triton_512_hf": _load(runs / "chain_triton_512_hf.json"),
        "chain_triton_2100_hf": _load(runs / "chain_triton_2100_hf.json"),
        "chain_golden_2100": _load(runs / "chain_golden_2100.json"),
    }
    repeat = _load(runs / "xrt_reference_triton_repeat.json")
    if repeat is not None:
        shutil.copyfile(runs / "xrt_reference_triton_repeat.json",
                        DATA / "xrt_reference_triton_repeat.json")
    measured = {name: provenance(x) for name, x in (
        *chains.items(),
        ("xrt_reference_triton", triton),
        ("xrt_reference_triton_repeat", repeat),
        ("xrt_reference_golden", golden),
        ("xrt_reference_hf_bf16", hf_x),
        ("profile_strict", strict),
        ("profile_hf", hf),
    )}
    result = {
        "row": "full_model_chain follow-up: reference scoring of xrt_tokens.json + single-GPU "
               "profiling",
        "status": args.status,
        "collected_at_commit": source_commit(),
        "measured_commits": measured,''')
rep('''        "experiments": {
            "chain_triton_512_hf_pre_change": chain_record(
                _load(runs / "chain_triton_512_hf_base.json")),
            "chain_triton_512_hf": chain_record(_load(runs / "chain_triton_512_hf.json")),
            "chain_triton_2100_hf": chain_record(_load(runs / "chain_triton_2100_hf.json")),
            "chain_golden_2100": chain_record(_load(runs / "chain_golden_2100.json")),
            "xrt_scoring": xrt_record(golden, triton, hf_x),
            "xrt_triton_repeat": _repeat_record(_load(runs / "xrt_reference_triton_first.json"),
                                                triton),''', '''        "experiments": {
            **{name: chain_record(x) for name, x in chains.items()},
            "xrt_scoring": xrt_record(golden, triton, hf_x),
            "xrt_triton_repeat": _repeat_record(repeat, triton),''')
p.write_text(s)
print("ok")
