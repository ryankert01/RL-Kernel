from pathlib import Path

p = Path("/home/ubuntu/RL-Kernel/.claude/worktrees/wf_65f5032c-579-2/rl_engine/models/muse_glimmer/runtime/vllm_xrt_reference.py")
s = p.read_text()
old = '''    from rl_engine.models.muse_glimmer.reference.strict_model import MuseGlimmerStrictModel
    from rl_engine.models.muse_glimmer.reference.weights import MuseGlimmerWeights
    from rl_engine.validation.operators.accelerator import disable_tf32
'''
new = '''    from rl_engine.models.muse_glimmer.reference.strict_model import MuseGlimmerStrictModel
    from rl_engine.models.muse_glimmer.reference.weights import MuseGlimmerWeights
    from rl_engine.models.muse_glimmer.runtime.run_vllm_strict import _git_head, _git_status
    from rl_engine.validation.operators.accelerator import disable_tf32
'''
assert old in s
s = s.replace(old, new, 1)
old = '''            "backend": f"rl_engine.models.muse_glimmer.reference triton backend, mode={mode}",
            "score_ms": (time.perf_counter() - t1) * 1e3,
            "sequences": seqs,
        }'''
new = '''            "backend": f"rl_engine.models.muse_glimmer.reference triton backend, mode={mode}",
            "score_ms": (time.perf_counter() - t1) * 1e3,
            "commit": _git_head(),
            **_git_status(),
            "command": "python -m rl_engine.models.muse_glimmer.runtime.vllm_xrt_reference "
            + " ".join(args),
            "checkpoint": checkpoint,
            "sequences": seqs,
        }'''
assert old in s
s = s.replace(old, new, 1)
p.write_text(s)
print("ok")
