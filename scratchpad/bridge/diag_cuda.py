"""Who initializes CUDA in the orchestrator during a profiled mock run? (never lets it happen)"""
import sys
import traceback

import torch
import torch.cuda

calls = []


def _no_init(*args, **kwargs):
    calls.append("".join(traceback.format_stack(limit=25)))
    raise RuntimeError("diag: CUDA init blocked")


torch.cuda._lazy_init = _no_init
orig_is_init = torch.cuda.is_initialized


def _is_init():
    value = orig_is_init()
    if value:
        calls.append("is_initialized() returned True:\n" + "".join(traceback.format_stack(limit=8)))
    return value


torch.cuda.is_initialized = _is_init

from rl_engine.models.muse_glimmer.e2e import run_poc_grpo  # noqa: E402

code = run_poc_grpo.main([
    "--mock", "--steps", "2", "--prompts-per-step", "2", "--samples-per-prompt", "2",
    "--max-tokens", "8", "--lora-rank", "4", "--threads", "2", "--trainer-bridge", "on",
    "--profile-step", "1", "--mode", "strict/strict", "--run-dir", sys.argv[1],
])
print("exit", code, "cuda init attempts:", len(calls))
for c in calls[:3]:
    print(c)
