"""CPU-side probe: entry point visible, strict install patches apply, op labels."""

import importlib.metadata as md
import os
import sys
import warnings

sys.path.insert(0, "/home/ubuntu/RL-Kernel/.claude/worktrees/agent-a7944ab684b3629cf")
warnings.filterwarnings("ignore")
os.environ["RL_KERNEL_MUSEGLIMMER_STRICT"] = "1"

print("entry points:", [(e.name, e.value) for e in md.entry_points(group="vllm.general_plugins")])
import rl_engine  # noqa: E402

print("rl_engine from", rl_engine.__file__)
from rl_engine.models.muse_glimmer.runtime.strict_ops import resolve_ops  # noqa: E402

ops = resolve_ops()
for name, label in ops.labels.items():
    print(f"  {name:36s} {label}")
from rl_engine.integrations.engines.rollout.vllm.runtime import register_vllm_plugin  # noqa: E402

register_vllm_plugin()
from rl_engine.models.muse_glimmer.runtime.vllm_strict import get_state  # noqa: E402

state = get_state()
print("installed hooks:")
for hook in state.hooks:
    print("  ", hook)
from vllm.v1.attention.backends.registry import AttentionBackendEnum  # noqa: E402

print("FLASH_ATTN class:", AttentionBackendEnum.FLASH_ATTN.get_class())
print("second register is a no-op:", register_vllm_plugin() is None and len(state.hooks))
