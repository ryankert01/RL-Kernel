import sys

sys.path.insert(0, "/home/ubuntu/RL-Kernel/.claude/worktrees/agent-a7944ab684b3629cf")
import warnings

warnings.filterwarnings("ignore")
import torch
import vllm

print("vllm", vllm.__version__, "torch", torch.__version__)
from vllm.v1.attention.backends.fa_utils import (
    is_fa_version_supported,
    is_flash_attn_varlen_func_available,
)

print("fa varlen available:", is_flash_attn_varlen_func_available())
for v in (2, 3, 4):
    try:
        print("fa%d supported:" % v, is_fa_version_supported(v))
    except Exception as e:  # noqa: BLE001
        print("fa%d check err:" % v, type(e).__name__, e)
from vllm.v1.attention.backends.registry import AttentionBackendEnum

print("CUSTOM in enum:", hasattr(AttentionBackendEnum, "CUSTOM"))
import importlib.metadata as md

print("entry points:", [(e.name, e.value) for e in md.entry_points(group="vllm.general_plugins")])
from rl_engine.models.muse_glimmer.loader import check_runtime_binding  # noqa: F401

print("loader ok")
from vllm.model_executor.models.registry import ModelRegistry

print("archs:", [a for a in ModelRegistry.get_supported_archs() if "Muse" in a])
import vllm.inputs as vi

print("TokensPrompt:", hasattr(vi, "TokensPrompt"))
from vllm.v1.kv_cache_interface import KVCacheLayout

print("KVCacheLayout members:", [m.name for m in KVCacheLayout])
from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend

print("FA layouts:", FlashAttentionBackend.supported_kv_cache_layouts())
