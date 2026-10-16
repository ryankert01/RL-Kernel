import torch

from rl_engine.models.muse_glimmer.runtime.strict_ops import probe_ops, resolve_ops
from rl_engine.models.muse_glimmer.runtime.strict_kernels import resolve_ops as mk

ops = probe_ops(resolve_ops(), torch.device("cuda"))
print("vllm strict_ops labels:", ops.labels["rope_table"], ops.labels["qknorm_scaled_rope"])
print("demotions:", [d for d in ops.demotions if "rope" in d.get("name", "") or "rope" in str(d)])
try:
    m = mk()
    print("megatron strict_kernels sources:", m.sources.get("rope_table"), m.sources.get("qknorm_scaled_rope"))
except Exception as e:  # noqa: BLE001
    print("strict_kernels.strict_ops() call failed:", type(e).__name__, e)
