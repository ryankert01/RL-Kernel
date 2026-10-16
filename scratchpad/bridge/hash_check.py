import torch

from rl_engine.models.muse_glimmer.e2e import sync
from rl_engine.models.muse_glimmer.runtime import megatron_lora

sd = {
    "layers.0.q_proj.lora_A": torch.zeros(1, 2, dtype=torch.bfloat16),
    "layers.0.q_proj.lora_B": torch.ones(3, 1, dtype=torch.bfloat16),
}
a = sync.lora_sha256(sd)
b = megatron_lora.lora_sha256(sd)
print("e2e/sync (muse_glimmer_lora_sha256_v1):", a)
print("runtime/megatron_lora               :", b)
print("equal:", a == b)
