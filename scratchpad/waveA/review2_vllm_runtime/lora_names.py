"""How does vLLM 0.31 map the PEFT names written by vllm_native_lora? (CPU only, no model)."""
from vllm.lora.utils import parse_fine_tuned_lora_name
from vllm.model_executor.models.muse_glimmer import MuseGlimmerForCausalLM as M

from rl_engine.models.muse_glimmer.e2e.sync import LORA_PROJECTIONS
from rl_engine.models.muse_glimmer.runtime.vllm_native_lora import peft_key

mapper = M.hf_to_vllm_mapper.get_rename_mapper()
print("packed_modules_mapping", M.packed_modules_mapping)
seen = {}
for proj in LORA_PROJECTIONS:
    for mat in ("A", "B"):
        key = peft_key(f"layers.0.{proj}.lora_{mat}")
        name, is_a = parse_fine_tuned_lora_name(key, mapper)
        print(f"{proj:14s} {mat}: {key} -> {name} (lora_a={is_a})")
        seen.setdefault((name, is_a), []).append(proj)
coll = {k: v for k, v in seen.items() if len(v) > 1}
print("COLLISIONS:", coll)
