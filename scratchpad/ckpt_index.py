import json
import os

root = os.path.expanduser("~/models/Muse-Glimmer-30B")
idx = json.load(open(os.path.join(root, "model.safetensors.index.json")))
wm = idx["weight_map"]
for k in (
    "lm_head.weight",
    "model.language_model.norm.weight",
    "model.language_model.embed_tokens.weight",
):
    print(k, "->", wm.get(k))
print("REVISION:", open(os.path.join(root, "REVISION")).read().strip())
cfg = json.load(open(os.path.join(root, "config.json")))
tc = cfg.get("text_config", cfg)
keys = (
    "vocab_size",
    "hidden_size",
    "output_multiplier",
    "final_logit_softcapping",
    "output_soft_cap_temp",
)
print({k: tc.get(k) for k in keys})
print("top-level keys with 'soft' or 'mult':", [k for k in tc if "soft" in k or "mult" in k])
