"""Extract the JSON printed by vllm_tp_sim from its lease log (usage: extract_sim.py log out commit)."""

import json
import sys

log, out, commit = sys.argv[1:4]
lines = open(log).read().splitlines()
start = next(i for i, line in enumerate(lines) if line == "{")
end = max(i for i, line in enumerate(lines) if line == "}")
doc = json.loads("\n".join(lines[start : end + 1]))
doc["command"] = (
    "~/bin/gpu-lease -n 1 --pool 4-6 -- env PYTHONPATH=$WT ~/envs/vllm/bin/python -m "
    "rl_engine.models.muse_glimmer.runtime.vllm_tp_sim"
)
doc["commit"] = commit
doc["what"] = (
    "binding per-rank code (strict_attention_forward on stand-in modules with the rank's heads, "
    "strict_mlp_forward, strict_logits, strict_selected_logp) of one real-shape decoder layer "
    "(local and global) + padded lm_head at simulated TP 2/4/8 (threads + in-process rank-ordered "
    "all-gather on ONE GPU), with and without a rank-16 LoRA; torch.equal vs TP1"
)
json.dump(doc, open(out, "w"), indent=1)
print(doc["all_equal"], [(e["tp"], e["lora"], e["layer"][:6], e["all_equal"]) for e in doc["results"]])
