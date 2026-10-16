"""Reviewer probe 2 (HF): (a) does the 'rope_inv_freq_cuda' setting compute anything on CUDA?
(b) are the reported argmax flips at bf16 top-1/top-2 ties of the baseline logits?
argv: committed.json tokens.json out.json"""
import json
import os
import sys
import time

import torch

from rl_engine.models.muse_glimmer.ablation import ablation_hf as A
from rl_engine.validation.operators.accelerator import disable_tf32

committed = json.load(open(sys.argv[1]))
seqs, _ = A.load_xrt_tokens(sys.argv[2])
dev = torch.device("cuda:0")
disable_tf32("cuda")
res = {}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


from transformers import AutoConfig  # noqa: E402

cfg_text = AutoConfig.from_pretrained(os.path.expanduser("~/models/Muse-Glimmer-30B"),
                                      trust_remote_code=False).text_config
cpu = A.hf_rope_inv_freq(cfg_text)
via_harness = A.hf_rope_inv_freq(cfg_text, dev)
base = cfg_text.rope_parameters["rope_theta"]
on_cuda = 1.0 / (base ** (torch.arange(0, 128, 2, dtype=torch.float, device=dev) / 128))
res["harness_cuda_vs_cpu_bits_differ"] = int((via_harness.cpu().view(torch.int32)
                                              != cpu.view(torch.int32)).sum())
res["true_cuda_pow_vs_cpu_bits_differ"] = int((on_cuda.cpu().view(torch.int32)
                                               != cpu.view(torch.int32)).sum())
res["harness_inv_freq_device"] = str(via_harness.device)
log(f"inv_freq: {res}")

model, lm_head, cfg = A.load_text_model(os.path.expanduser("~/models/Muse-Glimmer-30B"), dev)
tie_pos = {}
with torch.no_grad():
    for s in seqs:
        h = A.hidden_alone(model, s)
        z = A.head_logits(h, lm_head, cfg, "bf16")
        top = z.topk(2, dim=-1).values
        tie_pos[s.id] = [int(i) for i in torch.nonzero(top[:, 0] == top[:, 1]).flatten()]
res["tie_positions"] = tie_pos
res["n_ties"] = sum(len(v) for v in tie_pos.values())
per_setting = {}
for r in committed["records"]:
    if r["engine"] != "hf" or not r.get("argmax_flip"):
        continue
    flips = [i for i, f in enumerate(r["argmax_flip"]) if f]
    at_tie = sum(1 for i in flips if i in set(tie_pos[r["sequence"]]))
    acc = per_setting.setdefault(r["setting"], [0, 0])
    acc[0] += len(flips)
    acc[1] += at_tie
res["flips_total_and_at_baseline_ties"] = per_setting
log(f"ties {res['n_ties']}; flips (total, at ties): {per_setting}")
json.dump(res, open(sys.argv[3], "w"), indent=1)
log("done")
