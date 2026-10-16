"""Reviewer probe 3 (HF): the factor 'inv_freq computed on CUDA' done for real (pow on the GPU),
scored with the baseline driver.  argv: committed.json tokens.json out.json"""
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
model, lm_head, cfg = A.load_text_model(os.path.expanduser("~/models/Muse-Glimmer-30B"), dev)
theta = cfg.rope_parameters["rope_theta"]
dim = cfg.head_dim
cuda_inv = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float, device=dev) / dim))
rot = model.rotary_emb
saved = (rot.inv_freq, rot.original_inv_freq)
res = {"inv_freq_bits_differ": int((cuda_inv.view(torch.int32)
                                    != saved[0].view(torch.int32)).sum())}
A._set_rope_buffers(model, cuda_inv)
var, _ = A.score_all(model, lm_head, cfg, seqs, driver="alone")
rot.inv_freq, rot.original_inv_freq = saved
cb = [p["logp_hex"] for p in committed["baselines"]["hf"]["per_sequence"]]
per = [sum(x != y for x, y in zip(A.fp32_hex(v.logp), c)) for v, c in zip(var, cb)]
base_lp = [torch.tensor(p["logp"]) for p in committed["baselines"]["hf"]["per_sequence"]]
d = torch.cat([(v.logp.double() - b.double()).abs() for v, b in zip(var, base_lp)])
res.update({"tokens_differ_per_seq": per, "tokens_differ": sum(per),
            "max_abs_dlogp": float(d.max()), "mean_abs_dlogp": float(d.mean())})
print(f"[{time.strftime('%H:%M:%S')}] {res}", flush=True)
json.dump(res, open(sys.argv[3], "w"), indent=1)
