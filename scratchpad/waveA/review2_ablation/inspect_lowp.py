"""Where do the low-probability gold tokens sit?  (tokenizer.json read as data via `tokenizers`)."""
import json
import math
import os
import sys

from tokenizers import Tokenizer

d = json.load(open(sys.argv[1]))
tok = Tokenizer.from_file(os.path.expanduser("~/models/Muse-Glimmer-30B/tokenizer.json"))
xrt = json.load(open(sys.argv[2]))
ids_by = {s["id"]: s["token_ids"] for s in xrt["sequences"]}
base = d["baselines"]["hf"]["per_sequence"]
n_low = n_in_calc = n_very_low = 0
examples = []
for p in base:
    ids = ids_by[p["sequence"]]
    cs = p["completion_start"]
    text_so_far = ""
    for i, (lp, t) in enumerate(zip(p["logp"], p["targets"])):
        prefix = tok.decode(ids[cs:cs + i])
        # inside a <<...>> span if the last '<<' after the completion start is not yet closed
        inside = prefix.rfind("<<") > prefix.rfind(">>")
        piece = tok.decode([t])
        is_open = piece.strip().startswith("<<") or piece.strip() == "<"
        if math.exp(lp) <= 0.1:
            n_low += 1
            if inside or "<<" in piece:
                n_in_calc += 1
            elif len(examples) < 25:
                examples.append((p["sequence"], i, round(lp, 3), repr(piece)))
        if lp < -20:
            n_very_low += 1
print("p<=0.1 tokens:", n_low, "inside or opening <<calc>>:", n_in_calc, "logp<-20:", n_very_low)
print("low-p tokens outside <<calc>> spans (first 25):")
for e in examples:
    print("  ", e)
# print long-window completion text sample
lw = ids_by["gsm8k_long_window"]
print(repr(tok.decode(lw[2261:2261 + 60])))
print(repr(tok.decode(ids_by["gsm8k_test_0"][136:191])))
