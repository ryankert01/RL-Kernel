"""Robust version: character spans of every completion token vs <<...>> spans."""
import json
import math
import os
import re
import sys

from tokenizers import Tokenizer

d = json.load(open(sys.argv[1]))
tok = Tokenizer.from_file(os.path.expanduser("~/models/Muse-Glimmer-30B/tokenizer.json"))
xrt = json.load(open(sys.argv[2]))
ids_by = {s["id"]: s["token_ids"] for s in xrt["sequences"]}
base = d["baselines"]["hf"]["per_sequence"]
tot = {"low": 0, "low_calc": 0, "all": 0, "all_calc": 0, "low_first_tok": 0}
for p in base:
    ids = ids_by[p["sequence"]]
    cs = p["completion_start"]
    comp = ids[cs:]
    # cumulative decode to get char spans (decode of prefixes)
    spans = []
    prev = ""
    for i in range(len(comp)):
        cur = tok.decode(comp[: i + 1])
        spans.append((len(prev), len(cur)))
        prev = cur
    text = prev
    calc = [(m.start(), m.end()) for m in re.finditer(r"<<.*?>>", text)]
    for i, (lp, (a, b)) in enumerate(zip(p["logp"], spans)):
        in_calc = any(a < ce and b > cs_ for cs_, ce in calc)
        tot["all"] += 1
        tot["all_calc"] += in_calc
        if math.exp(lp) <= 0.1:
            tot["low"] += 1
            tot["low_calc"] += in_calc
            tot["low_first_tok"] += i == 0
print(tot)
