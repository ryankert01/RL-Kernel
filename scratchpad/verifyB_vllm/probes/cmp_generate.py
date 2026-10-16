import json
import sys

mine = json.load(open(sys.argv[1]))
data = json.load(open(sys.argv[2]))


def as_list(v):
    if isinstance(v, str):
        import ast

        return ast.literal_eval(v)
    return v


for g in mine["generate"]:
    t = g.get("temperature")
    lora = bool(g.get("lora"))
    refs = [
        e for e in data["generate"]
        if e["mode"] == "strict" and e["tp"] == 1 and e.get("temperature") == t
        and bool(e.get("lora")) == lora
    ]
    for ref in refs:
        mt, rt = as_list(g["token_ids"]), as_list(ref["token_ids"])
        mh, rh = as_list(g["logprobs_hex"]), as_list(ref["logprobs_hex"])
        same_tok = sum(1 for a, b in zip(mt, rt) if a == b)
        same_bits = sum(1 for a, b, c, d in zip(mt, rt, mh, rh) if a == b and c == d)
        print(f"T={t} lora={lora} mine(tp{mine['tp']}@{mine['commit'][:7]}) vs committed strict "
              f"tp1@{ref.get('commit', ref.get('run'))}: identical tokens {same_tok}/{len(rt)}, "
              f"tokens+bits {same_bits}/{len(rt)}")
