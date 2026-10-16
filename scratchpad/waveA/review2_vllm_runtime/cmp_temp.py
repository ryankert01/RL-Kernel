"""Are the T=0 and T=1 generations different? (read-only)"""
import json
import math
import sys

d = json.load(open(sys.argv[1]))
G = d["generate"]
for mode in ("native", "strict"):
    for tp in (1, 2):
        for lora in (False, True):
            gs = [g for g in G if g["mode"] == mode and g["tp"] == tp and bool(g["lora"]) == lora
                  and not (mode == "native" and g["engine_lora_enabled"] and not g["lora"])]
            t0 = [g for g in gs if g["temperature"] == 0.0]
            t1 = [g for g in gs if g["temperature"] == 1.0]
            if not t0 or not t1:
                continue
            a, b = t0[0], t1[0]
            same = a["token_ids"] == b["token_ids"]
            ndiff = sum(x != y for r1, r2 in zip(a["token_ids"], b["token_ids"]) for x, y in zip(r1, r2))
            first = [next((i for i, (x, y) in enumerate(zip(r1, r2)) if x != y), None)
                     for r1, r2 in zip(a["token_ids"], b["token_ids"])]
            # mean logprob of sampled tokens
            m0 = sum(x for r in a["logprobs"] for x in r) / 512
            m1 = sum(x for r in b["logprobs"] for x in r) / 512
            print(mode, tp, "lora" if lora else "base", "T0==T1 tokens:", same, "diff tokens", ndiff,
                  "first div", first, "mean lp T0 %.4f T1 %.4f" % (m0, m1))
            print("   T0 text0:", repr(a.get("texts", [""])[0][:200]))
            print("   T1 text0:", repr(b.get("texts", [""])[0][:200]))
