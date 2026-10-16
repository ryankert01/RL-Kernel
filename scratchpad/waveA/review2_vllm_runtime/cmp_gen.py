"""Check generate records in vllm_runtime.json (read-only)."""
import json
import struct
import sys

d = json.load(open(sys.argv[1]))
G = d["generate"]


def key(g):
    lora = g["lora"]
    lsha = lora if isinstance(lora, str) else (lora.get("sha256") if isinstance(lora, dict) else None)
    return (g["mode"], g["tp"], bool(lsha), g["engine_lora_enabled"], g["temperature"])


def bits(v):
    return struct.pack(">f", float(v)).hex()


by = {}
for g in G:
    by.setdefault(key(g), []).append(g)
for k, v in by.items():
    print(k, len(v), "lora field type", type(v[0]["lora"]).__name__)

pos_total = 0
cnt_total = 0
for g in G:
    if g["mode"] != "strict":
        continue
    lp = [x for row in g["logprobs"] for x in row]
    pos = [x for x in lp if x > 0]
    print("strict", g["tp"], "lora" if g["lora"] else "base", g["temperature"], "n=", len(lp), "positive", len(pos), max(pos) if pos else None)
    if g["tp"] == 1 and not g["lora"]:
        pos_total += len(pos)
        cnt_total += len(lp)
print("strict TP1 base positive", pos_total, "of", cnt_total)


def find(mode, tp, lora, temp):
    for g in G:
        if g["mode"] == mode and g["tp"] == tp and bool(g["lora"]) == lora and g["temperature"] == temp and not (mode == "native" and g["engine_lora_enabled"]):
            return g
    return None


for lora in (False, True):
    for temp in (0.0, 1.0):
        a = find("strict", 1, lora, temp)
        b = find("strict", 2, lora, temp)
        if not a or not b:
            print("missing", lora, temp)
            continue
        same_tok = a["token_ids"] == b["token_ids"]
        same_bits = [[bits(x) for x in r] for r in a["logprobs"]] == [[bits(x) for x in r] for r in b["logprobs"]]
        print("strict tp1 vs tp2 lora=%s T=%s tokens_equal=%s logprob_bits_equal=%s" % (lora, temp, same_tok, same_bits))
for temp in (0.0, 1.0):
    a = find("native", 1, False, temp)
    b = find("native", 2, False, temp)
    if a and b:
        ntok = sum(x != y for r1, r2 in zip(a["token_ids"], b["token_ids"]) for x, y in zip(r1, r2))
        first_div = []
        for r1, r2 in zip(a["token_ids"], b["token_ids"]):
            idx = next((i for i, (x, y) in enumerate(zip(r1, r2)) if x != y), None)
            first_div.append(idx)
        print("native tp1 vs tp2 T=%s token diffs=%d first divergence per seq=%s" % (temp, ntok, first_div))
    s = find("strict", 1, False, temp)
    if a and s:
        first_div = []
        for r1, r2 in zip(a["token_ids"], s["token_ids"]):
            idx = next((i for i, (x, y) in enumerate(zip(r1, r2)) if x != y), None)
            first_div.append(idx)
        print("native tp1 vs strict tp1 T=%s first divergence per seq=%s" % (temp, first_div))

# finish reasons & lengths
for g in G:
    fr = g.get("finish_reasons")
    lens = [len(r) for r in g["token_ids"]]
    print(g["mode"], g["tp"], bool(g["lora"]), g["temperature"], "lens", lens, "finish", fr)
# decode_vs_prefill structure
dv = G[0]["decode_vs_prefill"]
print(json.dumps({k: (v if not isinstance(v, list) else (v[:1])) for k, v in dv.items()})[:1500])
