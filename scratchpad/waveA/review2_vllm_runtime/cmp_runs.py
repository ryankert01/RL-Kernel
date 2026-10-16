import json
import struct
import sys

a = json.load(open(sys.argv[1]))
b = json.load(open(sys.argv[2]))
ga, gb = a["generate"][0], b["generate"][0]
print("A", a["mode"], a["tp"], a.get("status"), "B", b["mode"], b["tp"], b.get("status"))
print("B decode_vs_prefill mismatched", gb["decode_vs_prefill"]["mismatched"], "of", gb["decode_vs_prefill"]["tokens"])
assert ga["ids"] == gb["ids"]
tok_seq_diff = 0
first_div = []
bits_equal_prefix = 0
bits_total_prefix = 0
for ta, tb, la, lb in zip(ga["token_ids"], gb["token_ids"], ga["logprobs"], gb["logprobs"]):
    d = next((i for i, (x, y) in enumerate(zip(ta, tb)) if x != y), None)
    if d is not None:
        tok_seq_diff += 1
        first_div.append(d)
    end = len(ta) if d is None else d + 1  # logprob of the diverging token is of a different token
    end = len(ta) if d is None else d
    for x, y in zip(la[:end], lb[:end]):
        bits_total_prefix += 1
        bits_equal_prefix += struct.pack(">f", x) == struct.pack(">f", y)
print("sequences with different tokens", tok_seq_diff, "of", len(ga["token_ids"]), "first divergence idx", sorted(first_div)[:20])
print("logprob bits equal on common prefix", bits_equal_prefix, "of", bits_total_prefix)
w = (b.get("meta") or {}).get("workers", [])
for x in w:
    print("worker", x.get("tp_rank"), x.get("all_ops_strict"), x.get("backend_ids", {}).get("collective"), x.get("kv_ownership", {}).get("kv_heads_cached"))
