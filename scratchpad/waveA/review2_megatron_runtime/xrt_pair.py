import json
import sys

a = json.load(open(sys.argv[1]))
b = json.load(open(sys.argv[2]))
tot = mm = 0
for sa, sb in zip(a["sequences"], b["sequences"], strict=True):
    assert sa["id"] == sb["id"]
    m = sum(1 for x, y in zip(sa["logprobs_hex"], sb["logprobs_hex"], strict=True) if x != y)
    tot += len(sa["logprobs_hex"])
    mm += m
print(sys.argv[1].rsplit("/", 1)[-1], "vs", sys.argv[2].rsplit("/", 1)[-1], f"{mm}/{tot} bit mismatches",
      "commits", a.get("commit"), b.get("commit"))
