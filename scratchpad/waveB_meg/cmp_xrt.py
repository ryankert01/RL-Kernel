import json
import sys

a = json.load(open(sys.argv[1]))
b = json.load(open(sys.argv[2]))
mism = tot = 0
for sa, sb in zip(a["sequences"], b["sequences"], strict=True):
    assert sa["id"] == sb["id"]
    tot += len(sa["logprobs_hex"])
    mism += sum(x != y for x, y in zip(sa["logprobs_hex"], sb["logprobs_hex"], strict=True))
print(sys.argv[1].rsplit("/", 1)[-1], a.get("commit"), "vs", sys.argv[2].rsplit("/", 1)[-1],
      b.get("commit"), f"{mism}/{tot}")
