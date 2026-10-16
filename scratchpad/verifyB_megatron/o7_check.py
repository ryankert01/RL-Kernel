"""Are the tokens that changed between the wave A and wave B strict dumps only near-certain ones?"""
import json
import struct
import sys

a = json.load(open(sys.argv[1]))
b = json.load(open(sys.argv[2]))


def vals(j):
    out = []
    for s in j["sequences"]:
        out.append([struct.unpack(">f", bytes.fromhex(h))[0] for h in s["logprobs_hex"]])
    return out


va, vb = vals(a), vals(b)
changed = []
for sa, sb in zip(va, vb, strict=True):
    for x, y in zip(sa, sb, strict=True):
        if struct.pack(">f", x) != struct.pack(">f", y):
            changed.append((x, y))
print("changed", len(changed))
print("old max", max(c[0] for c in changed), "old min", min(c[0] for c in changed))
print("new max", max(c[1] for c in changed), "new min", min(c[1] for c in changed))
print("max |diff|", max(abs(c[0] - c[1]) for c in changed))
print("old > 0:", sum(c[0] > 0 for c in changed), "new > 0:", sum(c[1] > 0 for c in changed))
allv = [x for s in va for x in s]
print("old unchanged tokens with |lp|<1e-3:", sum(abs(x) < 1e-3 for x in allv) - sum(abs(c[0]) < 1e-3 for c in changed))
print("old changed tokens with |lp|>=1e-2:", sum(abs(c[0]) >= 1e-2 for c in changed))
