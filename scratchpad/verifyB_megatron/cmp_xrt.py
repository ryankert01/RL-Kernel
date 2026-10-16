"""Bitwise compare xrt logprob dumps (hex fields) and print provenance."""
import hashlib
import json
import sys
from pathlib import Path

d = Path(sys.argv[1])
names = sys.argv[2:]
data = {}
for n in names:
    p = d / n
    raw = p.read_bytes()
    j = json.loads(raw)
    data[n] = j
    meta = {k: j.get(k) for k in ("commit", "dirty", "runtime", "mode", "tp", "backend", "schema")}
    print(n, hashlib.sha256(raw).hexdigest()[:16], meta)


def hexes(j):
    seqs = j.get("sequences") or j.get("results") or j.get("samples")
    out = []
    for s in seqs:
        h = s.get("logprobs_hex") or s.get("hex")
        out.append(h)
    return out


base = names[0]
hb = hexes(data[base])
for n in names[1:]:
    ho = hexes(data[n])
    mism = 0
    tot = 0
    for a, b in zip(hb, ho, strict=True):
        assert len(a) == len(b)
        tot += len(a)
        mism += sum(x != y for x, y in zip(a, b))
    print(f"{n} vs {base}: {mism}/{tot} mismatches")
