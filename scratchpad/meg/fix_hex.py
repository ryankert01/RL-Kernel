"""Rewrite signed-int32 hex strings ('-3df7d49e') as the uint32 bit pattern ('c2082b62').

The transformation is exact (two's complement of the same 32 bits). It also cross-checks every
rewritten string against the stored fp32 value when a parallel 'logprobs' list exists.
"""
import json
import struct
import sys


def fix(h):
    if isinstance(h, str) and h.startswith("-"):
        return "%08x" % (int(h, 16) & 0xFFFFFFFF)
    return h


def walk(obj, stats):
    if isinstance(obj, dict):
        if "logprobs_hex" in obj and "logprobs" in obj:
            for v, h in zip(obj["logprobs"], obj["logprobs_hex"], strict=True):
                want = "%08x" % struct.unpack("<I", struct.pack("<f", v))[0]
                assert fix(h) == want, (h, v)
                stats["checked"] += 1
        return {k: walk(v, stats) for k, v in obj.items()}
    if isinstance(obj, list):
        return [walk(v, stats) for v in obj]
    if isinstance(obj, str) and obj.startswith("-") and len(obj) <= 9:
        try:
            out = fix(obj)
            stats["fixed"] += 1
            return out
        except ValueError:
            return obj
    return obj


for path in sys.argv[1:]:
    stats = {"fixed": 0, "checked": 0}
    data = json.load(open(path))
    data = walk(data, stats)
    with open(path, "w") as fh:
        json.dump(data, fh, indent=1 if path.endswith("run_tp1_part1.json") or "run_tp" in path else None)
        fh.write("\n")
    print(path, stats)
