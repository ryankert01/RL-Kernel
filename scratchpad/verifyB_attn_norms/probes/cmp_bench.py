"""Paired base/new forward bench rows: strict median ms and ratios (from committed raw files)."""
import json
import sys

a = json.load(open(sys.argv[1]))
b = json.load(open(sys.argv[2]))
for sec, key in (("prefill_forward", "T"), ("decode_forward", "B")):
    for ra, rb in zip(a[sec], b[sec]):
        assert ra[key] == rb[key] and ra["window"] == rb["window"]
        print(f"{sec:16s} {key}={ra[key]:5d} win={str(ra['window']):5s} base={ra['strict_ms']:.4f} "
              f"new={rb['strict_ms']:.4f} new/base={rb['strict_ms'] / ra['strict_ms']:.3f} "
              f"speedup={ra['strict_ms'] / rb['strict_ms']:.2f} sdpa_ratio={rb.get('ratio')} "
              f"flash_ratio={rb.get('flash_ratio')}")
