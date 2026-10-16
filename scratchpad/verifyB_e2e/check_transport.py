import json
import statistics
import sys
from collections import defaultdict

d = json.load(open(sys.argv[1]))
t = d["transport"]
rows = t["payloads"]
print("row keys", sorted(rows[0].keys()))
groups = defaultdict(list)
bit = defaultdict(list)
for r in rows:
    mode = r.get("tcp_nodelay", r.get("variant"))
    size = r.get("payload_bytes", r.get("bytes"))
    groups[(size, mode)].append(r.get("rtt_ms", r.get("rtt_s", 0) * 1e3 if r.get("rtt_s") is not None else None))
    bit[(size, mode)].append(r.get("bit_exact"))
for k in sorted(groups, key=lambda x: (x[0], str(x[1]))):
    v = groups[k]
    print(k, "n", len(v), "median", round(statistics.median(v), 1), "min", round(min(v), 1), "max", round(max(v), 1), "bit_exact all", all(bit[k]))
print("real_lora", json.dumps(t["real_lora_rank16"], default=str)[:1500])
print("bridge_equivalence names", [b.get("name") or b.get("variant") for b in d["bridge_equivalence"]])
for b in d["bridge_equivalence"]:
    if "torchrun" in json.dumps(b.get("name") or b.get("variant")):
        print({k: v for k, v in b.items() if k in ("name", "variant", "note", "all_bit_exact")})
