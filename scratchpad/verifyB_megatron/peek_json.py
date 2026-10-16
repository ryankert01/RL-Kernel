"""Summarize megatron_runtime.json structure and key claims."""
import json
import sys


def short(v, n=300):
    s = json.dumps(v)
    return s if len(s) <= n else s[:n] + "..."


d = json.load(open(sys.argv[1]))
print("top keys:", list(d.keys()))
for k, v in d.items():
    if isinstance(v, dict):
        print(f"== {k}: keys {list(v.keys())[:40]}")
    else:
        print(f"== {k}: {short(v)}")
path = sys.argv[2:] if len(sys.argv) > 2 else []
node = d
for p in path:
    node = node[p] if not isinstance(node, list) else node[int(p)]
if path:
    if isinstance(node, dict):
        for k, v in node.items():
            print(f"  -- {k}: {short(v, 600)}")
    else:
        print(short(node, 4000))
