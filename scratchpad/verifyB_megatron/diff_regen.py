"""Compare the regenerated megatron_runtime.json with the committed one, section by section."""
import json
import sys

a = json.load(open(sys.argv[1]))
b = json.load(open(sys.argv[2]))
for key in sorted(set(a) | set(b)):
    if key not in a or key not in b:
        print(key, "only in", "committed" if key in a else "regen")
        continue
    same = json.dumps(a[key], sort_keys=True) == json.dumps(b[key], sort_keys=True)
    print(key, "EQUAL" if same else "DIFFERS")
    if not same and isinstance(a[key], dict):
        for k2 in sorted(set(a[key]) | set(b[key])):
            s2 = json.dumps(a[key].get(k2), sort_keys=True) == json.dumps(b[key].get(k2), sort_keys=True)
            if not s2:
                print("   ", k2, "differs")
