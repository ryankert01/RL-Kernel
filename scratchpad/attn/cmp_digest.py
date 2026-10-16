import json
import sys

old = json.load(open(sys.argv[1]))
new = json.load(open(sys.argv[2]))
print("old modules", old.get("modules"))
print("new modules", new.get("modules"))
out = {"forward": {}, "backward": {}}
for part in ("forward", "backward"):
    keys = sorted(set(old[part]) | set(new[part]))
    same = diff = 0
    for k in keys:
        a, b = old[part].get(k), new[part].get(k)
        if isinstance(a, list) and isinstance(b, list):
            names = ("y", "o", "lse") if part == "forward" else ("dq", "dk", "dv", "dgate")
            flags = {n: (x == y) for n, x, y in zip(names, a, b)}
            out[part][k] = flags
            if all(flags.values()):
                same += 1
            else:
                diff += 1
                print(part, k, flags)
        else:
            out[part][k] = {"old": a if not isinstance(a, list) else "ok",
                            "new": b if not isinstance(b, list) else "ok"}
            print(part, k, "old:", str(a)[:90], "new:", str(b)[:90])
    print(part, "cases identical", same, "differ", diff, "of", len(keys))
if len(sys.argv) > 3:
    json.dump(out, open(sys.argv[3], "w"), indent=1)
