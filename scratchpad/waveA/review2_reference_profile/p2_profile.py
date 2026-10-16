import json, sys
D = sys.argv[1]
d = json.load(open(f"{D}/reference_profile.json"))


def shape(o, depth=0, maxd=3, pre=""):
    if depth > maxd:
        return
    if isinstance(o, dict):
        for k, v in o.items():
            t = type(v).__name__
            extra = ""
            if isinstance(v, list):
                extra = f" len={len(v)}"
            elif not isinstance(v, dict):
                extra = f" = {str(v)[:120]}"
            print(f"{pre}{k}: {t}{extra}")
            if isinstance(v, dict):
                shape(v, depth + 1, maxd, pre + "  ")
            elif isinstance(v, list) and v and isinstance(v[0], dict) and depth < maxd:
                print(f"{pre}  [0]:")
                shape(v[0], depth + 2, maxd, pre + "    ")


shape(d, maxd=int(sys.argv[2]) if len(sys.argv) > 2 else 3)
