import json
import sys

d = json.load(open(sys.argv[1]))
sec = sys.argv[2]


def short(v):
    s = json.dumps(v)
    return s if len(s) < 160 else s[:160] + "..."


if sec == "equiv":
    for b in d["bridge_equivalence"]:
        print(b["name"], b["server_python"], b["server_torch"], b["launcher"], b["tp"],
              b["all_bit_exact"], b.get("last_export_rank_sha256"))
        for s in b["steps"]:
            print("   ", {k: short(v) for k, v in s.items()})
    print(json.dumps(d["bridge_equivalence"][0]["calls"][:2], indent=1)[:2500])
else:
    obj = d
    for part in sec.split("."):
        obj = obj[int(part)] if isinstance(obj, list) else obj[part]
    print(json.dumps(obj, indent=1)[: int(sys.argv[3]) if len(sys.argv) > 3 else 6000])
