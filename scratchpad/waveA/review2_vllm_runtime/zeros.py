import json
import sys

d = json.load(open(sys.argv[1]))
for g in d["generate"]:
    lp = [x for r in g["logprobs"] for x in r]
    z = sum(1 for x in lp if x == 0.0)
    small = sum(1 for x in lp if abs(x) < 1e-4)
    big = sum(1 for x in lp if x < -0.1)
    print(g["mode"], g["tp"], "lora" if g["lora"] else ("native-lora-engine" if g["engine_lora_enabled"] else "base"),
          g["temperature"], "exact0", z, "|lp|<1e-4", small, "lp<-0.1", big, "of", len(lp))
