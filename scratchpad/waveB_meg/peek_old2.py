import json
import sys

d = json.load(open(sys.argv[1]))
for tp in ("tp1", "tp2"):
    c = d["runs"][tp]["checks"]
    print(tp, "grad_repeat passes", [p["info"]["fwd_bwd_s"] for p in c["grad_repeat"]["passes"]],
          [p["fwd_s"] for p in c["grad_repeat"]["passes"]])
    print(tp, "step", c["step"]["info"].get("fwd_bwd_s"), c["step"]["info"].get("grad_norm"))
    print(tp, "xrt seconds", c["xrt_strict"]["seconds"], "native", c["native"]["seconds"],
          c["native"]["max_abs_vs_strict"], c["native"]["mean_abs_vs_strict"])
    if "identity" in c:
        print(tp, "identity", c["identity"]["equal"], c["identity"]["total"], c["identity"].get("seconds"))
