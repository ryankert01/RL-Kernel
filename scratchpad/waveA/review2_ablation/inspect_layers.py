import json
import sys

d = json.load(open(sys.argv[1]))


def f(L, p):
    x = L[p]
    return "%.3f/%.2e/rows%d/first%s" % (x["frac_elems_differ"], x["mean_abs"],
                                         x["rows_with_any_diff"], x["first_row_with_diff"])


for r in d["layer_records"]:
    L = {x["point"]: x for x in r["layers"]}
    print(r["sequence"][:12], r["setting"][:26].ljust(26), "emb", f(L, "embedding"), "L0",
          f(L, "layer_0"), "L1", f(L, "layer_1"), "L12", f(L, "layer_12"), "L51",
          f(L, "layer_51"), "fin", f(L, "final_norm"), "rms51", round(L["layer_51"]["ref_rms"], 2),
          "rmsfin", round(L["final_norm"]["ref_rms"], 3))
