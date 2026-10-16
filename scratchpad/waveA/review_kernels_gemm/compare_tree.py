import json
import os

R = os.path.dirname(os.path.abspath(__file__)) + "/"
b = json.load(open(R + "probe_tree_base.json"))
h = json.load(open(R + "probe_tree_head.json"))
print("module", h["module"])
for rb, rh in zip(b["rows"], h["rows"]):
    key = {k: rb[k] for k in ("kind", "K", "M") if k in rb}
    print(key, "numel", rb["numel"], "| base vs_tree", rb["mismatch_vs_tree"], "vs_folded",
          rb["mismatch_vs_folded"], "bk", rb.get("block_k"), "| head vs_tree",
          rh["mismatch_vs_tree"], "vs_folded", rh["mismatch_vs_folded"], "bk", rh.get("block_k"))
print("IR base", b["ir_K512_M256"])
print("IR head", h["ir_K512_M256"])
same = sum(b["fwd_hashes"][k] == h["fwd_hashes"][k] for k in b["fwd_hashes"])
print("fwd hashes equal:", same, "/", len(b["fwd_hashes"]))
