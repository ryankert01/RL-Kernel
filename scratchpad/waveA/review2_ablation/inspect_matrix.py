import json
import sys

d = json.load(open(sys.argv[1]))
rows = {}
seq_ids = []
for r in d["records"]:
    if r["sequence"] not in seq_ids:
        seq_ids.append(r["sequence"])
    rows.setdefault((r["engine"], r["setting"]), {})[r["sequence"]] = r
print(" " * 44 + " ".join(s[-6:].rjust(9) for s in seq_ids))
for (e, n), per in rows.items():
    if e != "hf":
        continue
    print(f"{e}/{n}"[:43].ljust(44) + " ".join(
        f"{per[s]['n_bits_differ']}/{per[s]['n_tokens']}".rjust(9) for s in seq_ids))
