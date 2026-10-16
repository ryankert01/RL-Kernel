import json, sys, struct
D = sys.argv[1]
for f in ["xrt_reference_golden.json", "xrt_reference_triton.json", "xrt_reference_hf_bf16.json"]:
    d = json.load(open(f"{D}/{f}"))
    print(f, {k: (v if not isinstance(v, (list, dict)) else type(v).__name__) for k, v in d.items()})
    print("  commit", d.get("commit"), "source", d.get("source"))
    if "checks" in d:
        print("  modes", d["checks"]["modes"], d["checks"]["all_byte_equal_to_train"])
        for r in d["checks"]["sequences"]:
            print("   ", r["id"], r["length"], r["completion_start"], r["n_tokens"],
                  {m: (v.get("equal_tokens"), v.get("n_tokens")) if "skipped" not in v else "skip"
                   for m, v in r["modes"].items()})
    print("  n seqs", len(d["sequences"]), sum(s["n_tokens"] for s in d["sequences"]))
    # consistency: hex vs float list
    bad = 0
    for s in d["sequences"]:
        for h, x in zip(s["logprobs_hex"], s["logprobs"]):
            v = struct.unpack("<f", struct.pack("<I", int(h, 16)))[0]
            if not (v == x):
                bad += 1
    print("  hex/float mismatches", bad)

tok = json.load(open(f"{D}/xrt_tokens.json"))
print("tokens schema", tok.get("schema"), [ (s["id"], len(s["token_ids"]), s["completion_start"]) for s in tok["sequences"]])
