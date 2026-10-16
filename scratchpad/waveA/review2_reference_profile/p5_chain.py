import json, sys
D = sys.argv[1]
d = json.load(open(f"{D}/reference_profile.json"))
E = d["experiments"]
for name in ("chain_triton_512_hf_pre_change", "chain_triton_512_hf", "chain_triton_2100_hf", "chain_golden_2100"):
    c = E[name]
    print("==", name, c["status"], c["backend"], c["prompt_len"], c["completion_len"], "all_eq", c["all_paths_byte_equal"], "drifts", c["drifts"])
    for p in c["paths"]:
        print(f"   {p['name']:24s} {p['mode']:8s} n_chunks={p['n_chunks']} {p['equal_tokens']}/{p['n_tokens']} eq={p['byte_equal_to_train']} {p['seconds']:.2f}s peak={p['peak_alloc_gib']:.2f}")
    h = c.get("hf_anchor")
    if h:
        print("   hf", h.get("hf"))
        print("   positions", h["positions"], "cs", h["completion_start"])
        for k in ("logits_all_positions", "logits_completion_positions", "logp_all", "logp_completion"):
            print("   ", k, {kk: vv for kk, vv in h[k].items() if not isinstance(vv, list)})
        print("    argmax all/comp", h["argmax_agreement_all"], h["argmax_agreement_completion"])
        pl = h.get("per_layer")
        if pl:
            print("    per_layer n", len(pl), "keys", list(pl[0].keys()))
            for r in pl:
                if r.get("layer") in (9, 19, 29, 39, 49) or r.get("name") in ("layer_9", "layer_49"):
                    print("     ", r)
            print("     first", pl[0]); print("     last", pl[-1])
        s = h.get("series")
        if s:
            print("    series lens", {k: len(v) for k, v in s.items()}, "max abs_dlogp", max(s["abs_dlogp"]), "max dlogit", max(s["max_abs_dlogit"]))
a = E["chain_triton_512_hf_pre_change"]["train_logp_series"]
b = E["chain_triton_512_hf"]["train_logp_series"]
print("pre/post train series identical:", a == b)
