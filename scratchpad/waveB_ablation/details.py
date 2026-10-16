"""Cross-run repeats and layer-0 detail points from a merged ablation.json (argv[1])."""
import json
import sys

d = json.load(open(sys.argv[1]))
for p in d.get("previous_runs", []):
    print("previous", p["path"].split("/")[-1], p["commit"][:8] if p["commit"] else None,
          {k: (None if v is None else f"{v['tokens_differ']}/{v['n_tokens']}")
           for k, v in p.items() if k in ("hf", "strict")})
pts = ["rope_cos", "rope_sin", "layer_0.input_layernorm", "layer_0.self_attn.q_proj",
       "layer_0.self_attn.k_proj", "layer_0.self_attn.v_proj", "layer_0.self_attn.gate_proj",
       "layer_0.self_attn.o_proj.input", "layer_0.self_attn.o_proj", "layer_0.mlp.gate_proj",
       "layer_0.mlp.down_proj"]
for r in d["layer_records"]:
    allp = {x["point"]: x for x in r["layers"] + r.get("detail_points", [])}
    s = " ".join(
        f"{p.replace('layer_0.', '').replace('self_attn.', '')}={allp[p]['frac_elems_differ']:.2g}"
        f"/{allp[p]['rows_with_any_diff']}r@{allp[p]['first_row_with_diff']}"
        for p in pts if p in allp and allp[p]["rows_with_any_diff"])
    l0 = allp["layer_0"]
    print(f"{r['sequence'][:10]} {r['setting'][:22]:22s} first={r['first_point_with_diff']} | {s} | "
          f"L0={l0['frac_elems_differ']:.3g}/{l0['rows_with_any_diff']}r")
