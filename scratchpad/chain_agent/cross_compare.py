"""Four-way anchor: strict(golden) / strict(triton) / HF bf16 / HF fp32 on the same 544 tokens."""

import json
import sys

import torch

from rl_engine.models.muse_glimmer.reference.chain_harness import _tensor_diff

runs = sys.argv[1]
triton_dir = sys.argv[2] if len(sys.argv) > 2 else "triton_512_32"
out_name = sys.argv[3] if len(sys.argv) > 3 else "cross_compare.json"
golden = torch.load(f"{runs}/main_512_32/strict_ref.pt")
hf16 = torch.load(f"{runs}/main_512_32/hf_bf16.pt")
hf32 = torch.load(f"{runs}/hf_fp32_512_32.pt")
try:
    triton = torch.load(f"{runs}/{triton_dir}/strict_ref.pt")
except FileNotFoundError:
    triton = None
assert torch.equal(golden["token_ids"], hf32["token_ids"]) and torch.equal(
    golden["token_ids"], hf16["token_ids"]
)
ids = golden["token_ids"]
t = ids.numel()
lp_start = 511  # logits positions predicting completion tokens: 511..542
V = 202048


def logp_of(logits):
    return torch.log_softmax(logits.double(), -1)[torch.arange(t - 1), ids[1:]].float()


cands = {"strict_golden": golden["logits"][:, :V], "hf_bf16": hf16["logits"][:, :V]}
if triton is not None:
    cands["strict_triton"] = triton["logits"][:, :V]
ref = hf32["logits"][:, :V]
ref_lp = logp_of(ref)
out = {}
for name, lg in cands.items():
    comp = slice(lp_start, t - 1)
    lp = logp_of(lg) if name != "strict_golden" else golden["logp"]
    if name == "strict_triton":
        lp = triton["logp"]
    out[name + "_vs_hf_fp32"] = {
        "logits_all": _tensor_diff(lg, ref),
        "logits_completion": _tensor_diff(lg[comp], ref[comp]),
        "logp_completion": _tensor_diff(lp[comp], ref_lp[comp]),
        "argmax_agree_all": float((lg.argmax(-1) == ref.argmax(-1)).float().mean()),
    }
if triton is not None:
    g, tr = cands["strict_golden"], cands["strict_triton"]
    out["strict_golden_vs_strict_triton"] = {
        "logits_all": _tensor_diff(tr, g),
        "logp_completion": _tensor_diff(triton["logp"][lp_start:], golden["logp"][lp_start:]),
        "argmax_agree_all": float((g.argmax(-1) == tr.argmax(-1)).float().mean()),
    }
print(json.dumps(out, indent=1))

# per-layer residual stream vs fp32 HF (hs[0] = embedding, hs[i+1] = layer i output)
hs32 = [h.float() for h in hf32["hidden_states"]]
hs16 = [h.float() for h in hf16["hidden_states"]]
print("\nlayer | golden-vs-fp32 mean|Δ| max|Δ| | hf_bf16-vs-fp32 mean|Δ| max|Δ| | "
      + ("triton-vs-fp32 mean|Δ| max|Δ| | " if triton else "") + "rms(fp32)")
rows = [("emb", golden["embedding"], hs16[0], hs32[0], None if triton is None else triton["embedding"])]
for i in range(51):
    rows.append((str(i), golden["residuals"][i], hs16[i + 1], hs32[i + 1],
                 None if triton is None else triton["residuals"][i]))
rows.append(("final", golden["final_hidden"], hs16[-1], hs32[-1],
             None if triton is None else triton["final_hidden"]))
table = []
for name, g, h16, h32, tr in rows:
    dg = _tensor_diff(g, h32)
    dh = _tensor_diff(h16, h32)
    line = f"{name:>5} | {dg['mean_abs']:.3e} {dg['max_abs']:.2e} | {dh['mean_abs']:.3e} {dh['max_abs']:.2e} |"
    rec = {"layer": name, "golden_vs_fp32": dg, "hf_bf16_vs_fp32": dh, "rms_fp32": dg["ref_rms"]}
    if tr is not None:
        dt = _tensor_diff(tr, h32)
        line += f" {dt['mean_abs']:.3e} {dt['max_abs']:.2e} |"
        rec["triton_vs_fp32"] = dt
    line += f" {dg['ref_rms']:.3e}"
    print(line)
    table.append(rec)
json.dump({"summary": out, "per_layer": table}, open(f"{runs}/{out_name}", "w"), indent=1)
