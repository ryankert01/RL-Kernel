import json
import sys

runs = sys.argv[1]
triton_name = sys.argv[2] if len(sys.argv) > 2 else "triton_512_32"
m = json.load(open(f"{runs}/main_512_32.json"))
hf = m["hf_compare"]
print("golden HF anchor:", json.dumps({k: v for k, v in hf.items() if k != "per_layer"}, indent=1))
print("\nper-layer (golden vs HF bf16), every 5th layer + extremes:")
for r in hf["per_layer"]:
    L = r["layer"]
    if L in ("embedding", "final_norm") or (isinstance(L, int) and (L % 5 == 0 or L in (49, 50))):
        print(f"{str(L):>10} max|Δ|={r['max_abs']:.3e} mean|Δ|={r['mean_abs']:.3e} rms={r['ref_rms']:.3e}")
print("\nmodel memory GiB:", m["model"]["weights"]["memory_gib"], "load s", m["model"]["weights"]["load_seconds"])
w = json.load(open(f"{runs}/window_2100_32.json"))
print("\nwindow run paths:", [(r["spec"]["name"], round(r["seconds"], 1)) for r in w["chain"]["results"]])
try:
    t = json.load(open(f"{runs}/{triton_name}.json"))
except FileNotFoundError:
    sys.exit(0)
import torch  # noqa: E402

golden_ref = torch.load(f"{runs}/main_512_32/strict_ref.pt")
g_train = golden_ref["logp"][511:543].tolist()  # all golden paths are byte-equal to train
t_logp = {r["spec"]["name"]: r["logp"] for r in t["chain"]["results"]}
print(f"\ngolden(train) vs {triton_name} per path: max|Δlogp| / mean|Δlogp| over 32 tokens")
for name, lp in t_logp.items():
    diffs = [abs(a - b) for a, b in zip(g_train, lp)]
    print(f"  {name:28s} max {max(diffs):.5f}  mean {sum(diffs) / len(diffs):.5f}")
print("\ntriton timings:", [(r["spec"]["name"], round(r["seconds"], 2)) for r in t["chain"]["results"]])
thf = t.get("hf_compare")
if thf:
    print("triton HF anchor:", json.dumps({k: v for k, v in thf.items() if k != "per_layer"}, indent=1))
    print("triton ops provenance:", t["chain"]["ops_provenance"])
