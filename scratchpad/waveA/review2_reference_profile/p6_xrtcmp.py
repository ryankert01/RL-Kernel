import json, sys, struct
D = sys.argv[1]
L = {n: json.load(open(f"{D}/xrt_reference_{n}.json")) for n in ("golden", "triton", "hf_bf16")}


def f(h):
    return struct.unpack("<f", struct.pack("<I", int(h, 16)))[0]


def cmp(a, b):
    A = {s["id"]: s for s in L[a]["sequences"]}
    B = {s["id"]: s for s in L[b]["sequences"]}
    eq = n = 0
    allab = []
    per = {}
    for k in A:
        ha, hb = A[k]["logprobs_hex"], B[k]["logprobs_hex"]
        assert len(ha) == len(hb)
        e = sum(x == y for x, y in zip(ha, hb))
        ab = [abs(f(x) - f(y)) for x, y in zip(ha, hb)]
        eq += e; n += len(ha); allab += ab
        per[k] = (round(max(ab), 4), round(sum(ab) / len(ab), 5), e)
    print(f"{a} vs {b}: {eq}/{n} equal, max {max(allab):.4f}, mean {sum(allab)/len(allab):.5f}")
    for k, v in per.items():
        print("    ", k, v)


cmp("golden", "triton")
cmp("triton", "hf_bf16")
cmp("golden", "hf_bf16")
hf = L["hf_bf16"]
hits = sum(sum(s["argmax_is_token"]) for s in hf["sequences"])
n = sum(len(s["argmax_is_token"]) for s in hf["sequences"])
print("HF argmax==token", hits, n, hits / n)
# positive / zero logp?
for nm, x in L.items():
    vals = [f(h) for s in x["sequences"] for h in s["logprobs_hex"]]
    print(nm, "min", min(vals), "max", max(vals), "n>=0", sum(v >= 0 for v in vals), "neg zero", sum(1 for s in x["sequences"] for h in s["logprobs_hex"] if h == "80000000"), "nan", sum(v != v for v in vals))
# the profile JSON's derived numbers
d = json.load(open(f"{D}/reference_profile.json"))
X = d["experiments"]["xrt_scoring"]
print("profile golden_vs_triton", X["golden_vs_triton"]["equal_tokens"], X["golden_vs_triton"]["n_tokens"])
for k in ("triton_vs_hf_bf16", "golden_vs_hf_bf16"):
    print(k, X[k]["equal_tokens"], X[k]["max_abs_dlogp"], X[k]["mean_abs_dlogp"])
print("repeat", d["experiments"]["xrt_triton_repeat"])
print("commands:")
for c in d["commands"]:
    print("  ", c if isinstance(c, str) else json.dumps(c)[:400])
