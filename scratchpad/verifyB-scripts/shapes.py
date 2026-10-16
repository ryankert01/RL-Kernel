import json, sys, math
h1 = json.load(open(sys.argv[1])); h2 = json.load(open(sys.argv[2]))
allt = {}
for h in (h1, h2):
    for k, v in h.items():
        if k == "__metadata__": continue
        allt[k] = v
for k in sorted(allt):
    if k.startswith("model.language_model.layers.0.") or k.startswith("language_model.layers.0.") or "layers.0." in k and "vision" not in k:
        print(k, allt[k]["dtype"], allt[k]["shape"])
for k in sorted(allt):
    if "layers" not in k and "vision" not in k and "visual" not in k:
        print(k, allt[k]["dtype"], allt[k]["shape"])
print("norm-like non-layer tensors:", [k for k in allt if ("norm" in k and "layers" not in k)])
print("q_norm/k_norm/embed_norm any:", [k for k in allt if "q_norm" in k or "k_norm" in k or "embed_norm" in k])
print("3.87*sqrt(128)=", 3.87*math.sqrt(128), " 1/sqrt(26)=", 1/math.sqrt(26), " 3.87/sqrt(128)=", 3.87/math.sqrt(128), " 43.7840518911/128=", 43.7840518911/128)
