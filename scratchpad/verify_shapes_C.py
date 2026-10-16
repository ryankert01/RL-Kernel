import json, sys, re, collections
h = {}
for f in ("hf/h1.json","hf/h2.json"):
    d = json.load(open(f))
    for k,v in d.items():
        if k == "__metadata__": continue
        h[k] = v
print("tensors", len(h))
text = {k:v for k,v in h.items() if "language_model" in k or k.startswith("lm_head")}
print("text tensors", len(text), "dtypes", set(v["dtype"] for v in text.values()))
# per-layer shapes
per = collections.defaultdict(set)
for k,v in text.items():
    m = re.match(r"model\.language_model\.layers\.(\d+)\.(.*)\.weight", k)
    if m:
        per[m.group(2)].add(tuple(v["shape"]))
for name in sorted(per): print(name, per[name])
layers = set(int(re.match(r"model\.language_model\.layers\.(\d+)\.", k).group(1)) for k in text if ".layers." in k)
print("layers", len(layers), min(layers), max(layers))
print("bias tensors", [k for k in h if "bias" in k and "language_model" in k])
for k in ("model.language_model.embed_tokens.weight","lm_head.weight","model.language_model.norm.weight"):
    print(k, h.get(k))
print([k for k in text if "embed" in k or "norm" in k and "layers" not in k])
print("non-layer text", [k for k in text if ".layers." not in k])
# params
tp = sum(eval("*".join(map(str,v["shape"]))) if v["shape"] else 1 for v in text.values())
print("text params", tp)
print(4096+256+256, 4096+256+256+4096, 2*19968, 1/26**0.5, 43.7840518911/128**0.5, 3.87*128**0.5)
