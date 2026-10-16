import json, sys, re, collections

h1, h2, idx = sys.argv[1], sys.argv[2], sys.argv[3]
hdr = {}
for p in (h1, h2):
    with open(p) as f:
        d = json.load(f)
    for k, v in d.items():
        if k == "__metadata__":
            print(p, "metadata:", v)
            continue
        hdr[k] = v

print("tensor count:", len(hdr))
dtypes = collections.Counter(v["dtype"] for v in hdr.values())
print("dtypes:", dict(dtypes))

# per-layer tensors of layer 0 (local) and layer 3 (global)
for L in (0, 3):
    print(f"\n--- layer {L} ---")
    for k in sorted(hdr):
        if k.startswith(f"model.language_model.layers.{L}."):
            print(k.split(f"layers.{L}.")[1], hdr[k]["dtype"], hdr[k]["shape"])

print("\n--- non-layer text tensors ---")
for k in sorted(hdr):
    if "language_model.layers." not in k and "vision" not in k:
        print(k, hdr[k]["dtype"], hdr[k]["shape"])

# consistency: all 52 layers have identical suffix->shape
suffix_shapes = collections.defaultdict(set)
layer_ids = set()
for k, v in hdr.items():
    m = re.match(r"model\.language_model\.layers\.(\d+)\.(.*)", k)
    if m:
        layer_ids.add(int(m.group(1)))
        suffix_shapes[m.group(2)].add(tuple(v["shape"]))
print("\nlayers present:", min(layer_ids), "..", max(layer_ids), "count", len(layer_ids))
for s, shapes in sorted(suffix_shapes.items()):
    print(s, sorted(shapes), "UNIFORM" if len(shapes) == 1 else "VARIES")

# param totals
def numel(shape):
    n = 1
    for s in shape:
        n *= s
    return n

text = sum(numel(v["shape"]) for k, v in hdr.items() if k.startswith("model.language_model.") or k == "lm_head.weight")
vision = sum(numel(v["shape"]) for k, v in hdr.items() if "vision" in k or "perception" in k)
other = sum(numel(v["shape"]) for k, v in hdr.items() if not (k.startswith("model.language_model.") or k == "lm_head.weight" or "vision" in k or "perception" in k))
total = text + vision + other
print("\ntext params:", text, "vision params:", vision, "other:", other, "total:", total)
other_keys = [k for k in hdr if not (k.startswith("model.language_model.") or k == "lm_head.weight" or "vision" in k or "perception" in k)]
print("other keys:", other_keys[:20])
text_count = sum(1 for k in hdr if k.startswith("model.language_model.") or k == "lm_head.weight")
print("text tensor count:", text_count)

# per layer GEMM param count / per-layer total
per_layer = sum(numel(v["shape"]) for k, v in hdr.items() if k.startswith("model.language_model.layers.0."))
print("params per layer:", per_layer, " x52 =", per_layer * 52)
emb = numel(hdr["model.language_model.embed_tokens.weight"]["shape"])
lm = numel(hdr["lm_head.weight"]["shape"])
print("embed:", emb, "lm_head:", lm)
# bytes at bf16
print("text bytes bf16:", text * 2, "=", text * 2 / 1e9, "GB (1e9)")
print("total bytes bf16:", total * 2, "=", total * 2 / 1e9, "GB (1e9)", total * 2 / 2**30, "GiB")

with open(idx) as f:
    ix = json.load(f)
print("\nindex metadata:", ix.get("metadata"))
wm = ix["weight_map"]
print("weight_map entries:", len(wm))
print("shards:", sorted(set(wm.values())))

# attention gate GEMM: which layers have self_attn.gate_proj
gp = [k for k in hdr if k.endswith("self_attn.gate_proj.weight")]
print("self_attn.gate_proj count:", len(gp))
# any qk_norm weights? any biases?
print("qk_norm/q_norm/k_norm weights:", [k for k in hdr if "norm" in k and "self_attn" in k][:5])
print("biases in text model:", [k for k in hdr if k.endswith(".bias") and "language_model" in k][:5])
print("embed_norm weights:", [k for k in hdr if "embed_norm" in k])
