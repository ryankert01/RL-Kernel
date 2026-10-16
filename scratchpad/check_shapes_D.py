import json, sys
base = sys.argv[1]
h1 = json.load(open(base + "/h1.json")); h2 = json.load(open(base + "/h2.json"))
allt = {}
for h in (h1, h2):
    for k, v in h.items():
        if k == "__metadata__": continue
        allt[k] = v
def show(name):
    v = allt.get(name)
    print(name, v["dtype"] if v else None, v["shape"] if v else None)
for n in ["model.language_model.layers.0.self_attn.q_proj.weight","model.language_model.layers.0.self_attn.k_proj.weight","model.language_model.layers.0.self_attn.v_proj.weight","model.language_model.layers.0.self_attn.gate_proj.weight","model.language_model.layers.0.self_attn.o_proj.weight","model.language_model.layers.0.mlp.gate_proj.weight","model.language_model.layers.0.mlp.up_proj.weight","model.language_model.layers.0.mlp.down_proj.weight","lm_head.weight","model.language_model.embed_tokens.weight","model.language_model.norm.weight"]:
    show(n)
# any fused names?
fused = [k for k in allt if any(s in k for s in ("qkv", "gate_up", "linear_fc1", "linear_qkv"))]
print("fused-looking tensors:", fused[:5], len(fused))
idx = json.load(open(base + "/model.safetensors.index.json"))
print("metadata:", idx.get("metadata"))
# arithmetic
print("4096/8 =", 4096/8, "heads/shard =", 512/128)
print("19968/8 =", 19968/8)
print("202048 % 128 =", 202048 % 128, "202048 % 1024 =", 202048 % 1024, "202048/1024 =", 202048/1024)
print("padded 128*8:", -(-202048//1024)*1024, "per-rank TP8:", -(-202048//1024)*1024//8, "pad count:", -(-202048//1024)*1024 - 202048)
print("padded 128*4 (TP4):", -(-202048//512)*512, " padded 128*2:", -(-202048//256)*256, " padded 128:", -(-202048//128)*128)
print("202048 % 64 =", 202048 % 64, "202048/8 =", 202048/8, "202048/8 % 64 =", (202048/8) % 64)
n_rep = 32//2
for tp in (1,2,4,8):
    per = 32//tp
    groups = []
    for r in range(tp):
        heads = list(range(r*per, (r+1)*per))
        kvs = sorted(set(h//n_rep for h in heads))
        groups.append((r, heads[0], heads[-1], kvs))
    print("TP", tp, [(r, f"q{a}-{b}", f"kv{k}") for r,a,b,k in groups])
