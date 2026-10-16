import sys
import hashlib
import torch

paths = sys.argv[1:]
dumps = [torch.load(p, map_location="cpu", weights_only=True) for p in paths]
base = dumps[0]
print("keys", len(base["grads"]), "elements", sum(t.numel() for t in base["grads"].values()))
zero = sum(1 for t in base["grads"].values() if float(t.abs().max()) == 0.0)
print("zero tensors in", paths[0], zero)
# grads that are exactly equal to the LoRA B init or suspicious? check min nonzero fraction
nz = sum(int((t != 0).sum()) for t in base["grads"].values())
print("nonzero elements", nz)
for p, d in zip(paths[1:], dumps[1:]):
    g1, g2 = base["grads"], d["grads"]
    assert sorted(g1) == sorted(g2)
    mism = 0
    mt = 0
    mx = 0.0
    for k in sorted(g1):
        a = g1[k].float().contiguous()
        b = g2[k].float().contiguous()
        m = int((a.view(torch.int32) != b.view(torch.int32)).sum())
        mism += m
        mt += 1 if m else 0
        if m:
            mx = max(mx, float((a.double() - b.double()).abs().max()))
    lpm = [int((x.float().view(torch.int32) != y.float().view(torch.int32)).sum()) for x, y in zip(base["lps"], d["lps"])]
    print(f"{paths[0]} vs {p}: mismatching elements {mism}, tensors {mt}, max_abs {mx}, lp mism {lpm}")


def agg(g):
    return hashlib.sha256("".join(hashlib.sha256(g[k].float().contiguous().view(torch.int32).numpy().tobytes()).hexdigest() for k in sorted(g)).encode()).hexdigest()


for p, d in zip(paths, dumps):
    print(p, "aggregate", agg(d["grads"]))
