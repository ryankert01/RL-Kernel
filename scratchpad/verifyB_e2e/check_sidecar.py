"""Does the trainer agree with vLLM's PREFILL where it disagrees with vLLM's decode? (native smoke)"""
import sys

import torch

side = torch.load(sys.argv[1], map_location="cpu", weights_only=True)
print("keys", sorted(side.keys()) if isinstance(side, dict) else type(side))
for k, v in (side.items() if isinstance(side, dict) else []):
    if isinstance(v, list):
        print(" ", k, "list", len(v), type(v[0]).__name__ if v else None)
    elif isinstance(v, torch.Tensor):
        print(" ", k, tuple(v.shape), v.dtype)
    else:
        print(" ", k, str(v)[:120])

ex = side["extra"]
pre = ex.get("rollout_prefill_log_probs")
tr, ro = side["train_log_probs"], side["rollout_log_probs"]
def bits(t):
    return torch.as_tensor(t, dtype=torch.float32).view(torch.int32)
tot = tr_ne_ro = tr_ne_pre = pre_ne_ro = both = 0
for i in range(len(tr)):
    a, b, c = bits(tr[i]), bits(ro[i]), bits(pre[i])
    n = min(len(a), len(b), len(c))
    a, b, c = a[:n], b[:n], c[:n]
    tot += n
    x, y, z = (a != b), (a != c), (c != b)
    tr_ne_ro += int(x.sum()); tr_ne_pre += int(y.sum()); pre_ne_ro += int(z.sum())
    both += int((x & ~y).sum())
print(f"tokens {tot}: train!=rollout_decode {tr_ne_ro}; train!=rollout_prefill {tr_ne_pre}; "
      f"prefill!=decode {pre_ne_ro}; train==prefill but !=decode {both}")
