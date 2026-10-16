"""Native smoke: is the trainer-vs-rollout drift inside vLLM (decode vs prefill) or between engines?"""
import sys

import torch

side = torch.load(sys.argv[1], map_location="cpu", weights_only=True)
tr, ro = side["train_log_probs"], side["rollout_log_probs"]
pre = side["extra"]["rollout_prefill_log_probs"]
mx = {"train_vs_decode": 0.0, "train_vs_prefill": 0.0, "prefill_vs_decode": 0.0}
cnt = {"train_vs_decode": 0, "train_vs_prefill": 0, "prefill_vs_decode": 0,
       "train_eq_prefill_ne_decode": 0, "train_ne_both_while_decode_eq_prefill": 0, "tokens": 0}
for i in range(len(tr)):
    a = torch.as_tensor(tr[i], dtype=torch.float32)
    b = torch.as_tensor(ro[i], dtype=torch.float32)
    c = torch.as_tensor(pre[i], dtype=torch.float32)
    n = min(len(a), len(b), len(c))
    a, b, c = a[:n], b[:n], c[:n]
    ab = a.view(torch.int32) != b.view(torch.int32)
    ac = a.view(torch.int32) != c.view(torch.int32)
    cb = c.view(torch.int32) != b.view(torch.int32)
    cnt["tokens"] += n
    cnt["train_vs_decode"] += int(ab.sum())
    cnt["train_vs_prefill"] += int(ac.sum())
    cnt["prefill_vs_decode"] += int(cb.sum())
    cnt["train_eq_prefill_ne_decode"] += int((ab & ~ac).sum())
    cnt["train_ne_both_while_decode_eq_prefill"] += int((ab & ac & ~cb).sum())
    mx["train_vs_decode"] = max(mx["train_vs_decode"], float((a.double() - b.double()).abs().max()))
    mx["train_vs_prefill"] = max(mx["train_vs_prefill"], float((a.double() - c.double()).abs().max()))
    mx["prefill_vs_decode"] = max(mx["prefill_vs_decode"], float((c.double() - b.double()).abs().max()))
print(cnt)
print(mx)
