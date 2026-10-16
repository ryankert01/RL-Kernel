import sys
import time

import torch

threads = int(sys.argv[1]) if len(sys.argv) > 1 else 0
if threads:
    torch.set_num_threads(threads)
print("threads", torch.get_num_threads())

from rl_engine.models.muse_glimmer.e2e.drift import compare_logprobs
from rl_engine.models.muse_glimmer.e2e.mocks import (
    ByteTokenizer,
    MockRolloutEngine,
    MockTrainer,
    strict_exp,
    strict_log,
)

x = torch.linspace(-20, 20, 10001)
print("exp max rel err", ((strict_exp(x) - torch.exp(x)).abs() / torch.exp(x)).max().item())
y = torch.linspace(1e-3, 1e3, 10001)
print("log max abs err", (strict_log(y) - torch.log(y)).abs().max().item())

tok = ByteTokenizer()
prompts = [tok.encode_chat([{"role": "user", "content": f"What is {i} + {i}?"}]) for i in range(3)]


def run(ts, rs, perturb=0.0):
    tr = MockTrainer("m", tensor_parallel_size=1, gpu_ids=[], lora_rank=4, lr=1e-3, strict=ts, seed=1)
    ro = MockRolloutEngine(
        "m", tensor_parallel_size=1, gpu_ids=[], strict=rs, max_model_len=512, seed=2,
        lora_rank=4, perturb=perturb,
    )
    sd, sha = tr.export_lora()
    ro.load_lora(sd, sha)
    t0 = time.time()
    groups = ro.generate(prompts, n=2, max_tokens=20, temperature=1.0)
    t_gen = time.time() - t0
    seqs = [p + s.token_ids for p, g in zip(prompts, groups) for s in g]
    starts = [len(p) for p, g in zip(prompts, groups) for s in g]
    roll = [s.logprobs for g in groups for s in g]
    t0 = time.time()
    train = tr.token_logprobs(seqs, starts)
    t_train = time.time() - t0
    m = compare_logprobs([t.detach() for t in train], roll)
    pre = ro.logprobs(seqs, starts)
    mi = compare_logprobs(pre, roll)
    return tr, ro, seqs, starts, train, roll, m, mi, t_gen, t_train


tr, ro, seqs, starts, train, roll, m, mi, t_gen, t_train = run(True, True)
print("strict/strict gen %.2fs train %.2fs" % (t_gen, t_train))
print("strict/strict drift:", m.bitwise_mismatch_count, m.max_abs_diff, m.bitwise_equal)
single = [tr.token_logprobs([s], [c])[0].detach() for s, c in zip(seqs, starts)]
print("batched vs single trainer:", compare_logprobs(single, [t.detach() for t in train]).bitwise_equal)
print("rollout prefill vs decode:", mi.bitwise_equal)
for ts, rs in [(False, False), (True, False), (False, True)]:
    _, _, _, _, _, _, m, mi, _, _ = run(ts, rs)
    name = ("strict" if ts else "native") + "/" + ("strict" if rs else "native")
    print(name, m.bitwise_mismatch_count, m.token_count, "%.3e" % m.max_abs_diff, m.first_mismatch,
          "| rollout internal", mi.bitwise_mismatch_count, "%.3e" % mi.max_abs_diff)
_, _, _, _, _, _, m, mi, _, _ = run(True, True, perturb=1e-3)
print("perturbed strict/strict", m.bitwise_mismatch_count, m.token_count, "%.3e" % m.max_abs_diff, m.first_mismatch)

sd, sha = tr.export_lora()
loss = sum((t * torch.randn_like(t)).mean() for t in train)
info = tr.step(loss)
print(info)
sd2, sha2 = tr.export_lora()
print("sha changed:", sha2 != sha)
