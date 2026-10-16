"""Report-only: does the GRPO update move a dense, learnable mock reward? (strict/strict)."""

import json
import random
import sys
from pathlib import Path

import torch

torch.set_num_threads(8)

from rl_engine.models.muse_glimmer.e2e import data, loop, mocks

run_dir = Path(sys.argv[1])
lr = float(sys.argv[2]) if len(sys.argv) > 2 else 1e-2


class FirstTokenParityTask:
    name = "first-token-parity"

    def __init__(self):
        self.inner = data.SyntheticArithmeticTask(mocks.ByteTokenizer())

    def sample(self, rng, count):
        return self.inner.sample(rng, count)

    def reward(self, completion_ids, prompt):
        return 1.0 if int(completion_ids[0]) % 2 == 0 else 0.0

    def decode(self, ids):
        return self.inner.decode(ids)

    def describe(self):
        return {"task": self.name, "reward": "1.0 iff first completion token id is even"}


cfg = loop.LoopConfig(
    run_dir=run_dir, steps=40, prompts_per_step=8, samples_per_prompt=4, max_tokens=8,
    lora_rank=16, lr=lr, mode="strict/strict", mock=True, task="first-token-parity", seed=1234,
)
trainer = mocks.MockTrainer(
    "mock", tensor_parallel_size=4, gpu_ids=[4, 5, 6, 7], lora_rank=16, lr=lr, strict=True,
    seed=1234,
)
rollout = mocks.MockRolloutEngine(
    "mock", tensor_parallel_size=2, gpu_ids=[0, 1], strict=True, max_model_len=1024, seed=1234,
    lora_rank=16,
)
summary = loop.run_loop(cfg, FirstTokenParityTask(), rollout, trainer, log=None)
r = summary["reward"]["trajectory"]
print("reward trajectory:", [round(x, 3) for x in r])
print("first 10 mean %.3f  last 10 mean %.3f" % (sum(r[:10]) / 10, sum(r[-10:]) / 10))
print("drift totals", json.dumps(summary["drift_totals"]))
print("sync", summary["weight_sync"]["passed"], "/", summary["weight_sync"]["checks"])
