"""Probe: CUDA mock trainer, bridged (same env) vs in process, with ratio != 1.

Under the bridge the GRPO loss/backward runs on the CPU (client), in process it runs on the
trainer's device. The GPU smoke only compared ratio == 1 (exp(0) == 1 exactly on both devices).
"""

import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, os.environ["REPO"])
from rl_engine.models.muse_glimmer.e2e import bridge, grpo, mocks  # noqa: E402

HERE = Path(__file__).resolve().parent
SEQS = [[256, 3, 5, 7, 9, 11, 13], [256, 4, 6, 8, 10], [256, 1, 2, 3, 4, 5, 6, 7], [256, 9, 9]]
STARTS = [3, 2, 4, 1]
ADV = torch.tensor([1.0, -1.0, 0.5, -0.5])
LEASE = bridge.parse_visible_devices(os.environ.get("CUDA_VISIBLE_DEVICES"))


def steps(trainer, shift: float, n: int = 3):
    out = []
    for _ in range(n):
        lps = trainer.token_logprobs(SEQS, STARTS)
        gen = torch.Generator().manual_seed(11)
        rollout = [
            (t.detach().cpu() + shift * torch.randn(t.shape, generator=gen)).to(t.device)
            for t in lps
        ]
        loss = grpo.grpo_token_loss(lps, rollout, ADV.to(lps[0].device)).loss
        info = trainer.step(loss)
        sd, sha = trainer.export_lora()
        out.append((float(info["grad_norm"]), sha))
    return out


for shift in (0.0, 0.05):
    local = steps(mocks.MockTrainer("mock", tensor_parallel_size=1, gpu_ids=[], lora_rank=4,
                                    lr=1e-3, strict=True, seed=7, device="cuda"), shift)
    r = bridge.RemoteTrainer("mock", tensor_parallel_size=1, gpu_ids=LEASE[:1], lora_rank=4,
                             lr=1e-3, strict=True, seed=7, python=sys.executable,
                             factory=bridge.MOCK_TRAINER_FACTORY,
                             factory_kwargs={"device": "cuda"}, work_dir=HERE / f"ratio_{shift}")
    try:
        remote = steps(r, shift)
    finally:
        r.close()
    for i, (a, b) in enumerate(zip(local, remote)):
        print(f"shift {shift} step {i}: grad_norm local {a[0]!r} bridged {b[0]!r} "
              f"equal {a[0] == b[0]} | sha equal {a[1] == b[1]}")
