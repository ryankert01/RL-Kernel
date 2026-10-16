"""Mutation probe: does test_cpu_mock_trainer_never_initializes_cuda fail when CPUSafeAdam is
replaced by plain torch.optim.Adam? Run under a GPU lease (plain Adam may create a context)."""

import sys

import torch

from rl_engine.models.muse_glimmer.e2e import grpo, mocks

SEQS = [[256, 3, 5, 7, 9, 11, 13], [256, 4, 6, 8, 10], [256, 1, 2, 3, 4, 5, 6, 7], [256, 9, 9]]
STARTS = [3, 2, 4, 1]
ADV = torch.tensor([1.0, -1.0, 0.5, -0.5])


def run(label: str) -> None:
    orig = torch.cuda._lazy_init

    def blocked(*a, **k):
        raise AssertionError("CUDA init attempted")

    torch.cuda._lazy_init = blocked
    try:
        tr = mocks.MockTrainer("mock", tensor_parallel_size=1, gpu_ids=[], lora_rank=4, lr=1e-3,
                               strict=True, seed=7)
        for _ in range(2):
            lps = tr.token_logprobs(SEQS, STARTS)
            tr.step(grpo.grpo_token_loss(lps, [t.detach() for t in lps], ADV).loss)
        print(label, "-> no CUDA init attempt detected; cuda initialized:",
              torch.cuda.is_initialized())
    except AssertionError as exc:
        print(label, "-> guard tripped:", exc)
    finally:
        torch.cuda._lazy_init = orig


which = sys.argv[1]
if which == "plain":
    mocks.CPUSafeAdam = torch.optim.Adam  # mutation
run(which)
