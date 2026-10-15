# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""GPU smoke of the cross-environment trainer bridge (row ``final_e2e_lora``).

Run under a one-GPU lease (the tests read the leased physical id from CUDA_VISIBLE_DEVICES and
skip without one; they never pick a GPU themselves):

    ~/bin/gpu-lease -n 1 --pool 6-7 -- env PYTHONPATH=$PWD ~/envs/vllm/bin/python -m pytest \
        tests/models/muse_glimmer/test_e2e_bridge_gpu.py -q

The real ``StrictTrainer`` is pinned to the 30B shapes (``megatron_strict._check_config``), so a
toy-config StrictTrainer smoke is not possible; these tests drive the CUDA ``MockTrainer``
through the real transport instead (placement, device round trips, surrogate exactness on CUDA).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch

from rl_engine.models.muse_glimmer.e2e import bridge, grpo, mocks

MEGATRON_PY = Path("~/envs/megatron/bin/python").expanduser()
LEASE = bridge.parse_visible_devices(os.environ.get("CUDA_VISIBLE_DEVICES"))
pytestmark = [
    pytest.mark.skipif(not LEASE, reason="needs a GPU lease (CUDA_VISIBLE_DEVICES with ids)"),
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]

SEQS = [[256, 3, 5, 7, 9, 11, 13], [256, 4, 6, 8, 10], [256, 1, 2, 3, 4, 5, 6, 7], [256, 9, 9]]
STARTS = [3, 2, 4, 1]
ADVANTAGES = torch.tensor([1.0, -1.0, 0.5, -0.5])


def _bits(t: torch.Tensor) -> bytes:
    return bridge.tensor_bytes(t)


def test_surrogate_gradient_exact_on_cuda() -> None:
    lengths = [5, 64, 1000]
    src = torch.randn(sum(lengths), device="cuda", requires_grad=True)
    logps = list(torch.split(src * 1.0, lengths))
    gen = torch.Generator().manual_seed(0)
    grads = [torch.randn(n, generator=gen) * torch.logspace(-40, 38, n) for n in lengths]
    grads[0][:4] = torch.tensor([0.0, -0.0, 1e-45, -3.4e38])
    got = torch.autograd.grad(bridge.surrogate_loss(logps, grads), logps)
    for g_got, g_want in zip(got, grads):
        assert g_got.device.type == "cuda"
        assert _bits(g_got) == _bits(g_want)


def _steps(trainer, steps: int = 2):
    out = []
    for _ in range(steps):
        lps = trainer.token_logprobs(SEQS, STARTS)
        loss = grpo.grpo_token_loss(lps, [t.detach().clone() for t in lps], ADVANTAGES).loss
        info = trainer.step(loss)
        state_dict, sha = trainer.export_lora()
        out.append(([_bits(t) for t in lps], info["grad_norm"], sha))
    return out


@pytest.mark.skipif(not MEGATRON_PY.exists(), reason="~/envs/megatron/bin/python missing")
@pytest.mark.parametrize("which", ["megatron_env", "same_env"])
def test_cuda_trainer_through_bridge_sees_exactly_the_leased_gpu(tmp_path: Path, which) -> None:
    gpu = LEASE[:1]
    parent_uuid = str(torch.cuda.get_device_properties(0).uuid)
    python = MEGATRON_PY if which == "megatron_env" else Path(sys.executable)
    remote = bridge.RemoteTrainer(
        "mock", tensor_parallel_size=1, gpu_ids=gpu, lora_rank=4, lr=1e-3, strict=True, seed=7,
        python=str(python), factory=bridge.MOCK_TRAINER_FACTORY,
        factory_kwargs={"device": "cuda"}, work_dir=tmp_path / which, call_timeout_s=300.0,
        startup_timeout_s=300.0,
    )
    try:
        hello = remote.wait_ready()
        devices = hello["devices"]
        assert devices["cuda_visible_devices"] == ",".join(str(g) for g in gpu)
        assert devices["device_count"] == 1
        assert devices["devices"][0]["uuid"] == parent_uuid  # same physical GPU as the lease
        got = _steps(remote)
        calls = [c for c in remote.calls if c["op"] in ("token_logprobs", "step")]
        assert all((c["server"].get("memory") or {}).get("peak_allocated_gib") is not None
                   for c in calls)
        assert remote.provenance()["device"] == "cuda"
    finally:
        remote.close()
    want = _steps(
        mocks.MockTrainer("mock", tensor_parallel_size=1, gpu_ids=[], lora_rank=4, lr=1e-3,
                          strict=True, seed=7, device="cuda")
    )
    if which == "same_env":
        assert got == want  # the bridge itself adds no numerical difference
    else:
        # Different torch builds (cu129 vs cu130): report-only, the bridge is not the variable.
        for (lp_r, _, _), (lp_l, _, _) in zip(got, want):
            assert len(lp_r) == len(lp_l)
