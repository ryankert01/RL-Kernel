"""Probe: echo RTT through the trainer server with and without client-side TCP_NODELAY, plus one
round trip of the real-shape rank-16 LoRA (200 MiB). CPU only (server CUDA_VISIBLE_DEVICES="")."""

import os
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, os.environ["REPO"])
from rl_engine.models.muse_glimmer.e2e import bench_bridge, bridge  # noqa: E402
from rl_engine.models.muse_glimmer.e2e.sync import lora_sha256  # noqa: E402

HERE = Path(__file__).resolve().parent
MEG = str(Path("~/envs/megatron/bin/python").expanduser())


def run(label: str) -> None:
    r = bridge.RemoteTrainer("mock", tensor_parallel_size=1, gpu_ids=[], lora_rank=4, lr=1e-3,
                             strict=True, seed=7, python=MEG, factory=bridge.MOCK_TRAINER_FACTORY,
                             work_dir=HERE / f"transport_{label}",
                             extra_env={"OMP_NUM_THREADS": "2"})
    try:
        for size in (16 << 10, 256 << 10, 1 << 20):
            p = torch.randint(0, 256, (size,), dtype=torch.uint8)
            rtts = []
            for _ in range(5):
                echoed, st = r.ping(p)
                assert torch.equal(echoed, p)
                rtts.append(st["rtt_s"] * 1e3)
            print(f"{label}: {size >> 10} KiB median rtt {statistics.median(rtts):.2f} ms {rtts}")
        if label == "nodelay":
            lora = bench_bridge._real_shape_lora()
            nbytes = sum(t.numel() * t.element_size() for t in lora.values())
            echoed, st = r.ping(lora)
            print(f"real LoRA {len(lora)} tensors {nbytes} B rtt {st['rtt_s']*1e3:.0f} ms "
                  f"sha_equal {lora_sha256(echoed) == lora_sha256(lora)}")
    finally:
        r.close()


run("nodelay")
bridge.set_nodelay = lambda conn: None  # client side only; the server keeps its own NODELAY
run("client_nagle")
