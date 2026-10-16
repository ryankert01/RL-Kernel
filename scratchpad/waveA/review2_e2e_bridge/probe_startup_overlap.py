"""Probe: does RemoteTrainer(wait=False) let the trainer build while the caller does other work?

run_poc_grpo.build_engines launches the trainer with wait=False, builds vLLM, then wait_ready().
The server accepts the client connection BEFORE building the trainer; the client connects only in
wait_ready(). If so, the build cannot start until wait_ready() -> no overlap.

The probe uses a mock factory with a 6 s constructor delay (SlowMockTrainer below is injected via a
tiny module written next to this script and put on PYTHONPATH of the server).
"""

import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, os.environ["REPO"])

from rl_engine.models.muse_glimmer.e2e import bridge  # noqa: E402

work = HERE / "overlap_work"
work.mkdir(exist_ok=True)
mod = HERE / "slowmock_mod"
mod.mkdir(exist_ok=True)
(mod / "slowmock.py").write_text(
    "import time\n"
    "from rl_engine.models.muse_glimmer.e2e.mocks import MockTrainer\n"
    "class SlowMockTrainer(MockTrainer):\n"
    "    def __init__(self, *a, **k):\n"
    "        time.sleep(6.0)\n"
    "        super().__init__(*a, **k)\n"
)
t0 = time.monotonic()
r = bridge.RemoteTrainer(
    "mock", tensor_parallel_size=1, gpu_ids=[], lora_rank=4, lr=1e-3, strict=True, seed=7,
    python=str(Path("~/envs/megatron/bin/python").expanduser()), factory="slowmock:SlowMockTrainer",
    work_dir=work, wait=False, extra_env={"PYTHONPATH": f"{os.environ['REPO']}:{mod}"},
)
# Simulate "vLLM builds in this process" for 10 s, longer than the 6 s trainer build.
time.sleep(10.0)
t_wait = time.monotonic()
hello = r.wait_ready()
t_ready = time.monotonic()
r.close()
print(f"wait_ready() blocked for {t_ready - t_wait:.2f} s after 10 s of 'rollout build'")
print(f"server build_s={hello['build_s']:.2f}")
print("--- server log ---")
print((work / "trainer_server.log").read_text())
