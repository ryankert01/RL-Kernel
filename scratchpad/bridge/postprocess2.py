import json
from pathlib import Path

p = Path("/home/ubuntu/RL-Kernel/.claude/worktrees/wf_7b064961-36f-3/reports/muse_glimmer_poc/data/e2e_bridge.json")
d = json.loads(p.read_text())
d["gpu_smoke"]["commit"] = "a8ef05d145f4639bb96c9d13aeb5d4130494dbea"
d["env"].pop("client", None)
w = "W=<worktree>; cd $W; "
d["commands"] = [
    "[0738241, CPU, no lease] " + w + "PYTHONPATH=$W ~/envs/vllm/bin/python -m "
    "rl_engine.models.muse_glimmer.e2e.bench_bridge --cpu --hash-check --work-dir <scratch>",
    "[a8ef05d, GPU 7] " + w + "~/bin/gpu-lease -n 1 --pool 6-7 --timeout 1800 -- env "
    "PYTHONPATH=$W ~/envs/vllm/bin/python -m rl_engine.models.muse_glimmer.e2e.bench_bridge "
    "--gpu --work-dir <scratch>",
    "[0738241, CPU] " + w + "PYTHONPATH=$W ~/envs/vllm/bin/python -m pytest "
    "tests/models/muse_glimmer/test_e2e_bridge.py tests/models/muse_glimmer/test_e2e_loop_mock.py"
    " -q -p no:cacheprovider   -> 58 passed",
    "[a8ef05d, GPU 7] " + w + "~/bin/gpu-lease -n 1 --pool 6-7 --timeout 1800 -- env "
    "PYTHONPATH=$W ~/envs/vllm/bin/python -m pytest tests/models/muse_glimmer/"
    "test_e2e_bridge_gpu.py -q -p no:cacheprovider   -> 3 passed",
]
d["status"] = (
    "complete for the bridge + instrumentation scope: CPU sections at 0738241 (no GPU), GPU "
    "smoke at a8ef05d (A100 GPU 7 under lease); the REAL 30B StrictTrainer behind the bridge "
    "and the real E2E were NOT RUN (integrator)"
)
d["incident"] = {
    "what": "CPU-only runs before 0738241 (tests and the first bench runs, no lease) created a "
    "CUDA context on GPU 0 for their duration: torch 2.13 Adam.step probes "
    "torch.accelerator.current_stream() even for CPU parameters. Fixed in mocks.CPUSafeAdam; "
    "test_cpu_mock_trainer_never_initializes_cuda guards it. Any CPU test that steps a "
    "torch.optim optimizer in a GPU-visible process has the same effect.",
}
p.write_text(json.dumps(d, indent=1, default=str) + "\n", encoding="utf-8")
print(sorted(d))
