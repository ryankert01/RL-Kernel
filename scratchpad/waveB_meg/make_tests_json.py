"""Build the `tests` block embedded into megatron_runtime.json (input for megatron_results --extra)."""
import json
import sys
from pathlib import Path

S = Path(sys.argv[1])
out = {
    "code_commit": "926da6585ea54d91100427b031112eea0f8cd5bd",
    "unit_tests": [
        {
            "command": "~/bin/gpu-lease -n 1 --pool 0-2 -- env PYTHONPATH=$WT ~/envs/megatron/bin/python "
                       "-m pytest tests/models/muse_glimmer/test_megatron_strict_binding.py -q -rs",
            "outcome": (S / "final/pytest_1gpu_megatron_env.txt").read_text().strip().splitlines()[-1],
            "skipped": "NCCL TP2 / TP4 (need 2 / 4 GPUs: run separately), opt-in 30B protocol test",
        },
        {
            "command": "~/bin/gpu-lease -n 1 --pool 0-2 -- env PYTHONPATH=$WT ~/envs/vllm/bin/python -m pytest "
                       "tests/models/muse_glimmer/test_megatron_strict_binding.py "
                       "tests/models/muse_glimmer/test_vllm_strict_binding.py::TestLoRA -q -rs",
            "outcome": (S / "final/pytest_1gpu_vllm_env.txt").read_text().strip().splitlines()[-1],
        },
    ],
    "protocol_tests_30b": [
        {
            "command": "~/bin/gpu-lease -n 2 --pool 0-2 --contiguous -- env RL_KERNEL_MUSEGLIMMER_E2E=1 "
                       "MUSEGLIMMER_TRAIN_TP=2 MUSEGLIMMER_TRAIN_STRICT=1 PYTHONPATH=$WT "
                       "~/envs/megatron/bin/python -m pytest tests/models/muse_glimmer/"
                       "test_megatron_strict_binding.py -k protocol -s -q",
            "log": [ln for ln in (S / "final/protocol_tp2_strict.log").read_text().splitlines()
                    if ln.startswith(("step", "1 passed", "FAILED"))],
        },
        {
            "command": "same with MUSEGLIMMER_TRAIN_STRICT=0 (native trainer)",
            "log": [ln for ln in (S / "final/protocol_tp2_native.log").read_text().splitlines()
                    if ln.startswith(("step", "1 passed", "FAILED"))],
        },
    ],
    "before_after_probe": {
        "script": "scratchpad/waveB_meg/probe_before_after.py (conditions of the new regression tests)",
        "wave_a_tree_f91e8b6": json.loads((S / "probe_before.json").read_text()),
        "this_tree": json.loads((S / "probe_after.json").read_text()),
    },
}
nccl = S / "final/pytest_nccl.txt"
if nccl.exists():
    out["unit_tests"].append({
        "command": "~/bin/gpu-lease -n 4 --pool 0-3 --contiguous -- env PYTHONPATH=$WT "
                   "~/envs/megatron/bin/python -m pytest tests/models/muse_glimmer/"
                   "test_megatron_strict_binding.py -k nccl -q -rs",
        "outcome": nccl.read_text().strip().splitlines()[-1],
    })
else:
    out["unit_tests"].append({"command": "NCCL TP2/TP4 tests", "outcome": "NOT RUN"})
Path(sys.argv[2]).write_text(json.dumps(out, indent=1) + "\n")
print("ok")
