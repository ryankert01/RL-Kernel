import json
import subprocess
import sys

S = "/tmp/claude-1000/-home-ubuntu-RL-Kernel--claude-worktrees-swift-oak-hsv7/80655dc5-1eae-4992-a93b-f6f87e8ab1d8/scratchpad/attn_norms_B"
W = "/home/ubuntu/RL-Kernel/.claude/worktrees/wf_65f5032c-579-4"
RAW = "reports/muse_glimmer_poc/data/kernels_attention_raw/waveB"
tests = {
    "owned_files": {
        "command": "PYTHONPATH=$WT ~/bin/gpu-lease -n 1 --pool 3,7 -- ~/envs/vllm/bin/python -m "
                   "pytest -q -p no:cacheprovider tests/models/muse_glimmer/"
                   "test_gated_attention_fwd.py tests/models/muse_glimmer/"
                   "test_gated_attention_bwd.py tests/models/muse_glimmer/test_norms.py -rs",
        "result": "223 passed, 1 skipped, 1 warning in 107.87s",
        "skipped": ["test_norms.py::test_fused_rejects_two_gpus (needs two visible GPUs)"],
        "per_file_collected": {"test_gated_attention_fwd.py": 101,
                               "test_gated_attention_bwd.py": 75, "test_norms.py": 48},
        "log": f"{RAW}/tests_owned.log.txt",
        "commit_measured": "060d7d4",
    },
    "full_single_gpu_suite_vllm_env": {
        "command": "PYTHONPATH=$WT ~/bin/gpu-lease -n 1 --pool 3,7 -- ~/envs/vllm/bin/python -m "
                   "pytest -q tests/models/muse_glimmer --ignore=tests/models/muse_glimmer/"
                   "distributed -rfEs",
        "result": "675 passed, 8 skipped, 1 warning in 322.86s",
        "log": f"{RAW}/full_suite_vllm.log.txt",
        "commit_measured": "060d7d4",
        "note": "a first run at 4262415 failed test_vllm_strict_binding::"
                "test_probe_keeps_strict_ops_under_cuda_default_device (host validator built a "
                "helper tensor on the CUDA default device); fixed in 060d7d4 with a regression test",
    },
    "megatron_env": {
        "command": "PYTHONPATH=$WT ~/bin/gpu-lease -n 1 --pool 3,7 -- ~/envs/megatron/bin/python -m "
                   "pytest -q tests/models/muse_glimmer/test_megatron_strict_binding.py "
                   "tests/models/muse_glimmer/test_strict_model_tiny.py -rfEs",
        "result": "35 passed, 2 skipped",
        "log": f"{RAW}/megatron_env_tests.log.txt",
        "commit_measured": "060d7d4",
    },
    "distributed": "NOT RUN in wave B (pool 3,7, one GPU at a time)",
}
commits = {
    "fail_closed_kernel_and_wrappers": "c449df8",
    "norms_golden_and_tests": "320d1d0",
    "compiled_arithmetic_docs": "01f17ae",
    "bench_driver": "d0852cd",
    "reports_and_raw": "4262415",
    "default_device_fix": "060d7d4",
    "bench": "d0852cd (forward kernel code identical to c449df8; later commits change only "
             "comments/docstrings in the kernel file)",
    "decode_overhead": "01f17ae (working tree with the uncommitted bench-driver edits of d0852cd, "
                       "which the probe does not use)",
    "ir": "320d1d0",
    "norms_mismatch": "d0852cd",
    "final_digest": "060d7d4",
}
reqs = open(f"{S}/requests.json").read()
cmd = [sys.executable, f"{S}/update_json.py", W, json.dumps(tests), json.dumps(commits),
       f"{S}/verdicts.json", reqs]
print(subprocess.run(cmd, capture_output=True, text=True))
