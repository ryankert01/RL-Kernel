#!/usr/bin/env bash
# Verifier: GPU bridge tests at b3e5066 (run under ~/bin/gpu-lease).
W=/home/ubuntu/RL-Kernel/.claude/worktrees/wf_65f5032c-579-7
cd "$W" || exit 3
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
PYTHONPATH="$W" timeout 900 "$HOME/envs/vllm/bin/python" -m pytest \
  tests/models/muse_glimmer/test_e2e_bridge_gpu.py -q -p no:cacheprovider -rA
echo "exit=$?"
