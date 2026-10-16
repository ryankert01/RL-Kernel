#!/usr/bin/env bash
W=/home/ubuntu/RL-Kernel/.claude/worktrees/wf_65f5032c-579-7
V=/tmp/claude-1000/-home-ubuntu-RL-Kernel--claude-worktrees-swift-oak-hsv7/80655dc5-1eae-4992-a93b-f6f87e8ab1d8/scratchpad/verifyB_e2e
cd "$W" || exit 3
CUDA_VISIBLE_DEVICES_UNSET=1 PYTHONPATH="$W" timeout 900 "$HOME/envs/vllm/bin/python" "$V/probe_bridge.py" "$V/probe_work"
echo "exit=$?"
