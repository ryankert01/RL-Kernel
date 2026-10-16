#!/bin/bash
# runs the small reviewer probes in the vLLM env (each in its own process)
R=/tmp/claude-1000/-home-ubuntu-RL-Kernel--claude-worktrees-swift-oak-hsv7/80655dc5-1eae-4992-a93b-f6f87e8ab1d8/scratchpad/waveA/review2_kernels_rope
export PYTHONPATH=/home/ubuntu/RL-Kernel/.claude/worktrees/wf_5e0d8071-7a1-6
cd /home/ubuntu/RL-Kernel/.claude/worktrees/wf_5e0d8071-7a1-6 || exit 1
~/envs/vllm/bin/python $R/probe_verify_memo.py
~/envs/vllm/bin/python $R/probe_negidx.py neg 2>&1 | tail -2
~/envs/vllm/bin/python $R/probe_negidx.py big 2>&1 | tail -2
