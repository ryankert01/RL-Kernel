#!/bin/bash
# one GPU lease: the full new timing series again (the first run overlapped another process)
set -u
WT=/home/ubuntu/RL-Kernel/.claude/worktrees/wf_5e0d8071-7a1-1
SP=/tmp/claude-1000/-home-ubuntu-RL-Kernel--claude-worktrees-swift-oak-hsv7/80655dc5-1eae-4992-a93b-f6f87e8ab1d8/scratchpad/attn
PY=/home/ubuntu/envs/vllm/bin/python
cd "$WT"
export PYTHONPATH="$WT"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
echo "== new full"
$PY -m rl_engine.models.muse_glimmer.ops.gated_attention_bench bench --iters 30 --out "$SP/bench_new2.json"
echo "rc=$?"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
echo "== done"
