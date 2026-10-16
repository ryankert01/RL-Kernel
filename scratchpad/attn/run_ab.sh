#!/bin/bash
# one GPU lease: backward before/after alternated in one process
set -u
WT=/home/ubuntu/RL-Kernel/.claude/worktrees/wf_5e0d8071-7a1-1
SP=/tmp/claude-1000/-home-ubuntu-RL-Kernel--claude-worktrees-swift-oak-hsv7/80655dc5-1eae-4992-a93b-f6f87e8ab1d8/scratchpad/attn
PY=/home/ubuntu/envs/vllm/bin/python
cd "$WT"
export PYTHONPATH="$WT"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
echo "== bwd-ab"
$PY -m rl_engine.models.muse_glimmer.ops.gated_attention_bench bwd-ab \
    --old-module "$SP/old_tree/rl_engine/models/muse_glimmer/triton/gated_attention_bwd.py" \
    --iters 10 --rounds 7 --out "$SP/bwd_ab.json"
echo "rc=$?"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
echo "== done"
