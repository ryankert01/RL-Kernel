#!/bin/bash
# one GPU lease: backward before/after back to back, then the full new series
set -u
WT=/home/ubuntu/RL-Kernel/.claude/worktrees/wf_5e0d8071-7a1-1
SP=/tmp/claude-1000/-home-ubuntu-RL-Kernel--claude-worktrees-swift-oak-hsv7/80655dc5-1eae-4992-a93b-f6f87e8ab1d8/scratchpad/attn
PY=/home/ubuntu/envs/vllm/bin/python
cp "$WT/rl_engine/models/muse_glimmer/ops/gated_attention_bench.py" "$SP/drv/ga_bench_driver.py"
cd "$SP/drv"
echo "== old backward"
PYTHONPATH="$SP/old_tree" $PY "$SP/drv/ga_bench_driver.py" bench --sections backward --iters 30 --out "$SP/bench_old_bwd.json"; echo "rc=$?"
cd "$WT"
echo "== new full"
PYTHONPATH="$WT" $PY -m rl_engine.models.muse_glimmer.ops.gated_attention_bench bench --iters 30 --out "$SP/bench_new.json"; echo "rc=$?"
echo "== done"
