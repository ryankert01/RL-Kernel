#!/bin/bash
# usage: run.sh <n_gpus> <pool> <logname> <run_vllm_strict args...>
# Runs one run_vllm_strict invocation under a GPU lease from the worktree root.
set -u
N=$1; POOL=$2; LOG=$3; shift 3
WT=/home/ubuntu/RL-Kernel/.claude/worktrees/wf_65f5032c-579-2
SCR=/tmp/claude-1000/-home-ubuntu-RL-Kernel--claude-worktrees-swift-oak-hsv7/80655dc5-1eae-4992-a93b-f6f87e8ab1d8/scratchpad/vllmB
cd "$WT" || exit 2
CONTIG=""
if [ "$N" -gt 1 ]; then CONTIG="--contiguous"; fi
~/bin/gpu-lease -n "$N" --pool "$POOL" $CONTIG --timeout 14400 -- \
  env PATH="$HOME/envs/vllm/bin:$PATH" PYTHONPATH="$WT" ~/envs/vllm/bin/python -m rl_engine.models.muse_glimmer.runtime.run_vllm_strict "$@" \
  > "$SCR/logs/$LOG.log" 2>&1
echo "exit=$?" >> "$SCR/logs/$LOG.log"
tail -40 "$SCR/logs/$LOG.log"
