#!/bin/bash
# usage: lease.sh <n_gpus> <pool> <logname> <command...>
# Runs a command from the worktree root under a GPU lease with PYTHONPATH=worktree.
set -u
N=$1; POOL=$2; LOG=$3; shift 3
WT=/home/ubuntu/RL-Kernel/.claude/worktrees/wf_65f5032c-579-2
SCR=/tmp/claude-1000/-home-ubuntu-RL-Kernel--claude-worktrees-swift-oak-hsv7/80655dc5-1eae-4992-a93b-f6f87e8ab1d8/scratchpad/vllmB
cd "$WT" || exit 2
CONTIG=""
if [ "$N" -gt 1 ]; then CONTIG="--contiguous"; fi
~/bin/gpu-lease -n "$N" --pool "$POOL" $CONTIG --timeout 14400 -- \
  env PYTHONPATH="${PYTHONPATH_OVERRIDE:-$WT}" "$@" > "$SCR/logs/$LOG.log" 2>&1
echo "exit=$?" >> "$SCR/logs/$LOG.log"
tail -15 "$SCR/logs/$LOG.log"
