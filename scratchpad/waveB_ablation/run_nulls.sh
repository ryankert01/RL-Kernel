#!/usr/bin/env bash
# Last null control: the HF baseline on GPU 3 (HF main ran on GPU 7), from the current clean HEAD.
set -u
WT=/home/ubuntu/RL-Kernel/.claude/worktrees/wf_65f5032c-579-6
S=/tmp/claude-1000/-home-ubuntu-RL-Kernel--claude-worktrees-swift-oak-hsv7/80655dc5-1eae-4992-a93b-f6f87e8ab1d8/scratchpad/waveB_ablation
P=$S/parts
cd "$WT" || exit 1
R=rl_engine/models/muse_glimmer/ablation/run_ablation.py
PY="env PYTHONPATH=$WT $HOME/envs/vllm/bin/python"
L="$HOME/bin/gpu-lease -n 1 --timeout 14400"
run() {  # pool name args...
  local pool=$1; shift
  local name=$1; shift
  echo "~/bin/gpu-lease -n 1 --pool $pool --timeout 14400 -- env PYTHONPATH=\$WT ~/envs/vllm/bin/python $R $*   # at $(git rev-parse --short HEAD)" | sed "s#$P#\$S/parts#g" > "$S/pending_command.txt"
  echo "=== $(date +%T) $name (pool $pool)"
  $L --pool "$pool" -- $PY "$R" "$@" > "$S/log_$name.txt" 2>&1
  local rc=$?
  echo "=== $(date +%T) $name rc=$rc"
  tail -2 "$S/log_$name.txt"
}
run 3 hf_baseline_gpu3 hf-baseline --out "$P/part_hf_baseline_gpu3.json"
echo "=== $(date +%T) all done"
