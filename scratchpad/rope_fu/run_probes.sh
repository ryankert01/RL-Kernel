#!/bin/bash
# usage: run_probes.sh <outdir> <probe...>   (probe names: f1 f2 f3 f4 f5 xenv xenv_megatron)
S=/tmp/claude-1000/-home-ubuntu-RL-Kernel--claude-worktrees-swift-oak-hsv7/80655dc5-1eae-4992-a93b-f6f87e8ab1d8/scratchpad/rope_fu
W=/home/ubuntu/RL-Kernel/.claude/worktrees/wf_7b064961-36f-6
OUT=$1
shift
mkdir -p "$OUT"
cd "$W" || exit 1
export PYTHONPATH="$W"
for p in "$@"; do
  if [ "$p" = "xenv_megatron" ]; then
    ~/envs/megatron/bin/python "$S/probe_findings.py" xenv "$OUT/$p.json" > "$OUT/$p.log" 2>&1
  else
    ~/envs/vllm/bin/python "$S/probe_findings.py" "$p" "$OUT/$p.json" > "$OUT/$p.log" 2>&1
  fi
  echo "$p exit $?"
done
