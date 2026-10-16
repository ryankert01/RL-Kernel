#!/bin/bash
# Four-mode mock matrix for the REPORT (8 steps each), run in parallel with 8 threads each.
set -u
cd /home/ubuntu/RL-Kernel/.claude/worktrees/agent-a7f90c5ca29b03916 || exit 1
OUT=/tmp/claude-1000/-home-ubuntu-RL-Kernel--claude-worktrees-swift-oak-hsv7/80655dc5-1eae-4992-a93b-f6f87e8ab1d8/scratchpad/e2e_runs
mkdir -p "$OUT"
PY=/home/ubuntu/envs/vllm/bin/python
export PYTHONPATH=/home/ubuntu/RL-Kernel/.claude/worktrees/agent-a7f90c5ca29b03916
for m in strict/strict strict/native native/strict native/native; do
  name="mode_${m//\//-}"
  "$PY" -m rl_engine.models.muse_glimmer.e2e.run_poc_grpo --mock --mode "$m" --steps 8 \
    --prompts-per-step 8 --samples-per-prompt 4 --max-tokens 64 --lora-rank 16 --lr 1e-5 \
    --seed 1234 --threads 8 --run-dir "$OUT/$name" > "$OUT/$name.log" 2>&1 &
done
wait
for m in strict-strict strict-native native-strict native-native; do
  echo "== $m"; tail -n 3 "$OUT/mode_$m.log"
done
