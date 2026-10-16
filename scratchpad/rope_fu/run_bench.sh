#!/bin/bash
# usage: run_bench.sh <label> <outdir>   -> series JSON + the repo bench script output
S=/tmp/claude-1000/-home-ubuntu-RL-Kernel--claude-worktrees-swift-oak-hsv7/80655dc5-1eae-4992-a93b-f6f87e8ab1d8/scratchpad/rope_fu
W=/home/ubuntu/RL-Kernel/.claude/worktrees/wf_7b064961-36f-6
LABEL=$1
OUT=$2
mkdir -p "$OUT"
cd "$W" || exit 1
export PYTHONPATH="$W"
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv > "$OUT/gpu_state_$LABEL.txt"
~/envs/vllm/bin/python "$S/bench_series.py" "$LABEL" "$OUT/bench_series_$LABEL.json" > "$OUT/bench_series_$LABEL.log" 2>&1
echo "bench_series exit $?"
~/envs/vllm/bin/python benchmarks/models/muse_glimmer/bench_qknorm_rope.py > "$OUT/bench_repo_$LABEL.log" 2>&1
echo "bench_repo exit $?"
