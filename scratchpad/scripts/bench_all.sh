#!/usr/bin/env bash
# Isolated throughput/profile runs (one at a time, nothing else on the host GPUs 0-1).
cd /home/ubuntu/RL-Kernel/.claude/worktrees/wf_7b064961-36f-1 || exit 1
export PYTHONPATH=/home/ubuntu/RL-Kernel/.claude/worktrees/wf_7b064961-36f-1
OUT=/tmp/claude-1000/-home-ubuntu-RL-Kernel--claude-worktrees-swift-oak-hsv7/80655dc5-1eae-4992-a93b-f6f87e8ab1d8/scratchpad/vllm_bench
mkdir -p "$OUT"
for cfg in "1 --native" "1 --strict" "2 --native" "2 --strict"; do
  set -- $cfg
  tp=$1; mode=${2#--}
  n=$tp
  extra=""
  if [ "$tp" = "2" ]; then extra="--contiguous"; fi
  ~/bin/gpu-lease -n "$n" --pool 0-1 $extra --timeout 3600 -- ~/envs/vllm/bin/python -m \
    rl_engine.models.muse_glimmer.runtime.run_vllm_strict --tp "$tp" "--$mode" \
    --prompt-file reports/muse_glimmer_poc/data/xrt_tokens.json --no-generate --bench --profile \
    --out "$OUT/${mode}_tp${tp}_bench.json" > "$OUT/${mode}_tp${tp}_bench.log" 2>&1
  echo "$mode tp$tp exit=$?"
done
echo ALL_DONE
