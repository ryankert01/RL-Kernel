#!/usr/bin/env bash
# Merge every part in parts/ into the scratchpad (CPU only), then summarize.
# usage: dryrun_merge.sh OUT.json [summarize filter]
WT=/home/ubuntu/RL-Kernel/.claude/worktrees/wf_65f5032c-579-6
S=/tmp/claude-1000/-home-ubuntu-RL-Kernel--claude-worktrees-swift-oak-hsv7/80655dc5-1eae-4992-a93b-f6f87e8ab1d8/scratchpad/waveB_ablation
OUT=$1; shift
cd "$WT" || exit 1
export PYTHONPATH=$WT
"$HOME/envs/vllm/bin/python" rl_engine/models/muse_glimmer/ablation/run_ablation.py merge \
  --parts $S/parts/part_*.json \
  --tokenizer "$HOME/models/Muse-Glimmer-30B/tokenizer.json" \
  --previous $S/round1/waveA_ablation.json $S/round1/parts/part_hf.json \
  --commands-file $S/commands.txt \
  --out "$OUT" > "$OUT.log" 2>&1
echo "merge rc=$?"
head -1 "$OUT.log"
"$HOME/envs/vllm/bin/python" $S/summarize.py "$OUT" "$@"
"$HOME/envs/vllm/bin/python" -c "import json,sys; d=json.load(open(sys.argv[1])); print(json.dumps(d['previous_runs'], indent=None)[:3000])" "$OUT"
