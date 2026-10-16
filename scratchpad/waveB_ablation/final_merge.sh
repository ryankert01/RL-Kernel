#!/usr/bin/env bash
# Merge every part in parts/ into reports/muse_glimmer_poc/data/ablation.json (CPU only).
WT=/home/ubuntu/RL-Kernel/.claude/worktrees/wf_65f5032c-579-6
S=/tmp/claude-1000/-home-ubuntu-RL-Kernel--claude-worktrees-swift-oak-hsv7/80655dc5-1eae-4992-a93b-f6f87e8ab1d8/scratchpad/waveB_ablation
cd "$WT" || exit 1
export PYTHONPATH=$WT
CMD="PYTHONPATH=\$WT ~/envs/vllm/bin/python rl_engine/models/muse_glimmer/ablation/run_ablation.py merge --parts \$S/parts/part_*.json --tokenizer ~/models/Muse-Glimmer-30B/tokenizer.json --previous \$S/round1/waveA_ablation.json \$S/round1/parts/part_hf.json --commands-file \$S/commands.txt --out reports/muse_glimmer_poc/data/ablation.json   # merge at $(git rev-parse --short HEAD)"
grep -v "run_ablation.py merge" $S/commands.txt > $S/commands.tmp && mv $S/commands.tmp $S/commands.txt
echo "$CMD" >> $S/commands.txt
"$HOME/envs/vllm/bin/python" rl_engine/models/muse_glimmer/ablation/run_ablation.py merge \
  --parts $S/parts/part_*.json \
  --tokenizer "$HOME/models/Muse-Glimmer-30B/tokenizer.json" \
  --previous $S/round1/waveA_ablation.json $S/round1/parts/part_hf.json \
  --commands-file $S/commands.txt \
  --out reports/muse_glimmer_poc/data/ablation.json > $S/final_merge.log 2>&1
echo "merge rc=$?"
head -1 $S/final_merge.log
"$HOME/envs/vllm/bin/python" -c "import json; d=json.load(open('reports/muse_glimmer_poc/data/ablation.json')); print(d['status'], d['status_missing'], d['commits_all_parts'], d['head_at_merge'], d['all_parts_at_one_clean_commit']); print(d['headline'])"
ls -la reports/muse_glimmer_poc/data/ablation.json
