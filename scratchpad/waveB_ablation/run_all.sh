#!/usr/bin/env bash
# Wave B ablation re-run: every part from one clean commit, one GPU per process, sequential.
# Null controls are pinned to the other GPU (factor 8) or the same GPU (factor 9) as the main part.
set -u
WT=/home/ubuntu/RL-Kernel/.claude/worktrees/wf_65f5032c-579-6
S=/tmp/claude-1000/-home-ubuntu-RL-Kernel--claude-worktrees-swift-oak-hsv7/80655dc5-1eae-4992-a93b-f6f87e8ab1d8/scratchpad/waveB_ablation
P=$S/parts
mkdir -p "$P"
cd "$WT" || exit 1
R=rl_engine/models/muse_glimmer/ablation/run_ablation.py
PY="env PYTHONPATH=$WT $HOME/envs/vllm/bin/python"
L="$HOME/bin/gpu-lease -n 1 --timeout 7200"
: > "$S/commands.txt"
echo "# All run from the worktree root WT=$WT at commit $(git rev-parse --short HEAD); S=<session scratchpad>/waveB_ablation" >> "$S/commands.txt"
run() {  # pool name args...
  local pool=$1; shift
  local name=$1; shift
  echo "~/bin/gpu-lease -n 1 --pool $pool --timeout 7200 -- env PYTHONPATH=\$WT ~/envs/vllm/bin/python $R $*" | sed "s#$P#\$S/parts#g" >> "$S/commands.txt"
  echo "=== $(date +%T) $name (pool $pool)"
  $L --pool "$pool" -- $PY "$R" "$@" > "$S/log_$name.txt" 2>&1
  local rc=$?
  echo "=== $(date +%T) $name rc=$rc"
  tail -2 "$S/log_$name.txt"
}
lease_of() { $HOME/envs/vllm/bin/python -c "import json,sys; print(json.load(open(sys.argv[1]))['env']['gpu_lease'])" "$1"; }
other() { if [ "$1" = "3" ]; then echo 7; else echo 3; fi; }

run 3,7 strict strict --out "$P/part_strict.json"
G=$(lease_of "$P/part_strict.json")
run "$(other "$G")" strict_gpu$(other "$G") strict --settings repeat_immediate --out "$P/part_strict_gpu$(other "$G").json"
run 3,7 hf hf --out "$P/part_hf.json"
G=$(lease_of "$P/part_hf.json")
run "$(other "$G")" hf_baseline_gpu$(other "$G") hf-baseline --out "$P/part_hf_baseline_gpu$(other "$G").json"
run "$G" hf_baseline_gpu$G hf-baseline --out "$P/part_hf_baseline_gpu$G.json"
run 3,7 hf_layers hf-layers --out "$P/part_hf_layers.json"
echo "=== $(date +%T) all done"
