#!/usr/bin/env bash
# Rest of the wave B ablation re-run (strict main already done on GPU 3). Takes whichever pool
# GPU frees first; null controls are placed relative to the main part's GPU. Sequential, one GPU
# per process, commit unchanged.
set -u
WT=/home/ubuntu/RL-Kernel/.claude/worktrees/wf_65f5032c-579-6
S=/tmp/claude-1000/-home-ubuntu-RL-Kernel--claude-worktrees-swift-oak-hsv7/80655dc5-1eae-4992-a93b-f6f87e8ab1d8/scratchpad/waveB_ablation
P=$S/parts
cd "$WT" || exit 1
R=rl_engine/models/muse_glimmer/ablation/run_ablation.py
PY="env PYTHONPATH=$WT $HOME/envs/vllm/bin/python"
L="$HOME/bin/gpu-lease -n 1 --timeout 14400"
# keep only the lines of parts that actually ran (strict main); the rest are re-added on run
grep -v "strict --settings repeat_immediate\| hf \| hf-baseline \| hf-layers " "$S/commands.txt" > "$S/commands.tmp" && mv "$S/commands.tmp" "$S/commands.txt"
run() {  # pool name args...
  local pool=$1; shift
  local name=$1; shift
  echo "~/bin/gpu-lease -n 1 --pool $pool --timeout 14400 -- env PYTHONPATH=\$WT ~/envs/vllm/bin/python $R $*" | sed "s#$P#\$S/parts#g" >> "$S/commands.txt"
  echo "=== $(date +%T) $name (pool $pool)"
  $L --pool "$pool" -- $PY "$R" "$@" > "$S/log_$name.txt" 2>&1
  local rc=$?
  echo "=== $(date +%T) $name rc=$rc"
  tail -2 "$S/log_$name.txt"
}
lease_of() { $HOME/envs/vllm/bin/python -c "import json,sys; print(json.load(open(sys.argv[1]))['env']['gpu_lease'])" "$1"; }
other() { if [ "$1" = "3" ]; then echo 7; else echo 3; fi; }
run 3,7 hf hf --out "$P/part_hf.json"
G=$(lease_of "$P/part_hf.json")
run "$G" hf_baseline_gpu$G hf-baseline --out "$P/part_hf_baseline_gpu$G.json"
run 3,7 hf_layers hf-layers --out "$P/part_hf_layers.json"
O=$(other "$G")
run "$O" hf_baseline_gpu$O hf-baseline --out "$P/part_hf_baseline_gpu$O.json"
SG=$(lease_of "$P/part_strict.json")
SO=$(other "$SG")
run "$SO" strict_gpu$SO strict --settings repeat_immediate --out "$P/part_strict_gpu$SO.json"
echo "=== $(date +%T) all done"
