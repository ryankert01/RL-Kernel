#!/bin/bash
# usage: final_collect.sh <code_commit>   (status and status line are read from status.txt)
set -eu
S=/tmp/claude-1000/-home-ubuntu-RL-Kernel--claude-worktrees-swift-oak-hsv7/80655dc5-1eae-4992-a93b-f6f87e8ab1d8/scratchpad/vllmB
WT=/home/ubuntu/RL-Kernel/.claude/worktrees/wf_65f5032c-579-2
D=reports/muse_glimmer_poc/data
STATUS=$(sed -n 1p "$S/status.txt")
STATUS_LINE=$(sed -n 2p "$S/status.txt")
cd "$WT"
python3 "$S/make_extra.py" "$S/extra.json" "$1" "$S/tests.json" "$S/tp_sim.json" "$STATUS" "$STATUS_LINE"
PYTHONPATH="$WT" ~/envs/vllm/bin/python -m rl_engine.models.muse_glimmer.runtime.vllm_collect_results \
  --runs "$S/runs" --data "$D" --reference "$S/ref/xrt_reference_train.json" \
  --reference "$S/ref/xrt_reference_prefill.json" --extra "$S/extra.json" --out "$D/vllm_runtime.json" > /dev/null
PYTHONPATH="$WT" ~/envs/vllm/bin/python -m rl_engine.models.muse_glimmer.runtime.vllm_report \
  "$D/vllm_runtime.json" rl_engine/models/muse_glimmer/runtime/REPORT_vllm_runtime_adapter.md > /dev/null
echo collected
