#!/bin/bash
# two-GPU lease: real CP2 (torchrun, NCCL) parity matrix with backward, then the sigmoid probe
set -u
WT=/home/ubuntu/RL-Kernel/.claude/worktrees/wf_5e0d8071-7a1-1
SP=/tmp/claude-1000/-home-ubuntu-RL-Kernel--claude-worktrees-swift-oak-hsv7/80655dc5-1eae-4992-a93b-f6f87e8ab1d8/scratchpad/attn
PY=/home/ubuntu/envs/vllm/bin/python
cd "$WT"
export PYTHONPATH="$WT"
echo "== torchrun CP2"
$PY -m torch.distributed.run --nproc_per_node=2 --master_port=29571 \
    rl_engine/models/muse_glimmer/runtime/check_cp_parity.py --backward \
    --lengths 2048 2100 4096 8192 --json "$SP/cp2_torchrun.json"
echo "rc=$?"
echo "== sigmoid probe"
$PY "$SP/probe_sigmoid.py" "$SP/old_tree/rl_engine/models/muse_glimmer/triton/gated_attention_bwd.py" "$SP/probe_sigmoid.json"
echo "rc=$?"
echo "== done"
