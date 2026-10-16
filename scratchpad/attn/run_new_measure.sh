#!/bin/bash
# one GPU lease: digest (new), bwd-diff (new vs old backward), parity (new)
set -u
WT=/home/ubuntu/RL-Kernel/.claude/worktrees/wf_5e0d8071-7a1-1
SP=/tmp/claude-1000/-home-ubuntu-RL-Kernel--claude-worktrees-swift-oak-hsv7/80655dc5-1eae-4992-a93b-f6f87e8ab1d8/scratchpad/attn
PY=/home/ubuntu/envs/vllm/bin/python
cd "$WT"
export PYTHONPATH="$WT"
echo "== digest new"; $PY -m rl_engine.models.muse_glimmer.ops.gated_attention_bench digest --out $SP/digest_new.json; echo "rc=$?"
echo "== bwd-diff"; $PY -m rl_engine.models.muse_glimmer.ops.gated_attention_bench bwd-diff --old-module $SP/old_tree/rl_engine/models/muse_glimmer/triton/gated_attention_bwd.py --out $SP/bwd_diff.json; echo "rc=$?"
echo "== parity"; $PY -m rl_engine.models.muse_glimmer.ops.gated_attention_bench parity --out $SP/parity.json; echo "rc=$?"
echo "== done"
