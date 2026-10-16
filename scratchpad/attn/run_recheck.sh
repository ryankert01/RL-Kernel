#!/bin/bash
# one GPU lease: owned single-GPU tests, digest (new, after the per-KV-head fold), parity
set -u
WT=/home/ubuntu/RL-Kernel/.claude/worktrees/wf_5e0d8071-7a1-1
SP=/tmp/claude-1000/-home-ubuntu-RL-Kernel--claude-worktrees-swift-oak-hsv7/80655dc5-1eae-4992-a93b-f6f87e8ab1d8/scratchpad/attn
PY=/home/ubuntu/envs/vllm/bin/python
cd "$WT"
export PYTHONPATH="$WT"
echo "== tests"
$PY -m pytest -q --tb=short -p no:cacheprovider \
    tests/models/muse_glimmer/test_gated_attention_fwd.py \
    tests/models/muse_glimmer/test_gated_attention_bwd.py \
    tests/models/muse_glimmer/distributed/test_cp_global_attention.py \
    tests/models/muse_glimmer/distributed/test_cp_local_attention.py \
    tests/models/muse_glimmer/distributed/test_cp_tp_topology.py -rs 2>&1 | tail -15
echo "rc=${PIPESTATUS[0]}"
echo "== digest new2"; $PY -m rl_engine.models.muse_glimmer.ops.gated_attention_bench digest --out $SP/digest_new2.json; echo "rc=$?"
echo "== parity"; $PY -m rl_engine.models.muse_glimmer.ops.gated_attention_bench parity --out $SP/parity.json; echo "rc=$?"
echo "== done"
