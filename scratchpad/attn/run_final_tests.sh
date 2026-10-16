#!/bin/bash
# two-GPU lease: every owned attention test at the final code (single-GPU files + CP2 tests)
set -u
WT=/home/ubuntu/RL-Kernel/.claude/worktrees/wf_5e0d8071-7a1-1
PY=/home/ubuntu/envs/vllm/bin/python
cd "$WT"
export PYTHONPATH="$WT"
echo "== single-GPU files"
$PY -m pytest -q --tb=short -p no:cacheprovider -rs \
    tests/models/muse_glimmer/test_gated_attention_fwd.py \
    tests/models/muse_glimmer/test_gated_attention_bwd.py 2>&1 | tail -4
echo "rc=${PIPESTATUS[0]}"
echo "== distributed CP files (2 GPUs visible)"
$PY -m pytest -q --tb=short -p no:cacheprovider -rs \
    tests/models/muse_glimmer/distributed/test_cp_global_attention.py \
    tests/models/muse_glimmer/distributed/test_cp_local_attention.py \
    tests/models/muse_glimmer/distributed/test_cp_tp_topology.py 2>&1 | tail -10
echo "rc=${PIPESTATUS[0]}"
echo "== done"
