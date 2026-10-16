#!/bin/bash
# All post-fix measurements at the committed HEAD, in one lease (one GPU).
S=/tmp/claude-1000/-home-ubuntu-RL-Kernel--claude-worktrees-swift-oak-hsv7/80655dc5-1eae-4992-a93b-f6f87e8ab1d8/scratchpad/rope_fu
F=$S/final
mkdir -p $F
git -C /home/ubuntu/RL-Kernel/.claude/worktrees/wf_7b064961-36f-6 rev-parse HEAD > $F/commit.txt
$S/run_pytest.sh $F/pytest_qknorm_rope_vllm.log vllm tests/models/muse_glimmer/test_qknorm_rope.py -q -s -rA
$S/run_pytest.sh $F/pytest_qknorm_rope_megatron.log megatron tests/models/muse_glimmer/test_qknorm_rope.py -q -s -rA
$S/run_probes.sh $F f1 f2 f3 f4 xenv xenv_megatron f5 f5c
$S/run_script.sh vllm $S/overhead_probe.py $F/overhead.log $F/overhead.json
$S/run_bench.sh final $F
