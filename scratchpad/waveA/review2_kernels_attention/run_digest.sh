#!/bin/bash
R=/tmp/claude-1000/-home-ubuntu-RL-Kernel--claude-worktrees-swift-oak-hsv7/80655dc5-1eae-4992-a93b-f6f87e8ab1d8/scratchpad/waveA/review2_kernels_attention
WT=/home/ubuntu/RL-Kernel/.claude/worktrees/wf_5e0d8071-7a1-9
PY=/home/ubuntu/envs/vllm/bin/python
cd $R/drv
echo "== new"; PYTHONPATH=$WT $PY $R/drv/ga_bench_driver.py digest --out $R/digest_new.json; echo rc=$?
echo "== old"; PYTHONPATH=$R/old_tree $PY $R/drv/ga_bench_driver.py digest --out $R/digest_old.json; echo rc=$?
