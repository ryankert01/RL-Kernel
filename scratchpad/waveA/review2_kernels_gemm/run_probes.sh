#!/bin/bash
# Runs inside one gpu-lease (1 GPU).  Reviewer probes for kernels_gemm.
D=/tmp/claude-1000/-home-ubuntu-RL-Kernel--claude-worktrees-swift-oak-hsv7/80655dc5-1eae-4992-a93b-f6f87e8ab1d8/scratchpad/waveA/review2_kernels_gemm
WT=/home/ubuntu/RL-Kernel/.claude/worktrees/wf_5e0d8071-7a1-5
PY=~/envs/vllm/bin/python
cd $D
echo "GPU: $CUDA_VISIBLE_DEVICES"
PYTHONPATH=$D/before_tree $PY $D/probe_bits_review.py $D/bits_before.json > $D/bits_before.log 2>&1; echo "bits_before $?"
PYTHONPATH=$WT $PY $D/probe_bits_review.py $D/bits_after.json > $D/bits_after.log 2>&1; echo "bits_after $?"
PYTHONPATH=$WT $PY $D/probe_extra.py $D/extra.json > $D/extra.log 2>&1; echo "extra $?"
for i in 1 2; do
  PYTHONPATH=$D/before_tree $PY $D/bench_overhead.py before_$i >> $D/bench.log 2>&1; echo "bench before_$i $?"
  PYTHONPATH=$WT $PY $D/bench_overhead.py after_$i >> $D/bench.log 2>&1; echo "bench after_$i $?"
done
