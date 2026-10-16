#!/bin/bash
# D2H count probe on head; paired quick decode/prefill bench base -> head -> base -> head.
V=/tmp/claude-1000/-home-ubuntu-RL-Kernel--claude-worktrees-swift-oak-hsv7/80655dc5-1eae-4992-a93b-f6f87e8ab1d8/scratchpad/verifyB_attn_norms
P=$V/probes
BASE=$V/base_tree
HEAD=/home/ubuntu/RL-Kernel/.claude/worktrees/wf_65f5032c-579-9
PY=$HOME/envs/vllm/bin/python
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
PYTHONPATH=$HEAD $PY $P/probe_d2h.py 2>&1 | grep -v Warning | tail -3
cd $V
for i in 1 2; do
  PYTHONPATH=$BASE $PY $BASE/rl_engine/models/muse_glimmer/ops/gated_attention_bench.py bench --quick --iters 30 --rounds 3 --sections decode prefill --out $V/bench_base_$i.json > /dev/null 2>&1
  echo "base bench $i rc=$?"
  PYTHONPATH=$HEAD $PY $HEAD/rl_engine/models/muse_glimmer/ops/gated_attention_bench.py bench --quick --iters 30 --rounds 3 --sections decode prefill --out $V/bench_head_$i.json > /dev/null 2>&1
  echo "head bench $i rc=$?"
done
