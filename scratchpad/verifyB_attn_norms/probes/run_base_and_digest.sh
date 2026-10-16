#!/bin/bash
# (1) sensitivity control: the same bad cases on f91e8b6 under memcheck must show invalid accesses
# (2) digests: head driver on the f91e8b6 export and on 54ac035
V=/tmp/claude-1000/-home-ubuntu-RL-Kernel--claude-worktrees-swift-oak-hsv7/80655dc5-1eae-4992-a93b-f6f87e8ab1d8/scratchpad/verifyB_attn_norms
P=$V/probes
BASE=$V/base_tree
HEAD=/home/ubuntu/RL-Kernel/.claude/worktrees/wf_65f5032c-579-9
PY=$HOME/envs/vllm/bin/python
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
mkdir -p $P/sanitizer_base
for c in q_end_beyond_rows k_end_beyond_rows page_out_of_range; do
  PYTHONPATH=$BASE PYTORCH_NO_CUDA_MEMORY_CACHING=1 compute-sanitizer --tool memcheck --print-limit 5 \
    $PY $P/child.py $c none > $P/sanitizer_base/$c.log 2>&1
  echo "BASE $c rc=$? invalid=$(grep -c 'Invalid __global__' $P/sanitizer_base/$c.log) child_ok=$(grep -c CHILD-OK $P/sanitizer_base/$c.log) $(grep -m1 'ERROR SUMMARY' $P/sanitizer_base/$c.log)"
done
cd $V
PYTHONPATH=$BASE $PY $BASE/rl_engine/models/muse_glimmer/ops/gated_attention_bench.py digest --out $V/digest_base_mine.json > $V/digest_base_mine.log 2>&1
echo "base digest rc=$?"
PYTHONPATH=$HEAD $PY $HEAD/rl_engine/models/muse_glimmer/ops/gated_attention_bench.py digest --out $V/digest_head_mine.json > $V/digest_head_mine.log 2>&1
echo "head digest rc=$?"
