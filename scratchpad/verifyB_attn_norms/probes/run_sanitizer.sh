#!/bin/bash
# Runs each bad-metadata case under compute-sanitizer memcheck (no caching allocator) and
# reports: exit code, whether the strict assert fired, CHILD-OK, and the memcheck error count.
P=/tmp/claude-1000/-home-ubuntu-RL-Kernel--claude-worktrees-swift-oak-hsv7/80655dc5-1eae-4992-a93b-f6f87e8ab1d8/scratchpad/verifyB_attn_norms/probes
OUT=$P/sanitizer
mkdir -p $OUT
export PYTHONPATH=/home/ubuntu/RL-Kernel/.claude/worktrees/wf_65f5032c-579-9
export PYTORCH_NO_CUDA_MEMORY_CACHING=1
PY=$HOME/envs/vllm/bin/python
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
run() {
  local tag=$1; shift
  compute-sanitizer --tool memcheck --print-limit 5 "$@" > $OUT/$tag.log 2>&1
  local rc=$?
  local inv=$(grep -c "Invalid __global__" $OUT/$tag.log)
  local summ=$(grep -m1 "ERROR SUMMARY" $OUT/$tag.log)
  local asrt=$(grep -c "invalid metadata" $OUT/$tag.log)
  local ok=$(grep -c "CHILD-OK" $OUT/$tag.log)
  local extra=$(grep -m1 -E "LEGAL-BITWISE|RAISED|NO-RAISE|version unchanged" $OUT/$tag.log)
  echo "$tag rc=$rc invalid_access_lines=$inv assert_msg=$asrt child_ok=$ok | $summ | $extra"
}
for c in max_seqlen_short_auto max_seqlen_short_token max_seqlen_short_head max_seqlen_short_paged \
         q_end_beyond_rows k_end_beyond_rows q_pos_beyond_keys q_pos_negative page_minus1_visible \
         page_out_of_range seq_lens_beyond_table seq_lens_below_len_q; do
  w=none
  if [ "$c" = page_minus1_visible ] || [ "$c" = q_pos_negative ]; then w=2048; fi
  run "t_$c" $PY $P/child.py $c $w
done
for c in mid_cu_k_nonmono cu_q_mid_beyond token_qpos_bad_second_tile head_qpos_bad_other_seq \
         paged_neg7_visible_global paged_page_eq_num_pages paged_seq_lens_negative \
         paged_chunk_minus1_visible_token legal_evicted_and_garbage stale_meta_cu_q \
         stale_meta_q_pos autograd_noncontig_qpos; do
  run "x_$c" $PY $P/extra_child.py $c
done
