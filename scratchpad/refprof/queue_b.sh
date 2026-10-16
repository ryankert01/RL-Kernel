#!/bin/bash
# Queue B: golden scoring, HF profile, golden window chain.
WT=/home/ubuntu/RL-Kernel/.claude/worktrees/wf_7b064961-36f-8
R=/tmp/claude-1000/-home-ubuntu-RL-Kernel--claude-worktrees-swift-oak-hsv7/80655dc5-1eae-4992-a93b-f6f87e8ab1d8/scratchpad/refprof/runs
PY=/home/ubuntu/envs/vllm/bin/python
CK=/home/ubuntu/models/Muse-Glimmer-30B
LEASE="/home/ubuntu/bin/gpu-lease -n 1 --pool 6-7 --timeout 14400 --"
cd "$WT" || exit 1
MODES="prefill,decode,chunked:prompt_boundary,chunked:window_boundary_2048"

echo "=== smoke hf profile $(date)"
$LEASE env PYTHONPATH="$WT" $PY rl_engine/models/muse_glimmer/reference/run_profile.py --target hf --checkpoint $CK --prefill-lens 128 --decode-batches 1,4 --decode-context 200 --repeats 1 --decode-repeats 2 --json-out "$R/smoke_profile_hf.json" > "$R/smoke_profile_hf.log" 2>&1
echo "rc=$?"

echo "=== xrt golden $(date)"
$LEASE env PYTHONPATH="$WT" $PY rl_engine/models/muse_glimmer/reference/run_chain.py \
  --checkpoint $CK --backend golden --score-file reports/muse_glimmer_poc/data/xrt_tokens.json \
  --score-out "$R/xrt_reference_golden.json" --score-check-modes "$MODES" > "$R/xrt_golden.log" 2>&1
echo "rc=$?"

echo "=== profile hf $(date)"
$LEASE env PYTHONPATH="$WT" $PY rl_engine/models/muse_glimmer/reference/run_profile.py \
  --target hf --checkpoint $CK --json-out "$R/profile_hf.json" > "$R/profile_hf.log" 2>&1
echo "rc=$?"

echo "=== chain golden 2100+32 $(date)"
$LEASE env PYTHONPATH="$WT" $PY rl_engine/models/muse_glimmer/reference/run_chain.py \
  --checkpoint $CK --backend golden --prompt-len 2100 --completion-len 32 \
  --json-out "$R/chain_golden_2100.json" > "$R/chain_golden_2100.log" 2>&1
echo "rc=$?"
echo "=== done $(date)"
