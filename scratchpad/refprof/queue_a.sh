#!/bin/bash
# Queue A: triton scoring, triton chain + HF anchor (512 and 2100), strict profile, real-weights pytest.
WT=/home/ubuntu/RL-Kernel/.claude/worktrees/wf_7b064961-36f-8
R=/tmp/claude-1000/-home-ubuntu-RL-Kernel--claude-worktrees-swift-oak-hsv7/80655dc5-1eae-4992-a93b-f6f87e8ab1d8/scratchpad/refprof/runs
PY=/home/ubuntu/envs/vllm/bin/python
CK=/home/ubuntu/models/Muse-Glimmer-30B
LEASE="/home/ubuntu/bin/gpu-lease -n 1 --pool 6-7 --timeout 14400 --"
cd "$WT" || exit 1
MODES="prefill,decode,chunked:prompt_boundary,chunked:window_boundary_2047,chunked:window_boundary_2048,chunked:window_boundary_2049"

echo "=== smoke strict profile $(date)"
$LEASE env PYTHONPATH="$WT" $PY rl_engine/models/muse_glimmer/reference/run_profile.py --target strict --checkpoint $CK --prefill-lens 128 --decode-batches 1,4 --decode-context 200 --repeats 1 --decode-repeats 2 --json-out "$R/smoke_profile_strict.json" > "$R/smoke_profile_strict.log" 2>&1
echo "rc=$?"

echo "=== xrt triton $(date)"
$LEASE env PYTHONPATH="$WT" $PY rl_engine/models/muse_glimmer/reference/run_chain.py \
  --checkpoint $CK --backend triton --score-file reports/muse_glimmer_poc/data/xrt_tokens.json \
  --score-out "$R/xrt_reference_triton.json" --score-check-modes "$MODES" > "$R/xrt_triton.log" 2>&1
echo "rc=$?"

echo "=== chain triton 512+32 hf $(date)"
$LEASE env PYTHONPATH="$WT" $PY rl_engine/models/muse_glimmer/reference/run_chain.py \
  --checkpoint $CK --backend triton --prompt-len 512 --completion-len 32 --hf-compare \
  --json-out "$R/chain_triton_512_hf.json" > "$R/chain_triton_512_hf.log" 2>&1
echo "rc=$?"

echo "=== chain triton 2100+32 hf $(date)"
$LEASE env PYTHONPATH="$WT" $PY rl_engine/models/muse_glimmer/reference/run_chain.py \
  --checkpoint $CK --backend triton --prompt-len 2100 --completion-len 32 --hf-compare \
  --json-out "$R/chain_triton_2100_hf.json" > "$R/chain_triton_2100_hf.log" 2>&1
echo "rc=$?"

echo "=== profile strict $(date)"
$LEASE env PYTHONPATH="$WT" $PY rl_engine/models/muse_glimmer/reference/run_profile.py \
  --target strict --checkpoint $CK --json-out "$R/profile_strict.json" > "$R/profile_strict.log" 2>&1
echo "rc=$?"

echo "=== pytest real weights triton $(date)"
$LEASE env PYTHONPATH="$WT" MUSEGLIMMER_CHECKPOINT=$CK MUSEGLIMMER_BACKEND=triton $PY -m pytest \
  tests/models/muse_glimmer/test_full_model_chain.py -q -s > "$R/pytest_full_chain_triton.log" 2>&1
echo "rc=$?"
echo "=== done $(date)"
