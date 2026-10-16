#!/bin/bash
# Wave B rope measurements + golden reference runs (one GPU).
# usage: queue2.sh <out> <worktree> <old tree> [<archive of HEAD for the golden runs>]
set -u
R=$1
W=$2
OLD=$3
ARCH=${4:-}
CK=$HOME/models/Muse-Glimmer-30B
VPY=$HOME/envs/vllm/bin/python
MPY=$HOME/envs/megatron/bin/python
S=$W/reports/muse_glimmer_poc/scripts/rope
TOK=reports/muse_glimmer_poc/data/xrt_tokens.json
REF=rl_engine/models/muse_glimmer/reference
mkdir -p "$R"
step() {
  local name=$1
  shift
  echo "[$(date +%T)] start $name"
  "$@" > "$R/$name.log" 2>&1
  echo "[$(date +%T)] $name exit $?"
}
nvidia-smi --query-gpu=index,name,memory.used --format=csv > "$R/gpu_queue2.txt"
cd "$OLD" || exit 1
step probe_wave_b_before env PYTHONPATH="$OLD" "$VPY" "$S/probe_wave_b.py" "$R/probe_wave_b_before.json"
cd "$W" || exit 1
git rev-parse HEAD > "$R/commit_queue2.txt"
git status --porcelain --untracked-files=no > "$R/dirty_queue2.txt"
step probe_wave_b_after env PYTHONPATH="$W" "$VPY" "$S/probe_wave_b.py" "$R/probe_wave_b_after.json"
git show 6a61f6e:rl_engine/models/muse_glimmer/triton/qknorm_rope.py > "$R/old_triton_qknorm_rope_6a61f6e.py"
step bench_graph env PYTHONPATH="$W" "$VPY" "$S/bench_graph.py" "$R/bench_graph.json" \
  --old-kernel "$R/old_triton_qknorm_rope_6a61f6e.py"
step pytest_after_vllm env PYTHONPATH="$W" "$VPY" -m pytest -p no:cacheprovider \
  tests/models/muse_glimmer/test_qknorm_rope.py tests/models/muse_glimmer/test_strict_model_tiny.py -q -rA
step pytest_after_megatron env PYTHONPATH="$W" "$MPY" -m pytest -p no:cacheprovider \
  tests/models/muse_glimmer/test_qknorm_rope.py tests/models/muse_glimmer/test_megatron_strict_binding.py \
  tests/models/muse_glimmer/test_strict_model_tiny.py -q -rA
if [ -n "$ARCH" ]; then
  cd "$ARCH" || exit 1
  step xrt_golden env PYTHONPATH="$ARCH" "$VPY" $REF/run_chain.py --checkpoint "$CK" --backend golden \
    --score-file $TOK --score-out "$R/xrt_reference_golden.json" \
    --score-check-modes prefill,decode,chunked:prompt_boundary,chunked:window_boundary_2048
  step chain_golden_2100 env PYTHONPATH="$ARCH" "$VPY" $REF/run_chain.py --checkpoint "$CK" \
    --backend golden --prompt-len 2100 --completion-len 32 --json-out "$R/chain_golden_2100.json"
fi
echo "[$(date +%T)] queue2 done"
