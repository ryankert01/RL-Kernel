#!/bin/bash
# Wave B reference runs (one GPU). usage: queue1.sh <runs dir> <worktree>
set -u
R=$1
W=$2
CK=$HOME/models/Muse-Glimmer-30B
PY=$HOME/envs/vllm/bin/python
TOK=reports/muse_glimmer_poc/data/xrt_tokens.json
REF=rl_engine/models/muse_glimmer/reference
mkdir -p "$R"
cd "$W" || exit 1
export PYTHONPATH="$W"
git rev-parse HEAD > "$R/commit_queue1.txt"
git status --porcelain --untracked-files=no > "$R/dirty_queue1.txt"
nvidia-smi --query-gpu=index,name,memory.used --format=csv > "$R/gpu_queue1.txt"
step() {
  local name=$1
  shift
  echo "[$(date +%T)] start $name"
  "$@" > "$R/$name.log" 2>&1
  echo "[$(date +%T)] $name exit $?"
}
step xrt_triton "$PY" $REF/run_chain.py --checkpoint "$CK" --backend triton \
  --score-file $TOK --score-out "$R/xrt_reference_triton.json" \
  --score-check-modes prefill,decode,chunked:prompt_boundary,chunked:block64,chunked:ragged,chunked:window_boundary_2047,chunked:window_boundary_2048,chunked:window_boundary_2049
step xrt_triton_repeat "$PY" $REF/run_chain.py --checkpoint "$CK" --backend triton \
  --score-file $TOK --score-out "$R/xrt_reference_triton_repeat.json" --score-check-modes prefill,decode
step pytest_full_chain_triton env MUSEGLIMMER_CHECKPOINT="$CK" MUSEGLIMMER_BACKEND=triton \
  "$PY" -m pytest -p no:cacheprovider tests/models/muse_glimmer/test_full_model_chain.py -q -s -rA
step xrt_hf "$PY" -m rl_engine.models.muse_glimmer.reference.xrt_scoring --checkpoint "$CK" \
  --score-file $TOK --score-out "$R/xrt_reference_hf_bf16.json"
step chain_triton_512_hf "$PY" $REF/run_chain.py --checkpoint "$CK" --backend triton \
  --prompt-len 512 --completion-len 32 --hf-compare --json-out "$R/chain_triton_512_hf.json"
step chain_triton_2100_hf "$PY" $REF/run_chain.py --checkpoint "$CK" --backend triton \
  --prompt-len 2100 --completion-len 32 --hf-compare --json-out "$R/chain_triton_2100_hf.json"
step profile_strict "$PY" $REF/run_profile.py --target strict --checkpoint "$CK" \
  --json-out "$R/profile_strict.json"
step profile_hf "$PY" $REF/run_profile.py --target hf --checkpoint "$CK" \
  --json-out "$R/profile_hf.json"
echo "[$(date +%T)] queue1 done"
