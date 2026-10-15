#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
#
# Reproduce the qknorm_scaled_rope (O4) review measurements into <out_dir> on ONE leased GPU.
#   usage: ~/bin/gpu-lease -n 1 -- reports/muse_glimmer_poc/scripts/rope/run_rope_measurements.sh \
#              <out_dir> [<tree>]            # <tree>: code under test (default: this checkout)
# Before/after: run once with <tree> = an old tree (git archive <sha> | tar -x -C <old>) and once
# with the current checkout. Outputs: probes (wave A f1..f5c, xenv in both envs; wave B), the
# eager series and repo benchmarks, the CUDA-graph benchmark (old 6a61f6e vs current kernel),
# the CPU overhead probe and the pytest logs of test_qknorm_rope.py in both envs. Assemble with
# build_json.py (wave A) and update_json_wave_b.py (wave B).
set -u
OUT=$1
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../../../.." && pwd)
TREE=${2:-$REPO}
VPY=$HOME/envs/vllm/bin/python
MPY=$HOME/envs/megatron/bin/python
mkdir -p "$OUT"
cd "$TREE" || exit 1
export PYTHONPATH="$TREE"
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv > "$OUT/gpu_state.txt"
run() {
  local name=$1
  shift
  "$@" > "$OUT/$name.log" 2>&1
  echo "$name exit $?"
}
for p in f1 f2 f3 f4 xenv f5 f5c; do
  run "$p" "$VPY" "$HERE/probe_findings.py" "$p" "$OUT/$p.json"
done
run xenv_megatron "$MPY" "$HERE/probe_findings.py" xenv "$OUT/xenv_megatron.json"
run probe_wave_b "$VPY" "$HERE/probe_wave_b.py" "$OUT/probe_wave_b.json"
run bench_series "$VPY" "$HERE/bench_series.py" series "$OUT/bench_series.json"
run bench_repo "$VPY" "$REPO/benchmarks/models/muse_glimmer/bench_qknorm_rope.py"
(cd "$REPO" && git show 6a61f6e:rl_engine/models/muse_glimmer/triton/qknorm_rope.py) \
  > "$OUT/old_triton_qknorm_rope_6a61f6e.py"
run bench_graph "$VPY" "$HERE/bench_graph.py" "$OUT/bench_graph.json" \
  --old-kernel "$OUT/old_triton_qknorm_rope_6a61f6e.py"
run overhead "$VPY" "$HERE/overhead_probe.py" "$OUT/overhead.json"
run pytest_qknorm_rope_vllm "$VPY" -m pytest -p no:cacheprovider \
  tests/models/muse_glimmer/test_qknorm_rope.py -q -rA
run pytest_qknorm_rope_megatron "$MPY" -m pytest -p no:cacheprovider \
  tests/models/muse_glimmer/test_qknorm_rope.py -q -rA
