#!/bin/bash
# usage: phase.sh <phase>   (runs the listed run_vllm_strict invocations; see cases)
set -u
S=/tmp/claude-1000/-home-ubuntu-RL-Kernel--claude-worktrees-swift-oak-hsv7/80655dc5-1eae-4992-a93b-f6f87e8ab1d8/scratchpad/vllmB
D=reports/muse_glimmer_poc/data
X="--prompt-file $D/xrt_tokens.json"
case "$1" in
  native_tp1)
    $S/run.sh 1 4-6 native_tp1 --tp 1 --native $X --temperatures 0,1 --max-tokens 64 --prefill-check \
      --xrt-score $D/xrt_tokens.json --xrt-out $D/xrt_vllm_native_tp1.json --max-num-seqs 256 \
      --decode-batches 256 --out $S/runs/native_tp1.json ;;
  native_tp2)
    $S/run.sh 2 4-6 native_tp2 --tp 2 --native $X --temperatures 0,1 --max-tokens 64 --prefill-check \
      --xrt-score $D/xrt_tokens.json --xrt-out $D/xrt_vllm_native_tp2.json --max-num-seqs 256 \
      --decode-batches 256 --out $S/runs/native_tp2.json ;;
  native_lora_tp1)
    $S/run.sh 1 4-6 native_lora_tp1 --tp 1 --native --native-lora $X --temperatures 0,1 --max-tokens 64 \
      --prefill-check --lora-random-check --xrt-score $D/xrt_tokens.json \
      --xrt-lora-out $D/xrt_vllm_native_lora_tp1.json --out $S/runs/native_lora_tp1.json ;;
  chunked_strict_tp1)
    $S/run.sh 1 4-6 strict_tp1_chunked --tp 1 --strict $X --no-generate --max-num-batched-tokens 512 \
      --max-num-seqs 256 --long-window --decode-batches 64 --contract-lora --count-null-pages --out $S/runs/strict_tp1_chunked.json ;;
  chunked_strict_tp2)
    $S/run.sh 2 4-6 strict_tp2_chunked --tp 2 --strict $X --no-generate --max-num-batched-tokens 512 \
      --max-num-seqs 256 --long-window --decode-batches 64 --contract-lora --count-null-pages --out $S/runs/strict_tp2_chunked.json ;;
  chunked_native_tp1)
    $S/run.sh 1 4-6 native_tp1_chunked --tp 1 --native $X --no-generate --max-num-batched-tokens 512 \
      --max-num-seqs 256 --long-window --decode-batches 64 --out $S/runs/native_tp1_chunked.json ;;
  chunked_native_tp2)
    $S/run.sh 2 4-6 native_tp2_chunked --tp 2 --native $X --no-generate --max-num-batched-tokens 512 \
      --max-num-seqs 256 --long-window --decode-batches 64 --out $S/runs/native_tp2_chunked.json ;;
  tightkv_strict_tp1)
    $S/run.sh 1 4-6 strict_tp1_tightkv --tp 1 --strict $X --no-generate --max-num-batched-tokens 512 \
      --max-num-seqs 256 --long-window --long-window-copies 8 --decode-batches 64 --contract-tag _tightkv \
      --num-gpu-blocks-override 1600 --count-null-pages --out $S/runs/strict_tp1_tightkv.json ;;
  tightkv_native_tp1)
    $S/run.sh 1 4-6 native_tp1_tightkv --tp 1 --native $X --no-generate --max-num-batched-tokens 512 \
      --max-num-seqs 256 --long-window --long-window-copies 8 --decode-batches 64 --contract-tag _tightkv \
      --num-gpu-blocks-override 1600 --out $S/runs/native_tp1_tightkv.json ;;
  bench)
    # isolated benches, one engine at a time
    $S/run.sh 1 4-6 bench_strict_tp1 --tp 1 --strict $X --no-generate --bench --bench-lora --profile \
      --out $S/runs/bench_strict_tp1.json
    $S/run.sh 2 4-6 bench_strict_tp2 --tp 2 --strict $X --no-generate --bench --bench-lora --profile \
      --out $S/runs/bench_strict_tp2.json
    $S/run.sh 1 4-6 bench_native_tp1 --tp 1 --native $X --no-generate --bench --profile \
      --out $S/runs/bench_native_tp1.json
    $S/run.sh 1 4-6 bench_native_cg_tp1 --tp 1 --native --cuda-graphs $X --no-generate --bench \
      --out $S/runs/bench_native_cg_tp1.json
    $S/run.sh 2 4-6 bench_native_tp2 --tp 2 --native $X --no-generate --bench --profile \
      --out $S/runs/bench_native_tp2.json
    $S/run.sh 2 4-6 bench_native_cg_tp2 --tp 2 --native --cuda-graphs $X --no-generate --bench \
      --out $S/runs/bench_native_cg_tp2.json ;;
  bench_cg)
    $S/run.sh 1 4-6 bench_native_cg_tp1 --tp 1 --native --cuda-graphs $X --no-generate --bench \
      --out $S/runs/bench_native_cg_tp1.json
    $S/run.sh 2 4-6 bench_native_cg_tp2 --tp 2 --native --cuda-graphs $X --no-generate --bench \
      --out $S/runs/bench_native_cg_tp2.json ;;
  tp4)
    $S/run.sh 4 4-7 strict_tp4 --tp 4 --strict $X --temperatures 0,1 --max-tokens 64 --prefill-check \
      --lora-random-check --xrt-score $D/xrt_tokens.json --xrt-out $D/xrt_vllm_strict_tp4.json \
      --xrt-lora-out $D/xrt_vllm_strict_lora_tp4.json --max-num-seqs 256 --decode-batches 256 \
      --out $S/runs/strict_tp4.json
    $S/run.sh 4 4-7 strict_tp4_chunked --tp 4 --strict $X --no-generate --max-num-batched-tokens 512 \
      --max-num-seqs 256 --long-window --decode-batches 64 --contract-lora --count-null-pages --out $S/runs/strict_tp4_chunked.json
    $S/run.sh 4 4-7 bench_strict_tp4 --tp 4 --strict $X --no-generate --bench --profile \
      --out $S/runs/bench_strict_tp4.json
    $S/run.sh 4 4-7 bench_native_tp4 --tp 4 --native $X --no-generate --bench \
      --out $S/runs/bench_native_tp4.json
    $S/run.sh 4 4-7 bench_native_cg_tp4 --tp 4 --native --cuda-graphs $X --no-generate --bench \
      --out $S/runs/bench_native_cg_tp4.json ;;
  *) echo "unknown phase $1"; exit 2 ;;
esac
