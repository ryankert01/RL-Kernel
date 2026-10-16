#!/usr/bin/env bash
# Verifier: CPU e2e tests at b3e5066 (read-only worktree).
W=/home/ubuntu/RL-Kernel/.claude/worktrees/wf_65f5032c-579-7
V=/tmp/claude-1000/-home-ubuntu-RL-Kernel--claude-worktrees-swift-oak-hsv7/80655dc5-1eae-4992-a93b-f6f87e8ab1d8/scratchpad/verifyB_e2e
cd "$W" || exit 3
uptime > "$V/cpu_tests.log"
git -C "$W" rev-parse HEAD >> "$V/cpu_tests.log"
PYTHONPATH="$W" timeout 1800 "$HOME/envs/vllm/bin/python" -m pytest \
  tests/models/muse_glimmer/test_e2e_bridge.py tests/models/muse_glimmer/test_e2e_loop_mock.py \
  -q -p no:cacheprovider -rA --durations=15 >> "$V/cpu_tests.log" 2>&1
echo "exit=$?" >> "$V/cpu_tests.log"
