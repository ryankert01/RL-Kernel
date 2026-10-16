#!/bin/bash
# usage: run_pytest.sh <logfile> <env: vllm|megatron> <pytest args...>
W=/home/ubuntu/RL-Kernel/.claude/worktrees/wf_7b064961-36f-6
LOG=$1
ENVNAME=$2
shift 2
cd "$W" || exit 1
export PYTHONPATH="$W"
~/envs/$ENVNAME/bin/python -m pytest -p no:cacheprovider "$@" > "$LOG" 2>&1
rc=$?
echo "pytest exit $rc ($LOG)"
tail -5 "$LOG"
