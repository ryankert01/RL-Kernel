#!/bin/bash
# usage: run_script.sh <env> <script.py> <logfile> [args...]   (runs from the worktree root)
W=/home/ubuntu/RL-Kernel/.claude/worktrees/wf_7b064961-36f-6
ENVNAME=$1
SCRIPT=$2
LOG=$3
shift 3
cd "$W" || exit 1
export PYTHONPATH="$W"
~/envs/$ENVNAME/bin/python "$SCRIPT" "$@" > "$LOG" 2>&1
echo "exit $? ($LOG)"
