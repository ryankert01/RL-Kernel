#!/usr/bin/env bash
# Builds feat/museglimmer-poc: a copy of feat/museglimmer-arch-fingerprint (d84be03) whose unpushed
# commits no longer add the fingerprint test to .github/workflows/ci.yml. The original branch is not touched.
set -euo pipefail
cd /home/ubuntu/RL-Kernel/.claude/worktrees/swift-oak-hsv7
git branch feat/museglimmer-poc d84be0375f1215fb2f6ee353f8ea760c86979270
FILTER_BRANCH_SQUELCH_WARNING=1 git filter-branch --index-filter '
e=$(git ls-files -s .github/workflows/ci.yml)
if [ -n "$e" ]; then
  mode=$(echo "$e" | cut -d" " -f1); blob=$(echo "$e" | cut -d" " -f2)
  new=$(git cat-file blob $blob | sed "s# tests/models/muse_glimmer/test_museglimmer_fingerprint.py##" | git hash-object -w --stdin)
  git update-index --cacheinfo $mode,$new,.github/workflows/ci.yml
fi' -- feat/museglimmer-poc --not --remotes=origin
echo "original: $(git rev-parse --short feat/museglimmer-arch-fingerprint)  copy: $(git rev-parse --short feat/museglimmer-poc)"
git diff --stat feat/museglimmer-arch-fingerprint feat/museglimmer-poc
