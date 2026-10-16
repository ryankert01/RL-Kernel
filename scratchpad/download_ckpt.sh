#!/bin/bash
# Resumable download of meta-models/Muse-Glimmer-30B at the pinned revision.
set -u
R=a4e59da52a7bc87ae7251dd5545c0dd437c44b68
OUT=$HOME/models/Muse-Glimmer-30B
TOK=$(cat $HOME/.cache/huggingface/token)
cd "$OUT" || exit 1
FILES="config.json generation_config.json model.safetensors.index.json processor_config.json tokenizer.json tokenizer_config.json chat_template.jinja README.md LICENSE USAGE_POLICY.md model-00001-of-00002.safetensors model-00002-of-00002.safetensors"
for f in $FILES; do
  for attempt in 1 2 3 4 5; do
    curl -sS -L -C - --retry 5 --retry-delay 5 -m 7200 -H "Authorization: Bearer $TOK" \
      -o "$f" "https://huggingface.co/meta-models/Muse-Glimmer-30B/resolve/$R/$f" && break
    echo "retry $attempt for $f" >&2; sleep 10
  done
done
echo "$R" > REVISION
echo "---- sha256 of shards ----"
sha256sum model-00001-of-00002.safetensors model-00002-of-00002.safetensors
ls -la
