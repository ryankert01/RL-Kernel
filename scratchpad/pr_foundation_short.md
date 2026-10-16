## Summary

Foundation row of RFC #501 (`museglimmer_arch_fingerprint`). It pins the checkpoint and the model semantics the other rows build on. No kernels, runtimes or CI changes.

- Pins `meta-models/Muse-Glimmer-30B` @ `a4e59da5` (config, index and per-shard SHA-256) and the semantics source `transformers==5.17.0` @ `856157a2`, the newest release that vLLM 0.31 and Megatron Bridge both accept.
- `OFFICIAL_FINGERPRINT` and `OFFICIAL_FROZEN_SEMANTICS` hold the config values and the RFC's to-freeze items. They also pin the layer schedule (global iff `i % 4 == 3`), four separate Q/K/V/gate GEMMs (fused layouts fail closed), canonical TP 8 and the vocabulary padded to 202,752.
- Adds the strict loader (hashes and safetensors headers, fails closed), the operator trace (1,071 nodes mapped to RFC rows) and `scripts/museglimmer_reference.py`.

The pin corrects four RFC values: `qk_scale_factor` is 3.87 and applies to Q only; the output gate is elementwise over 4,096 channels; the model is 29.78B parameters / 59.55 GB; vLLM supports it from 0.28.0.

## Test

```bash
PYTHONPATH=. MUSEGLIMMER_CHECKPOINT=<checkpoint> python -m pytest tests/models/muse_glimmer/test_museglimmer_fingerprint.py -q  # 37 passed
PYTHONPATH=. python -m pytest tests/models/muse_glimmer/test_museglimmer_fingerprint.py -q  # 36 passed, 1 skipped (needs the checkpoint)
```

black, isort and flake8 (versions from `.pre-commit-config.yaml`) are clean. The CI entry for these tests is a follow-up.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_01U3SxjVAWpLJB5L47T5WgDc
