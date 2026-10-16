## Summary

Foundation row of RFC #501 (`museglimmer_arch_fingerprint`). It pins the checkpoint and the model semantics that every later Muse-Glimmer-30B row builds on. This PR adds no kernels, runtime bindings or CI changes; those come as separate PRs per RFC row.

What it pins and provides:

- **Checkpoint identity** (`museglimmer_manifest.json`): `meta-models/Muse-Glimmer-30B` at revision `a4e59da52a7bc87ae7251dd5545c0dd437c44b68`, with the SHA-256 of `config.json`, of `model.safetensors.index.json` and of each of the two shards (59,553,435,272 bytes, 1,436 BF16 tensors).
- **Semantics source**: `transformers==5.17.0` @ `856157a2f3e9594954310df18fdccc31ffddebe9` (`modeling_muse_glimmer.py` SHA-256 `e69eb21d…`). 5.17.0 is the release that both vLLM 0.31 (`<5.18`) and Megatron Bridge main (`>=5.15,<5.18`) accept. For the text model it differs from 5.19.0 only in how the RoPE angles are formed (a K=1 fp32 matmul vs an elementwise product).
- **`OFFICIAL_FINGERPRINT`** (`workload.py`): 52 layers, hidden 6656, intermediate 19968, 32 Q / 2 KV heads, head_dim 128, sliding window 2048, vocab 202048, `qk_scale_factor` 3.87, `final_logit_softcapping` 20, `output_multiplier` 0.19611613513818404, eps 1e-5 / post-norm eps 1e-8, RoPE θ 500000.
- **`OFFICIAL_FROZEN_SEMANTICS`**: the RFC's to-freeze items read from the pinned modeling file. These are the weightless embedding RMSNorm, the `(1+w)` four-norm sandwich and its two eps, QK-norm with the 3.87 scale on Q only (after the cast), RoPE on sliding layers only (NoPE on full layers), the elementwise sigmoid output gate over 4096 channels fed by the `input_layernorm` output, SwiGLU, and the logit chain `lm_head → ×output_multiplier → ÷T → tanh → ×T`.
- **Layer schedule check**: layer `i` is `full_attention` iff `i % 4 == 3` (13 global, 39 sliding), and `layer_rope_theta` is 0 on full layers.
- **Projection grouping**: four separate attention GEMMs (`q_proj`, `k_proj`, `v_proj`, `gate_proj`) and separate MLP `gate_proj` / `up_proj`. Fused names (`qkv_proj`, `qkvg_proj`, `linear_qkv`, `gate_up_proj`, `linear_fc1`) fail closed.
- **Parallel layout**: canonical TP 8; vocabulary padded to 202,752 (198 × 1024).
- **Strict loader** (`loader.py`): verifies the hashes, parses the safetensors headers and fails closed on any mismatch.
- **Operator trace** (`operator_trace.py`): `MuseGlimmerSpec`, with node kinds mapped to RFC rows (`NODE_KIND_ROWS`) and the GEMM shapes. There are 1,071 nodes for the 52-layer text model.
- **`scripts/museglimmer_reference.py`**: prints or emits the pinned fingerprint and checks a local checkpoint.

## Corrections to the RFC text found while pinning

These values come from `config.json` and the pinned modeling file.

- `qk_scale_factor` is **3.87** (not 43.7840518911 = 3.87·√128), applied to **Q only**; the effective logit multiplier is 3.87/√128.
- The output gate is **elementwise over 4096 channels**, not per head. Its input is the `input_layernorm` output, and it is applied before `o_proj`.
- The RFC's config keys (`use_qk_norm`, `output_soft_cap_temp`, `normalize_tok_embeddings`, bare `rope_theta`) are vLLM's flat-schema names. In the HF config, QK-norm has no flag and the cap is `final_logit_softcapping`.
- Parameter count is 29.78B (27.85B text + 1.92B vision), and the checkpoint is 59.55 GB. 59.58 GB is the repo's total storage.
- vLLM has native support from **v0.28.0**, not 0.27.0.

## Test

```bash
PYTHONPATH=. MUSEGLIMMER_CHECKPOINT=~/models/Muse-Glimmer-30B python -m pytest tests/models/muse_glimmer/test_museglimmer_fingerprint.py -q
# 33 passed in 3.61s
PYTHONPATH=. python -m pytest tests/models/muse_glimmer/test_museglimmer_fingerprint.py -q -rs
# 32 passed, 1 skipped in 3.55s   (the shard-hash test needs MUSEGLIMMER_CHECKPOINT)
PYTHONPATH=. python scripts/museglimmer_reference.py --checkpoint ~/models/Muse-Glimmer-30B --verify-shard-sha256
# semantics_source: transformers==5.17.0 @ 856157a2f3e9594954310df18fdccc31ffddebe9
# layer_types: 39 sliding_attention / 13 full_attention; canonical_tp: 8  padded_vocab_size: 202752
# operator_trace: 1071 nodes
# checkpoint: OK (627 text tensors, all shards hashed)
```

Environment: NVIDIA A100-SXM4-80GB, Python 3.12, torch 2.13.0+cu130, transformers 5.17.0 (the tests themselves run on CPU).

Lint with the versions pinned in `.pre-commit-config.yaml` (black 24.4.2, isort 5.13.2, flake8 7.0.0, line length 100): clean. Trailing whitespace and end-of-file checks also pass.

## Not in this PR

- No CI entry yet: the fingerprint test is not added to `.github/workflows/ci.yml` (follow-up).
- The WS1 kernel rows, WS2 TP/CP rows, the vLLM / Megatron Bridge bindings and the E2E row will come as separate PRs against `test-museglimmer`.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_01U3SxjVAWpLJB5L47T5WgDc
