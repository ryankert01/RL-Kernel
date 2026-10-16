# REPORT — `museglimmer_ablation` (one-factor mismatch probes, RFC #501)

**STATUS: partial.** The session was cut off before any real-weight run. The harness, CLI
and unit tests are committed; **no factor has been measured on the real weights yet.** The
results table below is empty by design — do not quote numbers from it.

## Resume command (one A100, ~56 GB, text weights only, ~10–15 min for all factors)

```bash
cd <worktree>   # branch worktree-agent-aff98d33dd3f64af6
~/bin/gpu-lease -n 1 --pool 0-5 -- env PYTHONPATH=$PWD ~/envs/vllm/bin/python \
  -m rl_engine.models.muse_glimmer.e2e.run_ablation \
  --checkpoint ~/models/Muse-Glimmer-30B \
  --out rl_engine/models/muse_glimmer/e2e/ablation_results.json
```

The JSON is rewritten after every factor, so a second cut-off still leaves partial results;
`--factors <names>` reruns a subset. Unit tests (CPU, no checkpoint):

```bash
PYTHONPATH=$PWD ~/envs/vllm/bin/python -m pytest tests/models/muse_glimmer/test_ablation_factors.py -q
```

## Summary

`rl_engine/models/muse_glimmer/e2e/ablation_hf.py` loads `MuseGlimmerTextModel`
(transformers 5.17.0) on the meta device and fills it from the safetensors shards
(`model.language_model.*` + `lm_head.weight` only; the vision tower is never instantiated),
bf16, eager attention, TF32 off. Eight chat-templated GSM8K questions are padded with a fixed
instruction passage to exactly 512 tokens; 64 greedy completion tokens come from the baseline
model so every variant scores the same tokens. Metrics per factor: max/mean |Δlogp| of the 64
selected tokens, argmax-change fraction, first index with |Δlogp| > 1e-3, non-finite hidden
states (forward hooks on every decoder layer), wall-clock. Two built-in gates:
`patched_identity` (replacement attention forward at default knobs must be bitwise HF) and
`restore_check` (baseline reproduced bitwise after all factors exited).

## Factors implemented (registry names in `factor_registry`)

| # | brief item | factor name(s) |
| --- | --- | --- |
| 1 | RoPE on global layers | `rope_on_global_layers` |
| 2 | NoPE everywhere | `nope_all_layers` |
| 3 | window | `sliding_window_256/512/1024/4096/none` (256/512 added: at 576 tokens, 1024/4096/none are trivially identical to 2048; eviction timing needs >W tokens — open issue) |
| 4 | gate | `gate_off`, `gate_per_head_mean` |
| 5 | Q/K scale | `qk_scale_43.78`, `qk_scale_on_k_too`, `qk_scale_folded_bf16`, `qk_scale_folded_fp32_scores`, `fp32_scores_only` |
| 6 | embedding norm | `embed_norm_off`, `embed_norm_eps_1e-6` |
| 7 | soft cap | `softcap_swapped`, `softcap_off`, `softcap_fp32` |
| 8 | post_norm_eps | `post_norm_eps_1e-5` |
| 9 | centered gain | `centered_norm_plain_w` |
| 10 | q/k single rounding | `qk_single_rounding` (fp32 norm→scale→RoPE→one bf16 cast) |
| 11 | fp32 logits | `lm_head_fp32` |
| 12 | runtime | `tf32_on`, `sdpa`, `batch_of_8`, `chunked_prefill_256` |
| 13 | RoPE table | `rope_table_elementwise` (5.19.0-style) |

## Results (factor × metrics) — NOT YET MEASURED

| factor | max abs dlogp | mean abs dlogp | argmax change | first drift | nonfinite | wall s |
| --- | --- | --- | --- | --- | --- | --- |
| (run the resume command; `run_ablation.py` prints this table ranked by drift) | | | | | | |

## Hypotheses to verify (not results)

- `rope_table_elementwise` and `sliding_window_{1024,4096,none}` should be bitwise equal to
  baseline (fp32 p·inv_freq is exact in both forms with TF32 off; windows above the sequence
  length are no-ops). If so, the RFC's "window size / eviction timing" factor must be run at
  >2048 tokens (`hybrid_attention_regression`) to mean anything.
- `tf32_on` touches only the fp32 RoPE-table matmul in a bf16 model, so it measures RoPE
  phase sensitivity, not GEMM nondeterminism.
- `centered_norm_plain_w`, `embed_norm_off`, `nope_all_layers`, `rope_on_global_layers`,
  `qk_scale_43.78` are expected to be catastrophic (argmax changes on most tokens): the RFC's
  43.78 is a factor-11 error against the checkpoint's 3.87 (`qk_scale_factor`, config.json).
- `gate_per_head_mean` quantifies how far the RFC's "per-head gate" reading is from the real
  elementwise 4096-wide gate (frozen `output_gate.granularity = elementwise`).
- `qk_single_rounding`, `softcap_fp32`, `lm_head_fp32`, `fp32_scores_only`, `sdpa`,
  `batch_of_8`, `chunked_prefill_256` are the implementation-divergence magnitudes — the
  tolerance budget a strict contract must pin to zero.

## Open issues

- `qk_single_rounding` upcasts the bf16 cos/sin table unless fp32 tables are attached to the
  attention modules (`_ablation_cos32/_sin32`); attach them to match the contract's fp32 RoPE.
- Eviction-timing probe (window 256 + chunked prefill 128 vs one-shot) is not wired; the
  `DynamicCache` sliding layer keeps `W-1` tokens.
- `greedy_complete` recomputes the full prefix per token (no cache): 64 × 8 forwards, fine.

## RFC feedback (what can be said before measuring)

- `qk_scale_factor` in the RFC model-contract table (43.7840518911) contradicts config.json
  (3.87). The RFC should say 3.87, Q only, after the weightless fp32 RMSNorm.
- "per-head sigmoid gate" should read "elementwise sigmoid gate over the 4096 flattened
  channels, computed from the input_layernorm output, applied before o_proj".
- The Experiments list should add: q/k rounding points, soft-cap dtype, lm_head dtype,
  attention backend (eager/SDPA), batch size, chunked prefill, RoPE-table construction —
  the divergences real runtimes actually have; "layer type" and "RoPE on global layers" are
  pins only a broken loader would violate.

## Time spent

~1 h reading (preamble, RFC, HF modeling, workload pins, contract) and writing harness, CLI,
tests; cut off before the GPU run.
