## Goal and scope

Support train–rollout consistency for `meta-models/Muse-Glimmer-30B` in RL-Kernel. Given the same tokens, weights, positions, masks and sampling context, strict training and rollout must produce exactly equal selected-token log-probabilities.

Muse Glimmer is a 29.6B dense model, served by vLLM and SGLang and trainable through Megatron Bridge. It adds three things the repo does not cover today: gated, scale-normalized attention; alternating local (sliding-window, RoPE) and global (full causal, no positional encoding) layers on 2 KV heads; and a normalized embedding with a multiplied, soft-capped logit path. The BF16 checkpoint is 59.58 GB, so everything except full-parameter end-to-end training fits one 8× H100 node.

- **Phase A:** the full 52-layer text model. Single-GPU kernels (WS1), multi-GPU execution (WS2), integration and a LoRA end-to-end run.
- **Phase B:** the vision input path.

Speculative rollout with the bundled DFlash drafter is left to a follow-up RFC; strict mode here runs with speculative decoding off. Quantized checkpoints, video input and the sampling RNG contract are out of scope.

No row is blocked on another RFC. Rows extend the Qwen3-8B dense track. Sibling rows in #415 and #434 are cited as "if landed", and the logit soft cap is shared through #542.

## Model contract

`museglimmer_arch_fingerprint` pins the checkpoint revision and confirms every value below against `config.json` and the modeling code. Values are from the model card and the published `muse_glimmer` configs.

| Component | Contract |
| --- | --- |
| Model | `MuseGlimmerForConditionalGeneration`; BF16; hidden size 6656; 52 decoder layers |
| Layer schedule | `[Local, Local, Local, Global]` × 13; global layers at 3, 7, …, 51 |
| Attention | GQA, 32 Q heads / 2 KV heads × 128; Q 6656→4096, K and V 6656→256, O 4096→6656 |
| Q/K norm | `use_qk_norm: true`, `qk_scale_factor: 43.7840518911` |
| Output gate | per-head sigmoid gate from `self_attn.gate_proj` |
| Local layers | causal, `sliding_window: 2048`; RoPE, `rope_theta: 500000` |
| Global layers | full causal; no positional encoding |
| FFN | SwiGLU, 6656→19,968→6656 |
| Norm | RMSNorm, `rms_norm_eps: 1e-5`; `post_norm_eps: 1e-8` |
| Embedding | vocabulary 202,048, input and output untied; `normalize_tok_embeddings: true` |
| Logits | `output_multiplier: 0.19611613513818404` (1/√26); `output_soft_cap_temp: 20.0` |
| Context | 131,072+ |
| Vision (Phase B) | frozen ViT-G/14 perception encoder, about 1.8B; `vision_output_dim: 6144`; up to 4,096 visual tokens per image |

The fingerprint also has to freeze what the configs do not state: the gate's width and where it is applied; the Q/K norm form; what `post_norm_eps` governs; the embedding normalization form and dtype; the soft-cap formula, dtype and order relative to the multiplier; the RoPE rotated-channel count; the local-layer cache eviction rule; and one grouping of the Q/K/V/gate projections (Megatron Bridge fuses all four, vLLM fuses Q/K/V only).

Upstream support: vLLM native model (0.27.0+, vllm#51655), SGLang (sglang#34262), and Megatron Bridge `bridge.models.muse_glimmer` with an 8× H100 LoRA recipe. The VIME provider is new work.

### Architecture

<img width="937" height="193" alt="Muse Glimmer layer schedule" src="https://github.com/user-attachments/assets/0f939e27-6c6d-4bc6-bc5b-9a563a706650" />

<img width="855" height="811" alt="One Muse Glimmer decoder layer" src="https://github.com/user-attachments/assets/fda4bb84-ffd2-4e07-9bf4-d2380bf84a03" />

`[G]` boxes reuse the deterministic GEMM. `[F*]` boxes are fused strict operators. The output gate is part of `[F3]` and `[F4]`.

## Numerical requirements

The existing WS1 numerical standard applies:

- Fixed accumulator precision and reduction order. No Split-K, Stream-K, split-KV or cross-block atomics unless the merge tree is fixed by contract. No TF32 or fast math. Casts only at declared points.
- **Batch invariance:** a logical row's bytes do not change with batch size or batch position.
- **Sequence-split invariance:** a token's output is identical in training, prefill and decode, for any chunked-prefill split and any eviction timing around the 2,048 window.
- **One projection grouping** for Q/K/V/gate on both sides. A runtime that keeps its native fused projection is not in strict mode.
- **Multi-GPU execution reproduces the single-GPU result byte for byte:** the forward pass first (G1), then gradients (G2).
- **Fail closed** on a Transformers-fallback model class, a non-canonical projection grouping, an unsupported window or eviction mode, a native reducing collective, a zigzag CP layout, an undeclared KV ownership map, a weight-sync hash mismatch, or speculative decoding.

Parity uses `torch.equal` between two strict paths. Correctness uses `allclose` against an independent FP64 reference.

**Platforms.** Each kernel PR carries the CUDA kernel and the Triton kernel, with the same arithmetic. ROCm runs the Triton kernel; no ROCm-specific kernel code is added. PRs are validated on CUDA, including CUDA == Triton bitwise. ROCm is tested periodically on the Triton path once most rows have landed, and CUDA ↔ ROCm equality is measured then.

## Work items

Claim a row with your handle and open a PR against `test-museglimmer`. One row = one PR.

A fused kernel is one row, because batch invariance and performance are accepted on the fused operator. Rows are split only between kernels and between WS1 and WS2.

Status: 🙋 open → ⏳ in progress → 👀 in review → ✅ merged. "If landed" means the sibling row is reused when it exists; otherwise the work is new.

### Foundation

| Work item | Scope | Reuse | Claim | PR | Status |
| --- | --- | --- | --- | --- | --- |
| `museglimmer_arch_fingerprint` | Pin checkpoint, layer schedule, to-freeze items, projection grouping, canonical TP and padded vocabulary; strict loader; operator trace | New | @ryankert01 | | ⏳ |

### WS1: single-GPU kernels

All differentiable operators need forward and backward.

| Work item | Scope | Reuse | Claim | PR | Status |
| --- | --- | --- | --- | --- | --- |
| `normalized_embedding` | Token lookup + embedding normalization | #243 | | | 🙋 |
| `fused_add_rmsnorm` | [F1] Residual add + RMSNorm; post-norm and final norm | norm infra | | | 🙋 |
| `dense_gemm_qualification` | [G] Deterministic GEMM on the Q/K/V/gate, O and FFN shapes | #180 / #322 / #343 | | | 🙋 |
| `qknorm_scaled_rope` | [F2] Scaled Q/K norm; RoPE on local layers only | #228 | | | 🙋 |
| `global_gated_attention` | [F4] Full causal NoPE GQA + output gate + KV cache; training, prefill, decode | #240 / #230; #434 `nope_gqa_attention` if landed | | | 🙋 |
| `local_gated_attention_train` | [F3] Sliding-window GQA + output gate, training path | #240; #415 `sliding_attention_core` if landed | | | 🙋 |
| `local_gated_attention_rollout` | [F3] The same arithmetic for prefill and decode; window KV cache and eviction; byte-equal to the training row | #230 / #240 | | | 🙋 |
| `swiglu_ffn` | [F5] Gate/up GEMM + SwiGLU + down GEMM | #280 / #322 / #343 | | | 🙋 |
| `lm_head_softcap_logprob` | LM head + multiplier + soft cap + selected-token logprob | #243 / #204 / #336; soft cap #542 | | | 🙋 |

### WS2: multi-GPU

No new communication primitive is needed. Rows reuse the deterministic collectives (`create_deterministic_collective`), the canonical TP tree (`RL_KERNEL_STRICT_CANONICAL_TP` = 8), the vocab-tile log-probability (#241, #336) and the CP attention contract (#235, `allgather` layout). Specific to Muse Glimmer:

- **Row-parallel sums.** O projection over 4,096 (8 shards of 512, 4 heads each) and FFN down over 19,968 (8 shards of 2,496).
- **Vocabulary.** 202,048 is not a multiple of 128 × canonical TP, so it is padded to 202,752. The multiplier and soft cap are applied before the tile partials, and padded logits are excluded.
- **2 KV heads.** At TP4 and TP8 each rank holds a copy of the KV head its Q heads use, so the forward pass needs no communication. For the backward pass, the proposal is to evaluate K/V gradients per canonical Q-head group (4 heads) and combine them in the canonical tree at every physical TP, including TP1.
- **CP.** Local layers apply the window on absolute positions. The gate is applied after the `(out, lse)` merge.

| Work item | Scope | Reuse | Claim | PR | Status |
| --- | --- | --- | --- | --- | --- |
| `tp_canonical_tree` | O projection and FFN down on the canonical tree; TP2/4/8 == TP1 | deterministic collectives; Qwen3 WS2 | | | 🙋 |
| `vocab_parallel_softcap_logp` | Padded vocabulary, pinned tiles, soft cap before partials; TP2/4/8 == TP1 | #241 / #336 | | | 🙋 |
| `kv_head_replication` | KV ownership map in both runtimes; K/V gradient rule | New; #415 `ws2_tp8_kv_replication` if landed | | | 🙋 |
| `cp_global_attention` | Global layers on the #235 contract; CP2/4 == CP1; zigzag fails closed | #235 | | | 🙋 |
| `cp_local_attention` | Local layers with the window; CP2/4 == CP1; one TP × CP topology | #235; #415 `cp_sliding_window_policy` if landed | | | 🙋 |
| `weight_sync_identity` | Trainer-to-rollout sync changes no logical weight; hash after every sync; one declared LoRA form | #352; #520 if landed | | | 🙋 |
| `backward_topology_invariance` | G2: gradients repeatable at a fixed topology, then equal across TP and CP | Qwen3 WS2 | | | 🙋 |

### Validation and integration

| Work item | Scope | Reuse | Claim | PR | Status |
| --- | --- | --- | --- | --- | --- |
| `full_model_chain` | 52 layers with real weights; training vs prefill vs decode; first-drift localization | #315 | | | 🙋 |
| `hybrid_attention_regression` | Lengths 2047 / 2048 / 2049, chunked prefill across the window, decode past eviction | New | | | 🙋 |
| `vllm_runtime_adapter` | Strict binding in vLLM's native model; provenance read-back | #338 / #360 | | | 🙋 |
| `megatron_vime_provider` | Strict binding in the Megatron Bridge model; VIME model script; weight mapping | New; #352 | | | 🙋 |
| `museglimmer_ablation` | One-factor mismatch probes | #230 | | | 🙋 |
| `final_e2e_lora` | LoRA RL run on one 8-GPU node, with rollout and training at different physical TP | #377 | | | 🙋 |
| `final_e2e_full_param` | Optional: full-parameter run on a larger cluster | needs hardware | | | 🙋 |
| `vision_input_path` | Phase B: perception encoder, projection and visual-token placement | New | | | 🙋 |

Suggested order: fingerprint → WS1 → `full_model_chain` → WS2 → integration → end-to-end. Within WS1, attention goes Q/K norm → global → local training → local rollout, and the other rows can run in parallel. `tp_canonical_tree` and `vocab_parallel_softcap_logp` do not need the attention rows.

## Acceptance

### Operator PRs

- [ ] PyTorch golden reference, CUDA kernel and Triton kernel; CUDA == Triton bitwise.
- [ ] Forward and backward correctness against an independent FP64 reference.
- [ ] Batch-invariance tests, and sequence-split tests for stateful operators.
- [ ] Documented reduction order, precision, cast points and supported geometry.
- [ ] Benchmark on Muse Glimmer shapes against the native path.
- [ ] Actual-backend trace and a reproducible validation command.

### WS1 and WS2

- [ ] All WS1 operators pass, then the full 52-layer chain: training, prefill and decode agree.
- [ ] G1: selected-token logprobs at TP2/4/8 and CP2/4 match the single-GPU result, on the rollout side and on the training side, and the two sides match at different physical TP.
- [ ] Weight sync changes no logical weight.
- [ ] G2: gradients match across TP and CP.
- [ ] Provenance records physical and canonical TP, collective backend, vocab tiles and padded size, KV ownership, CP layout, weight-sync hash and fallback state.
- [ ] Strict vs native collective cost is reported at Muse Glimmer shapes.

### Experiments

Mismatch probes change one factor at a time: layer type, RoPE on global layers, window size and eviction timing, gate, Q/K scale, embedding normalization, soft-cap order, projection grouping, KV ownership, a native reducing collective, zigzag CP, merged vs unmerged LoRA.

Run native/native, strict/native, native/strict and strict/strict training–rollout combinations. Report logprob drift, first drift, reward and performance.

## Open questions

1. **Projection grouping.** One fused Q/K/V/gate GEMM (as Megatron), fused Q/K/V plus a separate gate (as vLLM), or four GEMMs?
2. **K/V gradient rule.** Is evaluation per canonical Q-head group the right rule, and can #415 and #434 share it?
3. **LoRA form.** Unmerged `W·x + B·(A·x)` on both sides, or merged once on the trainer? Coordinate with #520.
4. **Backward gate.** Is G2 required for closeout, or is repeatability at a fixed topology enough for the first claim?
5. **Eviction policy.** Evict eagerly at the window edge, or keep a fixed over-allocation so the speculative follow-up never has to restore evicted entries?
6. **End-to-end scale.** Is a LoRA run enough for closeout?
7. **ROCm test rounds.** Who runs them, and at which milestones?
8. **Performance.** A fixed ratio to native per operator, or report-only at first?

## Follow-up: speculative rollout

The checkpoint ships a DFlash drafter (`meta-models/Muse-Glimmer-30B-assistant`; 5 layers, 16-token blocks) that reads target hidden states at layers {1, 13, 25, 37, 49}. A follow-up RFC would add three contracts: one verify forward over k drafted tokens equals k single-token decodes; a rejected tail leaves every cache byte-equal to the non-speculative state; and the hidden-state tap changes no target byte.

## References

- [Muse Glimmer-30B checkpoint](https://huggingface.co/meta-models/Muse-Glimmer-30B)
- [Transformers model doc](https://huggingface.co/docs/transformers/main/en/model_doc/muse_glimmer)
- [vLLM recipe](https://recipes.vllm.ai/meta-models/Muse-Glimmer-30B)
- [TP-invariant tree kernels](https://arxiv.org/abs/2511.17826)
- [DFlash](https://arxiv.org/abs/2602.06036)

### If you are interested, just ping below!

