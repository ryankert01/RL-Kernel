Status: Proposed  
Target: post-v0.1.0 community roadmap  
Related: #386

Reference Qwen3-8B Dense track: #180, #204, #228, #230, #240, #243, #280, #315, #322, #336, #338, #343, #352, #360, #361, #377.

---

## 1. Motivation

RL-Kernel has already closed the dense Qwen3-8B train/rollout consistency loop on CUDA and ROCm, including WS1 operator contracts, WS2 distributed execution, VIME/vLLM integration, ablation, and end-to-end experiments.

google/gemma-4-31B-it is the next dense model target. The goal is the same:

> Given the same logical tokens, weights, positions, masks, multimodal inputs, and sampling context, training and rollout must execute the same declared arithmetic contract and produce exactly equal selected-token log-probabilities in strict mode.

This RFC covers CUDA and ROCm, and splits the work into two phases:

- Phase A — text backbone: close WS1, WS2, ablation, vLLM/VIME integration, and end-to-end consistency for the 31B text backbone.
- Phase B — multimodal path: add the Gemma 4 vision encoder, vision soft-token construction/injection, bidirectional vision-token mask semantics, and full VLM train/rollout validation.

The split keeps the text-path consistency work unblocked while VIME-side Gemma 4 multimodal support is brought up.

---

## 2. Checkpoint fingerprint

The implementation must be pinned to the exact checkpoint/runtime revision used by each validation run. The current google/gemma-4-31B-it architecture relevant to this RFC is:

| Item | Value |
| --- | --- |
| Architecture | Gemma4ForConditionalGeneration |
| dtype | BF16 |
| Text hidden size | 5376 |
| Text intermediate size | 21504 |
| Decoder layers | 60 |
| Attention schedule | 5 × sliding_attention + 1 × full_attention, repeated |
| Q heads | 32 |
| Sliding KV heads / head dim | 16 / 256 |
| Global KV heads / head dim | 4 / 512 |
| Sliding window | 1024 |
| Max positions | 262144 |
| Vocabulary | 262144 |
| Activation | gelu_pytorch_tanh |
| Final logit softcap | 30.0 |
| Input/output embedding | tied |
| Global K/V semantics | attention_k_eq_v=true |
| Sliding RoPE | standard RoPE, theta 10000 |
| Global RoPE | proportional p-RoPE, partial factor 0.25, theta 1e6 |
| Vision-token attention | use_bidirectional_attention="vision" |
| MoE | disabled for this checkpoint |
| PLE | disabled for this checkpoint |
| Shared-KV decoder layers | disabled for this checkpoint |
| Audio path | disabled |

Vision-side fingerprint:

| Item | Value |
| --- | --- |
| Vision hidden size | 1152 |
| Intermediate size | 4304 |
| Layers | 27 |
| Heads / KV heads | 16 / 16 |
| Head dim | 72 |
| Patch size | 16 |
| Pooling kernel | 3 |
| RoPE theta | 100 |
| Default soft-token length | 280 |

Upstream references:

- https://huggingface.co/google/gemma-4-31B-it/blob/main/config.json
- https://github.com/huggingface/transformers/tree/main/src/transformers/models/gemma4
- https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/gemma4.py
- https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/gemma4_mm.py

### Upstream runtime status

- vLLM: Gemma 4 inference already exists, including the hybrid text attention path and multimodal model path. RL-Kernel therefore needs strict operator binding and runtime provenance, not a new serving implementation.
- VIME: as of 2026-09-17, the current vime_plugins/models/ tree does not contain a Gemma 4 provider. VIME support is therefore treated as new integration work in this RFC.

---

## 3. Architecture map

<img width="1500" alt="Gemma-4-31B-it architecture and RL-Kernel consistency boundaries" src="https://raw.githubusercontent.com/RL-Align/RL-Kernel/main/docs/assets/rfc/gemma4-31b/gemma4-31b-architecture.svg">

**Figure 1 — Gemma-4-31B-it architecture and RL-Kernel consistency boundaries.** The text backbone is closed first; the image path joins through deterministic vision soft-token construction and embedding merge.

The central difference from Qwen3-8B is that Gemma 4 does not have one uniform attention layer. Sliding and global layers have different head geometry, KV ownership, RoPE, mask semantics, cache behavior, and K/V projection semantics. They must therefore have separate runtime identities and validation rows.

### One decoder layer

<img width="1500" alt="Gemma 4 hybrid decoder layer strict operator boundaries" src="https://raw.githubusercontent.com/RL-Align/RL-Kernel/main/docs/assets/rfc/gemma4-31b/gemma4-31b-hybrid-layer.svg">

**Figure 2 — One Gemma 4 decoder layer.** WS1 qualifies each arithmetic boundary; WS2 wraps the same boundary with deterministic TP/CP/SP ownership and communication order.

---

## 4. Numerical contract

The existing RL-Kernel WS1 numerical standard remains authoritative. Gemma 4 does not get a weaker consistency definition.

For every strict reduction-bearing boundary:

1. accumulator precision is declared and fixed;
2. reduction/addition order is documented and independent of batch size, batch position, packing, launch geometry, SM/CU count, or runtime occupancy;
3. Split-K / Stream-K / split-KV / cross-CTA atomics are forbidden unless the split and merge tree is itself fixed by the contract;
4. TF32 and fast-math reassociation are disabled in strict mode;
5. casts occur only at declared epilogue points;
6. one logical row is invariant to unrelated batch rows, padding mutation, chunking, and rank placement;
7. unsupported strict geometry fails closed rather than silently falling back to a production/native path;
8. runtime provenance records requested backend, actual backend, kernel/implementation ID, schedule ID, split policy, accumulator dtype, layer type, device, dtype, shape fingerprint, and fallback state.

Platform acceptance:

- CUDA: exact train/rollout selected-token logprob equality under the pinned contract.
- ROCm: exact train/rollout selected-token logprob equality under the pinned contract.
- CUDA ↔ ROCm: cross-platform byte equality is tested and reported, but it is not assumed. Any unavoidable platform-level transcendental/compiler difference must use an explicit named comparator profile with provenance.

---

## 5. Kernel and work-item table

Status: 🙋 open → ⏳ in progress → 👀 in review → ✅ merged

To claim a task, put your handle in the GitHub column and open a PR. A row is not complete until both the implementation and the row-specific validation contract are included.

Legend for reuse:

- Reuse — existing Qwen3 arithmetic kernel can be reused after Gemma shape qualification.
- Extend — existing infrastructure is reusable, but Gemma-specific arithmetic or semantics are required.
- New — a new strict operator or integration boundary is required.

| Work item | What it does | Track | Platform | Reuse / dependency | GitHub | PR | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gemma4_arch_fingerprint | Freeze checkpoint revision, 60-layer type schedule, head geometry, RoPE, K=V, norm/residual order, tied weights, softcap, multimodal mask semantics | foundation | CUDA + ROCm | New |  @haotianxue31 |  | 🙋 |
| gemma4_operator_trace | Machine-readable per-layer operator/runtime trace consumed by WS1/WS2 gates | foundation | CUDA + ROCm | Extend WS1 chain infra | @haotianxue31  |  | 🙋 |
| scaled_token_embedding | Embedding lookup plus Gemma embedding scaling for vocab 262144 / hidden 5376 | WS1 | CUDA + ROCm | Extend #243 |  @haotianxue31 |  | 🙋 |
| tied_embedding_lm_head_contract | Shared input/output weight aliasing and deterministic shared-gradient accumulation | WS1 | CUDA + ROCm | Extend #243 |  @haotianxue31 |  | 🙋 |
| gemma_rmsnorm | Gemma RMSNorm arithmetic and cast order for block/final norms | WS1 | CUDA + ROCm | Extend existing norm infra |  @haotianxue31 |  | 🙋 |
| qk_rmsnorm_d256_d512 | Per-head Q/K RMSNorm for sliding D256 and global D512 | WS1 | CUDA + ROCm | Extend Qwen QK-norm infra | @haotianxue31  |  | 🙋 |
| sliding_qkv_gemm | Q 5376→8192, K/V 5376→4096 deterministic projections | WS1 | CUDA + ROCm | Reuse #180 + ROCm det-GEMM | @haotianxue31  |  | 🙋 |
| global_q_gemm | Global Q projection 5376→16384 | WS1 | CUDA + ROCm | Reuse #180 + ROCm det-GEMM | @haotianxue31  |  | 🙋 |
| global_k_eq_v_projection | Freeze global K=V parameter mapping, forward identity and backward gradient semantics | WS1 | CUDA + ROCm | New | @haotianxue31  |  | 🙋 |
| sliding_rope_d256 | Standard RoPE for sliding layers, D256, theta 10000 | WS1 | CUDA + ROCm | Extend #228 | @haotianxue31  |  | 🙋 |
| global_proportional_p_rope | Partial proportional p-RoPE for global layers, D512, factor 0.25, theta 1e6 | WS1 | CUDA + ROCm | New | @haotianxue31  |  | 🙋 |
| sliding_attention_core | Causal local attention, window 1024, Q32/KV16, D256 | WS1 | CUDA + ROCm | Extend #240; ROCm AITER/CK or Triton strict path | @haotianxue31   |  | 🙋 |
| global_attention_core | Full attention, Q32/KV4, D512, K=V, Gemma mask semantics | WS1 | CUDA + ROCm | Extend #240; ROCm strict attention path | @haotianxue31  |  | 🙋 |
| hybrid_kv_cache_contract | Sliding/global KV-cache layout, page identity, prefill/chunked-prefill/decode handoff | WS1 | CUDA + ROCm | Extend #230/#240 | @haotianxue31  |  | 🙋 |
| attention_o_gemm | Sliding 8192→5376 and global 16384→5376 O-projection shape qualification | WS1 | CUDA + ROCm | Reuse #180 |  @haotianxue31 |  | 🙋 |
| post_attention_residual | Freeze post-attention norm and residual ordering/cast points | WS1 | CUDA + ROCm | Extend chain harness | @haotianxue31  |  | 🙋 |
| ffn_gate_up_gemm | 5376→21504 ×2 deterministic gate/up projection | WS1 | CUDA + ROCm | Reuse #180/#322/#343 | @cwgan  |  | 🙋 |
| gelu_tanh_and_mul | Deterministic gelu_pytorch_tanh(gate) × up forward/backward | WS1 | CUDA + ROCm | New; #280 only supplies activation infra |  @cwgan |  | 🙋 |
| ffn_down_gemm | 21504→5376 deterministic down projection | WS1 | CUDA + ROCm | Reuse #180/#322/#343 |  @cwgan |  | 🙋 |
| post_ffn_residual | Pre-FFN norm → MLP → post-FFN norm → residual ordering | WS1 | CUDA + ROCm | Extend chain harness | @FED4  |  | 🙋 |
| tied_lm_head_projection | Fixed-K tied LM-head projection 5376→262144 | WS1 | CUDA + ROCm | Extend #243/#336 |  |  | 🙋 |
| final_logit_softcap | Strict FP32 30 × tanh(logit / 30) before LSE/logprob | WS1 | CUDA + ROCm | New | @Jungle430  |  | 🙋 |
| softcapped_selected_logprob | Selected-token logprob after softcap with fixed vocab reduction | WS1 | CUDA + ROCm | Extend #204/#336 |  |  | 🙋 |
| ws1_full_text_chain | Full 60-layer text model, real weights, train/decode parity, first-drift localization | WS1 closeout | CUDA + ROCm | Extend #315 |  |  | 🙋 |
| ws1_hybrid_attention_regression | Dedicated sliding/global alternation + cache parity + batch/chunk invariance suite | WS1 closeout | CUDA + ROCm | New |  |  | 🙋 |
| ws2_tp2 | Exact TP2 equivalence against WS1 | WS2 | CUDA + ROCm | Extend Qwen3 WS2 |  |  | 🙋 |
| ws2_tp4 | Exact TP4 equivalence against WS1 | WS2 | CUDA + ROCm | Extend Qwen3 WS2 |  |  | 🙋 |
| ws2_tp4_cp2 | Exact TP4/CP2 equivalence with fixed CP ownership and reduction order | WS2 | CUDA + ROCm | Extend Qwen3 WS2 |  |  | 🙋 |
| ws2_tp4_cp2_sp | Sequence-parallel acceptance on top of TP4/CP2 | WS2 | CUDA + ROCm | Extend Qwen3 WS2 |  |  | 🙋 |
| ws2_tp8_kv_replication | Define and validate replicated KV-head ownership because global attention has only 4 KV heads | WS2 | CUDA + ROCm | New |  |  | 🙋 |
| cp_sliding_window_policy | Correctness-first CP contract for local windows; optimize to deterministic halo exchange only after parity | WS2 | CUDA + ROCm | New |  |  | 🙋 |
| vocab_parallel_softcap_logp | Fixed global-vocab reduction after tied LM head + FP32 softcap | WS2 | CUDA + ROCm | Extend #336 |  |  | 🙋 |
| ws2_runtime_provenance | Read back actual TP/CP/SP topology, attention/GEMM/logp backend IDs, schedule IDs and fallback state | WS2 | CUDA + ROCm | Extend #338/#352/#360 |  |  | 🙋 |
| gemma4_ablation_overlay | Extend #230 taxonomy with Gemma-specific factors without replacing A0–A7/C0–C2 | ablation | CUDA + ROCm | Extend #230 |  |  | 🙋 |
| ablation_hybrid_layer_type | Sliding vs global layer identity / wrong-layer injection signature | ablation | CUDA + ROCm | New |  |  | 🙋 |
| ablation_rope_family | Sliding RoPE vs global p-RoPE mismatch signature | ablation | CUDA + ROCm | New |  |  | 🙋 |
| ablation_global_k_eq_v | Independent K/V vs K=V mapping mismatch signature | ablation | CUDA + ROCm | New |  |  | 🙋 |
| ablation_attention_window_mask | Sliding window / full mask / vision-bidirectional mask mismatch signature | ablation | CUDA + ROCm | New |  |  | 🙋 |
| ablation_gelu_tanh | Native vs strict GELU-tanh gating mismatch signature | ablation | CUDA + ROCm | New |  |  | 🙋 |
| ablation_tied_weight | Untied vs tied embedding/LM-head and shared-gradient mismatch signature | ablation | CUDA + ROCm | New |  |  | 🙋 |
| ablation_logit_softcap | Softcap location, dtype, and on/off mismatch signature | ablation | CUDA + ROCm | New |  |  | 🙋 |
| ablation_hybrid_kv_cache | Sliding/global KV layout, block plan, merge order, and prefill/decode mismatch signature | ablation | CUDA + ROCm | New |  |  | 🙋 |
| ablation_distributed_ownership | TP/CP/SP ownership perturbation on top of #230 A4/C0 controls | ablation | CUDA + ROCm | Extend distributed ablation infra |  |  | 🙋 |
| vllm_runtime_adapter_gemma4 | Bind RL-Kernel strict operators into the existing vLLM Gemma 4 text runtime | integration | CUDA + ROCm | Extend #338/#360 |  |  | 🙋 |
| vime_provider_gemma4_text | New Gemma 4 text provider/model registration/mapping in VIME | integration | CUDA + ROCm | New |  |  | 🙋 |
| vime_checkpoint_mapping_gemma4 | HF ↔ Megatron/VIME parameter mapping for the Gemma 4 text backbone | integration | CUDA + ROCm | New |  |  | 🙋 |
| vime_linear_logp_provider_gemma4 | Gemma 4 selected-logprob path aware of tied LM head and final softcap | integration | CUDA + ROCm | Extend #352 |  |  | 🙋 |
| multimodal_vision_encoder_trace | Freeze Gemma 4 vision tower contract, pooling, soft-token count, and merge semantics | phase B | CUDA + ROCm | New |  |  | 🙋 |
| multimodal_vision_token_injection | Deterministic vision soft-token construction and merge into the text stream | phase B | CUDA + ROCm | New |  |  | 🙋 |
| multimodal_bidirectional_mask | Vision-token bidirectional mask semantics inside the global attention path | phase B | CUDA + ROCm | New |  |  | 🙋 |
| multimodal_ws1_full_chain | Full VLM WS1 chain including the image path | phase B | CUDA + ROCm | New |  |  | 🙋 |
| multimodal_ws2_tp_cp | Distributed multimodal path with TP/CP acceptance | phase B | CUDA + ROCm | New |  |  | 🙋 |
| final_e2e_vime_vs_rlkernel_cuda | End-to-end Gemma 4 text training/rollout experiment: native path vs RL-Kernel + VIME on CUDA | final | CUDA | Extend #377 methodology |  |  | 🙋 |
| final_e2e_vime_vs_rlkernel_rocm | End-to-end Gemma 4 text training/rollout experiment: native path vs RL-Kernel + VIME on ROCm | final | ROCm | Extend #390 / #400 methodology |  |  | 🙋 |
| cuda_perf_closeout | CUDA strict-path performance optimization after correctness closeout | perf | CUDA | Extend #365/#367/#351/#354 |  |  | 🙋 |
| rocm_perf_closeout | ROCm strict-path performance optimization after correctness closeout | perf | ROCm | Extend #396/#390 |  |  | 🙋 |

---

## 6. Recommended claim order

For contributors, the recommended order is:

1. gemma4_arch_fingerprint
2. gemma4_operator_trace
3. scaled_token_embedding / tied_embedding_lm_head_contract
4. gemma_rmsnorm / qk_rmsnorm_d256_d512
5. sliding_qkv_gemm / global_q_gemm / global_k_eq_v_projection
6. sliding_rope_d256 / global_proportional_p_rope
7. sliding_attention_core / global_attention_core / hybrid_kv_cache_contract
8. ffn_gate_up_gemm / gelu_tanh_and_mul / ffn_down_gemm
9. tied_lm_head_projection / final_logit_softcap / softcapped_selected_logprob
10. ws1_full_text_chain / ws1_hybrid_attention_regression
11. ws2_tp2 → ws2_tp4 → ws2_tp4_cp2 → ws2_tp4_cp2_sp
12. gemma4_ablation_overlay and Gemma-specific ablation rows
13. vllm_runtime_adapter_gemma4 / vime_provider_gemma4_text / vime_checkpoint_mapping_gemma4
14. Phase B multimodal rows
15. final end-to-end experiments and performance closeout

---

## 7. Contribution notes

- A work item is not done until its validation contract is landed together with the implementation.
- Sliding and global attention are two different acceptance items. A contributor should not claim “Gemma attention done” by covering only one side.
- TP8 is a separate WS2 acceptance item because the global path has only 4 KV heads and therefore introduces replicated KV-head ownership.
- Correctness comes before optimization. In particular, the CP sliding-window path should first land a correctness-first deterministic contract before any optimized halo-exchange implementation is accepted.
- Phase B multimodal work must preserve the exact same strict logprob contract used by the text-only path.


### if you are interested just ping below!

