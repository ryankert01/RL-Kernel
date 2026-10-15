# Muse-Glimmer-30B strict train–rollout POC — report (DRAFT, assembled as results land)

Date: 2026-10-15. Hardware: 1 node, 8× A100-SXM4-80GB (NOT the RFC's H100), driver 580.105, nvcc 12.8.
Branch: `feat/museglimmer-arch-fingerprint` (POC; targets `test-museglimmer`). Checkpoint `meta-models/Muse-Glimmer-30B` @ `a4e59da5` (both shard SHA-256 verified).

## 0. One-paragraph verdict
(to fill at the end)

## 1. What was attempted and why (scope)
- Phase A text model only; no Phase B vision, no DFlash/speculative, no full-parameter run, no ROCm (no AMD hardware here).
- Every RFC row implemented at POC quality by a parallel agent team (13 agents, disjoint file ownership, worktree isolation, GPU leases), merged by one integrator against one numerical contract (`docs/contracts/muse-glimmer-30b-strict-contract.md`).
- Kernels: PyTorch golden + Triton only. No CUDA twins (community rows).
- Runtimes: real vLLM 0.31.0 and real Megatron Bridge 0.7.0 (git main), NOT VIME. The Qwen3 E2E in this repo ran through VIME + Megatron-LM; porting VIME to Megatron Bridge is left to the community.

## 2. Environment facts that constrain the RFC
| Item | Finding |
|---|---|
| Transformers pin triangle | vLLM 0.31.0 pins `transformers<5.18`; Megatron Bridge main pins `>=5.15,<5.18` and Python 3.12; the checkpoint was written with 5.15.0.dev0. Only **5.17.0** satisfies everyone → the fingerprint's semantics source moved from 5.19.0 to 5.17.0. 5.17→5.19 text-model diff: RoPE table built by a TF32-sensitive matmul (5.17) vs fp32 elementwise multiply (5.19). |
| vLLM native support | present from **v0.28.0** (PR #51655 merged 2026-08-14), not 0.27.0. The native model fuses Q/K/V (`QKVParallelLinear`) AND gate/up (`MergedColumnParallelLinear`); `output_gate_proj` is separate; `MuseGlimmerRMSNorm` is a plain nn.Module; softcap is applied in bf16 after the LogitsProcessor. |
| Megatron Bridge | PyPI 0.3.1 has no Muse Glimmer and pins transformers<5; git main (0.7.0) has it: HybridModel, TE RMSNorm with zero-centered gamma, nn.RMSNorm q/k norm with 3.87 folded into softmax_scale, fused gated `linear_qkv` (`MuseGlimmerQKVGMapping`), softcap after the CE module. |
| torch/Triton | vLLM env torch 2.13.0+cu130, Megatron env torch 2.13.0+cu129 (TE needs nvcc-compatible CUDA); **Triton 3.7.1 on both** — required for bitwise kernel parity across runtimes. |
| Repo prior art on A100 | The repo's strict attention (FA4 CuTe) and linear-logp kernels are SM90-only; the compiled `_C` extension cannot be built against a CUDA-13 torch with nvcc 12.8; the existing Triton tree GEMM runs at 5 TFLOP/s. Only the integration scaffolding (plugin entry point, attention-backend registration, plan/readback) was reusable. |

## 3. Numerical contract (summary) — see docs/contracts/muse-glimmer-30b-strict-contract.md
(to summarize: casts, orders, tree, tiles; and the deviations the implementers declared)

## 4. Results per row
### 4.1 WS1 (single GPU) — all merged
| Row | Status | Tests | Parity gates | Perf vs native | Deviations / notes |
|---|---|---|---|---|---|
| museglimmer_arch_fingerprint | ✅ | 33 | n/a | n/a | semantics pin 5.17.0 |
| dense_gemm_qualification | ✅ | 80 | G1/G2/TP1==tree(TP2/4/8)/G4/G5 | 108–134 TFLOP/s @M=4096 (1.9–2.5× cuBLAS time); decode floor 0.07 ms | tiles are perf-only (24 configs bitwise equal); K%256 required; dY→bf16 declared |
| normalized_embedding + fused_add_rmsnorm | ✅ | 43 | G1/G2/G4/G5 + fusion==composition | 0.13–0.31× eager time | rsqrt = IEEE 1/sqrt; fused op keeps fp32 residual grad |
| qknorm_scaled_rope | ✅ | 39 | kernel == golden bitwise; G1–G5 | 12.6× faster than eager | HF double rounding differs by up to 0.25 in q̂ (0.14% beyond bf16 tol) |
| global/local gated attention (fwd) | ✅ | 63 | train==chunked==decode==paged==eviction bitwise, lengths 64..8192 | prefill 4096: 1.31× SDPA time; paged decode 1.6× | num_warps must match across tiles (row-sum tree); −inf guard; page size 16 |
| gated attention (bwd) | ✅ | 24 | repeat/batch-invariant; TP2/4/8 sub-trees bitwise | 1.8–2.4× SDPA bwd time | bf16 casts of do/p/dS declared; gradient tolerance must be scale-free |
| swiglu_ffn | ✅ | 20 | G1/G2/G4/G5 | FFN 2.6× cuBLAS time (GEMMs 99%) | tree on all GEMMs; dW needs T%8 |
| lm_head_softcap_logprob (+ vocab tiles) | ✅ | 15 | G1/G2/G4/G5; tile-order & simulated TP1/2/4/8 bitwise | (to re-measure with fast_det_gemm) | tanh: libdevice == torch bitwise; division by tensor T; K%256 fallback added by integrator |

### 4.2 Chain, WS2, integration, E2E
(to fill: full_model_chain, hybrid_attention_regression (covered at op level by the attention G3 tests), tp_canonical_tree, vocab_parallel_softcap_logp, kv_head_replication, cp_global/local_attention, weight_sync_identity, backward_topology_invariance, vllm_runtime_adapter, megatron binding, museglimmer_ablation, final_e2e_lora)

## 5. Ups and downs (chronological)
- Session rate limit (429) cut seven agents mid-task; all resumed from saved state.
- Agent worktrees were created from `main`, not the POC branch; every agent had to merge the POC branch first (one reset its branch instead). Harmless but worth knowing for the next run.
- The LM-head op silently started using `fast_det_gemm` once it landed and its tiny-shape tests broke (K%256 contract) → integrator fallback.
- (more to add)

## 6. How the RFC must change
(from the 56-edit audit digest + implementer feedback; to be compiled into a paste-ready list)

## 7. Profiling summary
(to fill)

## 8. What is left for the community
(to fill)
