# REPORT — megatron_vime_provider + weight_sync_identity (Megatron Bridge strict binding)

**STATUS: partial.** Code complete and committed; the 30B GPU runs (a)-(f) did NOT run before the session window closed (only the Bridge load/weight probe ran). Unit tests: see tests/models/muse_glimmer/test_megatron_strict_binding.py (result in the final handback). Time ~2.5 h.

## Files
- runtime/megatron_strict.py: install_museglimmer_megatron_strict(model) re-drives the 52-layer forward from Bridge parameters (no class patches): O1 normalized_embedding (full table), O2 fused norms (final norm = HF tensor reloaded because Bridge stores w-1 in bf16 lossily), O3 four column GEMMs on VIEWS of fused linear_qkv ([q 16g..|gate 16g..|k g|v g] per KV group of 4352 rows), two on linear_fc1 halves, row-parallel o/down via fast_det_gemm_partial + rank-ordered all-gather + combine_tree, O4 qknorm_scaled_rope (RoPE only on sliding layers), O5 gated_attention (window = config.window_size[0]+1; at TP2 head shards are all-gathered and attention computed redundantly since the kernels are fixed to 32Q/2KV heads), O6 swiglu, O7 canonical 8 vocab tiles (rank owns 8/t; Bridge unpadded shards re-tiled; softcap_tile_stats/launch_dz/merge_tile_stats; equals the sibling op at TP1). strict_token_logprobs(model, ids, targets) -> fp32 (B,T-1) with autograd, per-layer recompute. Fails closed on non-native class, config drift, CP>1/zigzag, SP, TP>2.
- runtime/megatron_lora.py: unmerged LoRA y = bf16(acc32(Wx) + sum_k B[:,k] * bf16(Ax)[k]) (rank-16 sum = sequential fp32 loop of exact products, one rounding); B=0 is an exact bitwise no-op; fp32 masters; column: A replicated/B row-sharded, row: A col-sharded/B replicated; export keys layers.{i}.{proj}.lora_{A,B} full bf16 + sha256 over a canonical byte stream; import verifies the hash. Column-dX reduction = tree over the 8 canonical output-row chunks of the full N (contract gap, declared).
- runtime/megatron_weights_identity.py: reconstructs all 627 logical HF text tensors (un-fuse qkv/fc1, gather TP) and compares sha256 + torch.equal with safetensors bytes.
- runtime/strict_trainer.py: StrictTrainer (exact orchestrator interface; rank 0 in-process, TP workers spawned and driven by broadcast commands; AdamW on LoRA params only).
- runtime/run_megatron_strict.py: CLI writing the vLLM-format dump with --checks identity,repeat,native,step,grad_repeat,lora_noop,bench.
- runtime/strict_kernels.py: sibling-op resolution with labelled fallbacks (never labelled strict), canonical chunked GEMM wrapper, small-K loop, TP all-gather (NCCL all_gather = data movement; never a reducing collective).

## Weight identity (probe on the real checkpoint, layer 0 + globals)
embed_tokens, lm_head, q/k/v/gate_proj (un-fused), o_proj, gate/up (un-fused), down, all four decoder norms: torch.equal True. model.language_model.norm.weight: FALSE - Bridge RMSNorm2ZeroCenteredRMSNormMapping stores w-1 in bf16; bf16(w_meg+1) differs at 1120/6656 elements (HF span -4.94..4.38). Strict path reloads the HF tensor. Full 627 sweep (--checks identity) not run; expected 626/627 + this exception.

## Runs (a)-(f): NOT RUN (cut-off). Known: Bridge load 39.2 s, 55.5 GiB (incl. 3.6 GiB vision tower, dropped by the trainer).

## Resume
P=~/envs/megatron/bin/python; export PYTHONPATH=$PWD
$P -c "import json;print(json.dumps({\"token_ids\":[list(range(200000,200256))]*8}))" > /tmp/prompts.json
~/bin/gpu-lease -n 1 --pool 6-7 -- $P -m rl_engine.models.muse_glimmer.runtime.run_megatron_strict --tp 1 --prompt-file /tmp/prompts.json --out /tmp/lp_tp1.json --identity-out /tmp/identity.json --checks identity,repeat,lora_noop,native,grad_repeat,step,bench
~/bin/gpu-lease -n 2 --pool 6-7 -- $P -m rl_engine.models.muse_glimmer.runtime.run_megatron_strict --tp 2 --prompt-file /tmp/prompts.json --out /tmp/lp_tp2.json --checks repeat   # then diff logprobs_hex

## Deviations / open issues
1. Column-parallel dX order declared here (not in contract). 2. TP2 attention redundant. 3. final_norm identity exception bypassed. 4. vision tower dropped; --native forward-only. 5. 30B E2E unverified.

## RFC feedback
- megatron_vime_provider: "re-drive the forward from Bridge parameters" not "patch modules"; TP in {1,2} unless KV-group-aware kernels; Bridge unpadded vocab shards must be re-tiled canonically.
- weight_sync_identity: expected Bridge result 626/627 + norm.weight; hash the HF-form tensors the trainer computes with, over the canonical stream in megatron_lora.lora_state_bytes.
- Q1: four logical GEMMs on views of the fused weight (any storage). Q3: unmerged form above on both sides; merging changes bits.

## Unit test status (vllm env, GPU 6; pytest is missing in ~/envs/megatron)
2 passed (fused layout x2), then FAILED test_column_linear_lora_zero_is_bitwise_noop_and_repeatable at line 159 (values differ entirely, not a rounding issue) - likely the strided w view passed to fast_det_gemm_partial or the N-padding path in strict_kernels._GemmBackend.partial; remaining GPU tests not reached (-x). UNRESOLVED at cut-off; resume by running the pytest command above with ~/envs/vllm/bin/python.
