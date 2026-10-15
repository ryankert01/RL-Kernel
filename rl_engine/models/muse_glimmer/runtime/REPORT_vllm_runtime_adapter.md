# REPORT — `vllm_runtime_adapter` (RFC #501, Muse-Glimmer-30B strict POC)

**STATUS: partial.** Code, unit tests and plugin wiring are complete and committed; the real-model
vLLM runs (a)–(e) did NOT complete in the session window. The first strict TP1 engine launch failed
at model construction (`MuseGlimmerForCausalLM.__init__() missing 'vllm_config'`: vLLM 0.31 inspects
the model `__init__` signature to choose the construction call and my `*args/**kwargs` wrapper hid the
keyword). The fix (explicit `(*, vllm_config, prefix)` signature) is committed but UNTESTED against
the engine. Everything below "Runs" is design + unit-test evidence only.

## Summary
* `vllm_strict.py` — `install_museglimmer_vllm_strict()` patches vLLM 0.31's NATIVE
  `MuseGlimmerForCausalLM` (registry alias `MuseGlimmerForConditionalGeneration`) so the strict
  rollout computes the contract's "Model assembly" with the sibling strict ops (all 14 ops resolve to
  `rl_engine.models.muse_glimmer.ops.*` after merging `feat/museglimmer-arch-fingerprint`; labelled
  fallbacks remain, are probed/demoted per op at model init and are never called "strict").
* `vllm_attention_backend.py` — `FlashAttentionBackend` subclass registered as the override of
  `AttentionBackendEnum.FLASH_ATTN`; routes decoder attention to `strict_gated_attention_fwd_paged`.
* `vllm_plugin.py` — `register()` reached from the repo entry point when `RL_KERNEL_MUSEGLIMMER_STRICT=1`
  (one dispatch block at the top of `register_vllm_plugin()` in
  `rl_engine/integrations/engines/rollout/vllm/runtime.py`; nothing else touched there).
* `strict_rollout_engine.py` — `StrictRolloutEngine` with exactly the requested interface.
* `run_vllm_strict.py` — CLI (`--tp 1|2 --strict/--native --prompt-file --out`, `--prefill-check`,
  `--lora-zero-check`, `--score prev.json`) writing `{token_ids, logprobs, logprobs_hex, prompt_token_ids, meta}`.
* `fallback_ops.py` / `strict_ops.py` — labelled fallbacks + per-op resolver/probe.
* `tests/models/muse_glimmer/test_vllm_strict_binding.py` — 24 passed, 1 skipped (opt-in E2E) on A100.

Env exception used once: `~/.local/bin/uv pip install --python ~/envs/vllm/bin/python --no-deps -e <worktree>`
(entry point visible to vLLM workers; no extension built — uv's isolated build env has no torch).

## Hooks
FLASH_ATTN enum override -> strict backend; `ParallelLMHead.__init__/weight_loader` (vocab 202048 ->
padded 202752 so rank r of t owns tiles [8r/t, 8(r+1)/t)); `MuseGlimmerForCausalLM.__init__`
(`check_runtime_binding`, config pins, eager/bf16-KV checks, TP info, op probe, provenance);
`compute_logits` (native bf16 softcap logits for SAMPLING only + publishes hidden states);
`MuseGlimmerModel.embed_input_ids` (O1; TP>1 exact all-gather of vocab-sharded rows);
`MuseGlimmerDecoderLayer.forward` (O2 incl. fused post/pre norm); `MuseGlimmerAttention.forward`
(four det GEMMs on canonical slices, `qknorm_scaled_rope`, gate side-channel, row-parallel o_proj tree,
LoRA); `MuseGlimmerMLP.forward` (gate/up slices, `swiglu`, row-parallel down tree, LoRA);
`muse_glimmer.Attention` factory (TP>1 full 32/2 heads, KV replicated);
V2 `vllm.v1.worker.gpu.sample.sampler.Sampler.__call__` (sampled-token logprob column overwritten
with strict O7); `prompt_logprob.compute_prompt_logprobs_with_chunking` (prompt-token column, used by
`logprobs()`). Strict: whole model assembly. Native (in provenance): token selection, ranks, extra
top-k columns, KV write (`reshape_and_cache_flash`, a copy).

## Fused-weight slice mapping (rank r of t)
`qkv_proj.weight` rows [0,q_size)=q heads [32r/t,32(r+1)/t); [q_size,q_size+kv_size)=k;
[q_size+kv_size,q_size+2kv_size)=v (q_size=4096/t, kv_size=128*max(1,2/t)); `output_gate_proj.weight`
rows = gate channels of the same q heads. `gate_up_proj.weight` rows [0,F/t)=mlp gate, [F/t,2F/t)=up
— vLLM ALSO fuses gate/up (RFC says "Q/K/V only"). `o_proj (H,4096/t)` / `down_proj (H,F/t)` =
canonical K chunks `rank_chunk_ids(t,r)` (512/2496): `fast_det_gemm_partial` -> rank-ordered
all-gather -> `combine_tree` -> one bf16 cast. lm_head: padded shard = canonical tiles; logp =
`softcap_tile_stats` per tile -> all-gather stats -> `combine_tile_stats` -> `zhat[y]-logZ`.

## vLLM 0.31 notes for the RFC
Model Runner V2 is the CUDA default (patch `gpu.sample.sampler.Sampler.__call__`, not
`v1.sample.Sampler.forward`); `VLLM_ATTENTION_BACKEND` gone -> `attention_config={"backend":"FLASH_ATTN"}`
+ `register_backend` override; `get_kv_cache_shape` gone -> per-layer cache is `(P, H, 16, 2D)` K|V
packed, so the backend reinterprets the (L-outermost, contiguous) layer buffer as `(2,P,16,H,D)` and
owns `do_kv_cache_update`; null/evicted pages are id 0 in vLLM vs -1 in the strict op (mapped);
the loader inspects `__init__` signatures (keep `(*, vllm_config, prefix)`); `LLM.collective_rpc`
carries LoRA tensors/provenance to workers; the strict kernel pins 32/2 heads so TP>1 uses
replicated KV + exact all-gather of q/k/v/gate (bit-identical to TP1) — RFC feedback: make H_Q/H_KV
runtime parameters or state the replicated-KV design.

## LoRA
`load_lora(state_dict, sha256)`: keys `layers.{i}.{q_proj|k_proj|v_proj|gate_proj|o_proj|mlp.gate_proj|mlp.up_proj|mlp.down_proj}.lora_{A,B}`;
column-parallel slots keep A replicated, shard B rows like the base weight; row-parallel slots shard A
columns by canonical K chunks (+tree), keep B. `y = bf16(fp32(main) + fp32(bf16(det_gemm(bf16(det_gemm(x,A)),B))))`,
B-GEMM reduces over K=r in one chunk. Zero B is a bitwise no-op (unit-tested, torch.equal). sha256 recorded.

## Test results
```
~/bin/gpu-lease -n 1 --pool 6-7 -- ~/envs/vllm/bin/python -m pytest tests/models/muse_glimmer/test_vllm_strict_binding.py -q -x
[gpu-lease] using GPUs [6]
........................s                                                [100%]
24 passed, 1 skipped in 11.66s
```

## Runs (NOT completed) — resume commands
```
cd <scratch>/runs
~/bin/gpu-lease -n 1 --pool 6-7 -- ~/envs/vllm/bin/python -m rl_engine.models.muse_glimmer.runtime.run_vllm_strict --tp 1 --native --max-tokens 32 --out native_tp1.json
~/bin/gpu-lease -n 1 --pool 6-7 -- ~/envs/vllm/bin/python -m rl_engine.models.muse_glimmer.runtime.run_vllm_strict --tp 1 --strict --max-tokens 32 --prefill-check --lora-zero-check --out strict_tp1.json
~/bin/gpu-lease -n 1 --pool 6-7 -- ~/envs/vllm/bin/python -m rl_engine.models.muse_glimmer.runtime.run_vllm_strict --tp 1 --strict --score native_tp1.json --out strict_tp1_score.json
~/bin/gpu-lease -n 2 --pool 6-7 -- ~/envs/vllm/bin/python -m rl_engine.models.muse_glimmer.runtime.run_vllm_strict --tp 2 --strict --score native_tp1.json --out strict_tp2_score.json
```
Compare `logprobs_hex` between strict_tp1_score and strict_tp2_score (b); `decode_vs_prefill_equal`
in strict_tp1 (c); `tokens_per_s` native vs strict (d); `meta.workers[*]` provenance (e).

## Deviations / open issues
KV-replicated redundant attention at TP>1; repo `DeterministicCollective` needs `_C` (absent) ->
`torch.distributed.all_gather_into_tensor` (bit-exact data movement) + local tree, recorded in
`backend_ids.collective`; `enforce_eager=True` required; eviction beyond max_model_len untested;
`prompt_logprob_token_ids` path not patched.

## Time spent
~4.5 h.
