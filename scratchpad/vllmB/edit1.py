from pathlib import Path

p = Path("/home/ubuntu/RL-Kernel/.claude/worktrees/wf_65f5032c-579-2/rl_engine/models/muse_glimmer/runtime/vllm_strict.py")
s = p.read_text()
old = '''def _bind_model(state: StrictBindingState, model: Any, vllm_config: Any) -> None:
    speculative = (
        vllm_config is not None and getattr(vllm_config, "speculative_config", None) is not None
    )
'''
new = '''def check_strict_engine_config(vllm_config: Any) -> dict[str, str]:
    """Fail closed on vLLM features that address KV pages in the NATIVE layout (or otherwise
    bypass the strict path); returns the per-feature verdicts recorded in provenance.

    The strict backend re-views each layer's KV region as ``(2, B, N, H, D)`` (all K pages, then
    all V pages), so a native page ``b`` (bytes ``[b*N*H*2D, (b+1)*N*H*2D)`` of the layer region)
    is NOT the strict page ``b``: any native code that copies, zeroes, offloads or transfers KV
    pages by block id would move the wrong bytes.  None of them is proven safe, so all are
    rejected.
    """

    if vllm_config is None:
        return {"engine_config": "not checked (no vllm_config: unit test)"}
    verdicts: dict[str, str] = {}
    problems: list[str] = []
    cache = getattr(vllm_config, "cache_config", None)
    parallel = getattr(vllm_config, "parallel_config", None)
    model_config = getattr(vllm_config, "model_config", None)

    def need(ok: bool, name: str, detail: str) -> None:
        verdicts[name] = "passed" if ok else f"REJECTED: {detail}"
        if not ok:
            problems.append(f"{name}: {detail}")

    need(
        not bool(getattr(cache, "enable_prefix_caching", False)),
        "prefix_caching_off",
        "enable_prefix_caching=True (partial-hit copy-on-write copies native page ranges)",
    )
    need(
        getattr(cache, "kv_offloading_size", None) is None,
        "kv_offloading_off",
        f"kv_offloading_size={getattr(cache, 'kv_offloading_size', None)}",
    )
    need(
        getattr(vllm_config, "kv_transfer_config", None) is None,
        "kv_connector_off",
        "kv_transfer_config is set (KV connectors / transfer / offload read native pages)",
    )
    need(
        getattr(vllm_config, "speculative_config", None) is None,
        "speculative_decoding_off",
        "speculative_config is set",
    )
    need(
        not bool(getattr(cache, "kv_sharing_fast_prefill", False)),
        "kv_sharing_fast_prefill_off",
        "kv_sharing_fast_prefill=True",
    )
    dcp = int(getattr(parallel, "decode_context_parallel_size", 1) or 1)
    pcp = int(getattr(parallel, "prefill_context_parallel_size", 1) or 1)
    need(dcp == 1, "decode_context_parallel_off", f"decode_context_parallel_size={dcp}")
    need(pcp == 1, "prefill_context_parallel_off", f"prefill_context_parallel_size={pcp}")
    pp = int(getattr(parallel, "pipeline_parallel_size", 1) or 1)
    need(pp == 1, "pipeline_parallel_off", f"pipeline_parallel_size={pp} (untested)")
    mode = str(getattr(model_config, "logprobs_mode", "raw_logprobs"))
    need(
        mode == "raw_logprobs",
        "logprobs_mode_raw",
        f"logprobs_mode={mode!r} (strict overwrites the raw selected-token logprob only)",
    )
    if problems:
        raise RuntimeError(
            "strict Muse Glimmer binding: engine features that bypass the strict KV/logit path "
            "are enabled (fail closed): " + "; ".join(problems)
        )
    return verdicts


def _bind_model(state: StrictBindingState, model: Any, vllm_config: Any) -> None:
    state.checks.update(check_strict_engine_config(vllm_config))
    speculative = (
        vllm_config is not None and getattr(vllm_config, "speculative_config", None) is not None
    )
'''
assert old in s
s = s.replace(old, new, 1)
old = '''        impl = attention.attn.impl
        if int(impl.num_heads) != NUM_HEADS or int(impl.num_kv_heads) != NUM_KV_HEADS:
            raise RuntimeError(
                f"layer {attention.layer_idx}: vLLM Attention has {impl.num_heads}/"
                f"{impl.num_kv_heads} heads; strict mode needs the full {NUM_HEADS}/{NUM_KV_HEADS}"
            )'''
new = '''        impl = attention.attn.impl
        heads_q, heads_kv = rank_head_counts(state.tp_size)
        if (
            int(impl.num_heads) != heads_q
            or int(impl.num_kv_heads) != heads_kv
            or int(attention.q_size) != heads_q * HEAD_DIM
            or int(attention.kv_size) != heads_kv * HEAD_DIM
        ):
            raise RuntimeError(
                f"layer {attention.layer_idx}: vLLM Attention has {impl.num_heads}/"
                f"{impl.num_kv_heads} heads (q_size {attention.q_size}, kv_size "
                f"{attention.kv_size}); strict TP{state.tp_size} needs {heads_q}/{heads_kv}"
            )'''
assert old in s
s = s.replace(old, new, 1)
old = '''    state.cfg["window_keys_incl_self"] = int(pins["sliding_window"])
    state.cfg["sampling"] = "native"
    state.persist()'''
new = '''    state.cfg["window_keys_incl_self"] = int(pins["sliding_window"])
    state.cfg["sampling"] = "strict_logits"
    heads_q, heads_kv = rank_head_counts(state.tp_size)
    state.checks["attention_heads_per_rank"] = f"{heads_q}/{heads_kv}"
    state.persist()'''
assert old in s
s = s.replace(old, new, 1)
p.write_text(s)
print("ok")
