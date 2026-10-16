from pathlib import Path

p = Path("/home/ubuntu/RL-Kernel/.claude/worktrees/wf_65f5032c-579-2/rl_engine/models/muse_glimmer/runtime/vllm_strict.py")
s = p.read_text()

# 1. Drop gather_heads + the KV-replicating Attention factory; add the module-level attention body.
start = s.index("def gather_heads(")
end = s.index("def _patch_model(state: StrictBindingState) -> None:")
new_block = '''def strict_attention_forward(
    state: StrictBindingState, module: Any, positions: torch.Tensor, hidden_states: torch.Tensor
) -> torch.Tensor:
    """The strict ``MuseGlimmerAttention.forward`` body on THIS rank's heads (also driven by the
    TP simulation, so the real slicing code is what the simulation checks).

    Four det GEMMs on the rank's canonical slices of ``qkv_proj`` / ``output_gate_proj`` (+ strict
    LoRA), O4 on the rank's ``32/t`` q heads and ``max(1, 2/t)`` kv heads, the gate on the side
    channel, ``module.attn`` (the strict paged backend with ``num_q_heads``/``num_kv_heads`` =
    the rank's heads), then the row-parallel ``o_proj`` tree.  No head all-gather: heads never
    interact in O5, so the rank's attention output is the matching slice of the TP1 output bit
    for bit, and it is exactly the K columns of the rank's ``o_proj`` shard.
    """

    ops = state.ops
    layer_idx = int(module.layer_idx)
    lora = state.lora.get(layer_idx, {})
    heads_q, heads_kv = rank_head_counts(state.tp_size)
    if int(module.q_size) != heads_q * HEAD_DIM or int(module.kv_size) != heads_kv * HEAD_DIM:
        raise RuntimeError(
            f"layer {layer_idx}: q_size/kv_size {module.q_size}/{module.kv_size} do not match "
            f"TP{state.tp_size} heads {heads_q}/{heads_kv} (fail closed)"
        )
    w_q, w_k, w_v = canonical_qkv_slices(module.qkv_proj.weight, module.q_size, module.kv_size)
    w_g = module.output_gate_proj.weight
    q = column_parallel_project(state, hidden_states, w_q, lora.get("q_proj"))
    k = column_parallel_project(state, hidden_states, w_k, lora.get("k_proj"))
    v = column_parallel_project(state, hidden_states, w_v, lora.get("v_proj"))
    gate = column_parallel_project(state, hidden_states, w_g, lora.get("gate_proj"))
    tokens = hidden_states.shape[0]
    q_heads = q.view(tokens, heads_q, HEAD_DIM)
    k_heads = k.view(tokens, heads_kv, HEAD_DIM)
    if module.use_rope:
        cos, sin = _rope_for(state, positions)
    else:
        cos = sin = None
    q_heads, k_heads = ops.qknorm_scaled_rope(
        q_heads,
        k_heads,
        cos,
        sin,
        eps=float(module.qk_norm.eps),
        qk_scale=float(module.scale_query_by),
    )
    setattr(module.attn, attn_backend.GATE_ATTR, gate.contiguous())
    try:
        attended = module.attn(
            q_heads.reshape(tokens, heads_q * HEAD_DIM),
            k_heads.reshape(tokens, heads_kv * HEAD_DIM),
            v,
        )
    finally:
        setattr(module.attn, attn_backend.GATE_ATTR, None)
    state.call_counts["attention_forward"] += 1
    return row_parallel_project(
        state, attended.contiguous(), module.o_proj.weight, lora.get("o_proj")
    )


def strict_mlp_forward(state: StrictBindingState, module: Any, x: torch.Tensor) -> torch.Tensor:
    """The strict ``MuseGlimmerMLP.forward`` body (canonical gate/up slices, O6, down tree)."""

    lora = state.lora.get(int(getattr(module, "_rlk_layer_idx", -1)), {})
    w_gate, w_up = canonical_gate_up_slices(module.gate_up_proj.weight)
    gate = column_parallel_project(state, x, w_gate, lora.get("mlp.gate_proj"))
    up = column_parallel_project(state, x, w_up, lora.get("mlp.up_proj"))
    hidden = state.ops.swiglu(gate, up)
    state.call_counts["mlp_forward"] += 1
    return row_parallel_project(state, hidden, module.down_proj.weight, lora.get("mlp.down_proj"))


def _forbid(state: StrictBindingState, what: str) -> Callable[..., Any]:
    def guard(*_args: Any, **_kwargs: Any) -> Any:
        state.call_counts[f"forbidden.{what}"] += 1
        raise RuntimeError(
            f"strict Muse Glimmer: {what} would address KV pages in vLLM's native layout, but the "
            "strict backend re-views every layer region as (2, B, N, H, D) (fail closed)"
        )

    return guard


def _patch_kv_page_writers(state: StrictBindingState) -> None:
    """Make the native page-addressed KV writers of the model runners raise (defence in depth
    behind :func:`check_strict_engine_config`)."""

    from vllm.v1.worker import utils as worker_utils
    from vllm.v1.worker.gpu import model_runner as v2_runner

    if getattr(worker_utils.KVBlockZeroer, _MARK, None) is None:
        setattr(worker_utils.KVBlockZeroer, _MARK, worker_utils.KVBlockZeroer.zero_block_ids)
        worker_utils.KVBlockZeroer.zero_block_ids = _forbid(state, "KVBlockZeroer.zero_block_ids")
        state.hooks.append("vllm.v1.worker.utils.KVBlockZeroer.zero_block_ids (raises)")
    modules: list[Any] = [v2_runner]
    try:
        from vllm.v1.worker import gpu_model_runner as v1_runner

        modules.append(v1_runner)
    except Exception:  # noqa: BLE001 - the V1 runner is not used by the 0.31 CUDA default
        pass
    for module in modules:
        if getattr(module, _MARK + "copy", None) is not None:
            continue
        setattr(module, _MARK + "copy", module.copy_kv_cache_blocks_inplace)
        module.copy_kv_cache_blocks_inplace = _forbid(state, "copy_kv_cache_blocks_inplace")
        state.hooks.append(f"{module.__name__}.copy_kv_cache_blocks_inplace (raises)")
    for name in ("_init_kv_zero_meta",):
        cls = v2_runner.GPUModelRunner
        if getattr(cls, _MARK + name, None) is None and hasattr(cls, name):
            setattr(cls, _MARK + name, getattr(cls, name))
            setattr(cls, name, _forbid(state, f"GPUModelRunner.{name} (KV-block zeroing)"))
            state.hooks.append(f"{v2_runner.__name__}.GPUModelRunner.{name} (raises)")


'''
s = s[:start] + new_block + s[end:]

# 2. _patch_model: strict compute_logits, module-level bodies, no factory.
old = '''    cls = mg.MuseGlimmerForCausalLM
    if hasattr(cls, _MARK):
        return
    _patch_attention_factory(state)
    original_init = cls.__init__'''
new = '''    cls = mg.MuseGlimmerForCausalLM
    if hasattr(cls, _MARK):
        return
    original_init = cls.__init__'''
assert old in s
s = s.replace(old, new, 1)
old = '''    def compute_logits(self: Any, hidden_states: torch.Tensor) -> torch.Tensor | None:
        logits = original_logits(self, hidden_states)
        state.logits_ctx = hidden_states
        state.call_counts["compute_logits.native_for_sampling"] += 1
        return logits
'''
new = '''    def compute_logits(self: Any, hidden_states: torch.Tensor) -> torch.Tensor | None:
        # Strict sampling logits (module docstring); the native cuBLAS lm_head is never run.
        logits, z_tiles = strict_logits(state, hidden_states)
        if state.stash_logits:
            # The sampler patch turns the same fp32 tiles into the selected-token O7 logprob.
            state.logits_ctx = (hidden_states, z_tiles)
        state.call_counts["compute_logits.strict"] += 1
        state.call_counts["compute_logits.strict_rows"] += int(hidden_states.shape[0])
        return logits
'''
assert old in s
s = s.replace(old, new, 1)
start = s.index("    def attention_forward(\n        self: Any, positions: torch.Tensor, hidden_states: torch.Tensor")
end = s.index("    setattr(cls, _MARK, original_init)")
new = '''    def attention_forward(
        self: Any, positions: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        return strict_attention_forward(state, self, positions, hidden_states)

    def mlp_forward(self: Any, x: torch.Tensor) -> torch.Tensor:
        return strict_mlp_forward(state, self, x)

'''
s = s[:start] + new + s[end:]
s = s.replace('''    original_logits = cls.compute_logits
''', '''    original_logits = cls.compute_logits  # kept under _MARK for provenance only
''', 1)
old = '''    setattr(cls, _MARK, original_init)
    cls.__init__ = init_wrapped'''
new = '''    setattr(cls, _MARK, original_init)
    setattr(cls, _MARK + "compute_logits", original_logits)
    cls.__init__ = init_wrapped'''
assert old in s
s = s.replace(old, new, 1)
p.write_text(s)
print("ok")
