from pathlib import Path

p = Path("/home/ubuntu/RL-Kernel/.claude/worktrees/wf_65f5032c-579-2/rl_engine/models/muse_glimmer/runtime/vllm_strict.py")
s = p.read_text()

# strict_logits: empty batch guard
old = '''    z_tiles = strict_z_tiles(state, hidden)
    cfg = state.cfg
    multiplier = float(cfg["output_multiplier"])
    softcap = float(cfg["final_logit_softcapping"])
    rows = hidden.shape[0]
    local = torch.empty('''
new = '''    rows = hidden.shape[0]
    if rows == 0:
        return hidden.new_empty((0, REAL_VOCAB), dtype=torch.float32), []
    z_tiles = strict_z_tiles(state, hidden)
    cfg = state.cfg
    multiplier = float(cfg["output_multiplier"])
    softcap = float(cfg["final_logit_softcapping"])
    local = torch.empty('''
assert old in s
s = s.replace(old, new, 1)

# batch-sharded sampling would bypass compute_logits
old = '''    _bind_final_norm(state, model.model.norm)
    state.model = model'''
new = '''    if hasattr(model, "compute_logits_local"):
        raise RuntimeError(
            "strict Muse Glimmer: the model exposes compute_logits_local (batch-sharded sampling "
            "would bypass the strict compute_logits); fail closed"
        )
    state.checks["batch_sharded_sampling"] = "inactive (no compute_logits_local)"
    _bind_final_norm(state, model.model.norm)
    state.model = model'''
assert old in s
s = s.replace(old, new, 1)

# worker_load_lora: recompute the canonical hash over the tensors actually sharded
start = s.index("def worker_load_lora(\n")
end = s.index("def worker_load_lora_file(")
new = '''def worker_load_lora(
    _worker: Any, state_dict: Mapping[str, torch.Tensor], sha256: str
) -> dict[str, Any]:
    """``LLM.collective_rpc`` target: install strict unmerged LoRA slots on this rank.

    The rank converts the received tensors to bf16 (a no-op for a valid package), RECOMPUTES the
    canonical ``muse_glimmer_lora_sha256_v1`` hash over exactly those full tensors (the ones it
    then shards and copies to the device), refuses the load if it differs from the declared
    hash, and from then on reports the recomputed value (``provenance()["lora_sha256"]``), never
    the declared string.  ``sha256 == ""`` with an empty state dict clears the slots.
    """

    from rl_engine.models.muse_glimmer.e2e.sync import lora_sha256

    state = require_state()
    if state.model is None:
        raise RuntimeError("LoRA can only be loaded after the Muse Glimmer model is bound")
    device = next(state.model.parameters()).device
    t0 = time.perf_counter()
    loaded = {key: value.to(torch.bfloat16).contiguous() for key, value in state_dict.items()}
    recomputed = lora_sha256(loaded) if loaded else None
    hash_ms = (time.perf_counter() - t0) * 1e3
    if loaded and recomputed != str(sha256):
        raise RuntimeError(
            f"rank {state.tp_rank}: the LoRA tensors it loaded hash to {recomputed}, declared "
            f"{sha256} (muse_glimmer_lora_sha256_v1; fail closed, previous adapter kept)"
        )
    if not loaded and str(sha256):
        raise RuntimeError(f"rank {state.tp_rank}: empty LoRA state dict declared as {sha256}")
    pairs = parse_lora_state_dict(loaded)
    new_slots: dict[int, dict[str, tuple[torch.Tensor, torch.Tensor]]] = {}
    for (layer, name), (lora_a, lora_b) in pairs.items():
        a_local, b_local = shard_lora(
            name, lora_a, lora_b, tp_size=state.tp_size, tp_rank=state.tp_rank
        )
        new_slots.setdefault(layer, {})[name] = (
            a_local.to(device).contiguous(),
            b_local.to(device).contiguous(),
        )
    state.lora = new_slots
    state.lora_sha256 = recomputed if pairs else None
    state.lora_sha256_declared = str(sha256) if pairs else None
    state.call_counts["load_lora"] += 1
    state.persist()
    return {
        "rank": state.tp_rank,
        "layers": len(new_slots),
        "slots": sum(len(v) for v in new_slots.values()),
        "sha256": state.lora_sha256,
        "recomputed_sha256": state.lora_sha256,
        "declared_sha256": state.lora_sha256_declared,
        "rehash_ms": hash_ms,
        "install_ms": (time.perf_counter() - t0) * 1e3,
        "rl_engine_file": _rl_engine_file(),
    }


'''
s = s[:start] + new + s[end:]

# worker_check_lora_rule: formula independent of the binding helpers
old = '''    * ``binding``: the forward path the engine uses (``column_parallel_project`` /
      ``row_parallel_project``);
    * ``formula``: ``bf16(acc32(W x) + loop_k B[:, k] z[k])`` with ``z = bf16(acc32(A x))``, the
      fp32 loop written out here independently (exact bf16 products, sequential fp32 adds);'''
new = '''    * ``binding``: the forward path the engine uses (``column_parallel_project`` /
      ``row_parallel_project``);
    * ``formula``: ``bf16(acc32(W x) + loop_k B[:, k] z[k])`` with ``z = bf16(acc32(A x))``
      written out here WITHOUT the binding's helpers: ``acc32`` and ``A x`` call the strict GEMM
      module ``ops.det_gemm.fast_det_gemm`` directly (imported here, unpadded shapes, not the
      binding's op table or ``_padded_partial``) and the rank loop is a Python loop of exact bf16
      products and sequential fp32 adds;
    * ``golden``: the same formula with the fp32 golden GEMM ``fallback_ops.det_gemm`` (a
      different kernel: tolerance view, ``max_abs_vs_fp32_golden``, not expected bitwise);'''
assert old in s
s = s.replace(old, new, 1)
old = '''    from rl_engine.models.muse_glimmer.runtime import megatron_lora as ml
    from rl_engine.models.muse_glimmer.runtime.strict_kernels import TPComm
'''
new = '''    from rl_engine.models.muse_glimmer.ops.det_gemm import fast_det_gemm
    from rl_engine.models.muse_glimmer.runtime import megatron_lora as ml
    from rl_engine.models.muse_glimmer.runtime.strict_kernels import TPComm

    def formula(gemm: Callable[..., torch.Tensor], x, w, lora_a, lora_b) -> torch.Tensor:
        acc = gemm(x, w, out_dtype=torch.float32)
        z = gemm(x, lora_a, out_dtype=torch.float32).to(torch.bfloat16)
        zf, bf = z.float(), lora_b.float()
        loop = torch.zeros_like(acc)
        for k in range(zf.shape[1]):
            loop = loop + zf[:, k : k + 1] * bf[:, k].unsqueeze(0)
        return (acc + loop).to(torch.bfloat16), loop
'''
assert old in s
s = s.replace(old, new, 1)
old = '''        if name in _COLUMN_PARALLEL:
            y_bind = column_parallel_project(state, x, w, (lora_a, lora_b))
            y_base = column_parallel_project(state, x, w)
            acc = state.ops.det_gemm(x, w, out_dtype=torch.float32)
            y_meg = ml.StrictColumnLinear.apply(
                x, lora_a.float(), lora_b.float(), comm, int(w.shape[0]), w
            )
        else:
            y_bind = row_parallel_project(state, x, w, (lora_a, lora_b))
            y_base = row_parallel_project(state, x, w)
            acc = state.ops.det_gemm_partial(x, w, tuple(range(CANONICAL_TP)))
            y_meg = ml.StrictRowLinear.apply(
                x, w, lora_a.float(), lora_b.float(), comm, int(w.shape[1])
            )
        z = _padded_partial(state, x, lora_a, tuple(range(CANONICAL_TP))).to(torch.bfloat16)
        zf, bf = z.float(), lora_b.float()
        loop = torch.zeros_like(acc)
        for k in range(zf.shape[1]):
            loop = loop + zf[:, k : k + 1] * bf[:, k].unsqueeze(0)
        y_formula = (acc + loop).to(torch.bfloat16)
        ref64 = x.double() @ w.double().t() + (x.double() @ lora_a.double().t()) @ (
            lora_b.double().t()
        )
        out["slots"][name] = {
            "shape_x": list(x.shape),
            "shape_w": list(w.shape),
            "rank": int(lora_a.shape[0]),
            "binding_equals_formula": bool(torch.equal(y_bind, y_formula)),
            "binding_equals_megatron": bool(torch.equal(y_bind, y_meg.to(torch.bfloat16))),'''
new = '''        if name in _COLUMN_PARALLEL:
            y_bind = column_parallel_project(state, x, w, (lora_a, lora_b))
            y_base = column_parallel_project(state, x, w)
            y_meg = ml.StrictColumnLinear.apply(
                x, lora_a.float(), lora_b.float(), comm, int(w.shape[0]), w
            )
        else:
            y_bind = row_parallel_project(state, x, w, (lora_a, lora_b))
            y_base = row_parallel_project(state, x, w)
            y_meg = ml.StrictRowLinear.apply(
                x, w, lora_a.float(), lora_b.float(), comm, int(w.shape[1])
            )
        y_formula, loop = formula(fast_det_gemm, x, w, lora_a, lora_b)
        y_golden, _loop = formula(fb.det_gemm, x, w, lora_a, lora_b)
        ref64 = x.double() @ w.double().t() + (x.double() @ lora_a.double().t()) @ (
            lora_b.double().t()
        )
        out["slots"][name] = {
            "shape_x": list(x.shape),
            "shape_w": list(w.shape),
            "rank": int(lora_a.shape[0]),
            "binding_equals_formula": bool(torch.equal(y_bind, y_formula)),
            "binding_equals_megatron": bool(torch.equal(y_bind, y_meg.to(torch.bfloat16))),
            "binding_equals_fp32_golden_bits": bool(torch.equal(y_bind, y_golden)),
            "max_abs_vs_fp32_golden": float((y_bind.double() - y_golden.double()).abs().max()),'''
assert old in s
s = s.replace(old, new, 1)
s = s.replace('''    "column_parallel_project",
    "get_state",''', '''    "check_prompt_logprobs_request",
    "check_sampling_request",
    "check_strict_engine_config",
    "column_parallel_project",
    "get_state",''', 1)
s = s.replace('''    "lm_head_tiles_for_rank",
    "parse_lora_state_dict",''', '''    "lm_head_tiles_for_rank",
    "logp_from_z_tiles",
    "parse_lora_state_dict",
    "rank_head_counts",''', 1)
s = s.replace('''    "strict_selected_logp",
    "tree_over_ranks",''', '''    "strict_attention_forward",
    "strict_logits",
    "strict_mlp_forward",
    "strict_selected_logp",
    "strict_z_tiles",
    "tree_over_ranks",''', 1)
p.write_text(s)
print("ok")
