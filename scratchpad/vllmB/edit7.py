from pathlib import Path

p = Path("/home/ubuntu/RL-Kernel/.claude/worktrees/wf_65f5032c-579-2/rl_engine/models/muse_glimmer/runtime/vllm_tp_sim.py")
s = p.read_text()
old = s[s.index('"""Single-GPU simulation'):s.index('from __future__ import annotations')]
new = '''"""Single-GPU simulation of the vLLM strict binding at TP 2/4/8 (one decoder layer + lm_head).

The vLLM engine needs one GPU per TP rank, so TP4/TP8 cannot run on a small lease.  This module
runs the binding's OWN per-rank code for ``tp`` simulated ranks as ``tp`` threads on one device,
with an in-process rank-ordered all-gather (a barrier + ``torch.cat``; pure data movement, like
the NCCL all-gather the engine uses).  Every rank:

* holds its shard exactly as vLLM 0.31 lays it out (``QKVParallelLinear``: q heads
  ``[32r/t, 32(r+1)/t)``, the KV head ``(2r)//t`` replicated when ``t > 2``; ``output_gate_proj`` /
  ``gate_up_proj`` row shards; ``o_proj`` / ``down_proj`` column shards; the padded lm_head split
  into canonical vocab tiles), and the LoRA with ``shard_lora``;
* runs the binding functions unchanged: ``strict_attention_forward`` on a stand-in
  ``MuseGlimmerAttention`` module (the real slicing, head-count, O4, gate side-channel and
  ``o_proj`` tree code; its ``attn`` calls the strict varlen op with the rank's
  ``num_q_heads``/``num_kv_heads`` exactly like the paged backend), ``strict_mlp_forward``,
  ``strict_logits`` (TP-gathered sampling logits) and ``strict_selected_logp``;
* uses the strict sibling ops for norms, qk-norm/RoPE, attention (varlen), SwiGLU.

The result of every rank is compared with ``torch.equal`` against the same code at TP1.  What it
does NOT cover (needs the real engine at that TP): vLLM's paged KV cache layout per rank, the
TP embedding gather, NCCL itself.
"""

'''
s = s.replace(old, new, 1)
start = s.index("def rank_forward(")
end = s.index("def _state(ops: Any, tp: int, rank: int, collective: Any)")
new_fn = '''class _SimAttn:
    """Stand-in for vLLM's ``Attention`` layer: reads the gate side channel and runs the strict
    varlen op on this rank's heads (the paged backend passes the same head counts)."""

    def __init__(self, state: vs.StrictBindingState, positions: torch.Tensor, window: int | None):
        self.state, self.positions, self.window = state, positions, window
        self.num_heads, self.num_kv_heads = vs.rank_head_counts(state.tp_size)
        setattr(self, vs.attn_backend.GATE_ATTR, None)

    def __call__(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        gate = getattr(self, vs.attn_backend.GATE_ATTR)
        if gate is None:
            raise RuntimeError("gate side channel not set")
        t = q.shape[0]
        cu = torch.tensor([0, t], dtype=torch.int32, device=q.device)
        y, _o, _lse = self.state.ops.gated_attention_varlen(
            q.view(t, self.num_heads, D).contiguous(),
            k.view(t, self.num_kv_heads, D).contiguous(),
            v.view(t, self.num_kv_heads, D).contiguous(),
            gate,
            cu,
            cu,
            self.positions.to(torch.int32),
            window=self.window,
            num_q_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
        )
        return y


def _attention_module(
    state: vs.StrictBindingState, rw: dict, positions: torch.Tensor, *, local: bool
) -> SimpleNamespace:
    return SimpleNamespace(
        layer_idx=0 if local else 3,
        qkv_proj=SimpleNamespace(weight=rw["qkv"]),
        output_gate_proj=SimpleNamespace(weight=rw["g"]),
        o_proj=SimpleNamespace(weight=rw["o"]),
        q_size=rw["q_size"],
        kv_size=rw["kv_size"],
        use_rope=local,
        qk_norm=SimpleNamespace(eps=1e-5),
        scale_query_by=3.87,
        attn=_SimAttn(state, positions, 2048 if local else None),
    )


def rank_forward(
    state: vs.StrictBindingState, problem: dict, *, use_lora: bool, local: bool = True
) -> dict[str, torch.Tensor]:
    """The binding's decoder-layer + lm_head code for one (simulated) rank."""

    ops, w, tp, rank = state.ops, problem["w"], state.tp_size, state.tp_rank
    rw = _rank_weights(problem, tp, rank)
    layer = 0 if local else 3
    state.lora = {}
    if use_lora:
        for name, (a, b) in problem["lora"].items():
            state.lora.setdefault(layer, {})[name] = vs.shard_lora(
                name, a, b, tp_size=tp, tp_rank=rank
            )
    x = problem["x"]
    t = x.shape[0]
    positions = torch.arange(t, device=x.device)
    x_n = ops.centered_rmsnorm(x, w["n_in"], eps=1e-5)
    attention = _attention_module(state, rw, positions, local=local)
    attn = vs.strict_attention_forward(state, attention, positions, x_n)
    r, x_f = ops.fused_post_norm_residual_pre_norm(
        attn, x, w["n_pa"], 1e-8, w["n_pf"], 1e-5, pre_form="centered"
    )
    mlp = SimpleNamespace(
        _rlk_layer_idx=layer,
        gate_up_proj=SimpleNamespace(weight=rw["gate_up"]),
        down_proj=SimpleNamespace(weight=rw["down"]),
    )
    f = vs.strict_mlp_forward(state, mlp, x_f)
    post = ops.centered_rmsnorm(f, w["n_po"], eps=1e-8)
    r = (r.float() + post.float()).to(torch.bfloat16)
    hidden = ops.weighted_rmsnorm(r, w["n_fin"], eps=1e-5)
    state.lm_head = SimpleNamespace(weight=rw["lm"])
    logp = vs.strict_selected_logp(state, hidden, problem["targets"])
    logits, z_tiles = vs.strict_logits(state, hidden)
    logp_from_tiles = vs.logp_from_z_tiles(state, z_tiles, problem["targets"])
    return {
        "attn_out": attn,
        "mlp_out": f,
        "hidden": hidden,
        "logp": logp,
        "logp_sampler_path": logp_from_tiles,
        "logits": logits,
        "argmax": logits.argmax(dim=-1),
    }


'''
s = s[:start] + new_fn + s[end:]
old = '''    for use_lora in (False, True):
        ref = rank_forward(_state(ops, 1, 0, None), problem, use_lora=use_lora)
        for tp in tps:'''
new = '''    for use_lora, local in ((False, True), (True, True), (False, False), (True, False)):
        ref = rank_forward(_state(ops, 1, 0, None), problem, use_lora=use_lora, local=local)
        ref_logits = ref.pop("logits")
        for tp in tps:'''
assert old in s
s = s.replace(old, new, 1)
old = '''                    st = _state(ops, tp, rank, _SimCollective(group, rank))
                    outs[rank] = rank_forward(st, problem, use_lora=use_lora)'''
new = '''                    st = _state(ops, tp, rank, _SimCollective(group, rank))
                    outs[rank] = rank_forward(st, problem, use_lora=use_lora, local=local)'''
assert old in s
s = s.replace(old, new, 1)
old = '''            entry: dict[str, Any] = {
                "tp": tp,
                "lora": use_lora,'''
new = '''            entry: dict[str, Any] = {
                "tp": tp,
                "lora": use_lora,
                "layer": "local (RoPE, window 2048)" if local else "global (NoPE)",'''
assert old in s
s = s.replace(old, new, 1)
old = '''            if not errors:
                checks = {}
                for name in ("q", "k", "attn_out", "mlp_out", "hidden", "logp"):
                    checks[name] = [bool(torch.equal(o[name], ref[name])) for o in outs]'''
new = '''            if not errors:
                checks = {}
                for name in (
                    "attn_out", "mlp_out", "hidden", "logp", "logp_sampler_path", "argmax"
                ):
                    checks[name] = [bool(torch.equal(o[name], ref[name])) for o in outs]
                checks["logits"] = [bool(torch.equal(o.pop("logits"), ref_logits)) for o in outs]
                checks["sampler_logp_equals_prefill_logp"] = [
                    bool(torch.equal(o["logp_sampler_path"], o["logp"])) for o in outs
                ]'''
assert old in s
s = s.replace(old, new, 1)
p.write_text(s)
print("ok")
