from pathlib import Path
import sys

p = Path(sys.argv[1])
s = p.read_text()


def rep(old, new, count=1):
    global s
    assert s.count(old) == count, (old[:90], s.count(old))
    s = s.replace(old, new)


rep('''``L.self_attention.q/k_layernorm``   O4 ``qknorm_scaled_rope`` (3.87 on q, RoPE on sliding layers)
``L.self_attention.core_attention``  O5 ``gated_attention`` (gate inside, window from config)''',
    '''``L.self_attention.q/k_layernorm``   O4 ``qknorm_scaled_rope`` on the rank's heads (3.87 on q, RoPE)
``L.self_attention.core_attention``  O5 per-rank ``gated_attention`` (32/t Q heads, KV head (2r)//t)''')

rep('''so HF ``q_proj`` = concat over g of the first block, ``gate_proj`` the second, ``k_proj``/``v_proj``
the third/fourth. At TP ``t <= 2`` rank ``r``'s MCore shard is groups ``[2r/t, 2(r+1)/t)``
and its canonical q/gate/k/v row chunks are views of it. At TP 4/8 MCore splits the fused rows
evenly (no clean head partition); the binding re-shards them to canonical chunks at bind time
(see ``MuseGlimmerStrictBinding._qkv_views``). Attention runs on all 32/2 heads on every rank
(head shards all-gathered), so it is the TP1 computation at every TP.
"""''', '''so HF ``q_proj`` = concat over g of the first block, ``gate_proj`` the second, ``k_proj``/``v_proj``
the third/fourth. At TP ``t <= 2`` rank ``r``'s MCore shard is groups ``[2r/t, 2(r+1)/t)``
and its canonical q/gate/k/v row chunks are views of it. At TP 4/8 MCore splits the fused rows
evenly (no clean head partition); the binding re-shards them to canonical chunks at bind time
(see ``MuseGlimmerStrictBinding._qkv_views``).

Attention runs PER RANK on the rank's own heads (``runtime.megatron_attention``): Q heads
``[32r/t, 32(r+1)/t)`` (= the rank's canonical q/gate rows and ``o_proj`` input columns, so
nothing is gathered) and KV head ``(2r)//t``; at TP4/8 the KV head is replicated (its k/v
columns are all-gathered among the ranks, data movement) and the replicas' fp32 dK/dV nodes are
folded with the canonical tree before the bf16 rounding. Every bit equals TP1.

Strict mode fails closed: every op must resolve to its sibling strict kernel
(``strict_kernels.resolve_ops``) unless ``allow_fallback=True`` (unit tests only).
"""''')

rep('''import json
import math
import os
import time''', '''import gc
import json
import math
import os
import time''')

rep('''from rl_engine.models.muse_glimmer.runtime import strict_kernels as sk
from rl_engine.models.muse_glimmer.runtime.megatron_lora import (
    DEFAULT_LORA_RANK,
    LayerLoRA,''', '''from rl_engine.models.muse_glimmer.runtime import strict_kernels as sk
from rl_engine.models.muse_glimmer.runtime.megatron_attention import (
    attention_core,
    rank_head_geometry,
)
from rl_engine.models.muse_glimmer.runtime.megatron_lora import (
    DEFAULT_LORA_RANK,
    LORA_GRAD_RULE,
    LayerLoRA,''')

rep('''_FINAL_NORM_HF_KEY = "model.language_model.norm.weight"
SUPPORTED_TP = (1, 2, 4, 8)
''', '''_FINAL_NORM_HF_KEY = "model.language_model.norm.weight"
SUPPORTED_TP = (1, 2, 4, 8)
# TP sizes with a real 30B run on record (identity, xrt, grads vs TP1); others need the opt-in.
RUN_TESTED_TP = (1, 2, 4)
UNTESTED_TP_ENV = "MUSE_MEGATRON_ALLOW_UNTESTED_TP"
VISION_MODULES = ("vision_tower", "vision_adapter", "vision_projection")
''')

rep('''    if tp_world not in SUPPORTED_TP:
        raise StrictBindingError(
            f"tp={tp_world}: the strict binding supports TP {SUPPORTED_TP} (8 canonical chunks)"
        )
''', '''    if tp_world not in SUPPORTED_TP:
        raise StrictBindingError(
            f"tp={tp_world}: the strict binding supports TP {SUPPORTED_TP} (8 canonical chunks)"
        )
    if tp_world not in RUN_TESTED_TP and os.environ.get(UNTESTED_TP_ENV) != str(tp_world):
        raise StrictBindingError(
            f"tp={tp_world} has no real-model run on record (tested: {RUN_TESTED_TP}); set "
            f"{UNTESTED_TP_ENV}={tp_world} to run it as an experiment"
        )
''')

rep('''def _load_hf_tensor(hf_path: str | Path, name: str) -> torch.Tensor:''', '''def drop_vision_modules(model: Any) -> dict[str, Any]:
    """Delete the vision tower / adapter / projection and RELEASE their GPU memory.

    Bridge's ``hook_hf_module_setattr_for_tp_grad_sync`` stores a bound ``__setattr__`` in every
    vision submodule's ``__dict__`` (a reference cycle), so ``delattr`` alone frees nothing until
    a garbage collection; the parameter storages are released explicitly here, the cycles broken
    and collected. Returns the vision parameter bytes and the measured drop of
    ``torch.cuda.memory_allocated`` (both GiB).
    """
    gib = 2**30
    present = [name for name in VISION_MODULES if hasattr(model, name)]
    param_bytes = 0
    before = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    for name in present:
        module = getattr(model, name)
        for sub in module.modules():
            sub.__dict__.pop("__setattr__", None)  # break the bound-method cycle
            for _, param in list(sub.named_parameters(recurse=False)):
                param_bytes += param.numel() * param.element_size()
                param.data = torch.empty(0, dtype=param.dtype, device=param.device)
            for key, buf in list(sub._buffers.items()):
                if buf is not None:
                    sub._buffers[key] = torch.empty(0, dtype=buf.dtype, device=buf.device)
        delattr(model, name)
        del module
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    after = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    return {
        "dropped": present,
        "vision_param_gib": param_bytes / gib,
        "allocated_drop_gib": (before - after) / gib,
    }


def _load_hf_tensor(hf_path: str | Path, name: str) -> torch.Tensor:''')

rep('''        drop_vision: bool,
        lora_b_init_std: float = 0.0,
    ) -> None:
        self.model = model
        self.dims = _dims_from_manifest()
        self.ops = sk.resolve_ops()
        self.flags = sk.set_deterministic_torch()''', '''        drop_vision: bool,
        lora_b_init_std: float = 0.0,
        allow_fallback: bool = False,
    ) -> None:
        self.model = model
        self.dims = _dims_from_manifest()
        self.allow_fallback = bool(allow_fallback)
        # Fail closed: raises StrictOpsUnavailable unless every op is a sibling strict kernel.
        self.ops = sk.resolve_ops(allow_fallback=self.allow_fallback)
        self.flags = sk.set_deterministic_torch()''')

rep('''        self.lora_rank = int(lora_rank)
        self.seed = int(seed)
        self._rope_cache: dict[tuple[int, ...], tuple[torch.Tensor, torch.Tensor]] = {}
        self.notes: list[str] = []

        for p in model.parameters():
            p.requires_grad_(False)
        gib = 2**30
        self.memory_gib: dict[str, float] = {"bound_start": torch.cuda.memory_allocated() / gib}
        if drop_vision:
            for name in ("vision_tower", "vision_adapter", "vision_projection"):
                if hasattr(model, name):
                    delattr(model, name)
                    self.notes.append(f"dropped {name} (text-only trainer)")
            torch.cuda.empty_cache()
        self.memory_gib["after_vision_drop"] = torch.cuda.memory_allocated() / gib
''', '''        self.lora_rank = int(lora_rank)
        self.seed = int(seed)
        self.head_geometry = rank_head_geometry(self.tp_rank, self.tp_world)
        self.notes: list[str] = []

        for p in model.parameters():
            p.requires_grad_(False)
        gib = 2**30
        self.memory_gib: dict[str, float] = {"bound_start": torch.cuda.memory_allocated() / gib}
        self.vision_drop: dict[str, Any] | None = None
        if drop_vision:
            self.vision_drop = drop_vision_modules(model)
            self.notes.append(
                f"dropped {self.vision_drop['dropped']} (text-only trainer): "
                f"{self.vision_drop['vision_param_gib']:.3f} GiB of vision parameters, "
                f"memory_allocated fell by {self.vision_drop['allocated_drop_gib']:.3f} GiB"
            )
        self.memory_gib["after_vision_drop"] = torch.cuda.memory_allocated() / gib
        self.rope_table_sha256 = None
        if self.ops.rope_table_sha256 is not None:
            # the canonical (cached, fixed-size) RoPE table; logged so both runtimes can compare
            self.rope_table_sha256 = self.ops.rope_table_sha256(self.device)
''')

rep('''    def _col(self, x: torch.Tensor, blocks: Sequence[torch.Tensor], layer: int, name: str, n_total: int):
        a, b = self._lora_pair(layer, name)
        return StrictColumnLinear.apply(x, a, b, self.comm, n_total, *blocks)

    def _row(self, x: torch.Tensor, w_local: torch.Tensor, layer: int, name: str, k_total: int):
        a, b = self._lora_pair(layer, name)
        return StrictRowLinear.apply(x, w_local, a, b, self.comm, k_total)

    def rope(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        key = tuple(positions.tolist()) if positions.numel() <= 4096 else None
        if key is not None and key in self._rope_cache:
            return self._rope_cache[key]
        cos, sin = self.ops.rope_table(
            positions, theta=self.dims.rope_theta, dim=self.dims.head_dim, device=self.device
        )
        if key is not None:
            self._rope_cache[key] = (cos, sin)
        return cos, sin
''', '''    def _col(self, x: torch.Tensor, blocks: Sequence[torch.Tensor], layer: int, name: str,
             n_total: int, segments):
        a, b = self._lora_pair(layer, name)
        return StrictColumnLinear.apply(x, a, b, self.comm, n_total, *blocks, segments=segments)

    def _row(self, x: torch.Tensor, w_local: torch.Tensor, layer: int, name: str, k_total: int,
             segments):
        a, b = self._lora_pair(layer, name)
        return StrictRowLinear.apply(x, w_local, a, b, self.comm, k_total, segments=segments)

    def rope(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """fp32 ``(cos, sin)`` rows for ``positions``: exact copies (``index_select``) of the
        canonical cached table (``ops.qknorm_rope.rope_rows``); fixed memory, no per-packing
        cache."""
        return self.ops.rope_rows(positions.to(self.device), device=self.device)
''')

rep('''    def _layer(
        self,
        i: int,
        resid: torch.Tensor,
        x_n: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cu_seqlens: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        d = self.dims
        lw = self.layers[i]
        t = x_n.shape[0]
        q = self._col(x_n, lw.q_blocks, i, "q_proj", d.q_width)
        k = self._col(x_n, lw.k_blocks, i, "k_proj", d.kv_width)
        v = self._col(x_n, lw.v_blocks, i, "v_proj", d.kv_width)
        g = self._col(x_n, lw.gate_blocks, i, "gate_proj", d.q_width)
        if self.tp_world > 1:
            # The strict O4/O5 kernels take the full 32 Q / 2 KV heads: gather the canonical
            # column shards (data movement only), compute attention redundantly (= TP1 bits and
            # TP1 dQ/dK/dV on every rank), keep the local head columns.
            q = GatherHeads.apply(q, self.comm)
            k = GatherHeads.apply(k, self.comm)
            v = GatherHeads.apply(v, self.comm)
            g = GatherHeads.apply(g, self.comm)
        q3 = q.view(t, d.heads, d.head_dim)
        k3 = k.view(t, d.kv_heads, d.head_dim)
        if lw.sliding:
            qh, kh = self.ops.qknorm_scaled_rope(q3, k3, cos, sin, eps=d.rms_eps, qk_scale=d.qk_scale)
            window: int | None = d.window
        else:
            qh, kh = self.ops.qknorm_scaled_rope(q3, k3, None, None, eps=d.rms_eps, qk_scale=d.qk_scale)
            window = None
        y = self.ops.gated_attention(
            qh, kh, v.view(t, d.kv_heads, d.head_dim), g, cu_seqlens, cu_seqlens, positions, window=window
        )
        y_local = SliceLocalHeads.apply(y.reshape(t, d.q_width), self.comm)
        o = self._row(y_local, lw.o_local, i, "o_proj", d.q_width)
        resid, x_f = self.ops.fused_post_norm_residual_pre_norm(
            o, resid, lw.w_post_attn, d.post_eps, lw.w_pre_ffn, d.rms_eps, pre_form="centered"
        )
        gm = self._col(x_f, (lw.fc1_gate_local,), i, "mlp.gate_proj", d.ffn)
        up = self._col(x_f, (lw.fc1_up_local,), i, "mlp.up_proj", d.ffn)
        h = self.ops.silu_mul(gm, up)
        f = self._row(h, lw.down_local, i, "mlp.down_proj", d.ffn)''', '''    def _layer(
        self,
        i: int,
        resid: torch.Tensor,
        x_n: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cu_seqlens: torch.Tensor,
        positions: torch.Tensor,
        segments,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        d = self.dims
        lw = self.layers[i]
        q = self._col(x_n, lw.q_blocks, i, "q_proj", d.q_width, segments)
        k = self._col(x_n, lw.k_blocks, i, "k_proj", d.kv_width, segments)
        v = self._col(x_n, lw.v_blocks, i, "v_proj", d.kv_width, segments)
        g = self._col(x_n, lw.gate_blocks, i, "gate_proj", d.q_width, segments)
        # Per-rank heads: q/g/y stay local; k/v of the rank's KV head (gathered at TP4/8).
        y_local = attention_core(
            q, k, v, g, ops=self.ops, comm=self.comm,
            cos=cos if lw.sliding else None, sin=sin if lw.sliding else None,
            cu_seqlens=cu_seqlens, positions=positions,
            window=d.window if lw.sliding else None, rms_eps=d.rms_eps, qk_scale=d.qk_scale,
        )
        del q, k, v, g
        o = self._row(y_local, lw.o_local, i, "o_proj", d.q_width, segments)
        resid, x_f = self.ops.fused_post_norm_residual_pre_norm(
            o, resid, lw.w_post_attn, d.post_eps, lw.w_pre_ffn, d.rms_eps, pre_form="centered"
        )
        gm = self._col(x_f, (lw.fc1_gate_local,), i, "mlp.gate_proj", d.ffn, segments)
        up = self._col(x_f, (lw.fc1_up_local,), i, "mlp.up_proj", d.ffn, segments)
        h = self.ops.silu_mul(gm, up)
        del gm, up
        f = self._row(h, lw.down_local, i, "mlp.down_proj", d.ffn, segments)''')

rep('''    def hidden_states(
        self, tokens: torch.Tensor, cu_seqlens: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        """Final-normed hidden states ``(T, H)`` bf16 for packed tokens."""
        d = self.dims
        resid = self.ops.normalized_embedding(self.embed_full, tokens, eps=d.rms_eps)
        x_n = self.ops.centered_rmsnorm(resid, self.layers[0].w_input, eps=d.rms_eps)
        cos, sin = self.rope(positions)
        cu = cu_seqlens.to(device=self.device, dtype=torch.int32)
        pos = positions.to(device=self.device, dtype=torch.int32)
        for i in range(d.layers):
            if self.checkpoint_layers and torch.is_grad_enabled():
                resid, x_n = torch_checkpoint(
                    self._layer, i, resid, x_n, cos, sin, cu, pos, use_reentrant=False
                )
            else:
                resid, x_n = self._layer(i, resid, x_n, cos, sin, cu, pos)
        return x_n  # final norm output (weighted form)
''', '''    def hidden_states(
        self,
        tokens: torch.Tensor,
        cu_seqlens: torch.Tensor,
        positions: torch.Tensor,
        segments: Sequence[tuple[int, int]] | None = None,
    ) -> torch.Tensor:
        """Final-normed hidden states ``(T, H)`` bf16 for packed tokens.

        ``segments`` are the packed sequences' ``(start, end)`` token ranges (default: derived
        from ``cu_seqlens``); they select the packing-invariant LoRA gradient rule.
        """
        d = self.dims
        if segments is None:
            bounds = cu_seqlens.tolist()
            segments = tuple(zip(bounds[:-1], bounds[1:]))
        segments = tuple((int(s), int(e)) for s, e in segments)
        resid = self.ops.normalized_embedding(self.embed_full, tokens, eps=d.rms_eps)
        x_n = self.ops.centered_rmsnorm(resid, self.layers[0].w_input, eps=d.rms_eps)
        cos, sin = self.rope(positions)
        cu = cu_seqlens.to(device=self.device, dtype=torch.int32)
        pos = positions.to(device=self.device, dtype=torch.int32)
        for i in range(d.layers):
            if self.checkpoint_layers and torch.is_grad_enabled():
                resid, x_n = torch_checkpoint(
                    self._layer, i, resid, x_n, cos, sin, cu, pos, segments, use_reentrant=False
                )
            else:
                resid, x_n = self._layer(i, resid, x_n, cos, sin, cu, pos, segments)
        return x_n  # final norm output (weighted form)
''')

rep('''    def forward_packed(
        self,
        tokens: torch.Tensor,
        cu_seqlens: torch.Tensor,
        positions: torch.Tensor,
        select_rows: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        h = self.hidden_states(tokens, cu_seqlens, positions)''', '''    def forward_packed(
        self,
        tokens: torch.Tensor,
        cu_seqlens: torch.Tensor,
        positions: torch.Tensor,
        select_rows: torch.Tensor,
        targets: torch.Tensor,
        segments: Sequence[tuple[int, int]] | None = None,
    ) -> torch.Tensor:
        h = self.hidden_states(tokens, cu_seqlens, positions, segments)''')

rep('''            "tp": self.tp_world,
            "tp_collective": self.comm.backend,
            "ops": self.ops.provenance(),
            "torch_flags": self.flags,
            "lora": {
                "rank": self.lora_rank,
                "form": "unmerged: y = bf16(acc32(W x) + sum_k B[:,k] * bf16(A x)[k])",
                "enabled": self.lora_enabled,
                "seed": self.seed,
            },''', '''            "tp": self.tp_world,
            "tp_collective": self.comm.backend,
            "ops": self.ops.provenance(),
            "allow_fallback": self.allow_fallback,
            "torch_flags": self.flags,
            "attention_heads_per_rank": dict(self.head_geometry),
            "rope_table_sha256": self.rope_table_sha256,
            "lora": {
                "rank": self.lora_rank,
                "form": "unmerged: y = bf16(acc32(W x) + sum_k B[:,k] * bf16(A x)[k])",
                "grad_rule": LORA_GRAD_RULE,
                "enabled": self.lora_enabled,
                "seed": self.seed,
            },
            "vision_drop": self.vision_drop,''')

start = s.index("class GatherHeads(torch.autograd.Function):")
end = s.index("class StrictLMHeadLogprob(torch.autograd.Function):")
s = s[:start] + s[end:]

rep('''        dlogp32 = dlogp.to(torch.float32).contiguous()
        parts = []
        for j, tile in enumerate(tile_ids):''', '''        dlogp32 = dlogp.to(torch.float32).contiguous()
        tree = sk.TreeAccumulator()  # canonical tile tree, built as the tiles finish
        for j, tile in enumerate(tile_ids):''')
rep('''            parts.append(sk.gemm_fp32(dz, w_t.t()))  # 8-chunk tree inside the tile (sibling rule)
            del dz
        dh = comm.all_gather_tree(sk.gemm_backend().combine(parts))''', '''            tree.push(sk.gemm_fp32(dz, w_t.t()))  # 8-chunk tree inside the tile (sibling rule)
            del dz
        dh = comm.all_gather_tree(tree.result())''')

rep('''    drop_vision: bool = True,
    lora_b_init_std: float = 0.0,
) -> MuseGlimmerStrictBinding:
    """Bind the strict operator set to a Bridge ``MuseGlimmerModel`` (fails closed).

    ``lora_b_init_std > 0`` draws LoRA ``B`` from ``N(0, std)`` (TP-invariant CPU generator)
    instead of zeros; parity tests only (makes every LoRA gradient non-trivial).
    """''', '''    drop_vision: bool = True,
    lora_b_init_std: float = 0.0,
    allow_fallback: bool = False,
) -> MuseGlimmerStrictBinding:
    """Bind the strict operator set to a Bridge ``MuseGlimmerModel`` (fails closed).

    ``lora_b_init_std > 0`` draws LoRA ``B`` from ``N(0, std)`` (TP-invariant CPU generator)
    instead of zeros; parity tests only (makes every LoRA gradient non-trivial).
    ``allow_fallback=True`` accepts labelled non-strict ops (unit tests only); the default
    raises ``StrictOpsUnavailable`` when any strict kernel is missing or broken.
    """''')
rep('''        drop_vision=drop_vision,
        lora_b_init_std=lora_b_init_std,
    )
    inner._muse_strict_binding = binding''', '''        drop_vision=drop_vision,
        lora_b_init_std=lora_b_init_std,
        allow_fallback=allow_fallback,
    )
    inner._muse_strict_binding = binding''')

rep('''    tokens, cu, positions = pack_sequences(seqs, binding.device)
    select = torch.cat([torch.arange(i * t, i * t + t - 1) for i in range(b)]).to(binding.device)
    logp = binding.forward_packed(tokens, cu, positions, select, targets.reshape(-1))''', '''    tokens, cu, positions = pack_sequences(seqs, binding.device)
    select = torch.cat([torch.arange(i * t, i * t + t - 1) for i in range(b)]).to(binding.device)
    segments = tuple((i * t, (i + 1) * t) for i in range(b))
    logp = binding.forward_packed(tokens, cu, positions, select, targets.reshape(-1), segments)''')

rep('''__all__ = [
    "GatherHeads",
    "LayerWeights",
    "ModelDims",
    "MuseGlimmerStrictBinding",
    "NATIVE_BRIDGE_MODEL_CLASS",
    "SliceLocalHeads",
    "StrictBindingError",
    "StrictLMHeadLogprob",
    "binding_lora_parameters",''', '''__all__ = [
    "LayerWeights",
    "ModelDims",
    "MuseGlimmerStrictBinding",
    "NATIVE_BRIDGE_MODEL_CLASS",
    "RUN_TESTED_TP",
    "SUPPORTED_TP",
    "StrictBindingError",
    "StrictLMHeadLogprob",
    "binding_lora_parameters",
    "drop_vision_modules",''')
p.write_text(s)
print("ok")
