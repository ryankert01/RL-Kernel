# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Single-GPU simulation of the vLLM strict binding at TP 2/4/8 (one decoder layer + lm_head).

The vLLM engine needs one GPU per TP rank, so TP4/TP8 cannot run on a 2-GPU lease.  This module
runs the binding's OWN per-rank arithmetic for ``tp`` simulated ranks as ``tp`` threads on one
device, with an in-process rank-ordered all-gather (a barrier + ``torch.stack``; pure data
movement, like the NCCL all-gather the engine uses).  Every rank:

* slices its shard exactly as vLLM 0.31 lays it out (``QKVParallelLinear``: q heads
  ``[32r/t, 32(r+1)/t)``, the KV head ``r // (t/2)`` replicated when ``t > 2``;
  ``output_gate_proj`` / ``gate_up_proj`` row shards; ``o_proj`` / ``down_proj`` column shards;
  the padded lm_head split into canonical vocab tiles), and the LoRA with ``shard_lora``;
* calls the binding functions unchanged: ``column_parallel_project`` (+LoRA), ``gather_heads``
  (KV-replicated attention), ``row_parallel_project`` (sub-tree partial + all-gather + canonical
  tree, +LoRA), ``strict_selected_logp`` (tile stats + all-gather + tree);
* uses the strict sibling ops for norms, qk-norm/RoPE, attention (varlen), SwiGLU.

The result of every rank is compared with ``torch.equal`` against the same code at TP1.  What it
does NOT cover (needs the real engine at that TP): vLLM's paged KV cache with replicated KV heads,
the ``Attention`` factory patch, the TP embedding gather, NCCL itself.
"""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace
from typing import Any

import torch

from rl_engine.models.muse_glimmer.runtime import vllm_strict as vs
from rl_engine.models.muse_glimmer.runtime.strict_ops import probe_ops, resolve_ops

H, D, NQ, NKV, F = vs.HIDDEN, vs.HEAD_DIM, vs.NUM_HEADS, vs.NUM_KV_HEADS, vs.FFN


class _SimGroup:
    """Rank-ordered all-gather between threads (barrier, then stack)."""

    def __init__(self, world: int) -> None:
        self.world = world
        self.slots: list[torch.Tensor | None] = [None] * world
        self.barrier = threading.Barrier(world, timeout=600)
        self.calls = 0

    def all_gather(self, rank: int, x: torch.Tensor) -> torch.Tensor:
        self.slots[rank] = x
        self.barrier.wait()
        out = torch.cat([s for s in self.slots], dim=0)  # (world * rows, cols), rank order
        self.barrier.wait()
        if rank == 0:
            self.calls += 1
        return out


class _SimCollective:
    def __init__(self, group: _SimGroup, rank: int) -> None:
        self.group, self.rank = group, rank
        self.backend_id = "in-process thread all_gather (simulation)"

    def all_gather(self, x: torch.Tensor) -> torch.Tensor:
        return self.group.all_gather(self.rank, x.contiguous())


def _rand(gen: torch.Generator, *shape: int, scale: float, device: torch.device) -> torch.Tensor:
    return (torch.randn(*shape, generator=gen) * scale).to(device=device, dtype=torch.bfloat16)


def build_problem(*, tokens: int, seed: int, lora_rank: int, device: torch.device) -> dict:
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    w = {
        "q": _rand(gen, NQ * D, H, scale=H**-0.5, device=device),
        "k": _rand(gen, NKV * D, H, scale=H**-0.5, device=device),
        "v": _rand(gen, NKV * D, H, scale=H**-0.5, device=device),
        "g": _rand(gen, NQ * D, H, scale=H**-0.5, device=device),
        "o": _rand(gen, H, NQ * D, scale=(NQ * D) ** -0.5, device=device),
        "mg": _rand(gen, F, H, scale=H**-0.5, device=device),
        "mu": _rand(gen, F, H, scale=H**-0.5, device=device),
        "md": _rand(gen, H, F, scale=F**-0.5, device=device),
        "n_in": _rand(gen, H, scale=0.1, device=device),
        "n_pa": _rand(gen, H, scale=0.1, device=device),
        "n_pf": _rand(gen, H, scale=0.1, device=device),
        "n_po": _rand(gen, H, scale=0.1, device=device),
        "n_fin": (1.0 + torch.randn(H, generator=gen) * 0.1).to(device, torch.bfloat16),
    }
    lm = torch.zeros(vs.PADDED_VOCAB, H, dtype=torch.bfloat16, device=device)
    lm[: vs.REAL_VOCAB] = _rand(gen, vs.REAL_VOCAB, H, scale=1.0, device=device)
    w["lm"] = lm
    lora = {}
    for name, (k_in, n_out) in vs.LORA_SHAPES.items():
        lora[name] = (
            _rand(gen, lora_rank, k_in, scale=k_in**-0.5, device=device),
            _rand(gen, n_out, lora_rank, scale=0.02, device=device),
        )
    x = _rand(gen, tokens, H, scale=3.0, device=device)
    targets = torch.randint(0, vs.REAL_VOCAB, (tokens,), generator=gen).to(device)
    # Make sure the targets hit the first, a middle and the last real tile.
    targets[0], targets[-1] = 3, vs.REAL_VOCAB - 1
    return {"w": w, "lora": lora, "x": x, "targets": targets}


def _rank_weights(problem: dict, tp: int, rank: int) -> dict[str, torch.Tensor]:
    w = problem["w"]
    qh = NQ // tp
    q_rows = slice(rank * qh * D, (rank + 1) * qh * D)
    kv_per_rank = max(1, NKV // tp)
    replicas = max(1, tp // NKV)
    kv_first = (rank // replicas) * kv_per_rank
    kv_rows = slice(kv_first * D, (kv_first + kv_per_rank) * D)
    fused_qkv = torch.cat([w["q"][q_rows], w["k"][kv_rows], w["v"][kv_rows]]).contiguous()
    f_rows = slice(rank * F // tp, (rank + 1) * F // tp)
    gate_up = torch.cat([w["mg"][f_rows], w["mu"][f_rows]]).contiguous()
    o_cols = slice(rank * NQ * D // tp, (rank + 1) * NQ * D // tp)
    tiles = vs.CANONICAL_TP // tp
    lm_rows = slice(rank * tiles * vs.TILE_ROWS, (rank + 1) * tiles * vs.TILE_ROWS)
    return {
        "qkv": fused_qkv,
        "q_size": qh * D,
        "kv_size": kv_per_rank * D,
        "g": w["g"][q_rows].contiguous(),
        "o": w["o"][:, o_cols].contiguous(),
        "gate_up": gate_up,
        "down": w["md"][:, f_rows].contiguous(),
        "lm": w["lm"][lm_rows].contiguous(),
    }


def rank_forward(
    state: vs.StrictBindingState, problem: dict, *, use_lora: bool
) -> dict[str, torch.Tensor]:
    """The binding's decoder-layer + lm_head arithmetic for one (simulated) rank."""

    ops, w, tp, rank = state.ops, problem["w"], state.tp_size, state.tp_rank
    rw = _rank_weights(problem, tp, rank)
    lora: dict[str, Any] = {}
    if use_lora:
        for name, (a, b) in problem["lora"].items():
            lora[name] = vs.shard_lora(name, a, b, tp_size=tp, tp_rank=rank)
    x = problem["x"]
    t = x.shape[0]
    positions = torch.arange(t, device=x.device)
    x_n = ops.centered_rmsnorm(x, w["n_in"], eps=1e-5)
    w_q, w_k, w_v = vs.canonical_qkv_slices(rw["qkv"], rw["q_size"], rw["kv_size"])
    q = vs.column_parallel_project(state, x_n, w_q, lora.get("q_proj"))
    k = vs.column_parallel_project(state, x_n, w_k, lora.get("k_proj"))
    v = vs.column_parallel_project(state, x_n, w_v, lora.get("v_proj"))
    g = vs.column_parallel_project(state, x_n, rw["g"], lora.get("gate_proj"))
    if tp > 1:
        replicas = max(1, tp // NKV)
        q = vs.gather_heads(state, q)
        k = vs.gather_heads(state, k, replicas=replicas)
        v = vs.gather_heads(state, v, replicas=replicas)
        g = vs.gather_heads(state, g)
    cos, sin = vs._rope_for(state, positions)
    q_hat, k_hat = ops.qknorm_scaled_rope(
        q.view(t, NQ, D), k.view(t, NKV, D), cos, sin, eps=1e-5, qk_scale=3.87
    )
    cu = torch.tensor([0, t], dtype=torch.int32, device=x.device)
    y, _o, _lse = ops.gated_attention_varlen(
        q_hat.contiguous(),
        k_hat.contiguous(),
        v.view(t, NKV, D).contiguous(),
        g.contiguous(),
        cu,
        cu,
        positions.to(torch.int32),
        window=2048,
    )
    if tp > 1:
        q_size = rw["q_size"]
        y = y[:, rank * q_size : (rank + 1) * q_size].contiguous()
    attn = vs.row_parallel_project(state, y, rw["o"], lora.get("o_proj"))
    r, x_f = ops.fused_post_norm_residual_pre_norm(
        attn, x, w["n_pa"], 1e-8, w["n_pf"], 1e-5, pre_form="centered"
    )
    w_gate, w_up = vs.canonical_gate_up_slices(rw["gate_up"])
    gate = vs.column_parallel_project(state, x_f, w_gate, lora.get("mlp.gate_proj"))
    up = vs.column_parallel_project(state, x_f, w_up, lora.get("mlp.up_proj"))
    h = ops.swiglu(gate, up)
    f = vs.row_parallel_project(state, h, rw["down"], lora.get("mlp.down_proj"))
    post = ops.centered_rmsnorm(f, w["n_po"], eps=1e-8)
    r = (r.float() + post.float()).to(torch.bfloat16)
    hidden = ops.weighted_rmsnorm(r, w["n_fin"], eps=1e-5)
    state.lm_head = SimpleNamespace(weight=rw["lm"])
    logp = vs.strict_selected_logp(state, hidden, problem["targets"])
    return {"q": q, "k": k, "attn_out": attn, "mlp_out": f, "hidden": hidden, "logp": logp}


def _state(ops: Any, tp: int, rank: int, collective: Any) -> vs.StrictBindingState:
    state = vs.StrictBindingState(ops=ops)
    state.tp_size, state.tp_rank = tp, rank
    state.collective = collective
    state.collective_label = "simulation"
    state.cfg.update(
        {
            "output_multiplier": float(vs.OFFICIAL_FINGERPRINT["output_multiplier"]),
            "final_logit_softcapping": float(vs.OFFICIAL_FINGERPRINT["final_logit_softcapping"]),
            "rope_theta": float(vs.OFFICIAL_FINGERPRINT["rope_parameters"]["rope_theta"]),
            "rope_table_positions": 4096,
        }
    )
    return state


def simulate(
    tps: tuple[int, ...] = (2, 4, 8),
    *,
    tokens: int = 37,
    seed: int = 0,
    lora_rank: int = 16,
    device: torch.device | str = "cuda",
) -> dict[str, Any]:
    """Run TP1 and each simulated TP; return per-tensor bitwise verdicts (rank by rank)."""

    device = torch.device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    ops = probe_ops(resolve_ops(), device)
    problem = build_problem(tokens=tokens, seed=seed, lora_rank=lora_rank, device=device)
    report: dict[str, Any] = {
        "tokens": tokens,
        "seed": seed,
        "lora_rank": lora_rank,
        "op_impls": dict(ops.labels),
        "results": [],
    }
    for use_lora in (False, True):
        ref = rank_forward(_state(ops, 1, 0, None), problem, use_lora=use_lora)
        for tp in tps:
            group = _SimGroup(tp)
            outs: list[dict[str, torch.Tensor] | None] = [None] * tp
            errors: list[str] = []

            def run(rank: int) -> None:
                try:
                    st = _state(ops, tp, rank, _SimCollective(group, rank))
                    outs[rank] = rank_forward(st, problem, use_lora=use_lora)
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"rank {rank}: {type(exc).__name__}: {exc}")
                    group.barrier.abort()

            t0 = time.perf_counter()
            threads = [threading.Thread(target=run, args=(r,)) for r in range(tp)]
            for th in threads:
                th.start()
            for th in threads:
                th.join()
            torch.cuda.synchronize(device)
            entry: dict[str, Any] = {
                "tp": tp,
                "lora": use_lora,
                "seconds": time.perf_counter() - t0,
                "all_gathers": group.calls,
                "errors": errors,
            }
            if not errors:
                checks = {}
                for name in ("q", "k", "attn_out", "mlp_out", "hidden", "logp"):
                    checks[name] = [bool(torch.equal(o[name], ref[name])) for o in outs]
                entry["equal_to_tp1"] = checks
                entry["all_equal"] = all(all(v) for v in checks.values())
                entry["logp_max_abs_vs_tp1"] = max(
                    float((o["logp"] - ref["logp"]).abs().max().item()) for o in outs
                )
            else:
                entry["all_equal"] = False
            report["results"].append(entry)
    report["all_equal"] = all(e["all_equal"] for e in report["results"])
    return report


if __name__ == "__main__":  # pragma: no cover - manual run
    import json
    import sys

    result = simulate()
    print(json.dumps(result, indent=1))
    sys.exit(0 if result["all_equal"] else 1)
