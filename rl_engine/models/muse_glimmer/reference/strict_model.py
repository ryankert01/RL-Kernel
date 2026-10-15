# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Strict PyTorch reference of the Muse-Glimmer-30B text model (RFC #501 ``full_model_chain``).

One operator composition (contract "Model assembly") serves every execution mode:

* ``forward_train``: all T tokens at once, attention over this pass's own K/V (varlen form);
* ``prefill`` / chunked prefill / ``decode_step``: tokens are written to a paged KV cache first
  and attention reads the cache (paged form), exactly as a rollout engine does.

The only thing that changes between modes is where K/V come from; every row-local op, every GEMM
and the attention arithmetic are the same callables, so selected-token log-probabilities must be
byte-equal across modes. Per-node outputs can be captured (CPU copies, operator-trace node names)
for first-drift localization.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import torch

from rl_engine.models.muse_glimmer.reference.backend import OpSet, resolve_ops
from rl_engine.models.muse_glimmer.reference.paged_kv import PagedKVCache
from rl_engine.models.muse_glimmer.reference.weights import MuseGlimmerWeights, StrictConfig
from rl_engine.validation.operators.accelerator import disable_tf32

MODES = ("train", "prefill", "chunked", "decode", "per_token")


@dataclass
class Session:
    """Rollout state of one sequence: a paged KV cache and the number of committed tokens."""

    cache: PagedKVCache
    seq: int = 0

    @property
    def length(self) -> int:
        return self.cache.seq_len(self.seq)


@dataclass
class KVPlan:
    """Where the K/V of one ``_run`` call go: ``lens[j]`` new tokens of cache sequence
    ``seqs[j]`` at absolute positions ``starts[j]..``, packed in this order in the token axis.

    ``commit=False`` writes the new K/V into the cache slots but neither advances the sequence
    lengths nor evicts (a scratch step; the next call rewrites the same slots).
    """

    cache: PagedKVCache
    seqs: list[int]
    starts: list[int]
    lens: list[int]
    commit: bool = True
    _seq_index: torch.Tensor | None = field(default=None, repr=False)

    @property
    def total(self) -> int:
        return int(sum(self.lens))

    def positions(self, device: torch.device) -> torch.Tensor:
        return torch.cat(
            [torch.arange(s, s + n, device=device) for s, n in zip(self.starts, self.lens)]
        )


class MuseGlimmerStrictModel:
    """Full text model on the strict operator set (golden fallbacks or sibling Triton ops)."""

    def __init__(
        self,
        weights: MuseGlimmerWeights,
        backend: str = "golden",
        *,
        ops: OpSet | None = None,
        allow_partial: bool = False,
        gemm_impl: str = "triton_fp32",
    ):
        self.weights = weights
        self.cfg: StrictConfig = weights.cfg
        self.ops = ops if ops is not None else resolve_ops(
            backend, allow_partial=allow_partial, gemm_impl=gemm_impl
        )
        self.device = weights.device
        self._capture: dict[str, list[torch.Tensor]] | None = None
        self._timer: Any = None
        disable_tf32(self.device.type)

    # -- profiling hook -----------------------------------------------------------------------
    def set_timer(self, timer: Any) -> None:
        """Install (or remove with ``None``) a region timer, e.g.
        :class:`rl_engine.models.muse_glimmer.reference.profiling.RegionTimer`. It only records
        CUDA events around the op calls; the arithmetic and every bit are unchanged."""
        self._timer = timer

    def _region(self, cls: str, kind: str, layer: int | None = None):
        if self._timer is None:
            return nullcontext()
        return self._timer.region(cls, kind, layer)

    def _gemm_for(self, cls: str, kind: str, layer: int | None = None) -> Callable[..., Any]:
        gemm = self.ops.gemm
        if self._timer is None:
            return gemm
        return self._timer.wrap(gemm, cls, kind, layer)

    # -- capture ------------------------------------------------------------------------------
    def begin_capture(self) -> None:
        self._capture = {}

    def end_capture(self) -> dict[str, torch.Tensor]:
        """Return captured node outputs concatenated along the token axis (CPU, exact dtype)."""
        cap = self._capture or {}
        self._capture = None
        return {name: torch.cat(parts, dim=0) for name, parts in cap.items()}

    def _rec(self, name: str, value: torch.Tensor) -> None:
        if self._capture is not None:
            self._capture.setdefault(name, []).append(value.detach().to("cpu", copy=True))

    # -- sessions -----------------------------------------------------------------------------
    def new_session(self, max_seq_len: int) -> Session:
        cache = PagedKVCache(
            num_layers=self.cfg.num_hidden_layers,
            n_kv_heads=self.cfg.num_key_value_heads,
            head_dim=self.cfg.head_dim,
            max_seq_len=int(max_seq_len),
            device=self.device,
            dtype=torch.bfloat16,
        )
        return Session(cache=cache)

    def new_batch_cache(self, max_seq_len: int, max_seqs: int) -> PagedKVCache:
        """A paged KV cache holding ``max_seqs`` sequences (for :meth:`decode_batch`)."""
        return PagedKVCache(
            num_layers=self.cfg.num_hidden_layers,
            n_kv_heads=self.cfg.num_key_value_heads,
            head_dim=self.cfg.head_dim,
            max_seq_len=int(max_seq_len),
            device=self.device,
            dtype=torch.bfloat16,
            max_seqs=int(max_seqs),
        )

    # -- the shared composition ---------------------------------------------------------------
    def _run(
        self,
        token_ids: torch.Tensor,
        positions: torch.Tensor,
        kv: KVPlan | Session | None,
        targets: torch.Tensor | None,
        *,
        return_logits: bool = False,
    ) -> dict[str, torch.Tensor]:
        """THE composition (contract "Model assembly"). ``kv is None``: training forward over
        positions ``0..T-1`` (varlen attention over this pass's own K/V). Otherwise the tokens
        are appended to the paged cache per :class:`KVPlan` (a :class:`Session` is the
        one-sequence plan) and attention reads the cache; several sequences may be packed
        (batched decode / prefill), each row's arithmetic is unchanged by its neighbours."""
        with self._region("forward", "model"):
            return self._run_inner(token_ids, positions, kv, targets, return_logits)

    def _run_inner(
        self,
        token_ids: torch.Tensor,
        positions: torch.Tensor,
        kv: KVPlan | Session | None,
        targets: torch.Tensor | None,
        return_logits: bool,
    ) -> dict[str, torch.Tensor]:
        cfg, ops, w = self.cfg, self.ops, self.weights
        n = int(token_ids.shape[0])
        if isinstance(kv, Session):
            start = int(positions[0].item())
            kv = KVPlan(kv.cache, [kv.seq], [start], [n])
        if kv is None:
            if not torch.equal(positions, torch.arange(n, device=positions.device)):
                raise ValueError("training forward requires positions 0..T-1")
            cu = torch.tensor([0, n], dtype=torch.int32, device=self.device)
        else:
            if kv.total != n:
                raise ValueError(f"KV plan covers {kv.total} tokens, got {n}")
            cu_host = [0]
            for c in kv.lens:
                cu_host.append(cu_host[-1] + int(c))
            cu = torch.tensor(cu_host, dtype=torch.int32, device=self.device)
            seq_lens = torch.tensor(
                [s + c for s, c in zip(kv.starts, kv.lens)], dtype=torch.int32, device=self.device
            )
            if kv.cache.max_seqs == 1 and kv.seqs == [0]:
                seq_index = None  # the whole (one-row) block table, as a view
            else:
                seq_index = torch.tensor(kv.seqs, dtype=torch.long, device=self.device)
        window = cfg.sliding_window

        with self._region("embedding", "model"):
            r = ops.normalized_embedding(w.embed_tokens, token_ids, eps=cfg.rms_norm_eps)
        self._rec("embedding", r)
        kind0 = "local" if w.layers[0].is_local else "global"
        with self._region("norm", kind0, 0):
            x_n = ops.centered_rmsnorm(r, w.layers[0].input_layernorm, eps=cfg.rms_norm_eps)
        self._rec("layers.0.input_layernorm", x_n)

        for layer in w.layers:
            i = layer.index
            tag = f"layers.{i}."
            kind = "local" if layer.is_local else "global"
            with self._region("qkvg_gemm", kind, i):
                q = ops.gemm(x_n, ops.weight_arg(layer.q_proj))
                k = ops.gemm(x_n, ops.weight_arg(layer.k_proj))
                v = ops.gemm(x_n, ops.weight_arg(layer.v_proj))
                g = ops.gemm(x_n, ops.weight_arg(layer.attn_gate_proj))
            self._rec(tag + "q_proj", q)
            self._rec(tag + "k_proj", k)
            self._rec(tag + "v_proj", v)
            self._rec(tag + "attn_gate_proj", g)
            q = q.view(n, cfg.num_attention_heads, cfg.head_dim)
            k = k.view(n, cfg.num_key_value_heads, cfg.head_dim)
            v = v.view(n, cfg.num_key_value_heads, cfg.head_dim)
            with self._region("qknorm_rope", kind, i):
                if layer.is_local:
                    cos, sin = ops.rope_table(
                        positions, theta=cfg.rope_theta, dim=cfg.head_dim, device=self.device
                    )
                else:
                    cos = sin = None
                q_hat, k_hat = ops.qknorm_scaled_rope(
                    q, k, cos, sin, eps=cfg.rms_norm_eps, qk_scale=cfg.qk_scale_factor
                )
            if layer.is_local:
                self._rec(tag + "rope_q", q_hat)
                self._rec(tag + "rope_k", k_hat)
            else:
                self._rec(tag + "q_norm", q_hat)
                self._rec(tag + "k_norm", k_hat)
            layer_window = window if layer.is_local else None
            if kv is None:
                with self._region("attention", kind, i):
                    y, o, _lse = ops.attention_varlen(
                        q_hat, k_hat, v, g, cu, cu, positions.to(torch.int32), window=layer_window
                    )
            else:
                cache = kv.cache
                with self._region("kv_write", kind, i):
                    off = 0
                    for seq, s0, c in zip(kv.seqs, kv.starts, kv.lens):
                        cache.append(i, seq, k_hat[off : off + c], v[off : off + c], s0)
                        off += c
                    k_cache, v_cache, block_table = cache.layer_views(i)
                    if seq_index is not None:
                        block_table = block_table.index_select(0, seq_index)
                with self._region("attention", kind, i):
                    y, o, _lse = ops.attention_paged(
                        q_hat, g, k_cache, v_cache, block_table, seq_lens, cu, window=layer_window
                    )
            self._rec(tag + "attn", o)
            self._rec(tag + "output_gate", y)
            with self._region("o_gemm", kind, i):
                attn_out = _row_parallel(ops, y, layer.o_proj)
            self._rec(tag + "o_proj", attn_out)
            with self._region("norm", kind, i):
                r, x_f = ops.fused_post_norm_residual_pre_norm(
                    attn_out,
                    r,
                    layer.post_attention_layernorm,
                    cfg.post_norm_eps,
                    layer.pre_feedforward_layernorm,
                    cfg.rms_norm_eps,
                    pre_form="centered",
                )
            self._rec(tag + "residual_attn", r)
            self._rec(tag + "pre_feedforward_layernorm", x_f)
            with self._region("ffn_act", kind, i):
                f = ops.swiglu_ffn(
                    x_f,
                    ops.weight_arg(layer.mlp_gate_proj),
                    ops.weight_arg(layer.up_proj),
                    ops.weight_arg(layer.down_proj),
                    gemm=self._gemm_for("ffn_gemm", kind, i),
                )
            self._rec(tag + "down_proj", f)
            last = i == cfg.num_hidden_layers - 1
            with self._region("norm", kind, i):
                r, x_n = ops.fused_post_norm_residual_pre_norm(
                    f,
                    r,
                    layer.post_feedforward_layernorm,
                    cfg.post_norm_eps,
                    w.final_norm if last else w.layers[i + 1].input_layernorm,
                    cfg.rms_norm_eps,
                    pre_form="weighted" if last else "centered",
                )
            self._rec(tag + "residual_mlp", r)
            self._rec("final_norm" if last else f"layers.{i + 1}.input_layernorm", x_n)

        if kv is not None and kv.commit:
            with self._region("kv_evict", "model"):
                for seq, c in zip(kv.seqs, kv.lens):
                    kv.cache.advance(seq, c)
                    for layer in w.layers:
                        if layer.is_local:
                            kv.cache.evict_window(layer.index, seq, window)

        hidden = x_n
        out: dict[str, torch.Tensor] = {"hidden": hidden}
        head_arg = ops.lm_head_arg(w.lm_head)
        if targets is not None:
            with self._region("softcap_logprob", "model"):
                valid = targets >= 0
                rows = hidden if bool(valid.all()) else hidden[valid]
                logp_valid = ops.lm_head_softcap_logprob(
                    rows,
                    head_arg,
                    targets[valid] if rows is not hidden else targets,
                    multiplier=cfg.output_multiplier,
                    softcap=cfg.final_logit_softcapping,
                    vocab_size=cfg.vocab_size,
                    gemm=self._gemm_for("lm_head_gemm", "model"),
                )
                if rows is hidden:
                    logp = logp_valid
                else:
                    # Row-local op: scoring only the rows that have a target leaves their bits
                    # unchanged (batch invariance); rows without a target report NaN.
                    logp = torch.full((n,), float("nan"), dtype=torch.float32, device=self.device)
                    logp = logp.masked_scatter(valid, logp_valid)
            self._rec("logprob", logp)
            out["logp"] = logp
        if return_logits:
            from rl_engine.models.muse_glimmer.reference.golden_ops import (
                lm_head_softcap_logits,
            )

            out["logits"] = lm_head_softcap_logits(
                hidden,
                head_arg,
                multiplier=cfg.output_multiplier,
                softcap=cfg.final_logit_softcapping,
                vocab_size=cfg.vocab_size,
                gemm=ops.gemm,
            )[:, : cfg.vocab_size]
        return out

    # -- public entry points ------------------------------------------------------------------
    def forward_train(
        self,
        token_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        *,
        return_logits: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Training-mode forward over ``token_ids`` (T,). ``targets`` (T,) long, −1 = none.

        Differentiable (used for the tiny-config backward gate); runs attention in the varlen
        form over this pass's own K/V. Returns ``{"hidden", "logp"[, "logits"]}``.
        """
        token_ids = token_ids.to(device=self.device, dtype=torch.long)
        positions = torch.arange(token_ids.shape[0], device=self.device)
        tgt = None if targets is None else targets.to(device=self.device, dtype=torch.long)
        return self._run(token_ids, positions, None, tgt, return_logits=return_logits)

    @torch.no_grad()
    def prefill(
        self,
        session: Session,
        token_ids: torch.Tensor,
        *,
        targets: torch.Tensor | None = None,
        chunk_sizes: Sequence[int] | None = None,
    ) -> torch.Tensor:
        """Append ``token_ids`` to the session (one shot or in ``chunk_sizes`` chunks) through
        the paged KV cache; returns fp32 logp of ``targets`` (NaN where target is −1)."""
        token_ids = token_ids.to(device=self.device, dtype=torch.long)
        n = int(token_ids.shape[0])
        if targets is None:
            targets = torch.full((n,), -1, dtype=torch.long, device=self.device)
        targets = targets.to(device=self.device, dtype=torch.long)
        sizes = [n] if chunk_sizes is None else [int(c) for c in chunk_sizes]
        if sum(sizes) != n or any(c <= 0 for c in sizes):
            raise ValueError(f"chunk_sizes {sizes} must be positive and sum to {n}")
        outs = []
        base = session.length
        off = 0
        for c in sizes:
            positions = torch.arange(base + off, base + off + c, device=self.device)
            out = self._run(token_ids[off : off + c], positions, session, targets[off : off + c])
            outs.append(out["logp"])
            off += c
        return torch.cat(outs, dim=0)

    @torch.no_grad()
    def decode_step(
        self, session: Session, token: int | torch.Tensor, target: int | None = None
    ) -> torch.Tensor:
        """One rollout step: feed ``token`` at the next position; logp of ``target`` (fp32)."""
        tok = torch.as_tensor([int(token)], dtype=torch.long, device=self.device)
        tgt = torch.as_tensor(
            [-1 if target is None else int(target)], dtype=torch.long, device=self.device
        )
        positions = torch.arange(session.length, session.length + 1, device=self.device)
        return self._run(tok, positions, session, tgt)["logp"]

    @torch.no_grad()
    def decode_batch(
        self,
        cache: PagedKVCache,
        seqs: Sequence[int],
        tokens: Sequence[int] | torch.Tensor,
        targets: Sequence[int] | torch.Tensor | None = None,
        *,
        commit: bool = True,
    ) -> torch.Tensor:
        """One rollout step for ``len(seqs)`` sequences of ``cache`` packed into ONE call
        (batched decode: every GEMM has M = batch, attention gets one query per sequence).

        ``tokens[b]`` is fed at the next position of ``seqs[b]``; returns the fp32 logp of
        ``targets[b]`` (NaN where −1). ``commit=False`` leaves the cache lengths unchanged
        (the K/V slots are rewritten by the next call), which lets the same step be replayed
        at another batch size for the batch-invariance gate (contract G2)."""
        seqs = [int(s) for s in seqs]
        lens_host = [int(x) for x in cache.seq_lens.tolist()]
        plan = KVPlan(cache, seqs, [lens_host[s] for s in seqs], [1] * len(seqs), commit=commit)
        tok = torch.as_tensor(tokens, dtype=torch.long).to(self.device).reshape(-1)
        if targets is None:
            tgt = torch.full((len(seqs),), -1, dtype=torch.long, device=self.device)
        else:
            tgt = torch.as_tensor(targets, dtype=torch.long).to(self.device).reshape(-1)
        if tok.numel() != len(seqs) or tgt.numel() != len(seqs):
            raise ValueError("need one token and one target per sequence")
        return self._run(tok, plan.positions(self.device), plan, tgt)["logp"]

    def logprobs(
        self,
        prompt_ids: torch.Tensor,
        completion_ids: torch.Tensor,
        *,
        mode: str = "train",
        chunk_sizes: Sequence[int] | None = None,
    ) -> torch.Tensor:
        """Selected-token contract: fp32 logp of every completion token given the prefix.

        ``mode``: ``train`` (one differentiable training forward over prompt+completion),
        ``prefill`` (one-shot paged prefill of prompt+completion[:-1]), ``chunked`` (the same
        in ``chunk_sizes`` chunks), ``decode`` (one-shot prompt prefill, then one decode step
        per completion token), ``per_token`` (every token, prompt included, as its own step).
        """
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        prompt = prompt_ids.to(device=self.device, dtype=torch.long)
        completion = completion_ids.to(device=self.device, dtype=torch.long)
        lp, lc = int(prompt.shape[0]), int(completion.shape[0])
        if lp < 1 or lc < 1:
            raise ValueError("need at least one prompt and one completion token")
        full = torch.cat([prompt, completion])
        total = lp + lc
        if mode == "train":
            targets = torch.cat([full[1:], torch.full((1,), -1, device=self.device)])
            with torch.no_grad():
                return self.forward_train(full, targets)["logp"][lp - 1 : total - 1]
        scored, scored_targets = full[: total - 1], full[1:]
        session = self.new_session(total)
        if mode in ("prefill", "chunked", "per_token"):
            if mode == "prefill":
                sizes = None
            elif mode == "per_token":
                sizes = [1] * (total - 1)
            else:
                if chunk_sizes is None:
                    raise ValueError("mode='chunked' needs chunk_sizes")
                sizes = list(chunk_sizes)
            logp = self.prefill(session, scored, targets=scored_targets, chunk_sizes=sizes)
            return logp[lp - 1 :]
        # decode: prompt one shot, then token by token through the completion
        prompt_targets = torch.cat([prompt[1:], completion[:1]])
        out = [self.prefill(session, prompt, targets=prompt_targets)[-1:]]
        for j in range(lc - 1):
            out.append(self.decode_step(session, completion[j], int(completion[j + 1])))
        return torch.cat(out, dim=0)

    # -- introspection ------------------------------------------------------------------------
    def describe(self) -> dict[str, Any]:
        return {
            "weights": self.weights.describe(),
            "ops": self.ops.describe(),
            "modes": list(MODES),
        }


def _row_parallel(ops: OpSet, a: torch.Tensor, lin) -> torch.Tensor:
    """o_proj / down_proj: contract O3 row-parallel form through the active GEMM."""
    from rl_engine.models.muse_glimmer.reference.golden_ops import row_parallel_gemm

    return row_parallel_gemm(ops.gemm, a, ops.weight_arg(lin))


def nanmean_loss(logp: torch.Tensor) -> torch.Tensor:
    """-mean of the valid (non-NaN) selected log-probabilities; the tiny backward objective."""
    valid = ~torch.isnan(logp)
    if int(valid.sum().item()) == 0:
        raise ValueError("no valid targets")
    return -(torch.where(valid, logp, torch.zeros_like(logp)).sum() / valid.sum())


__all__ = ["KVPlan", "MODES", "MuseGlimmerStrictModel", "Session", "nanmean_loss"]
