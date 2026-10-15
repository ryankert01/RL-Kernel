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

from dataclasses import dataclass
from typing import Any, Sequence

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
        disable_tf32(self.device.type)

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

    # -- the shared composition ---------------------------------------------------------------
    def _run(
        self,
        token_ids: torch.Tensor,
        positions: torch.Tensor,
        session: Session | None,
        targets: torch.Tensor | None,
        *,
        return_logits: bool = False,
    ) -> dict[str, torch.Tensor]:
        cfg, ops, w = self.cfg, self.ops, self.weights
        n = int(token_ids.shape[0])
        if session is None and not torch.equal(
            positions, torch.arange(n, device=positions.device)
        ):
            raise ValueError("training forward requires positions 0..T-1")
        start = int(positions[0].item())
        seq_len_after = start + n
        cu = torch.tensor([0, n], dtype=torch.int32, device=self.device)
        window = cfg.sliding_window

        r = ops.normalized_embedding(w.embed_tokens, token_ids, eps=cfg.rms_norm_eps)
        self._rec("embedding", r)
        x_n = ops.centered_rmsnorm(r, w.layers[0].input_layernorm, eps=cfg.rms_norm_eps)
        self._rec("layers.0.input_layernorm", x_n)

        for layer in w.layers:
            i = layer.index
            tag = f"layers.{i}."
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
            if session is None:
                y, o, _lse = ops.attention_varlen(
                    q_hat, k_hat, v, g, cu, cu, positions, window=layer_window
                )
            else:
                cache = session.cache
                cache.append(i, session.seq, k_hat, v, start)
                k_cache, v_cache, block_table = cache.layer_views(i)
                seq_lens = torch.tensor([seq_len_after], dtype=torch.long, device=self.device)
                y, o, _lse = ops.attention_paged(
                    q_hat, g, k_cache, v_cache, block_table, seq_lens, cu, window=layer_window
                )
            self._rec(tag + "attn", o)
            self._rec(tag + "output_gate", y)
            attn_out = _row_parallel(ops, y, layer.o_proj)
            self._rec(tag + "o_proj", attn_out)
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
            f = ops.swiglu_ffn(
                x_f,
                ops.weight_arg(layer.mlp_gate_proj),
                ops.weight_arg(layer.up_proj),
                ops.weight_arg(layer.down_proj),
                gemm=ops.gemm,
            )
            self._rec(tag + "down_proj", f)
            last = i == cfg.num_hidden_layers - 1
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

        if session is not None:
            session.cache.advance(session.seq, n)
            for layer in w.layers:
                if layer.is_local:
                    session.cache.evict_window(layer.index, session.seq, window)

        hidden = x_n
        out: dict[str, torch.Tensor] = {"hidden": hidden}
        head_arg = ops.lm_head_arg(w.lm_head)
        if targets is not None:
            logp = ops.lm_head_softcap_logprob(
                hidden,
                head_arg,
                targets,
                multiplier=cfg.output_multiplier,
                softcap=cfg.final_logit_softcapping,
                vocab_size=cfg.vocab_size,
                gemm=ops.gemm,
            )
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


__all__ = ["MODES", "MuseGlimmerStrictModel", "Session", "nanmean_loss"]
