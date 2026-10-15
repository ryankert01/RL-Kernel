# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""``StrictTrainer``: the Megatron Bridge LoRA trainer the E2E orchestrator drives.

It satisfies ``rl_engine.models.muse_glimmer.e2e.engines.TrainerLike`` exactly::

    trainer = StrictTrainer(model_path, tensor_parallel_size=2, gpu_ids=[0, 1],
                            lora_rank=16, lr=1e-5, strict=True, seed=1234)
    lps = trainer.token_logprobs(token_ids, completion_start)   # list of fp32 (n_i,) tensors
    loss = sum((lp * g).sum() for lp, g in zip(lps, grads))      # ANY scalar built from lps
    info = trainer.step(loss)                                     # one AdamW step on the LoRA
    state, sha = trainer.export_lora()                            # bf16 layers.{i}.{proj}.lora_{A,B}
    trainer.provenance(); trainer.close()

Process model (no torchrun): the CALLING process is TP rank 0 and the only rank that returns
results. For ``tensor_parallel_size=t > 1`` it spawns ``t-1`` worker processes
(``torch.multiprocessing`` spawn context) for ranks 1..t-1; the ranks form an NCCL group over ``tcp://127.0.0.1:<free port>``
and Megatron's model-parallel state is initialised with ``tensor_model_parallel_size=tp``.
``gpu_ids`` are CUDA ordinals as seen by the calling process (i.e. indices into its
``CUDA_VISIBLE_DEVICES``); rank ``r`` uses ``gpu_ids[r]``. Every public call is broadcast to the
workers as a ``(command, args, kwargs)`` tuple (``broadcast_object_list``) and executed on every
rank in the same order, so the collectives inside the strict ops pair up.

Training step semantics (micro-batched recompute). ``token_logprobs`` runs the strict forward
WITHOUT an autograd graph (micro-batches of at most ``micro_batch_tokens`` packed tokens), keeps
the fp32 values, and returns them as views of one leaf tensor that requires grad, so the caller
can build any scalar loss from them. ``step(loss)`` takes ``g = dloss/dlogp`` on rank 0
(``torch.autograd.grad``), broadcasts ``g``, and every rank recomputes each micro-batch WITH the
graph (per-layer activation recompute), checks that the recomputed log-probabilities equal the
returned ones bitwise (the strict path is batch- and repeat-invariant, so they must), and
back-propagates ``g``. In strict mode a non-zero mismatch count FAILS CLOSED: the micro-batch is
not back-propagated, every LoRA gradient of the step is discarded and
:class:`StrictRecomputeMismatch` is raised, so no optimizer step uses it (the native mode only
counts mismatches in ``recompute_logp_mismatches``). LoRA weight gradients follow
``megatron_lora.LORA_GRAD_RULE``: per-sequence det_gemm dW partials folded left to right in
sample order, so they do not depend on ``micro_batch_tokens`` (packing) or the TP size. Then AdamW
(``foreach=False``, elementwise per parameter) updates the fp32 LoRA masters. The gradient norm is
computed from the TP-gathered full gradients in canonical key order, so it is identical at every
TP size.

``strict=False`` is the NATIVE trainer: Bridge's own forward (TE/cuBLAS kernels, bf16 logits)
with the same LoRA parameterisation added by forward hooks in plain bf16 matmuls
(``megatron_native_lora``), one sequence per forward, MCore full per-layer recompute for the
backward. Same protocol, same export schema/hash; TP 1/2 only.
"""

from __future__ import annotations

import datetime
import os
import socket
import time
import traceback
from typing import Any, Sequence

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from rl_engine.models.muse_glimmer.runtime import strict_kernels as sk
from rl_engine.models.muse_glimmer.runtime.megatron_lora import (
    DEFAULT_LORA_RANK,
    LORA_GRAD_RULE,
    LayerLoRA,
    export_lora,
    gather_lora,
    lora_parameters,
    lora_sha256,
    muse_glimmer_projection_shapes,
    replicated_lora_grads_agree,
)
from rl_engine.models.muse_glimmer.runtime.megatron_native_lora import (
    NATIVE_LORA_FORM,
    NativeLoRAHooks,
    chunked_token_logprobs,
    native_logprobs_with_grad,
)
from rl_engine.models.muse_glimmer.runtime.megatron_strict import (
    SUPPORTED_TP,
    StrictBindingError,
    _dims_from_manifest,
    drop_vision_modules,
    install_museglimmer_megatron_strict,
    load_bridge_model,
    native_token_logprobs,
    pack_sequences,
)

DEFAULT_MICRO_BATCH_TOKENS = 4096  # memory knob only: LoRA gradient bits do not depend on it
PACKING_RULE = (
    "micro_batches(lengths, micro_batch_tokens): greedy, order-preserving packs of <= budget "
    "tokens (a longer sequence is its own pack); the logprob and LoRA-gradient bits do not "
    "depend on the packing (batch-invariant forward, per-sequence gradient fold)"
)


class StrictRecomputeMismatch(RuntimeError):
    """Strict mode: the recomputed forward differs from the returned log-probabilities."""


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _init_distributed(rank: int, world: int, gpu: int, port: int) -> None:
    torch.cuda.set_device(gpu)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world)
    os.environ["LOCAL_RANK"] = str(gpu)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(
        "nccl",
        rank=rank,
        world_size=world,
        init_method=f"tcp://127.0.0.1:{port}",
        timeout=datetime.timedelta(minutes=30),
        device_id=torch.device("cuda", gpu),
    )
    from megatron.core import parallel_state
    from megatron.core.tensor_parallel import model_parallel_cuda_manual_seed

    parallel_state.initialize_model_parallel(tensor_model_parallel_size=world)
    model_parallel_cuda_manual_seed(0)


def micro_batches(lengths: Sequence[int], budget: int) -> list[list[int]]:
    """Greedy, order-preserving grouping of sequence indices into packs of ``<= budget`` tokens.

    A sequence longer than the budget forms its own pack. Depends only on the lengths and the
    budget (never on the TP size), so every rank and every TP size uses the same packs.
    """
    groups: list[list[int]] = []
    cur: list[int] = []
    used = 0
    for i, n in enumerate(lengths):
        if cur and used + n > budget:
            groups.append(cur)
            cur, used = [], 0
        cur.append(i)
        used += n
    if cur:
        groups.append(cur)
    return groups


def _validate_request(token_ids: list[list[int]], completion_start: list[int]) -> list[int]:
    if len(token_ids) != len(completion_start):
        raise ValueError("token_ids and completion_start must have the same length")
    spans = []
    for i, (seq, start) in enumerate(zip(token_ids, completion_start, strict=True)):
        n = len(seq)
        if not 1 <= int(start) < n:
            raise ValueError(f"sample {i}: completion_start {start} out of range for length {n}")
        spans.append(n - int(start))
    return spans


class _RankState:
    """Everything one TP rank holds; rank 0 and the workers run identical command code."""

    def __init__(
        self,
        rank: int,
        world: int,
        gpu: int,
        port: int,
        *,
        model_path: str,
        lora_rank: int,
        lr: float,
        strict: bool,
        seed: int,
        checkpoint_layers: bool,
        micro_batch_tokens: int,
        lora_b_init_std: float,
        weight_decay: float,
    ) -> None:
        self.rank = rank
        self.world = world
        self.strict = strict
        self.step_count = 0
        self.lr = lr
        self.micro_batch_tokens = int(micro_batch_tokens)
        self.device = torch.device("cuda", gpu)
        torch.manual_seed(seed)
        self.flags = sk.set_deterministic_torch()
        _init_distributed(rank, world, gpu, port)
        self.model, self.bridge, self.load_seconds = load_bridge_model(
            model_path, tensor_parallel_size=world
        )
        self.binding = None
        self.optimizer = None
        self.native_lora: list[LayerLoRA] | None = None
        self.native_hooks: NativeLoRAHooks | None = None
        self._pending: dict[str, Any] | None = None
        self.last_step_info: dict[str, Any] = {}
        if strict:
            self.binding = install_museglimmer_megatron_strict(
                self.model, hf_path=model_path, lora_rank=lora_rank, seed=seed,
                checkpoint_layers=checkpoint_layers, lora_b_init_std=lora_b_init_std,
            )
            self.comm = self.binding.comm
        else:
            self.vision_drop = drop_vision_modules(self.model)
            for p in self.model.parameters():
                p.requires_grad_(False)
            self.comm = sk.TPComm(_tp_group())
            if lora_rank > 0:
                d = _dims_from_manifest()
                shapes = muse_glimmer_projection_shapes(
                    hidden=d.hidden, q_width=d.q_width, kv_width=d.kv_width, ffn=d.ffn
                )
                self.native_lora = [
                    LayerLoRA(i, rank=lora_rank, shapes=shapes, tp_rank=self.comm.rank,
                              tp_world=world, seed=seed, device=self.device,
                              b_init_std=lora_b_init_std)
                    for i in range(d.layers)
                ]
                self.native_hooks = NativeLoRAHooks(
                    self.model, self.native_lora, tp_world=world, heads=d.heads,
                    kv_heads=d.kv_heads, head_dim=d.head_dim,
                )
        params = lora_parameters(self._loras())
        if params:
            self.optimizer = torch.optim.AdamW(
                params, lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=weight_decay,
                foreach=False, fused=False,
            )

    def _loras(self) -> list[LayerLoRA]:
        if self.binding is not None:
            return list(self.binding.lora or [])
        return list(self.native_lora or [])

    # -------------------------------------------------------------- strict forward
    def _strict_values(
        self, token_ids: list[list[int]], starts: list[int], groups: list[list[int]], *, grad: bool
    ) -> list[torch.Tensor]:
        """Per micro-batch packed fp32 logp (graph kept iff ``grad``)."""
        outs = []
        for group in groups:
            seqs = [token_ids[i] for i in group]
            tokens, cu, positions = pack_sequences(seqs, self.device)
            rows: list[int] = []
            targets: list[int] = []
            segments: list[tuple[int, int]] = []
            offset = 0
            for i in group:
                seq, start = token_ids[i], int(starts[i])
                rows.extend(range(offset + start - 1, offset + len(seq) - 1))
                targets.extend(seq[start:])
                segments.append((offset, offset + len(seq)))
                offset += len(seq)
            select = torch.tensor(rows, dtype=torch.int64, device=self.device)
            tgt = torch.tensor(targets, dtype=torch.int64, device=self.device)
            with torch.set_grad_enabled(grad):
                outs.append(
                    self.binding.forward_packed(tokens, cu, positions, select, tgt, tuple(segments))
                )
        return outs

    def _values(
        self, token_ids: list[list[int]], starts: list[int], groups: list[list[int]], *, grad: bool
    ) -> list[torch.Tensor]:
        if self.strict:
            return self._strict_values(token_ids, starts, groups, grad=grad)
        outs = []
        for group in groups:  # native: one sequence per group (B=1, no padding semantics)
            (i,) = group
            outs.append(native_logprobs_with_grad(
                self.model, token_ids[i], int(starts[i]), self.device, grad=grad
            ))
        return outs

    def cmd_logprobs(self, token_ids: list[list[int]], starts: list[int], keep: bool) -> torch.Tensor:
        """Packed fp32 logp (no graph). ``keep`` registers the request for the next ``step``."""
        _validate_request(token_ids, starts)
        if self.strict:
            groups = micro_batches([len(s) for s in token_ids], self.micro_batch_tokens)
        else:
            groups = [[i] for i in range(len(token_ids))]
        values = [v.detach() for v in self._values(token_ids, starts, groups, grad=False)]
        packed = torch.cat(values) if len(values) > 1 else values[0]
        if keep:
            self._pending = {
                "token_ids": token_ids, "starts": list(starts), "groups": groups, "values": values,
            }
        return packed

    def cmd_backward(self, grad_packed: torch.Tensor) -> dict[str, Any]:
        """Recompute each pending micro-batch with the graph and back-propagate ``grad_packed``."""
        if self._pending is None:
            raise RuntimeError("no pending token_logprobs request; call token_logprobs first")
        pend, self._pending = self._pending, None
        grad = grad_packed.to(self.device, torch.float32)
        mismatches = 0
        checked = 0
        offset = 0
        self._sync()
        t0 = time.perf_counter()
        for group, stored in zip(pend["groups"], pend["values"], strict=True):
            (logp,) = self._values(pend["token_ids"], pend["starts"], [group], grad=True)
            bad = int((logp.detach().view(torch.int32) != stored.view(torch.int32)).sum())
            mismatches += bad
            checked += stored.numel()
            if bad and self.strict:
                # Fail closed BEFORE back-propagating this micro-batch: drop every gradient of
                # the step (earlier micro-batches included) so no optimizer step can apply it.
                del logp
                self.cmd_zero_grad()
                raise StrictRecomputeMismatch(
                    f"strict recompute differs from the returned log-probabilities in {bad} of "
                    f"{stored.numel()} values (micro-batch {group}); step aborted, LoRA "
                    "gradients discarded"
                )
            g = grad[offset : offset + stored.numel()]
            offset += stored.numel()
            logp.backward(g)
            del logp
        self._sync()
        if offset != grad.numel():
            raise RuntimeError(f"gradient has {grad.numel()} entries, pending request {offset}")
        return {
            "recompute_logp_mismatches": mismatches,
            "recompute_logp_checked": checked,
            "micro_batches": len(pend["groups"]),
            "fwd_bwd_s": time.perf_counter() - t0,
        }

    def _sync(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def cmd_set_micro_batch_tokens(self, tokens: int) -> None:
        if int(tokens) < 1:
            raise ValueError("micro_batch_tokens must be positive")
        self.micro_batch_tokens = int(tokens)

    def _grad_norm(self) -> float:
        grads = gather_lora(self._loras(), self.comm, source="grad", dtype=torch.float32,
                            to_cpu=False)
        total = torch.zeros((), dtype=torch.float32, device=self.device)
        for key in sorted(grads):
            g = grads[key]
            total = total + (g * g).sum()
        return float(total.sqrt().item())

    def cmd_optimizer_step(self) -> dict[str, Any]:
        if self.optimizer is None:
            raise RuntimeError("step requires lora_rank > 0")
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        grad_norm = self._grad_norm()
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        self.step_count += 1
        params = lora_parameters(self._loras())
        return {
            "grad_norm": grad_norm,
            "lr": self.lr,
            "step": self.step_count,
            "optimizer": "AdamW(betas=(0.9,0.999), eps=1e-8, foreach=False)",
            "optimizer_s": time.perf_counter() - t0,
            "num_lora_params_local": int(sum(p.numel() for p in params)),
        }

    def cmd_zero_grad(self) -> None:
        for p in lora_parameters(self._loras()):
            p.grad = None

    def cmd_lora_grads(self) -> dict[str, Any]:
        """TP-gathered fp32 LoRA gradients (CPU) + replicated-gradient agreement counts."""
        grads = gather_lora(self._loras(), self.comm, source="grad", dtype=torch.float32)
        checked, differing = replicated_lora_grads_agree(self._loras(), self.comm)
        return {"grads": grads, "replicated_checked": checked, "replicated_differing": differing}

    def cmd_export(self) -> tuple[dict[str, torch.Tensor], str]:
        return export_lora(self._loras(), self.comm)

    def cmd_reset_lora_b(self, std: float, seed: int) -> None:
        for lora in self._loras():
            lora.reset_b(std, seed)

    def cmd_set_lora_enabled(self, flag: bool) -> None:
        if self.binding is not None:
            self.binding.lora_enabled = bool(flag) and self.binding.lora is not None
        if self.native_hooks is not None:
            self.native_hooks.enabled = bool(flag)

    def cmd_native(self, token_ids: list[list[int]], starts: list[int]) -> list[torch.Tensor]:
        outs = []
        for seq, start in zip(token_ids, starts, strict=True):
            lp = native_token_logprobs(self.model, torch.tensor([seq], dtype=torch.int64))
            outs.append(lp[0, int(start) - 1 :].float().cpu())
        return outs

    def cmd_identity(self, model_path: str) -> dict[str, Any]:
        from rl_engine.models.muse_glimmer.runtime.megatron_weights_identity import (
            check_weight_identity,
        )

        t0 = time.perf_counter()
        rep = check_weight_identity(self.model, model_path, self.comm)
        out = rep.to_dict()
        out["seconds"] = time.perf_counter() - t0
        return out

    def cmd_memory(self, reset: bool) -> list[dict[str, float]]:
        torch.cuda.synchronize()
        mine = {
            "rank": self.rank,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "allocated_gib": torch.cuda.memory_allocated() / 2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        }
        if reset:
            torch.cuda.reset_peak_memory_stats()
        if self.world == 1:
            return [mine]
        out: list[Any] = [None] * self.world
        dist.all_gather_object(out, mine, group=self.comm.group)
        return out

    def cmd_native_fwd_bwd(self, seq: list[int]) -> dict[str, Any]:
        """Bridge native forward + backward (activation grads through all 52 layers; base weights
        frozen, full per-layer recompute) of ``-mean(logp)``: the native reference cost of a LoRA
        training step minus the tiny LoRA GEMMs."""
        inner = self.model
        cfg = inner.config
        saved = (cfg.recompute_granularity, cfg.recompute_method, cfg.recompute_num_layers)
        cfg.recompute_granularity, cfg.recompute_method, cfg.recompute_num_layers = "full", "uniform", 1
        handle = inner.embedding.register_forward_hook(
            lambda _m, _i, out: out.detach().requires_grad_(True)
        )
        was_training = inner.training
        inner.train()
        try:
            ids = torch.tensor([seq], dtype=torch.int64, device=self.device)
            t = ids.shape[1]
            pos = torch.arange(t, device=self.device).unsqueeze(0)
            mask = torch.triu(torch.ones(t, t, dtype=torch.bool, device=self.device), 1)[None, None]
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            logits = inner(input_ids=ids, position_ids=pos, attention_mask=mask,
                           runtime_gather_output=True)
            lp = chunked_token_logprobs(logits[0, :-1], ids[0, 1:])
            loss = -lp.mean()
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            loss.backward()
            torch.cuda.synchronize()
            t2 = time.perf_counter()
            del logits, lp, loss
        finally:
            handle.remove()
            inner.train(was_training)
            cfg.recompute_granularity, cfg.recompute_method, cfg.recompute_num_layers = saved
        return {"forward_s": t1 - t0, "backward_s": t2 - t1}

    def provenance(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "engine": "StrictTrainer" if self.strict else "StrictTrainer(native Bridge forward)",
            "runtime": "megatron-bridge",
            "strict": self.strict,
            "tp": self.world,
            "load_seconds": self.load_seconds,
            "step_count": self.step_count,
            "lr": self.lr,
            "micro_batch_tokens": self.micro_batch_tokens,
            "packing_rule": PACKING_RULE if self.strict else "one sequence per forward (native)",
            "torch_flags": self.flags,
        }
        if self.binding is not None:
            out.update(self.binding.provenance())
        else:
            out["lora"] = {"rank": len(self._loras()) and self._loras()[0].rank,
                           "form": NATIVE_LORA_FORM, "enabled": bool(self.native_hooks
                           and self.native_hooks.enabled)}
            out["forward"] = "megatron.bridge MuseGlimmerModel.forward (TE kernels), bf16 logits"
            out["vision_drop"] = getattr(self, "vision_drop", None)
        if self.strict:
            out["lora_grad_rule"] = LORA_GRAD_RULE
            out["recompute_check"] = (
                "bitwise; a mismatch raises StrictRecomputeMismatch (fail closed)"
            )
        return out


def _tp_group() -> Any:
    try:
        from megatron.core import parallel_state

        return parallel_state.get_tensor_model_parallel_group()
    except Exception:  # noqa: BLE001
        return None


def _worker_main(rank: int, world: int, gpu: int, port: int, kwargs: dict[str, Any]) -> None:
    try:
        state = _RankState(rank, world, gpu, port, **kwargs)
        while True:
            payload: list[Any] = [None]
            dist.broadcast_object_list(payload, src=0)
            cmd, args, kw = payload[0]
            if cmd == "close":
                break
            try:
                getattr(state, f"cmd_{cmd}")(*args, **kw)
            except StrictRecomputeMismatch:
                # Raised identically on every rank at the same point (the compared values are
                # the TP-gathered logprobs); rank 0 reports it, the worker keeps serving.
                traceback.print_exc()
        dist.barrier()
        dist.destroy_process_group()
    except BaseException:  # noqa: BLE001 - a dead worker must not leave rank 0 hanging silently
        traceback.print_exc()
        os._exit(1)


class StrictTrainer:
    """Trainer for the Muse-Glimmer-30B strict POC (RFC #501 ``megatron_vime_provider``).

    Constructor arguments match ``e2e.mocks.MockTrainer``; the extra keywords are POC knobs.
    """

    def __init__(
        self,
        model_path: str,
        *,
        tensor_parallel_size: int,
        gpu_ids: list[int],
        lora_rank: int = DEFAULT_LORA_RANK,
        lr: float = 1e-5,
        strict: bool = True,
        seed: int = 0,
        checkpoint_layers: bool = True,
        micro_batch_tokens: int = DEFAULT_MICRO_BATCH_TOKENS,
        lora_b_init_std: float = 0.0,
        weight_decay: float = 0.0,
    ) -> None:
        if tensor_parallel_size not in SUPPORTED_TP:
            raise StrictBindingError(f"StrictTrainer supports tensor_parallel_size in {SUPPORTED_TP}")
        if len(gpu_ids) != tensor_parallel_size:
            raise ValueError("gpu_ids must list one GPU per TP rank")
        self.tp = int(tensor_parallel_size)
        self.gpu_ids = [int(g) for g in gpu_ids]
        self.lora_rank = int(lora_rank)
        self._closed = False
        self._workers: list[Any] = []
        self._leaf: torch.Tensor | None = None
        self._export_sha: str | None = None
        kwargs = dict(
            model_path=os.path.expanduser(str(model_path)), lora_rank=self.lora_rank, lr=float(lr),
            strict=bool(strict), seed=int(seed), checkpoint_layers=bool(checkpoint_layers),
            micro_batch_tokens=int(micro_batch_tokens), lora_b_init_std=float(lora_b_init_std),
            weight_decay=float(weight_decay),
        )
        port = _free_port()
        ctx = mp.get_context("spawn")
        t0 = time.time()
        for rank in range(1, self.tp):
            proc = ctx.Process(
                target=_worker_main, args=(rank, self.tp, self.gpu_ids[rank], port, kwargs)
            )
            proc.daemon = True
            proc.start()
            self._workers.append(proc)
        self._state = _RankState(0, self.tp, self.gpu_ids[0], port, **kwargs)
        self.init_seconds = time.time() - t0

    # -------------------------------------------------------------- command plumbing
    def _call(self, cmd: str, *args: Any, **kwargs: Any) -> Any:
        if self._closed:
            raise RuntimeError("StrictTrainer is closed")
        if self.tp > 1:
            dead = [p.pid for p in self._workers if not p.is_alive()]
            if dead:
                raise RuntimeError(f"TP worker process(es) {dead} died; see their stderr")
            dist.broadcast_object_list([(cmd, args, kwargs)], src=0)
        return getattr(self._state, f"cmd_{cmd}")(*args, **kwargs)

    # -------------------------------------------------------------- TrainerLike
    def token_logprobs(
        self, token_ids: list[list[int]], completion_start: list[int]
    ) -> list[torch.Tensor]:
        """fp32 ``(len_i - completion_start_i,)`` tensors with autograd history (rank 0)."""
        token_ids = [[int(t) for t in seq] for seq in token_ids]
        starts = [int(s) for s in completion_start]
        spans = _validate_request(token_ids, starts)
        packed = self._call("logprobs", token_ids, starts, True)
        self._leaf = packed.detach().clone().requires_grad_(True)
        return list(torch.split(self._leaf, spans))

    def score(self, token_ids: list[list[int]], completion_start: list[int]) -> list[torch.Tensor]:
        """Like :meth:`token_logprobs` but detached and not registered for :meth:`step`."""
        token_ids = [[int(t) for t in seq] for seq in token_ids]
        starts = [int(s) for s in completion_start]
        spans = _validate_request(token_ids, starts)
        packed = self._call("logprobs", token_ids, starts, False)
        return [t.detach() for t in torch.split(packed, spans)]

    def backward(self, loss: torch.Tensor) -> dict[str, Any]:
        """Accumulate LoRA gradients of ``loss`` (built from the last ``token_logprobs``)."""
        if self._state.optimizer is None:
            raise RuntimeError("training requires lora_rank > 0")
        if self._leaf is None:
            raise RuntimeError("call token_logprobs before step/backward")
        leaf, self._leaf = self._leaf, None
        (grad,) = torch.autograd.grad(loss, [leaf])
        return self._call("backward", grad.detach().float().cpu())

    def step(self, loss: torch.Tensor) -> dict[str, Any]:
        """One AdamW step on the LoRA for ANY scalar ``loss`` built from the last logprobs."""
        info = self.backward(loss)
        info.update(self._call("optimizer_step"))
        info["loss"] = float(loss.detach().float().item())
        self._export_sha = None
        return info

    def export_lora(self) -> tuple[dict[str, torch.Tensor], str]:
        state, sha = self._call("export")
        self._export_sha = sha
        return state, sha

    def provenance(self) -> dict[str, Any]:
        out = self._state.provenance()
        if self._state._loras() and self._export_sha is None and not self._closed:
            self.export_lora()
        out["lora_sha256"] = self._export_sha
        out["lora_hash_schema"] = "muse_glimmer_lora_sha256_v1"
        out["gpu_ids"] = list(self.gpu_ids)
        out["init_seconds"] = self.init_seconds
        out["process_model"] = (
            "rank 0 = calling process (returns results); ranks 1..tp-1 = spawned workers; "
            "NCCL tcp://127.0.0.1"
        )
        return out

    def close(self) -> None:
        if self._closed:
            return
        if self.tp > 1:
            if all(p.is_alive() for p in self._workers):
                dist.broadcast_object_list([("close", (), {})], src=0)
                dist.barrier()
            for proc in self._workers:
                proc.join(timeout=120)
                if proc.is_alive():
                    proc.terminate()
            self._workers.clear()
        self._closed = True
        try:
            from megatron.core import parallel_state

            parallel_state.destroy_model_parallel()
        except Exception:  # noqa: BLE001
            pass
        if dist.is_initialized():
            dist.destroy_process_group()

    # -------------------------------------------------------------- POC checks (all ranks)
    def zero_grad(self) -> None:
        self._call("zero_grad")

    def lora_grads(self) -> dict[str, Any]:
        return self._call("lora_grads")

    def set_lora_enabled(self, flag: bool) -> None:
        self._call("set_lora_enabled", bool(flag))

    def set_micro_batch_tokens(self, tokens: int) -> None:
        """Change the packing budget (memory knob; the bits do not depend on it)."""
        self._call("set_micro_batch_tokens", int(tokens))

    def reset_lora_b(self, std: float, seed: int) -> None:
        """Re-draw LoRA ``B`` (TP-invariant); parity tests only (``std=0`` restores the no-op)."""
        self._call("reset_lora_b", float(std), int(seed))
        self._export_sha = None

    def native_logprobs(
        self, token_ids: list[list[int]], completion_start: list[int]
    ) -> list[torch.Tensor]:
        return self._call("native", token_ids, completion_start)

    def weight_identity(self, model_path: str) -> dict[str, Any]:
        return self._call("identity", os.path.expanduser(model_path))

    def memory(self, reset: bool = False) -> list[dict[str, float]]:
        return self._call("memory", bool(reset))

    def native_fwd_bwd(self, seq: list[int]) -> dict[str, Any]:
        return self._call("native_fwd_bwd", [int(t) for t in seq])

    @property
    def binding(self):
        return self._state.binding

    @property
    def model(self):
        return self._state.model

    def __enter__(self) -> "StrictTrainer":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


__all__ = [
    "DEFAULT_MICRO_BATCH_TOKENS",
    "PACKING_RULE",
    "StrictRecomputeMismatch",
    "StrictTrainer",
    "micro_batches",
]
