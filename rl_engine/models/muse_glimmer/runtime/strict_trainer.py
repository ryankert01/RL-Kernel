# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""``StrictTrainer``: the Megatron Bridge trainer the E2E orchestrator drives.

Process model: rank 0 is the calling process (so the returned log-probabilities carry a real
autograd graph); ranks ``1..tp-1`` are spawned workers that replay every command rank 0
broadcasts (``torch.distributed.broadcast_object_list``). Log-probabilities are replicated on all
TP ranks, so the orchestrator's loss gradient ``dL/dlogp`` (computed on rank 0) is broadcast and
every rank runs the same backward on its own graph; replicated LoRA parameters receive identical
updates on every rank, sharded ones their shard.

``strict=False`` keeps Bridge's native forward (no LoRA, no gradients) for reference dumps only.
"""

from __future__ import annotations

import os
import socket
import time
from typing import Any

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from rl_engine.models.muse_glimmer.runtime import strict_kernels as sk
from rl_engine.models.muse_glimmer.runtime.megatron_lora import DEFAULT_LORA_RANK
from rl_engine.models.muse_glimmer.runtime.megatron_strict import (
    StrictBindingError,
    binding_lora_parameters,
    export_binding_lora,
    install_museglimmer_megatron_strict,
    load_bridge_model,
    native_token_logprobs,
    pack_sequences,
)


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
    dist.init_process_group("nccl", rank=rank, world_size=world, init_method=f"tcp://127.0.0.1:{port}")
    from megatron.core import parallel_state
    from megatron.core.tensor_parallel import model_parallel_cuda_manual_seed

    parallel_state.initialize_model_parallel(tensor_model_parallel_size=world)
    model_parallel_cuda_manual_seed(0)


class _RankState:
    """Everything one TP rank holds; identical code path on rank 0 and the workers."""

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
    ) -> None:
        self.rank = rank
        self.world = world
        self.strict = strict
        self.step_count = 0
        self.lr = lr
        torch.manual_seed(seed)
        sk.set_deterministic_torch()
        _init_distributed(rank, world, gpu, port)
        self.model, self.bridge, self.load_seconds = load_bridge_model(model_path, tensor_parallel_size=world)
        self.binding = None
        self.optimizer = None
        self._last_logp: torch.Tensor | None = None
        if strict:
            self.binding = install_museglimmer_megatron_strict(
                self.model, hf_path=model_path, lora_rank=lora_rank, seed=seed,
                checkpoint_layers=checkpoint_layers,
            )
            params = binding_lora_parameters(self.binding)
            if params:
                self.optimizer = torch.optim.AdamW(
                    params, lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, foreach=False
                )
        self.device = torch.device("cuda", gpu)

    # -------------------------------------------------------------- commands
    def logprobs(self, token_ids: list[list[int]], completion_start: list[int]) -> list[torch.Tensor]:
        if self.strict:
            tokens, cu, positions = pack_sequences(token_ids, self.device)
            rows, targets, spans = [], [], []
            offset = 0
            for seq, start in zip(token_ids, completion_start, strict=True):
                n = len(seq)
                if not 1 <= start < n:
                    raise ValueError(f"completion_start {start} out of range for length {n}")
                rows.extend(range(offset + start - 1, offset + n - 1))
                targets.extend(seq[start:])
                spans.append(n - start)
                offset += n
            select = torch.tensor(rows, dtype=torch.int64, device=self.device)
            tgt = torch.tensor(targets, dtype=torch.int64, device=self.device)
            logp = self.binding.forward_packed(tokens, cu, positions, select, tgt)
            self._last_logp = logp
            return list(torch.split(logp, spans))
        out = []
        for seq, start in zip(token_ids, completion_start, strict=True):
            lp = native_token_logprobs(self.model, torch.tensor([seq], dtype=torch.int64))
            out.append(lp[0, start - 1 :].detach())
        return out

    def backward(self, grad_packed: torch.Tensor) -> dict[str, Any]:
        if self._last_logp is None:
            raise RuntimeError("no logprobs with a live graph; call token_logprobs first")
        self._last_logp.backward(grad_packed.to(self._last_logp.device, torch.float32))
        self._last_logp = None
        params = binding_lora_parameters(self.binding)
        local_sq = torch.zeros((), dtype=torch.float32, device=self.device)
        for p in params:  # fixed parameter order
            if p.grad is not None:
                local_sq = local_sq + (p.grad.float() * p.grad.float()).sum()
        parts = self.binding.comm.all_gather(local_sq.reshape(1))
        total = parts[0]
        for part in parts[1:]:
            total = total + part
        grad_norm = float(total.sqrt().item())
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.step_count += 1
        return {
            "grad_norm": grad_norm,
            "lr": self.lr,
            "step": self.step_count,
            "num_lora_params": int(sum(p.numel() for p in params)),
        }

    def export(self) -> tuple[dict[str, torch.Tensor], str]:
        return export_binding_lora(self.binding)

    def provenance(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "strict": self.strict,
            "tp": self.world,
            "load_seconds": self.load_seconds,
            "step_count": self.step_count,
            "lr": self.lr,
        }
        if self.binding is not None:
            out.update(self.binding.provenance())
        return out


def _worker_main(rank: int, world: int, gpu: int, port: int, kwargs: dict[str, Any]) -> None:
    state = _RankState(rank, world, gpu, port, **kwargs)
    while True:
        payload = [None]
        dist.broadcast_object_list(payload, src=0)
        cmd, args = payload[0]
        if cmd == "logprobs":
            state.logprobs(*args)
        elif cmd == "backward":
            state.backward(args[0])
        elif cmd == "export":
            state.export()
        elif cmd == "close":
            break
    dist.destroy_process_group()


class StrictTrainer:
    """Trainer interface for the Muse-Glimmer-30B strict POC (RFC #501 ``megatron_vime_provider``)."""

    def __init__(
        self,
        model_path: str,
        *,
        tensor_parallel_size: int,
        gpu_ids: list[int],
        lora_rank: int = DEFAULT_LORA_RANK,
        lr: float = 1e-4,
        strict: bool = True,
        seed: int = 0,
        checkpoint_layers: bool = True,
    ) -> None:
        if tensor_parallel_size not in (1, 2):
            raise StrictBindingError("StrictTrainer supports tensor_parallel_size 1 or 2")
        if len(gpu_ids) != tensor_parallel_size:
            raise ValueError("gpu_ids must list one GPU per TP rank")
        self.tp = tensor_parallel_size
        self.gpu_ids = list(gpu_ids)
        self._workers: list[mp.Process] = []
        kwargs = dict(
            model_path=os.path.expanduser(model_path), lora_rank=lora_rank, lr=lr, strict=strict,
            seed=seed, checkpoint_layers=checkpoint_layers,
        )
        port = _free_port()
        ctx = mp.get_context("spawn")
        for rank in range(1, self.tp):
            proc = ctx.Process(target=_worker_main, args=(rank, self.tp, self.gpu_ids[rank], port, kwargs))
            proc.daemon = True
            proc.start()
            self._workers.append(proc)
        t0 = time.time()
        self._state = _RankState(0, self.tp, self.gpu_ids[0], port, **kwargs)
        self.init_seconds = time.time() - t0

    def _broadcast(self, cmd: str, *args: Any) -> None:
        if self.tp > 1:
            dist.broadcast_object_list([(cmd, args)], src=0)

    def token_logprobs(self, token_ids: list[list[int]], completion_start: list[int]) -> list[torch.Tensor]:
        self._broadcast("logprobs", token_ids, completion_start)
        return self._state.logprobs(token_ids, completion_start)

    def step(self, loss: torch.Tensor) -> dict:
        if not self._state.strict or self._state.optimizer is None:
            raise RuntimeError("step requires strict=True and lora_rank > 0")
        logp = self._state._last_logp
        if logp is None:
            raise RuntimeError("call token_logprobs before step")
        (grad,) = torch.autograd.grad(loss, [logp], retain_graph=False)
        grad = grad.detach().float().cpu()
        self._broadcast("backward", grad)
        return self._state.backward(grad)

    def export_lora(self) -> tuple[dict[str, torch.Tensor], str]:
        self._broadcast("export")
        return self._state.export()

    def provenance(self) -> dict:
        out = self._state.provenance()
        out["gpu_ids"] = self.gpu_ids
        out["init_seconds"] = self.init_seconds
        return out

    @property
    def binding(self):
        return self._state.binding

    @property
    def model(self):
        return self._state.model

    def close(self) -> None:
        self._broadcast("close")
        for proc in self._workers:
            proc.join(timeout=120)
        self._workers.clear()
        if dist.is_initialized():
            dist.destroy_process_group()


__all__ = ["StrictTrainer"]
