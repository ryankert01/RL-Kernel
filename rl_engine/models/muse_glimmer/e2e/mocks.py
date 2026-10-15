# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""CPU mock engines with the FIXED ``StrictRolloutEngine`` / ``StrictTrainer`` surface.

Both mocks share one tiny LM ("the checkpoint": deterministic init from ``TINY_BASE_SEED``) with
LoRA on the eight projection names of the real key schema. The two numerics paths are a
miniature of the RFC:

* **strict**: fp32 only, built from exactly-rounded elementwise primitives (``+ - * / sqrt max
  floor frexp`` and integer bit ops) plus polynomial ``exp``/``log``; GEMMs are a sequential K
  loop of ``mul`` then ``add``; the causal running mean is a sequential prefix sum. Every token's
  bits are therefore independent of batch size, batch position, padding, sequence split and
  decode-vs-prefill BY CONSTRUCTION (no BLAS, no vectorized transcendental with a scalar tail).
  Casts: LoRA A/B are held as bf16 (trainer: fp32 master -> bf16 -> fp32 at use) and upcast once;
  nothing else is rounded below fp32.
* **native**: ``F.linear`` (BLAS), ``torch.exp``/``log_softmax`` and bf16 casts whose placement
  is an engine-private choice (:class:`NativePolicy`): the native rollout rounds after every
  linear and keeps its running-mean state in bf16 (the KV-cache analog) with a Welford update in
  decode; the native trainer keeps fp32 activations, rounds once per block and uses a cumsum in
  prefill. Different-but-legitimate numerics per engine and per phase, as in the real runtimes.

Consequences the tests check: strict/strict drift is exactly zero; native/native, strict/native,
native/strict and a weight-perturbed rollout give nonzero drift; native decode-vs-prefill is
nonzero while strict decode-vs-prefill is zero.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterator, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from rl_engine.models.muse_glimmer.e2e.engines import RolloutSample
from rl_engine.models.muse_glimmer.e2e.sync import LORA_PROJECTIONS, WeightSyncError, lora_sha256

TINY_BASE_SEED = 20261015


@dataclass(frozen=True)
class TinyLMConfig:
    vocab_size: int = 259
    hidden_size: int = 32
    intermediate_size: int = 64
    num_layers: int = 2
    rms_norm_eps: float = 1e-5


@dataclass(frozen=True)
class NativePolicy:
    """Where a native engine rounds to bf16; each engine has its own, as in the real world."""

    name: str
    cast_each_linear: bool
    bf16_state: bool


ROLLOUT_NATIVE_POLICY = NativePolicy(
    "rollout-native: bf16 after every linear, bf16 running-mean state, Welford decode",
    cast_each_linear=True,
    bf16_state=True,
)
TRAINER_NATIVE_POLICY = NativePolicy(
    "trainer-native: fp32 activations, bf16 cast per block, cumsum prefill",
    cast_each_linear=False,
    bf16_state=False,
)


class ByteTokenizer:
    """Byte-level tokenizer: ids 0..255 are bytes, then BOS, EOS, PAD."""

    BOS = 256
    EOS = 257
    PAD = 258
    vocab_size = 259
    eos_token_ids = (EOS,)

    def encode_chat(self, messages: list[dict[str, str]]) -> list[int]:
        text = "".join(f"<{m['role']}>{m['content']}\n" for m in messages) + "<assistant>"
        return [self.BOS, *text.encode("utf-8")]

    def decode(self, token_ids: Sequence[int]) -> str:
        raw = bytes(int(t) for t in token_ids if 0 <= int(t) < 256)
        return raw.decode("utf-8", errors="replace")


# --------------------------------------------------------------------------------------------
# Strict primitives: exactly-rounded elementwise ops only (shape-invariant by construction).
# --------------------------------------------------------------------------------------------

_LN2 = 0.6931471805599453
_LN2_HI = 0.693145751953125  # exactly representable in fp32
_LN2_LO = 1.428606765330187e-06
_INV_LN2 = 1.4426950408889634
_SQRT_HALF = 0.7071067811865476
_EXP_HORNER = (1.0 / 5040.0, 1.0 / 720.0, 1.0 / 120.0, 1.0 / 24.0, 1.0 / 6.0, 0.5, 1.0, 1.0)


def _strict_exp_impl(x: torch.Tensor) -> torch.Tensor:
    """exp(x) via Cody-Waite range reduction, degree-7 Horner polynomial and exact 2^n scaling."""

    x = torch.clamp(x, min=-87.0, max=88.0)
    n = torch.floor(x * _INV_LN2 + 0.5)
    r = x - n * _LN2_HI
    r = r - n * _LN2_LO
    p = r * _EXP_HORNER[0] + _EXP_HORNER[1]
    for coefficient in _EXP_HORNER[2:]:
        p = p * r + coefficient
    exponent_bits = (n.to(torch.int32) + 127) << 23
    scale = exponent_bits.contiguous().view(torch.float32)
    return p * scale


def _strict_log_impl(x: torch.Tensor) -> torch.Tensor:
    """log(x) for x > 0 via frexp and the atanh series on [sqrt(1/2), sqrt(2))."""

    mantissa, exponent = torch.frexp(x)
    small = mantissa < _SQRT_HALF
    mantissa = torch.where(small, mantissa * 2.0, mantissa)
    exponent = torch.where(small, exponent - 1, exponent)
    f = mantissa - 1.0
    u = f / (2.0 + f)
    u2 = u * u
    poly = u2 * (2.0 / 9.0) + (2.0 / 7.0)
    poly = poly * u2 + (2.0 / 5.0)
    poly = poly * u2 + (2.0 / 3.0)
    poly = poly * u2 + 2.0
    return exponent.to(torch.float32) * _LN2 + u * poly


class _StrictExp(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
        y = _strict_exp_impl(x)
        ctx.save_for_backward(y)
        return y

    @staticmethod
    def backward(ctx: Any, grad: torch.Tensor) -> torch.Tensor:
        (y,) = ctx.saved_tensors
        return grad * y


class _StrictLog(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x)
        return _strict_log_impl(x)

    @staticmethod
    def backward(ctx: Any, grad: torch.Tensor) -> torch.Tensor:
        (x,) = ctx.saved_tensors
        return grad / x


def strict_exp(x: torch.Tensor) -> torch.Tensor:
    return _StrictExp.apply(x)


def strict_log(x: torch.Tensor) -> torch.Tensor:
    return _StrictLog.apply(x)


def strict_sigmoid(x: torch.Tensor) -> torch.Tensor:
    return 1.0 / (1.0 + strict_exp(-x))


def strict_silu(x: torch.Tensor) -> torch.Tensor:
    return x * strict_sigmoid(x)


def strict_sum_last(x: torch.Tensor) -> torch.Tensor:
    """Sequential sum over the last dim (keepdim), order k = 0..K-1."""

    acc = x[..., 0:1]
    for k in range(1, x.shape[-1]):
        acc = acc + x[..., k : k + 1]
    return acc


def strict_linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """``x @ weight.T`` as a sequential K loop: acc_k = acc_{k-1} + x[..., k] * weight[:, k]."""

    acc = x[..., 0:1] * weight[:, 0]
    for k in range(1, weight.shape[1]):
        acc = acc + x[..., k : k + 1] * weight[:, k]
    return acc


def strict_rmsnorm(x: torch.Tensor, eps: float) -> torch.Tensor:
    mean_square = strict_sum_last(x * x) / float(x.shape[-1])
    return x * (1.0 / torch.sqrt(mean_square + eps))


def strict_log_softmax(logits: torch.Tensor) -> torch.Tensor:
    shift = logits - logits.max(dim=-1, keepdim=True).values
    return shift - strict_log(strict_sum_last(strict_exp(shift)))


def native_rmsnorm(x: torch.Tensor, eps: float) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)


# --------------------------------------------------------------------------------------------
# Tiny LM with LoRA on the real key schema.
# --------------------------------------------------------------------------------------------


class LoRALinear(nn.Module):
    """Frozen base weight + LoRA (A: (r, K), B: (N, r)); bf16 LoRA storage on both sides."""

    def __init__(
        self, in_features: int, out_features: int, rank: int, *, trainable: bool, init_std: float
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.trainable = trainable
        weight = torch.randn(out_features, in_features) * init_std
        self.weight = nn.Parameter(weight, requires_grad=False)
        if trainable:
            self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
            self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        else:
            self.register_buffer("lora_A", torch.zeros(rank, in_features, dtype=torch.bfloat16))
            self.register_buffer("lora_B", torch.zeros(out_features, rank, dtype=torch.bfloat16))

    def lora_fp32(self) -> tuple[torch.Tensor, torch.Tensor]:
        a = self.lora_A.to(torch.bfloat16).to(torch.float32)
        b = self.lora_B.to(torch.bfloat16).to(torch.float32)
        return a, b

    def export(self) -> tuple[torch.Tensor, torch.Tensor]:
        a = self.lora_A.detach().to(torch.bfloat16).cpu().contiguous()
        b = self.lora_B.detach().to(torch.bfloat16).cpu().contiguous()
        return a, b

    @torch.no_grad()
    def load(self, a: torch.Tensor, b: torch.Tensor) -> None:
        if a.dtype != torch.bfloat16 or b.dtype != torch.bfloat16:
            raise WeightSyncError("LoRA tensors must be bf16")
        if tuple(a.shape) != tuple(self.lora_A.shape) or tuple(b.shape) != tuple(self.lora_B.shape):
            raise WeightSyncError(
                f"LoRA shape mismatch: got A{tuple(a.shape)} B{tuple(b.shape)}, "
                f"expected A{tuple(self.lora_A.shape)} B{tuple(self.lora_B.shape)}"
            )
        self.lora_A.copy_(a.to(self.lora_A.dtype))
        self.lora_B.copy_(b.to(self.lora_B.dtype))

    def forward_strict(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.lora_fp32()
        return strict_linear(x, self.weight) + strict_linear(strict_linear(x, a), b)

    def forward_native(self, x: torch.Tensor, *, cast: bool) -> torch.Tensor:
        a, b = self.lora_fp32()
        y = F.linear(x, self.weight) + F.linear(F.linear(x, a), b)
        return y.to(torch.bfloat16).to(torch.float32) if cast else y


State = tuple[torch.Tensor, torch.Tensor]  # (running sum or mean (B, d), count (B, 1))


class TinyBlock(nn.Module):
    def __init__(self, cfg: TinyLMConfig, rank: int, trainable: bool) -> None:
        super().__init__()
        d, f = cfg.hidden_size, cfg.intermediate_size
        self.eps = cfg.rms_norm_eps
        self.q_proj = LoRALinear(d, d, rank, trainable=trainable, init_std=d**-0.5)
        self.k_proj = LoRALinear(d, d, rank, trainable=trainable, init_std=d**-0.5)
        self.v_proj = LoRALinear(d, d, rank, trainable=trainable, init_std=d**-0.5)
        self.gate_proj = LoRALinear(d, d, rank, trainable=trainable, init_std=d**-0.5)
        self.o_proj = LoRALinear(d, d, rank, trainable=trainable, init_std=d**-0.5)
        self.mlp = nn.Module()
        self.mlp.gate_proj = LoRALinear(d, f, rank, trainable=trainable, init_std=d**-0.5)
        self.mlp.up_proj = LoRALinear(d, f, rank, trainable=trainable, init_std=d**-0.5)
        self.mlp.down_proj = LoRALinear(f, d, rank, trainable=trainable, init_std=f**-0.5)

    @staticmethod
    def _strict_causal_mean(k: torch.Tensor, state: State) -> tuple[torch.Tensor, State]:
        total, count = state
        outs = []
        for t in range(k.shape[1]):
            total = total + k[:, t]
            count = count + 1.0
            outs.append(total / count)
        return torch.stack(outs, dim=1), (total, count)

    @staticmethod
    def _native_causal_mean(
        k: torch.Tensor, state: State, *, decode: bool, bf16_state: bool
    ) -> tuple[torch.Tensor, State]:
        mean, count = state
        if decode:
            # Welford update, one token at a time (decode kernel); state optionally in bf16.
            count = count + 1.0
            mean = mean + (k[:, 0] - mean) / count
            if bf16_state:
                mean = mean.to(torch.bfloat16).to(torch.float32)
            return mean.unsqueeze(1), (mean, count)
        steps = torch.arange(1, k.shape[1] + 1, dtype=torch.float32, device=k.device)
        counts = count.unsqueeze(1) + steps.view(1, -1, 1)
        totals = (mean * count).unsqueeze(1) + torch.cumsum(k, dim=1)
        means = totals / counts
        last = means[:, -1]
        if bf16_state:
            last = last.to(torch.bfloat16).to(torch.float32)
        return means, (last, counts[:, -1])

    def forward_strict(self, x: torch.Tensor, state: State) -> tuple[torch.Tensor, State]:
        h = strict_rmsnorm(x, self.eps)
        q = strict_silu(self.q_proj.forward_strict(h))
        k = self.k_proj.forward_strict(h)
        v = self.v_proj.forward_strict(h)
        g = self.gate_proj.forward_strict(h)
        kmean, state = self._strict_causal_mean(k, state)
        a = (q * kmean + v) * strict_sigmoid(g)
        x = x + self.o_proj.forward_strict(a)
        h2 = strict_rmsnorm(x, self.eps)
        m = strict_silu(self.mlp.gate_proj.forward_strict(h2)) * self.mlp.up_proj.forward_strict(h2)
        x = x + self.mlp.down_proj.forward_strict(m)
        return x, state

    def forward_native(
        self, x: torch.Tensor, state: State, *, decode: bool, policy: NativePolicy
    ) -> tuple[torch.Tensor, State]:
        cast = policy.cast_each_linear
        h = native_rmsnorm(x, self.eps)
        q = F.silu(self.q_proj.forward_native(h, cast=cast))
        k = self.k_proj.forward_native(h, cast=cast)
        v = self.v_proj.forward_native(h, cast=cast)
        g = self.gate_proj.forward_native(h, cast=cast)
        kmean, state = self._native_causal_mean(
            k, state, decode=decode, bf16_state=policy.bf16_state
        )
        a = (q * kmean + v) * torch.sigmoid(g)
        x = x + self.o_proj.forward_native(a, cast=cast)
        h2 = native_rmsnorm(x, self.eps)
        m = F.silu(self.mlp.gate_proj.forward_native(h2, cast=cast))
        m = m * self.mlp.up_proj.forward_native(h2, cast=cast)
        x = x + self.mlp.down_proj.forward_native(m, cast=cast)
        return x.to(torch.bfloat16).to(torch.float32), state


class TinyLM(nn.Module):
    """2-block byte LM. Base weights come from ``base_seed`` (identical on both engines)."""

    def __init__(
        self,
        cfg: TinyLMConfig,
        *,
        lora_rank: int,
        trainable_lora: bool,
        base_seed: int = TINY_BASE_SEED,
        perturb: float = 0.0,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.lora_rank = lora_rank
        self.trainable_lora = trainable_lora
        self.base_seed = base_seed
        self.perturb = perturb
        d = cfg.hidden_size
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(base_seed)
            self.embed = nn.Parameter(torch.randn(cfg.vocab_size, d) * 0.1, requires_grad=False)
            self.layers = nn.ModuleList(
                [TinyBlock(cfg, lora_rank, trainable_lora) for _ in range(cfg.num_layers)]
            )
            self.lm_head = nn.Parameter(
                torch.randn(cfg.vocab_size, d) * d**-0.5, requires_grad=False
            )
        if perturb:
            noise = torch.randn(
                self.lm_head.shape, generator=torch.Generator().manual_seed(base_seed + 1)
            )
            with torch.no_grad():
                self.lm_head.add_(noise * perturb)

    def lora_modules(self) -> Iterator[tuple[str, LoRALinear]]:
        for i, layer in enumerate(self.layers):
            for proj in LORA_PROJECTIONS:
                module = layer.get_submodule(proj)
                assert isinstance(module, LoRALinear)
                yield f"layers.{i}.{proj}", module

    def lora_parameters(self) -> list[nn.Parameter]:
        params: list[nn.Parameter] = []
        for _, module in self.lora_modules():
            params.extend([module.lora_A, module.lora_B])  # type: ignore[list-item]
        return params

    @torch.no_grad()
    def init_lora(self, seed: int) -> None:
        generator = torch.Generator().manual_seed(seed)
        for _, module in self.lora_modules():
            bound = module.in_features**-0.5
            module.lora_A.uniform_(-bound, bound, generator=generator)
            module.lora_B.zero_()

    def export_lora(self) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        for name, module in self.lora_modules():
            a, b = module.export()
            out[f"{name}.lora_A"] = a
            out[f"{name}.lora_B"] = b
        return out

    def load_lora(self, state_dict: dict[str, torch.Tensor]) -> None:
        expected = set()
        for name, _ in self.lora_modules():
            expected.update({f"{name}.lora_A", f"{name}.lora_B"})
        if set(state_dict) != expected:
            missing = sorted(expected - set(state_dict))
            unexpected = sorted(set(state_dict) - expected)
            raise WeightSyncError(f"LoRA keys differ: missing {missing}, unexpected {unexpected}")
        for name, module in self.lora_modules():
            module.load(state_dict[f"{name}.lora_A"], state_dict[f"{name}.lora_B"])

    def initial_state(self, batch: int) -> list[State]:
        d = self.cfg.hidden_size
        return [
            (torch.zeros(batch, d), torch.zeros(batch, 1)) for _ in range(self.cfg.num_layers)
        ]

    def forward_strict(
        self, ids: torch.Tensor, states: list[State]
    ) -> tuple[torch.Tensor, list[State]]:
        x = self.embed[ids]
        new_states = []
        for layer, state in zip(self.layers, states):
            x, state = layer.forward_strict(x, state)
            new_states.append(state)
        return strict_linear(strict_rmsnorm(x, self.cfg.rms_norm_eps), self.lm_head), new_states

    def forward_native(
        self, ids: torch.Tensor, states: list[State], *, decode: bool, policy: NativePolicy
    ) -> tuple[torch.Tensor, list[State]]:
        x = self.embed[ids]
        new_states = []
        for layer, state in zip(self.layers, states):
            x, state = layer.forward_native(x, state, decode=decode, policy=policy)
            new_states.append(state)
        return F.linear(native_rmsnorm(x, self.cfg.rms_norm_eps), self.lm_head), new_states


def selected_logprobs(logp: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """``logp[:, t, ids[:, t + 1]]`` for t in 0..T-2, shape ``(B, T - 1)``."""

    return logp[:, :-1].gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)


def _merge_states(per_row: list[list[State]]) -> list[State]:
    merged = []
    for layer_states in zip(*per_row):
        merged.append(
            (
                torch.cat([s[0] for s in layer_states], dim=0),
                torch.cat([s[1] for s in layer_states], dim=0),
            )
        )
    return merged


class MockRolloutEngine:
    """CPU stand-in for ``StrictRolloutEngine`` (same constructor and methods)."""

    def __init__(
        self,
        model_path: str,
        *,
        tensor_parallel_size: int,
        gpu_ids: list[int],
        strict: bool,
        max_model_len: int,
        seed: int,
        lora_rank: int = 4,
        config: TinyLMConfig | None = None,
        perturb: float = 0.0,
    ) -> None:
        self.model_path = str(model_path)
        self.tensor_parallel_size = int(tensor_parallel_size)
        self.gpu_ids = [int(g) for g in gpu_ids]
        self.strict = bool(strict)
        self.max_model_len = int(max_model_len)
        self.seed = int(seed)
        self.cfg = config or TinyLMConfig()
        self.perturb = float(perturb)
        self.model = TinyLM(self.cfg, lora_rank=lora_rank, trainable_lora=False, perturb=perturb)
        self.model.eval()
        self.native_policy = ROLLOUT_NATIVE_POLICY
        self.tokenizer = ByteTokenizer()
        self._generator = torch.Generator().manual_seed(self.seed)
        self._lora_sha256: str | None = None
        self._lora_loads = 0
        self._generate_calls = 0

    def _forward(
        self, ids: torch.Tensor, states: list[State], *, decode: bool
    ) -> tuple[torch.Tensor, list[State]]:
        if self.strict:
            return self.model.forward_strict(ids, states)
        return self.model.forward_native(ids, states, decode=decode, policy=self.native_policy)

    def _log_softmax(self, logits: torch.Tensor) -> torch.Tensor:
        if self.strict:
            return strict_log_softmax(logits)
        return torch.log_softmax(logits, dim=-1)

    @torch.no_grad()
    def generate(
        self,
        prompt_token_ids: list[list[int]],
        *,
        n: int,
        max_tokens: int,
        temperature: float,
    ) -> list[list[RolloutSample]]:
        if n <= 0 or max_tokens <= 0:
            raise ValueError("n and max_tokens must be positive")
        self._generate_calls += 1
        rows = [list(p) for p in prompt_token_ids for _ in range(n)]
        for row in rows:
            if len(row) + max_tokens > self.max_model_len:
                raise ValueError("prompt + max_tokens exceeds max_model_len")
        per_row_states: list[list[State]] = []
        last_logits = []
        for row in rows:
            ids = torch.tensor([row], dtype=torch.long)
            logits, states = self._forward(ids, self.model.initial_state(1), decode=False)
            last_logits.append(logits[0, -1])
            per_row_states.append(states)
        states = _merge_states(per_row_states)
        logits = torch.stack(last_logits, dim=0)
        batch = len(rows)
        generated: list[list[int]] = [[] for _ in rows]
        logprobs: list[list[float]] = [[] for _ in rows]
        finished = [False] * batch
        reasons = ["length"] * batch
        eos = set(self.tokenizer.eos_token_ids)
        for _ in range(max_tokens):
            logp = self._log_softmax(logits)
            if temperature <= 0.0:
                next_ids = logits.argmax(dim=-1)
            else:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_ids = torch.multinomial(probs, 1, generator=self._generator).squeeze(1)
            chosen = logp.gather(-1, next_ids.unsqueeze(-1)).squeeze(-1)
            for b in range(batch):
                if finished[b]:
                    continue
                token = int(next_ids[b].item())
                generated[b].append(token)
                logprobs[b].append(float(chosen[b].item()))
                if token in eos:
                    finished[b] = True
                    reasons[b] = "stop"
            if all(finished):
                break
            step_logits, states = self._forward(next_ids.unsqueeze(1), states, decode=True)
            logits = step_logits[:, 0]
        groups: list[list[RolloutSample]] = []
        for p in range(len(prompt_token_ids)):
            group = []
            for j in range(n):
                b = p * n + j
                group.append(RolloutSample(generated[b], logprobs[b], reasons[b]))
            groups.append(group)
        return groups

    @torch.no_grad()
    def logprobs(
        self, token_ids: list[list[int]], completion_start: list[int]
    ) -> list[list[float]]:
        out: list[list[float]] = []
        for row, start in zip(token_ids, completion_start):
            ids = torch.tensor([list(row)], dtype=torch.long)
            logits, _ = self._forward(ids, self.model.initial_state(1), decode=False)
            selected = selected_logprobs(self._log_softmax(logits), ids)[0]
            out.append(selected[int(start) - 1 :].float().tolist())
        return out

    def load_lora(self, state_dict: dict[str, torch.Tensor], sha256: str) -> None:
        actual = lora_sha256(state_dict)
        if actual != sha256:
            raise WeightSyncError(f"load_lora: declared sha256 {sha256} != computed {actual}")
        self.model.load_lora(state_dict)
        self._lora_sha256 = sha256
        self._lora_loads += 1

    def provenance(self) -> dict[str, Any]:
        return {
            "engine": "MockRolloutEngine",
            "runtime": "cpu-mock (no vLLM)",
            "strict": self.strict,
            "logprob_path": (
                "strict: fp32 exactly-rounded primitives, polynomial exp/log, sequential sums"
                if self.strict
                else self.native_policy.name
            ),
            "tensor_parallel_size": self.tensor_parallel_size,
            "gpu_ids": list(self.gpu_ids),
            "max_model_len": self.max_model_len,
            "seed": self.seed,
            "lora_sha256": self._lora_sha256,
            "lora_loads": self._lora_loads,
            "generate_calls": self._generate_calls,
            "tiny_config": asdict(self.cfg),
            "base_seed": self.model.base_seed,
            "perturb": self.perturb,
            "logprob_temperature": 1.0,
        }

    def close(self) -> None:
        return None


class MockTrainer:
    """CPU stand-in for ``StrictTrainer`` (same constructor and methods)."""

    def __init__(
        self,
        model_path: str,
        *,
        tensor_parallel_size: int,
        gpu_ids: list[int],
        lora_rank: int,
        lr: float,
        strict: bool,
        seed: int,
        config: TinyLMConfig | None = None,
    ) -> None:
        self.model_path = str(model_path)
        self.tensor_parallel_size = int(tensor_parallel_size)
        self.gpu_ids = [int(g) for g in gpu_ids]
        self.lora_rank = int(lora_rank)
        self.lr = float(lr)
        self.strict = bool(strict)
        self.seed = int(seed)
        self.cfg = config or TinyLMConfig()
        self.model = TinyLM(self.cfg, lora_rank=self.lora_rank, trainable_lora=True)
        self.model.init_lora(self.seed)
        self.native_policy = TRAINER_NATIVE_POLICY
        self.optimizer = torch.optim.Adam(self.model.lora_parameters(), lr=self.lr)
        self.pad_id = ByteTokenizer.PAD
        self._step = 0

    def token_logprobs(
        self, token_ids: list[list[int]], completion_start: list[int]
    ) -> list[torch.Tensor]:
        if len(token_ids) != len(completion_start):
            raise ValueError("token_ids and completion_start must align")
        lengths = [len(row) for row in token_ids]
        width = max(lengths)
        ids = torch.full((len(token_ids), width), self.pad_id, dtype=torch.long)
        for i, row in enumerate(token_ids):
            ids[i, : len(row)] = torch.tensor(row, dtype=torch.long)
        states = self.model.initial_state(len(token_ids))
        if self.strict:
            logits, _ = self.model.forward_strict(ids, states)
            logp = strict_log_softmax(logits)
        else:
            logits, _ = self.model.forward_native(
                ids, states, decode=False, policy=self.native_policy
            )
            logp = torch.log_softmax(logits, dim=-1)
        selected = selected_logprobs(logp, ids)
        out = []
        for i, (length, start) in enumerate(zip(lengths, completion_start)):
            if not 0 < start < length:
                raise ValueError(f"sample {i}: completion_start {start} out of range for {length}")
            out.append(selected[i, start - 1 : length - 1])
        return out

    def step(self, loss: torch.Tensor) -> dict[str, Any]:
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        params = self.model.lora_parameters()
        squares = [p.grad.detach().pow(2).sum() for p in params if p.grad is not None]
        grad_norm = float(torch.stack(squares).sum().sqrt().item()) if squares else 0.0
        self.optimizer.step()
        self._step += 1
        return {
            "step": self._step,
            "loss": float(loss.detach().item()),
            "grad_norm": grad_norm,
            "lr": self.lr,
            "lora_param_count": int(sum(p.numel() for p in params)),
        }

    def export_lora(self) -> tuple[dict[str, torch.Tensor], str]:
        state_dict = self.model.export_lora()
        return state_dict, lora_sha256(state_dict)

    def provenance(self) -> dict[str, Any]:
        return {
            "engine": "MockTrainer",
            "runtime": "cpu-mock (no Megatron)",
            "strict": self.strict,
            "logprob_path": (
                "strict: fp32 exactly-rounded primitives, batched right-padded (invariant)"
                if self.strict
                else self.native_policy.name
            ),
            "tensor_parallel_size": self.tensor_parallel_size,
            "gpu_ids": list(self.gpu_ids),
            "lora_rank": self.lora_rank,
            "lr": self.lr,
            "optimizer": "Adam",
            "seed": self.seed,
            "optimizer_steps": self._step,
            "tiny_config": asdict(self.cfg),
            "base_seed": self.model.base_seed,
            "lora_storage": "fp32 master -> bf16 at use and at export",
        }

    def close(self) -> None:
        return None
