# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Benchmark O4 ``qknorm_scaled_rope`` (strict Triton) vs the torch-native HF-style composition.

Real Muse-Glimmer-30B shape: T=4096 tokens, q (T, 32, 128), k (T, 2, 128), bf16.  Both sides
receive precomputed cos/sin rows (fp32 for the strict op, bf16 for the HF path, as the model
does); table construction is excluded on both sides.  Forward and forward+backward latency.

Usage (GPU lease required on the shared box), from the repo root:
    ~/bin/gpu-lease -n 1 -- ~/envs/vllm/bin/python \\
        benchmarks/models/muse_glimmer/bench_qknorm_rope.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from rl_engine.models.muse_glimmer.ops.qknorm_rope import (  # noqa: E402
    QK_SCALE,
    RMS_NORM_EPS,
    qknorm_scaled_rope,
    rope_table,
)

HQ, HK, D = 32, 2, 128


def _hf_rmsnorm(x: torch.Tensor, eps: float) -> torch.Tensor:
    h = x.float()
    return (h * torch.pow(h.pow(2).mean(-1, keepdim=True) + eps, -0.5)).type_as(x)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    return torch.cat((-x[..., D // 2 :], x[..., : D // 2]), dim=-1)


def hf_native(q, k, cos_bf16, sin_bf16):
    """HF MuseGlimmerTextAttention semantics: norm->bf16, *3.87 in bf16, bf16 rotate_half."""
    qn = _hf_rmsnorm(q, RMS_NORM_EPS) * QK_SCALE
    kn = _hf_rmsnorm(k, RMS_NORM_EPS)
    c, s = cos_bf16.unsqueeze(1), sin_bf16.unsqueeze(1)
    return (qn * c) + (_rotate_half(qn) * s), (kn * c) + (_rotate_half(kn) * s)


def _time_ms(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dev = torch.device("cuda")
    T = args.tokens
    gen = torch.Generator(device=dev).manual_seed(0)
    q = (torch.randn(T, HQ, D, device=dev, generator=gen) * 2).to(torch.bfloat16)
    k = (torch.randn(T, HK, D, device=dev, generator=gen) * 2).to(torch.bfloat16)
    pos = torch.arange(T, device=dev)
    cos, sin = rope_table(pos, device=dev)
    cos_bf, sin_bf = cos.to(torch.bfloat16), sin.to(torch.bfloat16)
    dyq = torch.randn(q.shape, device=dev, generator=gen).to(torch.bfloat16)
    dyk = torch.randn(k.shape, device=dev, generator=gen).to(torch.bfloat16)

    def strict_fwd():
        return qknorm_scaled_rope(q, k, cos, sin)

    def native_fwd():
        return hf_native(q, k, cos_bf, sin_bf)

    def strict_fwd_bwd():
        qg, kg = q.clone().requires_grad_(True), k.clone().requires_grad_(True)
        torch.autograd.grad(qknorm_scaled_rope(qg, kg, cos, sin), (qg, kg), (dyq, dyk))

    def native_fwd_bwd():
        qg, kg = q.clone().requires_grad_(True), k.clone().requires_grad_(True)
        torch.autograd.grad(hf_native(qg, kg, cos_bf, sin_bf), (qg, kg), (dyq, dyk))

    rows = []
    for name, strict, native in (
        ("forward", strict_fwd, native_fwd),
        ("forward+backward", strict_fwd_bwd, native_fwd_bwd),
    ):
        ms_s = _time_ms(strict, args.warmup, args.iters)
        ms_n = _time_ms(native, args.warmup, args.iters)
        rows.append((name, ms_s, ms_n, ms_n / ms_s))
    bytes_fwd = 2 * (q.numel() + k.numel()) * 2 + (cos.numel() + sin.numel()) * 4
    print(f"device: {torch.cuda.get_device_name()}  T={T} q{tuple(q.shape)} k{tuple(k.shape)} bf16")
    print(f"{'pass':<18}{'strict Triton ms':>18}{'torch-native HF ms':>20}{'native/strict':>15}")
    for name, ms_s, ms_n, ratio in rows:
        print(f"{name:<18}{ms_s:>18.4f}{ms_n:>20.4f}{ratio:>15.2f}x")
    fwd_ms = rows[0][1]
    print(
        f"strict forward effective bandwidth: {bytes_fwd / fwd_ms / 1e6:.0f} GB/s "
        f"({bytes_fwd / 1e6:.1f} MB moved)"
    )


if __name__ == "__main__":
    main()
