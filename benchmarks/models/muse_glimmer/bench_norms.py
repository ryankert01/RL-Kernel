# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Strict Muse-Glimmer norms vs torch-native equivalents at the real shape (T=4096, H=6656).

Run:  PYTHONPATH=. ~/bin/gpu-lease -n 1 -- ~/envs/vllm/bin/python \
          benchmarks/models/muse_glimmer/bench_norms.py

Native baselines are the HF formulas in torch eager (bf16 in, fp32 compute, bf16 out), which is
what the trainer/rollout would run without the strict kernels. Forward is the contract benchmark;
backward numbers are informational (they include the fixed-order dw column reduction).
"""

from __future__ import annotations

import argparse
import statistics

import torch
import torch.nn.functional as F

from rl_engine.models.muse_glimmer.ops import norms as N


def _time_ms(fn, *, iters: int, warmup: int = 10) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples)


def _native_centered(x, w, eps):
    x32 = x.float()
    y = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps)
    return (y * (1.0 + w.float())).to(torch.bfloat16)


def _native_weighted(x, w, eps):
    x32 = x.float()
    y = x32 * torch.pow(x32.pow(2).mean(-1, keepdim=True) + eps, -0.5)
    return (y * w.float()).to(torch.bfloat16)


def _native_embedding(table, ids, eps):
    x32 = F.embedding(ids, table).float()
    return (x32 * torch.pow(x32.pow(2).mean(-1, keepdim=True) + eps, -0.5)).to(torch.bfloat16)


def _native_fused(b, r, w_post, eps_post, w_pre, eps_pre):
    rnew = r + _native_centered(b, w_post, eps_post)
    return rnew, _native_centered(rnew, w_pre, eps_pre)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--hidden", type=int, default=N.HIDDEN)
    parser.add_argument("--vocab", type=int, default=N.VOCAB)
    parser.add_argument("--iters", type=int, default=50)
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dev = "cuda"
    T, H, V = args.rows, args.hidden, args.vocab
    g = torch.Generator(device=dev).manual_seed(0)
    x = (torch.randn(T, H, device=dev, generator=g) * 2).to(torch.bfloat16)
    r = (torch.randn(T, H, device=dev, generator=g) * 3).to(torch.bfloat16)
    w = (torch.randn(H, device=dev, generator=g) * 0.1).to(torch.bfloat16)
    w2 = (torch.randn(H, device=dev, generator=g) * 0.1).to(torch.bfloat16)
    wf = (1 + torch.randn(H, device=dev, generator=g) * 0.1).to(torch.bfloat16)
    table = (torch.randn(V, H, device=dev, generator=g, dtype=torch.bfloat16) * 0.02).contiguous()
    ids = torch.randint(0, V, (T,), device=dev, generator=g)
    eps, eps_post = N.RMS_NORM_EPS, N.POST_NORM_EPS

    rows = []

    def bench(name, strict, native):
        s = _time_ms(strict, iters=args.iters)
        n = _time_ms(native, iters=args.iters)
        rows.append((name, s, n, s / n))

    bench(
        "centered_rmsnorm fwd",
        lambda: N.centered_rmsnorm(x, w, eps=eps),
        lambda: _native_centered(x, w, eps),
    )
    bench(
        "weighted_rmsnorm fwd",
        lambda: N.weighted_rmsnorm(x, wf, eps=eps),
        lambda: _native_weighted(x, wf, eps),
    )
    bench(
        "normalized_embedding fwd",
        lambda: N.normalized_embedding(table, ids, eps=eps),
        lambda: _native_embedding(table, ids, eps),
    )
    bench(
        "fused_post_norm_residual_pre_norm fwd",
        lambda: N.fused_post_norm_residual_pre_norm(x, r, w, eps_post, w2, eps),
        lambda: _native_fused(x, r, w, eps_post, w2, eps),
    )

    # Informational backward timings (forward + backward, same inputs).
    dy = torch.randn_like(x)

    def strict_norm_bwd():
        xx, ww = x.clone().requires_grad_(True), w.clone().requires_grad_(True)
        N.centered_rmsnorm(xx, ww, eps=eps).backward(dy)

    def native_norm_bwd():
        xx, ww = x.clone().requires_grad_(True), w.clone().requires_grad_(True)
        _native_centered(xx, ww, eps).backward(dy)

    bench("centered_rmsnorm fwd+bwd", strict_norm_bwd, native_norm_bwd)

    def strict_fused_bwd():
        xx, rr = x.clone().requires_grad_(True), r.clone().requires_grad_(True)
        ww, ww2 = w.clone().requires_grad_(True), w2.clone().requires_grad_(True)
        rnew, pre = N.fused_post_norm_residual_pre_norm(xx, rr, ww, eps_post, ww2, eps)
        torch.autograd.backward([rnew, pre], [dy, dy])

    def native_fused_bwd():
        xx, rr = x.clone().requires_grad_(True), r.clone().requires_grad_(True)
        ww, ww2 = w.clone().requires_grad_(True), w2.clone().requires_grad_(True)
        rnew, pre = _native_fused(xx, rr, ww, eps_post, ww2, eps)
        torch.autograd.backward([rnew, pre], [dy, dy])

    bench("fused_post_norm_residual_pre_norm fwd+bwd", strict_fused_bwd, native_fused_bwd)

    def strict_embed_bwd():
        tt = table.clone().requires_grad_(True)
        N.normalized_embedding(tt, ids, eps=eps).backward(dy)

    def native_embed_bwd():
        tt = table.clone().requires_grad_(True)
        _native_embedding(tt, ids, eps).backward(dy)

    bench("normalized_embedding fwd+bwd (V=%d)" % V, strict_embed_bwd, native_embed_bwd)

    name = torch.cuda.get_device_name()
    print(f"device={name} T={T} H={H} V={V} iters={args.iters} (median ms)")
    print("| op | strict ms | native ms | strict/native |")
    print("|---|---:|---:|---:|")
    for n, s, t, ratio in rows:
        print(f"| {n} | {s:.3f} | {t:.3f} | {ratio:.2f}x |")


if __name__ == "__main__":
    main()
