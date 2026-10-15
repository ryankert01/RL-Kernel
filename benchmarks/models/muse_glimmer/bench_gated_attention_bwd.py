# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Strict gated-attention backward vs torch SDPA backward at the Muse-Glimmer training shape.

Shape: one sequence of 4096 tokens, 32 q heads x 2 kv heads x 128, bf16. The strict path times
preprocess (gate VJP + do + delta) + dK/dV partials + dQ + tree combine; the native baseline
times ``F.scaled_dot_product_attention(..., is_causal=True, enable_gqa=True)`` backward (no
gate, no sliding window - SDPA has no window, so the local row is compared to the same causal
baseline). Run:

    PYTHONPATH=. ~/bin/gpu-lease -n 1 -- ~/envs/vllm/bin/python \
        benchmarks/models/muse_glimmer/bench_gated_attention_bwd.py
"""

from __future__ import annotations

import argparse
import time

import torch
import torch.nn.functional as F

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def _timeit(fn, iters: int, warmup: int = 3) -> float:
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
    parser.add_argument("--length", type=int, default=4096)
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()

    from rl_engine.models.muse_glimmer.ops import attention_golden_bwd as gold
    from rl_engine.models.muse_glimmer.triton import gated_attention_bwd as bwd

    dev = torch.device("cuda")
    length = args.length
    g = torch.Generator(device="cpu").manual_seed(0)
    q = torch.randn(length, 32, 128, generator=g)
    q = (q * torch.rsqrt(q.pow(2).mean(-1, keepdim=True) + 1e-5) * 3.87).to(dev, torch.bfloat16)
    k = torch.randn(length, 2, 128, generator=g)
    k = (k * torch.rsqrt(k.pow(2).mean(-1, keepdim=True) + 1e-5)).to(dev, torch.bfloat16)
    v = torch.randn(length, 2, 128, generator=g).to(dev, torch.bfloat16)
    gate = torch.randn(length, 4096, generator=g).to(dev, torch.bfloat16)
    dy = torch.randn(length, 4096, generator=g).to(dev, torch.bfloat16)
    cu = torch.tensor([0, length], dtype=torch.int32, device=dev)
    pos = torch.arange(length, dtype=torch.int32, device=dev)

    rows = []
    for name, window in (("global_gated_attention", None), ("local_gated_attention_train", 2048)):
        t0 = time.time()
        _y, o, lse = gold.golden_forward_fixed_order(q, k, v, gate, cu, cu, pos, window=window)
        torch.cuda.synchronize()
        fwd_s = time.time() - t0

        def strict():
            return bwd.strict_gated_attention_bwd(
                q, k, v, gate, o, lse, dy, cu, cu, pos, window=window
            )

        def strict_core():
            dgate, do, delta = bwd.gated_attention_bwd_preprocess(dy, o, gate)
            return bwd.strict_gated_attention_bwd_partials(
                q, k, v, do, lse, delta, cu, cu, pos, window=window
            )

        ms_strict = _timeit(strict, args.iters)
        ms_core = _timeit(strict_core, args.iters)
        rows.append((name, fwd_s, ms_strict, ms_core))

    # Native baseline: SDPA causal GQA backward (dq, dk, dv), bf16.
    qs = q.permute(1, 0, 2).unsqueeze(0).contiguous().requires_grad_(True)  # (1, 32, L, 128)
    ks = k.permute(1, 0, 2).unsqueeze(0).contiguous().requires_grad_(True)  # (1, 2, L, 128)
    vs = v.permute(1, 0, 2).unsqueeze(0).contiguous().requires_grad_(True)
    out = F.scaled_dot_product_attention(qs, ks, vs, is_causal=True, enable_gqa=True)
    dout = dy.view(length, 32, 128).permute(1, 0, 2).unsqueeze(0).contiguous()

    def native():
        return torch.autograd.grad(out, (qs, ks, vs), grad_outputs=dout, retain_graph=True)

    ms_native = _timeit(native, args.iters)

    print(f"\nMuse-Glimmer gated attention backward, Tq=Tk={length}, 32x2 heads, D=128, bf16, A100")
    print("| row | strict bwd total (ms) | strict core w/o tree (ms) | SDPA causal bwd (ms) |"
          " ratio strict/native | golden fwd (s, fallback) |")
    print("|---|---|---|---|---|---|")
    for name, fwd_s, ms_strict, ms_core in rows:
        print(
            f"| {name} | {ms_strict:.2f} | {ms_core:.2f} | {ms_native:.2f} | "
            f"{ms_strict / ms_native:.2f}x | {fwd_s:.2f} |"
        )
    print(f"\ntrace: {bwd.trace()['tile_config']}")


if __name__ == "__main__":
    main()
