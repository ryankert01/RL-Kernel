# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Strict ``lm_head_softcap_logprob`` vs the native torch path at the Muse-Glimmer-30B shape.

Native path: bf16 ``torch.matmul`` -> fp32 -> ``T * tanh(z * m / T)`` -> ``log_softmax`` ->
gather (materialises the full fp32 logits).  Strict path: 8 vocabulary tiles through
``det_gemm`` (fp32 out), Triton tile statistics, canonical tree.

Usage (GPU lease required)::

    ~/bin/gpu-lease -n 1 -- ~/envs/vllm/bin/python \
        benchmarks/models/muse_glimmer/bench_lm_head_softcap_logprob.py --tokens 4096 --backward
"""

from __future__ import annotations

import argparse
import time

import torch

from rl_engine.models.muse_glimmer.ops import lm_head_softcap_logprob as op

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def _time(fn, *, warmup: int, iters: int) -> tuple[float, float]:
    """(ms per call, peak extra MiB)."""

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    ms = (time.perf_counter() - start) * 1000.0 / iters
    peak = (torch.cuda.max_memory_allocated() - base) / 2**20
    return ms, peak


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--backward", action="store_true", help="also time forward+backward")
    parser.add_argument(
        "--gemm", choices=("auto", "legacy"), default="auto", help="strict GEMM backend"
    )
    args = parser.parse_args()

    device = torch.device("cuda")
    g = torch.Generator(device=device).manual_seed(0)
    tokens, hidden, vocab = args.tokens, op.HIDDEN_SIZE, op.VOCAB_SIZE
    w = (torch.randn(vocab, hidden, device=device, generator=g) * 0.02).to(torch.bfloat16)
    w_padded = op.pad_lm_head_weight(w)
    h = torch.randn(tokens, hidden, device=device, generator=g).to(torch.bfloat16)
    targets = torch.randint(0, vocab, (tokens,), device=device, generator=g)
    dlogp = torch.randn(tokens, device=device, generator=g)
    gemm = op.legacy_det_gemm_fp32 if args.gemm == "legacy" else None
    mult, cap = op.OUTPUT_MULTIPLIER, op.FINAL_LOGIT_SOFTCAP

    def native_forward(hh=h, ww=w):
        z = (hh @ ww.t()).float()
        zhat = op.softcap_golden(z, multiplier=mult, softcap=cap)
        return torch.log_softmax(zhat, dim=-1).gather(1, targets[:, None]).squeeze(1)

    def strict_forward(hh=h, ww=w_padded):
        return op.lm_head_softcap_logprob(hh, ww, targets, gemm=gemm)

    def native_fwd_bwd():
        hh = h.clone().requires_grad_(True)
        ww = w.clone().requires_grad_(True)
        native_forward(hh, ww).backward(dlogp)

    def strict_fwd_bwd():
        hh = h.clone().requires_grad_(True)
        ww = w_padded.clone().requires_grad_(True)
        strict_forward(hh, ww).backward(dlogp)

    with torch.no_grad():
        native_logp = native_forward()
        strict_logp = strict_forward()
    max_err = (native_logp - strict_logp).abs().max().item()

    rows = []
    with torch.no_grad():
        nat_ms, nat_mem = _time(native_forward, warmup=args.warmup, iters=args.iters)
        st_ms, st_mem = _time(strict_forward, warmup=args.warmup, iters=args.iters)
    rows.append(("forward", nat_ms, nat_mem, st_ms, st_mem))
    if args.backward:
        nat_ms, nat_mem = _time(native_fwd_bwd, warmup=args.warmup, iters=args.iters)
        st_ms, st_mem = _time(strict_fwd_bwd, warmup=args.warmup, iters=args.iters)
        rows.append(("forward+backward", nat_ms, nat_mem, st_ms, st_mem))

    flops = 2.0 * tokens * hidden * op.PADDED_VOCAB_SIZE
    print(f"device: {torch.cuda.get_device_name(device)}")
    print(f"shape: T={tokens} H={hidden} V={vocab} (padded {op.PADDED_VOCAB_SIZE}, 8 tiles)")
    print(f"strict gemm backend: {op.gemm_backend_name(gemm)}")
    print(f"max |native logp - strict logp| = {max_err:.3e}")
    print()
    print("| pass | native ms | native peak MiB | strict ms | strict peak MiB | strict/native |")
    print("| --- | ---: | ---: | ---: | ---: | ---: |")
    for name, n_ms, n_mem, s_ms, s_mem in rows:
        ratio = s_ms / n_ms
        print(f"| {name} | {n_ms:.1f} | {n_mem:.0f} | {s_ms:.1f} | {s_mem:.0f} | {ratio:.1f}x |")
    fwd = rows[0]
    print()
    print(
        f"forward GEMM throughput: native {flops / fwd[1] / 1e9:.1f} TFLOP/s, "
        f"strict {flops / fwd[3] / 1e9:.1f} TFLOP/s"
    )


if __name__ == "__main__":
    main()
