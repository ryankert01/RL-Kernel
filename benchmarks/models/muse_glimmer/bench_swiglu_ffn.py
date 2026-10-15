# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Benchmark RFC #501 row ``swiglu_ffn`` at the Muse-Glimmer-30B shape against torch native.

Reports (ms, ratio) for the SwiGLU kernel alone (T x F) and for the whole FFN forward (T tokens),
plus the time share of the three GEMMs vs the elementwise kernel, so the report can say what the
fast GEMM will buy. Run under a GPU lease::

    PYTHONPATH=. ~/bin/gpu-lease -n 1 -- ~/envs/vllm/bin/python \\
        benchmarks/models/muse_glimmer/bench_swiglu_ffn.py --tokens 4096
"""

from __future__ import annotations

import argparse
import statistics

import torch
import torch.nn.functional as F_

from rl_engine.models.muse_glimmer.ops.swiglu_ffn import (
    TritonDetGemmAdapter,
    default_gemm,
    swiglu,
    swiglu_ffn,
    trace,
)

HIDDEN, FFN = 6656, 19968


def _time_ms(fn, *, iters: int, warmup: int) -> float:
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dev = torch.device("cuda")
    tokens = args.tokens
    gen = torch.Generator(device=dev).manual_seed(0)

    def randn(shape, scale=1.0):
        return (torch.randn(shape, device=dev, generator=gen) * scale).to(torch.bfloat16)

    x = randn((tokens, HIDDEN))
    w_gate = randn((FFN, HIDDEN), HIDDEN**-0.5)
    w_up = randn((FFN, HIDDEN), HIDDEN**-0.5)
    w_down = randn((HIDDEN, FFN), FFN**-0.5)
    gemm = default_gemm()
    gemm_impl = trace(gemm)["tile_config"]["gemm"].get("impl", type(gemm).__name__)
    print(
        f"device={torch.cuda.get_device_name(dev)} tokens={tokens} H={HIDDEN} F={FFN} "
        f"gemm={gemm_impl}"
    )
    print(f"torch={torch.__version__}")
    timing = dict(iters=args.iters, warmup=args.warmup)
    rows: list[tuple[str, float, float]] = []

    # 1. SwiGLU elementwise kernel, T x F.
    g = randn((tokens, FFN), 3.0)
    u = randn((tokens, FFN))
    with torch.no_grad():
        strict_ms = _time_ms(lambda: swiglu(g, u), **timing)
        native_ms = _time_ms(lambda: F_.silu(g) * u, **timing)
    traffic_gb = 3 * g.numel() * 2 / 1e9  # two bf16 loads + one bf16 store per element
    print(
        f"swiglu  strict {strict_ms:.3f} ms ({traffic_gb/(strict_ms/1e3):.0f} GB/s)  "
        f"native {native_ms:.3f} ms"
    )
    rows.append(("swiglu (T x F) fwd", strict_ms, native_ms))

    # 2. Whole FFN forward.
    gemm_flops = 3 * 2.0 * tokens * HIDDEN * FFN
    with torch.no_grad():
        strict_ms = _time_ms(lambda: swiglu_ffn(x, w_gate, w_up, w_down, gemm=gemm), **timing)
        native_ms = _time_ms(
            lambda: F_.linear(F_.silu(F_.linear(x, w_gate)) * F_.linear(x, w_up), w_down), **timing
        )
    print(
        f"ffn fwd strict {strict_ms:.2f} ms ({gemm_flops/strict_ms/1e9:.0f} TFLOP/s)  "
        f"native {native_ms:.2f} ms ({gemm_flops/native_ms/1e9:.0f} TFLOP/s)"
    )
    rows.append(("swiglu_ffn fwd (T tokens)", strict_ms, native_ms))

    # 3. Share of time: GEMMs vs the elementwise kernel.
    with torch.no_grad():
        gate_ms = _time_ms(lambda: gemm(x, w_gate), **timing)
        up_ms = _time_ms(lambda: gemm(x, w_up), **timing)
        gg = gemm(x, w_gate)
        uu = gemm(x, w_up)
        act_ms = _time_ms(lambda: swiglu(gg, uu), **timing)
        hh = swiglu(gg, uu)
        down_ms = _time_ms(lambda: gemm(hh, w_down), **timing)
    gemms_ms = gate_ms + up_ms + down_ms
    total = gemms_ms + act_ms
    print(
        f"breakdown: gate {gate_ms:.2f} + up {up_ms:.2f} + down {down_ms:.2f} = "
        f"GEMMs {gemms_ms:.2f} ms ({100*gemms_ms/total:.1f}%), "
        f"swiglu {act_ms:.2f} ms ({100*act_ms/total:.1f}%)"
    )
    one_gemm_flops = 2.0 * tokens * HIDDEN * FFN
    print(
        f"per-GEMM throughput: gate {one_gemm_flops/gate_ms/1e9:.0f} TFLOP/s, "
        f"down {one_gemm_flops/down_ms/1e9:.0f} TFLOP/s"
    )

    # 4. Adapter detail: price of the 8-chunk fp32 tree done outside the kernel.
    adapter = gemm if isinstance(gemm, TritonDetGemmAdapter) else TritonDetGemmAdapter()
    with torch.no_grad():
        single_gate = _time_ms(
            lambda: adapter._partial(x, w_gate, 0, HIDDEN).to(torch.bfloat16), **timing
        )
        tree_gate = _time_ms(lambda: adapter(x, w_gate), **timing)
        single_down = _time_ms(
            lambda: adapter._partial(hh, w_down, 0, FFN).to(torch.bfloat16), **timing
        )
        tree_down = _time_ms(lambda: adapter(hh, w_down), **timing)
    print(
        f"adapter: gate GEMM single pass {single_gate:.2f} ms vs 8-chunk tree {tree_gate:.2f} ms; "
        f"down GEMM single pass {single_down:.2f} ms vs 8-chunk tree {tree_down:.2f} ms"
    )

    print()
    print("| case | strict ms | native ms | strict / native |")
    print("|---|---:|---:|---:|")
    for name, strict_ms, native_ms in rows:
        print(f"| {name} | {strict_ms:.3f} | {native_ms:.3f} | {strict_ms/native_ms:.2f}x |")
    print(f"| GEMM share of strict FFN fwd | {gemms_ms:.2f} | - | {100*gemms_ms/total:.1f}% |")
    print(f"| swiglu share of strict FFN fwd | {act_ms:.3f} | - | {100*act_ms/total:.1f}% |")


if __name__ == "__main__":
    main()
