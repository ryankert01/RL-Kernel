# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Qualify and time ``fast_det_gemm`` on the Muse-Glimmer-30B GEMM shapes (RFC #501 O3).

For every per-layer GEMM and the padded lm_head at M = 4096 (training / prefill) and M = 16
(decode): strict kernel vs cuBLAS bf16 (``torch.matmul``, TF32 off) vs the repo's existing
``TritonDetGemmOp``; plus the G1 error table vs an fp64 reference and a full autograd
forward+backward timing.  No autotuning happens here; the pinned tile constants are used.

    PYTHONPATH=. ~/bin/gpu-lease -n 1 -- ~/envs/vllm/bin/python \
        benchmarks/models/muse_glimmer/bench_det_gemm.py [--skip-legacy] [--out report.md]
"""

from __future__ import annotations

import argparse
import time

import torch

from rl_engine.models.muse_glimmer.ops import det_gemm as dg
from rl_engine.models.muse_glimmer.triton import det_gemm_tree as kt

try:
    from rl_engine.backends.shared.triton.gemm.det_gemm import _triton_tree_gemm as legacy_tree

    _HAS_LEGACY = True
except ImportError:  # pragma: no cover - benchmark only
    _HAS_LEGACY = False

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

DEV = "cuda"
SHAPES = [
    ("q_proj", 4096, 6656),
    ("k_proj", 256, 6656),
    ("v_proj", 256, 6656),
    ("attn_gate", 4096, 6656),
    ("o_proj", 6656, 4096),
    ("mlp_gate", 19968, 6656),
    ("up_proj", 19968, 6656),
    ("down_proj", 6656, 19968),
    ("lm_head", 202752, 6656),
]
LEGACY_TIME_CAP_S = 4.0


def _timeit(fn, iters: int = 20, warmup: int = 3) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def _legacy_ms(a: torch.Tensor, w_kn: torch.Tensor) -> float:
    """Time the existing TritonDetGemmOp; skip shapes where one call exceeds the cap."""

    if not _HAS_LEGACY:
        return float("nan")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    legacy_tree(a, w_kn)
    torch.cuda.synchronize()
    once = time.perf_counter() - t0
    if once > LEGACY_TIME_CAP_S:
        return once * 1e3
    return _timeit(lambda: legacy_tree(a, w_kn), iters=5, warmup=1)


def qualify(rows: list[str]) -> None:
    rows.append("## G1 error vs fp64 reference (M=256, a~N(0,1), w~N(0,0.02^2))")
    rows.append("")
    rows.append(
        "| shape | N | K | bf16 max_abs | bf16 max_abs / max abs ref | fp32 max_abs "
        "| fp32 max_rel |"
    )
    rows.append("|---|---|---|---|---|---|---|")
    for name, n_size, k_size in SHAPES:
        torch.manual_seed(0)
        a = torch.randn(256, k_size, device=DEV, dtype=torch.bfloat16)
        w = (torch.randn(n_size, k_size, device=DEV) * 0.02).to(torch.bfloat16)
        ref = dg.forward_fp32(a, w).double()
        y16 = dg.fast_det_gemm(a, w).double()
        y32 = dg.fast_det_gemm(a, w, out_dtype=torch.float32).double()
        scale = ref.abs().max().item()
        e16 = (y16 - ref).abs().max().item()
        e32 = (y32 - ref).abs()
        rel32 = (e32 / ref.abs().clamp_min(1e-3 * scale)).max().item()
        rows.append(
            f"| {name} | {n_size} | {k_size} | {e16:.3e} | {e16 / scale:.3e} | "
            f"{e32.max().item():.3e} | {rel32:.3e} |"
        )
        del ref, y16, y32
    rows.append("")


def bench(rows: list[str], skip_legacy: bool) -> None:
    for m_size in (4096, 16):
        rows.append(f"## Throughput at M={m_size} (bf16 out)")
        rows.append("")
        rows.append(
            "| shape | N | K | strict ms | strict TFLOP/s | cuBLAS ms | cuBLAS TFLOP/s | "
            "strict/cuBLAS | legacy TritonDetGemmOp ms | legacy/strict | class | regs/spills |"
        )
        rows.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for name, n_size, k_size in SHAPES:
            torch.manual_seed(0)
            a = torch.randn(m_size, k_size, device=DEV, dtype=torch.bfloat16)
            w = (torch.randn(n_size, k_size, device=DEV) * 0.02).to(torch.bfloat16)
            w_kn = w.t().contiguous()
            flop = 2.0 * m_size * n_size * k_size
            t_strict = _timeit(lambda: dg.fast_det_gemm(a, w))
            launch = kt.last_launch()
            t_cublas = _timeit(lambda: torch.matmul(a, w.t()))
            if skip_legacy or (name == "lm_head" and m_size > 16):
                t_legacy = float("nan")
            else:
                t_legacy = _legacy_ms(a, w_kn)
            legacy_ratio = f"{t_legacy / t_strict:.1f}x" if t_legacy == t_legacy else "n/a"
            legacy_str = f"{t_legacy:.3f}" if t_legacy == t_legacy else "skipped"
            rows.append(
                f"| {name} | {n_size} | {k_size} | {t_strict:.3f} | {flop / t_strict / 1e9:.1f} | "
                f"{t_cublas:.3f} | {flop / t_cublas / 1e9:.1f} | {t_strict / t_cublas:.2f}x | "
                f"{legacy_str} | {legacy_ratio} | {launch['shape_class']} | "
                f"{launch['n_regs']}/{launch['n_spills']} |"
            )
            del a, w, w_kn
            torch.cuda.empty_cache()
        rows.append("")


def bench_backward(rows: list[str]) -> None:
    rows.append("## Autograd forward+backward at M=4096 (dA + dW), strict vs torch bf16 autograd")
    rows.append("")
    rows.append("| shape | N | K | strict fwd+bwd ms | torch fwd+bwd ms | ratio | dA ms | dW ms |")
    rows.append("|---|---|---|---|---|---|---|---|")
    backward_shapes = [
        ("o_proj", 6656, 4096),
        ("mlp_gate", 19968, 6656),
        ("down_proj", 6656, 19968),
    ]
    for name, n_size, k_size in backward_shapes:
        m_size = 4096
        torch.manual_seed(0)
        a = torch.randn(m_size, k_size, device=DEV, dtype=torch.bfloat16, requires_grad=True)
        w = (torch.randn(n_size, k_size, device=DEV) * 0.02).to(torch.bfloat16)
        w.requires_grad_(True)
        g = torch.randn(m_size, n_size, device=DEV, dtype=torch.bfloat16)

        def strict():
            dg.fast_det_gemm(a, w).backward(g)

        def native():
            torch.nn.functional.linear(a, w).backward(g)

        t_strict = _timeit(strict, iters=5, warmup=2)
        t_native = _timeit(native, iters=5, warmup=2)
        t_da = _timeit(lambda: kt.tree_gemm(g, w), iters=5, warmup=2)
        t_dw = _timeit(lambda: kt.tree_gemm(g.t(), a.detach(), allow_pad=True), iters=5, warmup=2)
        rows.append(
            f"| {name} | {n_size} | {k_size} | {t_strict:.3f} | {t_native:.3f} | "
            f"{t_strict / t_native:.2f}x | {t_da:.3f} | {t_dw:.3f} |"
        )
        del a, w, g
        torch.cuda.empty_cache()
    rows.append("")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=None)
    parser.add_argument("--skip-legacy", action="store_true")
    parser.add_argument("--skip-qualify", action="store_true")
    args = parser.parse_args()
    import triton

    rows = [
        f"# fast_det_gemm on {torch.cuda.get_device_name()} "
        f"(sm{''.join(map(str, torch.cuda.get_device_capability()))}, torch {torch.__version__}, "
        f"triton {triton.__version__})",
        "",
        f"Pinned tile constants: {dg.trace()['tile_config']}",
        "",
    ]
    if not args.skip_qualify:
        qualify(rows)
    bench(rows, args.skip_legacy)
    bench_backward(rows)
    text = "\n".join(rows)
    print(text)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")


if __name__ == "__main__":
    main()
