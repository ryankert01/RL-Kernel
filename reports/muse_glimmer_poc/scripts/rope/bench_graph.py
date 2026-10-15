# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""GPU time of the O4 kernels without launch overhead: CUDA-graph replay.

For each T (local and global layers, forward and backward kernels) the launcher is captured
``LAUNCHES`` times into one CUDA graph; the graph is replayed ``REPLAYS`` times between two CUDA
events, so ``gpu_us_per_launch`` = elapsed / (REPLAYS * LAUNCHES) is device time only (no host
launch gaps). The same calls are also timed eagerly (events around ``iters`` back-to-back
launcher calls) to show where the host launch floor dominates. With ``--old-kernel <path>`` (the
6a61f6e ``triton/qknorm_rope.py``, int32 offsets) both kernels are timed interleaved in one
process and their output bits compared. Effective bandwidth is reported for T >= 8192 only
(below that the kernel is a few microseconds and the number is not meaningful).

usage: PYTHONPATH=<repo> python bench_graph.py <out.json> [--old-kernel <path>]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import statistics

import torch

from rl_engine.models.muse_glimmer.ops.qknorm_rope import rope_table
from rl_engine.models.muse_glimmer.triton import qknorm_rope as new

DEV = torch.device("cuda")
TS = [1, 16, 64, 256, 1024, 2048, 4096, 8192, 16384, 32768]
LAUNCHES = 20
REPLAYS = 10
ROUNDS = 5
KW = dict(eps=1e-5, qk_scale=3.87)


def load_old(path: str):
    spec = importlib.util.spec_from_file_location("qknorm_rope_old_kernel", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def graph_us(fn) -> float:
    """us per launch from graph replay (LAUNCHES launches captured, REPLAYS replays)."""
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()  # warm up / compile outside capture
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(LAUNCHES):
            fn()
    g.replay()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(REPLAYS):
        g.replay()
    e.record()
    torch.cuda.synchronize()
    us = s.elapsed_time(e) * 1e3 / (REPLAYS * LAUNCHES)
    del g
    return us


def eager_us(fn, iters: int) -> float:
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) * 1e3 / iters


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--old-kernel", default=None)
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    kernels = {"new": new}
    if args.old_kernel:
        kernels["old"] = load_old(args.old_kernel)
    rows = []
    for T in TS:
        g = torch.Generator(device=DEV).manual_seed(T)
        q = (torch.randn(T, 32, 128, device=DEV, generator=g) * 2).to(torch.bfloat16)
        k = (torch.randn(T, 2, 128, device=DEV, generator=g) * 2).to(torch.bfloat16)
        dyq = torch.randn(q.shape, device=DEV, generator=g).to(torch.bfloat16)
        dyk = torch.randn(k.shape, device=DEV, generator=g).to(torch.bfloat16)
        cos, sin = rope_table(torch.arange(T, device=DEV), device=DEV)
        for local in (True, False):
            c, s = (cos, sin) if local else (None, None)
            fns = {}
            for name, mod in kernels.items():
                fns[f"{name}_fwd"] = (lambda m=mod: m.qknorm_rope_forward_triton(q, k, c, s, **KW))
                fns[f"{name}_bwd"] = (
                    lambda m=mod: m.qknorm_rope_backward_triton(q, k, dyq, dyk, c, s, **KW))
            rec = {"T": T, "local": local}
            if "old" in kernels:
                a, b = fns["old_fwd"](), fns["new_fwd"]()
                rec["bitwise_equal_fwd"] = bool(torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]))
                a, b = fns["old_bwd"](), fns["new_bwd"]()
                rec["bitwise_equal_bwd"] = bool(torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]))
            samples = {n: [] for n in fns}
            for _ in range(ROUNDS):  # interleaved rounds
                for n, fn in fns.items():
                    samples[n].append(graph_us(fn))
            for n, v in samples.items():
                rec[f"{n}_gpu_us"] = statistics.median(v)
                rec[f"{n}_gpu_us_rounds"] = v
            iters = 200 if T <= 2048 else 50
            for n, fn in fns.items():
                rec[f"{n}_eager_us"] = eager_us(fn, iters)
            fwd_bytes = 2 * 2 * (q.numel() + k.numel()) + (2 * 4 * cos.numel() if local else 0)
            rec["fwd_bytes"] = fwd_bytes
            rec["new_fwd_GBps"] = (fwd_bytes / (rec["new_fwd_gpu_us"] * 1e-6) / 1e9
                                   if T >= 8192 else None)
            rows.append(rec)
            msg = " ".join(f"{n} {rec[f'{n}_gpu_us']:8.2f}" for n in fns)
            print(f"T={T:6d} {'local ' if local else 'global'} gpu us/launch: {msg}", flush=True)
    out = {
        "description": (
            f"CUDA-graph replay: {LAUNCHES} launcher calls captured per graph, {REPLAYS} "
            f"replays timed with CUDA events, median of {ROUNDS} interleaved rounds -> device "
            "time per launch without host launch overhead; *_eager_us = events around "
            "back-to-back eager launcher calls (includes the host launch floor)"),
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "loadavg": os.getloadavg(),
        "old_kernel": args.old_kernel,
        "T": TS,
        "series": rows,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
