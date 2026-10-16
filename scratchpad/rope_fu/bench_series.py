# Series benchmark for O4 qknorm_scaled_rope: strict Triton fwd / bwd-kernel / fwd+bwd(autograd)
# vs the torch-native HF-style composition, per token count T (also covers decode batch sizes).
# Usage: python bench_series.py <label> <out.json>
from __future__ import annotations

import json
import statistics
import sys

import torch

from benchmarks.models.muse_glimmer.bench_qknorm_rope import hf_native
from rl_engine.models.muse_glimmer.ops import qknorm_rope as op
from rl_engine.models.muse_glimmer.triton.qknorm_rope import (
    qknorm_rope_backward_triton,
    qknorm_rope_forward_triton,
)

HQ, HK, D = 32, 2, 128
TS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]


def time_ms(fn, iters, reps=5, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(reps):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(iters):
            fn()
        e.record()
        torch.cuda.synchronize()
        out.append(s.elapsed_time(e) / iters)
    return statistics.median(out), out


def main():
    label, path = sys.argv[1], sys.argv[2]
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = torch.device("cuda")
    rows = []
    for T in TS:
        g = torch.Generator(device=dev).manual_seed(T)
        q = (torch.randn(T, HQ, D, device=dev, generator=g) * 2).to(torch.bfloat16)
        k = (torch.randn(T, HK, D, device=dev, generator=g) * 2).to(torch.bfloat16)
        dyq = torch.randn(q.shape, device=dev, generator=g).to(torch.bfloat16)
        dyk = torch.randn(k.shape, device=dev, generator=g).to(torch.bfloat16)
        pos = torch.arange(T, device=dev)
        cos, sin = op.rope_table(pos, device=dev)
        cb, sb = cos.to(torch.bfloat16), sin.to(torch.bfloat16)
        iters = 200 if T <= 1024 else 50
        for local in (True, False):
            c, s = (cos, sin) if local else (None, None)
            kw = dict(eps=op.RMS_NORM_EPS, qk_scale=op.QK_SCALE)
            fwd_k = lambda: qknorm_rope_forward_triton(q, k, c, s, **kw)  # noqa: E731
            bwd_k = lambda: qknorm_rope_backward_triton(q, k, dyq, dyk, c, s, **kw)  # noqa: E731
            fwd_api = lambda: op.qknorm_scaled_rope(q, k, c, s)  # noqa: E731

            def fwd_bwd():
                qg, kg = q.detach().requires_grad_(True), k.detach().requires_grad_(True)
                torch.autograd.grad(op.qknorm_scaled_rope(qg, kg, c, s), (qg, kg), (dyq, dyk))

            rec = {"T": T, "local": local}
            for name, fn in (
                ("strict_fwd_kernel_ms", fwd_k),
                ("strict_bwd_kernel_ms", bwd_k),
                ("strict_fwd_api_ms", fwd_api),
                ("strict_fwd_bwd_autograd_ms", fwd_bwd),
            ):
                med, all_ = time_ms(fn, iters)
                rec[name] = med
                rec[name.replace("_ms", "_reps_ms")] = all_
            if local:
                med, all_ = time_ms(lambda: hf_native(q, k, cb, sb), iters)
                rec["native_hf_fwd_ms"] = med
                rec["native_hf_fwd_reps_ms"] = all_
            bytes_fwd = 2 * (q.numel() + k.numel()) * 2 + ((2 * cos.numel() * 4) if local else 0)
            rec["fwd_bytes"] = bytes_fwd
            rec["fwd_kernel_GBps"] = bytes_fwd / rec["strict_fwd_kernel_ms"] / 1e6
            rows.append(rec)
            print(
                f"T={T:6d} {'local ' if local else 'global'} fwd {rec['strict_fwd_kernel_ms']*1e3:8.1f} us"
                f"  bwd {rec['strict_bwd_kernel_ms']*1e3:8.1f} us  api {rec['strict_fwd_api_ms']*1e3:8.1f} us"
                f"  fwd+bwd {rec['strict_fwd_bwd_autograd_ms']*1e3:8.1f} us"
                + (f"  native {rec['native_hf_fwd_ms']*1e3:8.1f} us" if local else "")
                + f"  {rec['fwd_kernel_GBps']:7.0f} GB/s",
                flush=True,
            )
    out = {"label": label, "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
           "series": rows}
    with open(path, "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
