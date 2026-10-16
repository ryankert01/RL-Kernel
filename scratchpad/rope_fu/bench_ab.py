# Interleaved A/B: pre-fix kernel (6a61f6e, int32 offsets) vs current kernel (int64 offsets),
# same process, same inputs, alternating rounds.  Also checks the two give identical bits.
# usage: python bench_ab.py <out.json>
import importlib.util
import json
import os
import statistics
import sys
from pathlib import Path

import torch

from rl_engine.models.muse_glimmer.triton import qknorm_rope as new

spec = importlib.util.spec_from_file_location(
    "old_qknorm_rope", Path(__file__).resolve().parent / "old_triton_qknorm_rope_6a61f6e.py"
)
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)

DEV = torch.device("cuda")
TS = [1, 4, 16, 64, 256, 1024, 2048, 4096, 8192, 16384, 32768]
KW = dict(eps=1e-5, qk_scale=3.87)


def cuda_ms(fn, iters):
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def main():
    rows = []
    for T in TS:
        g = torch.Generator(device=DEV).manual_seed(T)
        q = (torch.randn(T, 32, 128, device=DEV, generator=g) * 2).to(torch.bfloat16)
        k = (torch.randn(T, 2, 128, device=DEV, generator=g) * 2).to(torch.bfloat16)
        dyq = torch.randn(q.shape, device=DEV, generator=g).to(torch.bfloat16)
        dyk = torch.randn(k.shape, device=DEV, generator=g).to(torch.bfloat16)
        from rl_engine.models.muse_glimmer.ops.qknorm_rope import rope_table

        cos, sin = rope_table(torch.arange(T, device=DEV), device=DEV)
        iters = 200 if T <= 2048 else 50
        for local in (True, False):
            c, s = (cos, sin) if local else (None, None)
            fns = {
                "old_fwd": lambda: old.qknorm_rope_forward_triton(q, k, c, s, **KW),
                "new_fwd": lambda: new.qknorm_rope_forward_triton(q, k, c, s, **KW),
                "old_bwd": lambda: old.qknorm_rope_backward_triton(q, k, dyq, dyk, c, s, **KW),
                "new_bwd": lambda: new.qknorm_rope_backward_triton(q, k, dyq, dyk, c, s, **KW),
            }
            a, b = fns["old_fwd"](), fns["new_fwd"]()
            same_fwd = torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
            a, b = fns["old_bwd"](), fns["new_bwd"]()
            same_bwd = torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
            for fn in fns.values():
                for _ in range(5):
                    fn()
            torch.cuda.synchronize()
            samples = {name: [] for name in fns}
            for _ in range(7):  # interleaved rounds
                for name, fn in fns.items():
                    samples[name].append(cuda_ms(fn, iters))
            rec = {"T": T, "local": local, "bitwise_equal_fwd": same_fwd,
                   "bitwise_equal_bwd": same_bwd}
            for name, v in samples.items():
                rec[name + "_ms"] = statistics.median(v)
                rec[name + "_rounds_ms"] = v
            rows.append(rec)
            print(
                f"T={T:6d} {'local ' if local else 'global'} fwd old {rec['old_fwd_ms']*1e3:7.1f} "
                f"new {rec['new_fwd_ms']*1e3:7.1f} us | bwd old {rec['old_bwd_ms']*1e3:7.1f} "
                f"new {rec['new_bwd_ms']*1e3:7.1f} us | equal fwd {same_fwd} bwd {same_bwd}",
                flush=True,
            )
    out = {"description": "interleaved A/B, median of 7 rounds, launcher incl. input checks",
           "old": "6a61f6e triton/qknorm_rope.py (int32 offsets)",
           "new": "current triton/qknorm_rope.py (int64 offsets, bf16-only checks)",
           "gpu": torch.cuda.get_device_name(), "loadavg_start": os.getloadavg(), "series": rows}
    out["loadavg_end"] = os.getloadavg()
    Path(sys.argv[1]).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
