import sys
import traceback
from pathlib import Path

import torch
import torch.cuda

calls = []


def _no_init(*args, **kwargs):
    calls.append("".join(traceback.format_stack(limit=25)))
    raise RuntimeError("diag: CUDA init blocked")


torch.cuda._lazy_init = _no_init

from rl_engine.models.muse_glimmer.e2e import bench_bridge as bb  # noqa: E402

work = Path(sys.argv[1])
for name, fn in (
    ("surrogate", lambda: bb.surrogate_exactness("cpu")),
    ("equiv", lambda: bb.bridge_equivalence(work / "eq", steps=1)),
    ("transport", lambda: bb.transport(work, repeats=1)),
    ("faults", lambda: bb.failure_handling(work / "faults")),
):
    before = len(calls)
    try:
        fn()
    except Exception as exc:  # noqa: BLE001
        print(name, "raised", type(exc).__name__, str(exc)[:200])
    print(name, "init attempts", len(calls) - before, "is_initialized", torch.cuda.is_initialized())
for c in calls[:2]:
    print(c)
