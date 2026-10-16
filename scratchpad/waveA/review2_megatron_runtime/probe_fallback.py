"""Probe: does the strict op resolution fail closed when a strict kernel is unavailable?"""
import os
import sys

import torch  # noqa: F401  (import torch before TE)

from rl_engine.models.muse_glimmer.runtime import strict_kernels as sk

mode = sys.argv[1]
if mode == "env":
    os.environ["MUSE_STRICT_ATTN"] = "off"
elif mode == "missing_norms":
    real = sk._optional_module

    def fake(path):
        if path.endswith("ops.norms"):
            return None  # simulate an import error in the sibling norms module
        return real(path)

    sk._optional_module = fake
ops = sk.resolve_ops()
prov = ops.provenance()
print(mode, "all_strict =", prov["all_strict"])
for k, v in prov.items():
    if k != "all_strict":
        print("   ", k, "->", v)
