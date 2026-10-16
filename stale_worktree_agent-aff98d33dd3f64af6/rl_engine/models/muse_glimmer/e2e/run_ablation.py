# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""CLI for the Muse-Glimmer-30B one-factor mismatch probes (RFC #501 ``museglimmer_ablation``).

Run under a GPU lease, e.g.::

    ~/bin/gpu-lease -n 1 --pool 0-5 -- env PYTHONPATH=$PWD ~/envs/vllm/bin/python \
        -m rl_engine.models.muse_glimmer.e2e.run_ablation \
        --checkpoint ~/models/Muse-Glimmer-30B --out ablation.json [--factors a b c]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rl_engine.models.muse_glimmer.e2e.ablation_hf import rank_factors, run_ablation


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default=str(Path.home() / "models" / "Muse-Glimmer-30B"))
    p.add_argument("--out", required=True, help="JSON file (written after every factor)")
    p.add_argument("--factors", nargs="*", default=None, help="subset of factor names")
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    res = run_ablation(a.checkpoint, None, a.factors, a.out, device=a.device)
    print(f"{'factor':34s} {'max|dlogp|':>11s} {'mean|dlogp|':>12s} {'argmax%':>8s} first")
    for name, mx in rank_factors(res.factors):
        r = res.factors[name]
        print(
            f"{name:34s} {mx:11.4g} {r['mean_abs_dlogp']:12.4g} "
            f"{100 * r['argmax_change_frac']:7.1f}% {r['first_drift_index']}"
        )
    print(json.dumps({"nonfinite": [k for k, v in res.factors.items() if v.get("nonfinite")]}))


if __name__ == "__main__":
    main()
