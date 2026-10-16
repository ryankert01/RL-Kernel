"""Start-up overlap measurement on the PRE-FIX bridge (tree of commit 87a4cd2: old bridge and
trainer server, new mocks with build_delay_s) using the bench code of 33a86ad.

The client runs from the extracted tree (sys.path[0]) and chdirs into it, so the trainer
server's ``python -m`` (cwd first on sys.path) also loads the pre-fix code.
"""

import os
import sys
from pathlib import Path

here = Path(__file__).resolve().parent
tree = here / "tree_87a4cd2"
sys.path.insert(0, str(tree))
os.environ["PYTHONPATH"] = str(tree)
os.chdir(tree)

from rl_engine.models.muse_glimmer.e2e import bench_bridge, bridge  # noqa: E402

assert Path(bridge.__file__).resolve().is_relative_to(tree), bridge.__file__
assert not hasattr(bridge, "STATEFUL_OPS"), "expected the pre-fix bridge"
sys.exit(
    bench_bridge.main(
        [
            "--startup-overlap",
            "--overlap-label",
            "before_fix",
            "--out",
            str(here / "overlap_before.json"),
            "--work-dir",
            str(here / "work_overlap_before"),
            "--command",
            "[pre-fix tree = git archive 87a4cd2 (old bridge/trainer_server, mocks with "
            "build_delay_s) + bench_bridge.py of 33a86ad; CPU, no lease] "
            "~/envs/vllm/bin/python <scratch>/run_overlap_before.py",
        ]
    )
)
