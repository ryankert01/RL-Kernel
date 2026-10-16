import pathlib

SRC = pathlib.Path(
    "/tmp/claude-1000/-home-ubuntu-RL-Kernel--claude-worktrees-swift-oak-hsv7/"
    "80655dc5-1eae-4992-a93b-f6f87e8ab1d8/scratchpad/rope_fu"
)
DST = pathlib.Path("reports/muse_glimmer_poc/scripts/rope")
DST.mkdir(parents=True, exist_ok=True)
HDR = "# SPDX-License-Identifier: Apache-2.0\n# Copyright (c) 2026 RL-Kernel Contributors\n"

# probe_findings.py: unchanged logic
s = (SRC / "probe_findings.py").read_text()
s = s.replace(
    "# Probe the 5 review findings for O4 qknorm_scaled_rope. Writes one JSON object.\n"
    "# Usage: python probe_findings.py <which: f1|f2|f3|f4|f5|xenv> <out.json>\n",
    HDR + "# Wave A probes of the 5 review findings for O4 qknorm_scaled_rope (qknorm_rope:1..5).\n"
    "# Writes one JSON object. Run from the repo root (or with PYTHONPATH=<repo>) on one GPU:\n"
    "#   python probe_findings.py <which: f1|f2|f3|f4|f5|f5b|f5c|xenv> <out.json>\n"
    "# f5/f5b/f5c allocate ~20-30 GiB. Against an old tree: PYTHONPATH=<old tree>.\n",
)
(DST / "probe_findings.py").write_text(s)

s = (SRC / "bench_series.py").read_text()
s = s.replace(
    "# Series benchmark for O4 qknorm_scaled_rope:",
    HDR + "# Series benchmark for O4 qknorm_scaled_rope (eager, CUDA events; launch-bound below\n"
    "# T~4096 - see bench_graph.py for device time without launch overhead):",
)
s = s.replace("# Usage: python bench_series.py <label> <out.json>",
              "# Usage: PYTHONPATH=<repo> python bench_series.py <label> <out.json>")
(DST / "bench_series.py").write_text(s)

s = (SRC / "bench_ab.py").read_text()
s = s.replace(
    "# Interleaved A/B: pre-fix kernel (6a61f6e, int32 offsets) vs current kernel (int64 offsets),\n"
    "# same process, same inputs, alternating rounds.  Also checks the two give identical bits.\n"
    "# usage: python bench_ab.py <out.json>\n",
    HDR + "# Interleaved A/B: pre-fix kernel (6a61f6e, int32 offsets) vs current kernel (int64\n"
    "# offsets), same process, same inputs, alternating rounds (eager launcher calls, CUDA events).\n"
    "# Also checks the two give identical bits.\n"
    "# usage: git show 6a61f6e:rl_engine/models/muse_glimmer/triton/qknorm_rope.py > old.py\n"
    "#        PYTHONPATH=<repo> python bench_ab.py <out.json> old.py\n",
)
s = s.replace(
    '''spec = importlib.util.spec_from_file_location(
    "old_qknorm_rope", Path(__file__).resolve().parent / "old_triton_qknorm_rope_6a61f6e.py"
)
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
''',
    '''spec = importlib.util.spec_from_file_location("old_qknorm_rope", Path(sys.argv[2]).resolve())
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
''',
)
assert "sys.argv[2]" in s
(DST / "bench_ab.py").write_text(s)

s = (SRC / "overhead_probe.py").read_text()
s = s.replace(
    "# Where does the ~300 us autograd fwd+bwd overhead come from?  Wall-clock (CPU) timing.\n",
    HDR + "# Where does the ~300 us autograd fwd+bwd overhead come from?  Wall-clock (CPU) timing.\n"
    "# usage: PYTHONPATH=<repo> python overhead_probe.py <out.json>\n",
)
(DST / "overhead_probe.py").write_text(s)

s = (SRC / "build_json.py").read_text()
s = s.replace(
    '''# Assemble reports/muse_glimmer_poc/data/kernels_rope.json from the probe/bench/pytest outputs.
# usage: python build_json.py <commit_after> <out.json>
import json
import re
import sys
from pathlib import Path

S = Path(__file__).resolve().parent
B, A = S / "before", S / "final"
commit_after, out = sys.argv[1], sys.argv[2]
SP = str(S)
W = "/home/ubuntu/RL-Kernel/.claude/worktrees/wf_7b064961-36f-6"
''',
    HDR + '''# Wave A: assemble reports/muse_glimmer_poc/data/kernels_rope.json from the probe/bench/pytest
# outputs of run_rope_measurements.sh (a "before" dir measured at 6a61f6e, an "after" dir at the
# fixed commit). Wave B fields are added on top by update_json_wave_b.py.
# usage: python build_json.py <before_dir> <after_dir> <commit_after> <out.json>
import json
import re
import sys
from pathlib import Path

B, A = Path(sys.argv[1]), Path(sys.argv[2])
commit_after, out = sys.argv[3], sys.argv[4]
SP = "reports/muse_glimmer_poc/scripts/rope"
''',
)
assert "sys.argv[4]" in s
(DST / "build_json.py").write_text(s)
print("ok")
