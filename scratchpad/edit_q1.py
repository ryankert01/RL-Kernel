import pathlib, re
p = pathlib.Path("rl_engine/models/muse_glimmer/ops/qknorm_rope.py")
s = p.read_text()

# ---- header docstring: public interface
s = s.replace('''    ``rope_table(positions, *, theta, dim, device) -> (cos, sin)`` fp32 ``(T, dim)``; THE table
    definition; CUDA devices only (a CPU build differs in ~19% of the elements)
    ``canonical_rope_table(device)`` the cached ``(cos, sin)`` for ``arange(131072)`` (one
    object per device; sha256 pinned in ``CANONICAL_ROPE_TABLE_SHA256``)
    ``rope_rows(positions)`` rows gathered from the canonical table (bitwise == ``rope_table``)
    ``qknorm_scaled_rope(q, k, cos, sin, *, eps, qk_scale, local=None)`` strict Triton op,
    bf16 -> bf16 (other dtypes raise), differentiable (deterministic row-local Triton backward);
    ``local=True/False`` makes the layer kind explicit and is checked against ``cos``/``sin``
''', '''    ``rope_table(positions, *, theta, dim, device) -> (cos, sin)`` fp32 ``(T, dim)``: rows of
    THE table (copies gathered from ``canonical_rope_table``); CUDA devices only (a CPU build
    differs in ~19% of the elements); positions must lie in ``[0, 131072)``
    ``canonical_rope_table(device, verify=False)`` the cached ``(cos, sin)`` for
    ``arange(131072)`` (one object per device, built once by the fixed fp32 torch-CUDA formula;
    sha256 pinned in ``CANONICAL_ROPE_TABLE_SHA256``; in-place changes are detected; building it
    during CUDA-graph capture raises - build it at init)
    ``rope_rows(positions)`` rows gathered from the canonical table (same as ``rope_table``)
    ``qknorm_scaled_rope(q, k, cos, sin, *, eps, qk_scale, local=None)`` strict Triton op,
    bf16 -> bf16 (other dtypes raise), differentiable (deterministic row-local Triton backward);
    ``local=True/False`` makes the layer kind explicit and is checked against ``cos``/``sin``;
    ``local=None`` (legacy, implicit) is counted, warns once, and raises when
    ``MUSEGLIMMER_O4_REQUIRE_LAYER_KIND=1`` or ``require_explicit_layer_kind(True)``
''')
s = s.replace('''    ``trace()`` provenance dict (contract R4)
"""''', '''    ``trace()`` provenance dict of the op (contract R4; static)
    ``runtime_provenance()`` what THIS process did: digest of the table it built (per device),
    explicit / implicit layer-kind call counts
"""''')

s = s.replace('''import hashlib
import threading
from typing import Any
''', '''import hashlib
import os
import threading
import warnings
from typing import Any
''')

s = s.replace('''CANONICAL_ROPE_TABLE_SHA256 = "088dbb8c93a55ce79d189181c46ac06fbdb61dab6c3914218bbe40ed3ea56d9c"
''', '''CANONICAL_ROPE_TABLE_SHA256 = "088dbb8c93a55ce79d189181c46ac06fbdb61dab6c3914218bbe40ed3ea56d9c"
# Set to 1 (e.g. by the E2E harness) to make the layer kind mandatory: local=None raises.
LAYER_KIND_ENV = "MUSEGLIMMER_O4_REQUIRE_LAYER_KIND"
''')

s = s.replace('''    "CANONICAL_ROPE_TABLE_SHA256",
    "GLOBAL_LAYER_PERIOD",
    "HEAD_DIM",
''', '''    "CANONICAL_ROPE_TABLE_SHA256",
    "GLOBAL_LAYER_PERIOD",
    "HEAD_DIM",
    "LAYER_KIND_ENV",
''')
s = s.replace('''    "qknorm_scaled_rope_fp64",
    "rope_rows",
    "rope_table",
    "rope_table_fp64",
    "rope_table_sha256",
    "trace",
]''', '''    "qknorm_scaled_rope_fp64",
    "require_explicit_layer_kind",
    "rope_rows",
    "rope_table",
    "rope_table_fp64",
    "rope_table_sha256",
    "runtime_provenance",
    "trace",
]''')
p.write_text(s)
