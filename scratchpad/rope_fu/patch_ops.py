import sys

p = sys.argv[1]
s = open(p).read()


def rep(old, new, count=1):
    global s
    assert s.count(old) == count, (old[:80], s.count(old))
    s = s.replace(old, new)


rep('''Public interface (imported by the reference-model and attention agents)
    ``rope_table(positions, *, theta, dim, device) -> (cos, sin)`` fp32 ``(T, dim)``
    ``qknorm_scaled_rope(q, k, cos, sin, *, eps, qk_scale)`` strict Triton op, bf16 -> bf16,
    differentiable (deterministic row-local Triton backward)
''', '''Public interface (imported by the reference-model and attention agents)
    ``rope_table(positions, *, theta, dim, device) -> (cos, sin)`` fp32 ``(T, dim)``; THE table
    definition; CUDA devices only (a CPU build differs in ~19% of the elements)
    ``canonical_rope_table(device)`` the cached ``(cos, sin)`` for ``arange(131072)`` (one
    object per device; sha256 pinned in ``CANONICAL_ROPE_TABLE_SHA256``)
    ``rope_rows(positions)`` rows gathered from the canonical table (bitwise == ``rope_table``)
    ``qknorm_scaled_rope(q, k, cos, sin, *, eps, qk_scale, local=None)`` strict Triton op,
    bf16 -> bf16 (other dtypes raise), differentiable (deterministic row-local Triton backward);
    ``local=True/False`` makes the layer kind explicit and is checked against ``cos``/``sin``
    ``qknorm_scaled_rope_at(q, k, positions, *, local | layer_idx)`` the contract's assembly
    signature ``qknorm_scaled_rope(q, k, pos, local?)``: explicit layer kind, canonical table
    ``layer_is_local(i)`` the frozen schedule (layer i is global iff i % 4 == 3, 52 layers)
''')

rep('''from typing import Any

import torch
''', '''import hashlib
import threading
from typing import Any

import torch
''')

rep('''ROPE_THETA = 500000.0
RMS_NORM_EPS = 1e-5
QK_SCALE = 3.87
''', '''ROPE_THETA = 500000.0
RMS_NORM_EPS = 1e-5
QK_SCALE = 3.87
NUM_LAYERS = 52
GLOBAL_LAYER_PERIOD = 4  # layer i is global (NoPE) iff i % 4 == 3
ROPE_TABLE_POSITIONS = 131072  # max_position_embeddings: rows of the canonical table
# sha256(cos.bytes || sin.bytes) of rope_table(arange(131072), device=cuda) (fp32, little
# endian).  Measured on A100 with torch 2.13.0+cu130 (vLLM env) and torch 2.13.0+cu129
# (Megatron env): identical, so both runtimes build the same table.
CANONICAL_ROPE_TABLE_SHA256 = "088dbb8c93a55ce79d189181c46ac06fbdb61dab6c3914218bbe40ed3ea56d9c"
''')

rep('''__all__ = [
    "HEAD_DIM",
    "QK_SCALE",
    "RMS_NORM_EPS",
    "ROPE_THETA",
    "QKNormScaledRopeOp",
    "qknorm_scaled_rope",
    "qknorm_scaled_rope_backward_fp32",
    "qknorm_scaled_rope_fp32",
    "qknorm_scaled_rope_fp64",
    "rope_table",
    "rope_table_fp64",
    "trace",
]''', '''__all__ = [
    "CANONICAL_ROPE_TABLE_SHA256",
    "GLOBAL_LAYER_PERIOD",
    "HEAD_DIM",
    "NUM_LAYERS",
    "QK_SCALE",
    "RMS_NORM_EPS",
    "ROPE_TABLE_POSITIONS",
    "ROPE_THETA",
    "QKNormScaledRopeOp",
    "canonical_rope_table",
    "layer_is_local",
    "qknorm_scaled_rope",
    "qknorm_scaled_rope_at",
    "qknorm_scaled_rope_backward_fp32",
    "qknorm_scaled_rope_fp32",
    "qknorm_scaled_rope_fp64",
    "rope_rows",
    "rope_table",
    "rope_table_fp64",
    "rope_table_sha256",
    "trace",
]''')

rep('''    This is the table definition for BOTH runtimes (training and rollout); it is a pure
    function of ``positions`` and deterministic, so callers cache it (e.g. build it once for
    ``arange(max_positions)`` and gather rows with ``positions``).  Bits are specific to the
    device's fp32 ``cos``/``sin``/``pow`` implementation: build it on the device that runs the
    kernel, as the contract says (fixed fp32 torch-CUDA function).
    """
    return _rope_table_in(positions, theta=theta, dim=dim, device=device, dtype=torch.float32)
''', '''    This is the table definition for BOTH runtimes (training and rollout); it is a pure,
    deterministic and elementwise function of ``positions``, so the row of position ``p`` has
    the same bits whether it is built alone or as part of ``arange(N)``.  Prefer
    :func:`canonical_rope_table` / :func:`rope_rows` (built once per device, cached, fingerprint
    pinned).  Bits are specific to the fp32 ``cos``/``sin``/``pow`` implementation, so the
    contract fixes the builder to torch-CUDA: a non-CUDA ``device`` raises (fail closed; a CPU
    build differs from the CUDA build in ~19 % of the elements at T=4096 and flips ~0.1 % of
    the bf16 q_hat outputs).  Use :func:`rope_table_fp64` for references on any device.
    """
    dev = torch.device(device)
    if dev.type != "cuda":
        raise RuntimeError(
            "contract O4: the RoPE table is built by a fixed fp32 torch-CUDA function; got "
            f"device {dev}. CPU/other-device builds differ in the last bits (~19% of cos/sin "
            "elements). Build on the CUDA device that runs the kernel (canonical_rope_table)."
        )
    return _rope_table_in(positions, theta=theta, dim=dim, device=dev, dtype=torch.float32)


_CANONICAL_TABLES: dict[tuple[int, float, int], tuple[torch.Tensor, torch.Tensor]] = {}
_CANONICAL_SHA: dict[tuple[int, float, int], str] = {}
_CANONICAL_LOCK = threading.Lock()


def _cuda_index(device: torch.device | str | int | None) -> int:
    if device is None:
        return torch.cuda.current_device()
    dev = torch.device("cuda", device) if isinstance(device, int) else torch.device(device)
    if dev.type != "cuda":
        raise RuntimeError(
            f"contract O4: the canonical RoPE table lives on a CUDA device; got {dev}"
        )
    return dev.index if dev.index is not None else torch.cuda.current_device()


def canonical_rope_table(
    device: torch.device | str | int | None = None,
    *,
    theta: float = ROPE_THETA,
    dim: int = HEAD_DIM,
    verify: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """THE cached fp32 ``(cos, sin)`` for positions ``arange(ROPE_TABLE_POSITIONS)``.

    Built once per ``(cuda index, theta, dim)`` with :func:`rope_table` and returned as the SAME
    tensor objects on every call (``canonical_rope_table(d)[0] is canonical_rope_table(d)[0]``);
    callers must not modify them in place.  64 MiB each at the model's dim=128.
    ``verify=True`` checks the bits against the pinned :data:`CANONICAL_ROPE_TABLE_SHA256`
    (defined for the model's theta/dim only) and raises on a mismatch - call it once per process
    in each runtime to prove that both build the same table.
    """
    key = (_cuda_index(device), float(theta), int(dim))
    with _CANONICAL_LOCK:
        tables = _CANONICAL_TABLES.get(key)
        if tables is None:
            dev = torch.device("cuda", key[0])
            pos = torch.arange(ROPE_TABLE_POSITIONS, device=dev)
            cos, sin = rope_table(pos, theta=key[1], dim=key[2], device=dev)
            tables = (cos.contiguous(), sin.contiguous())
            _CANONICAL_TABLES[key] = tables
    if verify:
        if (key[1], key[2]) != (ROPE_THETA, HEAD_DIM):
            raise ValueError("verify=True is only defined for theta=5e5, dim=128")
        got = rope_table_sha256(key[0])
        if got != CANONICAL_ROPE_TABLE_SHA256:
            raise RuntimeError(
                f"canonical RoPE table sha256 {got} != pinned {CANONICAL_ROPE_TABLE_SHA256}: this "
                "torch/CUDA build computes different cos/sin bits; train/rollout parity is broken"
            )
    return tables


def rope_table_sha256(
    device: torch.device | str | int | None = None,
    *,
    theta: float = ROPE_THETA,
    dim: int = HEAD_DIM,
) -> str:
    """sha256 hex of ``cos.bytes || sin.bytes`` of :func:`canonical_rope_table` (cached).

    Log it in each runtime's provenance; equal digests == bitwise-identical tables."""
    key = (_cuda_index(device), float(theta), int(dim))
    digest = _CANONICAL_SHA.get(key)
    if digest is None:
        cos, sin = canonical_rope_table(key[0], theta=theta, dim=dim)
        h = hashlib.sha256()
        for t in (cos, sin):
            h.update(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
        digest = h.hexdigest()
        _CANONICAL_SHA[key] = digest
    return digest


def rope_rows(
    positions: torch.Tensor,
    *,
    device: torch.device | str | int | None = None,
    theta: float = ROPE_THETA,
    dim: int = HEAD_DIM,
) -> tuple[torch.Tensor, torch.Tensor]:
    """fp32 ``(cos, sin)`` rows ``(T, dim)`` for absolute ``positions``, gathered (exact copies,
    ``index_select``) from :func:`canonical_rope_table` on ``device`` (default: the positions'
    CUDA device, else the current one).  Bitwise equal to ``rope_table(positions)``.

    Positions must be int32/int64, 1-D and inside ``[0, ROPE_TABLE_POSITIONS)``; out-of-range
    positions raise ``ValueError`` (checked on the host, one sync; skipped during CUDA-graph
    capture, where ``index_select``'s device-side bounds check still fails loudly)."""
    positions = torch.as_tensor(positions)
    if positions.dim() != 1:
        raise ValueError(f"positions must be 1-D (T,), got shape {tuple(positions.shape)}")
    if positions.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"positions must be int32 or int64, got {positions.dtype}")
    if device is None and positions.device.type == "cuda":
        device = positions.device
    cos, sin = canonical_rope_table(device, theta=theta, dim=dim)
    capturing = positions.device.type == "cuda" and torch.cuda.is_current_stream_capturing()
    if positions.numel() and not capturing:
        lo, hi = int(positions.min()), int(positions.max())
        if lo < 0 or hi >= ROPE_TABLE_POSITIONS:
            raise ValueError(
                f"position out of range: [{lo}, {hi}] not inside [0, {ROPE_TABLE_POSITIONS}) "
                "(max_position_embeddings)"
            )
    idx = positions.to(device=cos.device, dtype=torch.long)
    return cos.index_select(0, idx), sin.index_select(0, idx)


def layer_is_local(layer_idx: int) -> bool:
    """Frozen layer schedule: layer ``i`` (0..51) is global/NoPE iff ``i % 4 == 3``, else local
    (sliding window + RoPE): 39 local and 13 global layers."""
    i = int(layer_idx)
    if not 0 <= i < NUM_LAYERS:
        raise ValueError(f"layer_idx must be in [0, {NUM_LAYERS}), got {layer_idx}")
    return i % GLOBAL_LAYER_PERIOD != GLOBAL_LAYER_PERIOD - 1
''')

rep('''def qknorm_scaled_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor | None,
    sin: torch.Tensor | None,
    *,
    eps: float = RMS_NORM_EPS,
    qk_scale: float = QK_SCALE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Strict O4 op: ``(q (T,32,128) bf16, k (T,2,128) bf16, cos/sin (T,128) fp32 | None)``
    -> ``(q_hat bf16, k_hat bf16)``.  ``cos is None`` selects the global (NoPE) layer path.

    Differentiable w.r.t. ``q`` and ``k`` with a deterministic, batch-invariant Triton backward.
    """
    return _QKNormScaledRopeFunction.apply(q, k, cos, sin, float(eps), float(qk_scale))
''', '''def _check_layer_kind(
    local: bool | None, cos: torch.Tensor | None, sin: torch.Tensor | None
) -> None:
    if local is None:
        return
    has_tables = cos is not None or sin is not None
    if bool(local) and not has_tables:
        raise ValueError("local=True (sliding-window RoPE layer) needs cos/sin rows; got None")
    if not bool(local) and has_tables:
        raise ValueError("local=False (global NoPE layer) must not be given cos/sin rows")


def qknorm_scaled_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor | None,
    sin: torch.Tensor | None,
    *,
    eps: float = RMS_NORM_EPS,
    qk_scale: float = QK_SCALE,
    local: bool | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Strict O4 op: ``(q (T,HQ,128) bf16, k (T,HK,128) bf16, cos/sin (T,128) fp32 | None)``
    -> ``(q_hat bf16, k_hat bf16)``.  HQ/HK are 32/2 at TP1 and the per-rank head shards
    16/8/4 and 1 at TP2/4/8 (bitwise equal to the slice of the full call).

    Layer kind: pass ``local=True`` (sliding layer, RoPE; ``cos``/``sin`` required) or
    ``local=False`` (global NoPE layer; ``cos``/``sin`` must be None) - a mismatch raises.
    ``local=None`` is the legacy implicit form (``cos is None`` selects NoPE).
    Non-bf16 ``q``/``k`` raise ``TypeError`` (no fallback).

    Differentiable w.r.t. ``q`` and ``k`` with a deterministic, batch-invariant Triton backward.
    """
    _check_layer_kind(local, cos, sin)
    return _QKNormScaledRopeFunction.apply(q, k, cos, sin, float(eps), float(qk_scale))


def qknorm_scaled_rope_at(
    q: torch.Tensor,
    k: torch.Tensor,
    positions: torch.Tensor | None,
    *,
    local: bool | None = None,
    layer_idx: int | None = None,
    eps: float = RMS_NORM_EPS,
    qk_scale: float = QK_SCALE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The contract's assembly signature ``qknorm_scaled_rope(q, k, pos, local?)``.

    The layer kind is mandatory: give ``local`` or ``layer_idx`` (or both, consistent).  Local
    layers gather ``cos``/``sin`` rows for ``positions`` from :func:`canonical_rope_table` on
    ``q``'s device (so the table bits cannot come from another builder); global layers ignore
    ``positions`` (may be None).  Same bits as :func:`qknorm_scaled_rope` with ``rope_table``
    rows."""
    if local is None and layer_idx is None:
        raise ValueError("give local=True/False or layer_idx: the layer kind must be explicit")
    if layer_idx is not None:
        from_idx = layer_is_local(layer_idx)
        if local is not None and bool(local) != from_idx:
            raise ValueError(
                f"local={local} contradicts layer_idx={layer_idx} (global iff i % 4 == 3)"
            )
        local = from_idx
    if not local:
        return qknorm_scaled_rope(q, k, None, None, eps=eps, qk_scale=qk_scale, local=False)
    if positions is None:
        raise ValueError("a local (RoPE) layer needs positions")
    positions = torch.as_tensor(positions)
    if positions.dim() != 1 or positions.shape[0] != q.shape[0]:
        raise ValueError(
            f"positions must be (T,) with T={q.shape[0]}; got shape {tuple(positions.shape)}"
        )
    cos, sin = rope_rows(positions, device=q.device)
    return qknorm_scaled_rope(q, k, cos, sin, eps=eps, qk_scale=qk_scale, local=True)
''')

rep('''        "scale": {"q": QK_SCALE, "k": 1.0, "applied": "fp32 after norm, before rotation"},
        "eps": RMS_NORM_EPS,
        "rope_theta": ROPE_THETA,
        "tile_config": tile_config(HQ, HK, D),''', '''        "scale": {"q": QK_SCALE, "k": 1.0, "applied": "fp32 after norm, before rotation"},
        "eps": RMS_NORM_EPS,
        "rope_theta": ROPE_THETA,
        "rope_table": {
            "builder": "rl_engine.models.muse_glimmer.ops.qknorm_rope.rope_table",
            "device_type": "cuda",
            "formula": (
                "inv_freq = 1 / torch.pow(theta, arange(0,128,2,fp32)/128); "
                "angle = float(p)[:,None] * inv_freq[None,:]; emb = cat(angle, angle); "
                "cos, sin = emb.cos(), emb.sin()  (fp32, torch-CUDA)"
            ),
            "cache": "canonical_rope_table(device): one (cos, sin) object per CUDA device",
            "rows": "rope_rows(positions) = index_select of the canonical table (exact copies)",
            "positions": ROPE_TABLE_POSITIONS,
            "sha256": CANONICAL_ROPE_TABLE_SHA256,
        },
        "layer_selection": (
            "explicit: local=True (RoPE, cos/sin required) | local=False (NoPE, cos/sin must "
            "be None), mismatch raises; qknorm_scaled_rope_at requires local or layer_idx; "
            "schedule: layer i is global iff i % 4 == 3 (52 layers); local=None = legacy "
            "implicit (cos is None -> NoPE)"
        ),
        "dtype_policy": "q/k bf16 only; fp16/fp32 raise TypeError (no fallback)",
        "tile_config": tile_config(HQ, HK, D),''')

rep('''    def __call__(self, q, k, cos, sin):
        return self.forward(q, k, cos, sin)

    def forward(self, q, k, cos, sin):
        return qknorm_scaled_rope(q, k, cos, sin, eps=self.eps, qk_scale=self.qk_scale)
''', '''    def __call__(self, q, k, cos, sin, *, local: bool | None = None):
        return self.forward(q, k, cos, sin, local=local)

    def forward(self, q, k, cos, sin, *, local: bool | None = None):
        return qknorm_scaled_rope(
            q, k, cos, sin, eps=self.eps, qk_scale=self.qk_scale, local=local
        )
''')
open(p, "w").write(s)
print("patched", p)
