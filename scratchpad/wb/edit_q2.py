import pathlib

p = pathlib.Path("rl_engine/models/muse_glimmer/ops/qknorm_rope.py")
s = p.read_text()
start = s.index("def rope_table(\n")
end = s.index("def layer_is_local(")
new = '''def rope_table(
    positions: torch.Tensor,
    *,
    theta: float = ROPE_THETA,
    dim: int = HEAD_DIM,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """fp32 ``(cos, sin)`` rows of shape ``(T, dim)`` for absolute ``positions`` (int32|int64).

    Definition (all fp32, on a CUDA ``device``)::

        inv_freq = 1 / theta ** (arange(0, dim, 2) / dim)      # (dim/2,)
        angle    = positions.float()[:, None] * inv_freq[None]  # (T, dim/2), elementwise
        emb      = cat(angle, angle)                             # (T, dim)
        cos, sin = emb.cos(), emb.sin()

    This is the table for BOTH runtimes (training and rollout).  The formula is elementwise, so
    the row of position ``p`` has the same bits whether it is built alone or as part of
    ``arange(N)``; contract O4 nevertheless asks for ONE table "built once by a fixed fp32
    torch-CUDA function, cached, identical object on both runtimes", so this function returns
    copies of the rows of :func:`canonical_rope_table` (built once per device and ``(theta,
    dim)``, version-checked) instead of building a private table: every caller of the strict
    builder (the vLLM and Megatron bindings and the reference model bind ``rope_table``) reads
    the same cached object.  Positions must lie in ``[0, ROPE_TABLE_POSITIONS)`` (else
    ``ValueError``; one device->host sync for the range check, see :func:`rope_rows`).  Bits are
    specific to the fp32 ``cos``/``sin``/``pow`` implementation, so the builder is fixed to
    torch-CUDA: a non-CUDA ``device`` raises (fail closed; a CPU build differs from the CUDA
    build in ~19 % of the elements at T=4096 and flips ~0.1 % of the bf16 q_hat outputs).  Use
    :func:`rope_table_fp64` for references on any device.
    """
    dev = torch.device(device)
    if dev.type != "cuda":
        raise RuntimeError(
            "contract O4: the RoPE table is built by a fixed fp32 torch-CUDA function; got "
            f"device {dev}. CPU/other-device builds differ in the last bits (~19% of cos/sin "
            "elements). Build on the CUDA device that runs the kernel (canonical_rope_table)."
        )
    return rope_rows(positions, device=dev, theta=theta, dim=dim)


_TableKey = tuple[int, float, int]
_CANONICAL_TABLES: dict[_TableKey, tuple[torch.Tensor, torch.Tensor]] = {}
# (cos._version, sin._version) right after the build: any in-place change bumps them.
_CANONICAL_VERSIONS: dict[_TableKey, tuple[int, int]] = {}
# digest memo: key -> (versions the digest was computed at, hex digest)
_CANONICAL_SHA: dict[_TableKey, tuple[tuple[int, int], str]] = {}
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


def _versions(tables: tuple[torch.Tensor, torch.Tensor]) -> tuple[int, int]:
    return int(tables[0]._version), int(tables[1]._version)


def _check_unmodified(key: _TableKey, tables: tuple[torch.Tensor, torch.Tensor]) -> None:
    """Fail closed if a caller changed the shared table in place (no sync: version counters)."""
    if _versions(tables) != _CANONICAL_VERSIONS[key]:
        raise RuntimeError(
            f"the canonical RoPE table on cuda:{key[0]} (theta={key[1]}, dim={key[2]}) was "
            "modified in place after it was built (tensor version counter changed); its rows "
            "no longer follow the O4 formula and train/rollout parity is broken. Callers must "
            "never write to the canonical_rope_table() tensors."
        )


def _build_canonical(key: _TableKey) -> tuple[torch.Tensor, torch.Tensor]:
    dev = torch.device("cuda", key[0])
    if torch.cuda.is_current_stream_capturing():
        # A build recorded into a CUDA graph is not executed until the first replay: the cache
        # would hold uninitialised graph-pool memory for every eager caller until then.
        raise RuntimeError(
            "the canonical RoPE table must be built before CUDA-graph capture: call "
            "canonical_rope_table(device, verify=True) once at init (before warm-up/capture)"
        )
    # Normal tensors even when the first caller runs under torch.inference_mode() (vLLM): the
    # version counters (in-place detection) only exist on normal tensors.
    with torch.inference_mode(False), torch.no_grad():
        pos = torch.arange(ROPE_TABLE_POSITIONS, device=dev)
        cos, sin = _rope_table_in(pos, theta=key[1], dim=key[2], device=dev, dtype=torch.float32)
        return cos.contiguous(), sin.contiguous()


def _digest(tables: tuple[torch.Tensor, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for t in tables:
        h.update(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def canonical_rope_table(
    device: torch.device | str | int | None = None,
    *,
    theta: float = ROPE_THETA,
    dim: int = HEAD_DIM,
    verify: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """THE cached fp32 ``(cos, sin)`` for positions ``arange(ROPE_TABLE_POSITIONS)``.

    Built once per ``(cuda index, theta, dim)`` with the O4 formula and returned as the SAME
    tensor objects on every call (``canonical_rope_table(d)[0] is canonical_rope_table(d)[0]``).
    64 MiB each at the model's dim=128.  Fail-closed rules:

    * callers must not modify the tables in place: every call checks the tensors' version
      counters (no device sync) and raises ``RuntimeError`` if they changed since the build
      (writes through ``.data`` or foreign aliases bypass the counters; ``verify=True`` still
      catches those);
    * the first build must not happen during CUDA-graph capture (``RuntimeError``): runtimes call
      ``canonical_rope_table(device, verify=True)`` once at init, before warm-up and capture;
    * ``verify=True`` re-hashes the table (no memo; ~0.3 s) and compares it with the pinned
      :data:`CANONICAL_ROPE_TABLE_SHA256` (defined for the model's theta/dim only); a mismatch
      raises - call it once per process in each runtime to prove that both build the same table.
    """
    key = (_cuda_index(device), float(theta), int(dim))
    with _CANONICAL_LOCK:
        tables = _CANONICAL_TABLES.get(key)
        if tables is None:
            tables = _build_canonical(key)
            _CANONICAL_TABLES[key] = tables
            _CANONICAL_VERSIONS[key] = _versions(tables)
    _check_unmodified(key, tables)
    if verify:
        if (key[1], key[2]) != (ROPE_THETA, HEAD_DIM):
            raise ValueError("verify=True is only defined for theta=5e5, dim=128")
        got = _digest(tables)
        _check_unmodified(key, tables)
        with _CANONICAL_LOCK:
            _CANONICAL_SHA[key] = (_versions(tables), got)
        if got != CANONICAL_ROPE_TABLE_SHA256:
            raise RuntimeError(
                f"canonical RoPE table sha256 {got} != pinned {CANONICAL_ROPE_TABLE_SHA256}: the "
                "table bits differ from the pinned build (different torch/CUDA math library, or "
                "the table was overwritten); train/rollout parity is broken"
            )
    return tables


def rope_table_sha256(
    device: torch.device | str | int | None = None,
    *,
    theta: float = ROPE_THETA,
    dim: int = HEAD_DIM,
) -> str:
    """sha256 hex of ``cos.bytes || sin.bytes`` of :func:`canonical_rope_table` as it is NOW.

    Memoised per table and invalidated by the tables' version counters (an in-place change
    also makes :func:`canonical_rope_table` raise).  Log it in each runtime's provenance;
    equal digests == bitwise-identical tables."""
    key = (_cuda_index(device), float(theta), int(dim))
    tables = canonical_rope_table(key[0], theta=theta, dim=dim)
    versions = _versions(tables)
    memo = _CANONICAL_SHA.get(key)
    if memo is not None and memo[0] == versions:
        return memo[1]
    digest = _digest(tables)
    with _CANONICAL_LOCK:
        _CANONICAL_SHA[key] = (versions, digest)
    return digest


def rope_rows(
    positions: torch.Tensor,
    *,
    device: torch.device | str | int | None = None,
    theta: float = ROPE_THETA,
    dim: int = HEAD_DIM,
    check_range: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """fp32 ``(cos, sin)`` rows ``(T, dim)`` for absolute ``positions``, gathered (exact copies,
    ``index_select``) from :func:`canonical_rope_table` on ``device`` (default: the positions'
    CUDA device, else the current one).  Bitwise equal to the O4 formula evaluated at
    ``positions``; this is what :func:`rope_table` returns.

    Positions must be int32/int64, 1-D and inside ``[0, ROPE_TABLE_POSITIONS)``; out-of-range
    positions raise ``ValueError``.  The check costs one device->host sync for CUDA positions;
    it is skipped during CUDA-graph capture or with ``check_range=False`` (hot loops that
    validated the positions once), where ``index_select``'s device-side bounds check still
    fails loudly (it never clamps).  The canonical table itself must already exist when this is
    captured (see :func:`canonical_rope_table`).  Hoist one call per forward out of the layer
    loop."""
    positions = torch.as_tensor(positions)
    if positions.dim() != 1:
        raise ValueError(f"positions must be 1-D (T,), got shape {tuple(positions.shape)}")
    if positions.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"positions must be int32 or int64, got {positions.dtype}")
    if device is None and positions.device.type == "cuda":
        device = positions.device
    cos, sin = canonical_rope_table(device, theta=theta, dim=dim)
    capturing = positions.device.type == "cuda" and torch.cuda.is_current_stream_capturing()
    if check_range and positions.numel() and not capturing:
        lo, hi = torch.stack(torch.aminmax(positions)).tolist()  # one sync
        if lo < 0 or hi >= ROPE_TABLE_POSITIONS:
            raise ValueError(
                f"position out of range: [{lo}, {hi}] not inside [0, {ROPE_TABLE_POSITIONS}) "
                "(max_position_embeddings)"
            )
    idx = positions.to(device=cos.device, dtype=torch.long)
    return cos.index_select(0, idx), sin.index_select(0, idx)


'''
s = s[:start] + new + s[end:]
p.write_text(s)
print("ok")
