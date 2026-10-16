import pathlib

p = pathlib.Path("rl_engine/models/muse_glimmer/ops/qknorm_rope.py")
s = p.read_text()

old_check = s[s.index("def _check_layer_kind(\n"):s.index("def qknorm_scaled_rope(\n")]
new_check = '''# Layer-kind bookkeeping (contract O4 "Global layers: no rotation"): which form each call used.
_LAYER_KIND_CALLS = {"explicit_local": 0, "explicit_global": 0, "implicit_local": 0,
                     "implicit_global": 0}
_REQUIRE_LAYER_KIND: bool | None = None  # None: the LAYER_KIND_ENV environment variable decides
_WARNED_IMPLICIT = False


def require_explicit_layer_kind(flag: bool | None = True) -> None:
    """Process-wide switch for the legacy implicit form ``local=None``.

    ``True``: ``local=None`` raises ``ValueError`` (fail closed; what the E2E harness should
    use once every call site passes ``local=``); ``False``: allowed (counted, warns once);
    ``None``: defer to the environment variable ``MUSEGLIMMER_O4_REQUIRE_LAYER_KIND``
    (``1`` = required), which is the default."""
    global _REQUIRE_LAYER_KIND
    _REQUIRE_LAYER_KIND = None if flag is None else bool(flag)


def layer_kind_required() -> bool:
    """True when ``local=None`` is rejected (see :func:`require_explicit_layer_kind`)."""
    if _REQUIRE_LAYER_KIND is not None:
        return _REQUIRE_LAYER_KIND
    return os.environ.get(LAYER_KIND_ENV, "").strip().lower() not in ("", "0", "false", "no")


def _check_layer_kind(
    local: bool | None, cos: torch.Tensor | None, sin: torch.Tensor | None
) -> None:
    global _WARNED_IMPLICIT
    has_tables = cos is not None or sin is not None
    if local is None:
        if layer_kind_required():
            raise ValueError(
                "qknorm_scaled_rope: the layer kind must be explicit (local=True for a "
                "sliding-window RoPE layer, local=False for a global NoPE layer); local=None is "
                f"rejected because {LAYER_KIND_ENV}=1 or require_explicit_layer_kind(True)"
            )
        _LAYER_KIND_CALLS["implicit_local" if has_tables else "implicit_global"] += 1
        if not _WARNED_IMPLICIT:
            _WARNED_IMPLICIT = True
            warnings.warn(
                "qknorm_scaled_rope(local=None): the layer kind is implied by `cos is None`, so "
                "a sliding layer given no table silently runs NoPE. Pass local=True/False "
                f"(set {LAYER_KIND_ENV}=1 to make it mandatory). Warned once; counted in "
                "runtime_provenance().",
                RuntimeWarning,
                stacklevel=3,
            )
        return
    if bool(local) and not has_tables:
        raise ValueError("local=True (sliding-window RoPE layer) needs cos/sin rows; got None")
    if not bool(local) and has_tables:
        raise ValueError("local=False (global NoPE layer) must not be given cos/sin rows")
    _LAYER_KIND_CALLS["explicit_local" if local else "explicit_global"] += 1


'''
s = s.replace(old_check, new_check)

s = s.replace('''    Layer kind: pass ``local=True`` (sliding layer, RoPE; ``cos``/``sin`` required) or
    ``local=False`` (global NoPE layer; ``cos``/``sin`` must be None) - a mismatch raises.
    ``local=None`` is the legacy implicit form (``cos is None`` selects NoPE).
    Non-bf16 ``q``/``k`` raise ``TypeError`` (no fallback).''', '''    Layer kind: pass ``local=True`` (sliding layer, RoPE; ``cos``/``sin`` required) or
    ``local=False`` (global NoPE layer; ``cos``/``sin`` must be None) - a mismatch raises.
    ``local=None`` is the legacy implicit form (``cos is None`` selects NoPE): it is counted in
    :func:`runtime_provenance`, warns once per process, and raises ``ValueError`` when
    ``MUSEGLIMMER_O4_REQUIRE_LAYER_KIND=1`` (or :func:`require_explicit_layer_kind`).
    ``cos``/``sin`` should be rows of :func:`canonical_rope_table` (``rope_rows`` /
    ``rope_table``); the op cannot tell where a table came from.
    Non-bf16 ``q``/``k`` raise ``TypeError`` (no fallback).''')

old_trace_tab = '''            "cache": "canonical_rope_table(device): one (cos, sin) object per CUDA device",
            "rows": "rope_rows(positions) = index_select of the canonical table (exact copies)",
            "positions": ROPE_TABLE_POSITIONS,
            "sha256": CANONICAL_ROPE_TABLE_SHA256,
        },
        "layer_selection": (
            "explicit: local=True (RoPE, cos/sin required) | local=False (NoPE, cos/sin must "
            "be None), mismatch raises; qknorm_scaled_rope_at requires local or layer_idx; "
            "schedule: layer i is global iff i % 4 == 3 (52 layers); local=None = legacy "
            "implicit (cos is None -> NoPE)"
        ),'''
new_trace_tab = '''            "cache": "canonical_rope_table(device): one (cos, sin) object per CUDA device, "
                     "built once (not during CUDA-graph capture), in-place changes raise",
            "rows": "rope_table(positions) = rope_rows(positions) = index_select of the "
                    "canonical table (exact copies)",
            "positions": ROPE_TABLE_POSITIONS,
            "sha256_pinned": CANONICAL_ROPE_TABLE_SHA256,
            "sha256_built": "see runtime_provenance()['rope_table_sha256_built']",
        },
        "layer_selection": (
            "local=True (RoPE, cos/sin required) | local=False (NoPE, cos/sin must be None), "
            "mismatch raises; qknorm_scaled_rope_at requires local or layer_idx; schedule: "
            "layer i is global iff i % 4 == 3 (52 layers); local=None = legacy implicit "
            "(cos is None -> NoPE), counted and warned, rejected when "
            f"{LAYER_KIND_ENV}=1; which form callers used: runtime_provenance()"
        ),
        "scope": "describes the op (static); runtime_provenance() reports what this process did",'''
assert old_trace_tab in s
s = s.replace(old_trace_tab, new_trace_tab)

old_cls = '''class QKNormScaledRopeOp:'''
new_rt = '''def runtime_provenance(*, compute_digest: bool = True) -> dict[str, Any]:
    """What THIS process actually did (measured, not a description of the op):

    ``rope_table_sha256_built``: ``{"cuda:<i>": sha256}`` of every canonical table built in this
    process at the model's theta/dim (``{}`` if none was built; computed on demand, memoised by
    version counter; ``compute_digest=False`` reports ``None`` per device instead), plus
    ``equals_pinned`` per device; ``layer_kind_calls``: how many strict calls used
    ``local=True``/``False`` and how many used the implicit legacy form;
    ``layer_kind_required``: whether ``local=None`` is currently rejected."""
    built: dict[str, str | None] = {}
    for key in sorted(_CANONICAL_TABLES):
        if (key[1], key[2]) != (ROPE_THETA, HEAD_DIM):
            continue
        built[f"cuda:{key[0]}"] = rope_table_sha256(key[0]) if compute_digest else None
    return {
        "op": "qknorm_scaled_rope",
        "rope_table_sha256_pinned": CANONICAL_ROPE_TABLE_SHA256,
        "rope_table_sha256_built": built,
        "rope_table_equals_pinned": {
            d: (None if h is None else h == CANONICAL_ROPE_TABLE_SHA256) for d, h in built.items()
        },
        "other_tables_built": [
            {"cuda": k[0], "theta": k[1], "dim": k[2]}
            for k in sorted(_CANONICAL_TABLES) if (k[1], k[2]) != (ROPE_THETA, HEAD_DIM)
        ],
        "layer_kind_calls": dict(_LAYER_KIND_CALLS),
        "layer_kind_required": layer_kind_required(),
    }


class QKNormScaledRopeOp:'''
s = s.replace(old_cls, new_rt)
p.write_text(s)
print("ok")
