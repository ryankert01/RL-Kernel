import pathlib

p = pathlib.Path("rl_engine/models/muse_glimmer/reference/golden_ops.py")
s = p.read_text()
start = s.index("_ROPE_TABLES: dict[")
end = s.index("def _rotate_half(")
new = '''def rope_table(
    positions: torch.Tensor, *, theta: float, dim: int, device: torch.device | str
) -> tuple[torch.Tensor, torch.Tensor]:
    """O4 table: cos/sin fp32 ``(T, dim)`` of ``p * theta^(-2i/dim)`` for absolute positions.

    Delegates to THE builder, :func:`rl_engine.models.muse_glimmer.ops.qknorm_rope.rope_table`:
    rows of the canonical table built once per (CUDA device, theta, dim) by the fixed fp32
    torch-CUDA function, so the golden and the Triton backends (and the runtimes) read the same
    cached object. Non-CUDA devices and positions outside ``[0, 131072)`` raise (fail closed:
    a CPU build differs in ~19 % of the elements).
    """
    from rl_engine.models.muse_glimmer.ops.qknorm_rope import rope_table as _canonical_rows

    return _canonical_rows(positions, theta=float(theta), dim=int(dim), device=device)


'''
s = s[:start] + new + s[end:]

old = '''    eps: float = 1e-5,
    qk_scale: float = 3.87,
) -> tuple[torch.Tensor, torch.Tensor]:
    """O4: per-head weightless rmsnorm (pow -0.5) in fp32; q *= qk_scale; rotate_half RoPE in
    fp32 over all channels when cos/sin are given (local layers); → bf16 once."""
    qf = q.float()'''
new2 = '''    eps: float = 1e-5,
    qk_scale: float = 3.87,
    local: bool | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """O4: per-head weightless rmsnorm (pow -0.5) in fp32; q *= qk_scale; rotate_half RoPE in
    fp32 over all channels when cos/sin are given (local layers); → bf16 once.

    ``local=True/False`` states the layer kind explicitly and must agree with ``cos``/``sin``
    (a local layer needs the table, a global layer must not get one; mismatch → ValueError).
    ``local=None`` keeps the legacy implicit form (``cos is None`` → NoPE)."""
    if (cos is None) != (sin is None):
        raise ValueError("cos and sin must both be given (local layer) or both None (global)")
    if local is not None and bool(local) != (cos is not None):
        raise ValueError(
            f"local={local} but cos/sin are {'None' if cos is None else 'given'}: a local "
            "(RoPE) layer needs the table rows, a global (NoPE) layer must not get them"
        )
    qf = q.float()'''
assert old in s
s = s.replace(old, new2)
p.write_text(s)
print("ok")
