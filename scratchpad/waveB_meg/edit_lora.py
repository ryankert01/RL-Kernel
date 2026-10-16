from pathlib import Path
import sys

p = Path(sys.argv[1])
s = p.read_text()


def rep(old, new, count=1):
    global s
    assert s.count(old) == count, (old[:80], s.count(old))
    s = s.replace(old, new)


rep('''Backward reduction orders (contract gap, declared here): the input gradient of a
column-parallel projection is the fixed tree over the 8 canonical output-row chunks of the
FULL ``N`` (one whole-chunk GEMM each); the LoRA contributions use the same chunks; base and
LoRA fp32 input gradients are added once (``dx = bf16(dx_base + dx_lora)``); parameter
gradients reduce over the (zero-padded) token axis with the canonical 8-chunk tree.
"""''', '''Backward reduction orders (contract gap, declared here): the input gradient of a
column-parallel projection is the fixed tree over the 8 canonical output-row chunks of the
FULL ``N`` (one whole-chunk GEMM each); the LoRA contributions use the same chunks; base and
LoRA fp32 input gradients are added once (``dx = bf16(dx_base + dx_lora)``).

LoRA weight gradients (reduction over TOKENS), :data:`LORA_GRAD_RULE`:

* per sequence ``s`` of the packed micro-batch (``segments``), ``g_s`` = the det_gemm dW rule
  over its ``T_s`` tokens: ``T_pad = roundup(T_s, 256)``, 8 chunks of ``T_pad/8`` rows, rows
  ``>= T_s`` masked to zero, fp32 tree (``fast_det_gemm(..., ragged_k=True)``, the rule
  ``ops.det_gemm.DW_CHUNK_RULE`` declares);
* the step gradient is the LEFT FOLD in fp32 over the request's sequences in sample order,
  ``((g_0 + g_1) + g_2) + ...``, written straight into the fp32 master's ``.grad`` (the op
  accumulates; it returns ``None`` to autograd for these inputs).

So the gradient bits do not depend on how the sequences are packed into micro-batches
(``micro_batch_tokens``) nor on the TP size; they depend on the sample order, which is part of
the data. Without ``segments`` (single-sequence callers, the vLLM rule check) the whole input is
one segment and the gradient is returned through autograd as usual.
"""''')

rep('''from rl_engine.models.muse_glimmer.runtime.strict_kernels import (
    CANONICAL_CHUNKS,
    TPComm,
    ceil_to,
    gemm_backend,
    gemm_fp32,
    gemm_single_chunk_fp32,
    local_chunk_ids,
    small_k_fp32,
)''', '''from rl_engine.models.muse_glimmer.runtime.strict_kernels import (
    CANONICAL_CHUNKS,
    TOKEN_PAD,
    TPComm,
    TreeAccumulator,
    gemm_backend,
    gemm_fp32,
    gemm_single_chunk_fp32,
    local_chunk_ids,
    small_k_fp32,
)''')

rep('''_TOKEN_PAD = CANONICAL_CHUNKS * 64
DEFAULT_LORA_RANK = 16


def _pad_tokens_t(x_t: torch.Tensor, m_pad: int) -> torch.Tensor:
    """Zero-pad the token axis (dim 1 of a transposed activation) to ``m_pad``."""
    m = x_t.shape[1]
    if m == m_pad:
        return x_t.contiguous()
    return torch.nn.functional.pad(x_t, (0, m_pad - m)).contiguous()


def _token_grad_gemm(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    """``lhs.T @ rhs`` reducing over tokens: ``lhs (M, P)``, ``rhs (M, Q)`` → fp32 ``(P, Q)``.

    Tokens are zero-padded to a multiple of 8*64 so the reduction forms 8 canonical chunks.
    """
    m_pad = ceil_to(lhs.shape[0], _TOKEN_PAD)
    a = _pad_tokens_t(lhs.t(), m_pad)
    w = _pad_tokens_t(rhs.t(), m_pad)
    return gemm_fp32(a, w)
''', '''DEFAULT_LORA_RANK = 16
LORA_GRAD_RULE = (
    "per sequence: det_gemm dW rule over its tokens (T_pad = roundup(T, %d), 8 chunks of "
    "T_pad/8 rows, fp32 tree); step gradient = left fold in fp32 over the request's sequences "
    "in sample order, accumulated into the fp32 master .grad (packing- and TP-invariant)"
    % TOKEN_PAD
)

Segments = tuple[tuple[int, int], ...]


def _token_grad_gemm(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    """``lhs.T @ rhs`` reducing over tokens: ``lhs (T, P)``, ``rhs (T, Q)`` -> fp32 ``(P, Q)``.

    The det_gemm dW rule: ``T_pad = roundup(T, 256)``, 8 chunks of ``T_pad/8`` token rows
    (masked tail), fp32 chunk tree (``strict_kernels._GemmBackend.token_grad``).
    """
    return gemm_backend().token_grad(lhs, rhs)


def normalize_segments(segments: Any, tokens: int) -> Segments | None:
    """Validate ``((start, end), ...)`` covering ``[0, tokens)`` contiguously, in order."""
    if segments is None:
        return None
    out = tuple((int(s), int(e)) for s, e in segments)
    pos = 0
    for s, e in out:
        if s != pos or e <= s:
            raise ValueError(f"segments {out} do not tile [0, {tokens}) in order")
        pos = e
    if pos != tokens:
        raise ValueError(f"segments end at {pos}, the input has {tokens} tokens")
    return out


def _accumulate_token_grad(
    param: torch.Tensor, lhs: torch.Tensor, rhs: torch.Tensor, segments: Segments
) -> None:
    """``param.grad += g_s`` for each segment in order (``g_s`` = :func:`_token_grad_gemm`)."""
    with torch.no_grad():
        for start, end in segments:
            part = _token_grad_gemm(lhs[start:end], rhs[start:end])
            if part.shape != param.shape:
                raise RuntimeError(f"LoRA grad shape {tuple(part.shape)} != {tuple(param.shape)}")
            if param.grad is None:
                param.grad = part
            else:
                param.grad.add_(part)
''')

start = s.index("class StrictColumnLinear(torch.autograd.Function):")
end = s.index("@dataclass(frozen=True)\nclass ProjectionShape:")
s = s[:start] + Path(sys.argv[2]).read_text() + s[end:]

rep('''__all__ = [
    "COLUMN_PROJECTIONS",
    "DEFAULT_LORA_RANK",
    "LayerLoRA",''', '''__all__ = [
    "COLUMN_PROJECTIONS",
    "DEFAULT_LORA_RANK",
    "LORA_GRAD_RULE",
    "LayerLoRA",''')
rep('''    "muse_glimmer_projection_shapes",
    "replicated_lora_grads_agree",
]''', '''    "muse_glimmer_projection_shapes",
    "normalize_segments",
    "replicated_lora_grads_agree",
]''')
p.write_text(s)
print("ok")
