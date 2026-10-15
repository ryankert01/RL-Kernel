# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Strict gated-attention forward kernel for Muse-Glimmer-30B (contract O5, RFC #501).

One Triton kernel body serves training, one-shot prefill, chunked prefill, decode and
paged-KV rollout. The arithmetic a query row sees is a pure function of its own q vector,
its absolute position and the K/V bits of the keys it may attend to; nothing depends on the
batch composition, the batch slot, the chunking of the prefill or the cache layout.

Reduction order (per query row, fp32 unless stated):

* KV is traversed in blocks of ``BLOCK_N = 64`` keys aligned at absolute key position 0,
  ascending. For a tile whose rows span positions ``[p_min, p_max]`` the visited blocks are
  ``floor(max(0, p_min - W + 1) / 64) .. floor(p_max / 64)`` on local layers and
  ``0 .. floor(p_max / 64)`` on global layers. Blocks that are fully masked for a row are
  bitwise no-ops (score = -inf -> p = 0, running max unchanged, rescale ``exp(0) = 1``), so
  visiting extra blocks because a tile spans several positions never changes a row's bits.
* ``scores = tl.dot(q_tile, K_block^T)`` (bf16 x bf16 -> fp32, K = 128), ``* s`` in fp32 with
  ``s = 128 ** -0.5``, masked to ``-inf``.
* Online softmax in this exact order: ``m_new = max(m, rowmax(scores))``;
  ``alpha = exp(m - m_new)``; ``p = exp(scores - m_new)``; ``l = l * alpha + rowsum(p)``;
  ``acc = acc * alpha + tl.dot(p -> bf16, V_block)``; ``m = m_new``. The ``p -> bf16``
  rounding before the PV dot is the declared cast. When ``m_new`` is still ``-inf`` (the row
  has not met a visible key yet) the exponent subtracts 0 instead, which keeps the
  ``(m, l, acc) = (-inf, 0, 0)`` state exactly and avoids ``-inf - -inf``.
* ``o = acc / l`` (fp32, IEEE round-to-nearest division) ``-> bf16``;
  ``y = (o_bf16 -> fp32) * sigmoid(g -> fp32)`` in fp32 ``-> bf16``, elementwise over the 4096
  flattened channels of the gate projection; ``lse = m + log(l)`` fp32.
* GQA: query head ``h`` reads KV head ``h // 16``.

Cast points: inputs bf16; scores/p/l/acc fp32; ``p`` rounded to bf16 once before the PV dot;
``o`` rounded to bf16 once; ``y`` rounded to bf16 once. No other rounding.

Two tile shapes share this body and must be bitwise identical per row (verified by the test
suite): ``TOKEN`` tiles (``BLOCK_M = 64`` consecutive queries of one head) for
prefill/training, and ``HEAD`` tiles (``BLOCK_M = 16`` = the 16 query heads of one GQA group
at a single query position) for decode / short chunks, which read each K/V block once per KV
head instead of once per query head.

Why both tiles use ``num_warps = 4``: the only reduction whose result depends on the thread
layout is ``rowsum(p)`` over the 64 keys of a block (``rowmax`` is exact in any order and the
``tl.dot`` K-accumulation is per element). Triton 3.7.1 (sm80) assigns the warps of the MMA
layout along the KEY axis for this kernel (``warpsPerCTA = [1, num_warps]``), so each warp owns
``64 / num_warps`` columns and the sum tree is "thread-local partials -> quad shuffles ->
cross-warp combine". That tree is identical for both tile heights as long as the number of
warps is identical; measured on A100: 4 warps for both tiles -> ``torch.equal`` on y/o/lse;
1, 2 or 8 warps for the HEAD tile -> last-bit differences in ``l`` (and therefore o, lse).
The test suite pins this (``test_token_vs_head_tiles_bitwise``).
"""

from __future__ import annotations

import triton
import triton.language as tl
from triton.language.extra import libdevice

# Pinned Muse-Glimmer-30B attention geometry (museglimmer_manifest.json).
NUM_Q_HEADS = 32
NUM_KV_HEADS = 2
HEAD_DIM = 128
GQA_GROUP = NUM_Q_HEADS // NUM_KV_HEADS  # 16
PAGE_SIZE = 16

# Contract tile constants (R6: compile-time, never autotuned).
BLOCK_N = 64
BLOCK_M_TOKEN = 64
BLOCK_M_HEAD = GQA_GROUP
NUM_WARPS_TOKEN = 4
NUM_WARPS_HEAD = 4  # must equal NUM_WARPS_TOKEN: same warp split along keys, same sum tree
NUM_STAGES = 2

NEG_INF = tl.constexpr(float("-inf"))
INT32_MAX = tl.constexpr(2147483647)


@triton.jit
def _gated_attention_fwd_kernel(
    q_ptr,
    gate_ptr,
    k_ptr,
    v_ptr,
    y_ptr,
    o_ptr,
    lse_ptr,
    q_pos_ptr,
    cu_q_ptr,
    cu_k_ptr,
    seq_lens_ptr,
    block_table_ptr,
    sm_scale,
    window,
    max_pages,
    H_Q: tl.constexpr,
    H_KV: tl.constexpr,
    D: tl.constexpr,
    PAGE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    LOCAL: tl.constexpr,
    PAGED: tl.constexpr,
    HEAD_TILE: tl.constexpr,
):
    """One program = one query tile of one sequence (see module docstring).

    grid = (tiles, sequences, H_Q) for TOKEN tiles, (max_len_q, sequences, H_KV) for HEAD
    tiles. ``q``/``gate``/``y``/``o`` are packed ``(T, H_Q*D)`` rows, ``lse`` is ``(T, H_Q)``.
    Varlen K/V are ``(Tk, H_KV, D)``; paged caches are ``(num_pages, PAGE, H_KV, D)``.
    """
    GROUP: tl.constexpr = H_Q // H_KV
    pid_m = tl.program_id(0)
    b = tl.program_id(1)
    pid_h = tl.program_id(2)

    q_start = tl.load(cu_q_ptr + b).to(tl.int64)
    q_end = tl.load(cu_q_ptr + b + 1).to(tl.int64)
    len_q = q_end - q_start

    offs_d = tl.arange(0, D)
    offs_n = tl.arange(0, BLOCK_N)

    if HEAD_TILE:
        tl.static_assert(BLOCK_M == GROUP, "HEAD tiles hold exactly one GQA group")
        tok = pid_m.to(tl.int64)
        if tok >= len_q:
            return
        kv_head = pid_h
        row_tok = tok + tl.zeros([BLOCK_M], dtype=tl.int64)
        row_head = pid_h * GROUP + tl.arange(0, BLOCK_M)
        row_valid = tl.arange(0, BLOCK_M) >= 0
        pos_scalar = tl.load(q_pos_ptr + q_start + tok)
        pos_rows = pos_scalar + tl.zeros([BLOCK_M], dtype=tl.int32)
        p_min = pos_scalar
        p_max = pos_scalar
    else:
        tok0 = pid_m.to(tl.int64) * BLOCK_M
        if tok0 >= len_q:
            return
        kv_head = pid_h // GROUP
        row_tok = tok0 + tl.arange(0, BLOCK_M).to(tl.int64)
        row_head = pid_h + tl.zeros([BLOCK_M], dtype=tl.int32)
        row_valid = row_tok < len_q
        pos_rows = tl.load(q_pos_ptr + q_start + row_tok, mask=row_valid, other=-1)
        p_min = tl.min(tl.where(row_valid, pos_rows, INT32_MAX), axis=0)
        p_max = tl.max(pos_rows, axis=0)

    # q tile (BLOCK_M, D) bf16. Row offsets are shared with gate/y/o (same (T, H_Q*D) layout).
    row_off = (q_start + row_tok) * (H_Q * D) + row_head.to(tl.int64) * D
    q = tl.load(q_ptr + row_off[:, None] + offs_d[None, :], mask=row_valid[:, None], other=0.0)

    if PAGED:
        len_k = tl.load(seq_lens_ptr + b)
        bt_row = block_table_ptr + b.to(tl.int64) * max_pages
    else:
        k_start = tl.load(cu_k_ptr + b).to(tl.int64)
        len_k = (tl.load(cu_k_ptr + b + 1).to(tl.int64) - k_start).to(tl.int32)

    if LOCAL:
        win_lo = p_min - window  # keys must satisfy k_pos > win_lo
        first_blk = tl.maximum(win_lo + 1, 0) // BLOCK_N
    else:
        first_blk = 0
    last_blk = tl.minimum(p_max, len_k - 1) // BLOCK_N

    m_i = tl.full([BLOCK_M], NEG_INF, dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)

    for blk in range(first_blk, last_blk + 1):
        start_n = blk * BLOCK_N
        k_pos = start_n + offs_n
        key_ok = (k_pos < len_k) & (k_pos <= p_max)
        if LOCAL:
            key_ok = key_ok & (k_pos > win_lo)
        if PAGED:
            page = tl.load(bt_row + k_pos // PAGE, mask=key_ok, other=-1)
            key_ok = key_ok & (page >= 0)
            page = tl.where(page >= 0, page, 0).to(tl.int64)
            kv_off = page * (PAGE * H_KV * D) + (k_pos % PAGE).to(tl.int64) * (H_KV * D)
        else:
            kv_off = (k_start + k_pos.to(tl.int64)) * (H_KV * D)
        kv_off = kv_off + kv_head * D
        k = tl.load(
            k_ptr + kv_off[:, None] + offs_d[None, :], mask=key_ok[:, None], other=0.0
        )
        v = tl.load(
            v_ptr + kv_off[:, None] + offs_d[None, :], mask=key_ok[:, None], other=0.0
        )

        scores = tl.dot(q, tl.trans(k), allow_tf32=False)
        scores = scores * sm_scale
        keep = key_ok[None, :] & row_valid[:, None] & (k_pos[None, :] <= pos_rows[:, None])
        if LOCAL:
            keep = keep & (k_pos[None, :] > (pos_rows[:, None] - window))
        scores = tl.where(keep, scores, NEG_INF)

        m_blk = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, m_blk)
        m_safe = tl.where(m_new == NEG_INF, 0.0, m_new)
        alpha = tl.exp(m_i - m_safe)
        p = tl.exp(scores - m_safe[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v, allow_tf32=False)
        m_i = m_new

    # IEEE round-to-nearest divisions (libdevice div_rn) so the golden's torch division is
    # the same operation; Triton's default fp32 `/` may lower to the approximate div.full.
    l_b = tl.broadcast_to(l_i[:, None], (BLOCK_M, D))
    o = libdevice.div_rn(acc, l_b)
    o_bf16 = o.to(tl.bfloat16)
    g = tl.load(
        gate_ptr + row_off[:, None] + offs_d[None, :], mask=row_valid[:, None], other=0.0
    ).to(tl.float32)
    one = tl.full([BLOCK_M, D], 1.0, dtype=tl.float32)
    gate = libdevice.div_rn(one, one + tl.exp(-g))
    y = (o_bf16.to(tl.float32) * gate).to(tl.bfloat16)

    tl.store(o_ptr + row_off[:, None] + offs_d[None, :], o_bf16, mask=row_valid[:, None])
    tl.store(y_ptr + row_off[:, None] + offs_d[None, :], y, mask=row_valid[:, None])
    lse = m_i + tl.log(l_i)
    tl.store(lse_ptr + (q_start + row_tok) * H_Q + row_head.to(tl.int64), lse, mask=row_valid)


__all__ = [
    "BLOCK_M_HEAD",
    "BLOCK_M_TOKEN",
    "BLOCK_N",
    "GQA_GROUP",
    "HEAD_DIM",
    "NUM_KV_HEADS",
    "NUM_Q_HEADS",
    "NUM_STAGES",
    "NUM_WARPS_HEAD",
    "NUM_WARPS_TOKEN",
    "PAGE_SIZE",
    "_gated_attention_fwd_kernel",
]
