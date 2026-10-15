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

Compiled form (Triton 3.7.1, sm80, default ``enable_fp_fusion=True``; PTX/TTGIR inspected):
``tl.exp(x)`` is ``ex2.approx.f32(x * log2 e)`` (the exp2 form), ``tl.log`` is libdevice
``__nv_logf`` with FTZ, ``libdevice.div_rn`` is ``div.rn.ftz.f32`` (round-to-nearest, subnormals
flushed), ``l * alpha + rowsum(p)`` contracts to one ``fma.rn.f32``, and the PV ``tl.dot``
accumulates into ``acc * alpha`` inside the MMA. Every tile / TP / paged variant compiles the
same sequence (bitwise parity); the torch golden does not (tolerance reference only).

Cast points: inputs bf16; scores/p/l/acc fp32; ``p`` rounded to bf16 once before the PV dot;
``o`` rounded to bf16 once; ``y`` rounded to bf16 once. No other rounding.

Two tile shapes share this body and must be bitwise identical per row (verified by the test
suite): ``TOKEN`` tiles (``BLOCK_M = 64`` consecutive queries of one head) for
prefill/training, and ``HEAD`` tiles (``BLOCK_M = 16`` rows holding the query heads of one GQA
group at a single query position) for decode / short chunks, which read each K/V block once per
KV head instead of once per query head.

Head counts are constexpr parameters (``H_Q``, ``H_KV``) so a tensor-parallel rank runs the same
body on its head shard: TP1 ``32/2`` (GQA group 16), TP2 ``16/1`` (group 16), TP4 ``8/1`` and
TP8 ``4/1`` with a replicated KV head (group 8 / 4). A HEAD tile always has ``BLOCK_M = 16``
rows; when the shard's group is smaller the unused rows are masked (they load zeros, produce
``-inf`` scores and are never stored), so the valid rows run the identical instruction stream
(same tile height, same warp split) and keep their bits. Heads never interact, so a rank's
output equals the matching slice of the full-geometry call bit for bit (test suite).

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

#: Per-rank ``(H_Q, H_KV)`` the strict attention kernels accept: TP1, TP2 (own KV head), TP4 /
#: TP8 (one replicated KV head). Rank ``r`` of TP ``t`` holds Q heads ``[32r/t, 32(r+1)/t)`` and
#: KV head ``floor(2r/t)``; its GQA group is ``H_Q // H_KV`` (16, 16, 8, 4).
SUPPORTED_HEAD_GEOMETRIES = ((NUM_Q_HEADS, NUM_KV_HEADS), (16, 1), (8, 1), (4, 1))


def check_head_geometry(num_q_heads: int, num_kv_heads: int) -> tuple[int, int]:
    """Validate a per-rank ``(H_Q, H_KV)`` against :data:`SUPPORTED_HEAD_GEOMETRIES`."""
    geom = (int(num_q_heads), int(num_kv_heads))
    if geom not in SUPPORTED_HEAD_GEOMETRIES:
        raise ValueError(
            f"(num_q_heads, num_kv_heads)={geom} is not a Muse-Glimmer TP shard; supported: "
            f"{SUPPORTED_HEAD_GEOMETRIES}"
        )
    return geom


NEG_INF = tl.constexpr(float("-inf"))
INT32_MAX = tl.constexpr(2147483647)


@triton.jit
def sigmoid_rn(x):
    """Output-gate sigmoid shared by the forward and the backward: ``1 / (1 + exp(-x))`` fp32.

    ``tl.exp`` fp32 (``ex2.approx(x * log2 e)``) and a round-to-nearest division
    (``libdevice.div_rn`` = ``div.rn.ftz.f32``): Triton's default fp32 ``/`` may lower to the
    approximate ``div.full.f32``. Both kernels must call this
    helper so the gradient differentiates the same function that produced ``y``.
    """
    one = tl.zeros_like(x) + 1.0
    return libdevice.div_rn(one, one + tl.exp(-x))


@triton.jit(do_not_specialize=["num_q_tok", "num_k_tok"])
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
    status_ptr,
    sm_scale,
    window,
    max_pages,
    num_q_tok,
    num_k_tok,
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

    Fail closed (``status_ptr``: one int32 the wrapper sets to 1 and asserts on): a program
    that finds invalid metadata stores 0 there. Checked before any data access: ``cu_q`` (and
    ``cu_k`` on the varlen path) start at 0, are non-decreasing and end at ``num_q_tok``
    (``num_k_tok``); the grid covers every query (``len_q <= rows of grid axis 0``, i.e. the
    caller's ``max_seqlen_q`` is not under-reported); paged: ``len_q <= seq_lens[b] <=
    max_pages * PAGE``. Such a program returns without reading or writing anything. Per row
    (varlen): ``0 <= q_pos < len_k``. Per visible key (paged): its block-table entry lies in
    ``[0, num_k_tok)`` (= ``num_pages``); a bad entry is never dereferenced. Paged query
    positions are derived here: ``pos = seq_lens[b] - len_q + t`` (the last ``len_q`` keys).
    """
    GROUP: tl.constexpr = H_Q // H_KV
    pid_m = tl.program_id(0)
    b = tl.program_id(1)
    pid_h = tl.program_id(2)
    last_b = tl.num_programs(1) - 1

    q_start = tl.load(cu_q_ptr + b).to(tl.int64)
    q_end = tl.load(cu_q_ptr + b + 1).to(tl.int64)
    len_q = q_end - q_start

    # ---- fail-closed metadata checks (no data is touched before they pass) ----------------
    bad = (q_start < 0) | (q_end < q_start) | (q_end > num_q_tok)
    bad = bad | ((b == 0) & (q_start != 0)) | ((b == last_b) & (q_end != num_q_tok))
    if HEAD_TILE:
        bad = bad | (len_q > tl.num_programs(0))
    else:
        bad = bad | (len_q > tl.num_programs(0).to(tl.int64) * BLOCK_M)
    if PAGED:
        len_k = tl.load(seq_lens_ptr + b)
        bad = bad | (len_k < len_q) | (len_k > max_pages * PAGE)
        bt_row = block_table_ptr + b.to(tl.int64) * max_pages
    else:
        k_start = tl.load(cu_k_ptr + b).to(tl.int64)
        k_end = tl.load(cu_k_ptr + b + 1).to(tl.int64)
        bad = bad | (k_start < 0) | (k_end < k_start) | (k_end > num_k_tok)
        bad = bad | ((b == 0) & (k_start != 0)) | ((b == last_b) & (k_end != num_k_tok))
        len_k = (k_end - k_start).to(tl.int32)
    if bad:
        tl.store(status_ptr, 0)
        return

    offs_d = tl.arange(0, D)
    offs_n = tl.arange(0, BLOCK_N)

    if HEAD_TILE:
        tl.static_assert(BLOCK_M >= GROUP, "a HEAD tile holds one whole GQA group")
        tok = pid_m.to(tl.int64)
        if tok >= len_q:
            return
        kv_head = pid_h
        row_tok = tok + tl.zeros([BLOCK_M], dtype=tl.int64)
        row_head = pid_h * GROUP + tl.arange(0, BLOCK_M)
        # rows >= GROUP exist only for TP4/TP8 shards (group 8/4); they are masked everywhere
        row_valid = tl.arange(0, BLOCK_M) < GROUP
        if PAGED:
            # the queries are the last len_q keys of the sequence (len_q <= len_k checked)
            pos_scalar = len_k - len_q.to(tl.int32) + tok.to(tl.int32)
        else:
            pos_scalar = tl.load(q_pos_ptr + q_start + tok)
            if (pos_scalar < 0) | (pos_scalar >= len_k):
                tl.store(status_ptr, 0)
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
        if PAGED:
            pos_rows = len_k - len_q.to(tl.int32) + row_tok.to(tl.int32)
            pos_rows = tl.where(row_valid, pos_rows, -1)
        else:
            pos_rows = tl.load(q_pos_ptr + q_start + row_tok, mask=row_valid, other=-1)
            pos_bad = row_valid & ((pos_rows < 0) | (pos_rows >= len_k))
            zero_rows = tl.zeros([BLOCK_M], dtype=tl.int32)
            tl.store(status_ptr + zero_rows, zero_rows, mask=pos_bad)
        p_min = tl.min(tl.where(row_valid, pos_rows, INT32_MAX), axis=0)
        p_max = tl.max(pos_rows, axis=0)

    # q tile (BLOCK_M, D) bf16. Row offsets are shared with gate/y/o (same (T, H_Q*D) layout).
    row_off = (q_start + row_tok) * (H_Q * D) + row_head.to(tl.int64) * D
    q = tl.load(q_ptr + row_off[:, None] + offs_d[None, :], mask=row_valid[:, None], other=0.0)

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
            # Every key_ok key is needed by some row of the tile (rows are consecutive
            # positions), so its page must be live: an entry outside [0, num_pages) is a cache
            # bug (evicted pages hold only keys <= p_min - W, never key_ok). Fail closed and
            # never dereference it.
            page = tl.load(bt_row + k_pos // PAGE, mask=key_ok, other=0)
            page_bad = key_ok & ((page < 0) | (page >= num_k_tok))
            zero_keys = tl.zeros([BLOCK_N], dtype=tl.int32)
            tl.store(status_ptr + zero_keys, zero_keys, mask=page_bad)
            key_ok = key_ok & (page >= 0) & (page < num_k_tok)
            page = tl.where(key_ok, page, 0).to(tl.int64)
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

    # Round-to-nearest divisions (libdevice div_rn -> div.rn.ftz.f32: IEEE-RN except that
    # subnormals flush to zero); Triton's default fp32 `/` may lower to the approximate div.full.
    l_b = tl.broadcast_to(l_i[:, None], (BLOCK_M, D))
    o = libdevice.div_rn(acc, l_b)
    o_bf16 = o.to(tl.bfloat16)
    g = tl.load(
        gate_ptr + row_off[:, None] + offs_d[None, :], mask=row_valid[:, None], other=0.0
    ).to(tl.float32)
    gate = sigmoid_rn(g)
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
    "SUPPORTED_HEAD_GEOMETRIES",
    "_gated_attention_fwd_kernel",
    "check_head_geometry",
    "sigmoid_rn",
]
