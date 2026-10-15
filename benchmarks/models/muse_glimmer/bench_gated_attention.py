# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Strict gated attention (O5) vs torch SDPA bf16 on the real Muse-Glimmer-30B shapes.

Cases: prefill Tq = 4096 (B = 1, 32 heads) and decode B = 64 at 4096 context, each for the
global (causal) and the local (window 2048) layer kind. The native baseline is
``F.scaled_dot_product_attention`` bf16 with ``enable_gqa=True`` (causal flag for the global
layer, an explicit boolean sliding-window mask for the local layer) followed by the same
``* sigmoid(gate)`` in fp32 -> bf16, so both sides do the same work. Decode SDPA reads a
contiguous ``(B, 2, 4096, 128)`` K/V; the strict path reads the paged cache.

Usage (from the repo root, PYTHONPATH set)::

    ~/bin/gpu-lease -n 1 -- ~/envs/vllm/bin/python \
        benchmarks/models/muse_glimmer/bench_gated_attention.py
"""

from __future__ import annotations

import argparse
import time

import torch
import torch.nn.functional as F

from rl_engine.models.muse_glimmer.ops import gated_attention as ga
from rl_engine.models.muse_glimmer.ops.paged_kv import PagedKVCache
from rl_engine.models.muse_glimmer.triton import gated_attention_fwd as kern

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

DEV = torch.device("cuda")
HQ, HKV, D, PAGE = kern.NUM_Q_HEADS, kern.NUM_KV_HEADS, kern.HEAD_DIM, kern.PAGE_SIZE
W = ga.SLIDING_WINDOW


def _time(fn, iters: int, warmup: int = 10) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def _inputs(n: int, seed: int):
    g = torch.Generator(device=DEV).manual_seed(seed)
    q = (torch.randn(n, HQ, D, device=DEV, generator=g) * 3.87).to(torch.bfloat16)
    k = torch.randn(n, HKV, D, device=DEV, generator=g).to(torch.bfloat16)
    v = torch.randn(n, HKV, D, device=DEV, generator=g).to(torch.bfloat16)
    gate = (torch.randn(n, HQ * D, device=DEV, generator=g) * 2.0).to(torch.bfloat16)
    return q, k, v, gate


def _sdpa_gated(q4, k4, v4, gate, *, is_causal: bool, mask):
    out = F.scaled_dot_product_attention(
        q4, k4, v4, attn_mask=mask, is_causal=is_causal, enable_gqa=True
    )  # (B, HQ, S, D)
    o = out.transpose(1, 2).reshape(gate.shape)
    return (o.float() * torch.sigmoid(gate.float())).to(torch.bfloat16)


def bench_prefill(n: int, window: int | None, iters: int) -> dict:
    q, k, v, gate = _inputs(n, seed=1)
    cu = torch.tensor([0, n], dtype=torch.int32, device=DEV)
    pos = torch.arange(n, dtype=torch.int32, device=DEV)

    def strict():
        return ga.strict_gated_attention_fwd_varlen(
            q, k, v, gate, cu, cu, pos, window=window, max_seqlen_q=n
        )

    q4 = q.transpose(0, 1).unsqueeze(0).contiguous()  # (1, HQ, n, D)
    k4 = k.transpose(0, 1).unsqueeze(0).contiguous()
    v4 = v.transpose(0, 1).unsqueeze(0).contiguous()
    if window is None:
        mask = None
        is_causal = True
    else:
        i = torch.arange(n, device=DEV)
        mask = (i[None, :] <= i[:, None]) & (i[None, :] > i[:, None] - window)
        is_causal = False

    def native():
        return _sdpa_gated(q4, k4, v4, gate, is_causal=is_causal, mask=mask)

    t_strict = _time(strict, iters)
    t_native = _time(native, iters)
    y_s = strict()[0]
    y_n = native()
    err = (y_s.float() - y_n.float()).abs().max().item()
    return {
        "case": f"prefill Tq={n} B=1 ({'global' if window is None else 'local'})",
        "strict_ms": t_strict,
        "native_ms": t_native,
        "ratio": t_strict / t_native,
        "max_abs_diff_vs_sdpa": err,
    }


def bench_decode(bsz: int, ctx: int, window: int | None, iters: int) -> dict:
    seqs = [_inputs(ctx, seed=100 + i) for i in range(bsz)]
    cache = PagedKVCache(
        num_pages=bsz * (ctx // PAGE), num_seqs=bsz, max_pages_per_seq=ctx // PAGE, device=DEV
    )
    for i, (_, k, v, _) in enumerate(seqs):
        cache.append(i, k, v)
    q = torch.cat([s[0][-1:] for s in seqs]).contiguous()
    gate = torch.cat([s[3][-1:] for s in seqs]).contiguous()
    cu_q = torch.arange(bsz + 1, dtype=torch.int32, device=DEV)

    def strict_paged():
        return ga.strict_gated_attention_fwd_paged(
            q, gate, cache.k_cache, cache.v_cache, cache.block_table, cache.seq_lens, cu_q,
            window=window, max_seqlen_q=1,
        )

    k_all = torch.cat([s[1] for s in seqs])
    v_all = torch.cat([s[2] for s in seqs])
    cu_k = torch.arange(0, bsz * ctx + 1, ctx, dtype=torch.int32, device=DEV)
    pos = torch.full((bsz,), ctx - 1, dtype=torch.int32, device=DEV)

    def strict_varlen():
        return ga.strict_gated_attention_fwd_varlen(
            q, k_all, v_all, gate, cu_q, cu_k, pos, window=window, max_seqlen_q=1
        )

    q4 = q.unsqueeze(2).contiguous()  # (B, HQ, 1, D)
    k4 = k_all.view(bsz, ctx, HKV, D).transpose(1, 2).contiguous()  # (B, HKV, ctx, D)
    v4 = v_all.view(bsz, ctx, HKV, D).transpose(1, 2).contiguous()
    if window is None:
        mask = None
    else:
        j = torch.arange(ctx, device=DEV)
        mask = ((j <= ctx - 1) & (j > ctx - 1 - window)).view(1, 1, 1, ctx)

    def native():
        return _sdpa_gated(q4, k4, v4, gate, is_causal=False, mask=mask)

    t_paged = _time(strict_paged, iters)
    t_varlen = _time(strict_varlen, iters)
    t_native = _time(native, iters)
    y_s = strict_paged()[0]
    y_n = native()
    err = (y_s.float() - y_n.float()).abs().max().item()
    return {
        "case": f"decode B={bsz} ctx={ctx} ({'global' if window is None else 'local'})",
        "strict_ms": t_paged,
        "strict_varlen_ms": t_varlen,
        "native_ms": t_native,
        "ratio": t_paged / t_native,
        "max_abs_diff_vs_sdpa": err,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=30)
    args = ap.parse_args()
    print(f"device: {torch.cuda.get_device_name()}  torch {torch.__version__}")
    rows = []
    t0 = time.time()
    for window in (None, W):
        rows.append(bench_prefill(4096, window, args.iters))
        rows.append(bench_decode(64, 4096, window, args.iters))
    print(
        "\n| case | strict ms | native SDPA ms | strict/native | max abs diff y "
        "(strict vs SDPA) |"
    )
    print("|---|---|---|---|---|")
    for r in rows:
        extra = f" (varlen K/V: {r['strict_varlen_ms']:.3f})" if "strict_varlen_ms" in r else ""
        print(
            f"| {r['case']} | {r['strict_ms']:.3f}{extra} | {r['native_ms']:.3f} | "
            f"{r['ratio']:.2f}x | {r['max_abs_diff_vs_sdpa']:.3e} |"
        )
    print(f"\ntrace: {ga.trace()['tile_config']}")
    print(f"total wall {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
