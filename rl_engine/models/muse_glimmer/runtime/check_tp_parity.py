# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Muse-Glimmer WS2 TP parity check (RFC #501): run with torchrun on 2, 4 or 8 GPUs.

    PYTHONPATH=<repo> torchrun --standalone --nproc_per_node={2,4,8} \\
        -m rl_engine.models.muse_glimmer.runtime.check_tp_parity [--tokens 64] [--rows 256]

Every rank builds identical seeded inputs, computes the TP-t result of each WS2 row through the
deterministic collective and the TP1 result locally, and rank 0 prints the equality matrix, the
declared KV ownership map, the canonical chunk -> rank table, the NCCL all-reduce negative
control and the timings of all-gather + tree vs NCCL all-reduce. Exit status 1 if any row is not
byte-equal to TP1.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import torch
import torch.distributed as dist

from rl_engine.models.muse_glimmer.runtime import kv_replication as kvr
from rl_engine.models.muse_glimmer.runtime import tp_canonical_tree as tct
from rl_engine.models.muse_glimmer.runtime import vocab_parallel_softcap_logp as vp
from rl_engine.models.muse_glimmer.runtime.tp_layout import official_layout


def _seeded(shape, device, seed, dtype=torch.bfloat16, on_device=False):
    if on_device:
        gen = torch.Generator(device=device).manual_seed(seed)
        return torch.randn(*shape, device=device, dtype=dtype, generator=gen)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(*shape, generator=gen).to(device=device, dtype=dtype)


def _row_parallel(name, rows, k_size, n_size, layout, rank, collective, device, seed, reps):
    a = _seeded((rows, k_size), device, seed)
    w = _seeded((n_size, k_size), device, seed + 1)
    k0, k1 = layout.k_range(rank, k_size // layout.canonical_tp)
    a_local, w_local = a[:, k0:k1].contiguous(), w[:, k0:k1].contiguous()
    out = tct.row_parallel_tree_gemm(a_local, w_local, layout, collective)
    rerun = tct.row_parallel_tree_gemm(a_local, w_local, layout, collective)
    reference = tct.row_parallel_tree_gemm_tp1(a, w, layout)
    root = tct.subtree_partial_fp32(a_local, w_local, layout.canonical_chunks(rank))
    control = tct.nccl_all_reduce_control(root)
    return {
        "row": name,
        "shape": f"M={rows} K={k_size} N={n_size}",
        "tp_equals_tp1": bool(torch.equal(out, reference)),
        "rerun_equal": bool(torch.equal(out, rerun)),
        "nccl_all_reduce_equals_tp1": bool(torch.equal(control, reference)),
        "nccl_all_reduce_mismatches": int((control != reference).sum()),
        "numel": int(reference.numel()),
        "timings_ms": tct.time_rank_reduction(root, collective, reps=reps),
    }


def _vocab(tokens, layout, rank, collective, device, seed):
    h = _seeded((tokens, 6656), device, seed)
    gen = torch.Generator(device="cpu").manual_seed(seed + 2)
    targets = torch.randint(0, layout.vocab_size, (tokens,), generator=gen)
    tile = layout.vocab_tile_rows
    special = [0, tile - 1, tile, 7 * tile, layout.vocab_size - 1]
    targets[: len(special)] = torch.tensor(special)
    targets = targets.to(device)
    w_pad = vp.pad_lm_head_weight(
        _seeded((layout.vocab_size, 6656), device, seed + 3, on_device=True)
    )
    v0, v1 = layout.vocab_range(rank)
    w_local = w_pad[v0:v1].contiguous()
    logp = vp.vocab_parallel_softcap_logp(h, w_local, targets, layout, collective)
    rerun = vp.vocab_parallel_softcap_logp(h, w_local, targets, layout, collective)
    reference = vp.vocab_parallel_softcap_logp(h, w_pad, targets, official_layout(1))
    bad = targets.clone()
    bad[0] = layout.vocab_size
    try:
        vp.vocab_parallel_softcap_logp(h, w_local, bad, layout, collective)
        raised = False
    except ValueError:
        raised = True
    return {
        "row": "vocab_parallel_softcap_logp",
        "shape": f"T={tokens} H=6656 V={layout.vocab_size}/{layout.padded_vocab_size}",
        "tp_equals_tp1": bool(torch.equal(logp, reference)),
        "rerun_equal": bool(torch.equal(logp, rerun)),
        "pad_target_raised": raised,
        "logp_min_max": [float(logp.min()), float(logp.max())],
    }


def _kv(lengths, window, layout, rank, collective, device, seed):
    total = sum(lengths)
    q = _seeded((total, 32, 128), device, seed + 4)
    k = _seeded((total, 2, 128), device, seed + 5)
    v = _seeded((total, 2, 128), device, seed + 6)
    gate = _seeded((total, 4096), device, seed + 7)
    cu = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], device=device)
    pos = torch.cat([torch.arange(n, device=device) for n in lengths])
    y_rank = kvr.rank_kv_replicated_attention(q, k, v, gate, cu, pos, layout, rank, window=window)
    reference = kvr.gated_attention_fwd(q, k, v, gate, cu, pos, window=window)
    c0, c1 = layout.q_columns(rank)
    gathered = collective.all_gather(y_rank.t().contiguous()).t()
    return {
        "row": "kv_head_replication(forward)",
        "shape": f"T={total} seqs={list(lengths)} window={window}",
        "tp_equals_tp1": bool(torch.equal(y_rank, reference[:, c0:c1])),
        "gathered_equals_tp1": bool(torch.equal(gathered, reference)),
        "rerun_equal": bool(
            torch.equal(
                y_rank,
                kvr.rank_kv_replicated_attention(
                    q, k, v, gate, cu, pos, layout, rank, window=window
                ),
            )
        ),
        "kv_head": list(layout.kv_heads(rank)),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=256, help="M for o_proj / down")
    parser.add_argument("--tokens", type=int, default=64, help="T for the vocab row")
    parser.add_argument("--kv-lengths", default="160,352", help="sequence lengths for the KV row")
    parser.add_argument("--kv-window", type=int, default=0, help="0 = global layer, else window")
    parser.add_argument("--reps", type=int, default=20, help="timing repetitions")
    parser.add_argument("--seed", type=int, default=20261015)
    parser.add_argument("--skip-vocab", action="store_true", help="skip the 2.7 GB LM head row")
    args = parser.parse_args(argv)

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend="nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    layout = official_layout(world)
    layout.fail_closed_if_undeclared(world_size=world)
    collective = tct.make_collective(dist.group.WORLD, device)

    rows: list[dict[str, Any]] = [
        _row_parallel(
            "tp_canonical_tree(o_proj)", args.rows, 4096, 6656, layout, rank, collective, device,
            args.seed, args.reps,
        ),
        _row_parallel(
            "tp_canonical_tree(down)", args.rows, 19968, 6656, layout, rank, collective, device,
            args.seed + 10, args.reps,
        ),
    ]
    if not args.skip_vocab:
        rows.append(_vocab(args.tokens, layout, rank, collective, device, args.seed + 20))
    lengths = tuple(int(x) for x in args.kv_lengths.split(","))
    window = args.kv_window or None
    rows.append(_kv(lengths, window, layout, rank, collective, device, args.seed + 30))

    gathered: list[Any] = [None] * world
    dist.all_gather_object(gathered, rows)
    ok = True
    if rank == 0:
        print(f"# Muse-Glimmer WS2 TP parity  tp={world}  collective={collective.backend_id}")
        print(f"# gemm={tct.GEMM_BACKEND}\n# tree={tct.TREE_BACKEND}")
        print(f"# softcap={vp.SOFTCAP_BACKEND}\n# attention={kvr.ATTENTION_BACKEND}")
        print("\n## Equality matrix (TP{} vs TP1, per rank)".format(world))
        header = f"{'row':34s} {'shape':44s} " + " ".join(f"r{r}" for r in range(world)) + "  rerun"
        print(header)
        for index, row in enumerate(rows):
            flags = [gathered[r][index]["tp_equals_tp1"] for r in range(world)]
            reruns = [gathered[r][index]["rerun_equal"] for r in range(world)]
            ok &= all(flags) and all(reruns)
            marks = " ".join("==" if f else "!=" for f in flags)
            print(f"{row['row']:34s} {row['shape']:44s} {marks}  {'ok' if all(reruns) else 'DIFF'}")
            if "gathered_equals_tp1" in row:
                ok &= all(gathered[r][index]["gathered_equals_tp1"] for r in range(world))
            if "pad_target_raised" in row:
                ok &= all(gathered[r][index]["pad_target_raised"] for r in range(world))
        print("\n## Negative control: NCCL all_reduce(SUM) of the fp32 sub-tree roots -> bf16")
        for row in rows:
            if "nccl_all_reduce_equals_tp1" in row:
                print(
                    f"{row['row']:34s} equals TP1: {row['nccl_all_reduce_equals_tp1']}  "
                    f"mismatches: {row['nccl_all_reduce_mismatches']}/{row['numel']}"
                )
        print("\n## Timing per call (ms), rank 0")
        for row in rows:
            if "timings_ms" in row:
                timings = ", ".join(f"{k}={v:.3f}" for k, v in row["timings_ms"].items())
                print(f"{row['row']:34s} {timings}")
        print("\n## Canonical chunk / tile -> rank")
        print(json.dumps(tct.trace(layout)["chunk_to_rank"]))
        print("\n## KV ownership map (declared)")
        for entry in layout.kv_ownership_map()["ranks"]:
            print(
                f"rank {entry['rank']}: q heads {entry['q_heads']}  kv heads {entry['kv_heads']}  "
                f"o_proj K {entry['o_proj_k']}  down K {entry['down_k']}  "
                f"vocab tiles {entry['vocab_tiles']}"
            )
        print(f"\nRESULT: {'PASS' if ok else 'FAIL'}")
    ok_tensor = torch.tensor([1 if ok else 0], device=device)
    dist.broadcast(ok_tensor, src=0)
    if hasattr(collective, "close"):
        collective.close()
    dist.destroy_process_group()
    return 0 if int(ok_tensor) else 1


if __name__ == "__main__":
    sys.exit(main())
