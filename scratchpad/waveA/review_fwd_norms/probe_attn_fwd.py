# Adversarial review probes for the strict gated-attention FORWARD (contract O5).
# Read-only: imports repo code, writes nothing into the repo. Prints one JSON line per probe.
from __future__ import annotations

import json
import re
import sys

import torch

from rl_engine.models.muse_glimmer.ops import gated_attention as ga
from rl_engine.models.muse_glimmer.ops.paged_kv import PagedKVCache
from rl_engine.models.muse_glimmer.triton import gated_attention_fwd as kern

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
DEV = torch.device("cuda")
HQ, HKV, D, W = kern.NUM_Q_HEADS, kern.NUM_KV_HEADS, kern.HEAD_DIM, ga.SLIDING_WINDOW
RESULTS: dict[str, dict] = {}


def emit(name: str, **kw) -> None:
    RESULTS[name] = kw
    print(json.dumps({"probe": name, **kw}), flush=True)


def inputs(n: int, seed: int):
    g = torch.Generator(device=DEV).manual_seed(seed)
    q = (torch.randn(n, HQ, D, device=DEV, generator=g) * 3.87).to(torch.bfloat16)
    k = torch.randn(n, HKV, D, device=DEV, generator=g).to(torch.bfloat16)
    v = torch.randn(n, HKV, D, device=DEV, generator=g).to(torch.bfloat16)
    gate = (torch.randn(n, HQ * D, device=DEV, generator=g) * 2.0).to(torch.bfloat16)
    return q, k, v, gate


def i32(vals):
    return torch.tensor(list(vals), dtype=torch.int32, device=DEV)


def one_shot(q, k, v, gate, window, **kw):
    n = q.shape[0]
    cu = i32([0, n])
    pos = torch.arange(n, dtype=torch.int32, device=DEV)
    return ga.strict_gated_attention_fwd_varlen(q, k, v, gate, cu, cu, pos, window=window, **kw)


def ndiff(a, b) -> int:
    a, b = a.float(), b.float()
    same = (a == b) | (torch.isnan(a) & torch.isnan(b))
    return int((~same).sum().item())


# ---------------------------------------------------------------- A1 undersized max_seqlen_q
def probe_max_seqlen_undersized():
    n = 200
    q, k, v, gate = inputs(n, 1)
    ref = one_shot(q, k, v, gate, None)
    # Fill the caching allocator's next blocks with a sentinel so skipped rows are visible.
    junk = torch.full((n, HQ * D), 7.0, dtype=torch.bfloat16, device=DEV)
    del junk
    raised = None
    try:
        got = one_shot(q, k, v, gate, None, max_seqlen_q=64)
    except Exception as e:  # noqa: BLE001
        raised = repr(e)
        got = None
    if got is None:
        emit("A1_max_seqlen_q_undersized", raised=raised)
        return
    rows_bad = int(((got[0].float() != ref[0].float()).any(dim=1)).sum().item())
    emit(
        "A1_max_seqlen_q_undersized",
        raised=raised,
        n_rows=n,
        max_seqlen_q_passed=64,
        rows_differing_from_correct=rows_bad,
        first_bad_row=int(((got[0].float() != ref[0].float()).any(dim=1)).nonzero()[0].item())
        if rows_bad
        else None,
        sample_value_row_150=got[0][150, :4].float().tolist(),
        y_has_nan=bool(torch.isnan(got[0].float()).any().item()),
    )


# ---------------------------------------------------------------- A2 -1 page inside the window
def probe_missing_page_in_window():
    n = 300
    q, k, v, gate = inputs(n, 2)
    ref = one_shot(q, k, v, gate, W)
    cache = PagedKVCache(num_pages=40, num_seqs=1, max_pages_per_seq=-(-n // 16), device=DEV)
    cache.append(0, k, v)
    # Simulate an allocator bug: logical page 5 (keys 80..95, all visible to the query at 299)
    cache.block_table[0, 5] = -1
    raised = None
    try:
        y, o, lse = ga.strict_gated_attention_fwd_paged(
            q[-1:].contiguous(), gate[-1:].contiguous(), cache.k_cache, cache.v_cache,
            cache.block_table, cache.seq_lens, i32([0, 1]), window=W,
        )
    except Exception as e:  # noqa: BLE001
        raised = repr(e)
        y = None
    if y is None:
        emit("A2_missing_page_inside_window", raised=raised)
        return
    emit(
        "A2_missing_page_inside_window",
        raised=raised,
        y_elements_differing_from_reference=ndiff(y, ref[0][-1:]),
        lse_diff=float((lse - ref[2][-1:]).abs().max().item()),
        y_has_nan=bool(torch.isnan(y.float()).any().item()),
        verdict="silently dropped 16 visible keys (no error, finite output)"
        if not torch.isnan(y.float()).any()
        else "nan",
    )


# ---------------------------------------------------------------- A3 seq_lens beyond the table
def probe_seq_len_past_block_table():
    n0, n1 = 64, 64
    q0, k0, v0, g0 = inputs(n0, 3)
    _, k1, v1, _ = inputs(n1, 4)
    cache = PagedKVCache(num_pages=16, num_seqs=2, max_pages_per_seq=4, device=DEV)
    cache.append(0, k0, v0)
    cache.append(1, k1, v1)
    # Caller claims seq 0 has 80 keys (5 pages) but the table is only 4 pages wide.
    seq_lens = cache.seq_lens.clone()
    seq_lens[0] = 80
    qd = q0[-1:].contiguous()
    gd = g0[-1:].contiguous()
    cu = i32([0, 1, 1])  # one decode token for seq 0, none for seq 1
    raised = None
    try:
        y_a, _, lse_a = ga.strict_gated_attention_fwd_paged(
            qd, gd, cache.k_cache, cache.v_cache, cache.block_table, seq_lens, cu, window=None
        )
        # perturb ONLY sequence 1's KV pages
        for p in cache.allocated_pages(1):
            cache.k_cache[p] = torch.randn_like(cache.k_cache[p].float()).to(torch.bfloat16)
            cache.v_cache[p] = torch.randn_like(cache.v_cache[p].float()).to(torch.bfloat16)
        y_b, _, lse_b = ga.strict_gated_attention_fwd_paged(
            qd, gd, cache.k_cache, cache.v_cache, cache.block_table, seq_lens, cu, window=None
        )
    except Exception as e:  # noqa: BLE001
        raised = repr(e)
        y_a = y_b = None
    if y_a is None:
        emit("A3_seq_lens_past_block_table", raised=raised)
        return
    emit(
        "A3_seq_lens_past_block_table",
        raised=raised,
        seq0_output_changes_when_seq1_kv_changes=not torch.equal(y_a, y_b),
        y_elements_changed=ndiff(y_a, y_b),
        lse_change=float((lse_a - lse_b).abs().max().item()),
    )


# ---------------------------------------------------------------- A4 window off-by-one vs G1 tol
def probe_window_off_by_one_undetected():
    from rl_engine.contracts.numerical import load_contract, resolve_tolerance

    tol = resolve_tolerance(
        load_contract(), judgment="forward_accuracy", op_class="attention", dtype=torch.bfloat16
    )
    n = 2300  # the G1 test length that crosses the window
    q, k, v, gate = inputs(n, 11 + n)
    pos = torch.arange(n, dtype=torch.int32, device=DEV)
    yg, og, lseg = ga.gated_attention_forward_fp32(q, k, v, gate, pos, W)
    out = {}
    for wbad in (W - 1, W + 1):
        y, o, lse = one_shot(q, k, v, gate, wbad)
        ok_y = torch.allclose(y.float(), yg.float(), atol=tol.atol, rtol=tol.rtol)
        ok_o = torch.allclose(o.float(), og.float(), atol=tol.atol, rtol=tol.rtol)
        out[f"window_{wbad}"] = {
            "G1_y_allclose_passes": bool(ok_y),
            "G1_o_allclose_passes": bool(ok_o),
            "max_abs_y": float((y.float() - yg.float()).abs().max().item()),
            "max_abs_lse": float((lse - lseg).abs().max().item()),
            "G1_lse_lt_1e-4_passes": bool((lse - lseg).abs().max().item() < 1e-4),
        }
    yw, ow, lw = one_shot(q, k, v, gate, W)
    out["window_2048_reference"] = {
        "max_abs_y": float((yw.float() - yg.float()).abs().max().item()),
        "max_abs_lse": float((lw - lseg).abs().max().item()),
    }
    emit("A4_window_off_by_one_vs_G1_tolerance", atol=tol.atol, rtol=tol.rtol, **out)


# ---------------------------------------------------------------- A5 mixed batches (coverage gap)
def probe_mixed_batches():
    # varlen batch of chunked-prefill pieces (cu_q != cu_k), each vs its own one-shot rows
    lens = [300, 2100, 70]
    starts = [250, 2040, 0]
    seqs = [inputs(n, 40 + i) for i, n in enumerate(lens)]
    refs = [one_shot(*s, W) for s in seqs]
    q = torch.cat([s[0][a:] for s, a in zip(seqs, starts)])
    gate = torch.cat([s[3][a:] for s, a in zip(seqs, starts)])
    k = torch.cat([s[1] for s in seqs])
    v = torch.cat([s[2] for s in seqs])
    lq = [n - a for n, a in zip(lens, starts)]
    cu_q = i32([0, *torch.tensor(lq).cumsum(0).tolist()])
    cu_k = i32([0, *torch.tensor(lens).cumsum(0).tolist()])
    pos = torch.cat([torch.arange(a, n, dtype=torch.int32, device=DEV) for n, a in zip(lens, starts)])
    y, o, lse = ga.strict_gated_attention_fwd_varlen(q, k, v, gate, cu_q, cu_k, pos, window=W)
    off, bad_varlen = 0, 0
    for r, a, m in zip(refs, starts, lq):
        bad_varlen += ndiff(y[off : off + m], r[0][a:]) + ndiff(lse[off : off + m], r[2][a:])
        off += m
    # paged batch mixing a 50-token chunk, a 1-token decode and a 9-token chunk
    cache = PagedKVCache(num_pages=400, num_seqs=3, max_pages_per_seq=140, device=DEV)
    for i, s in enumerate(seqs):
        cache.append(i, s[1], s[2])
    take = [50, 1, 9]
    qp = torch.cat([s[0][-t:] for s, t in zip(seqs, take)])
    gp = torch.cat([s[3][-t:] for s, t in zip(seqs, take)])
    cu_p = i32([0, *torch.tensor(take).cumsum(0).tolist()])
    bad_paged = {}
    for mode in (None, "token", "head"):
        yp, op_, lp = ga.strict_gated_attention_fwd_paged(
            qp, gp, cache.k_cache, cache.v_cache, cache.block_table, cache.seq_lens, cu_p,
            window=W, tile_mode=mode,
        )
        off, bad = 0, 0
        for r, t in zip(refs, take):
            bad += ndiff(yp[off : off + t], r[0][-t:]) + ndiff(lp[off : off + t], r[2][-t:])
            off += t
        bad_paged[str(mode)] = bad
    emit(
        "A5_mixed_batches_not_in_test_suite",
        varlen_chunked_batch_mismatches=bad_varlen,
        paged_mixed_prefill_decode_mismatches=bad_paged,
    )


# ---------------------------------------------------------------- A6 storage alignment
def probe_alignment():
    n = 300
    q, k, v, gate = inputs(n, 50)
    out = {}
    for window in (None, W):
        ref = one_shot(q, k, v, gate, window)

        def mis(t):
            buf = torch.empty(t.numel() + 1, dtype=t.dtype, device=DEV)
            view = buf[1:].view(t.shape)
            view.copy_(t)
            return view

        qm, km, vm, gm = mis(q), mis(k), mis(v), mis(gate)
        assert qm.is_contiguous() and qm.data_ptr() % 16 == 2
        got = one_shot(qm, km, vm, gm, window)
        out["global" if window is None else "local"] = sum(ndiff(a, b) for a, b in zip(got, ref))
    emit("A6_misaligned_inputs_2byte_offset", element_mismatches=out)


# ---------------------------------------------------------------- A7 bad positions / seq_lens
def probe_bad_positions():
    n = 128
    q, k, v, gate = inputs(n, 60)
    cu = i32([0, n])
    out = {}
    # q_pos beyond the keys provided (caller passed a too-short cu_seqlens_k)
    cu_k = i32([0, 100])
    pos = torch.arange(n, dtype=torch.int32, device=DEV)
    try:
        y, o, lse = ga.strict_gated_attention_fwd_varlen(
            q, k[:100].contiguous(), v[:100].contiguous(), gate, cu, cu_k, pos, window=None
        )
        out["q_pos_ge_len_k"] = {
            "raised": None,
            "rows_100_127_finite": bool(torch.isfinite(y[100:].float()).all().item()),
        }
    except Exception as e:  # noqa: BLE001
        out["q_pos_ge_len_k"] = {"raised": repr(e)}
    # negative position -> no visible key -> 0/0
    pos2 = pos.clone()
    pos2[5] = -3
    try:
        y, o, lse = ga.strict_gated_attention_fwd_varlen(q, k, v, gate, cu, cu, pos2, window=None)
        out["negative_q_pos"] = {
            "raised": None,
            "row5_y_nan": bool(torch.isnan(y[5].float()).any().item()),
            "row5_lse": float(lse[5, 0].item()),
        }
    except Exception as e:  # noqa: BLE001
        out["negative_q_pos"] = {"raised": repr(e)}
    # paged: seq_lens smaller than the number of new queries
    cache = PagedKVCache(num_pages=16, num_seqs=1, max_pages_per_seq=8, device=DEV)
    cache.append(0, k[:20], v[:20])
    sl = cache.seq_lens.clone()
    sl[0] = 10
    try:
        y, o, lse = ga.strict_gated_attention_fwd_paged(
            q[:20].contiguous(), gate[:20].contiguous(), cache.k_cache, cache.v_cache,
            cache.block_table, sl, i32([0, 20]), window=None,
        )
        out["paged_seq_len_lt_len_q"] = {
            "raised": None,
            "rows_with_nan": int(torch.isnan(y.float()).any(dim=1).sum().item()),
        }
    except Exception as e:  # noqa: BLE001
        out["paged_seq_len_lt_len_q"] = {"raised": repr(e)}
    emit("A7_unvalidated_positions", **out)


# ---------------------------------------------------------------- A8 compiled code facts
def probe_compiled_code():
    q, k, v, gate = inputs(70, 70)
    one_shot(q, k, v, gate, W, tile_mode="token")
    one_shot(q, k, v, gate, W, tile_mode="head")
    caches = kern._gated_attention_fwd_kernel.device_caches[torch.cuda.current_device()][0]
    facts = []
    for compiled in caches.values():
        ptx = compiled.asm["ptx"]
        ttgir = compiled.asm["ttgir"]
        # does the PV dot take the rescaled accumulator as its C operand?
        dots = [ln.strip() for ln in ttgir.splitlines() if "tt.dot" in ln]
        facts.append(
            {
                "ex2.approx": len(re.findall(r"ex2\.approx", ptx)),
                "lg2.approx": len(re.findall(r"lg2\.approx", ptx)),
                "div.rn.f32": len(re.findall(r"div\.rn\.f32", ptx)),
                "div.full.f32": len(re.findall(r"div\.full\.f32", ptx)),
                "fma.rn.f32": len(re.findall(r"fma\.rn\.f32", ptx)),
                "mma.sync": len(re.findall(r"mma\.sync", ptx)),
                "tt.dot_lines": [d[:160] for d in dots],
            }
        )
    emit("A8_compiled_code_facts", variants=facts)


# ---------------------------------------------------------------- A9 training forward binding
def probe_training_binding():
    from rl_engine.models.muse_glimmer.ops import gated_attention_autograd as A
    from rl_engine.models.muse_glimmer.runtime import strict_kernels as S

    n = 300
    q, k, v, gate = inputs(n, 90)
    cu = i32([0, n])
    pos = torch.arange(n, dtype=torch.int32, device=DEV)
    ref = one_shot(q, k, v, gate, W)
    y_auto = A.gated_attention(q, k, v, gate, cu, cu, pos, window=W)
    ops = S.resolve_ops()
    emit(
        "A9_training_forward_binding",
        autograd_FORWARD_IMPL=A.FORWARD_IMPL,
        autograd_forward_is_fallback=A._strict_fwd is None,
        autograd_y_mismatch_vs_strict_kernel=ndiff(y_auto, ref[0]),
        megatron_strict_kernels_attention_source=str(ops.sources["gated_attention"]),
    )


if __name__ == "__main__":
    which = sys.argv[1:] or [
        "max_seqlen_undersized",
        "missing_page_in_window",
        "seq_len_past_block_table",
        "window_off_by_one_undetected",
        "mixed_batches",
        "alignment",
        "bad_positions",
        "compiled_code",
        "training_binding",
    ]
    for name in which:
        try:
            globals()[f"probe_{name}"]()
        except Exception as e:  # noqa: BLE001
            emit(name, crashed=repr(e)[:400])
    print("ALL_RESULTS " + json.dumps(RESULTS))
