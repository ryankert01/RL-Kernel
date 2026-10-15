# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
# ruff: noqa: E501  (kept as run in wave A; lines are measurement plumbing)
# Wave A probes of the 5 review findings for O4 qknorm_scaled_rope (qknorm_rope:1..5).
# Writes one JSON object. Run from the repo root (or with PYTHONPATH=<repo>) on one GPU:
#   python probe_findings.py <which: f1|f2|f3|f4|f5|f5b|f5c|xenv> <out.json>
# f5/f5b/f5c allocate ~20-30 GiB. Against an old tree: PYTHONPATH=<old tree>.
from __future__ import annotations

import hashlib
import json
import sys
import time

import torch

from rl_engine.models.muse_glimmer.ops import qknorm_rope as op

DEV = torch.device("cuda")
N_MAX = 131072  # max_position_embeddings
HQ, HK, D = 32, 2, 128


def sha(*ts):
    h = hashlib.sha256()
    for t in ts:
        h.update(t.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def make_qk(T, seed, scale=2.0, hq=HQ, hk=HK):
    g = torch.Generator(device="cpu").manual_seed(seed)
    q = (torch.randn(T, hq, D, generator=g) * scale).to(torch.bfloat16).to(DEV)
    k = (torch.randn(T, hk, D, generator=g) * scale).to(torch.bfloat16).to(DEV)
    return q, k


def env():
    import triton

    return {
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "triton": triton.__version__,
        "python": sys.executable,
    }


def f1():
    out = {}
    pos = torch.arange(N_MAX)
    cc, sc = op.rope_table(pos.to(DEV), device=DEV)
    # CPU build accepted?
    try:
        c_cpu, s_cpu = op.rope_table(pos, device="cpu")
        out["cpu_build_accepted"] = True
    except Exception as e:  # noqa: BLE001
        out["cpu_build_accepted"] = False
        out["cpu_build_error"] = f"{type(e).__name__}: {e}"
        c_cpu = s_cpu = None
    if c_cpu is not None:
        dc = c_cpu.to(DEV) != cc
        ds = s_cpu.to(DEV) != sc
        out["cpu_vs_cuda_table"] = {
            "positions": N_MAX,
            "elements": cc.numel(),
            "cos_mismatch": int(dc.sum()),
            "sin_mismatch": int(ds.sum()),
            "cos_max_abs": float((c_cpu.to(DEV) - cc).abs().max()),
            "sin_max_abs": float((s_cpu.to(DEV) - sc).abs().max()),
        }
        # per position bucket series (2048-position buckets) for plotting
        nb = N_MAX // 2048
        out["cpu_vs_cuda_table"]["bucket_size"] = 2048
        out["cpu_vs_cuda_table"]["cos_mismatch_frac_per_bucket"] = [
            float(x) for x in dc.view(nb, -1).float().mean(1).tolist()
        ]
        out["cpu_vs_cuda_table"]["sin_mismatch_frac_per_bucket"] = [
            float(x) for x in ds.view(nb, -1).float().mean(1).tolist()
        ]
        # first 4096 positions only (contract T)
        out["cpu_vs_cuda_table"]["first4096"] = {
            "cos_mismatch": int(dc[:4096].sum()),
            "sin_mismatch": int(ds[:4096].sum()),
            "elements": 4096 * D,
        }
        exp_c = torch.arange(0, D, 2, dtype=torch.float32, device=DEV) / D
        exp_h = torch.arange(0, D, 2, dtype=torch.float32) / D
        inv_c = 1.0 / torch.pow(op.ROPE_THETA, exp_c)
        inv_h = 1.0 / torch.pow(op.ROPE_THETA, exp_h)
        neq = (inv_h.to(DEV) != inv_c).nonzero().flatten().tolist()
        out["cpu_vs_cuda_inv_freq_mismatch_channels"] = neq
        # effect on the op output at T=4096 (contract shape) and at the long-context tail
        for name, lo in (("pos_0_4096", 0), ("pos_126976_131072", N_MAX - 4096)):
            q, k = make_qk(4096, seed=3)
            rows = slice(lo, lo + 4096)
            a = op.qknorm_scaled_rope(q, k, cc[rows], sc[rows])
            b = op.qknorm_scaled_rope(q, k, c_cpu[rows].to(DEV), s_cpu[rows].to(DEV))
            out[f"op_flips_cpu_table_{name}"] = {
                "q_hat_mismatch": int((a[0] != b[0]).sum()),
                "q_hat_elements": a[0].numel(),
                "k_hat_mismatch": int((a[1] != b[1]).sum()),
                "k_hat_elements": a[1].numel(),
            }
    # cross-implementation identity on CUDA (other table builders in the repo)
    impls = {}
    from rl_engine.models.muse_glimmer.reference import golden_ops
    from rl_engine.models.muse_glimmer.runtime import fallback_ops, strict_kernels

    cands = {
        "reference.golden_ops.rope_table": lambda p: golden_ops.rope_table(
            p, theta=op.ROPE_THETA, dim=D, device=DEV
        ),
        "runtime.fallback_ops.rope_table": lambda p: fallback_ops.rope_table(
            p, theta=op.ROPE_THETA, dim=D, device=DEV
        ),
        "runtime.strict_kernels._fb_rope_table": lambda p: strict_kernels._fb_rope_table(
            p, theta=op.ROPE_THETA, dim=D, device=DEV
        ),
        "ops.qknorm_rope.rope_table(int32 positions)": lambda p: op.rope_table(
            p.to(torch.int32), device=DEV
        ),
    }
    for name, fn in cands.items():
        c, s = fn(pos.to(DEV))
        impls[name] = {
            "cos_mismatch": int((c != cc).sum()),
            "sin_mismatch": int((s != sc).sum()),
            "elements": cc.numel(),
        }
    out["cuda_cross_impl_vs_ops_rope_table"] = impls
    out["canonical_table_sha256"] = {"positions": N_MAX, "sha256(cos||sin)": sha(cc, sc)}
    return out


def f2():
    out = {}
    T = 64
    q, k = make_qk(T, seed=1)
    pos = torch.arange(T, device=DEV)
    cos, sin = op.rope_table(pos, device=DEV)
    for dt in (torch.float32, torch.float16):
        rec = {}
        try:
            a = op.qknorm_scaled_rope(q.to(dt), k.to(dt), cos, sin)
            rec["accepted"] = True
            rec["out_dtype"] = str(a[0].dtype)
            ref = op.qknorm_scaled_rope(q, k, cos, sin)
            rec["mismatch_vs_bf16_path_after_bf16_cast"] = int(
                (a[0].to(torch.bfloat16) != ref[0]).sum()
            )
            rec["mismatch_vs_bf16_path_without_cast_fp32_compare"] = int(
                (a[0].float() != ref[0].float()).sum()
            )
            qg = q.to(dt).clone().requires_grad_(True)
            kg = k.to(dt).clone().requires_grad_(True)
            oq, ok = op.qknorm_scaled_rope(qg, kg, cos, sin)
            dq, dk = torch.autograd.grad((oq, ok), (qg, kg), (torch.ones_like(oq), torch.ones_like(ok)))
            rec["grad_dtype"] = str(dq.dtype)
        except Exception as e:  # noqa: BLE001
            rec["accepted"] = False
            rec["error"] = f"{type(e).__name__}: {e}"
        out[str(dt)] = rec
    return out


def _shard_cases():
    for t in (2, 4, 8):
        hq = HQ // t
        for r in range(t):
            kv = (r * hq) // 16  # GQA: q head h reads kv head h // 16
            yield t, r, slice(r * hq, (r + 1) * hq), kv


def f3():
    out = {"cases": []}
    for T in (1, 7, 4096):
        q, k = make_qk(T, seed=10 + T)
        pos = torch.arange(T, device=DEV) + 100
        cos, sin = op.rope_table(pos, device=DEV)
        g = torch.Generator(device="cpu").manual_seed(99)
        dyq = torch.randn(q.shape, generator=g).to(torch.bfloat16).to(DEV)
        dyk = torch.randn(k.shape, generator=g).to(torch.bfloat16).to(DEV)
        for local in (True, False):
            c, s = (cos, sin) if local else (None, None)
            qg, kg = q.clone().requires_grad_(True), k.clone().requires_grad_(True)
            fq, fk = op.qknorm_scaled_rope(qg, kg, c, s)
            fdq, fdk = torch.autograd.grad((fq, fk), (qg, kg), (dyq, dyk))
            for t, r, qs, kv in _shard_cases():
                q_r = q[:, qs].contiguous().requires_grad_(True)
                k_r = k[:, kv : kv + 1].contiguous().requires_grad_(True)
                oq, ok = op.qknorm_scaled_rope(q_r, k_r, c, s)
                dq, dk = torch.autograd.grad(
                    (oq, ok), (q_r, k_r), (dyq[:, qs].contiguous(), dyk[:, kv : kv + 1].contiguous())
                )
                # dK of a replicated KV head: each rank's dk is the grad through its own k_hat;
                # with the same dy for k it equals the full call's dk slice.
                out["cases"].append(
                    {
                        "T": T,
                        "local": local,
                        "tp": t,
                        "rank": r,
                        "q_heads": [qs.start, qs.stop],
                        "kv_head": kv,
                        "BLOCK_HQ": op.tile_config(HQ // t, 1)["BLOCK_HQ"],
                        "BLOCK_HK": op.tile_config(HQ // t, 1)["BLOCK_HK"],
                        "q_hat_mismatch": int((oq != fq[:, qs]).sum()),
                        "k_hat_mismatch": int((ok != fk[:, kv : kv + 1]).sum()),
                        "dq_mismatch": int((dq != fdq[:, qs]).sum()),
                        "dk_mismatch": int((dk != fdk[:, kv : kv + 1]).sum()),
                        "q_elements": oq.numel(),
                        "k_elements": ok.numel(),
                    }
                )
    out["total_mismatch"] = sum(
        c["q_hat_mismatch"] + c["k_hat_mismatch"] + c["dq_mismatch"] + c["dk_mismatch"]
        for c in out["cases"]
    )
    return out


def f4():
    T = 4096
    q, k = make_qk(T, seed=5)
    cos, sin = op.rope_table(torch.arange(T, device=DEV), device=DEV)
    a = op.qknorm_scaled_rope(q, k, cos, sin)
    try:
        b = op.qknorm_scaled_rope(q, k, None, None)
        res = {"none_on_local_layer_raised": False}
        res["q_hat_mismatch_vs_rope"] = int((a[0] != b[0]).sum())
        res["k_hat_mismatch_vs_rope"] = int((a[1] != b[1]).sum())
        res["elements"] = a[0].numel() + a[1].numel()
    except Exception as e:  # noqa: BLE001
        res = {"none_on_local_layer_raised": True, "error": f"{type(e).__name__}: {e}"}
    # explicit flag available?
    try:
        op.qknorm_scaled_rope(q, k, None, None, local=True)
        res["explicit_local_flag"] = "accepted_without_error"
    except TypeError as e:
        res["explicit_local_flag"] = f"TypeError: {e}"
    except ValueError as e:
        res["explicit_local_flag"] = f"ValueError (fails closed): {e}"
    return res


def f5():
    """T*HQ*D >= 2^31: first overflowing token row is t = 2^31 / (32*128) = 524288."""
    T = 524288 + 64
    res = {"T": T, "numel_q": T * HQ * D, "int32_max": 2**31 - 1}
    torch.manual_seed(0)
    q = (torch.randn(T, HQ, D, device=DEV) * 2).to(torch.bfloat16)
    k = (torch.randn(T, HK, D, device=DEV) * 2).to(torch.bfloat16)
    check = list(range(524288 - 64, T))
    for local in (False, True):
        tag = "local" if local else "global"
        try:
            if local:
                cos, sin = op.rope_table(torch.arange(T, device=DEV) % 131072, device=DEV)
            else:
                cos = sin = None
            oq, ok = op.qknorm_scaled_rope(q, k, cos, sin)
            torch.cuda.synchronize()
            rq, rk = op.qknorm_scaled_rope(
                q[check].contiguous(),
                k[check].contiguous(),
                None if cos is None else cos[check].contiguous(),
                None if sin is None else sin[check].contiguous(),
            )
            torch.cuda.synchronize()
            bad_rows = [
                t for i, t in enumerate(check) if not torch.equal(oq[t], rq[i])
            ]
            res[tag] = {
                "raised": False,
                "rows_checked": len(check),
                "rows_checked_range": [check[0], check[-1]],
                "rows_mismatch_q": len(bad_rows),
                "first_bad_row": bad_rows[0] if bad_rows else None,
                "rows_below_2p31_mismatch": len([t for t in bad_rows if t < 524288]),
                "rows_at_or_above_2p31_mismatch": len([t for t in bad_rows if t >= 524288]),
            }
            # backward
            qg = q.clone().requires_grad_(True)
            kg = k.clone().requires_grad_(True)
            o = op.qknorm_scaled_rope(qg, kg, cos, sin)
            dq, dk = torch.autograd.grad(o, (qg, kg), (torch.ones_like(o[0]), torch.ones_like(o[1])))
            torch.cuda.synchronize()
            sq = q[check].contiguous().requires_grad_(True)
            sk = k[check].contiguous().requires_grad_(True)
            so = op.qknorm_scaled_rope(
                sq, sk,
                None if cos is None else cos[check].contiguous(),
                None if sin is None else sin[check].contiguous(),
            )
            sdq, _ = torch.autograd.grad(
                so, (sq, sk), (torch.ones_like(so[0]), torch.ones_like(so[1]))
            )
            bad = [t for i, t in enumerate(check) if not torch.equal(dq[t], sdq[i])]
            res[tag]["bwd_rows_mismatch_dq"] = len(bad)
            del qg, kg, o, dq, dk
        except Exception as e:  # noqa: BLE001
            res[tag] = {"raised": True, "error": f"{type(e).__name__}: {str(e)[:400]}"}
            break
    res["peak_mem_gib"] = torch.cuda.max_memory_allocated() / 2**30
    return res


def f5b():
    """Same overflow, but with a 6 GiB tensor allocated just before q so the wrapped
    (X - 4 GiB) addresses land in mapped memory: is the failure then silent?"""
    T = 524288 + 64
    res = {"T": T}
    pad = torch.zeros(6 * 2**30 // 2, dtype=torch.bfloat16, device=DEV)
    q = torch.randn(T, HQ, D, device=DEV, dtype=torch.bfloat16)
    k = torch.randn(T, HK, D, device=DEV, dtype=torch.bfloat16)
    res["addr_gap_q_minus_pad_end_MiB"] = (
        q.data_ptr() - (pad.data_ptr() + pad.numel() * 2)
    ) / 2**20
    q_head_before = q[:1024].clone()
    check = list(range(524288 - 64, T))
    ref = op.qknorm_scaled_rope(q[check].contiguous(), k[check].contiguous(), None, None)
    torch.cuda.synchronize()
    try:
        oq, ok = op.qknorm_scaled_rope(q, k, None, None)
        torch.cuda.synchronize()
        res["raised"] = False
        res["oq_minus_q_addr_GiB"] = (oq.data_ptr() - q.data_ptr()) / 2**30
        bad = [t for i, t in enumerate(check) if not torch.equal(oq[t], ref[0][i])]
        res["rows_checked_range"] = [check[0], check[-1]]
        res["rows_mismatch_q"] = len(bad)
        res["first_bad_row"] = bad[0] if bad else None
        res["pad_elements_modified"] = int((pad != 0).sum())
        res["q_input_rows_0_1024_modified_elements"] = int((q[:1024] != q_head_before).sum())
    except Exception as e:  # noqa: BLE001
        res["raised"] = True
        res["error"] = f"{type(e).__name__}: {str(e)[:300]}"
    return res


def f5c():
    """Overflow with an allocation layout that maps the wrapped addresses (X - 4 GiB) for both
    the input and the output, to test whether the failure can be SILENT.  The caching allocator
    hands out large blocks at descending addresses, so: tmp (future output block), pad1 (below
    tmp), q, pad2 (below q); then free tmp so the op's output reuses it."""
    T = 524288 + 64
    res = {"T": T}
    nq = T * HQ * D
    tmp = torch.empty(nq, dtype=torch.bfloat16, device=DEV)
    pad1 = torch.zeros(6 * 2**30 // 2, dtype=torch.bfloat16, device=DEV)
    q = torch.randn(T, HQ, D, device=DEV, dtype=torch.bfloat16)
    pad2 = torch.full((6 * 2**30 // 2,), 2.0, dtype=torch.bfloat16, device=DEV)
    k = torch.randn(T, HK, D, device=DEV, dtype=torch.bfloat16)
    tmp_addr = tmp.data_ptr()
    res["addr_GiB"] = {
        "tmp": tmp_addr / 2**30,
        "pad1": pad1.data_ptr() / 2**30,
        "q": q.data_ptr() / 2**30,
        "pad2": pad2.data_ptr() / 2**30,
    }
    del tmp
    check = list(range(524288 - 64, T))
    ref = op.qknorm_scaled_rope(q[check].contiguous(), k[check].contiguous(), None, None)
    torch.cuda.synchronize()
    try:
        oq, ok = op.qknorm_scaled_rope(q, k, None, None)
        torch.cuda.synchronize()
        res["raised"] = False
        res["oq_reused_tmp_block"] = oq.data_ptr() == tmp_addr
        bad = [t for i, t in enumerate(check) if not torch.equal(oq[t], ref[0][i])]
        res["rows_checked_range"] = [check[0], check[-1]]
        res["rows_mismatch_q"] = len(bad)
        res["first_bad_row"] = bad[0] if bad else None
        res["rows_below_2p31_mismatch"] = len([t for t in bad if t < 524288])
        res["pad1_elements_modified"] = int((pad1 != 0).sum())
        res["pad2_elements_modified"] = int((pad2 != 2.0).sum())
    except Exception as e:  # noqa: BLE001
        res["raised"] = True
        res["error"] = f"{type(e).__name__}: {str(e)[:300]}"
    return res


def xenv():
    """Cross-runtime identity: table + op outputs fingerprinted (run in both envs)."""
    pos = torch.arange(N_MAX, device=DEV)
    cc, sc = op.rope_table(pos, device=DEV)
    res = {"table_sha256": sha(cc, sc)}
    T = 4096
    q, k = make_qk(T, seed=7)
    g = torch.Generator(device="cpu").manual_seed(8)
    dyq = torch.randn(q.shape, generator=g).to(torch.bfloat16).to(DEV)
    dyk = torch.randn(k.shape, generator=g).to(torch.bfloat16).to(DEV)
    res["inputs_sha256"] = sha(q, k, dyq, dyk)
    for local in (True, False):
        c, s = (cc[:T], sc[:T]) if local else (None, None)
        qg, kg = q.clone().requires_grad_(True), k.clone().requires_grad_(True)
        oq, ok = op.qknorm_scaled_rope(qg, kg, c, s)
        dq, dk = torch.autograd.grad((oq, ok), (qg, kg), (dyq, dyk))
        res["local" if local else "global"] = {"fwd_sha256": sha(oq, ok), "bwd_sha256": sha(dq, dk)}
    # long-context rows
    c, s = cc[N_MAX - T :], sc[N_MAX - T :]
    oq, ok = op.qknorm_scaled_rope(q, k, c, s)
    res["local_pos_126976_131072_fwd_sha256"] = sha(oq, ok)
    return res


if __name__ == "__main__":
    which, path = sys.argv[1], sys.argv[2]
    t0 = time.time()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    res = {"probe": which, "env": env()}
    res.update(globals()[which]())
    res["wall_s"] = round(time.time() - t0, 2)
    with open(path, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1)[:6000])
