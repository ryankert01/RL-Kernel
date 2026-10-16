"""det_gemm:1 probe: kernel vs combine_tree(8 single-chunk leaves) across ITERS; IR inspection.

Usage: python probe_iters.py <out_json> [<ir_dir>]
"""

import json
import re
import sys
from dataclasses import asdict

import torch
import triton

from rl_engine.models.muse_glimmer.ops import det_gemm as dg
from rl_engine.models.muse_glimmer.ops.canonical_tree import combine_tree
from rl_engine.models.muse_glimmer.triton import det_gemm_tree as kt

torch.backends.cuda.matmul.allow_tf32 = False
DEV = "cuda"
BF = torch.bfloat16
OUT = sys.argv[1]
IR_DIR = sys.argv[2] if len(sys.argv) > 2 else None


def rand(*s, scale=1.0, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    return (torch.randn(*s, device=DEV) * scale).to(BF)


def leaves_tree(a, b_kn, num_chunks=8):
    """Reference: num_chunks single-chunk kernel launches (no in-kernel adds) + torch tree."""
    k = a.size(1)
    chunk = k // num_chunks
    parts = []
    for c in range(num_chunks):
        cols = slice(c * chunk, (c + 1) * chunk)
        parts.append(kt.tree_gemm(a[:, cols], b_kn[cols], out_dtype=torch.float32, num_chunks=1))
    return combine_tree(parts)


def ir_stats(compiled):
    asm = compiled.asm
    ttir = asm.get("ttir", "")
    ttgir = asm.get("ttgir", "")
    ptx = asm.get("ptx", "")
    dots = re.findall(r"tt\.dot (%[\w#]+), (%[\w#]+), (%[\w#]+)", ttir)
    zero_c = 0
    # A dot whose accumulator is a constant zero tensor (arith.constant dense<0.0>)
    zero_names = set(re.findall(r"(%[\w#]+) = arith\.constant dense<0\.000000e\+00>", ttir))
    for _, _, c in dots:
        if c in zero_names:
            zero_c += 1
    return {
        "ttir_scf_for": ttir.count("scf.for"),
        "ttir_tt_dot": ttir.count("tt.dot "),
        "ttir_dot_with_zero_acc": zero_c,
        "ttir_dot_with_nonzero_acc": len(dots) - zero_c,
        "ttir_addf_tensor": len(re.findall(r"arith\.addf %[\w#]+, %[\w#]+ : tensor<", ttir)),
        "ttgir_mma": ttgir.count("triton_gpu.mma") + ttgir.count("ttg.nvidia_mma")
        + ttgir.count("#mma"),
        "ptx_mma_sync": ptx.count("mma.sync"),
        "ptx_add_f32": len(re.findall(r"\badd(\.rn)?\.f32\b", ptx)),
    }


def compile_variant(m, n, k, block_k_override=None, num_chunks=8, cfg=None):
    a = rand(m, k)
    b = rand(k, n, scale=0.02)
    out = torch.empty((m, n), dtype=torch.float32, device=DEV)
    _, c = kt.select_config(m)
    c = cfg or c
    plan = kt.plan_chunks(k, num_chunks, block_k=c.block_k)
    bk = block_k_override or plan.block_k
    grid = (triton.cdiv(m, c.block_m) * triton.cdiv(n, c.block_n),)
    compiled = kt.det_gemm_tree_kernel[grid](
        a, b, out, m, n, k, a.stride(0), a.stride(1), b.stride(0), b.stride(1),
        out.stride(0), out.stride(1), CHUNK_LEN=plan.chunk_len, NUM_CHUNKS=num_chunks,
        BLOCK_M=c.block_m, BLOCK_N=c.block_n, BLOCK_K=bk, GROUP_M=c.group_m,
        EVEN_M=m % c.block_m == 0, EVEN_N=n % c.block_n == 0, MASK_K=False,
        num_warps=c.num_warps, num_stages=c.num_stages,
    )
    return compiled, plan, bk


def main():
    rec = {"forward_sweep": [], "backward_paths": [], "ir": [], "tile_variants": []}
    # 1) forward: kernel (8 chunks) vs 8 single-chunk leaves + torch tree, many K
    for k in (256, 512, 768, 1024, 1536, 2048, 4096, 6656):
        for m in (16, 64, 256):
            a, w = rand(m, k, seed=k + m), rand(384, k, scale=0.02)
            full = dg.fast_det_gemm(a, w, out_dtype=torch.float32)
            plan = kt.last_launch()["plan"]
            ref = leaves_tree(a, w.t())
            nd = int((full != ref).sum().item())
            rec["forward_sweep"].append({
                "M": m, "N": 384, "K": k, "chunk_len": plan["chunk_len"],
                "block_k": plan["block_k"], "iters": plan["chunk_len"] // plan["block_k"],
                "equal_to_tree": bool(nd == 0), "mismatch": nd, "numel": full.numel(),
                "max_abs_diff": float((full - ref).abs().max().item()),
            })
            print(rec["forward_sweep"][-1], flush=True)
    # 2) backward paths at model geometry
    torch.manual_seed(1)
    for name, n_red, k_out in (("k_proj_dA", 256, 6656), ("q_proj_dA", 4096, 6656)):
        for m in (1, 64, 4096):
            w = rand(n_red, k_out, scale=0.02)
            dy = rand(m, n_red)
            da = kt.tree_gemm(dy, w, out_dtype=torch.float32, allow_pad=True)
            plan = kt.last_launch()["plan"]
            ref = leaves_tree(dy, w)
            nd = int((da != ref).sum().item())
            rec["backward_paths"].append({
                "path": name, "M": m, "reduce_len": n_red, "chunk_len": plan["chunk_len"],
                "block_k": plan["block_k"], "iters": plan["chunk_len"] // plan["block_k"],
                "equal_to_tree": nd == 0, "mismatch": nd, "numel": da.numel(),
            })
            print(rec["backward_paths"][-1], flush=True)
    for tokens in (16, 256, 300, 512, 4096):
        x = rand(tokens, 1024)
        dy = rand(tokens, 384)
        dw = kt.tree_gemm(dy.t(), x, out_dtype=torch.float32, allow_pad=True)
        plan = kt.last_launch()["plan"]
        pad = plan["k_pad"]
        dyp = torch.zeros(pad, 384, device=DEV, dtype=BF)
        xp = torch.zeros(pad, 1024, device=DEV, dtype=BF)
        dyp[:tokens], xp[:tokens] = dy, x
        ref = leaves_tree(dyp.t(), xp)
        nd = int((dw != ref).sum().item())
        rec["backward_paths"].append({
            "path": "dW_tokens", "M": tokens, "reduce_len": tokens, "k_pad": pad,
            "chunk_len": plan["chunk_len"], "block_k": plan["block_k"],
            "iters": plan["chunk_len"] // plan["block_k"], "equal_to_tree": nd == 0,
            "mismatch": nd, "numel": dw.numel(),
        })
        print(rec["backward_paths"][-1], flush=True)
    # 3) BLOCK_K variants at K=512 / 256 / 6656
    for k in (256, 512, 6656):
        a, w = rand(64, k, seed=7), rand(256, k, scale=0.02)
        ref = leaves_tree(a, w.t())
        for bk in (32, 64):
            if (k // 8) % bk:
                continue
            cfg = kt.TileConfig(64, 64, bk, 4, 3)
            plan = kt.plan_chunks(k, block_k=bk)
            if plan.block_k != bk:
                continue
            y = kt.tree_gemm(a, w.t(), out_dtype=torch.float32, config=cfg)
            nd = int((y != ref).sum().item())
            rec["tile_variants"].append({
                "K": k, "block_k": bk, "iters": (k // 8) // bk, "equal_to_tree": nd == 0,
                "mismatch": nd, "numel": y.numel(),
            })
            print(rec["tile_variants"][-1], flush=True)
    # 4) IR inspection
    for k, bk in ((256, None), (512, None), (512, 32), (4096, None), (256, 16), (512, 16)):
        try:
            compiled, plan, used_bk = compile_variant(64, 256, k, block_k_override=bk,
                                                      cfg=kt.TileConfig(64, 64, 64, 4, 3))
        except Exception as exc:  # noqa: BLE001
            rec["ir"].append({"K": k, "block_k": bk, "error": repr(exc)[:300]})
            continue
        st = ir_stats(compiled)
        st.update({"K": k, "chunk_len": plan.chunk_len, "block_k": used_bk,
                   "iters": plan.chunk_len // used_bk})
        rec["ir"].append(st)
        print(st, flush=True)
        if IR_DIR:
            for key in ("ttir", "ttgir", "ptx"):
                with open(f"{IR_DIR}/K{k}_bk{used_bk}.{key}", "w") as f:
                    f.write(compiled.asm[key])
    rec["env"] = {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
                  "triton": triton.__version__}
    with open(OUT, "w") as f:
        json.dump(rec, f, indent=1)


if __name__ == "__main__":
    main()
