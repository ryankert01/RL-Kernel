"""Verifier probe (norms:R1): kernel inv vs golden inv, old golden (ss / float(H)) vs IEEE golden
(div_ieee), and the bf16 flip counts, for the norm-input distribution and the embedding table
distribution of the owner's mismatch probe. Also CPU vs CUDA equality of the golden."""
import json
import sys

import torch

from rl_engine.models.muse_glimmer.ops import norms as N
from rl_engine.models.muse_glimmer.triton import norms as K

DEV = torch.device("cuda")
H = N.HIDDEN
EPS = N.RMS_NORM_EPS


def bf16(shape, seed, scale=1.0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    return (torch.randn(*shape, device=DEV, generator=g) * scale).to(torch.bfloat16)


def inv_old(x32):
    ms = N._sum_squares_fp32(x32) / float(x32.shape[-1])
    return 1.0 / torch.sqrt(ms + EPS)


def inv_exact_from(ss):
    ms = (ss.double() / float(H)).float()  # correctly rounded fp32 division
    return 1.0 / torch.sqrt(ms + EPS)


out = {}
for name, rows in (
    ("norm_input_scale2", bf16((4096, H), 4106, 2.0)),
    ("embedding_rows_scale0.02", None),
):
    if rows is None:
        table = bf16((50000, H), 7, 0.02)
        gen = torch.Generator(device=DEV).manual_seed(8)
        ids = torch.randint(0, 50000, (4096,), device=DEV, generator=gen)
        y_k, inv_k = K.norm_forward(table, None, eps=EPS, form=K.FORM_WEIGHTLESS, token_ids=ids)
        rows = table.index_select(0, ids)
    else:
        y_k, inv_k = K.norm_forward(rows, None, eps=EPS, form=K.FORM_WEIGHTLESS)
    x32 = rows.float()
    ss = N._sum_squares_fp32(x32)
    i_old = inv_old(x32)
    i_new = N.inv_rms_fp32(x32, EPS)
    i_exact = inv_exact_from(ss)
    y_old = (x32 * i_old[:, None]).to(torch.bfloat16)
    y_new = (x32 * i_new[:, None]).to(torch.bfloat16)
    rec = {
        "rows": int(rows.shape[0]),
        "golden_new_equals_exact_div_rows": int((i_new == i_exact).sum()),
        "golden_old_equals_exact_div_rows": int((i_old == i_exact).sum()),
        "kernel_vs_old_inv_mismatch_rows": int((inv_k != i_old).sum()),
        "kernel_vs_new_inv_mismatch_rows": int((inv_k != i_new).sum()),
        "kernel_vs_old_y_flips": int((y_k != y_old).sum()),
        "kernel_vs_new_y_flips": int((y_k != y_new).sum()),
    }
    # CPU golden equals CUDA golden with div_ieee
    i_cpu = N.inv_rms_fp32(x32.cpu(), EPS)
    ss_cpu = N._sum_squares_fp32(x32.cpu())
    rec["ss_cpu_equals_cuda_rows"] = int((ss_cpu == ss.cpu()).sum())
    same_ss = ss_cpu == ss.cpu()
    rec["inv_cpu_equals_cuda_rows_where_ss_equal"] = int(((i_cpu == i_new.cpu()) & same_ss).sum())
    out[name] = rec
print(json.dumps(out, indent=1))
json.dump(out, open(sys.argv[1], "w"), indent=1)
