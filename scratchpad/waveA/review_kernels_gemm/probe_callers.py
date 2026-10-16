"""Reviewer probe: bits of GEMM call patterns used by OTHER modules (not in the bit ledger).

* megatron_lora._token_grad_gemm -> strict_kernels.gemm_fp32 -> fast_det_gemm_partial(ids 0..7)
  with K = T padded to a multiple of 512 (T <= 512 -> K = 512, chunk 64).
* vllm_strict._lora_gemm(..., k_chunks=1) for LoRA B with K = rank (32 / 64).
Usage: PYTHONPATH=<tree> python probe_callers.py <out.json>
"""

import hashlib
import json
import sys

import torch

from rl_engine.models.muse_glimmer.ops import det_gemm as dg

BF = torch.bfloat16


def h16(t):
    return hashlib.sha256(t.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()[:16]


def rnd(*s, scale=1.0):
    return (torch.randn(*s, device="cuda") * scale).to(BF)


res = {"module": dg.__file__}
for t_pad in (512, 1024):
    torch.manual_seed(t_pad)
    dz_t = rnd(16, t_pad)  # (r, T_pad): LoRA dA = dz^T x
    x_t = rnd(6656, t_pad)
    res[f"megatron_token_grad_Tpad{t_pad}"] = h16(dg.fast_det_gemm_partial(dz_t, x_t, list(range(8))))
for r in (32, 64):
    torch.manual_seed(r)
    z = rnd(64, r)
    b = rnd(4096, r, scale=0.02)
    res[f"vllm_lora_b_r{r}_kchunks1"] = h16(dg.fast_det_gemm(z, b, k_chunks=1))
json.dump(res, open(sys.argv[1], "w"), indent=1)
print(json.dumps(res, indent=1))
