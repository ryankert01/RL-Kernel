"""Reviewer probe: FFN backward at T=64 (base raised), lm_head host syncs and decode latency.

Usage: PYTHONPATH=<tree> python probe_ffn_lm.py <label> <out.json>
"""

import inspect
import json
import sys
import time
import warnings

import torch

from rl_engine.models.muse_glimmer.ops import lm_head_softcap_logprob as lm
from rl_engine.models.muse_glimmer.ops import swiglu_ffn as ffn

BF = torch.bfloat16
label, out_path = sys.argv[1], sys.argv[2]
res = {"label": label, "module": ffn.__file__}


def rnd(*s, scale=1.0):
    return (torch.randn(*s, device="cuda") * scale).to(BF)


# FFN backward with weight grads at T=64 and T=300 (real shape)
torch.manual_seed(0)
H, F = 6656, 19968
wg, wu, wd = rnd(F, H, scale=0.02), rnd(F, H, scale=0.02), rnd(H, F, scale=0.02)
for T in (64, 300):
    x = rnd(T, H).requires_grad_(True)
    g, u, d = (t.clone().requires_grad_(True) for t in (wg, wu, wd))
    try:
        y = ffn.swiglu_ffn(x, g, u, d)
        y.backward(rnd(T, H))
        res[f"ffn_bwd_T{T}"] = "ok"
        # dW_down vs explicit zero-padded tokens (ragged rule claim)
    except Exception as exc:  # noqa: BLE001
        res[f"ffn_bwd_T{T}"] = f"{type(exc).__name__}: {str(exc)[:160]}"
del wg, wu, wd
torch.cuda.empty_cache()

# lm_head: real shape
V, VP = lm.VOCAB_SIZE, lm.PADDED_VOCAB_SIZE
torch.manual_seed(1)
w = torch.zeros(VP, H, device="cuda", dtype=BF)
w[:V] = rnd(V, H, scale=0.02)
has_vt = "validate_targets" in inspect.signature(lm.lm_head_softcap_logprob).parameters
res["has_validate_targets"] = has_vt


def count_syncs(fn):
    torch.cuda.synchronize()
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        torch.cuda.set_sync_debug_mode("warn")
        try:
            fn()
        finally:
            torch.cuda.set_sync_debug_mode(0)
    return sum("synchroniz" in str(r.message).lower() for r in rec)


h1 = rnd(1, H)
t1 = torch.tensor([123], device="cuda")
lm.lm_head_softcap_logprob(h1, w, t1)  # warm-up / compile
res["syncs_validated"] = count_syncs(lambda: lm.lm_head_softcap_logprob(h1, w, t1))
if has_vt:
    lm.lm_head_softcap_logprob(h1, w, t1, validate_targets=False)
    res["syncs_unvalidated"] = count_syncs(
        lambda: lm.lm_head_softcap_logprob(h1, w, t1, validate_targets=False)
    )
    bad = torch.tensor([V + 5], device="cuda")
    res["unvalidated_padded_target_logp"] = float(
        lm.lm_head_softcap_logprob(h1, w, bad, validate_targets=False)[0]
    )
    res["unvalidated_negative_target_logp"] = float(
        lm.lm_head_softcap_logprob(h1, w, torch.tensor([-1], device="cuda"),
                                   validate_targets=False)[0]
    )


def timed(fn, n=60):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    t0 = time.perf_counter()
    ev0.record()
    for _ in range(n):
        fn()
    ev1.record()
    torch.cuda.synchronize()
    return ev0.elapsed_time(ev1) / n, (time.perf_counter() - t0) * 1e3 / n


ms_ev, ms_wall = timed(lambda: lm.lm_head_softcap_logprob(h1, w, t1))
res["lm_T1_event_ms"], res["lm_T1_wall_ms"] = ms_ev, ms_wall
json.dump(res, open(out_path, "w"), indent=1)
print(json.dumps(res, indent=1))
