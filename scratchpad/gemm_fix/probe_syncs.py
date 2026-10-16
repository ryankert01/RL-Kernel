"""Count host syncs of the lm_head forward (sync debug warnings + profiler events)."""

import json
import sys
import warnings

import torch

from rl_engine.models.muse_glimmer.ops import lm_head_softcap_logprob as op


def count(fn):
    torch.cuda.synchronize()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        torch.cuda.set_sync_debug_mode("warn")
        try:
            fn()
        finally:
            torch.cuda.set_sync_debug_mode("default")
    return sum("synchronizing" in str(w.message) for w in caught)


def profiler_syncs(fn):
    from torch.profiler import ProfilerActivity, profile

    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    names = [e.name for e in prof.events()]
    return {
        "cudaStreamSynchronize": sum(n == "cudaStreamSynchronize" for n in names),
        "_local_scalar_dense": sum(n == "aten::_local_scalar_dense" for n in names),
    }


g = torch.Generator().manual_seed(0)
H = int(sys.argv[2]) if len(sys.argv) > 2 else 256
h = torch.randn(32, H, generator=g).to(torch.bfloat16).cuda()
w = (torch.randn(1900, H, generator=g) * 0.1).to(torch.bfloat16)
wp = op.pad_lm_head_weight(w, padded_vocab=2048).cuda()
t = torch.randint(0, 1900, (32,), generator=g).cuda()
kw = dict(vocab_size=1900)
res = {}
with torch.no_grad():
    op.lm_head_softcap_logprob(h, wp, t, **kw)
    x = t.clone()
    res["bool_any_single_check"] = count(lambda: bool((x < 0).any()))
    res["forward_validated_warn_count"] = count(lambda: op.lm_head_softcap_logprob(h, wp, t, **kw))
    res["forward_validated_profiler"] = profiler_syncs(
        lambda: op.lm_head_softcap_logprob(h, wp, t, **kw))
    try:
        res["forward_unvalidated_warn_count"] = count(
            lambda: op.lm_head_softcap_logprob(h, wp, t, validate_targets=False, **kw))
        res["forward_unvalidated_profiler"] = profiler_syncs(
            lambda: op.lm_head_softcap_logprob(h, wp, t, validate_targets=False, **kw))
    except TypeError as exc:
        res["forward_unvalidated"] = f"n/a ({exc})"
print(json.dumps(res))
with open(sys.argv[1], "w") as f:
    json.dump(res, f)
