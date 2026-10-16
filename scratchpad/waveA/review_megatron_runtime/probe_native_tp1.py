"""Reviewer probe: native trainer (strict=False) TP1 LoRA grads vs the strict trainer's grads.

Same LoRA init (seed 0), B ~ N(0, 1e-3) seed 1, same surrogate loss as run_megatron_strict's
grad_repeat check. Saves the native grads; the comparison (cosine / rel L2 per tensor) is
done on CPU against the implementer's strict grads_tp1.pt.
"""
import json
import os
import sys
import time

import torch

OUT = os.path.dirname(os.path.abspath(__file__))
WT = "/home/ubuntu/RL-Kernel/.claude/worktrees/wf_7b064961-36f-12"
MEG = "/tmp/claude-1000/-home-ubuntu-RL-Kernel--claude-worktrees-swift-oak-hsv7/80655dc5-1eae-4992-a93b-f6f87e8ab1d8/scratchpad/meg"


def main():
    from rl_engine.models.muse_glimmer.runtime.run_megatron_strict import surrogate_loss, surrogate_weights
    from rl_engine.models.muse_glimmer.runtime.strict_trainer import StrictTrainer

    xrt = json.load(open(f"{WT}/reports/muse_glimmer_poc/data/xrt_tokens.json"))
    seqs = [list(map(int, s["token_ids"])) for s in xrt["sequences"]]
    starts = [int(s["completion_start"]) for s in xrt["sequences"]]
    spans = [len(s) - st for s, st in zip(seqs, starts)]
    out = {}
    t0 = time.time()
    tr = StrictTrainer(os.path.expanduser("~/models/Muse-Glimmer-30B"), tensor_parallel_size=1, gpu_ids=[0],
                       lora_rank=16, lr=1e-5, strict=False, seed=0)
    out["init_s"] = time.time() - t0
    weights = surrogate_weights(spans)
    tr.reset_lora_b(1e-3, 1)
    lp = tr.token_logprobs(seqs, starts)
    info = tr.backward(surrogate_loss(lp, weights))
    out["info"] = info
    g = tr.lora_grads()["grads"]
    torch.save({"grads": g, "lps": [x.detach().cpu() for x in lp]}, f"{OUT}/grads_native_tp1.pt")
    tr.close()
    ref = torch.load(f"{MEG}/grads_tp1.pt", map_location="cpu", weights_only=True)
    per = []
    for k in sorted(ref["grads"]):
        a = ref["grads"][k].double().flatten()
        b = g[k].double().flatten()
        na, nb = a.norm().item(), b.norm().item()
        cos = float((a @ b) / (na * nb)) if na and nb else float("nan")
        per.append({"key": k, "strict_l2": na, "native_l2": nb, "cos": cos,
                    "rel_l2": float((a - b).norm() / na) if na else float("nan")})
    out["per_tensor"] = per
    lps_ref = ref["lps"]
    out["logp_max_abs_native_vs_strict_with_B"] = max(
        float((x.double() - y.double()).abs().max()) for x, y in zip(lp, lps_ref))
    out["strict_total_l2"] = float(sum(p["strict_l2"] ** 2 for p in per) ** 0.5)
    out["native_total_l2"] = float(sum(p["native_l2"] ** 2 for p in per) ** 0.5)
    json.dump(out, open(f"{OUT}/probe_native_tp1.json", "w"), indent=1, default=str)
    print("done", out["strict_total_l2"], out["native_total_l2"], flush=True)


if __name__ == "__main__":
    sys.exit(main())
