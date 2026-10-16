"""Reviewer probe (read-only w.r.t. the repo): strict TP1 at HEAD 0bb1a6a.

1. score xrt tokens (micro_batch_tokens 8192) -> bitwise vs committed xrt_megatron_strict_tp1.json
2. B ~ N(0,1e-3) seed 1, surrogate loss, grads at mbt 8192 -> bitwise vs implementer grads_tp1.pt
3. same at mbt 4096 (2 micro-batches) twice -> repeat determinism; logp equality vs 8192; grad diff vs 8192
4. AdamW step at mbt 8192 -> LoRA sha vs claimed 64d30065...
"""
import json
import os
import struct
import sys
import time

import torch

OUT = os.path.dirname(os.path.abspath(__file__))
WT = "/home/ubuntu/RL-Kernel/.claude/worktrees/wf_7b064961-36f-12"
MEG = "/tmp/claude-1000/-home-ubuntu-RL-Kernel--claude-worktrees-swift-oak-hsv7/80655dc5-1eae-4992-a93b-f6f87e8ab1d8/scratchpad/meg"


def bits(t):
    return t.detach().float().cpu().contiguous().view(torch.int32)


def main():
    from rl_engine.models.muse_glimmer.runtime.run_megatron_strict import surrogate_loss, surrogate_weights
    from rl_engine.models.muse_glimmer.runtime.strict_trainer import StrictTrainer

    xrt = json.load(open(f"{WT}/reports/muse_glimmer_poc/data/xrt_tokens.json"))
    seqs = [list(map(int, s["token_ids"])) for s in xrt["sequences"]]
    starts = [int(s["completion_start"]) for s in xrt["sequences"]]
    spans = [len(s) - st for s, st in zip(seqs, starts)]
    ref = json.load(open(f"{WT}/reports/muse_glimmer_poc/data/xrt_megatron_strict_tp1.json"))
    out = {}
    t0 = time.time()
    tr = StrictTrainer(os.path.expanduser("~/models/Muse-Glimmer-30B"), tensor_parallel_size=1, gpu_ids=[0],
                       lora_rank=16, lr=1e-5, strict=True, seed=0, micro_batch_tokens=8192)
    out["init_s"] = time.time() - t0
    out["ops"] = tr.provenance().get("ops")
    lps = tr.score(seqs, starts)
    mism = 0
    for lp, s in zip(lps, ref["sequences"]):
        hx = [f"{v & 0xFFFFFFFF:08x}" for v in bits(lp).tolist()]
        mism += sum(1 for a, b in zip(hx, s["logprobs_hex"]) if a != b)
    out["xrt_vs_committed_mismatches"] = mism
    print("xrt vs committed", mism, flush=True)

    weights = surrogate_weights(spans)
    tr.reset_lora_b(1e-3, 1)

    def grads_once():
        lp = tr.token_logprobs(seqs, starts)
        info = tr.backward(surrogate_loss(lp, weights))
        g = tr.lora_grads()["grads"]
        tr.zero_grad()
        return [x.detach().cpu() for x in lp], g, info

    lp8, g8, info8 = grads_once()
    out["info_8192"] = info8
    ref_g = torch.load(f"{MEG}/grads_tp1.pt", map_location="cpu", weights_only=True)["grads"]
    gm = sum(int((bits(g8[k]) != bits(ref_g[k])).sum()) for k in ref_g)
    out["grads8192_vs_committed_tp1_mismatches"] = gm
    print("grads 8192 vs committed", gm, info8, flush=True)

    tr._state.micro_batch_tokens = 4096
    lp4a, g4a, info4a = grads_once()
    lp4b, g4b, info4b = grads_once()
    out["info_4096"] = [info4a, info4b]
    out["grads4096_repeat_mismatches"] = sum(int((bits(g4a[k]) != bits(g4b[k])).sum()) for k in g4a)
    out["logp4096_vs_8192_mismatches"] = sum(int((bits(a) != bits(b)).sum()) for a, b in zip(lp4a, lp8))
    diff_el = 0
    rel = []
    for k in g8:
        a, b = g8[k].double(), g4a[k].double()
        diff_el += int((bits(g8[k]) != bits(g4a[k])).sum())
        n = a.norm().item()
        rel.append(((a - b).norm().item() / n) if n else 0.0)
    out["grads4096_vs_8192_mismatching_elements"] = diff_el
    out["grads4096_vs_8192_max_rel_l2"] = max(rel)
    print("4096:", {k: out[k] for k in out if "4096" in k}, flush=True)

    tr._state.micro_batch_tokens = 8192
    _, sha0 = tr.export_lora()
    lp = tr.token_logprobs(seqs, starts)
    info = tr.step(surrogate_loss(lp, weights))
    _, sha1 = tr.export_lora()
    out["step_info"] = info
    out["sha_before"] = sha0
    out["sha_after"] = sha1
    out["sha_after_matches_claim"] = sha1 == "64d3006511501f36241ca72a9189341ab90af3e35ca505d0b97f1d9407842ca3"
    print("step", info, sha1, flush=True)
    torch.save({"grads": g8}, f"{OUT}/grads_strict_tp1_head.pt")
    tr.close()
    json.dump(out, open(f"{OUT}/probe_strict_tp1.json", "w"), indent=1, default=str)
    print("done", flush=True)


if __name__ == "__main__":
    sys.exit(main())
