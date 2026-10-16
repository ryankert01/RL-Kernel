"""Reviewer probe (strict reference, triton backend, one GPU): reproduce committed strict numbers
at HEAD and extend continuous-batching decode to 65 sequences (decode GEMM M crosses the
decode16 / decode64 / prefill shape classes).  argv: committed.json tokens.json out.json"""
import json
import os
import sys
import time

import torch

from rl_engine.models.muse_glimmer.ablation import ablation_hf as A
from rl_engine.models.muse_glimmer.ablation import ablation_strict as S
from rl_engine.models.muse_glimmer.reference.strict_model import MuseGlimmerStrictModel
from rl_engine.models.muse_glimmer.reference.weights import MuseGlimmerWeights
from rl_engine.validation.operators.accelerator import disable_tf32

committed = json.load(open(sys.argv[1]))
seqs, _ = A.load_xrt_tokens(sys.argv[2])
out_path = sys.argv[3]
dev = torch.device("cuda:0")
disable_tf32("cuda")
res = {"env": A.env_info(dev)}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def hexes(scores):
    return [A.fp32_hex(s.logp) for s in scores]


def ndiff(a, b):
    return sum(x != y for sa, sb in zip(a, b) for x, y in zip(sa, sb))


w = MuseGlimmerWeights.load_checkpoint(os.path.expanduser("~/models/Muse-Glimmer-30B"),
                                       device=dev, layout="nk")
model = MuseGlimmerStrictModel(w, "triton")
res["ops"] = model.ops.describe()["provenance"]
log("strict model loaded")
base = S.score_alone(model, seqs, "train")
bh = hexes(base)
cb = [p["logp_hex"] for p in committed["baselines"]["strict"]["per_sequence"]]
res["baseline_vs_committed"] = ndiff(bh, cb)
log(f"strict baseline vs committed: {res['baseline_vs_committed']}")

packed = S.score_packed_batch(model, seqs, "packed")
res["packed_vs_base"] = ndiff(hexes(packed), bh)
paged = S.score_paged_batch_prefill(model, seqs)
res["paged_prefill_vs_base"] = ndiff(hexes(paged), bh)
log(f"packed {res['packed_vs_base']} paged {res['paged_prefill_vs_base']}")

# 8 copies of the 8 short sequences + the long one: 65 active rows in the first decode steps
short = [s for s in seqs if s.id != "gsm8k_long_window"]
long = [s for s in seqs if s.id == "gsm8k_long_window"]
big = [A.SeqSpec(f"{s.id}#{k}", s.token_ids, s.completion_start) for k in range(8) for s in short]
big += long
src = {s.id: i for i, s in enumerate(seqs)}
torch.cuda.reset_peak_memory_stats(dev)
t0 = time.perf_counter()
bd, steps = S.score_batched_decode(model, big)
res["batched_decode65_wall_s"] = time.perf_counter() - t0
res["batched_decode65_peak_gib"] = torch.cuda.max_memory_allocated(dev) / 2**30
res["batched_decode65_steps"] = steps
diffs = 0
for s, sc in zip(big, bd):
    b = bh[src[s.id.split("#")[0]]]
    diffs += sum(x != y for x, y in zip(A.fp32_hex(sc.logp), b))
res["batched_decode65_vs_base_tokens_differ"] = diffs
res["batched_decode65_n_tokens"] = sum(s.n_completion for s in big)
log(f"batched decode 65: {diffs}/{res['batched_decode65_n_tokens']} differ, {steps} steps, "
    f"{res['batched_decode65_wall_s']:.1f}s peak {res['batched_decode65_peak_gib']:.1f} GiB")
json.dump(res, open(out_path, "w"), indent=1)
log(f"wrote {out_path}")
