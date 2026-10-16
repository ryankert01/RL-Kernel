"""Reviewer probe (real weights, triton backend): xrt re-score repeatability, batched decode at
B=256 (ragged short contexts) and B=4 across the window with eviction, ties to the train
forward, T=8192 train == prefill == chunked, T=2 edge, decode_batch API edge cases."""
import json
import pathlib
import sys
import time

import torch

WT = pathlib.Path(sys.argv[1])
OUT = pathlib.Path(sys.argv[2])
CK = str(pathlib.Path("~/models/Muse-Glimmer-30B").expanduser())

from rl_engine.models.muse_glimmer.reference.strict_model import MuseGlimmerStrictModel, Session  # noqa
from rl_engine.models.muse_glimmer.reference.weights import MuseGlimmerWeights  # noqa
from rl_engine.models.muse_glimmer.reference.xrt_scoring import score_token_file  # noqa
from rl_engine.validation.operators.accelerator import disable_tf32  # noqa

res = {}


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def bits(t):
    return t.detach().float().contiguous().cpu().view(torch.int32)


def beq(a, b):
    return bool(torch.equal(bits(a), bits(b)))


RESCORE = len(sys.argv) > 3 and sys.argv[3] == "rescore"
dev = torch.device("cuda:0")
disable_tf32("cuda")
t0 = time.perf_counter()
w = MuseGlimmerWeights.load_checkpoint(CK, device=dev, layout="nk")
model = MuseGlimmerStrictModel(w, "triton")
log(f"loaded {time.perf_counter() - t0:.1f}s; golden nodes {model.ops.golden_nodes()}")

# 1. re-score the xrt token file: train hex vs committed file, plus one check mode
xrt = WT / "reports/muse_glimmer_poc/data/xrt_tokens.json"
t1 = time.perf_counter()
sc = score_token_file(model, xrt, check_modes=("prefill", "chunked:window_boundary_2048")) if RESCORE else None
RESCORE_DONE = sc is not None
committed = json.loads((WT / "reports/muse_glimmer_poc/data/xrt_reference_triton.json").read_text())
cm = {s["id"]: s["logprobs_hex"] for s in committed["sequences"]}
eq = n = 0
for s in (sc["sequences"] if sc else []):
    h = cm[s["id"]]
    eq += sum(a == b for a, b in zip(s["logprobs_hex"], h))
    n += len(h)
res["xrt_rescore"] = {"equal_to_committed": eq, "n": n, "seconds": time.perf_counter() - t1,
                      "checks_all_equal": sc["checks"]["all_byte_equal_to_train"] if sc else None,
                      "modes": {r["id"]: {m: v.get("equal_tokens", "skip") for m, v in r["modes"].items()}
                                for r in (sc["checks"]["sequences"] if sc else [])}}
log(f"xrt rescore: {eq}/{n} equal to committed")

# 2. B=256 batched decode, ragged short contexts
gen = torch.Generator().manual_seed(123)
B = 256
lens = [17 + (b * 37) % 113 for b in range(B)]
prompts = [torch.randint(0, 200000, (L,), generator=gen) for L in lens]
cache = model.new_batch_cache(max(lens) + 4, B)
t1 = time.perf_counter()
for b in range(B):
    model.prefill(Session(cache=cache, seq=b), prompts[b])
torch.cuda.synchronize()
log(f"prefilled {B} seqs in {time.perf_counter() - t1:.1f}s")
toks = torch.randint(0, 200000, (B,), generator=gen)
tgts = torch.randint(0, 200000, (B,), generator=gen)
t1 = time.perf_counter()
full = model.decode_batch(cache, list(range(B)), toks, tgts, commit=False).cpu()
log(f"B=256 step {time.perf_counter() - t1:.2f}s")
probe = [0, 1, 2, 63, 64, 127, 128, 200, 254, 255]
single = {b: model.decode_batch(cache, [b], toks[b:b + 1], tgts[b:b + 1], commit=False).cpu()
          for b in probe}
r2 = {}
r2["b1_vs_b256"] = sum(beq(single[b], full[b:b + 1]) for b in probe)
rev = list(range(B))[::-1]
fr = model.decode_batch(cache, rev, toks[rev], tgts[rev], commit=False).cpu()
r2["reversed_b256_vs_b256"] = int((bits(fr) == bits(full[rev])).sum())
for sub in ([255, 0], [5, 200, 17], list(range(1, 18)), [b for b in range(B) if b != 5]):
    o = model.decode_batch(cache, sub, toks[sub], tgts[sub], commit=False).cpu()
    r2[f"sub_len{len(sub)}"] = int((bits(o) == bits(full[sub])).sum())
# tie to the train forward for 3 sequences
tie = 0
for b in (0, 128, 255):
    lp = model.logprobs(prompts[b], torch.tensor([int(toks[b]), int(tgts[b])]), mode="train").cpu()
    tie += beq(lp[1:2], full[b:b + 1])
r2["b256_vs_train_forward"] = tie
# targets=None and duplicate seqs (API edge cases)
try:
    o = model.decode_batch(cache, [0, 1], toks[:2], None, commit=False)
    r2["targets_none"] = f"ok: {o.tolist()}"
except Exception as exc:  # noqa
    r2["targets_none"] = f"raised {type(exc).__name__}: {str(exc)[:200]}"
try:
    o = model.decode_batch(cache, [3, 3], [int(toks[3]), 77], [int(tgts[3]), int(tgts[3])],
                           commit=False).cpu()
    r2["duplicate_seq"] = {"raised": False, "row0_equals_single": beq(o[:1], full[3:4]),
                           "values": o.tolist(), "single": full[3:4].tolist()}
except Exception as exc:  # noqa
    r2["duplicate_seq"] = {"raised": True, "err": f"{type(exc).__name__}: {str(exc)[:200]}"}
log(f"B=256 part 1: {r2}")
# commit the full step, then step 2 at B=256 vs B=1
model.decode_batch(cache, list(range(B)), toks, tgts, commit=True)
toks2 = torch.randint(0, 200000, (B,), generator=gen)
tgts2 = torch.randint(0, 200000, (B,), generator=gen)
full2 = model.decode_batch(cache, list(range(B)), toks2, tgts2, commit=False).cpu()
r2["step2_b1_vs_b256"] = sum(
    beq(model.decode_batch(cache, [b], toks2[b:b + 1], tgts2[b:b + 1], commit=False).cpu(),
        full2[b:b + 1]) for b in probe)
lp = model.logprobs(prompts[7], torch.tensor([int(toks[7]), int(toks2[7]), int(tgts2[7])]),
                    mode="train").cpu()
r2["step2_vs_train_seq7"] = beq(lp[2:3], full2[7:8])
res["decode_b256"] = r2
log(f"B=256: {r2}")
del cache
torch.cuda.empty_cache()

# 3. B=4 across the window, with page eviction
lens4 = [2100, 2047, 2200, 2064]
pr4 = [torch.randint(0, 200000, (L,), generator=gen) for L in lens4]
cache = model.new_batch_cache(max(lens4) + 8, 4)
for b in range(4):
    model.prefill(Session(cache=cache, seq=b), pr4[b])
r3 = {"evictions_after_prefill": cache.evictions}
hist = [[] for _ in range(4)]
outs = [[] for _ in range(4)]
for step in range(3):
    tk = torch.randint(0, 200000, (4,), generator=gen)
    tg = torch.randint(0, 200000, (4,), generator=gen)
    f4 = model.decode_batch(cache, [0, 1, 2, 3], tk, tg, commit=False).cpu()
    s4 = torch.cat([model.decode_batch(cache, [b], tk[b:b + 1], tg[b:b + 1], commit=False).cpu()
                    for b in range(4)])
    r3[f"step{step}_b1_vs_b4"] = int((bits(f4) == bits(s4)).sum())
    model.decode_batch(cache, [2, 0, 3, 1], tk[[2, 0, 3, 1]], tg[[2, 0, 3, 1]], commit=True)
    for b in range(4):
        hist[b].append(int(tk[b]))
        outs[b].append((int(tg[b]), f4[b:b + 1]))
r3["evictions_after_decode"] = cache.evictions
tie = 0
for b in range(4):
    comp = torch.tensor(hist[b] + [outs[b][-1][0]])
    lp = model.logprobs(pr4[b], comp, mode="train").cpu()
    # decode step j fed hist[b][j] at position len+j and scored outs[b][j][0]; lp[j+1] scores comp[j+1]
    ok = all(beq(lp[j + 1:j + 2], outs[b][j][1]) for j in range(2))
    # last step target = comp[-1]
    ok = ok and beq(lp[3:4], outs[b][2][1])
    tie += ok
r3["b4_vs_train_forward_seqs"] = tie
res["decode_b4_window"] = r3
log(f"B=4 window: {r3}")
del cache
torch.cuda.empty_cache()

# 4. T=8192: train == one-shot paged prefill == chunked [4096, 4095]; T=2 edge
ids = torch.randint(0, 200000, (8192,), generator=gen)
r4 = {}
for mode, cs in (("train", None), ("prefill", None), ("chunked", [4096, 4095]),
                 ("chunked", [2047, 2049, 4095])):
    torch.cuda.reset_peak_memory_stats(dev)
    t1 = time.perf_counter()
    lp = model.logprobs(ids[:8000], ids[8000:], mode=mode, chunk_sizes=cs).cpu()
    torch.cuda.synchronize()
    key = mode if cs is None else f"{mode}:{'/'.join(map(str, cs))}"
    r4[key] = {"s": round(time.perf_counter() - t1, 2),
               "peak_gib": round(torch.cuda.max_memory_allocated(dev) / 2**30, 2), "lp": lp}
base = r4["train"]["lp"]
for k in list(r4):
    r4[k]["equal_to_train"] = int((bits(r4[k]["lp"]) == bits(base)).sum())
    r4[k]["n"] = int(base.numel())
    del r4[k]["lp"]
small = torch.randint(0, 200000, (2,), generator=gen)
lps = {m: model.logprobs(small[:1], small[1:], mode=m).cpu() for m in ("train", "prefill", "decode")}
r4["T2_modes_equal"] = beq(lps["train"], lps["prefill"]) and beq(lps["train"], lps["decode"])
res["long_T8192"] = r4
log(f"T=8192: {r4}")
OUT.write_text(json.dumps(res, indent=1, default=str))
log(f"wrote {OUT}")
cache = model.new_batch_cache(64, 4)
for b in range(4):
    model.prefill(Session(cache=cache, seq=b), torch.randint(0, 200000, (20,), generator=gen))
try:
    model.decode_batch(cache, [-1], [5], [6], commit=False)
    torch.cuda.synchronize()
    log("negative seq: accepted silently")
except Exception as exc:  # noqa
    log(f"negative seq: {type(exc).__name__}: {str(exc)[:120]}")
