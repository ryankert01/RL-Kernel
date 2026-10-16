"""Reviewer probe (HF, one GPU): reproduce committed numbers at HEAD and test the TF32 claim
outside the batch-1 'alone' shape.  argv: committed_ablation.json tokens.json out.json"""
import json
import os
import sys
import time

import torch

from rl_engine.models.muse_glimmer.ablation import ablation_hf as A
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


def per_seq_diff(a, b):
    return [sum(x != y for x, y in zip(sa, sb)) for sa, sb in zip(a, b)]


rec_hex = {}
for r in committed["records"]:
    rec_hex.setdefault((r["engine"], r["setting"]), {})[r["sequence"]] = r["logp_hex"]
order = [s.id for s in seqs]


def committed_hex(engine, setting):
    return [rec_hex[(engine, setting)][sid] for sid in order]


# ---- 1. RoPE K=1 GEMM under TF32 at the shapes the other drivers produce -------------------
inv = A.hf_rope_inv_freq(type("C", (), {})()) if False else None  # placeholder (needs cfg)
from transformers import AutoConfig  # noqa: E402

cfg_text = AutoConfig.from_pretrained(os.path.expanduser("~/models/Muse-Glimmer-30B"),
                                      trust_remote_code=False).text_config
inv = A.hf_rope_inv_freq(cfg_text).to(dev)
ids, mask, pos_left, _ = A.padded_batch(seqs, "left")
shapes = {
    "alone_b1_L2439": torch.arange(2439, device=dev)[None],
    "padded_left_b9_L2439": pos_left.to(dev),
    "padded_right_b9_L2439": torch.arange(2439, device=dev)[None].expand(9, -1).contiguous(),
    "decode_b9_L1": torch.tensor([[2300], [150], [200], [120], [250], [260], [180], [280], [2400]],
                                 device=dev),
    "decode_b1_L1": torch.tensor([[2437]], device=dev),
}
probe = {}
for name, pos in shapes.items():
    outs = {}
    for flag in (False, True):
        with A.tf32(flag):
            e = inv[None, :, None].float().expand(pos.shape[0], -1, 1)
            outs[flag] = (e @ pos[:, None, :].float()).transpose(1, 2)
            torch.cuda.synchronize(dev)
    probe[name] = int((outs[False] != outs[True]).sum())
res["rope_gemm_tf32_on_vs_off_elems_differ"] = probe
log(f"rope GEMM tf32 probe: {probe}")

# ---- 2. HF model: reproduce + combined factors ---------------------------------------------
model, lm_head, cfg = A.load_text_model(os.path.expanduser("~/models/Muse-Glimmer-30B"), dev)
log("model loaded")
base, _ = A.score_all(model, lm_head, cfg, seqs, driver="alone")
bh = hexes(base)
cb = [p["logp_hex"] for p in committed["baselines"]["hf"]["per_sequence"]]
res["baseline_vs_committed_tokens_differ"] = ndiff(bh, cb)
log(f"baseline vs committed: {res['baseline_vs_committed_tokens_differ']}")

pr, _ = A.score_all(model, lm_head, cfg, seqs, driver="padded_right")
prh = hexes(pr)
res["padded_right_vs_committed_record"] = ndiff(prh, committed_hex("hf", "padded_batch9_right"))
res["padded_right_vs_baseline"] = ndiff(prh, bh)
log(f"padded_right: vs committed {res['padded_right_vs_committed_record']}, vs base "
    f"{res['padded_right_vs_baseline']}")
with A.tf32(True):
    prt, _ = A.score_all(model, lm_head, cfg, seqs, driver="padded_right")
res["padded_right_tf32_on_vs_tf32_off"] = ndiff(hexes(prt), prh)
log(f"padded_right tf32 on vs off: {res['padded_right_tf32_on_vs_tf32_off']}")

l1, _ = A.score_all(model, lm_head, cfg, seqs, driver="prefill_minus_last")
l1h = hexes(l1)
res["prefill_L_minus_1_vs_committed_record"] = ndiff(l1h, committed_hex("hf", "prefill_L_minus_1"))
for chunk in (64, 256, 1024):
    ch, _ = A.score_all(model, lm_head, cfg, seqs, driver="chunked", chunk=chunk)
    chh = hexes(ch)
    res[f"chunked_{chunk}_vs_prefill_L_minus_1_per_seq"] = per_seq_diff(chh, l1h)
    res[f"chunked_{chunk}_vs_committed_record"] = ndiff(chh, committed_hex("hf", f"chunked_prefill_{chunk}"))
    log(f"chunk {chunk}: vs L-1 per seq {res[f'chunked_{chunk}_vs_prefill_L_minus_1_per_seq']}, "
        f"vs committed {res[f'chunked_{chunk}_vs_committed_record']}")

with A.tf32(True):
    bdt, _ = A.score_all(model, lm_head, cfg, seqs, driver="batched_decode_left")
res["batched_decode_tf32_on_vs_committed_tf32_off"] = ndiff(
    hexes(bdt), committed_hex("hf", "rollout_like_batched_decode"))
log(f"batched decode tf32 on vs committed off: {res['batched_decode_tf32_on_vs_committed_tf32_off']}")

# ---- 3. argmax ties in the bf16 soft-capped baseline logits ---------------------------------
ties = 0
ties_any_top = 0
with torch.no_grad():
    for s in seqs:
        h = A.hidden_alone(model, s)
        z = A.head_logits(h, lm_head, cfg, "bf16")
        top = z.topk(2, dim=-1).values
        ties += int((top[:, 0] == top[:, 1]).sum())
res["baseline_bf16_argmax_top2_ties"] = ties
log(f"bf16 baseline argmax ties (top1 == top2): {ties}")
json.dump(res, open(out_path, "w"), indent=1)
log(f"wrote {out_path}")
