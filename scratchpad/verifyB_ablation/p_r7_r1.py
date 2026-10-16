"""Probe: (R1) GPU pow vs CPU pow of inv_freq; (R7) packed_pass multi-segment paged prefill vs the
reference's own KVPlan multi-sequence path, and multi-segment varlen vs per-sequence forward_train."""
import json

import torch

from rl_engine.models.muse_glimmer.ablation import ablation_hf as A
from rl_engine.models.muse_glimmer.ablation import ablation_strict as S
from rl_engine.models.muse_glimmer.reference.strict_model import KVPlan, MuseGlimmerStrictModel
from rl_engine.models.muse_glimmer.reference.weights import MuseGlimmerWeights, StrictConfig

out = {}
# ---- R1
from transformers import AutoConfig

cfg = AutoConfig.from_pretrained("/home/ubuntu/models/Muse-Glimmer-30B")
tc = getattr(cfg, "text_config", cfg)
cpu = A.hf_rope_inv_freq(tc)
old_harness = A.hf_rope_inv_freq(tc, "cuda")  # the pre-fix path: CPU pow, copied to GPU
new = A.rope_inv_freq_formula(tc, "cuda")
out["r1_old_harness_vs_cpu_entries_differ"] = int((A.fp32_bits(old_harness.cpu()) != A.fp32_bits(cpu)).sum())
out["r1_gpu_pow_vs_cpu_entries_differ"] = int((A.fp32_bits(new.cpu()) != A.fp32_bits(cpu)).sum())
out["r1_formula_on_cpu_equal_hf"] = bool(torch.equal(A.fp32_bits(A.rope_inv_freq_formula(tc, "cpu")), A.fp32_bits(cpu)))

# ---- R7
torch.manual_seed(0)
scfg = StrictConfig.tiny()
w = MuseGlimmerWeights.random(scfg, device="cuda", seed=1, layout="both")
model = MuseGlimmerStrictModel(w, "golden")
gen = torch.Generator().manual_seed(5)
toks = torch.randint(0, scfg.vocab_size, (260,), generator=gen).tolist()
seqs = [A.SeqSpec("x", tuple(toks[:41]), 30), A.SeqSpec("y", tuple(toks[:250]), 120),
        A.SeqSpec("z", tuple(toks[7:90]), 61)]
# (a) multi-segment paged prefill of L-1 tokens: packed_pass vs model._run(KVPlan) (private, probe only)
cache_a = model.new_batch_cache(max(s.length for s in seqs), len(seqs))
cache_b = model.new_batch_cache(max(s.length for s in seqs), len(seqs))
segs = [S.Segment(S._ids(s)[:-1], 0, S._train_targets(s, s.length - 1), slot=b) for b, s in enumerate(seqs)]
pp = torch.cat(S.packed_pass(model, segs, cache_a))
ids = torch.cat([S._ids(s)[:-1] for s in seqs]).cuda()
tg = torch.cat([S._train_targets(s, s.length - 1) for s in seqs]).cuda()
plan = KVPlan(cache_b, list(range(len(seqs))), [0] * len(seqs), [s.length - 1 for s in seqs])
with torch.no_grad():
    ref = model._run(ids, plan.positions(model.device), plan, tg)["logp"]
valid = tg >= 0
out["r7_paged_multiseg_prefill_vs_KVPlan_rows"] = int(valid.sum())
out["r7_paged_multiseg_prefill_vs_KVPlan_differ"] = int((S._bits(pp[valid.to(pp.device)]) != S._bits(ref[valid])).sum())
# (b) then one mixed step: 1 token of x, 5 tokens of z (continuing), compare again
# (c) multi-segment varlen vs per-sequence forward_train
segs = [S.Segment(S._ids(s), 0, S._train_targets(s)) for s in seqs]
outs = S.packed_pass(model, segs)
nd = 0
nrows = 0
for s, o in zip(seqs, outs):
    t = S._train_targets(s)
    r = model.forward_train(S._ids(s), t)["logp"]
    v = (t >= 0).cuda()
    nd += int((S._bits(o[v]) != S._bits(r[v])).sum())
    nrows += int(v.sum())
out["r7_varlen_multiseg_vs_forward_train_rows"] = nrows
out["r7_varlen_multiseg_vs_forward_train_differ"] = nd
print(json.dumps(out, indent=1))
