import json

import torch

from rl_engine.models.muse_glimmer.reference.chain_harness import (
    compare_strict_to_hf_outputs,
    hf_text_logits,
    hf_text_model_from_weights,
    strict_reference_outputs,
)
from rl_engine.models.muse_glimmer.reference.strict_model import MuseGlimmerStrictModel
from rl_engine.models.muse_glimmer.reference.weights import MuseGlimmerWeights, StrictConfig

DEVICE = torch.device("cuda")
for seed in (9, 10):
    cfg = StrictConfig.tiny()
    weights = MuseGlimmerWeights.random(cfg, device=DEVICE, seed=seed, layout="both")
    model = MuseGlimmerStrictModel(weights, "golden")
    gen = torch.Generator(device="cpu").manual_seed(21)
    ids = torch.randint(0, cfg.vocab_size, (230,), generator=gen)
    ref = strict_reference_outputs(model, ids)
    hf_model, lm_head = hf_text_model_from_weights(weights, DEVICE)
    hf_logits, hf_hidden = hf_text_logits(hf_model, lm_head, ids, cfg)
    report = compare_strict_to_hf_outputs(ref, hf_logits, hf_hidden, completion_start=200)
    print("seed", seed)
    print(json.dumps({k: v for k, v in report.items() if k != "per_layer"}, indent=1))
    for row in report["per_layer"]:
        print(row)
    lg = ref["logits"]
    print("strict logits std", float(lg.std()), "range", float(lg.min()), float(lg.max()))
    print("top1-top2 gap mean", float((lg.topk(2, -1).values[:, 0] - lg.topk(2, -1).values[:, 1]).mean()))
