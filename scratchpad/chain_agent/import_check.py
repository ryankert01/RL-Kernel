import torch

import rl_engine.models.muse_glimmer.reference.backend  # noqa: F401
import rl_engine.models.muse_glimmer.reference.chain_harness as h
import rl_engine.models.muse_glimmer.reference.golden_ops as g
import rl_engine.models.muse_glimmer.reference.paged_kv  # noqa: F401
import rl_engine.models.muse_glimmer.reference.run_chain  # noqa: F401
import rl_engine.models.muse_glimmer.reference.strict_model  # noqa: F401
import rl_engine.models.muse_glimmer.reference.weights as w
from rl_engine.models.muse_glimmer.operator_trace import MuseGlimmerSpec

spec = MuseGlimmerSpec.from_manifest()
cfg = w.StrictConfig.from_spec(spec)
assert cfg.node_names() == spec.node_names(), "names"
assert all(cfg.node_kind(n) == spec.node_kind(n) for n in cfg.node_names())
print("official nodes", len(cfg.node_names()))
t = w.StrictConfig.tiny()
print("tiny", t.layer_types, t.padded_vocab_size, w.StrictConfig.tiny(vocab_size=1000).padded_vocab_size)
print([s.name for s in h.default_path_specs(512, 32)])
print([s.name for s in h.default_path_specs(2100, 32)])
x = torch.randint(-8, 8, (5, 6656)).float()
assert torch.equal(g.tree_sum_last(x), x.sum(-1))
from transformers import MuseGlimmerTextConfig  # noqa: E402

c = MuseGlimmerTextConfig(**t.hf_text_config_kwargs())
print("hf cfg ok", c.layer_types, c.layer_rope_theta, c.rope_parameters)
print("OK")
