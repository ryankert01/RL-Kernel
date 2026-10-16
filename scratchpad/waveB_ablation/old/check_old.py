"""Run the new R1 / R7 assertions against the pre-fix modules (expected: both FAIL)."""
import ast
import importlib.util
import pathlib
import sys
import types

import torch

HERE = pathlib.Path(__file__).parent
spec = importlib.util.spec_from_file_location("old_ablation_hf", HERE / "old_ablation_hf.py")
OLD = importlib.util.module_from_spec(spec)
sys.modules["old_ablation_hf"] = OLD
spec.loader.exec_module(OLD)
from rl_engine.models.muse_glimmer.ablation import ablation_hf as NEW  # noqa: E402

cfg = types.SimpleNamespace(rope_parameters={"rope_theta": 500000.0}, head_dim=128,
                            hidden_size=4096, num_attention_heads=32)


def model():
    inv = NEW.hf_rope_inv_freq(cfg).to("cuda")
    return types.SimpleNamespace(config=cfg, rotary_emb=types.SimpleNamespace(
        inv_freq=inv, original_inv_freq=inv.clone()))


cuda = NEW.rope_inv_freq_formula(cfg, "cuda")
for name, mod in (("old", OLD), ("new", NEW)):
    m = model()
    with mod.rope_inv_freq_on(m, "cuda"):
        got = m.rotary_emb.inv_freq
        nd = int((NEW.fp32_bits(got) != NEW.fp32_bits(cuda)).sum())
    print(f"R1 {name}: installed table vs GPU-pow table: {nd}/64 entries differ -> "
          f"{'PASS' if nd == 0 else 'FAIL'}")

for name, path in (("old", HERE / "old_ablation_strict.py"),
                   (
                       "new",
                       pathlib.Path(sys.argv[1]) / "rl_engine/models/muse_glimmer/ablation/"
                                                   "ablation_strict.py")):
    tree = ast.parse(path.read_text())
    bad = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
                "rl_engine.models.muse_glimmer.reference"):
            bad += [a.name for a in node.names if a.name.startswith("_")]
        if isinstance(node, ast.Attribute) and node.attr.startswith("_") \
                and not node.attr.startswith("__"):
            bad.append(node.attr)
    print(f"R7 {name}: private names {bad} -> {'PASS' if not bad else 'FAIL'}")
