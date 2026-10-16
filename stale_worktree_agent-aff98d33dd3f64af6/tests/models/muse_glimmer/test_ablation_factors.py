# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""Unit tests for the ablation factor patches on a tiny random Muse Glimmer text config.

No checkpoint needed.  Each arithmetic factor must change the output; exiting the context
manager must leave the model bitwise unchanged; the replacement forward at default knobs must
be bitwise equal to the HF forward.  CPU, fp32-free bf16 path like the real harness.
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("transformers")
from transformers.models.muse_glimmer import modeling_muse_glimmer as hf  # noqa: E402
from transformers.models.muse_glimmer.configuration_muse_glimmer import (  # noqa: E402
    MuseGlimmerTextConfig,
)

from rl_engine.models.muse_glimmer.e2e import ablation_hf as ab  # noqa: E402

# Tiny config: window 8 on 24-token sequences, so every window factor changes the output.
NO_CHANGE_EXPECTED = {"patched_identity", "rope_table_elementwise"}
SKIP_CPU = {"tf32_on", "sdpa", "sliding_window_1024", "sliding_window_4096"}


def _tiny():
    torch.manual_seed(0)
    cfg = MuseGlimmerTextConfig(
        vocab_size=97,
        hidden_size=64,
        intermediate_size=96,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        sliding_window=8,
        layer_types=["sliding_attention", "sliding_attention", "sliding_attention", "full_attention"],
        layer_rope_theta=[10000.0, 10000.0, 10000.0, 0],
        qk_scale_factor=3.87,
        rms_norm_eps=1e-5,
        post_norm_eps=1e-8,
        output_multiplier=0.5,
        final_logit_softcapping=20.0,
        max_position_embeddings=256,
    )
    cfg._attn_implementation = "eager"
    model = hf.MuseGlimmerTextModel(cfg).to(torch.bfloat16).eval()
    for p in model.parameters():
        p.data.normal_(0, 0.2)
    head = torch.nn.Linear(64, 97, bias=False).to(torch.bfloat16)
    return model, head


def _score(model, head, seqs):
    with torch.no_grad():
        return torch.cat([s.logp for s in ab.score_sequences(model, head, seqs)])


@pytest.fixture(scope="module")
def setup():
    model, head = _tiny()
    g = torch.Generator().manual_seed(1)
    seqs = torch.randint(0, 97, (2, 24), generator=g).tolist()
    return model, head, seqs


@pytest.mark.parametrize("name", sorted(set(ab.factor_registry(_tiny()[0])) - SKIP_CPU))
def test_factor_changes_output_and_restores(setup, name, monkeypatch):
    model, head, seqs = setup
    monkeypatch.setattr(ab, "COMPLETION_TOKENS", 8)
    base = _score(model, head, seqs)
    with ab.factor_registry(model)[name]():
        var = _score(model, head, seqs)
    after = _score(model, head, seqs)
    assert torch.equal(base, after), f"{name} did not restore the model bitwise"
    if name in NO_CHANGE_EXPECTED or name == "baseline":
        assert torch.equal(base, var), f"{name} should be a no-op"
    else:
        assert not torch.equal(base, var), f"{name} changed nothing"


def test_compare_metrics(setup):
    model, head, seqs = setup
    base = ab.score_sequences(model, head, seqs)
    same = ab.compare(base, base)
    assert same["bitwise_equal"] and same["max_abs_dlogp"] == 0.0
    assert same["first_drift_index"] is None and same["argmax_change_frac"] == 0.0
