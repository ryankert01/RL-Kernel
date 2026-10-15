# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Real-weights full-model chain gate (RFC #501 full_model_chain).

Skipped unless ``MUSEGLIMMER_CHECKPOINT`` points at the complete pinned snapshot (``REVISION``
file present). Needs one 80 GB GPU: the bf16 text weights are ~52 GiB. Prompt 256 + completion
32 tokens: training forward == one-shot prefill == chunked prefills == decode, byte-equal
selected-token logp. Set ``MUSEGLIMMER_BACKEND=triton`` to run the sibling Triton ops.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from rl_engine.models.muse_glimmer.reference.chain_harness import compare_paths
from rl_engine.models.muse_glimmer.reference.run_chain import make_tokens
from rl_engine.models.muse_glimmer.reference.strict_model import MuseGlimmerStrictModel
from rl_engine.models.muse_glimmer.reference.weights import MuseGlimmerWeights

_CKPT = os.environ.get("MUSEGLIMMER_CHECKPOINT")
_BACKEND = os.environ.get("MUSEGLIMMER_BACKEND", "golden")

pytestmark = [
    pytest.mark.skipif(not _CKPT, reason="MUSEGLIMMER_CHECKPOINT not set"),
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device"),
]


def _complete(path: str) -> bool:
    return (Path(path).expanduser() / "REVISION").is_file()


@pytest.mark.skipif(_CKPT is not None and not _complete(_CKPT), reason="checkpoint incomplete")
def test_full_model_chain_prompt256_completion32():
    device = torch.device("cuda")
    free, _total = torch.cuda.mem_get_info(device)
    if free < 70 * 2**30:
        pytest.skip(f"need ~70 GiB free on {device}, have {free / 2**30:.1f} GiB")
    layout = "kn" if _BACKEND == "golden" else "nk"
    weights = MuseGlimmerWeights.load_checkpoint(_CKPT, device=device, layout=layout)
    assert weights.checkpoint_report is not None
    assert weights.checkpoint_report.text_tensor_count == 627
    model = MuseGlimmerStrictModel(weights, _BACKEND, allow_partial=_BACKEND == "triton")
    ids = make_tokens(256, 32, seed=20261015, vocab=weights.cfg.vocab_size)
    report = compare_paths(model, ids[:256], ids[256:], localize=True)
    print(report.markdown())
    assert all(torch.isfinite(r.logp).all() for r in report.results)
    assert report.all_equal, report.markdown()
