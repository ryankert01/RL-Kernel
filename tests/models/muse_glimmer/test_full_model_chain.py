# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Real-weights full-model gates (RFC #501 full_model_chain).

Skipped unless ``MUSEGLIMMER_CHECKPOINT`` points at the complete pinned snapshot (``REVISION``
file present). Needs one 80 GB GPU: the bf16 text weights are ~52 GiB (loaded once per module).

* chain: prompt 256 + completion 32 tokens: training forward == one-shot prefill == chunked
  prefills == decode, byte-equal selected-token logp. ``MUSEGLIMMER_BACKEND=triton`` runs the
  sibling Triton ops (default: golden).
* batched decode (contract G2 at the model level; triton backend only, skipped otherwise): one
  paged cache with 16 sequences, contexts 1..2100 (one beyond W + 16, so local layers evict
  pages); for 3 decode steps the same step at batch 1, 2, 4 (permuted) and 16 gives the same
  fp32 bits per sequence, and the committed decode logps of two sequences equal the training
  forward over the same tokens.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from rl_engine.models.muse_glimmer.reference.chain_harness import compare_paths
from rl_engine.models.muse_glimmer.reference.run_chain import make_tokens
from rl_engine.models.muse_glimmer.reference.strict_model import MuseGlimmerStrictModel, Session
from rl_engine.models.muse_glimmer.reference.weights import MuseGlimmerWeights

_CKPT = os.environ.get("MUSEGLIMMER_CHECKPOINT")
_BACKEND = os.environ.get("MUSEGLIMMER_BACKEND", "golden")

pytestmark = [
    pytest.mark.skipif(not _CKPT, reason="MUSEGLIMMER_CHECKPOINT not set"),
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device"),
]


def _complete(path: str) -> bool:
    return (Path(path).expanduser() / "REVISION").is_file()


@pytest.fixture(scope="module")
def real_model():
    if _CKPT is not None and not _complete(_CKPT):
        pytest.skip("checkpoint incomplete")
    device = torch.device("cuda")
    free, _total = torch.cuda.mem_get_info(device)
    if free < 70 * 2**30:
        pytest.skip(f"need ~70 GiB free on {device}, have {free / 2**30:.1f} GiB")
    layout = "kn" if _BACKEND == "golden" else "nk"
    weights = MuseGlimmerWeights.load_checkpoint(_CKPT, device=device, layout=layout)
    assert weights.checkpoint_report is not None
    assert weights.checkpoint_report.text_tensor_count == 627
    model = MuseGlimmerStrictModel(weights, _BACKEND, allow_partial=_BACKEND == "triton")
    if _BACKEND == "triton":
        assert not model.ops.golden_nodes(), model.ops.provenance  # the full Triton op set
    yield model
    del model, weights
    torch.cuda.empty_cache()


def test_full_model_chain_prompt256_completion32(real_model):
    model = real_model
    ids = make_tokens(256, 32, seed=20261015, vocab=model.cfg.vocab_size)
    report = compare_paths(model, ids[:256], ids[256:], localize=True)
    print(report.markdown())
    assert all(torch.isfinite(r.logp).all() for r in report.results)
    assert report.all_equal, report.markdown()


# contexts of the 16 sequences: 2100 > W + 16 (eviction), the window edge 2047..2064, and short
_DECODE_CONTEXTS = (2100, 2064, 2048, 2047, 1, 2, 17, 63, 64, 65, 100, 257, 300, 513, 600, 31)
_STEPS = 3


def _bits(x: torch.Tensor) -> torch.Tensor:
    return x.detach().float().cpu().contiguous().view(torch.int32)


@pytest.mark.skipif(_BACKEND != "triton", reason="batched-decode gate runs the triton op set")
def test_batched_decode_is_batch_invariant_with_eviction(real_model):
    model = real_model
    window = model.cfg.sliding_window
    assert max(_DECODE_CONTEXTS) > window + 16  # at least one sequence evicts pages
    n = len(_DECODE_CONTEXTS)
    vocab = min(model.cfg.vocab_size, 200000)
    gen = torch.Generator(device="cpu").manual_seed(501)
    prompts = [torch.randint(0, vocab, (c,), generator=gen) for c in _DECODE_CONTEXTS]
    for p in prompts:
        p[0] = 200000  # BOS
    completions = [torch.randint(0, vocab, (_STEPS + 1,), generator=gen) for _ in range(n)]
    cache = model.new_batch_cache(max(_DECODE_CONTEXTS) + _STEPS + 2, n)
    got: list[list[torch.Tensor]] = []
    for b, (p, c) in enumerate(zip(prompts, completions)):
        tgt = torch.cat([p[1:], c[:1]])
        got.append([model.prefill(Session(cache=cache, seq=b), p, targets=tgt)[-1:].cpu()])
    assert cache.evictions > 0, "no page was evicted: the window case is not exercised"
    seqs = list(range(n))
    sub_batches = ([5, 0], [3, 0, 15, 8], [15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0])
    for j in range(_STEPS):
        toks = [int(c[j]) for c in completions]
        tgts = [int(c[j + 1]) for c in completions]
        single = torch.cat([
            model.decode_batch(cache, [b], [toks[b]], [tgts[b]], commit=False).cpu()
            for b in seqs
        ])
        for sub in sub_batches:  # batch 2, permuted batch 4, reversed batch 16
            out = model.decode_batch(cache, sub, [toks[b] for b in sub], [tgts[b] for b in sub],
                                     commit=False)
            assert torch.equal(_bits(out), _bits(single[sub])), (j, sub)
        full = model.decode_batch(cache, seqs, toks, tgts, commit=True)
        assert torch.equal(_bits(full), _bits(single)), j
        for b in seqs:
            got[b].append(full[b : b + 1].cpu())
    assert [cache.seq_len(b) for b in seqs] == [c + _STEPS for c in _DECODE_CONTEXTS]
    # the committed batched-decode logps of the longest (evicting) and a short sequence equal
    # the training forward over the same tokens (train == decode across the batch)
    for b in (0, 6):
        train = model.logprobs(prompts[b], completions[b], mode="train")
        assert torch.equal(_bits(torch.cat(got[b])), _bits(train)), b
