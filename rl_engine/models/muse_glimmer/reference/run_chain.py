#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""CLI: run the Muse-Glimmer-30B full-model chain gate on real weights.

Example (one leased A100):
    ~/bin/gpu-lease -n 1 -- env PYTHONPATH=$PWD ~/envs/vllm/bin/python \
        rl_engine/models/muse_glimmer/reference/run_chain.py \
        --checkpoint ~/models/Muse-Glimmer-30B --backend golden \
        --prompt-len 512 --completion-len 32 --hf-compare --json-out chain.json
"""

from __future__ import annotations

import argparse
import gc
import json
import pathlib
import sys
import time

import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rl_engine.models.muse_glimmer.reference.chain_harness import (  # noqa: E402
    PathSpec,
    compare_paths,
    compare_to_hf,
    default_path_specs,
    strict_reference_outputs,
)
from rl_engine.models.muse_glimmer.reference.strict_model import (  # noqa: E402
    MuseGlimmerStrictModel,
)
from rl_engine.models.muse_glimmer.reference.weights import MuseGlimmerWeights  # noqa: E402
from rl_engine.validation.operators.accelerator import disable_tf32  # noqa: E402


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Muse-Glimmer-30B strict full-model chain gate")
    p.add_argument("--checkpoint", default="~/models/Muse-Glimmer-30B")
    p.add_argument("--backend", choices=("golden", "triton"), default="golden")
    p.add_argument("--allow-partial", action="store_true",
                   help="backend=triton: substitute golden for missing sibling ops (recorded)")
    p.add_argument("--gemm-impl", choices=("triton_fp32", "torch_fp32_debug"),
                   default="triton_fp32")
    p.add_argument("--prompt-len", type=int, default=512)
    p.add_argument("--completion-len", type=int, default=64)
    p.add_argument("--seed", type=int, default=20261015)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--hf-compare", action="store_true")
    p.add_argument("--hf-device", default=None,
                   help="device for the HF model (default: same device after freeing strict)")
    p.add_argument("--verify-shards", action="store_true", help="SHA-256 every shard (~60 GB)")
    p.add_argument("--no-localize", action="store_true")
    p.add_argument("--only", nargs="*", default=None,
                   help="subset of path names (e.g. train prefill decode)")
    p.add_argument("--extra-chunking", action="append", default=[],
                   help="comma-separated chunk sizes to add as a chunked path")
    p.add_argument("--json-out", default=None)
    p.add_argument("--save-dir", default=None,
                   help="also torch.save strict_ref.pt / hf_bf16.pt (logits, hidden states)")
    p.add_argument("--score-file", default=None,
                   help="scoring mode: a muse_glimmer_xrt_v1 token file (e.g. "
                        "reports/muse_glimmer_poc/data/xrt_tokens.json); writes the train-"
                        "forward logp of every completion token (muse_glimmer_xrt_logprobs_v1) "
                        "to --score-out and skips the random-token chain")
    p.add_argument("--score-out", default=None)
    p.add_argument("--score-check-modes", default="prefill,decode",
                   help="scoring mode: forms compared bitwise with the train forward (comma "
                        "list of prefill, decode, per_token, chunked:<n1>/<n2>/...; '' = none)")
    return p.parse_args(argv)


def make_tokens(prompt_len: int, completion_len: int, seed: int, vocab: int) -> torch.Tensor:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    ids = torch.randint(0, min(vocab, 200000), (prompt_len + completion_len,), generator=gen)
    ids[0] = 200000  # BOS of the pinned tokenizer
    return ids


def _score_main(args: argparse.Namespace, model: MuseGlimmerStrictModel, weights) -> int:
    """Scoring mode (``--score-file``): see :mod:`...reference.xrt_scoring`."""
    from rl_engine.models.muse_glimmer.reference.xrt_scoring import score_token_file

    if not args.score_out:
        raise SystemExit("--score-file needs --score-out")
    modes = [m for m in args.score_check_modes.split(",") if m]
    t0 = time.perf_counter()
    scored = score_token_file(model, args.score_file, check_modes=modes, log=_log)
    scored["seconds"] = time.perf_counter() - t0
    scored["checkpoint_revision_loaded"] = weights.checkpoint_report.revision
    scored["peak_allocated_gib"] = torch.cuda.max_memory_allocated(model.device) / 2**30
    pathlib.Path(args.score_out).write_text(json.dumps(scored, indent=1))
    n = sum(s["n_tokens"] for s in scored["sequences"])
    ok = scored["checks"]["all_byte_equal_to_train"]
    _log(f"wrote {args.score_out}: {n} tokens; {modes} byte-equal to train: {ok}")
    return 0 if ok else 1


def main(argv=None) -> int:
    args = parse_args(argv)
    device = torch.device(args.device)
    disable_tf32(device.type)
    _log(f"loading strict weights from {args.checkpoint} (layout for backend {args.backend})")
    t0 = time.perf_counter()
    layout = "kn" if args.backend == "golden" else "nk"
    weights = MuseGlimmerWeights.load_checkpoint(
        args.checkpoint, device=device, layout=layout, verify_shards=args.verify_shards,
        progress=_log,
    )
    _log(f"weights ready in {time.perf_counter() - t0:.1f}s: {weights.memory_bytes() / 2**30:.1f}"
         f" GiB on {device}; checkpoint {weights.checkpoint_report.revision}")
    model = MuseGlimmerStrictModel(
        weights, args.backend, allow_partial=args.allow_partial, gemm_impl=args.gemm_impl
    )
    _log(f"ops: golden={model.ops.golden_nodes()} sibling={model.ops.sibling_nodes()}")
    if args.score_file:
        return _score_main(args, model, weights)
    ids = make_tokens(args.prompt_len, args.completion_len, args.seed, weights.cfg.vocab_size)
    prompt, completion = ids[: args.prompt_len], ids[args.prompt_len :]
    specs = default_path_specs(args.prompt_len, args.completion_len,
                               window=weights.cfg.sliding_window)
    for i, extra in enumerate(args.extra_chunking):
        sizes = tuple(int(x) for x in extra.split(","))
        specs.append(PathSpec(f"chunked:extra{i}", "chunked", sizes))
    if args.only:
        specs = [s for s in specs if s.name in set(args.only)]
    report = compare_paths(model, prompt, completion, specs=specs,
                           localize=not args.no_localize, log=_log)
    print(report.markdown(), flush=True)
    payload = {
        "args": vars(args),
        "model": model.describe(),
        "chain": report.to_dict(),
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
    }
    if args.hf_compare:
        _log("strict reference outputs (train forward with capture) for the HF anchor")
        ref = strict_reference_outputs(model, ids)
        save_dir = pathlib.Path(args.save_dir) if args.save_dir else None
        if save_dir is not None:
            save_dir.mkdir(parents=True, exist_ok=True)
            torch.save(ref, save_dir / "strict_ref.pt")
            _log(f"saved {save_dir / 'strict_ref.pt'}")
        hf_device = torch.device(args.hf_device) if args.hf_device else device
        if hf_device == device:
            _log("freeing strict model before loading HF on the same device")
            del model, weights
            gc.collect()
            torch.cuda.empty_cache()
        hf = compare_to_hf(ref, checkpoint=str(pathlib.Path(args.checkpoint).expanduser()),
                           device=hf_device, completion_start=args.prompt_len, log=_log,
                           save_path=None if save_dir is None else save_dir / "hf_bf16.pt")
        payload["hf_compare"] = hf
        _log("HF anchor: logits(all) max|Δ|={:.4f}  logits(completion) max|Δ|={:.4f}  "
             "logp(completion) max|Δ|={:.5f} mean|Δ|={:.5f} argmax agree={:.4f}".format(
                 hf["logits_all_positions"]["max_abs"],
                 hf["logits_completion_positions"]["max_abs"],
                 hf["logp_completion"]["max_abs"], hf["logp_completion"]["mean_abs"],
                 hf["argmax_agreement_all"]))
        for row in hf.get("per_layer", []):
            _log("  layer {:>10}: max|Δ|={:.4e} mean|Δ|={:.4e} rms(hf)={:.4e} "
                 "max/rms={:.3e}".format(str(row["layer"]), row["max_abs"], row["mean_abs"],
                                         row["ref_rms"], row["max_abs_over_rms"]))
    if args.json_out:
        pathlib.Path(args.json_out).write_text(json.dumps(payload, indent=2, default=str))
        _log(f"wrote {args.json_out}")
    return 0 if report.all_equal else 1


if __name__ == "__main__":
    raise SystemExit(main())
