#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""CLI: op-class profile of the strict single-GPU Muse-Glimmer-30B model or of HF bf16 (sdpa).

Examples (one leased A100 each; strict and HF do not fit on one GPU together):
    ~/bin/gpu-lease -n 1 -- env PYTHONPATH=$PWD ~/envs/vllm/bin/python \
        rl_engine/models/muse_glimmer/reference/run_profile.py --target strict \
        --checkpoint ~/models/Muse-Glimmer-30B --json-out profile_strict.json
    ~/bin/gpu-lease -n 1 -- env PYTHONPATH=$PWD ~/envs/vllm/bin/python \
        rl_engine/models/muse_glimmer/reference/run_profile.py --target hf \
        --checkpoint ~/models/Muse-Glimmer-30B --json-out profile_hf.json

Method and op classes: see :mod:`rl_engine.models.muse_glimmer.reference.profiling`.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rl_engine.models.muse_glimmer.reference import profiling  # noqa: E402
from rl_engine.validation.operators.accelerator import disable_tf32  # noqa: E402


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Muse-Glimmer-30B op-class profile")
    p.add_argument("--target", choices=("strict", "hf"), required=True)
    p.add_argument("--checkpoint", default="~/models/Muse-Glimmer-30B")
    p.add_argument("--backend", choices=("triton", "golden"), default="triton",
                   help="strict backend (the profile is meant for triton)")
    p.add_argument("--prefill-lens", default="512,2048,4096")
    p.add_argument("--decode-batches", default="1,16")
    p.add_argument("--decode-context", type=int, default=2048)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--decode-repeats", type=int, default=5)
    p.add_argument("--no-paged", action="store_true", help="strict: skip paged prefill runs")
    p.add_argument("--no-profiler-check", action="store_true")
    p.add_argument("--hf-attn", default="sdpa")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--json-out", required=True)
    return p.parse_args(argv)


def _ints(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def main(argv=None) -> int:
    args = parse_args(argv)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    disable_tf32(device.type)
    ckpt = str(pathlib.Path(args.checkpoint).expanduser())
    t0 = time.perf_counter()
    common = dict(
        prefill_lens=_ints(args.prefill_lens),
        decode_batches=_ints(args.decode_batches),
        decode_context=args.decode_context,
        warmup=args.warmup,
        repeats=args.repeats,
        decode_repeats=args.decode_repeats,
        profiler_check=not args.no_profiler_check,
        log=_log,
    )
    if args.target == "strict":
        from rl_engine.models.muse_glimmer.reference.strict_model import MuseGlimmerStrictModel
        from rl_engine.models.muse_glimmer.reference.weights import MuseGlimmerWeights

        layout = "kn" if args.backend == "golden" else "nk"
        weights = MuseGlimmerWeights.load_checkpoint(ckpt, device=device, layout=layout,
                                                     progress=_log)
        load_s = time.perf_counter() - t0
        model = MuseGlimmerStrictModel(weights, args.backend)
        _log(f"strict weights {weights.memory_bytes() / 2**30:.2f} GiB loaded in {load_s:.1f}s")
        result = profiling.profile_strict(model, paged_prefill=not args.no_paged, **common)
        result["weights_gib"] = weights.memory_bytes() / 2**30
        result["ops_provenance"] = dict(model.ops.provenance)
        result["backend"] = args.backend
    else:
        from transformers import MuseGlimmerForConditionalGeneration

        hf = MuseGlimmerForConditionalGeneration.from_pretrained(
            ckpt, dtype=torch.bfloat16, attn_implementation=args.hf_attn
        )
        hf.eval()
        hf.to(device)
        load_s = time.perf_counter() - t0
        _log(f"HF bf16 ({args.hf_attn}) loaded in {load_s:.1f}s; "
             f"{torch.cuda.memory_allocated(device) / 2**30:.2f} GiB allocated")
        result = profiling.profile_hf(hf, **common)
        result["weights_gib"] = sum(p.numel() * p.element_size() for p in hf.parameters()) / 2**30
    result["load_seconds"] = load_s
    result["method"] = ("cuda_events: nested named regions around every op call, exclusive "
                        "time per class, median over repeats; clean totals without regions; "
                        "one torch.profiler pass per config for summed kernel time")
    result["args"] = vars(args)
    result["env"] = {
        "gpu": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }
    try:
        import triton

        result["env"]["triton"] = triton.__version__
    except ImportError:
        pass
    try:
        import transformers

        result["env"]["transformers"] = transformers.__version__
    except ImportError:
        pass
    pathlib.Path(args.json_out).write_text(json.dumps(result, indent=1))
    _log(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
