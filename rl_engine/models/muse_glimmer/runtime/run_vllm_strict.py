# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""CLI: native vs strict Muse Glimmer rollout through vLLM 0.31 with bit-pattern logprob dumps.

Examples (run under a GPU lease; the lease sets CUDA_VISIBLE_DEVICES)::

    python -m rl_engine.models.muse_glimmer.runtime.run_vllm_strict --tp 1 --native \
        --out native_tp1.json
    python -m rl_engine.models.muse_glimmer.runtime.run_vllm_strict --tp 1 --strict \
        --prefill-check --out strict_tp1.json
    python -m rl_engine.models.muse_glimmer.runtime.run_vllm_strict --tp 2 --strict \
        --score native_tp1.json --out strict_tp2_score.json

Output JSON: ``{"token_ids": [[...]], "logprobs": [[fp32...]], "logprobs_hex": [["%08x"...]],
"prompt_token_ids": [[...]], "meta": {...provenance, timing...}}`` plus, with ``--prefill-check``,
``prefill_logprobs`` / ``prefill_logprobs_hex`` / ``decode_vs_prefill_equal``.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import time
from pathlib import Path
from typing import Any

DEFAULT_MODEL = os.path.expanduser("~/models/Muse-Glimmer-30B")
DEFAULT_PROMPTS = [
    "The capital of France is",
    "Write one sentence about the ocean.",
    "Explain why the sky is blue in two sentences.",
    "List three prime numbers greater than 10:",
    "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
    "Translate to German: The weather is nice today.",
    "Q: What is 17 * 23?\nA:",
    "Once upon a time, in a small village by the sea,",
]


def fp32_hex(value: float) -> str:
    return struct.pack(">f", float(value)).hex()


def hex_rows(rows: list[list[float]]) -> list[list[str]]:
    return [[fp32_hex(v) for v in row] for row in rows]


def _load_prompts(path: str | None, tokenizer: Any, chat_template: bool) -> list[list[int]]:
    if path:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data = data.get("prompts") or data.get("prompt_token_ids")
        if not isinstance(data, list):
            raise ValueError("prompt file must hold a JSON list of strings or token-id lists")
    else:
        data = list(DEFAULT_PROMPTS)
    prompts: list[list[int]] = []
    for item in data:
        if isinstance(item, str):
            if chat_template:
                ids = tokenizer.apply_chat_template(
                    [{"role": "user", "content": item}], add_generation_prompt=True, tokenize=True
                )
                if hasattr(ids, "input_ids"):
                    ids = ids["input_ids"]
            else:
                ids = tokenizer(item, add_special_tokens=True)["input_ids"]
            prompts.append([int(t) for t in ids])
        else:
            prompts.append([int(t) for t in item])
    return prompts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--tp", type=int, default=1)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--strict", action="store_true", default=True)
    mode.add_argument("--native", action="store_true")
    parser.add_argument("--prompt-file", default=None, help="JSON list of strings or token-id lists")
    parser.add_argument("--chat-template", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu-ids", default=None, help="comma list; ignored when a lease set CUDA_VISIBLE_DEVICES")
    parser.add_argument(
        "--score",
        default=None,
        help="score the prompt_token_ids+token_ids of a previous output JSON instead of generating",
    )
    parser.add_argument("--prefill-check", action="store_true", help="after generate, rescore via logprobs()")
    parser.add_argument("--lora-zero-check", action="store_true", help="load an all-zero LoRA and re-score")
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    strict = not args.native

    from rl_engine.models.muse_glimmer.runtime.strict_rollout_engine import StrictRolloutEngine

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if args.gpu_ids:
        gpu_ids = [int(g) for g in args.gpu_ids.split(",")]
    elif visible:
        gpu_ids = list(range(len(visible.split(","))))
    else:
        gpu_ids = list(range(args.tp))

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    result: dict[str, Any] = {"mode": "score" if args.score else "generate", "strict": strict, "tp": args.tp}

    engine = StrictRolloutEngine(
        args.model,
        tensor_parallel_size=args.tp,
        gpu_ids=gpu_ids,
        strict=strict,
        max_model_len=args.max_model_len,
        seed=args.seed,
    )
    try:
        if args.score:
            previous = json.loads(Path(args.score).read_text(encoding="utf-8"))
            prompt_ids = previous["prompt_token_ids"]
            completions = previous["token_ids"]
            sequences = [p + c for p, c in zip(prompt_ids, completions, strict=True)]
            starts = [len(p) for p in prompt_ids]
            t0 = time.time()
            logprobs = engine.logprobs(sequences, starts)
            result.update(
                {
                    "prompt_token_ids": prompt_ids,
                    "token_ids": completions,
                    "logprobs": logprobs,
                    "logprobs_hex": hex_rows(logprobs),
                    "score_seconds": time.time() - t0,
                    "scored_from": args.score,
                }
            )
            if "logprobs_hex" in previous:
                result["bits_equal_to_scored_file"] = previous["logprobs_hex"] == result["logprobs_hex"]
                diffs = [
                    abs(a - b)
                    for ra, rb in zip(previous["logprobs"], logprobs, strict=True)
                    for a, b in zip(ra, rb, strict=True)
                ]
                result["max_abs_diff_to_scored_file"] = max(diffs) if diffs else 0.0
        else:
            prompts = _load_prompts(args.prompt_file, tokenizer, args.chat_template)
            t0 = time.time()
            samples = engine.generate(
                prompts, n=args.n, max_tokens=args.max_tokens, temperature=args.temperature
            )
            gen_seconds = time.time() - t0
            flat_tokens = [s.token_ids for group in samples for s in group]
            flat_logprobs = [s.logprobs for group in samples for s in group]
            flat_prompts = [p for p, group in zip(prompts, samples, strict=True) for _ in group]
            generated = sum(len(t) for t in flat_tokens)
            result.update(
                {
                    "prompt_token_ids": flat_prompts,
                    "token_ids": flat_tokens,
                    "logprobs": flat_logprobs,
                    "logprobs_hex": hex_rows(flat_logprobs),
                    "finish_reasons": [s.finish_reason for group in samples for s in group],
                    "texts": [tokenizer.decode(t) for t in flat_tokens],
                    "generate_seconds": gen_seconds,
                    "generated_tokens": generated,
                    "tokens_per_s": generated / gen_seconds if gen_seconds else None,
                }
            )
            if args.prefill_check:
                sequences = [p + c for p, c in zip(flat_prompts, flat_tokens, strict=True)]
                starts = [len(p) for p in flat_prompts]
                t1 = time.time()
                prefill = engine.logprobs(sequences, starts)
                result["prefill_seconds"] = time.time() - t1
                result["prefill_logprobs"] = prefill
                result["prefill_logprobs_hex"] = hex_rows(prefill)
                result["decode_vs_prefill_equal"] = result["prefill_logprobs_hex"] == result["logprobs_hex"]
                diffs = [
                    abs(a - b)
                    for ra, rb in zip(flat_logprobs, prefill, strict=True)
                    for a, b in zip(ra, rb, strict=True)
                ]
                result["decode_vs_prefill_max_abs_diff"] = max(diffs) if diffs else 0.0
                if args.lora_zero_check and strict:
                    import hashlib

                    import torch

                    state_dict: dict[str, torch.Tensor] = {}
                    from rl_engine.models.muse_glimmer.runtime.vllm_strict import LORA_SHAPES

                    for layer in (0, 51):
                        for name, (k, n) in LORA_SHAPES.items():
                            state_dict[f"layers.{layer}.{name}.lora_A"] = torch.randn(
                                args.lora_rank, k, dtype=torch.bfloat16
                            )
                            state_dict[f"layers.{layer}.{name}.lora_B"] = torch.zeros(
                                n, args.lora_rank, dtype=torch.bfloat16
                            )
                    digest = hashlib.sha256(b"zero-lora").hexdigest()
                    engine.load_lora(state_dict, digest)
                    with_lora = engine.logprobs(sequences, starts)
                    result["lora_zero_logprobs_hex"] = hex_rows(with_lora)
                    result["lora_zero_equal_to_prefill"] = result["lora_zero_logprobs_hex"] == result[
                        "prefill_logprobs_hex"
                    ]
                    result["lora_sha256"] = digest
                    engine.load_lora({}, "")
        result["meta"] = engine.provenance()
    finally:
        engine.close()
    Path(args.out).write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    summary = {k: v for k, v in result.items() if k in (
        "mode", "strict", "tp", "generated_tokens", "tokens_per_s", "generate_seconds", "score_seconds",
        "prefill_seconds", "decode_vs_prefill_equal", "decode_vs_prefill_max_abs_diff",
        "bits_equal_to_scored_file", "max_abs_diff_to_scored_file", "lora_zero_equal_to_prefill",
    )}
    print(json.dumps(summary, indent=2), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
