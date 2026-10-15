# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""CLI: strict (or native) Muse-Glimmer-30B token log-probabilities from Megatron Bridge.

Writes the same JSON as the vLLM agent's dump so the integrator can diff bit patterns::

    {"token_ids": [[...]], "logprobs": [[fp32...]], "logprobs_hex": [["3f8e...", ...]],
     "meta": {...provenance, checks...}}

Example (TP1, strict, with checks)::

    PYTHONPATH=. ~/bin/gpu-lease -n 1 --pool 6-7 -- ~/envs/megatron/bin/python -m \\
        rl_engine.models.muse_glimmer.runtime.run_megatron_strict --tp 1 --strict \\
        --prompt-file prompts.json --out logprobs_tp1.json --checks identity,repeat,native,step
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any

import torch

from rl_engine.models.muse_glimmer.runtime.megatron_strict import hex_bits
from rl_engine.models.muse_glimmer.runtime.strict_trainer import StrictTrainer


def _load_prompts(path: str, model_path: str, max_tokens: int) -> list[list[int]]:
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = None
    if isinstance(data, dict) and "token_ids" in data:
        seqs = [list(map(int, s)) for s in data["token_ids"]]
    elif isinstance(data, list) and data and isinstance(data[0], list):
        seqs = [list(map(int, s)) for s in data]
    else:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(model_path)
        lines = data if isinstance(data, list) else [ln for ln in text.splitlines() if ln.strip()]
        seqs = [tok(line, add_special_tokens=True)["input_ids"] for line in lines]
    return [s[:max_tokens] for s in seqs if len(s) >= 2]


def _dump(token_ids, logps, meta, out_path):
    payload = {
        "token_ids": token_ids,
        "logprobs": [lp.tolist() for lp in logps],
        "logprobs_hex": [hex_bits(lp) for lp in logps],
        "meta": meta,
    }
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=os.path.expanduser("~/models/Muse-Glimmer-30B"))
    ap.add_argument("--tp", type=int, default=1, choices=(1, 2))
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--strict", action="store_true", default=True)
    mode.add_argument("--native", action="store_true")
    ap.add_argument("--prompt-file", required=True, help="JSON {token_ids:[[...]]}, JSON list, or text lines")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--completion-start", type=int, default=1, help="first token position scored")
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-checkpoint", action="store_true", help="disable per-layer recompute")
    ap.add_argument(
        "--checks", default="",
        help="comma list of: identity, repeat, native, step, grad_repeat, lora_noop, bench",
    )
    ap.add_argument("--identity-out", default=None)
    ap.add_argument("--lora-out", default=None)
    args = ap.parse_args(argv)
    strict = not args.native
    checks = {c.strip() for c in args.checks.split(",") if c.strip()}

    seqs = _load_prompts(args.prompt_file, args.model, args.max_tokens)
    starts = [max(1, min(args.completion_start, len(s) - 1)) for s in seqs]
    trainer = StrictTrainer(
        args.model, tensor_parallel_size=args.tp, gpu_ids=list(range(args.tp)),
        lora_rank=args.lora_rank if strict else 0, lr=args.lr, strict=strict, seed=args.seed,
        checkpoint_layers=not args.no_checkpoint,
    )
    meta: dict[str, Any] = {"provenance": trainer.provenance(), "checks": {}, "timing": {}}
    meta["num_tokens"] = int(sum(len(s) for s in seqs))
    meta["completion_start"] = starts

    torch.cuda.synchronize()
    t0 = time.time()
    logps = trainer.token_logprobs(seqs, starts)
    torch.cuda.synchronize()
    meta["timing"]["forward_s"] = time.time() - t0
    logps_cpu = [lp.detach().float().cpu() for lp in logps]
    print(f"[run] forward {meta['timing']['forward_s']:.2f}s tokens={meta['num_tokens']}", flush=True)

    if "repeat" in checks:
        again = trainer.token_logprobs(seqs, starts)
        meta["checks"]["repeat_torch_equal"] = all(
            torch.equal(a.detach().cpu(), b) for a, b in zip(again, logps_cpu, strict=True)
        )
        print("[check] repeat:", meta["checks"]["repeat_torch_equal"], flush=True)
        logps = again

    if strict and "lora_noop" in checks:
        trainer.binding.lora_enabled = False
        without = trainer.token_logprobs(seqs, starts)
        trainer.binding.lora_enabled = True
        meta["checks"]["lora_zero_is_bitwise_noop"] = all(
            torch.equal(a.detach().cpu(), b) for a, b in zip(without, logps_cpu, strict=True)
        )
        print("[check] lora zero no-op:", meta["checks"]["lora_zero_is_bitwise_noop"], flush=True)
        logps = trainer.token_logprobs(seqs, starts)

    if strict and "native" in checks:
        from rl_engine.models.muse_glimmer.runtime.megatron_strict import native_token_logprobs

        diffs = []
        for seq, start, lp in zip(seqs, starts, logps_cpu, strict=True):
            nat = native_token_logprobs(trainer.model, torch.tensor([seq], dtype=torch.int64))[0, start - 1 :]
            diffs.append(float((nat.cpu() - lp).abs().max().item()))
        meta["checks"]["native_vs_strict_max_abs_diff"] = max(diffs)
        print("[check] native vs strict max abs diff:", max(diffs), flush=True)

    if strict and ("step" in checks or "grad_repeat" in checks):
        def loss_of(lps):
            return -torch.stack([lp.mean() for lp in lps]).mean()

        if "grad_repeat" in checks:
            params = [p for p in trainer.binding.lora[0].parameters()]
            (g1,) = torch.autograd.grad(loss_of(logps), [params[0]], retain_graph=True)
            logps2 = trainer.token_logprobs(seqs, starts)
            (g2,) = torch.autograd.grad(loss_of(logps2), [params[0]], retain_graph=True)
            meta["checks"]["lora_grad_repeat_torch_equal"] = bool(torch.equal(g1, g2))
            meta["checks"]["lora_grad_abs_max"] = float(g1.abs().max().item())
            print("[check] lora grad repeat:", meta["checks"]["lora_grad_repeat_torch_equal"], flush=True)
            logps = logps2
        torch.cuda.synchronize()
        t1 = time.time()
        stats = trainer.step(loss_of(logps))
        torch.cuda.synchronize()
        meta["timing"]["backward_step_s"] = time.time() - t1
        meta["checks"]["step"] = stats
        print("[step]", stats, f"{meta['timing']['backward_step_s']:.2f}s", flush=True)
        state, sha = trainer.export_lora()
        meta["checks"]["lora_export_sha256"] = sha
        meta["checks"]["lora_export_tensors"] = len(state)
        print("[export] sha256", sha, "tensors", len(state), flush=True)
        if args.lora_out:
            torch.save({"state_dict": state, "sha256": sha}, args.lora_out)
        after = trainer.token_logprobs(seqs, starts)
        meta["checks"]["post_step_max_abs_change"] = max(
            float((a.detach().cpu() - b).abs().max().item()) for a, b in zip(after, logps_cpu, strict=True)
        )

    if strict and "identity" in checks:
        from rl_engine.models.muse_glimmer.runtime.megatron_weights_identity import check_weight_identity

        t2 = time.time()
        rep = check_weight_identity(trainer.model, args.model, trainer.binding.comm)
        meta["checks"]["weight_identity"] = {
            "total": rep.total,
            "equal": rep.equal,
            "mismatches": [m.name for m in rep.mismatches],
            "seconds": time.time() - t2,
        }
        print(f"[identity] {rep.equal}/{rep.total} equal; mismatches={[m.name for m in rep.mismatches]}", flush=True)
        if args.identity_out:
            with open(args.identity_out, "w", encoding="utf-8") as fh:
                fh.write(rep.to_json())

    if "bench" in checks:
        torch.cuda.synchronize()
        t3 = time.time()
        lps = trainer.token_logprobs(seqs, starts)
        torch.cuda.synchronize()
        fwd = time.time() - t3
        meta["timing"]["bench_forward_s"] = fwd
        if strict and trainer.binding.lora is not None:
            loss = -torch.stack([lp.mean() for lp in lps]).mean()
            t4 = time.time()
            trainer.step(loss)
            torch.cuda.synchronize()
            meta["timing"]["bench_backward_step_s"] = time.time() - t4
        meta["timing"]["peak_mem_gib"] = torch.cuda.max_memory_allocated() / 2**30
        print("[bench]", meta["timing"], flush=True)

    _dump(seqs, logps_cpu, meta, args.out)
    print(f"[run] wrote {args.out}", flush=True)
    trainer.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
