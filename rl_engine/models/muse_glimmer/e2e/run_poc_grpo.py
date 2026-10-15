# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""CLI for the Muse-Glimmer-30B strict POC GRPO loop (``final_e2e_lora``).

Mock (CPU, no GPUs, no downloads unless ``--task gsm8k``):

    PYTHONPATH=. ~/envs/vllm/bin/python -m rl_engine.models.muse_glimmer.e2e.run_poc_grpo \\
        --mock --mode strict/strict --steps 3 --prompts-per-step 4 --samples-per-prompt 4 \\
        --max-tokens 32 --run-dir /tmp/poc_mock

Real run (one 8-GPU node; rollout TP2 on GPUs 0-1, trainer TP4 on GPUs 4-7):

    PYTHONPATH=. python -m rl_engine.models.muse_glimmer.e2e.run_poc_grpo \\
        --rollout-tp 2 --rollout-gpus 0,1 --train-tp 4 --train-gpus 4,5,6,7 \\
        --steps 8 --prompts-per-step 8 --samples-per-prompt 4 --max-tokens 256 \\
        --lora-rank 16 --lr 1e-5 --mode strict/strict --run-dir runs/poc/strict_strict

``--mode`` is ``<train>/<rollout>``: native/native, strict/native, native/strict, strict/strict.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Sequence

from rl_engine.models.muse_glimmer.e2e.data import (
    DEFAULT_DATA_DIR,
    DEFAULT_MODEL_PATH,
    GSM8KTask,
    SyntheticArithmeticTask,
    Task,
    download_gsm8k,
    load_gsm8k,
)
from rl_engine.models.muse_glimmer.e2e.loop import MODES, LoopConfig, run_loop


def _int_list(text: str) -> list[int]:
    return [int(part) for part in text.split(",") if part.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", default="strict/strict", choices=MODES, help="<train>/<rollout>")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--prompts-per-step", type=int, default=8)
    parser.add_argument("--samples-per-prompt", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--kl-coef", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--rollout-tp", type=int, default=2)
    parser.add_argument("--rollout-gpus", type=_int_list, default=[0, 1])
    parser.add_argument("--train-tp", type=int, default=4)
    parser.add_argument("--train-gpus", type=_int_list, default=[4, 5, 6, 7])
    parser.add_argument("--run-dir", default=None, help="default: runs/muse_glimmer_poc_grpo/<ts>_<mode>")
    parser.add_argument("--sync-via-file", action="store_true", help="LoRA transfer through a file")
    parser.add_argument(
        "--no-recheck-rollout-logprobs",
        action="store_true",
        help="skip the rollout prefill-vs-decode self-check (rollout.logprobs on the same tokens)",
    )
    parser.add_argument("--task", choices=("gsm8k", "arith"), default=None)
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--split", choices=("train", "test"), default="train")
    parser.add_argument("--limit-examples", type=int, default=None)
    parser.add_argument("--no-download", action="store_true", help="fail if GSM8K files are missing")
    parser.add_argument("--mock", action="store_true", help="CPU mock engines (no GPUs)")
    parser.add_argument("--mock-perturb", type=float, default=0.0, help="perturb mock rollout weights")
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="torch CPU threads (default: 4 with --mock, untouched otherwise)",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> LoopConfig:
    if args.run_dir is None:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        suffix = "_mock" if args.mock else ""
        run_dir = Path("runs/muse_glimmer_poc_grpo") / f"{stamp}_{args.mode.replace('/', '-')}{suffix}"
    else:
        run_dir = Path(args.run_dir)
    task = args.task or ("arith" if args.mock else "gsm8k")
    return LoopConfig(
        run_dir=run_dir,
        steps=args.steps,
        prompts_per_step=args.prompts_per_step,
        samples_per_prompt=args.samples_per_prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        lora_rank=args.lora_rank,
        lr=args.lr,
        clip_eps=args.clip_eps,
        kl_coef=args.kl_coef,
        mode=args.mode,
        seed=args.seed,
        model_path=args.model_path,
        max_model_len=args.max_model_len,
        rollout_tp=args.rollout_tp,
        rollout_gpus=list(args.rollout_gpus),
        train_tp=args.train_tp,
        train_gpus=list(args.train_gpus),
        sync_via_file=args.sync_via_file,
        recheck_rollout_logprobs=not args.no_recheck_rollout_logprobs,
        mock=args.mock,
        mock_perturb=args.mock_perturb,
        task=task,
        data_dir=args.data_dir,
        split=args.split,
        limit_examples=args.limit_examples,
    )


def build_engines(cfg: LoopConfig) -> tuple[Any, Any, Any]:
    """Return ``(rollout, trainer, tokenizer)``; mock on CPU or the real strict engines."""

    train_strict, rollout_strict = cfg.train_strict, cfg.rollout_strict
    if cfg.mock:
        from rl_engine.models.muse_glimmer.e2e.mocks import (
            ByteTokenizer,
            MockRolloutEngine,
            MockTrainer,
        )

        trainer = MockTrainer(
            cfg.model_path,
            tensor_parallel_size=cfg.train_tp,
            gpu_ids=cfg.train_gpus,
            lora_rank=cfg.lora_rank,
            lr=cfg.lr,
            strict=train_strict,
            seed=cfg.seed,
        )
        rollout = MockRolloutEngine(
            cfg.model_path,
            tensor_parallel_size=cfg.rollout_tp,
            gpu_ids=cfg.rollout_gpus,
            strict=rollout_strict,
            max_model_len=cfg.max_model_len,
            seed=cfg.seed,
            lora_rank=cfg.lora_rank,
            perturb=cfg.mock_perturb,
        )
        return rollout, trainer, ByteTokenizer()

    from rl_engine.models.muse_glimmer.e2e.data import HFChatTokenizer
    from rl_engine.models.muse_glimmer.e2e.engines import load_real_engine_classes

    rollout_cls, trainer_cls = load_real_engine_classes()
    tokenizer = HFChatTokenizer(cfg.model_path)
    # Trainer first: its initial LoRA defines the step-0 policy and is synced before rollout 0.
    trainer = trainer_cls(
        cfg.model_path,
        tensor_parallel_size=cfg.train_tp,
        gpu_ids=cfg.train_gpus,
        lora_rank=cfg.lora_rank,
        lr=cfg.lr,
        strict=train_strict,
        seed=cfg.seed,
    )
    rollout = rollout_cls(
        cfg.model_path,
        tensor_parallel_size=cfg.rollout_tp,
        gpu_ids=cfg.rollout_gpus,
        strict=rollout_strict,
        max_model_len=cfg.max_model_len,
        seed=cfg.seed,
    )
    return rollout, trainer, tokenizer


def build_task(cfg: LoopConfig, tokenizer: Any, *, download: bool = True) -> Task:
    if cfg.task == "arith":
        return SyntheticArithmeticTask(tokenizer)
    data_dir = Path(cfg.data_dir).expanduser()
    path = data_dir / f"{cfg.split}.jsonl"
    if not path.exists():
        if not download:
            raise FileNotFoundError(f"{path} missing; run data.download_gsm8k or drop --no-download")
        download_gsm8k(data_dir, (cfg.split,))
    examples = load_gsm8k(path, limit=cfg.limit_examples)
    return GSM8KTask(examples, tokenizer, split=cfg.split)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = config_from_args(args)
    threads = args.threads if args.threads is not None else (4 if args.mock else None)
    if threads:
        import torch

        torch.set_num_threads(threads)  # the mock's tiny elementwise ops drown in thread overhead
    rollout, trainer, tokenizer = build_engines(cfg)
    try:
        task = build_task(cfg, tokenizer, download=not args.no_download)
        print(f"run dir: {cfg.run_dir}  mode: {cfg.mode}  task: {task.name}  mock: {cfg.mock}")
        summary = run_loop(cfg, task, rollout, trainer)
    finally:
        rollout.close()
        trainer.close()
    totals = summary["drift_totals"]
    print(
        f"done: {summary['steps_completed']} steps, bitwise mismatches "
        f"{totals['bitwise_mismatch_count']}/{totals['token_count']}, max |Δ| "
        f"{totals['max_abs_diff']:.3e}, first drift {summary['first_drift']}, reward mean "
        f"{summary['reward']['mean']}, weight sync {summary['weight_sync']['passed']}/"
        f"{summary['weight_sync']['checks']} -> {Path(cfg.run_dir) / 'summary.json'}"
    )
    return 0 if summary["status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
