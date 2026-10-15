# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""CLI for the Muse-Glimmer-30B strict POC GRPO loop (``final_e2e_lora``).

Mock (CPU, no GPUs, no downloads unless ``--task gsm8k``):

    PYTHONPATH=. ~/envs/vllm/bin/python -m rl_engine.models.muse_glimmer.e2e.run_poc_grpo \\
        --mock --mode strict/strict --steps 3 --prompts-per-step 4 --samples-per-prompt 4 \\
        --max-tokens 32 --run-dir /tmp/poc_mock

Real run (lease the whole GPU set for this orchestrator; it runs in the vLLM env and starts the
trainer server with ``--trainer-python``; layout A = rollout TP1 on GPU 0 + train TP2 on 2,3):

    ~/bin/gpu-lease -n 8 --pool 0-7 -- env PYTHONPATH=$PWD ~/envs/vllm/bin/python \\
        -m rl_engine.models.muse_glimmer.e2e.run_poc_grpo \\
        --rollout-tp 1 --rollout-gpus 0 --train-tp 2 --train-gpus 2,3 \\
        --trainer-python ~/envs/megatron/bin/python --trainer-launcher process \\
        --steps 8 --prompts-per-step 8 --samples-per-prompt 4 --max-tokens 256 \\
        --lora-rank 16 --lr 1e-5 --mode strict/strict --run-dir runs/poc/strict_strict

``--mode`` is ``<train>/<rollout>``: native/native, strict/native, native/strict, strict/strict.

GPU placement is explicit: ``--rollout-gpus`` / ``--train-gpus`` are PHYSICAL ids inside the
lease. The orchestrator sets ``CUDA_VISIBLE_DEVICES`` for the processes it starts, from these
flags and nowhere else: the trainer server gets ``--train-gpus``; this process (whose children
are vLLM's engine-core / worker processes) gets ``--rollout-gpus`` before vLLM starts.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, Sequence

from rl_engine.models.muse_glimmer.e2e.bridge import (
    DEFAULT_TRAINER_FACTORY,
    DEFAULT_TRAINER_PYTHON,
    LAUNCHERS,
    MOCK_TRAINER_FACTORY,
    RemoteTrainer,
    plan_gpu_placement,
)
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
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
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
    parser.add_argument("--rollout-tp", type=int, default=1)
    parser.add_argument("--rollout-gpus", type=_int_list, default=[0], help="physical GPU ids")
    parser.add_argument("--train-tp", type=int, default=2)
    parser.add_argument("--train-gpus", type=_int_list, default=[2, 3], help="physical GPU ids")
    parser.add_argument(
        "--trainer-bridge",
        choices=("auto", "on", "off"),
        default="auto",
        help="run the trainer in a --trainer-python subprocess (auto: on for real engines)",
    )
    parser.add_argument(
        "--trainer-python",
        default=DEFAULT_TRAINER_PYTHON,
        help="Python of the trainer server (Megatron Bridge env)",
    )
    parser.add_argument(
        "--trainer-launcher",
        choices=LAUNCHERS,
        default="process",
        help="process: the trainer spawns its own TP ranks (today's StrictTrainer); "
        "torchrun: one server process per TP rank, rank 0 broadcasts every call",
    )
    parser.add_argument(
        "--trainer-factory",
        default=None,
        help=f"module:Class built by the server (default {DEFAULT_TRAINER_FACTORY}; "
        f"with --mock {MOCK_TRAINER_FACTORY})",
    )
    parser.add_argument("--trainer-timeout", type=float, default=3600.0, help="seconds per call")
    parser.add_argument(
        "--trainer-startup-timeout", type=float, default=3600.0, help="seconds to build"
    )
    parser.add_argument(
        "--mock-trainer-device",
        default="cpu",
        help="device of the bridged mock trainer (cuda uses --train-gpus)",
    )
    parser.add_argument(
        "--profile-step",
        type=int,
        default=None,
        help="torch.profiler top-ops tables for this step (rollout + trainer scoring/step)",
    )
    parser.add_argument(
        "--run-dir", default=None, help="default: runs/muse_glimmer_poc_grpo/<ts>_<mode>"
    )
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
    parser.add_argument(
        "--no-download", action="store_true", help="fail if GSM8K files are missing"
    )
    parser.add_argument("--mock", action="store_true", help="CPU mock engines (no GPUs)")
    parser.add_argument(
        "--mock-perturb", type=float, default=0.0, help="perturb mock rollout weights"
    )
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
        mode_tag = args.mode.replace("/", "-")
        run_dir = Path("runs/muse_glimmer_poc_grpo") / f"{stamp}_{mode_tag}{suffix}"
    else:
        run_dir = Path(args.run_dir)
    task = args.task or ("arith" if args.mock else "gsm8k")
    bridge = args.trainer_bridge == "on" or (args.trainer_bridge == "auto" and not args.mock)
    factory = args.trainer_factory or (
        MOCK_TRAINER_FACTORY if args.mock else DEFAULT_TRAINER_FACTORY
    )
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
        trainer_bridge=bridge,
        trainer_python=args.trainer_python,
        trainer_launcher=args.trainer_launcher,
        trainer_factory=factory,
        trainer_timeout_s=args.trainer_timeout,
        trainer_startup_timeout_s=args.trainer_startup_timeout,
        mock_trainer_device=args.mock_trainer_device,
        profile_step=args.profile_step,
    )


def _remote_trainer(cfg: LoopConfig, *, gpu_ids: list[int], wait: bool) -> RemoteTrainer:
    """The trainer behind the cross-environment bridge (server in ``cfg.trainer_python``)."""

    factory_kwargs: dict[str, Any] = {}
    extra_env: dict[str, str] = {}
    if cfg.mock and cfg.trainer_factory == MOCK_TRAINER_FACTORY:
        factory_kwargs["device"] = cfg.mock_trainer_device
        extra_env["OMP_NUM_THREADS"] = "4"  # the tiny mock drowns in thread overhead
    return RemoteTrainer(
        cfg.model_path,
        tensor_parallel_size=cfg.train_tp,
        gpu_ids=gpu_ids,
        lora_rank=cfg.lora_rank,
        lr=cfg.lr,
        strict=cfg.train_strict,
        seed=cfg.seed,
        python=cfg.trainer_python,
        launcher=cfg.trainer_launcher,
        factory=cfg.trainer_factory or DEFAULT_TRAINER_FACTORY,
        factory_kwargs=factory_kwargs,
        work_dir=Path(cfg.run_dir) / "trainer_server",
        startup_timeout_s=cfg.trainer_startup_timeout_s,
        call_timeout_s=cfg.trainer_timeout_s,
        extra_env=extra_env,
        wait=wait,
    )


def build_engines(cfg: LoopConfig) -> tuple[Any, Any, Any]:
    """Return ``(rollout, trainer, tokenizer)``; mock on CPU or the real strict engines.

    Real engines: the GPU placement is validated against this process's lease, the trainer
    server is launched first (``CUDA_VISIBLE_DEVICES`` = ``--train-gpus``) and loads while vLLM
    starts in this process (``CUDA_VISIBLE_DEVICES`` = ``--rollout-gpus``, inherited by vLLM's
    engine-core and worker processes); then the loop waits for the trainer to be ready.
    """

    train_strict, rollout_strict = cfg.train_strict, cfg.rollout_strict
    if cfg.mock:
        from rl_engine.models.muse_glimmer.e2e.mocks import (
            ByteTokenizer,
            MockRolloutEngine,
            MockTrainer,
        )

        if cfg.trainer_bridge:
            gpus = list(cfg.train_gpus) if cfg.mock_trainer_device.startswith("cuda") else []
            if gpus:
                plan_gpu_placement(
                    rollout_gpus=[],
                    rollout_tp=0,
                    train_gpus=gpus,
                    train_tp=len(gpus),
                    parent_visible=os.environ.get("CUDA_VISIBLE_DEVICES"),
                )
            trainer: Any = _remote_trainer(cfg, gpu_ids=gpus, wait=True)
        else:
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

    placement = plan_gpu_placement(
        rollout_gpus=cfg.rollout_gpus,
        rollout_tp=cfg.rollout_tp,
        train_gpus=cfg.train_gpus,
        train_tp=cfg.train_tp,
        parent_visible=os.environ.get("CUDA_VISIBLE_DEVICES"),
    )
    tokenizer = HFChatTokenizer(cfg.model_path)
    # Trainer first: its initial LoRA defines the step-0 policy and is synced before rollout 0.
    if cfg.trainer_bridge:
        trainer = _remote_trainer(cfg, gpu_ids=list(cfg.train_gpus), wait=False)
        rollout_cls = _rollout_class()
    else:
        rollout_cls, trainer_cls = load_real_engine_classes()
        trainer = trainer_cls(
            cfg.model_path,
            tensor_parallel_size=cfg.train_tp,
            gpu_ids=cfg.train_gpus,
            lora_rank=cfg.lora_rank,
            lr=cfg.lr,
            strict=train_strict,
            seed=cfg.seed,
        )
    try:
        # The ONLY place the orchestrator sets CUDA_VISIBLE_DEVICES for itself: before vLLM
        # starts, so its engine-core / worker children see exactly --rollout-gpus.
        os.environ["CUDA_VISIBLE_DEVICES"] = placement["rollout_cuda_visible_devices"]
        rollout = rollout_cls(
            cfg.model_path,
            tensor_parallel_size=cfg.rollout_tp,
            gpu_ids=cfg.rollout_gpus,
            strict=rollout_strict,
            max_model_len=cfg.max_model_len,
            seed=cfg.seed,
        )
        if isinstance(trainer, RemoteTrainer):
            trainer.wait_ready()
    except BaseException:
        trainer.close()
        raise
    return rollout, trainer, tokenizer


def _rollout_class() -> type:
    """``StrictRolloutEngine`` only (the trainer lives behind the bridge in another env)."""

    import importlib

    from rl_engine.models.muse_glimmer.e2e.engines import _ROLLOUT_MODULE, EngineUnavailable

    try:
        return importlib.import_module(_ROLLOUT_MODULE).StrictRolloutEngine
    except ImportError as exc:  # pragma: no cover - depends on the vLLM env
        raise EngineUnavailable(f"StrictRolloutEngine not importable ({exc})") from exc


def build_task(cfg: LoopConfig, tokenizer: Any, *, download: bool = True) -> Task:
    if cfg.task == "arith":
        return SyntheticArithmeticTask(tokenizer)
    data_dir = Path(cfg.data_dir).expanduser()
    path = data_dir / f"{cfg.split}.jsonl"
    if not path.exists():
        if not download:
            raise FileNotFoundError(
                f"{path} missing; run data.download_gsm8k or drop --no-download"
            )
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
