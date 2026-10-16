from pathlib import Path

p = Path("/home/ubuntu/RL-Kernel/.claude/worktrees/wf_7b064961-36f-3/rl_engine/models/muse_glimmer/e2e/run_poc_grpo.py")
s = p.read_text()
rep = []

rep.append(('''Real run (one 8-GPU node; rollout TP2 on GPUs 0-1, trainer TP4 on GPUs 4-7):

    PYTHONPATH=. python -m rl_engine.models.muse_glimmer.e2e.run_poc_grpo \\\\
        --rollout-tp 2 --rollout-gpus 0,1 --train-tp 4 --train-gpus 4,5,6,7 \\\\
        --steps 8 --prompts-per-step 8 --samples-per-prompt 4 --max-tokens 256 \\\\
        --lora-rank 16 --lr 1e-5 --mode strict/strict --run-dir runs/poc/strict_strict

``--mode`` is ``<train>/<rollout>``: native/native, strict/native, native/strict, strict/strict.
"""''', '''Real run (lease the whole GPU set for this orchestrator; it runs in the vLLM env and starts the
trainer server with ``--trainer-python``; layout A = rollout TP1 on GPU 0 + train TP2 on 2,3):

    ~/bin/gpu-lease -n 8 --pool 0-7 -- env PYTHONPATH=$PWD ~/envs/vllm/bin/python \\\\
        -m rl_engine.models.muse_glimmer.e2e.run_poc_grpo \\\\
        --rollout-tp 1 --rollout-gpus 0 --train-tp 2 --train-gpus 2,3 \\\\
        --trainer-python ~/envs/megatron/bin/python --trainer-launcher process \\\\
        --steps 8 --prompts-per-step 8 --samples-per-prompt 4 --max-tokens 256 \\\\
        --lora-rank 16 --lr 1e-5 --mode strict/strict --run-dir runs/poc/strict_strict

``--mode`` is ``<train>/<rollout>``: native/native, strict/native, native/strict, strict/strict.

GPU placement is explicit: ``--rollout-gpus`` / ``--train-gpus`` are PHYSICAL ids inside the
lease. The orchestrator sets ``CUDA_VISIBLE_DEVICES`` for the processes it starts, from these
flags and nowhere else: the trainer server gets ``--train-gpus``; this process (whose children
are vLLM's engine-core / worker processes) gets ``--rollout-gpus`` before vLLM starts.
"""'''))

rep.append(('''from rl_engine.models.muse_glimmer.e2e.data import (''', '''from rl_engine.models.muse_glimmer.e2e.bridge import (
    DEFAULT_TRAINER_FACTORY,
    DEFAULT_TRAINER_PYTHON,
    LAUNCHERS,
    MOCK_TRAINER_FACTORY,
    RemoteTrainer,
    plan_gpu_placement,
)
from rl_engine.models.muse_glimmer.e2e.data import ('''))

rep.append(('''    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--rollout-tp", type=int, default=2)
    parser.add_argument("--rollout-gpus", type=_int_list, default=[0, 1])
    parser.add_argument("--train-tp", type=int, default=4)
    parser.add_argument("--train-gpus", type=_int_list, default=[4, 5, 6, 7])''', '''    parser.add_argument("--max-model-len", type=int, default=4096)
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
    )'''))

rep.append(('''    task = args.task or ("arith" if args.mock else "gsm8k")
    return LoopConfig(''', '''    task = args.task or ("arith" if args.mock else "gsm8k")
    bridge = args.trainer_bridge == "on" or (args.trainer_bridge == "auto" and not args.mock)
    factory = args.trainer_factory or (MOCK_TRAINER_FACTORY if args.mock else DEFAULT_TRAINER_FACTORY)
    return LoopConfig('''))

rep.append(('''        split=args.split,
        limit_examples=args.limit_examples,
    )''', '''        split=args.split,
        limit_examples=args.limit_examples,
        trainer_bridge=bridge,
        trainer_python=args.trainer_python,
        trainer_launcher=args.trainer_launcher,
        trainer_factory=factory,
        trainer_timeout_s=args.trainer_timeout,
        trainer_startup_timeout_s=args.trainer_startup_timeout,
        mock_trainer_device=args.mock_trainer_device,
        profile_step=args.profile_step,
    )'''))

start = s.index("def build_engines(cfg: LoopConfig)")
end = s.index("def build_task(")
new_build = '''def _remote_trainer(cfg: LoopConfig, *, gpu_ids: list[int], wait: bool) -> RemoteTrainer:
    """The trainer behind the cross-environment bridge (server in ``cfg.trainer_python``)."""

    factory_kwargs: dict[str, Any] = {}
    if cfg.mock and cfg.trainer_factory == MOCK_TRAINER_FACTORY:
        factory_kwargs["device"] = cfg.mock_trainer_device
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


'''
s = s[:start] + new_build + s[end:]

rep2 = [('''import argparse
import sys
import time''', '''import argparse
import os
import sys
import time''')]
for a, b in rep + rep2:
    assert a in s, a[:80]
    s = s.replace(a, b)
p.write_text(s)
print("ok")
