import sys
from pathlib import Path

p = Path(sys.argv[1]) / "rl_engine/models/muse_glimmer/e2e/run_poc_grpo.py"
s = p.read_text()


def rep(old, new, count=1):
    global s
    assert s.count(old) == count, (s.count(old), old[:80])
    s = s.replace(old, new)


rep('''Real run (lease the whole GPU set for this orchestrator; it runs in the vLLM env and starts the
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
"""''', '''Real run (lease exactly the GPUs of one run for this orchestrator; it runs in the vLLM env and
starts the trainer server with ``--trainer-python``). Rollout TP2 + train TP2 on GPUs 0-3:

    ~/bin/gpu-lease -n 4 --pool 0-3 -- env PYTHONPATH=$PWD ~/envs/vllm/bin/python \\\\
        -m rl_engine.models.muse_glimmer.e2e.run_poc_grpo \\\\
        --rollout-tp 2 --train-tp 2 \\\\
        --trainer-python ~/envs/megatron/bin/python --trainer-launcher process \\\\
        --steps 8 --prompts-per-step 8 --samples-per-prompt 4 --max-tokens 256 \\\\
        --lora-rank 16 --lr 1e-5 --mode strict/strict --run-dir runs/poc/strict_strict

``--mode`` is ``<train>/<rollout>``: native/native, strict/native, native/strict, strict/strict.

GPU placement: ``--rollout-gpus`` / ``--train-gpus`` are PHYSICAL ids inside the lease. Omitted,
they default to the lease in order (rollout = the first ``--rollout-tp`` leased ids, trainer =
the next ``--train-tp``), so ``gpu-lease -n 3`` works without knowing which ids it will pick.
The orchestrator sets ``CUDA_VISIBLE_DEVICES`` for the processes it starts, from these ids and
nowhere else: the trainer server gets the train ids; this process (whose children are vLLM's
engine-core / worker processes) gets the rollout ids before vLLM starts. The trainer server is
launched first and builds while vLLM starts (``bridge.RemoteTrainer(wait=False)``).
""''' + '"')

rep('''    parser.add_argument("--rollout-tp", type=int, default=1)
    parser.add_argument("--rollout-gpus", type=_int_list, default=[0], help="physical GPU ids")
    parser.add_argument("--train-tp", type=int, default=2)
    parser.add_argument("--train-gpus", type=_int_list, default=[2, 3], help="physical GPU ids")''', '''    parser.add_argument("--rollout-tp", type=int, default=1)
    parser.add_argument(
        "--rollout-gpus",
        type=_int_list,
        default=None,
        help="physical GPU ids (default: the first --rollout-tp ids of the lease)",
    )
    parser.add_argument("--train-tp", type=int, default=2)
    parser.add_argument(
        "--train-gpus",
        type=_int_list,
        default=None,
        help="physical GPU ids (default: the next --train-tp ids of the lease)",
    )''')

rep('''def config_from_args(args: argparse.Namespace) -> LoopConfig:
    if args.run_dir is None:''', '''def _placement(args: argparse.Namespace) -> tuple[list[int], list[int]]:
    """Physical GPU ids of the rollout and the trainer (omitted flags come from the lease)."""

    gpu_trainer = args.mock and str(args.mock_trainer_device).startswith("cuda")
    if args.mock and not gpu_trainer:
        # CPU mocks: no GPU is used; record the flags (or nothing) for the config only.
        return list(args.rollout_gpus or []), list(args.train_gpus or [])
    if args.mock:
        _, train = default_gpu_ids(
            rollout_tp=0,
            train_tp=args.train_tp,
            parent_visible=os.environ.get("CUDA_VISIBLE_DEVICES"),
            rollout_gpus=[],
            train_gpus=args.train_gpus,
        )
        return list(args.rollout_gpus or []), train
    return default_gpu_ids(
        rollout_tp=args.rollout_tp,
        train_tp=args.train_tp,
        parent_visible=os.environ.get("CUDA_VISIBLE_DEVICES"),
        rollout_gpus=args.rollout_gpus,
        train_gpus=args.train_gpus,
    )


def config_from_args(args: argparse.Namespace) -> LoopConfig:
    rollout_gpus, train_gpus = _placement(args)
    if args.run_dir is None:''')

rep('''        rollout_tp=args.rollout_tp,
        rollout_gpus=list(args.rollout_gpus),
        train_tp=args.train_tp,
        train_gpus=list(args.train_gpus),''', '''        rollout_tp=args.rollout_tp,
        rollout_gpus=rollout_gpus,
        train_tp=args.train_tp,
        train_gpus=train_gpus,''')

rep('''    MOCK_TRAINER_FACTORY,
    RemoteTrainer,
    plan_gpu_placement,
)''', '''    MOCK_TRAINER_FACTORY,
    RemoteTrainer,
    default_gpu_ids,
    plan_gpu_placement,
)''')

rep('''    Real engines: the GPU placement is validated against this process's lease, the trainer
    server is launched first (``CUDA_VISIBLE_DEVICES`` = ``--train-gpus``) and loads while vLLM
    starts in this process (``CUDA_VISIBLE_DEVICES`` = ``--rollout-gpus``, inherited by vLLM's
    engine-core and worker processes); then the loop waits for the trainer to be ready.''', '''    Real engines: the GPU placement is validated against this process's lease, the trainer
    server is launched first (``CUDA_VISIBLE_DEVICES`` = train ids) and builds the trainer right
    away, while vLLM starts in this process (``CUDA_VISIBLE_DEVICES`` = rollout ids, inherited
    by vLLM's engine-core and worker processes); then ``wait_ready`` connects and waits for the
    rest of the trainer build (``provenance["trainer_start"]["bridge"]["hello"]["startup"]``
    records the overlap).''')
p.write_text(s)
print("patched", p)
