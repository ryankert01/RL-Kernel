"""Parse the REPORT's integrator command (no GPU, no engines)."""
import os

from rl_engine.models.muse_glimmer.e2e import bridge, run_poc_grpo

home = os.path.expanduser("~")
common = (f"--trainer-python {home}/envs/megatron/bin/python --trainer-launcher process "
          "--steps 8 --prompts-per-step 8 --samples-per-prompt 4 --max-tokens 256 "
          "--lora-rank 16 --lr 1e-5 --seed 1234").split()
for gpus, mode in ((("0,1", "2,3"), "strict/strict"), (("4,5", "6,7"), "native/native")):
    argv = ["--rollout-tp", "2", "--rollout-gpus", gpus[0], "--train-tp", "2", "--train-gpus",
            gpus[1], *common, "--mode", mode, "--run-dir", "/nonexistent/x"]
    cfg = run_poc_grpo.config_from_args(run_poc_grpo.build_parser().parse_args(argv))
    print(mode, cfg.rollout_tp, cfg.rollout_gpus, cfg.train_tp, cfg.train_gpus,
          cfg.train_strict, cfg.rollout_strict, cfg.trainer_bridge, cfg.trainer_launcher)
    lease = ",".join([*gpus[0].split(","), *gpus[1].split(",")])
    print(" placement", bridge.plan_gpu_placement(rollout_gpus=cfg.rollout_gpus, rollout_tp=2,
                                                 train_gpus=cfg.train_gpus, train_tp=2,
                                                 parent_visible=lease))
# Flag-less inside a 4-GPU lease (REPORT: "optional inside a 4-GPU lease").
print("flagless", bridge.default_gpu_ids(rollout_tp=2, train_tp=2, parent_visible="4,5,6,7"))
