from pathlib import Path

W = Path("/home/ubuntu/RL-Kernel/.claude/worktrees/wf_7b064961-36f-3")
fixes = {
    "rl_engine/models/muse_glimmer/e2e/bench_bridge.py": [
        ('''        commands.append("python -m rl_engine.models.muse_glimmer.e2e.run_poc_grpo " + " ".join(argv))''',
         '''        commands.append(
            "python -m rl_engine.models.muse_glimmer.e2e.run_poc_grpo " + " ".join(argv)
        )'''),
    ],
    "rl_engine/models/muse_glimmer/e2e/bridge.py": [
        ('''    """``sum_i (logprobs[i] * grads[i]).sum()``; its gradient w.r.t. ``logprobs[i]`` is ``grads[i]``.''',
         '''    """``sum_i (logprobs[i] * grads[i]).sum()``; d/d ``logprobs[i]`` is exactly ``grads[i]``.'''),
        ('''                f"gpu_ids {self.gpu_ids} must list one GPU per TP rank (TP{self.tensor_parallel_size})"''',
         '''                f"gpu_ids {self.gpu_ids} must list one GPU per TP rank "
                f"(TP{self.tensor_parallel_size})"'''),
        ('''        self.command = server_command(self.python, launcher, self.tensor_parallel_size, self.spec_path)''',
         '''        self.command = server_command(
            self.python, launcher, self.tensor_parallel_size, self.spec_path
        )'''),
        ('''            raise RemoteTrainerError(f"trainer bridge unusable after an earlier failure: {self._broken}")''',
         '''            raise RemoteTrainerError(
                f"trainer bridge unusable after an earlier failure: {self._broken}"
            )'''),
    ],
    "rl_engine/models/muse_glimmer/e2e/loop.py": [
        ('''                    trainer.arm_profile(profile_dir, f"step{step:04d}")  # type: ignore[attr-defined]''',
         '''                    arm = getattr(trainer, "arm_profile")
                    arm(profile_dir, f"step{step:04d}")'''),
    ],
    "rl_engine/models/muse_glimmer/e2e/run_poc_grpo.py": [
        ('''    factory = args.trainer_factory or (MOCK_TRAINER_FACTORY if args.mock else DEFAULT_TRAINER_FACTORY)''',
         '''    factory = args.trainer_factory or (
        MOCK_TRAINER_FACTORY if args.mock else DEFAULT_TRAINER_FACTORY
    )'''),
    ],
    "rl_engine/models/muse_glimmer/e2e/trainer_server.py": [
        ('''                errors = [f"rank {s['rank']}: {s.get('error')}" for s in statuses if not s.get("ok")]''',
         '''                errors = [
                    f"rank {s['rank']}: {s.get('error')}" for s in statuses if not s.get("ok")
                ]'''),
    ],
    "tests/models/muse_glimmer/test_e2e_bridge.py": [
        ('''    PYTHONPATH=<worktree> ~/envs/vllm/bin/python -m pytest tests/models/muse_glimmer/test_e2e_bridge.py -q''',
         '''    PYTHONPATH=<worktree> ~/envs/vllm/bin/python -m pytest \\
        tests/models/muse_glimmer/test_e2e_bridge.py -q'''),
        ('''        assert hello["devices"]["device_count"] == 0 and hello["devices"]["cuda_visible_devices"] == ""''',
         '''        assert hello["devices"]["device_count"] == 0
        assert hello["devices"]["cuda_visible_devices"] == ""'''),
        ('''    remote = _remote(tmp_path, factory=FAULTY, factory_kwargs={"fault": "raise", "fault_op": "step"})''',
         '''    remote = _remote(
        tmp_path, factory=FAULTY, factory_kwargs={"fault": "raise", "fault_op": "step"}
    )'''),
        ('''    assert records[0]["trainer_step"]["bridge"]["grad_elements"] == records[0]["drift"]["token_count"]''',
         '''    first = records[0]
    assert first["trainer_step"]["bridge"]["grad_elements"] == first["drift"]["token_count"]'''),
    ],
}
for rel, pairs in fixes.items():
    p = W / rel
    s = p.read_text()
    for a, b in pairs:
        assert a in s, (rel, a[:60])
        s = s.replace(a, b)
    p.write_text(s)
print("ok")
