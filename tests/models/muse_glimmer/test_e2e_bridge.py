# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""CPU tests of the cross-environment trainer bridge (row ``final_e2e_lora``).

The client runs in this interpreter (the vLLM env); the trainer server is a real subprocess
started with the Megatron env's Python (``~/envs/megatron/bin/python``) over the real transport,
holding the CPU ``MockTrainer``. Every server gets ``CUDA_VISIBLE_DEVICES=""`` (no GPU use).

    PYTHONPATH=<worktree> ~/envs/vllm/bin/python -m pytest \
        tests/models/muse_glimmer/test_e2e_bridge.py -q
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
import torch

from rl_engine.models.muse_glimmer.e2e import bridge, grpo, mocks, run_poc_grpo
from rl_engine.models.muse_glimmer.e2e.sync import WeightSyncError, lora_sha256

pytestmark = pytest.mark.unit

MEGATRON_PY = Path("~/envs/megatron/bin/python").expanduser()
requires_megatron_env = pytest.mark.skipif(
    not MEGATRON_PY.exists(), reason="~/envs/megatron/bin/python missing"
)
FAULTY = "rl_engine.models.muse_glimmer.e2e.mocks:FaultyMockTrainer"

SEQS = [[256, 3, 5, 7, 9, 11, 13], [256, 4, 6, 8, 10], [256, 1, 2, 3, 4, 5, 6, 7], [256, 9, 9]]
STARTS = [3, 2, 4, 1]
ADVANTAGES = torch.tensor([1.0, -1.0, 0.5, -0.5])


@pytest.fixture(autouse=True, scope="module")
def _few_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(min(4, old))
    yield
    torch.set_num_threads(old)


def _remote(tmp_path: Path, *, python: Path | str = MEGATRON_PY, launcher: str = "process",
            tp: int = 1, factory: str = bridge.MOCK_TRAINER_FACTORY, factory_kwargs=None,
            call_timeout_s: float = 120.0, startup_timeout_s: float = 120.0, name: str = "srv"):
    return bridge.RemoteTrainer(
        "mock", tensor_parallel_size=tp, gpu_ids=[], lora_rank=4, lr=1e-3, strict=True, seed=7,
        python=str(python), launcher=launcher, factory=factory,
        factory_kwargs=factory_kwargs or {}, work_dir=tmp_path / name,
        call_timeout_s=call_timeout_s, startup_timeout_s=startup_timeout_s,
        extra_env={"OMP_NUM_THREADS": "2"},
    )


def _local() -> mocks.MockTrainer:
    return mocks.MockTrainer(
        "mock", tensor_parallel_size=1, gpu_ids=[], lora_rank=4, lr=1e-3, strict=True, seed=7
    )


def _bits(t: torch.Tensor) -> bytes:
    return bridge.tensor_bytes(t)


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    # A zombie still answers kill(0); treat it as dead.
    try:
        with open(f"/proc/{pid}/stat", encoding="utf-8") as handle:
            return handle.read().split()[2] != "Z"
    except FileNotFoundError:
        return False


def _run_steps(trainer, steps: int = 3):
    """GRPO steps on fixed sequences; returns per-step (logprob bits, grad_norm, sha, sd)."""

    out = []
    for _ in range(steps):
        lps = trainer.token_logprobs(SEQS, STARTS)
        rollout = [t.detach().clone() for t in lps]
        loss = grpo.grpo_token_loss(lps, rollout, ADVANTAGES).loss
        info = trainer.step(loss)
        state_dict, sha = trainer.export_lora()
        out.append(([_bits(t) for t in lps], info["grad_norm"], sha, state_dict))
    return out


# --------------------------------------------------------------------------------------------
# Numerics of the hand-off (in process)
# --------------------------------------------------------------------------------------------


def _special_fp32(n: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    base = torch.randn(n, generator=g) * torch.logspace(-30, 30, n, dtype=torch.float32)
    special = torch.tensor(
        [0.0, -0.0, 1e-45, -1e-45, 1.1754942e-38, 3.4e38, -3.4e38, 1.0, -1.0, 2.0**-149],
        dtype=torch.float32,
    )
    k = min(n, special.numel())
    base[:k] = special[:k]
    return base


def test_surrogate_gradient_is_exactly_the_shipped_gradient() -> None:
    lengths = [7, 1, 33, 128, 5]
    packed_src = torch.randn(sum(lengths), dtype=torch.float32, requires_grad=True)
    packed = packed_src * 1.0  # a non-leaf "trainer" output, split into per-sample views
    logps = list(torch.split(packed, lengths))
    grads = [_special_fp32(n, seed=i) for i, n in enumerate(lengths)]
    surrogate = bridge.surrogate_loss(logps, grads)
    got = torch.autograd.grad(surrogate, logps, retain_graph=True)
    for g_got, g_want in zip(got, grads):
        assert _bits(g_got) == _bits(g_want)
    # Through the split: the trainer's packed tensor receives cat(g) bit for bit.
    (g_packed,) = torch.autograd.grad(surrogate, [packed])
    assert _bits(g_packed) == _bits(torch.cat(grads))


def test_surrogate_matches_in_process_backward_on_a_real_grpo_loss() -> None:
    trainer = _local()
    lps = trainer.token_logprobs(SEQS, STARTS)
    rollout = [t.detach().clone() + 1e-3 for t in lps]
    loss = grpo.grpo_token_loss(lps, rollout, ADVANTAGES).loss
    direct = torch.autograd.grad(loss, trainer.model.lora_parameters(), retain_graph=True,
                                 allow_unused=True)
    leaves = [t.detach().clone().requires_grad_(True) for t in lps]
    grpo.grpo_token_loss(leaves, rollout, ADVANTAGES).loss.backward()
    surrogate = bridge.surrogate_loss(lps, [leaf.grad for leaf in leaves])
    via = torch.autograd.grad(surrogate, trainer.model.lora_parameters(), allow_unused=True)
    for a, b in zip(direct, via):
        if a is None:
            assert b is None
            continue
        assert _bits(a) == _bits(b)


def test_surrogate_rejects_mismatches_and_forward_only_logprobs() -> None:
    lp = torch.zeros(3, requires_grad=True) * 1.0
    with pytest.raises(ValueError, match="shape"):
        bridge.surrogate_loss([lp], [torch.zeros(4)])
    with pytest.raises(TypeError, match="dtype"):
        bridge.surrogate_loss([lp], [torch.zeros(3, dtype=torch.float64)])
    with pytest.raises(RuntimeError, match="autograd history"):
        bridge.surrogate_loss([torch.zeros(3)], [torch.zeros(3)])


def test_wire_roundtrip_is_bit_exact() -> None:
    fp32 = _special_fp32(64, seed=3)
    fp32[10] = float("inf")
    fp32[11] = float("-inf")
    nan_bits = torch.tensor([0x7FC00001, 0x7FA00000, -0x00400001], dtype=torch.int32)
    fp32[12:15] = nan_bits.view(torch.float32)
    bf16_bits = torch.arange(-32768, 32768, 37, dtype=torch.int32).to(torch.int16)
    bf16 = bf16_bits.view(torch.bfloat16)
    big = torch.arange(1000, dtype=torch.float32)
    view = big[100:200]  # storage offset -> compacted on the wire
    payload = {"fp32": fp32, "bf16": bf16, "view": view, "ver": torch.__version__,
               "nested": [{"x": 1, "y": (2.5, None)}], "path": Path("/x")}
    data = bridge.encode_message(bridge.to_wire(payload))
    back = bridge.decode_message(data)
    assert _bits(back["fp32"]) == _bits(fp32)
    assert _bits(back["bf16"]) == _bits(bf16) and back["bf16"].dtype == torch.bfloat16
    assert _bits(back["view"]) == _bits(view) and back["view"].untyped_storage().nbytes() == 400
    assert back["ver"] == str(torch.__version__) and type(back["ver"]) is str
    assert back["nested"] == [{"x": 1, "y": [2.5, None]}] and back["path"] == "/x"
    assert len(data) < 64 * 4 + 1024 + 2 * 1772 + 400 + 4096


# --------------------------------------------------------------------------------------------
# Placement and command construction
# --------------------------------------------------------------------------------------------


def test_plan_gpu_placement_validates_flags() -> None:
    plan = bridge.plan_gpu_placement(rollout_gpus=[0], rollout_tp=1, train_gpus=[2, 3],
                                     train_tp=2, parent_visible="0,1,2,3,4,5,6,7")
    assert plan["rollout_cuda_visible_devices"] == "0"
    assert plan["train_cuda_visible_devices"] == "2,3"
    with pytest.raises(ValueError, match="overlap"):
        bridge.plan_gpu_placement(rollout_gpus=[0, 1], rollout_tp=2, train_gpus=[1, 2],
                                  train_tp=2, parent_visible=None)
    with pytest.raises(ValueError, match="outside"):
        bridge.plan_gpu_placement(rollout_gpus=[0], rollout_tp=1, train_gpus=[6, 7],
                                  train_tp=2, parent_visible="0,1,2,3")
    with pytest.raises(ValueError, match="TP2"):
        bridge.plan_gpu_placement(rollout_gpus=[0], rollout_tp=2, train_gpus=[2, 3],
                                  train_tp=2, parent_visible=None)
    with pytest.raises(ValueError, match="repeat"):
        bridge.plan_gpu_placement(rollout_gpus=[0], rollout_tp=1, train_gpus=[3, 3],
                                  train_tp=2, parent_visible=None)
    assert bridge.parse_visible_devices("") == []
    assert bridge.parse_visible_devices("GPU-abc") is None


def test_child_env_and_server_command(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLLM_FOO", "1")
    monkeypatch.setenv("RL_KERNEL_MUSEGLIMMER_STRICT", "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    env = bridge.child_env([2, 3], {"X": "y"})
    assert env["CUDA_VISIBLE_DEVICES"] == "2,3"
    assert "VLLM_FOO" not in env and "RL_KERNEL_MUSEGLIMMER_STRICT" not in env
    assert env["PYTHONPATH"].split(os.pathsep)[0] == str(bridge.REPO_ROOT)
    assert env["X"] == "y"
    assert bridge.child_env([])["CUDA_VISIBLE_DEVICES"] == ""
    cmd = bridge.server_command("py", "process", 2, Path("s.json"))
    assert cmd == ["py", "-m", bridge.SERVER_MODULE, "--spec", "s.json"]
    cmd = bridge.server_command("py", "torchrun", 2, Path("s.json"))
    assert cmd[:3] == ["py", "-m", "torch.distributed.run"] and "--nproc-per-node=2" in cmd
    assert cmd[-4:] == ["--module", bridge.SERVER_MODULE, "--spec", "s.json"]


# --------------------------------------------------------------------------------------------
# Real transport, real subprocess
# --------------------------------------------------------------------------------------------


@requires_megatron_env
@pytest.mark.parametrize("which", ["megatron_env", "same_env"])
def test_remote_trainer_equals_in_process_trainer_bitwise(tmp_path: Path, which: str) -> None:
    python = MEGATRON_PY if which == "megatron_env" else sys.executable
    remote = _remote(tmp_path, python=python)
    try:
        hello = remote.wait_ready()
        assert hello["devices"]["device_count"] == 0
        assert hello["devices"]["cuda_visible_devices"] == ""
        if which == "megatron_env":
            assert "cu129" in hello["versions"]["torch"]
        got = _run_steps(remote)
        prov = remote.provenance()
    finally:
        remote.close()
    want = _run_steps(_local())
    assert len(got) == len(want) == 3
    for (lp_r, gn_r, sha_r, sd_r), (lp_l, gn_l, sha_l, sd_l) in zip(got, want):
        assert lp_r == lp_l  # trainer logprob bits identical
        assert gn_r == gn_l  # same gradient norm (same gradients)
        assert sha_r == sha_l == lora_sha256(sd_r)
        assert all(_bits(sd_r[k]) == _bits(sd_l[k]) for k in sd_l)
    assert len({sha for _, _, sha, _ in got}) == 3  # the LoRA moves every step
    assert prov["engine"] == "MockTrainer" and prov["bridge"]["launcher"] == "process"
    assert not remote.alive()


@requires_megatron_env
def test_lora_export_bit_exact_and_hash_lie_detected(tmp_path: Path) -> None:
    remote = _remote(tmp_path)
    try:
        sd_remote, sha_remote = remote.export_lora()
    finally:
        remote.close()
    sd_local, sha_local = _local().export_lora()
    assert sha_remote == sha_local
    assert sorted(sd_remote) == sorted(sd_local)
    for name in sd_local:
        assert sd_remote[name].dtype == torch.bfloat16
        assert _bits(sd_remote[name]) == _bits(sd_local[name])
    assert remote.last_export["num_tensors"] == len(sd_local)

    liar = _remote(tmp_path, factory=FAULTY, factory_kwargs={"fault": "bad_hash",
                                                            "fault_op": "export_lora"}, name="liar")
    try:
        with pytest.raises(WeightSyncError, match="hash mismatch"):
            liar.export_lora()
    finally:
        liar.close()


@requires_megatron_env
def test_ping_large_payload_round_trips_bit_exact(tmp_path: Path) -> None:
    remote = _remote(tmp_path)
    try:
        payload = torch.randint(0, 256, (8 << 20,), dtype=torch.uint8)
        echoed, stats = remote.ping(payload)
        assert torch.equal(echoed, payload)
        assert stats["request_bytes"] > 8 << 20 and stats["response_bytes"] > 8 << 20
    finally:
        remote.close()


@requires_megatron_env
def test_server_crash_gives_clear_error_and_no_hang(tmp_path: Path) -> None:
    remote = _remote(tmp_path, factory=FAULTY, factory_kwargs={"fault": "exit", "fault_op": "step"})
    lps = remote.token_logprobs(SEQS, STARTS)
    loss = grpo.grpo_token_loss(lps, [t.detach() for t in lps], ADVANTAGES).loss
    t0 = time.monotonic()
    with pytest.raises(bridge.RemoteTrainerError, match="exit code 17") as info:
        remote.step(loss)
    assert time.monotonic() - t0 < 30.0
    assert "died during 'step'" in str(info.value)
    assert not remote.alive()
    with pytest.raises(bridge.RemoteTrainerError, match="unusable"):
        remote.provenance()
    remote.close()


@requires_megatron_env
def test_hung_server_times_out_and_is_killed(tmp_path: Path) -> None:
    remote = _remote(tmp_path, factory=FAULTY, call_timeout_s=3.0,
                     factory_kwargs={"fault": "hang", "fault_op": "token_logprobs"})
    pid = remote.server_pid
    t0 = time.monotonic()
    with pytest.raises(bridge.RemoteTrainerTimeout, match="within 3.0s"):
        remote.token_logprobs(SEQS, STARTS)
    elapsed = time.monotonic() - t0
    assert 3.0 <= elapsed < 30.0
    assert not _pid_alive(pid)
    remote.close()


@requires_megatron_env
def test_remote_exception_propagates_and_server_survives(tmp_path: Path) -> None:
    remote = _remote(
        tmp_path, factory=FAULTY, factory_kwargs={"fault": "raise", "fault_op": "step"}
    )
    try:
        lps = remote.token_logprobs(SEQS, STARTS)
        loss = grpo.grpo_token_loss(lps, [t.detach() for t in lps], ADVANTAGES).loss
        with pytest.raises(bridge.RemoteTrainerError, match="injected fault in step"):
            remote.step(loss)
        assert remote.alive()
        assert remote.provenance()["engine"] == "MockTrainer"
        with pytest.raises(bridge.RemoteTrainerError, match="unknown op"):
            remote._call("no_such_op", {})
        with pytest.raises(RuntimeError, match="token_logprobs before step"):
            remote.step(loss)  # the failed step consumed nothing on the client side
    finally:
        remote.close()
    assert not remote.alive()


@requires_megatron_env
@pytest.mark.parametrize(
    "factory,kwargs,pattern",
    [
        ("rl_engine.models.muse_glimmer.e2e.mocks:NoSuchTrainer", {}, "NoSuchTrainer"),
        (FAULTY, {"fault": "build"}, "injected fault: trainer construction failed"),
    ],
)
def test_startup_failure_reports_remote_traceback(tmp_path: Path, factory, kwargs, pattern) -> None:
    t0 = time.monotonic()
    with pytest.raises(bridge.RemoteTrainerError, match=pattern):
        _remote(tmp_path, factory=factory, factory_kwargs=kwargs)
    assert time.monotonic() - t0 < 60.0


CLIENT_SCRIPT = """
import sys, time
from pathlib import Path
from rl_engine.models.muse_glimmer.e2e import bridge
r = bridge.RemoteTrainer("mock", tensor_parallel_size=1, gpu_ids=[], lora_rank=4, lr=1e-3,
                         strict=True, seed=7, python=sys.argv[1],
                         factory=bridge.MOCK_TRAINER_FACTORY, work_dir=Path(sys.argv[2]))
print(r.server_pid, flush=True)
time.sleep(600)
"""


@requires_megatron_env
def test_server_terminates_when_client_is_killed(tmp_path: Path) -> None:
    env = dict(os.environ, PYTHONPATH=str(bridge.REPO_ROOT))
    client = subprocess.Popen(
        [sys.executable, "-c", CLIENT_SCRIPT, str(MEGATRON_PY), str(tmp_path / "orphan")],
        stdout=subprocess.PIPE, text=True, env=env,
    )
    try:
        server_pid = int(client.stdout.readline())
        assert _pid_alive(server_pid)
        client.send_signal(signal.SIGKILL)
        client.wait(timeout=10)
        deadline = time.monotonic() + 20.0
        while _pid_alive(server_pid) and time.monotonic() < deadline:
            time.sleep(0.2)
        assert not _pid_alive(server_pid), "trainer server outlived its client"
    finally:
        if client.poll() is None:
            client.kill()


@requires_megatron_env
def test_torchrun_tp2_all_ranks_execute_and_match(tmp_path: Path) -> None:
    remote = _remote(tmp_path, launcher="torchrun", tp=2, startup_timeout_s=180.0)
    try:
        hello = remote.wait_ready()
        assert hello["world"] == 2
        got = _run_steps(remote, steps=2)
        last_calls = remote.calls[-3:]
        prov = remote.provenance()
    finally:
        remote.close()
    want = _run_steps(_local(), steps=2)
    for (lp_r, gn_r, sha_r, _), (lp_l, gn_l, sha_l, _) in zip(got, want):
        assert lp_r == lp_l and gn_r == gn_l and sha_r == sha_l
    for call in last_calls:
        ranks = call["server"]["ranks"]
        assert [r["rank"] for r in ranks] == [0, 1]
        assert ranks[0]["op_counts"] == ranks[1]["op_counts"]  # same methods, same order
    assert remote.last_export["rank_sha256"] == [sha_r, sha_r]
    assert prov["bridge"]["launcher"] == "torchrun"
    assert not remote.alive()


@requires_megatron_env
def test_torchrun_rank1_crash_is_reported(tmp_path: Path) -> None:
    remote = _remote(tmp_path, launcher="torchrun", tp=2, startup_timeout_s=180.0,
                     call_timeout_s=120.0, factory=FAULTY,
                     factory_kwargs={"fault": "exit", "fault_op": "step", "fault_rank": 1})
    pid = remote.server_pid
    lps = remote.token_logprobs(SEQS, STARTS)
    loss = grpo.grpo_token_loss(lps, [t.detach() for t in lps], ADVANTAGES).loss
    t0 = time.monotonic()
    with pytest.raises(bridge.RemoteTrainerError, match="died during 'step'"):
        remote.step(loss)
    assert time.monotonic() - t0 < 60.0
    assert not _pid_alive(pid)
    remote.close()


# --------------------------------------------------------------------------------------------
# The GRPO CLI through the bridge
# --------------------------------------------------------------------------------------------


def _cli(run_dir: Path, *extra: str) -> dict:
    code = run_poc_grpo.main(
        ["--mock", "--mode", "strict/strict", "--steps", "3", "--prompts-per-step", "2",
         "--samples-per-prompt", "2", "--max-tokens", "8", "--lora-rank", "4", "--run-dir",
         str(run_dir), "--threads", "2", "--trainer-python", str(MEGATRON_PY), *extra]
    )
    assert code == 0
    return json.loads((run_dir / "summary.json").read_text())


@requires_megatron_env
def test_cli_mock_through_bridge_equals_in_process_and_writes_timeline(tmp_path: Path) -> None:
    local = _cli(tmp_path / "local", "--trainer-bridge", "off")
    remote = _cli(tmp_path / "remote", "--trainer-bridge", "on", "--profile-step", "1")
    assert local["config"]["trainer_bridge"] is False and remote["config"]["trainer_bridge"]
    for key in ("bitwise_mismatch_count", "token_count", "max_abs_diff"):
        assert remote["drift_totals"][key] == local["drift_totals"][key]
    assert remote["drift_totals"]["bitwise_mismatch_count"] == 0
    assert remote["reward"]["trajectory"] == local["reward"]["trajectory"]
    assert remote["weight_sync"]["final_sha256"] == local["weight_sync"]["final_sha256"]
    assert remote["weight_sync"]["all_passed"]
    timeline = remote["timeline"]
    assert timeline["phase_order"][0] == "rollout_generate" and timeline["steps"] == [0, 1, 2]
    assert timeline["profiled_steps"] == [1]
    assert all(len(series) == 3 for series in timeline["phases_s"].values())
    assert set(timeline["bridge"]) >= {"token_logprobs", "step", "export_lora"}
    assert timeline["bridge"]["export_lora"]["count"] == 3
    assert local["timeline"]["bridge"] is None
    records = [json.loads(line) for line in (tmp_path / "remote" / "steps.jsonl").read_text()
               .splitlines()]
    first = records[0]
    assert first["trainer_step"]["bridge"]["grad_elements"] == first["drift"]["token_count"]
    assert records[0]["timeline"]["tokens_generated"] == records[0]["num_completion_tokens"]
    profile = remote["profiles"][0]
    paths = [Path(profile["orchestrator_rollout_generate"]["path"])]
    paths += [Path(entry["path"]) for entry in profile["trainer_server"]]
    assert {entry["op"] for entry in profile["trainer_server"]} == {"token_logprobs", "step"}
    for path in paths:
        table = json.loads(path.read_text())
        assert table["schema"] == "muse_glimmer_top_ops_v1" and table["rows"]
    trainer_prov = json.loads((tmp_path / "remote" / "provenance.json").read_text())
    assert trainer_prov["trainer_start"]["bridge"]["server_cuda_visible_devices"] == ""
