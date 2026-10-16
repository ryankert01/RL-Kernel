import sys
from pathlib import Path

p = Path(sys.argv[1]) / "rl_engine/models/muse_glimmer/e2e/bench_bridge.py"
s = p.read_text()


def rep(old, new, count=1):
    global s
    assert s.count(old) == count, (s.count(old), old[:80])
    s = s.replace(old, new)


rep('''* ``bridge_equivalence``: RemoteTrainer (server in the Megatron env / in this env; process and
  torchrun TP2 launchers) vs the same trainer in process: per step logprob bit mismatches,
  grad norms, LoRA hash and byte mismatches.
* ``transport``: echo round trips of 1 KiB .. 256 MiB uint8 payloads and of a LoRA state dict
  with the REAL Muse-Glimmer-30B rank-16 key schema and shapes (832 bf16 tensors, 200 MiB):
  RTT, client encode/decode, server time, throughput, bit-exactness, lora_sha256 cost.
* ``failure_handling``: time from the failing call to the client-side exception for a server
  crash, a hang (timeout), a remote exception, startup failures, a killed client and a crashing
  torchrun rank.
* ``mock_e2e``: the GRPO CLI on the CPU mock engines with the trainer behind the bridge (all four
  modes) and in process (strict/strict): per-step phase timeline, drift, RPC breakdown.
* ``gpu_smoke`` (``--gpu``, run under a 1-GPU lease): placement (the server sees exactly the
  leased GPU, matched by UUID), CUDA surrogate exactness, a CUDA mock trainer through the bridge
  vs in process, server peak memory.

    PYTHONPATH=$PWD ~/envs/vllm/bin/python -m rl_engine.models.muse_glimmer.e2e.bench_bridge --cpu
    ~/bin/gpu-lease -n 1 --pool 6-7 -- env PYTHONPATH=$PWD ~/envs/vllm/bin/python \\\\
        -m rl_engine.models.muse_glimmer.e2e.bench_bridge --gpu
"""''', '''* ``bridge_equivalence``: RemoteTrainer (server in the Megatron env / in this env; process
  launcher, and a 2-rank torchrun control-plane check where every rank holds a full replica)
  vs the same trainer in process: per step logprob bit mismatches, grad norms, LoRA hash and
  byte mismatches.
* ``transport``: echo round trips of 1 KiB .. 256 MiB uint8 payloads and of a LoRA state dict
  with the REAL Muse-Glimmer-30B rank-16 key schema and shapes (832 bf16 tensors, 200 MiB),
  with TCP_NODELAY on (production) and off (Nagle, both ends), every repeat as a raw row: RTT,
  client encode/decode, server time, throughput, bit-exactness, lora_sha256 cost.
* ``startup_overlap`` (``--startup-overlap``): a mock trainer whose build takes
  ``build_delay_s`` is launched with ``wait=False``; the caller works ``caller_work_s`` (vLLM
  start-up stand-in) and then calls ``wait_ready``: blocked time and launch-to-ready per row.
* ``failure_handling``: time from the failing call to the client-side exception for a server
  crash, a hang (timeout), a remote exception in a stateful op, startup failures, a killed
  client, a crashing torchrun rank, divergent torchrun ranks and a forward-only trainer.
* ``mock_e2e``: the GRPO CLI on the CPU mock engines with the trainer behind the bridge (all four
  modes) and in process (strict/strict): per-step phase timeline, drift, RPC breakdown.
* ``gpu_smoke`` (``--gpu``, run under a 1-GPU lease): placement (the server sees exactly the
  leased GPU, matched by UUID), CUDA surrogate exactness, a CUDA mock trainer through the bridge
  vs in process, server peak memory.
* ``real_e2e_smoke`` (``--import-e2e-run DIR ...``): key numbers and raw per-step series of real
  ``run_poc_grpo`` run directories (30B engines), read from their summary/steps/provenance files
  and the trainer-server log.

    PYTHONPATH=$PWD ~/envs/vllm/bin/python -m rl_engine.models.muse_glimmer.e2e.bench_bridge \\\\
        --cpu --hash-check --startup-overlap
    ~/bin/gpu-lease -n 1 --pool 3,7 -- env PYTHONPATH=$PWD ~/envs/vllm/bin/python \\\\
        -m rl_engine.models.muse_glimmer.e2e.bench_bridge --gpu
"""''')

rep('''import argparse
import json
import os
import signal''', '''import argparse
import json
import os
import re
import signal''')

rep('''def _remote(work: Path, *, python: str = MEGATRON_PY, launcher: str = "process", tp: int = 1,
            factory: str = bridge.MOCK_TRAINER_FACTORY, factory_kwargs=None, gpu_ids=(),
            call_timeout_s: float = 300.0, startup_timeout_s: float = 300.0,
            threads: int = 2) -> bridge.RemoteTrainer:
    return bridge.RemoteTrainer(
        "mock", tensor_parallel_size=tp, gpu_ids=list(gpu_ids), lora_rank=4, lr=1e-3,
        strict=True, seed=7, python=python, launcher=launcher, factory=factory,
        factory_kwargs=factory_kwargs or {}, work_dir=work, call_timeout_s=call_timeout_s,
        startup_timeout_s=startup_timeout_s, extra_env={"OMP_NUM_THREADS": str(threads)},
    )''', '''def _remote(work: Path, *, python: str = MEGATRON_PY, launcher: str = "process", tp: int = 1,
            factory: str = bridge.MOCK_TRAINER_FACTORY, factory_kwargs=None, gpu_ids=(),
            call_timeout_s: float = 300.0, startup_timeout_s: float = 300.0,
            threads: int = 2, **kwargs: Any) -> bridge.RemoteTrainer:
    return bridge.RemoteTrainer(
        "mock", tensor_parallel_size=tp, gpu_ids=list(gpu_ids), lora_rank=4, lr=1e-3,
        strict=True, seed=7, python=python, launcher=launcher, factory=factory,
        factory_kwargs=factory_kwargs or {}, work_dir=work, call_timeout_s=call_timeout_s,
        startup_timeout_s=startup_timeout_s, extra_env={"OMP_NUM_THREADS": str(threads)},
        **kwargs,
    )''')

rep('''    for name, python, launcher, tp in (
        ("megatron_env_process", MEGATRON_PY, "process", 1),
        ("same_env_process", sys.executable, "process", 1),
        ("megatron_env_torchrun_tp2", MEGATRON_PY, "torchrun", 2),
    ):''', '''    notes = {
        "megatron_env_torchrun_2rank_replicated_control_plane": (
            "NOT tensor parallel: each of the 2 torchrun ranks builds a full-replica CPU "
            "MockTrainer (gpu_ids=[]) and rank 0 answers. Validates only the control plane: "
            "every op broadcast to every rank in order, per-rank status gather, rank LoRA-hash "
            "and logprob-digest agreement. No TP numerics are exercised."
        ),
    }
    for name, python, launcher, tp in (
        ("megatron_env_process", MEGATRON_PY, "process", 1),
        ("same_env_process", sys.executable, "process", 1),
        ("megatron_env_torchrun_2rank_replicated_control_plane", MEGATRON_PY, "torchrun", 2),
    ):''')

rep('''                "last_export_rank_sha256": rank_sha,
                "calls": calls,
            }
        )
    return out''', '''                "last_export_rank_sha256": rank_sha,
                "note": notes.get(name),
                "calls": calls,
            }
        )
    return out''')

rep('''def transport(work: Path, repeats: int = 3) -> dict[str, Any]:
    sizes = [1 << 10, 16 << 10, 256 << 10, 4 << 20, 32 << 20, 128 << 20, 256 << 20]
    remote = _remote(work / "transport")
    rows = []
    lora_rows = []
    try:
        remote.wait_ready()
        gen = torch.Generator().manual_seed(0)
        for size in sizes:
            payload = torch.randint(0, 256, (size,), dtype=torch.uint8, generator=gen)
            for rep in range(repeats):
                echoed, stats = remote.ping(payload)
                rows.append(
                    {
                        "payload_bytes": size,''', '''def transport(work: Path, repeats: int = 3) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    lora_rows: list[dict[str, Any]] = []
    sha_ms: dict[str, float] = {}
    for nodelay in (True, False):
        sha_ms[f"tcp_nodelay={nodelay}"] = _transport_variant(
            work / f"transport_nodelay_{int(nodelay)}", nodelay, repeats, rows, lora_rows
        )
    return {
        "note": "echo through the trainer server (megatron env, process launcher, CPU); one "
        "round trip = request + reply, each a torch.save blob over a localhost socket. "
        "tcp_nodelay=True is the production setting (bridge.set_nodelay on both ends); "
        "tcp_nodelay=False keeps Nagle on both ends (RemoteTrainer(tcp_nodelay=False)), measured "
        "in the same run for comparison. Every repeat is a raw row.",
        "repeats": repeats,
        "payloads": rows,
        "real_lora_rank16": {
            "shape_source": "museglimmer_manifest dims, 52 layers x 8 projections x (A, B), bf16",
            "lora_sha256_ms_before_send": sha_ms,
            "rows": lora_rows,
        },
    }


def _transport_variant(work: Path, nodelay: bool, repeats: int, rows: list, lora_rows: list
                       ) -> float:
    sizes = [1 << 10, 16 << 10, 256 << 10, 4 << 20, 32 << 20, 128 << 20, 256 << 20]
    remote = _remote(work, tcp_nodelay=nodelay)
    sha_s = 0.0
    try:
        hello = remote.wait_ready()
        assert hello["tcp_nodelay"] is nodelay
        gen = torch.Generator().manual_seed(0)
        for size in sizes:
            payload = torch.randint(0, 256, (size,), dtype=torch.uint8, generator=gen)
            for rep in range(repeats):
                echoed, stats = remote.ping(payload)
                rows.append(
                    {
                        "tcp_nodelay": nodelay,
                        "payload_bytes": size,''')

rep('''            lora_rows.append(
                {
                    "repeat": rep,
                    "num_tensors": len(lora),''', '''            lora_rows.append(
                {
                    "tcp_nodelay": nodelay,
                    "repeat": rep,
                    "num_tensors": len(lora),''')

rep('''    finally:
        remote.close()
    return {
        "note": "echo through the trainer server (megatron env, process launcher, CPU); one "
        "round trip = request + reply, each a torch.save blob over a localhost socket",
        "payloads": rows,
        "real_lora_rank16": {
            "shape_source": "museglimmer_manifest dims, 52 layers x 8 projections x (A, B), bf16",
            "lora_sha256_ms_before_send": sha_s * 1e3,
            "rows": lora_rows,
        },
    }''', '''    finally:
        remote.close()
    return sha_s * 1e3


def startup_overlap(work: Path, *, build_delay_s: float = 6.0,
                    caller_work_s: tuple[float, ...] = (0.0, 3.0, 6.0, 9.0, 12.0)
                    ) -> dict[str, Any]:
    """Does the trainer build overlap the caller's work between launch and ``wait_ready``?"""

    rows = []
    for caller in caller_work_s:
        t0 = time.monotonic()
        remote = _remote(work / f"overlap_{caller:g}", factory_kwargs={"build_delay_s": build_delay_s},
                         wait=False)
        try:
            time.sleep(caller)  # stands in for the orchestrator building vLLM
            t1 = time.monotonic()
            hello = remote.wait_ready()
            t2 = time.monotonic()
        finally:
            remote.close()
        rows.append(
            {
                "caller_work_s": caller,
                "build_delay_s": build_delay_s,
                "server_build_s": hello["build_s"],
                "wait_ready_blocked_s": t2 - t1,
                "launch_to_ready_s": t2 - t0,
                "startup": hello.get("startup"),
            }
        )
    return {
        "note": "process launcher, megatron env, CPU MockTrainer(build_delay_s) = a trainer whose "
        "build takes build_delay_s (the 30B StrictTrainer build took 100-164 s in the real "
        "smokes). Serial behaviour: launch_to_ready ~ caller_work + interpreter start + build; "
        "overlapped: ~ max(caller_work, interpreter start + build).",
        "build_delay_s": build_delay_s,
        "rows": rows,
    }''')

rep('''    for name, launcher, tp, kwargs, op, timeout in (
        ("server_exit_during_step", "process", 1, {"fault": "exit", "fault_op": "step"}, "step",
         120.0),
        ("server_hang_during_scoring", "process", 1,
         {"fault": "hang", "fault_op": "token_logprobs"}, "token_logprobs", 5.0),
        ("remote_exception_in_step", "process", 1, {"fault": "raise", "fault_op": "step"},
         "step", 120.0),
        ("torchrun_rank1_exit_during_step", "torchrun", 2,
         {"fault": "exit", "fault_op": "step", "fault_rank": 1}, "step", 120.0),
    ):
        remote = _remote(work / name, launcher=launcher, tp=tp, factory=faulty,
                         factory_kwargs=kwargs, call_timeout_s=timeout)
        pid = remote.server_pid
        try:
            if op == "step":
                loss = scored(remote)
                ms, err = _timed(lambda: remote.step(loss))
            else:
                ms, err = _timed(lambda: remote.token_logprobs(SEQS, STARTS))
            alive_after = remote.alive()
            survives = None
            if alive_after:
                survives = remote.provenance().get("engine") == "MockTrainer"
        finally:
            remote.close()''', '''    for name, launcher, tp, kwargs, op, timeout in (
        ("server_exit_during_step", "process", 1, {"fault": "exit", "fault_op": "step"}, "step",
         120.0),
        ("server_hang_during_scoring", "process", 1,
         {"fault": "hang", "fault_op": "token_logprobs"}, "token_logprobs", 5.0),
        ("remote_exception_in_step", "process", 1, {"fault": "raise", "fault_op": "step"},
         "step", 120.0),
        ("torchrun_rank1_exit_during_step", "torchrun", 2,
         {"fault": "exit", "fault_op": "step", "fault_rank": 1}, "step", 120.0),
        ("torchrun_rank1_divergent_lora_export", "torchrun", 2,
         {"fault": "diverge_lora", "fault_op": "export_lora", "fault_rank": 1}, "export_lora",
         120.0),
        ("torchrun_rank1_divergent_logprobs", "torchrun", 2,
         {"fault": "diverge_logprobs", "fault_op": "token_logprobs", "fault_rank": 1},
         "token_logprobs", 120.0),
        ("forward_only_trainer_scoring", "process", 1,
         {"fault": "forward_only", "fault_op": "token_logprobs"}, "score_check", 120.0),
    ):
        remote = _remote(work / name, launcher=launcher, tp=tp, factory=faulty,
                         factory_kwargs=kwargs, call_timeout_s=timeout)
        pid = remote.server_pid
        try:
            if op == "step":
                loss = scored(remote)
                ms, err = _timed(lambda: remote.step(loss))
            elif op == "export_lora":
                ms, err = _timed(remote.export_lora)
            elif op == "score_check":
                from rl_engine.models.muse_glimmer.e2e.loop import _check_train_logps

                ms, err = _timed(lambda: _check_train_logps(
                    remote.token_logprobs(SEQS, STARTS), SEQS, STARTS))
            else:
                ms, err = _timed(lambda: remote.token_logprobs(SEQS, STARTS))
            alive_after = remote.alive()
            survives = None
            if alive_after:
                try:
                    survives = remote.provenance().get("engine") == "MockTrainer"
                except bridge.RemoteTrainerError:
                    survives = False
        finally:
            remote.close()''')

rep('''def main(argv: list[str] | None = None) -> int:''', '''_LOG_LINE = re.compile(r"^\\[trainer_server rank(\\d+) (\\d\\d):(\\d\\d):(\\d\\d)\\] (.*)$")


def _server_log_events(path: Path) -> list[dict[str, Any]]:
    """``[trainer_server rank0 HH:MM:SS] message`` lines (rank, clock, seconds of day, text)."""

    events = []
    if not path.exists():
        return events
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = _LOG_LINE.match(line.strip())
        if match:
            rank, hh, mm, ss, text = match.groups()
            events.append({"rank": int(rank), "clock": f"{hh}:{mm}:{ss}",
                           "sec_of_day": int(hh) * 3600 + int(mm) * 60 + int(ss),
                           "message": text})
    return events


def import_e2e_run(run_dir: Path, *, commit: str | None) -> dict[str, Any]:
    """Key numbers and raw per-step series of one real ``run_poc_grpo`` run directory."""

    summary = json.loads((run_dir / "summary.json").read_text())
    steps = [json.loads(line) for line in (run_dir / "steps.jsonl").read_text().splitlines()
             if line.strip()]
    prov = json.loads((run_dir / "provenance.json").read_text())
    cfg = summary["config"]
    code = summary.get("code") or {}
    rollout0, trainer0 = prov.get("rollout_start", {}), prov.get("trainer_start", {})
    hello = (trainer0.get("bridge") or {}).get("hello") or {}
    events = _server_log_events(run_dir / "trainer_server" / "trainer_server.log")

    def first(prefix: str) -> dict[str, Any] | None:
        return next((e for e in events if e["message"].startswith(prefix)), None)

    start, connected, built = first("start "), first("client connected"), first("trainer ")
    close = first("close ")
    rows = []
    for rec in steps:
        trainer_step = rec.get("trainer_step") or {}
        rows.append({
            "step": rec["step"],
            "num_samples": rec["num_samples"],
            "num_completion_tokens": rec["num_completion_tokens"],
            "rewards": rec["rewards"],
            "advantages": rec["advantages"],
            "drift": rec["drift"],
            "rollout_decode_vs_prefill": rec.get("rollout_decode_vs_prefill"),
            "grpo": {k: rec["grpo"].get(k) for k in (
                "loss", "ratio_mean", "ratio_min", "ratio_max", "ratio_exactly_one_frac",
                "clip_frac")},
            "trainer_step": {k: trainer_step.get(k) for k in (
                "grad_norm", "recompute_logp_mismatches", "recompute_logp_checked",
                "micro_batches", "fwd_bwd_s", "optimizer_s", "num_lora_params_local")},
            "sync": rec["sync"],
            "phases_s": (rec.get("timeline") or {}).get("phases_s"),
            "step_s": rec["timings"]["step_s"],
            "bridge_calls": [
                {"op": c["op"], "rtt_s": c["rtt_s"],
                 "server_compute_s": (c.get("server") or {}).get("compute_s"),
                 "request_bytes": c["request_bytes"], "response_bytes": c["response_bytes"]}
                for c in rec.get("bridge_calls") or []
            ],
        })
    return {
        "run_dir": str(run_dir),
        "commit": code.get("git_commit") or commit,
        "commit_source": "summary.code" if code.get("git_commit") else (
            "--e2e-run-commit (the run directory records no commit)" if commit else None),
        "mode": summary["mode"],
        "status": summary["status"],
        "error": summary["error"],
        "config": {k: cfg.get(k) for k in (
            "steps", "prompts_per_step", "samples_per_prompt", "max_tokens", "temperature",
            "lora_rank", "lr", "seed", "rollout_tp", "rollout_gpus", "train_tp", "train_gpus",
            "trainer_launcher", "task", "split")},
        "rollout": {k: rollout0.get(k) for k in (
            "engine", "runtime", "strict", "lora_mode", "logprob_path", "tensor_parallel_size",
            "gpu_ids")} | {"load_seconds": (rollout0.get("timing") or {}).get("load_seconds")},
        "trainer": {k: trainer0.get(k) for k in (
            "engine", "runtime", "strict", "tp", "binding", "load_seconds", "init_seconds")}
        | {"build_s": hello.get("build_s"), "startup_s": hello.get("startup_s"),
           "versions": hello.get("versions")},
        "trainer_server_log": {
            "start": start and start["clock"],
            "client_connected": connected and connected["clock"],
            "trainer_built": built and built["clock"],
            "close": close and close["message"],
            "start_to_client_connected_s": (connected["sec_of_day"] - start["sec_of_day"])
            if start and connected else None,
            "start_to_trainer_built_s": (built["sec_of_day"] - start["sec_of_day"])
            if start and built else None,
            "events": [{"clock": e["clock"], "message": e["message"][:200]} for e in events
                       if not e["message"].startswith("trainer construction failed")][:40],
        },
        "drift_totals": summary["drift_totals"],
        "first_drift": summary["first_drift"],
        "weight_sync": summary["weight_sync"],
        "lora_sha256": {
            "rollout_start": rollout0.get("lora_sha256"),
            "trainer_start": trainer0.get("lora_sha256"),
            "rollout_end": (prov.get("rollout_end") or {}).get("lora_sha256"),
            "trainer_end": (prov.get("trainer_end") or {}).get("lora_sha256"),
        },
        "reward_trajectory": summary["reward"]["trajectory"],
        "timings_mean_s": summary["timings_mean_s"],
        "steps": rows,
    }


# Integration findings of this row against the sibling engines, with their current status.
INTEGRATION_FINDINGS = [
    {
        "id": "F1",
        "what": "runtime/megatron_lora.lora_sha256 (key\\\\n + shape\\\\n + bytes) was not "
        "muse_glimmer_lora_sha256_v1; StrictTrainer.export_lora reported the former and the "
        "bridge failed closed at the initial LoRA sync (reviewer's TP1 smoke: WeightSyncError).",
        "status": "fixed in the merged code: megatron_lora.lora_sha256 delegates to "
        "e2e.sync.lora_sha256",
        "evidence": "hash_schema_check all_equal (this JSON); real_e2e_smoke: trainer_start, "
        "rollout_start and every sync report the same canonical hash in both modes",
        "follow_up": "e2e_bridge:R11 - runtime imports e2e (dependency inversion): move the "
        "hash into a shared module (request to the integrator)",
    },
    {
        "id": "F2",
        "what": "StrictTrainer(strict=False) had no LoRA (step/export_lora raised).",
        "status": "fixed in the merged code (native trainer LoRA hooks)",
        "evidence": "real_e2e_smoke native/native: step ok, export_lora ok, weight sync 2/2",
    },
    {
        "id": "F3",
        "what": "StrictRolloutEngine(strict=False).load_lora raised.",
        "status": "fixed in the merged code (vLLM LoRARequest from a PEFT folder)",
        "evidence": "real_e2e_smoke native/native rollout lora_mode 'vLLM native punica LoRA "
        "(PEFT folder)', weight sync 2/2",
    },
    {
        "id": "F4",
        "what": "StrictRolloutEngine.provenance() had no top-level lora_sha256 and the workers "
        "stored the declared hash.",
        "status": "fixed in the merged code (top-level lora_sha256; None when worker hashes "
        "differ); e2e/sync no longer accepts per-worker values at all",
        "evidence": "real_e2e_smoke: rollout_sha256_source 'provenance.lora_sha256'; "
        "test_rollout_hash_only_from_the_engine_level_provenance_key",
    },
    {
        "id": "F5",
        "what": "StrictTrainer TP worker NCCL group used the default 10 min timeout.",
        "status": "fixed in the merged code (30 min, strict_trainer._init_distributed)",
        "evidence": "code reading",
    },
    {
        "id": "F6",
        "what": "StrictTrainer spawns its own TP workers, so it runs under --trainer-launcher "
        "process; under torchrun it fails closed on len(gpu_ids) == tp.",
        "status": "open (by design for the POC)",
        "evidence": "real_e2e_smoke uses the process launcher with train TP2",
    },
    {
        "id": "F7",
        "what": "StrictTrainer.close() took ~127 s at TP2 in both real smokes (trainer-server "
        "log 'close ... compute=127295 ms' / '127799 ms'): the TP worker join (timeout 120 s) "
        "expires and the worker is terminated.",
        "status": "open (runtime/strict_trainer.py, not this row)",
        "evidence": "real_e2e_smoke[*].trainer_server_log.close",
    },
    {
        "id": "F8",
        "what": "Before e2e_bridge:R3 the trainer server accepted the client before building, "
        "so the 30B trainer build ran after vLLM start-up instead of during it.",
        "status": "fixed in this row (build before accept); NOT re-measured on the 30B",
        "evidence": "real_e2e_smoke strict: server start 20:48:49, client connected 20:49:33, "
        "trainer built 20:52:17 (start-up 209.6 s for a 163.7 s build); startup_overlap rows",
    },
]

STATUS = (
    "bridge + instrumentation complete; review wave B fixes in (rank agreement, start-up "
    "overlap, stateful-failure retire, top-level rollout hash only, lease-relative placement). "
    "Real 30B: two one-step E2E smokes through the bridge by the integrator (real_e2e_smoke); "
    "the four-mode 8-step runs are NOT RUN (integrator)."
)


def main(argv: list[str] | None = None) -> int:''')

rep('''    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--command", default=None, help="command line to record in the JSON")
    args = parser.parse_args(argv)''', '''    parser.add_argument("--work-dir", default=None)
    parser.add_argument("--command", default=None, help="command line to record in the JSON")
    parser.add_argument("--startup-overlap", action="store_true",
                        help="trainer build vs caller work before wait_ready (CPU mock)")
    parser.add_argument("--overlap-label", default="current",
                        help="key under startup_overlap for this measurement")
    parser.add_argument("--import-e2e-run", nargs="*", default=None, metavar="RUN_DIR",
                        help="record real run_poc_grpo run directories under real_e2e_smoke")
    parser.add_argument("--e2e-run-commit", default=None,
                        help="commit the imported runs used (when their summary has none)")
    args = parser.parse_args(argv)''')

rep('''    if args.real_trainer_smoke:
        key = f"real_trainer_smoke_tp{args.train_tp}"
        data[key] = real_trainer_smoke(
            work / "real", tp=args.train_tp, launcher=args.trainer_launcher,
            model_path=args.model_path, num_sequences=args.num_sequences,
        )
        data[key]["commit"] = data["commit"]
    for tp in (1, 2):
        data.setdefault(
            f"real_trainer_smoke_tp{tp}",
            {
                "status": "NOT RUN",
                "reason": "StrictTrainer is pinned to the 30B shapes (megatron_strict._check_config"
                "), so no toy-config smoke exists; the 30B smoke is for the integrator's lease",
                "command": "~/bin/gpu-lease -n %d --pool 6-7 -- env PYTHONPATH=$PWD "
                "~/envs/vllm/bin/python -m rl_engine.models.muse_glimmer.e2e.bench_bridge "
                "--real-trainer-smoke --train-tp %d" % (tp, tp),
            },
        )
    out_path.parent.mkdir''', '''    if args.real_trainer_smoke:
        key = f"real_trainer_smoke_tp{args.train_tp}"
        data[key] = real_trainer_smoke(
            work / "real", tp=args.train_tp, launcher=args.trainer_launcher,
            model_path=args.model_path, num_sequences=args.num_sequences,
        )
        data[key]["commit"] = data["commit"]
    if args.startup_overlap:
        entry = startup_overlap(work / "overlap")
        entry["commit"] = data["commit"]
        data.setdefault("startup_overlap", {})[args.overlap_label] = entry
    if args.import_e2e_run:
        data["real_e2e_smoke"] = [
            import_e2e_run(Path(d), commit=args.e2e_run_commit) for d in args.import_e2e_run
        ]
    # Superseded hand-written entries (replaced by measured sections above).
    data.pop("transport_before_tcp_nodelay", None)
    for tp in (1, 2):
        stub = data.get(f"real_trainer_smoke_tp{tp}")
        if isinstance(stub, dict) and stub.get("status") == "NOT RUN":
            data.pop(f"real_trainer_smoke_tp{tp}")
    data["integration_findings"] = INTEGRATION_FINDINGS
    data["status"] = STATUS
    out_path.parent.mkdir''')

rep('''    parser.add_argument(
        "--real-trainer-smoke",
        action="store_true",
        help="REAL StrictTrainer (30B) behind the bridge on the leased GPUs (integrator only)",
    )''', '''    parser.add_argument(
        "--real-trainer-smoke",
        action="store_true",
        help="REAL StrictTrainer (30B) behind the bridge on the leased GPUs (score, step, export)",
    )''')
p.write_text(s)
print("patched", p)
