"""Verifier probes for the e2e wave B fixes (b3e5066). CPU only (gpu_ids=[]).

A  R3: wait=False, 6 s build, 10 s caller work -> wait_ready must block < 2 s; server log order.
B  R3 edge: wait=False + constructor raises -> wait_ready raises with the remote traceback,
   the server process is gone afterwards.
C  R3+R5: torchrun 2 ranks, wait=False, caller work 8 s (rank 1 waits in broadcast while rank 0
   waits for the client) -> calls work, export_lora rank_sha256 has 2 equal hashes,
   token_logprobs rank_logprobs_sha256 has 2 equal digests.
D  R5 steady state: torchrun, diverge_lora on rank 1 only on the 2nd export (fault_after=1)
   -> first export passes, second raises RemoteWeightSyncError, bridge retired.
E  R9: stateful failure in export_lora via 'raise' -> retired; ping afterwards refused.
"""
import json
import sys
import time
import traceback
from pathlib import Path

import torch

from rl_engine.models.muse_glimmer.e2e import bridge, grpo
from rl_engine.models.muse_glimmer.e2e.sync import WeightSyncError

OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
MEGATRON_PY = str(Path("~/envs/megatron/bin/python").expanduser())
FAULTY = "rl_engine.models.muse_glimmer.e2e.mocks:FaultyMockTrainer"
SEQS = [[256, 3, 5, 7, 9, 11, 13], [256, 4, 6, 8, 10]]
STARTS = [3, 2]
results = {}


def remote(name, **kw):
    base = dict(
        tensor_parallel_size=1, gpu_ids=[], lora_rank=4, lr=1e-3, strict=True, seed=7,
        python=MEGATRON_PY, launcher="process", factory=bridge.MOCK_TRAINER_FACTORY,
        factory_kwargs={}, work_dir=OUT / name, call_timeout_s=120.0, startup_timeout_s=180.0,
        extra_env={"OMP_NUM_THREADS": "2"},
    )
    base.update(kw)
    return bridge.RemoteTrainer("mock", **base)


def log_lines(r):
    return [l for l in Path(r.log_path).read_text(errors="replace").splitlines()
            if any(w in l for w in ("start pid", "client connected", "built in", "construction failed", "accepting"))]


def probe(name, fn):
    t0 = time.monotonic()
    try:
        results[name] = {"ok": True, **fn()}
    except Exception as exc:  # noqa: BLE001
        results[name] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:800],
                         "tb": traceback.format_exc()[-1500:]}
    results[name]["wall_s"] = time.monotonic() - t0
    print(name, json.dumps(results[name], default=str)[:1500], flush=True)


def a():
    r = remote("A", factory_kwargs={"build_delay_s": 6.0}, wait=False)
    try:
        time.sleep(10.0)
        t0 = time.monotonic()
        hello = r.wait_ready()
        blocked = time.monotonic() - t0
        return {"blocked_s": blocked, "build_s": hello["build_s"], "startup": hello["startup"],
                "log": log_lines(r), "pass": blocked < 2.0}
    finally:
        r.close()


def b():
    r = remote("B", factory=FAULTY, factory_kwargs={"fault": "build"}, wait=False)
    pid = r.server_pid
    time.sleep(4.0)
    alive_before_connect = r.alive()
    try:
        r.wait_ready()
        raised = None
    except bridge.RemoteTrainerError as exc:
        raised = f"{type(exc).__name__}: {str(exc)[:300]}"
    time.sleep(0.5)
    alive_after = r.alive()
    r.close()
    return {"alive_before_connect": alive_before_connect, "raised": raised,
            "server_alive_after": alive_after, "log": log_lines(r),
            "pass": raised is not None and not alive_after}


def c():
    r = remote("C", launcher="torchrun", tensor_parallel_size=2, wait=False)
    try:
        time.sleep(8.0)
        hello = r.wait_ready()
        lps = r.token_logprobs(SEQS, STARTS)
        last = r.calls[-1]
        loss = grpo.grpo_token_loss(lps, [t.detach() for t in lps], torch.tensor([1.0, -1.0])).loss
        r.step(loss)
        sd, sha = r.export_lora()
        return {"world": hello["world"], "rank_sha256": r.last_export.get("rank_sha256"),
                "sha": sha, "startup": hello["startup"],
                "pass": hello["world"] == 2 and r.last_export.get("rank_sha256") == [sha, sha]}
    finally:
        r.close()


def d():
    r = remote("D", launcher="torchrun", tensor_parallel_size=2, factory=FAULTY,
               factory_kwargs={"fault": "diverge_lora", "fault_op": "export_lora",
                               "fault_after": 1, "fault_rank": 1})
    pid = r.server_pid
    try:
        _, sha1 = r.export_lora()
        first_ok = r.last_export.get("rank_sha256") == [sha1, sha1]
        try:
            r.export_lora()
            second = "DID NOT RAISE"
        except WeightSyncError as exc:
            second = f"{type(exc).__name__}: {str(exc)[:300]}"
        try:
            r.ping(torch.zeros(4))
            after = "ping worked (bridge NOT retired)"
        except bridge.RemoteTrainerError as exc:
            after = f"refused: {str(exc)[:120]}"
        return {"first_ok": first_ok, "second": second, "after": after, "alive": r.alive(),
                "pass": first_ok and second.startswith("RemoteWeightSyncError") and after.startswith("refused") and not r.alive()}
    finally:
        r.close()


def e():
    r = remote("E", factory=FAULTY, factory_kwargs={"fault": "raise", "fault_op": "export_lora"})
    try:
        try:
            r.export_lora()
            first = "DID NOT RAISE"
        except bridge.RemoteTrainerError as exc:
            first = f"{type(exc).__name__}: {str(exc).splitlines()[0][:200]}"
        try:
            r.ping(torch.zeros(4))
            after = "ping worked"
        except bridge.RemoteTrainerError as exc:
            after = f"refused: {str(exc)[:120]}"
        return {"first": first, "after": after, "alive": r.alive(),
                "pass": after.startswith("refused") and not r.alive()}
    finally:
        r.close()


for name, fn in (("A_overlap", a), ("B_build_fail_wait_false", b), ("C_torchrun_late_connect", c),
                 ("D_torchrun_diverge_second_export", d), ("E_stateful_export_raise", e)):
    probe(name, fn)

(OUT / "probe_results.json").write_text(json.dumps(results, indent=1, default=str))
print("ALL_PASS", all(v.get("pass") for v in results.values()))
