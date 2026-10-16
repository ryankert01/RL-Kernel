"""Add status, the pre-fix transport observation and the integration findings to e2e_bridge.json."""
import json
from pathlib import Path

p = Path("/home/ubuntu/RL-Kernel/.claude/worktrees/wf_7b064961-36f-3/reports/muse_glimmer_poc/data/e2e_bridge.json")
d = json.loads(p.read_text())
if "client" in d["env"]:
    d["env"]["client_gpu_run"] = d["env"].pop("client")
d["status"] = (
    "complete for the bridge + instrumentation scope: CPU (megatron-env server) and 1-GPU "
    "(A100, lease GPU 7) runs passed; the REAL 30B StrictTrainer through the bridge and the "
    "real E2E were NOT RUN (integrator)"
)
d["transport_before_tcp_nodelay"] = {
    "source": "first bench run at commit 025b1f9 (before TCP_NODELAY), repeat index 1 of 3, "
    "copied from that run's output; the JSON of that run was replaced by the current one",
    "payload_bytes": [1024, 16384, 262144, 4194304, 33554432, 134217728, 268435456],
    "rtt_ms": [8.96, 93.96, 87.34, 40.29, 348.09, 1227.37, 2016.29],
    "mock_e2e_export_lora_overhead_ms_mean": 46.8,
    "note": "Nagle + delayed ACK on loopback: multiprocessing.connection writes header and "
    "payload separately above 16 KiB. Fixed by bridge.set_nodelay on both ends.",
}
d["integration_findings"] = [
    {
        "id": "F1",
        "severity": "blocks every mode with a strict trainer at the initial LoRA sync",
        "what": "runtime/megatron_lora.lora_sha256 (key\\n + shape\\n + bytes) is not "
        "muse_glimmer_lora_sha256_v1 (name|dtype|shape\\n + bytes) of e2e/sync.py; "
        "StrictTrainer.export_lora reports the former, export_lora_verified / RemoteTrainer "
        "recompute the latter and fail closed (WeightSyncError).",
        "evidence": "hash_schema_check (all_equal false)",
        "patch": "runtime/megatron_lora.py: def lora_sha256(state): from "
        "rl_engine.models.muse_glimmer.e2e.sync import lora_sha256 as canonical; "
        "return canonical(state)  (import_lora then verifies with the same schema)",
    },
    {
        "id": "F2",
        "severity": "blocks native/native and native/strict (native trainer)",
        "what": "StrictTrainer(strict=False) has no LoRA: step() raises 'step requires "
        "strict=True', export_lora() calls export_binding_lora(None) -> AttributeError.",
        "evidence": "code reading of runtime/strict_trainer.py at 307fcf2 (not run)",
        "patch": "native trainer needs Bridge-native forward + the same LoRA wrapper with "
        "gradients (or the RFC drops the native-train modes)",
    },
    {
        "id": "F3",
        "severity": "blocks native/native and strict/native (native rollout)",
        "what": "StrictRolloutEngine(strict=False).load_lora raises 'load_lora is only "
        "available on a strict engine'; the loop syncs LoRA before rollout 0 in every mode.",
        "evidence": "code reading of runtime/strict_rollout_engine.py at 307fcf2 (not run)",
        "patch": "native rollout needs vLLM LoRA (enable_lora + LoRARequest from the exported "
        "tensors) or the strict LoRA slots on the native model",
    },
    {
        "id": "F4",
        "severity": "weakens weight_sync_identity on the vLLM side (no hard failure)",
        "what": "StrictRolloutEngine.provenance() has no top-level lora_sha256 (only "
        "workers[i].lora_sha256), and worker_load_lora stores the DECLARED hash without "
        "recomputing it on the received tensors.",
        "evidence": "code reading; e2e/sync.rollout_lora_sha256 now accepts the per-worker "
        "hash when all TP workers agree (tested)",
        "patch": "load_lora: verify sync.lora_sha256(cpu_state) == sha256 before the RPC and "
        "recompute in worker_load_lora; expose provenance()['lora_sha256']",
    },
    {
        "id": "F5",
        "severity": "risk for train TP2 when one rollout step takes > 10 min",
        "what": "strict_trainer._init_distributed uses the default NCCL timeout (10 min); the "
        "TP worker idles in broadcast_object_list while the orchestrator generates, so a long "
        "rollout (or vLLM start-up after the trainer is built) can trip the NCCL watchdog.",
        "evidence": "code reading (not run)",
        "patch": "dist.init_process_group(..., timeout=datetime.timedelta(hours=6))",
    },
    {
        "id": "F6",
        "severity": "launcher choice",
        "what": "today's StrictTrainer spawns its own TP workers, so it must run under "
        "--trainer-launcher process; under torchrun the server passes gpu_ids=[LOCAL_RANK] "
        "and StrictTrainer fails closed on its len(gpu_ids)==tp check (no double spawn).",
        "evidence": "torchrun path tested with the mock trainer (TP2, gloo): all ranks run "
        "the same ops, rank LoRA hashes agree",
        "patch": "to use torchrun: StrictTrainer must build one rank per process when "
        "torch.distributed is already initialized",
    },
]
p.write_text(json.dumps(d, indent=1, default=str) + "\n", encoding="utf-8")
print(sorted(d))
