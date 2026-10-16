from pathlib import Path

p = Path("/home/ubuntu/RL-Kernel/.claude/worktrees/wf_7b064961-36f-3/rl_engine/models/muse_glimmer/e2e/loop.py")
s = p.read_text()
rep = []

rep.append(('''No VIME, no Ray: a single process drives two engine objects; GPU placement is the engines' job
(``gpu_ids``), the orchestrator never touches CUDA.
"""''', '''No VIME, no Ray: one orchestrator process drives two engine objects. The rollout lives in this
process (vLLM env); with ``trainer_bridge`` the trainer is a :class:`bridge.RemoteTrainer` whose
server runs in the Megatron env (``trainer_server.py``). The orchestrator never touches CUDA.

Timeline: every step records the wall seconds of the phases in :data:`timeline.PHASE_ORDER`
(``steps.jsonl["timeline"]``), the RemoteTrainer call stats (RTT vs server compute, bytes), and
``summary.json["timeline"]`` holds the per-phase series for plotting. ``profile_step`` captures
``torch.profiler`` top-ops tables of that step's rollout (orchestrator process) and of the
trainer scoring + train step (on the trainer server when bridged, else in-process).
"""'''))

rep.append(('''from rl_engine.models.muse_glimmer.e2e.sync import export_lora_verified, transfer_lora
''', '''from rl_engine.models.muse_glimmer.e2e.sync import export_lora_verified, transfer_lora
from rl_engine.models.muse_glimmer.e2e.timeline import PhaseClock, profiled, summarize_timeline
'''))

rep.append(('''    task: str = "gsm8k"
    data_dir: str = "~/data/gsm8k"
    split: str = "train"
    limit_examples: int | None = None
''', '''    task: str = "gsm8k"
    data_dir: str = "~/data/gsm8k"
    split: str = "train"
    limit_examples: int | None = None
    # Cross-environment trainer bridge (run_poc_grpo builds the RemoteTrainer from these).
    trainer_bridge: bool = False
    trainer_python: str = "~/envs/megatron/bin/python"
    trainer_launcher: str = "process"
    trainer_factory: str | None = None
    trainer_timeout_s: float = 3600.0
    trainer_startup_timeout_s: float = 3600.0
    mock_trainer_device: str = "cpu"
    # Profiling: capture torch.profiler top-ops tables for this step (None = off).
    profile_step: int | None = None
'''))

rep.append(('''    "rollout_time_s",
    "train_logp_time_s",
    "train_step_time_s",
    "sync_time_s",
    "step_time_s",
]''', '''    "rollout_time_s",
    "train_logp_time_s",
    "train_step_time_s",
    "sync_time_s",
    "step_time_s",
    "reward_time_s",
    "drift_time_s",
    "recheck_time_s",
    "sidecar_time_s",
    "lora_export_time_s",
    "lora_transfer_time_s",
    "rollout_tokens_per_s",
]'''))

rep.append(('''    timings: dict[str, float],
    advantages: torch.Tensor,
    samples_per_prompt: int,
) -> dict[str, Any]:
    lengths = [len(c) for c in flat["completions"]]''', '''    timings: dict[str, float],
    advantages: torch.Tensor,
    samples_per_prompt: int,
    timeline: dict[str, Any] | None = None,
    bridge_calls: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    lengths = [len(c) for c in flat["completions"]]'''))

rep.append(('''        "sync": sync,
        "timings": timings,
    }
    record.update(reward_stats(flat["rewards"], samples_per_prompt))
    return record''', '''        "sync": sync,
        "timings": timings,
        "timeline": timeline or {},
        "bridge_calls": bridge_calls,
    }
    record.update(reward_stats(flat["rewards"], samples_per_prompt))
    return record'''))

rep.append(('''        "sync_time_s": timings["sync_s"],
        "step_time_s": timings["step_s"],
    }''', '''        "sync_time_s": timings["sync_s"],
        "step_time_s": timings["step_s"],
        "reward_time_s": timings.get("reward_s", ""),
        "drift_time_s": timings.get("drift_s", ""),
        "recheck_time_s": timings.get("recheck_s", ""),
        "sidecar_time_s": timings.get("sidecar_s", ""),
        "lora_export_time_s": timings.get("lora_export_s", ""),
        "lora_transfer_time_s": timings.get("lora_transfer_s", ""),
        "rollout_tokens_per_s": (record.get("timeline") or {}).get("rollout_tokens_per_s", ""),
    }'''))

rep.append(('''    status: str,
    error: str | None,
    initial_sync: dict[str, Any],
) -> dict[str, Any]:''', '''    status: str,
    error: str | None,
    initial_sync: dict[str, Any],
    profiles: list[dict[str, Any]] | None = None,
    initial_bridge_calls: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:'''))

rep.append(('''        "timings_mean_s": timings,
        "formulas": {''', '''        "timings_mean_s": timings,
        "timeline": summarize_timeline(records) if records else None,
        "profiles": profiles or [],
        "bridge_initial_calls": initial_bridge_calls,
        "formulas": {'''))

rep.append(('''            "provenance": "provenance.json",
            "config": "config.json",
        },
    }''', '''            "provenance": "provenance.json",
            "config": "config.json",
            "profiles": "profile/*.json (top-ops tables, only with profile_step)",
        },
    }'''))
for a, b in rep:
    assert a in s, a[:80]
    s = s.replace(a, b)
p.write_text(s)
print("ok")
