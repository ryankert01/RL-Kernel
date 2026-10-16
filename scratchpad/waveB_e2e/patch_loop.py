import sys
from pathlib import Path

p = Path(sys.argv[1]) / "rl_engine/models/muse_glimmer/e2e/loop.py"
s = p.read_text()


def rep(old, new, count=1):
    global s
    assert s.count(old) == count, (s.count(old), old[:80])
    s = s.replace(old, new)


rep('''Timeline: every step records the wall seconds of the phases in :data:`timeline.PHASE_ORDER`
(``steps.jsonl["timeline"]``), the RemoteTrainer call stats (RTT vs server compute, bytes), and
``summary.json["timeline"]`` holds the per-phase series for plotting. ``profile_step`` captures
``torch.profiler`` top-ops tables of that step's rollout (orchestrator process) and of the
trainer scoring + train step (on the trainer server when bridged, else in-process).
"""''', '''Timeline: every step records the wall seconds of the phases in :data:`timeline.PHASE_ORDER`
(``steps.jsonl["timeline"]``), the RemoteTrainer call stats (RTT vs server compute, bytes), and
``summary.json["timeline"]`` holds the per-phase series for plotting. ``profile_step`` captures
``torch.profiler`` top-ops tables of that step's trainer scoring + train step (on the trainer
server when bridged, else in-process) and of the rollout call AS SEEN BY THE ORCHESTRATOR
PROCESS: vLLM V1 runs the model in EngineCore / worker subprocesses, so that table holds no GPU
kernels. The rollout engine's own profiler runs only if it offers ``start_profile`` /
``stop_profile`` (``StrictRolloutEngine`` does not today); the profile entry records which
(``rollout_engine_profiler``).

Weight-sync records name both hops: ``trainer_to_orchestrator`` (``in_process`` or ``bridge``:
the trainer-bridge socket) and ``orchestrator_to_rollout`` (``memory`` or ``file``);
``transport`` joins them (``bridge+memory``) for a bridged trainer.
"""''')

rep('''import csv
import json
import random
import traceback''', '''import csv
import json
import random
import subprocess
import traceback''')

rep('''    rollout_tp: int = 2
    rollout_gpus: list[int] = field(default_factory=lambda: [0, 1])
    train_tp: int = 4
    train_gpus: list[int] = field(default_factory=lambda: [4, 5, 6, 7])''', '''    # Physical GPU ids (run_poc_grpo fills them from the lease when the flags are omitted).
    rollout_tp: int = 1
    rollout_gpus: list[int] = field(default_factory=lambda: [0])
    train_tp: int = 2
    train_gpus: list[int] = field(default_factory=lambda: [1, 2])''')

rep('''def _json_default(value: Any) -> Any:''', '''REPO_ROOT = Path(__file__).resolve().parents[4]
ROLLOUT_PROFILE_SCOPE = (
    "orchestrator process only: vLLM's EngineCore/worker subprocesses (the model's GPU work) are "
    "not in this table unless the rollout engine profiles itself (rollout_engine_profiler)"
)


def code_provenance() -> dict[str, Any]:
    """The checkout this orchestrator runs from: commit and whether tracked files differ."""

    def git(*args: str) -> str | None:
        try:
            out = subprocess.run(
                ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True,
                timeout=30,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip()

    status = git("status", "--porcelain", "--untracked-files=no")
    return {
        "repo_root": str(REPO_ROOT),
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": None if status is None else bool(status),
    }


def _json_default(value: Any) -> Any:''')

rep('''    initial_bridge_calls: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    first_drift = None''', '''    initial_bridge_calls: list[dict[str, Any]] | None = None,
    code: dict[str, Any] | None = None,
) -> dict[str, Any]:
    first_drift = None''')

rep('''        "rfc_row": "final_e2e_lora",
        "mode": cfg.mode,''', '''        "rfc_row": "final_e2e_lora",
        "code": code,
        "mode": cfg.mode,''')

rep('''            "final_sha256": records[-1]["sync"]["exported_sha256"] if records else None,
            "transport": initial_sync["transport"],
        },''', '''            "final_sha256": records[-1]["sync"]["exported_sha256"] if records else None,
            "transport": initial_sync["transport"],
            "trainer_to_orchestrator": initial_sync.get("trainer_to_orchestrator"),
            "orchestrator_to_rollout": initial_sync.get("orchestrator_to_rollout"),
        },''')

rep('''def _engine_profile(engine: Any, action: str, enabled: bool) -> None:''', '''def _trainer_transport(trainer: Any) -> str:
    """``bridge`` when the trainer lives behind the cross-environment bridge."""

    return "bridge" if callable(getattr(trainer, "bridge_info", None)) else "in_process"


def _has_engine_profiler(engine: Any) -> bool:
    return callable(getattr(engine, "start_profile", None)) and callable(
        getattr(engine, "stop_profile", None)
    )


def _engine_profile(engine: Any, action: str, enabled: bool) -> None:''')

rep('''    emit = log or (lambda _message: None)
    rng = random.Random(cfg.seed)''', '''    emit = log or (lambda _message: None)
    code = code_provenance()
    rng = random.Random(cfg.seed)''')

rep('''    package = export_lora_verified(trainer, step=0)
    initial_sync = transfer_lora(
        package, rollout, via_file=(lora_dir / "step0000.pt") if cfg.sync_via_file else None
    )''', '''    trainer_transport = _trainer_transport(trainer)
    package = export_lora_verified(trainer, step=0)
    initial_sync = transfer_lora(
        package,
        rollout,
        via_file=(lora_dir / "step0000.pt") if cfg.sync_via_file else None,
        trainer_transport=trainer_transport,
    )''')

rep('''                    sync = transfer_lora(
                        package,
                        rollout,
                        via_file=(
                            (lora_dir / f"step{step + 1:04d}.pt") if cfg.sync_via_file else None
                        ),
                    )''', '''                    sync = transfer_lora(
                        package,
                        rollout,
                        via_file=(
                            (lora_dir / f"step{step + 1:04d}.pt") if cfg.sync_via_file else None
                        ),
                        trainer_transport=trainer_transport,
                    )''')

rep('''                    entry = {
                        "step": step,
                        "orchestrator_rollout_generate": prof_rollout,''', '''                    entry = {
                        "step": step,
                        "rollout_engine_profiler": _has_engine_profiler(rollout),
                        "orchestrator_rollout_generate_scope": ROLLOUT_PROFILE_SCOPE,
                        "orchestrator_rollout_generate": prof_rollout,''')

rep('''                profiles=profiles,
                initial_bridge_calls=initial_bridge_calls,
            )''', '''                profiles=profiles,
                initial_bridge_calls=initial_bridge_calls,
                code=code,
            )''')
p.write_text(s)
print("patched", p)
