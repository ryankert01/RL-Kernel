# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""The GRPO loop of the Muse-Glimmer-30B strict POC (``final_e2e_lora``).

Per step: sample prompts -> ``rollout.generate`` (n samples per prompt, logprobs at sampling
time) -> verifiable rewards -> ``trainer.token_logprobs`` on the SAME token ids -> drift metrics
(+ sidecar) -> GRPO loss -> ``trainer.step`` -> ``export_lora`` -> hash-verified transfer ->
``rollout.load_lora`` -> ``weight_sync_identity`` check -> JSON / CSV log.

Mode strings are ``<train>/<rollout>`` with each side ``strict`` or ``native`` (the RFC's
"training-rollout combinations"). The LoRA is synced BEFORE the first rollout so that the
step-k comparison is between two engines holding byte-identical weights; the update of step k is
visible to the rollout of step k+1 (on-policy, one optimizer step per batch).

No VIME, no Ray: one orchestrator process drives two engine objects. The rollout lives in this
process (vLLM env); with ``trainer_bridge`` the trainer is a :class:`bridge.RemoteTrainer` whose
server runs in the Megatron env (``trainer_server.py``). The orchestrator never touches CUDA.

Timeline: every step records the wall seconds of the phases in :data:`timeline.PHASE_ORDER`
(``steps.jsonl["timeline"]``), the RemoteTrainer call stats (RTT vs server compute, bytes), and
``summary.json["timeline"]`` holds the per-phase series for plotting. ``profile_step`` captures
``torch.profiler`` top-ops tables of that step's rollout (orchestrator process) and of the
trainer scoring + train step (on the trainer server when bridged, else in-process).
"""

from __future__ import annotations

import csv
import json
import random
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import torch

from rl_engine.models.muse_glimmer.e2e.data import Task
from rl_engine.models.muse_glimmer.e2e.drift import (
    DriftMetrics,
    as_fp32_logprobs,
    compare_logprobs,
    write_sidecar,
)
from rl_engine.models.muse_glimmer.e2e.engines import (
    RolloutEngineLike,
    TrainerLike,
    coerce_samples,
)
from rl_engine.models.muse_glimmer.e2e.grpo import (
    GRPOLossOutput,
    group_normalized_advantages,
    grpo_token_loss,
    reward_stats,
)
from rl_engine.models.muse_glimmer.e2e.sync import export_lora_verified, transfer_lora
from rl_engine.models.muse_glimmer.e2e.timeline import PhaseClock, profiled, summarize_timeline

MODES = ("native/native", "strict/native", "native/strict", "strict/strict")


def parse_mode(mode: str) -> tuple[bool, bool]:
    """``"<train>/<rollout>"`` -> ``(train_strict, rollout_strict)``."""

    parts = mode.strip().lower().split("/")
    if len(parts) != 2 or any(p not in ("strict", "native") for p in parts):
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    return parts[0] == "strict", parts[1] == "strict"


@dataclass
class LoopConfig:
    run_dir: Path
    steps: int = 8
    prompts_per_step: int = 8
    samples_per_prompt: int = 4
    max_tokens: int = 256
    temperature: float = 1.0
    lora_rank: int = 16
    lr: float = 1e-5
    clip_eps: float = 0.2
    kl_coef: float = 0.0
    mode: str = "strict/strict"
    seed: int = 1234
    model_path: str = "~/models/Muse-Glimmer-30B"
    max_model_len: int = 4096
    rollout_tp: int = 2
    rollout_gpus: list[int] = field(default_factory=lambda: [0, 1])
    train_tp: int = 4
    train_gpus: list[int] = field(default_factory=lambda: [4, 5, 6, 7])
    sync_via_file: bool = False
    recheck_rollout_logprobs: bool = True
    mock: bool = False
    mock_perturb: float = 0.0
    task: str = "gsm8k"
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

    @property
    def train_strict(self) -> bool:
        return parse_mode(self.mode)[0]

    @property
    def rollout_strict(self) -> bool:
        return parse_mode(self.mode)[1]

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["run_dir"] = str(self.run_dir)
        out["train_strict"] = self.train_strict
        out["rollout_strict"] = self.rollout_strict
        return out


CSV_COLUMNS = [
    "step",
    "reward_mean",
    "reward_std",
    "groups_with_any_success_frac",
    "num_samples",
    "num_completion_tokens",
    "completion_len_mean",
    "completion_len_max",
    "finish_stop_frac",
    "loss",
    "policy_loss",
    "kl",
    "ratio_mean",
    "ratio_exactly_one_frac",
    "clip_frac",
    "grad_norm",
    "bitwise_mismatch_count",
    "value_mismatch_count",
    "max_abs_diff",
    "mean_abs_diff",
    "mismatch_k3_kl",
    "first_mismatch_sample",
    "first_mismatch_token",
    "rollout_decode_vs_prefill_bitwise_mismatch_count",
    "rollout_decode_vs_prefill_max_abs_diff",
    "lora_sha256",
    "weight_sync_identity",
    "rollout_time_s",
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
]


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.tolist()
    return str(value)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, default=_json_default) + "\n", encoding="utf-8")


def _flatten_rollout(
    prompts: Sequence[Any], groups: Sequence[Sequence[Any]], task: Task, samples_per_prompt: int
) -> dict[str, Any]:
    sequences: list[list[int]] = []
    starts: list[int] = []
    rollout_logps: list[torch.Tensor] = []
    rewards: list[float] = []
    finish: list[str] = []
    completions: list[list[int]] = []
    if len(groups) != len(prompts):
        raise ValueError(f"rollout returned {len(groups)} groups for {len(prompts)} prompts")
    for prompt, group in zip(prompts, groups):
        if len(group) != samples_per_prompt:
            raise ValueError(f"group has {len(group)} samples, expected {samples_per_prompt}")
        for sample in group:
            if not sample.token_ids:
                raise ValueError("rollout returned an empty completion")
            sequences.append(list(prompt.token_ids) + list(sample.token_ids))
            starts.append(len(prompt.token_ids))
            rollout_logps.append(as_fp32_logprobs(sample.logprobs, label="rollout"))
            rewards.append(float(task.reward(sample.token_ids, prompt)))
            finish.append(sample.finish_reason)
            completions.append(list(sample.token_ids))
    return {
        "sequences": sequences,
        "starts": starts,
        "rollout_logps": rollout_logps,
        "rewards": rewards,
        "finish": finish,
        "completions": completions,
    }


def _check_train_logps(
    train_logps: Sequence[torch.Tensor], sequences: list[list[int]], starts: list[int]
) -> None:
    if len(train_logps) != len(sequences):
        raise ValueError(
            f"trainer returned {len(train_logps)} tensors for {len(sequences)} samples"
        )
    for i, (t, seq, start) in enumerate(zip(train_logps, sequences, starts)):
        expected = len(seq) - start
        if t.dim() != 1 or t.numel() != expected:
            raise ValueError(
                f"sample {i}: trainer logprobs shape {tuple(t.shape)}, expected ({expected},)"
            )
        if t.dtype != torch.float32:
            raise TypeError(f"sample {i}: trainer logprobs must be fp32, got {t.dtype}")
        if not t.requires_grad:
            raise ValueError(f"sample {i}: trainer logprobs carry no autograd history")


def _step_record(
    *,
    step: int,
    flat: dict[str, Any],
    drift: DriftMetrics,
    rollout_internal: DriftMetrics | None,
    loss_out: GRPOLossOutput,
    step_info: dict[str, Any],
    sync: dict[str, Any],
    timings: dict[str, float],
    advantages: torch.Tensor,
    samples_per_prompt: int,
    timeline: dict[str, Any] | None = None,
    bridge_calls: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    lengths = [len(c) for c in flat["completions"]]
    record: dict[str, Any] = {
        "step": step,
        "num_samples": len(lengths),
        "num_completion_tokens": int(sum(lengths)),
        "completion_len_mean": sum(lengths) / len(lengths),
        "completion_len_max": max(lengths),
        "finish_stop_frac": sum(1 for f in flat["finish"] if f == "stop") / len(lengths),
        "rewards": flat["rewards"],
        "advantages": [float(a) for a in advantages.tolist()],
        "grpo": loss_out.stats,
        "trainer_step": step_info,
        "drift": drift.to_dict(),
        "rollout_decode_vs_prefill": rollout_internal.to_dict() if rollout_internal else None,
        "sync": sync,
        "timings": timings,
        "timeline": timeline or {},
        "bridge_calls": bridge_calls,
    }
    record.update(reward_stats(flat["rewards"], samples_per_prompt))
    return record


def _csv_row(record: dict[str, Any]) -> dict[str, Any]:
    drift = record["drift"]
    internal = record.get("rollout_decode_vs_prefill") or {}
    first = drift.get("first_mismatch")
    grpo = record["grpo"]
    timings = record["timings"]
    return {
        "step": record["step"],
        "reward_mean": record["reward_mean"],
        "reward_std": record["reward_std"],
        "groups_with_any_success_frac": record["groups_with_any_success_frac"],
        "num_samples": record["num_samples"],
        "num_completion_tokens": record["num_completion_tokens"],
        "completion_len_mean": record["completion_len_mean"],
        "completion_len_max": record["completion_len_max"],
        "finish_stop_frac": record["finish_stop_frac"],
        "loss": grpo["loss"],
        "policy_loss": grpo["policy_loss"],
        "kl": grpo["kl"],
        "ratio_mean": grpo["ratio_mean"],
        "ratio_exactly_one_frac": grpo["ratio_exactly_one_frac"],
        "clip_frac": grpo["clip_frac"],
        "grad_norm": record["trainer_step"].get("grad_norm"),
        "bitwise_mismatch_count": drift["bitwise_mismatch_count"],
        "value_mismatch_count": drift["value_mismatch_count"],
        "max_abs_diff": drift["max_abs_diff"],
        "mean_abs_diff": drift["mean_abs_diff"],
        "mismatch_k3_kl": drift["mismatch_k3_kl"],
        "first_mismatch_sample": first[0] if first else "",
        "first_mismatch_token": first[1] if first else "",
        "rollout_decode_vs_prefill_bitwise_mismatch_count": internal.get(
            "bitwise_mismatch_count", ""
        ),
        "rollout_decode_vs_prefill_max_abs_diff": internal.get("max_abs_diff", ""),
        "lora_sha256": record["sync"]["exported_sha256"],
        "weight_sync_identity": record["sync"]["identity"],
        "rollout_time_s": timings["rollout_s"],
        "train_logp_time_s": timings["train_logp_s"],
        "train_step_time_s": timings["train_step_s"],
        "sync_time_s": timings["sync_s"],
        "step_time_s": timings["step_s"],
        "reward_time_s": timings.get("reward_s", ""),
        "drift_time_s": timings.get("drift_s", ""),
        "recheck_time_s": timings.get("recheck_s", ""),
        "sidecar_time_s": timings.get("sidecar_s", ""),
        "lora_export_time_s": timings.get("lora_export_s", ""),
        "lora_transfer_time_s": timings.get("lora_transfer_s", ""),
        "rollout_tokens_per_s": (record.get("timeline") or {}).get("rollout_tokens_per_s", ""),
    }


def _format_line(record: dict[str, Any]) -> str:
    drift = record["drift"]
    first = drift["first_mismatch"]
    first_text = f"({first[0]},{first[1]})" if first else "-"
    return (
        f"step {record['step']:3d} | reward {record['reward_mean']:.3f} "
        f"| loss {record['grpo']['loss']:+.5f} ratio {record['grpo']['ratio_mean']:.6f} "
        f"clip {record['grpo']['clip_frac']:.3f} "
        f"| drift bitwise {drift['bitwise_mismatch_count']}/{drift['token_count']} "
        f"max {drift['max_abs_diff']:.3e} first {first_text} "
        f"| sync {record['sync']['exported_sha256'][:12]} ok={record['sync']['identity']} "
        f"| {record['timings']['step_s']:.1f}s"
    )


def _summarize(
    cfg: LoopConfig,
    records: list[dict[str, Any]],
    provenance: dict[str, Any],
    task_description: dict[str, Any],
    *,
    status: str,
    error: str | None,
    initial_sync: dict[str, Any],
    profiles: list[dict[str, Any]] | None = None,
    initial_bridge_calls: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    first_drift = None
    total_tokens = 0
    total_bitwise = 0
    max_abs = 0.0
    for record in records:
        drift = record["drift"]
        total_tokens += drift["token_count"]
        total_bitwise += drift["bitwise_mismatch_count"]
        max_abs = max(max_abs, drift["max_abs_diff"])
        if first_drift is None and drift["bitwise_mismatch_count"] > 0:
            first_drift = {
                "step": record["step"],
                "sample": drift["first_mismatch"][0],
                "token": drift["first_mismatch"][1],
                "abs_diff": drift["first_mismatch_abs_diff"],
            }
    rewards = [r["reward_mean"] for r in records]
    timings = {
        key: (sum(r["timings"][key] for r in records) / len(records)) if records else None
        for key in ("rollout_s", "train_logp_s", "train_step_s", "sync_s", "step_s")
    }
    sync_checks = [r["sync"]["identity"] for r in records] + [initial_sync["identity"]]
    return {
        "status": status,
        "error": error,
        "orchestrator": "rl_engine.models.muse_glimmer.e2e (single process; no VIME, no Ray)",
        "rfc_row": "final_e2e_lora",
        "mode": cfg.mode,
        "train_strict": cfg.train_strict,
        "rollout_strict": cfg.rollout_strict,
        "config": cfg.to_dict(),
        "task": task_description,
        "provenance": provenance,
        "steps_completed": len(records),
        "first_drift": first_drift,
        "drift_totals": {
            "bitwise_mismatch_count": total_bitwise,
            "token_count": total_tokens,
            "bitwise_mismatch_frac": (total_bitwise / total_tokens) if total_tokens else None,
            "max_abs_diff": max_abs,
            "all_steps_bitwise_equal": bool(records) and total_bitwise == 0 and max_abs == 0.0,
        },
        "reward": {
            "first_step": rewards[0] if rewards else None,
            "last_step": rewards[-1] if rewards else None,
            "mean": (sum(rewards) / len(rewards)) if rewards else None,
            "trajectory": rewards,
        },
        "weight_sync": {
            "checks": len(sync_checks),
            "passed": int(sum(1 for c in sync_checks if c)),
            "all_passed": all(sync_checks),
            "initial_sha256": initial_sync["exported_sha256"],
            "final_sha256": records[-1]["sync"]["exported_sha256"] if records else None,
            "transport": initial_sync["transport"],
        },
        "timings_mean_s": timings,
        "timeline": summarize_timeline(records) if records else None,
        "profiles": profiles or [],
        "bridge_initial_calls": initial_bridge_calls,
        "formulas": {
            "bitwise_mismatch_count": "tokens whose 4 fp32 bytes differ (train vs rollout)",
            "max_abs_diff": "max |train - rollout| in fp64 over active completion tokens",
            "mismatch_k3_kl": "mean(exp(d) - d - 1), d = train - rollout",
            "first_drift": "first (step, sample, token) with a bitwise mismatch, batch order",
            "advantage": "(r - mean_group) / max(std_group, 1e-6), population std",
            "loss": (
                "-mean_tokens(min(rho*A, clip(rho, 1-eps, 1+eps)*A)), "
                "rho = exp(lp_train - lp_rollout)"
            ),
            "lora_sha256": "sha256 over sorted(name|dtype|shape\\n + raw LE bytes)",
        },
        "files": {
            "steps": "steps.jsonl",
            "rounds": "rounds.csv",
            "sidecars": "mismatch-sidecars/step{n:04d}.pt",
            "lora_packages": "lora/step{n:04d}.pt (only with sync_via_file)",
            "provenance": "provenance.json",
            "config": "config.json",
            "profiles": "profile/*.json (top-ops tables, only with profile_step)",
        },
    }


README_TEMPLATE = """# Muse-Glimmer-30B strict POC — GRPO LoRA loop ({mode})

Row `final_e2e_lora` of RFC #501. Orchestrator: `rl_engine.models.muse_glimmer.e2e` (one
process, no VIME, no Ray). Mode `{mode}` = `<train>/<rollout>`.

| Setting | Value |
|---|---|
| Rollout engine | {rollout_engine} (TP{rollout_tp}, GPUs {rollout_gpus}, strict={rollout_strict}) |
| Trainer engine | {trainer_engine} (TP{train_tp}, GPUs {train_gpus}, strict={train_strict}) |
| Task | {task} |
| Batch | {steps} steps × {prompts} prompts × {samples} samples |
| Generation | max_tokens {max_tokens}, temperature {temperature} |
| LoRA / optimizer | rank {lora_rank}, lr {lr}, clip ε {clip_eps}, KL coef {kl_coef} |
| Seed | {seed} |

| Metric | Value |
|---|---:|
| Status | {status} |
| Steps completed | {steps_completed} |
| First drift (step, sample, token) | {first_drift} |
| Bitwise mismatches / completion tokens | {bitwise} / {tokens} |
| Max abs logprob diff (all steps) | {max_abs:.3e} |
| Reward mean first / last / all steps | {reward_first} / {reward_last} / {reward_mean} |
| Weight-sync identity checks | {sync_passed} / {sync_checks} passed |
| Mean time per step (s) | {step_s} |
| Rollout / train-logp / train-step / sync (s) | {rollout_s} / {train_logp_s} / {train_step_s} |
| Sync (s) | {sync_s} |

Files: `steps.jsonl` (one JSON object per step with rewards, advantages, GRPO stats, drift and
sync records), `rounds.csv` (one row per step), `summary.json` (this table as data plus the
formulas), `provenance.json` (both engines before and after the run), `config.json`,
`mismatch-sidecars/step*.pt` (train/rollout logprobs, masks and token ids per step; recompute
the drift metrics offline with `drift.metrics_from_sidecar`).
"""


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_readme(run_dir: Path, summary: dict[str, Any]) -> Path:
    cfg = summary["config"]
    first = summary["first_drift"]
    prov = summary["provenance"]
    timings = summary["timings_mean_s"]
    text = README_TEMPLATE.format(
        mode=summary["mode"],
        rollout_engine=prov.get("rollout_start", {}).get("engine", "?"),
        trainer_engine=prov.get("trainer_start", {}).get("engine", "?"),
        rollout_tp=cfg["rollout_tp"],
        rollout_gpus=cfg["rollout_gpus"],
        rollout_strict=summary["rollout_strict"],
        train_tp=cfg["train_tp"],
        train_gpus=cfg["train_gpus"],
        train_strict=summary["train_strict"],
        task=summary["task"].get("task", "?"),
        steps=cfg["steps"],
        prompts=cfg["prompts_per_step"],
        samples=cfg["samples_per_prompt"],
        max_tokens=cfg["max_tokens"],
        temperature=cfg["temperature"],
        lora_rank=cfg["lora_rank"],
        lr=cfg["lr"],
        clip_eps=cfg["clip_eps"],
        kl_coef=cfg["kl_coef"],
        seed=cfg["seed"],
        status=summary["status"],
        steps_completed=summary["steps_completed"],
        first_drift=(
            f"({first['step']}, {first['sample']}, {first['token']}) |Δ|={first['abs_diff']:.3e}"
            if first
            else "none (bitwise equal at every step)"
        ),
        bitwise=summary["drift_totals"]["bitwise_mismatch_count"],
        tokens=summary["drift_totals"]["token_count"],
        max_abs=summary["drift_totals"]["max_abs_diff"],
        reward_first=_fmt(summary["reward"]["first_step"]),
        reward_last=_fmt(summary["reward"]["last_step"]),
        reward_mean=_fmt(summary["reward"]["mean"]),
        sync_passed=summary["weight_sync"]["passed"],
        sync_checks=summary["weight_sync"]["checks"],
        step_s=_fmt(timings["step_s"], 2),
        rollout_s=_fmt(timings["rollout_s"], 2),
        train_logp_s=_fmt(timings["train_logp_s"], 2),
        train_step_s=_fmt(timings["train_step_s"], 2),
        sync_s=_fmt(timings["sync_s"], 3),
    )
    path = run_dir / "README.md"
    path.write_text(text, encoding="utf-8")
    return path


def _drain_bridge_calls(trainer: Any) -> list[dict[str, Any]] | None:
    """RemoteTrainer call stats since the last drain (``None`` for an in-process trainer)."""

    drain = getattr(trainer, "drain_call_stats", None)
    return drain() if callable(drain) else None


def _engine_profile(engine: Any, action: str, enabled: bool) -> None:
    """Call ``engine.start_profile()`` / ``stop_profile()`` when the engine offers them."""

    if not enabled:
        return
    hook = getattr(engine, f"{action}_profile", None)
    if callable(hook):
        hook()


def run_loop(
    cfg: LoopConfig,
    task: Task,
    rollout: RolloutEngineLike,
    trainer: TrainerLike,
    *,
    log: Callable[[str], None] | None = print,
) -> dict[str, Any]:
    """Run ``cfg.steps`` GRPO steps; always writes summary.json/README.md, re-raises errors."""

    run_dir = Path(cfg.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "mismatch-sidecars").mkdir(exist_ok=True)
    _write_json(run_dir / "config.json", cfg.to_dict())
    emit = log or (lambda _message: None)
    rng = random.Random(cfg.seed)
    samples_per_prompt = cfg.samples_per_prompt
    task_description = task.describe()

    lora_dir = run_dir / "lora"
    profile_dir = run_dir / "profile"
    profiles: list[dict[str, Any]] = []
    package = export_lora_verified(trainer, step=0)
    initial_sync = transfer_lora(
        package, rollout, via_file=(lora_dir / "step0000.pt") if cfg.sync_via_file else None
    )
    provenance: dict[str, Any] = {
        "rollout_start": rollout.provenance(),
        "trainer_start": trainer.provenance(),
    }
    _write_json(run_dir / "provenance.json", provenance)
    initial_bridge_calls = _drain_bridge_calls(trainer)
    emit(
        f"initial sync ok: lora {initial_sync['exported_sha256'][:12]} "
        f"via {initial_sync['transport']}"
    )

    records: list[dict[str, Any]] = []
    status = "ok"
    error: str | None = None
    steps_path = run_dir / "steps.jsonl"
    rounds_path = run_dir / "rounds.csv"
    with steps_path.open("w", encoding="utf-8") as steps_file, rounds_path.open(
        "w", encoding="utf-8", newline=""
    ) as rounds_file:
        writer = csv.DictWriter(rounds_file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        try:
            for step in range(cfg.steps):
                clock = PhaseClock()
                prompts = task.sample(rng, cfg.prompts_per_step)
                profile_now = cfg.profile_step is not None and step == cfg.profile_step
                remote_profile = profile_now and callable(getattr(trainer, "arm_profile", None))
                if remote_profile:
                    trainer.arm_profile(profile_dir, f"step{step:04d}")  # type: ignore[attr-defined]

                with clock.phase("rollout_generate"), profiled(
                    profile_now,
                    out_path=profile_dir / f"orchestrator_step{step:04d}_rollout_generate.json",
                    label="orchestrator.rollout.generate",
                ) as prof_rollout:
                    _engine_profile(rollout, "start", profile_now)
                    try:
                        groups = coerce_samples(
                            rollout.generate(
                                [p.token_ids for p in prompts],
                                n=samples_per_prompt,
                                max_tokens=cfg.max_tokens,
                                temperature=cfg.temperature,
                            )
                        )
                    finally:
                        _engine_profile(rollout, "stop", profile_now)
                with clock.phase("reward"):
                    flat = _flatten_rollout(prompts, groups, task, samples_per_prompt)

                local_profile = profile_now and not remote_profile
                with clock.phase("trainer_logprobs"), profiled(
                    local_profile,
                    out_path=profile_dir / f"orchestrator_step{step:04d}_trainer_logprobs.json",
                    label="orchestrator.trainer.token_logprobs",
                ) as prof_logp:
                    train_logps = trainer.token_logprobs(flat["sequences"], flat["starts"])
                _check_train_logps(train_logps, flat["sequences"], flat["starts"])

                with clock.phase("drift_metrics"):
                    train_cpu = [t.detach().to("cpu", torch.float32) for t in train_logps]
                    drift = compare_logprobs(train_cpu, flat["rollout_logps"])
                rollout_internal = None
                prefill_logps = None
                with clock.phase("rollout_recheck"):
                    if cfg.recheck_rollout_logprobs:
                        prefill = rollout.logprobs(flat["sequences"], flat["starts"])
                        prefill_logps = [
                            as_fp32_logprobs(v, label="rollout-prefill") for v in prefill
                        ]
                        rollout_internal = compare_logprobs(prefill_logps, flat["rollout_logps"])
                with clock.phase("sidecar"):
                    write_sidecar(
                        run_dir / "mismatch-sidecars" / f"step{step:04d}.pt",
                        step=step,
                        token_ids=flat["sequences"],
                        completion_start=flat["starts"],
                        train_logprobs=train_cpu,
                        rollout_logprobs=flat["rollout_logps"],
                        metrics=drift,
                        provenance={"mode": cfg.mode, "lora_sha256": package.sha256},
                        extra={
                            "rewards": flat["rewards"],
                            "finish_reasons": flat["finish"],
                            "rollout_prefill_log_probs": prefill_logps,
                            "rollout_decode_vs_prefill": (
                                rollout_internal.to_dict() if rollout_internal else None
                            ),
                        },
                    )

                with clock.phase("grpo_loss"):
                    advantages = group_normalized_advantages(flat["rewards"], samples_per_prompt)
                    loss_out = grpo_token_loss(
                        train_logps,
                        flat["rollout_logps"],
                        advantages,
                        clip_eps=cfg.clip_eps,
                        kl_coef=cfg.kl_coef,
                    )
                with clock.phase("backward_step"), profiled(
                    local_profile,
                    out_path=profile_dir / f"orchestrator_step{step:04d}_trainer_step.json",
                    label="orchestrator.trainer.step",
                ) as prof_step:
                    step_info = trainer.step(loss_out.loss)

                with clock.phase("lora_export"):
                    package = export_lora_verified(trainer, step=step + 1)
                with clock.phase("lora_transfer_load"):
                    sync = transfer_lora(
                        package,
                        rollout,
                        via_file=(
                            (lora_dir / f"step{step + 1:04d}.pt") if cfg.sync_via_file else None
                        ),
                    )

                bridge_calls = _drain_bridge_calls(trainer)
                if profile_now:
                    entry = {
                        "step": step,
                        "orchestrator_rollout_generate": prof_rollout,
                        "orchestrator_trainer_logprobs": prof_logp or None,
                        "orchestrator_trainer_step": prof_step or None,
                        "trainer_server": [
                            {"op": c["op"], **(c.get("server") or {}).get("profile", {})}
                            for c in bridge_calls or []
                            if (c.get("server") or {}).get("profile")
                        ],
                    }
                    profiles.append(entry)
                phases = clock.as_dict()
                tokens = int(sum(len(c) for c in flat["completions"]))
                rollout_s = phases["rollout_generate"]
                timings = {
                    "rollout_s": rollout_s,
                    "train_logp_s": phases["trainer_logprobs"],
                    "train_step_s": phases["backward_step"],
                    "sync_s": phases["lora_export"] + phases["lora_transfer_load"],
                    "step_s": clock.elapsed(),
                    "reward_s": phases["reward"],
                    "drift_s": phases["drift_metrics"],
                    "recheck_s": phases["rollout_recheck"],
                    "sidecar_s": phases["sidecar"],
                    "lora_export_s": phases["lora_export"],
                    "lora_transfer_s": phases["lora_transfer_load"],
                }
                timeline = {
                    "profiled": profile_now,  # profiler overhead inflates this step's times
                    "phases_s": phases,
                    "tokens_generated": tokens,
                    "rollout_tokens_per_s": (tokens / rollout_s) if rollout_s > 0 else None,
                    "lora_bytes": int(
                        sum(t.numel() * t.element_size() for t in package.state_dict.values())
                    ),
                }
                record = _step_record(
                    step=step,
                    flat=flat,
                    drift=drift,
                    rollout_internal=rollout_internal,
                    loss_out=loss_out,
                    step_info=step_info,
                    sync=sync,
                    timings=timings,
                    advantages=advantages,
                    samples_per_prompt=samples_per_prompt,
                    timeline=timeline,
                    bridge_calls=bridge_calls,
                )
                records.append(record)
                steps_file.write(json.dumps(record, default=_json_default) + "\n")
                steps_file.flush()
                writer.writerow(_csv_row(record))
                rounds_file.flush()
                emit(_format_line(record))
        except BaseException as exc:  # noqa: B036 - always leave a summary behind
            status = "failed"
            error = "".join(traceback.format_exception_only(type(exc), exc)).strip()
            raise
        finally:
            try:
                provenance["rollout_end"] = rollout.provenance()
                provenance["trainer_end"] = trainer.provenance()
            except Exception as prov_exc:  # noqa: BLE001 - provenance is best effort here
                provenance["end_error"] = repr(prov_exc)
            _write_json(run_dir / "provenance.json", provenance)
            summary = _summarize(
                cfg,
                records,
                provenance,
                task_description,
                status=status,
                error=error,
                initial_sync=initial_sync,
                profiles=profiles,
                initial_bridge_calls=initial_bridge_calls,
            )
            _write_json(run_dir / "summary.json", summary)
            write_readme(run_dir, summary)
    return summary
