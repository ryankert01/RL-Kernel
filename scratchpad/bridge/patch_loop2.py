from pathlib import Path

p = Path("/home/ubuntu/RL-Kernel/.claude/worktrees/wf_7b064961-36f-3/rl_engine/models/muse_glimmer/e2e/loop.py")
s = p.read_text()
start = s.index("    lora_dir = run_dir / \"lora\"\n    package = export_lora_verified(trainer, step=0)")
end_marker = "                initial_sync=initial_sync,\n            )\n            _write_json(run_dir / \"summary.json\", summary)"
end = s.index(end_marker) + len(end_marker)

new = '''    lora_dir = run_dir / "lora"
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
                steps_file.write(json.dumps(record, default=_json_default) + "\\n")
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
            _write_json(run_dir / "summary.json", summary)'''
s = s[:start] + new + s[end:]

helpers = '''

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


def run_loop('''
s = s.replace("\n\ndef run_loop(", helpers, 1)
p.write_text(s)
print("ok")
