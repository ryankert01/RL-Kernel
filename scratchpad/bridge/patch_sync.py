from pathlib import Path

p = Path("/home/ubuntu/RL-Kernel/.claude/worktrees/wf_7b064961-36f-3/rl_engine/models/muse_glimmer/e2e/sync.py")
s = p.read_text()
old = '''    rollout.load_lora(loaded.state_dict, loaded.sha256)
    reported = rollout.provenance().get("lora_sha256")
    identity = weight_sync_identity(package.sha256, reported)
    if not identity:
        raise WeightSyncError(
            f"weight_sync_identity failed: exported {package.sha256}, rollout reports {reported!r}"
        )
    return {
        "transport": transport,
        "exported_sha256": package.sha256,
        "rollout_sha256": reported,
        "identity": identity,
        "file": str(via_file) if via_file is not None else None,
        "step": package.step,
    }'''
new = '''    rollout.load_lora(loaded.state_dict, loaded.sha256)
    reported, source = rollout_lora_sha256(rollout.provenance())
    identity = weight_sync_identity(package.sha256, reported)
    if not identity:
        raise WeightSyncError(
            f"weight_sync_identity failed: exported {package.sha256}, rollout reports "
            f"{reported!r} ({source})"
        )
    return {
        "transport": transport,
        "exported_sha256": package.sha256,
        "rollout_sha256": reported,
        "rollout_sha256_source": source,
        "identity": identity,
        "file": str(via_file) if via_file is not None else None,
        "step": package.step,
    }


def rollout_lora_sha256(provenance: Mapping[str, Any]) -> tuple[str | None, str]:
    """The LoRA hash a rollout engine reports holding, and where it was found.

    ``provenance["lora_sha256"]`` (the engine-level contract) wins. ``StrictRolloutEngine``
    (vLLM) reports it per TP worker instead (``provenance["workers"][i]["lora_sha256"]``); that
    is accepted only when every worker reports the same string.
    """

    top = provenance.get("lora_sha256")
    if top is not None:
        return (top if isinstance(top, str) else None), "provenance.lora_sha256"
    workers = provenance.get("workers")
    if isinstance(workers, list) and workers:
        values = [w.get("lora_sha256") if isinstance(w, Mapping) else None for w in workers]
        if all(isinstance(v, str) for v in values) and len(set(values)) == 1:
            return values[0], f"provenance.workers[*].lora_sha256 ({len(values)} ranks agree)"
        return None, f"provenance.workers[*].lora_sha256 missing or disagreeing: {values}"
    return None, "no lora_sha256 in rollout provenance"'''
assert old in s
s = s.replace(old, new)
p.write_text(s)
print("ok")
