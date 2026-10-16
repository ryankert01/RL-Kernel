from pathlib import Path

p = Path("/home/ubuntu/RL-Kernel/.claude/worktrees/wf_65f5032c-579-2/rl_engine/models/muse_glimmer/runtime/vllm_worker_tools.py")
s = p.read_text()
old = '''* :func:`worker_profile` — arms ``torch.profiler`` for exactly one future engine step (selected
  by its index after arming) and returns the top ops by CUDA time.
"""
'''
new = '''* :func:`worker_profile` — arms ``torch.profiler`` for exactly one future engine step (selected
  by its index after arming) and returns the top ops by CUDA time.
* :func:`worker_native_lora_report` — native mode: rebuilds the canonical LoRA state dict from the
  tensors vLLM's own LoRA manager loaded on this rank (PEFT names mapped back) and recomputes
  ``e2e.sync.lora_sha256`` on it, so the native engine's ``provenance()["lora_sha256"]`` is a
  worker-side recomputation, not the driver's echo.
* :func:`worker_rl_engine_file` — where this worker imported ``rl_engine`` from (the driver fails
  if it differs from its own: a stale editable install must not be what the workers run).
"""
'''
assert old in s
s = s.replace(old, new, 1)
old = '''import time
from typing import Any

import torch
'''
new = '''import re
import time
from pathlib import Path
from typing import Any

import torch
'''
assert old in s
s = s.replace(old, new, 1)
old = '''__all__ = ["worker_memory", "worker_profile", "worker_step_timer"]'''
new = '''def worker_rl_engine_file(_worker: Any = None) -> str | None:
    """``rl_engine.__file__`` as resolved in this worker process."""

    try:
        import rl_engine

        return str(Path(rl_engine.__file__).resolve())
    except Exception:  # noqa: BLE001
        return None


_NATIVE_LORA_MODULE = re.compile(r"layers\\.(\\d+)\\.(self_attn|mlp)\\.([a-z_]+)$")
# vLLM module (after packing / the rename mapper) -> canonical slot names, in pack order.
_NATIVE_TO_CANONICAL = {
    ("self_attn", "qkv_proj"): ("q_proj", "k_proj", "v_proj"),
    ("self_attn", "q_proj"): ("q_proj",),
    ("self_attn", "k_proj"): ("k_proj",),
    ("self_attn", "v_proj"): ("v_proj",),
    ("self_attn", "output_gate_proj"): ("gate_proj",),
    ("self_attn", "o_proj"): ("o_proj",),
    ("mlp", "gate_up_proj"): ("mlp.gate_proj", "mlp.up_proj"),
    ("mlp", "gate_proj"): ("mlp.gate_proj",),
    ("mlp", "up_proj"): ("mlp.up_proj",),
    ("mlp", "down_proj"): ("mlp.down_proj",),
}


def canonical_state_from_lora_model(lora_model: Any) -> tuple[dict[str, torch.Tensor], list[str]]:
    """``vllm.lora.lora_model.LoRAModel`` -> ``{layers.{i}.{proj}.lora_{A,B}: tensor}``.

    Raises on an unknown module name, a non-unit scaling (vLLM would have scaled ``lora_b``,
    i.e. it holds a different adapter) or a duplicate slot.  Returns the state dict and the list
    of vLLM module names it came from.
    """

    out: dict[str, torch.Tensor] = {}
    modules = []
    for name, weights in lora_model.loras.items():
        match = _NATIVE_LORA_MODULE.search(name)
        if match is None or (match.group(2), match.group(3)) not in _NATIVE_TO_CANONICAL:
            raise RuntimeError(f"native LoRA: unexpected vLLM LoRA module {name!r}")
        layer = int(match.group(1))
        slots = _NATIVE_TO_CANONICAL[(match.group(2), match.group(3))]
        a_list = weights.lora_a if isinstance(weights.lora_a, list) else [weights.lora_a]
        b_list = weights.lora_b if isinstance(weights.lora_b, list) else [weights.lora_b]
        scalings = weights.scaling if isinstance(weights.scaling, list) else [weights.scaling]
        if len(a_list) != len(slots) or len(b_list) != len(slots):
            raise RuntimeError(f"native LoRA: {name} packs {len(a_list)} slots, expected {slots}")
        for slot, a, b, scaling in zip(slots, a_list, b_list, scalings, strict=True):
            if a is None or b is None:
                continue
            if scaling not in (1, 1.0, None):
                raise RuntimeError(f"native LoRA: {name}/{slot} has scaling {scaling} != 1")
            for mat, tensor in (("A", a), ("B", b)):
                key = f"layers.{layer}.{slot}.lora_{mat}"
                if key in out:
                    raise RuntimeError(f"native LoRA: duplicate slot {key}")
                out[key] = tensor.detach().to("cpu").contiguous()
        modules.append(name)
    return out, modules


def worker_native_lora_report(worker: Any, lora_int_id: int) -> dict[str, Any]:
    """Recompute the canonical LoRA hash from the adapter vLLM's LoRA manager holds on this rank.

    The adapter must have been added (``LLMEngine.add_lora``); a missing adapter raises.  Also
    reports the number of slots with a non-zero ``B`` (a sync of a real adapter installs 416).
    """

    from rl_engine.models.muse_glimmer.e2e.sync import lora_sha256

    manager = getattr(getattr(worker, "model_runner", None), "lora_manager", None)
    if manager is None:
        raise RuntimeError("native LoRA report: this worker has no LoRA manager (enable_lora?)")
    lora_model = manager._adapter_manager.get_adapter(int(lora_int_id))
    if lora_model is None:
        raise RuntimeError(f"native LoRA report: adapter {lora_int_id} is not loaded on this rank")
    t0 = time.perf_counter()
    state, modules = canonical_state_from_lora_model(lora_model)
    digest = lora_sha256(state) if state else None
    nonzero_b = sum(
        1 for key, tensor in state.items() if key.endswith("lora_B") and bool(tensor.any())
    )
    rank = None
    try:
        from vllm.distributed import get_tensor_model_parallel_rank

        rank = int(get_tensor_model_parallel_rank())
    except Exception:  # noqa: BLE001
        pass
    return {
        "rank": rank,
        "lora_int_id": int(lora_int_id),
        "recomputed_sha256": digest,
        "slots": len(state) // 2,
        "slots_nonzero_b": nonzero_b,
        "vllm_modules": len(modules),
        "rehash_ms": (time.perf_counter() - t0) * 1e3,
        "source": "vLLM LoRAModel tensors on this worker (PEFT names mapped back)",
        "rl_engine_file": worker_rl_engine_file(),
    }


__all__ = [
    "canonical_state_from_lora_model",
    "worker_memory",
    "worker_native_lora_report",
    "worker_profile",
    "worker_rl_engine_file",
    "worker_step_timer",
]'''
assert old in s
s = s.replace(old, new, 1)
p.write_text(s)
print("ok")
