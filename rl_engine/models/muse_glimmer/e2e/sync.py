# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors
"""LoRA weight sync trainer -> rollout with a canonical content hash (``weight_sync_identity``).

``lora_sha256`` is THE hash both engines must agree on. Definition (schema
``muse_glimmer_lora_sha256_v1``):

    h = sha256()
    for name in sorted(state_dict):                      # Python str order (code points)
        t = state_dict[name].detach().cpu().contiguous()
        h.update(f"{name}|{t.dtype}|{','.join(str(d) for d in t.shape)}\\n".encode("utf-8"))
        h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())   # raw host-order bytes
    return h.hexdigest()

* ``t.dtype`` renders as ``torch.bfloat16``; shape renders as ``16,6656`` (no spaces; the empty
  string for a 0-d tensor).
* Raw bytes are the little-endian in-memory representation of the contiguous tensor (x86-64 and
  aarch64 are both little-endian; the hash is not portable to big-endian hosts by design).
* No device, stride, requires_grad or metadata is hashed; only name, dtype, shape and bytes.
* The LoRA key schema is ``layers.{i}.{proj}.lora_{A,B}`` with ``proj`` in
  ``q_proj|k_proj|v_proj|gate_proj|o_proj|mlp.gate_proj|mlp.up_proj|mlp.down_proj``,
  ``lora_A`` of shape ``(r, K)`` and ``lora_B`` of shape ``(N, r)``, both bf16.

Transport: in-memory (same process) or a ``torch.save`` package under the run directory
(separate processes). The hash is recomputed on load; the rollout reports the hash it holds in
``provenance()["lora_sha256"]`` and the loop checks it equals the exported one.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import torch

LORA_SHA256_SCHEMA = "muse_glimmer_lora_sha256_v1"
LORA_PACKAGE_SCHEMA = "muse_glimmer_lora_package_v1"
LORA_PROJECTIONS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "gate_proj",
    "o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)
LORA_KEY_RE = re.compile(
    r"^layers\.(?P<layer>\d+)\."
    r"(?P<proj>q_proj|k_proj|v_proj|gate_proj|o_proj|mlp\.gate_proj|mlp\.up_proj|mlp\.down_proj)"
    r"\.lora_(?P<mat>A|B)$"
)


class WeightSyncError(RuntimeError):
    """A LoRA package failed hash verification or schema validation."""


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    flat = tensor.detach().to("cpu").contiguous().reshape(-1)
    if flat.numel() == 0:
        return b""
    return flat.view(torch.uint8).numpy().tobytes()


def lora_sha256(state_dict: Mapping[str, torch.Tensor]) -> str:
    """Canonical sha256 of a LoRA state dict (see module docstring for the exact byte stream)."""

    digest = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = state_dict[name]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name}: expected a tensor, got {type(tensor).__name__}")
        shape = ",".join(str(int(dim)) for dim in tensor.shape)
        digest.update(f"{name}|{tensor.dtype}|{shape}\n".encode("utf-8"))
        digest.update(_tensor_bytes(tensor))
    return digest.hexdigest()


def validate_lora_state_dict(
    state_dict: Mapping[str, torch.Tensor], *, require_bf16: bool = True
) -> dict[str, Any]:
    """Check the key schema and A/B shape agreement; return a small summary."""

    if not state_dict:
        raise WeightSyncError("LoRA state dict is empty")
    ranks: set[int] = set()
    layers: set[int] = set()
    modules: dict[str, dict[str, torch.Tensor]] = {}
    for name, tensor in state_dict.items():
        match = LORA_KEY_RE.match(name)
        if match is None:
            raise WeightSyncError(f"LoRA key {name!r} does not match layers.<i>.<proj>.lora_<A|B>")
        if require_bf16 and tensor.dtype != torch.bfloat16:
            raise WeightSyncError(f"{name}: expected bf16, got {tensor.dtype}")
        if tensor.dim() != 2:
            raise WeightSyncError(f"{name}: expected a 2-D tensor, got shape {tuple(tensor.shape)}")
        layers.add(int(match["layer"]))
        module = f"layers.{match['layer']}.{match['proj']}"
        modules.setdefault(module, {})[match["mat"]] = tensor
    for module, mats in modules.items():
        if set(mats) != {"A", "B"}:
            raise WeightSyncError(f"{module}: needs both lora_A and lora_B, got {sorted(mats)}")
        a, b = mats["A"], mats["B"]
        if a.shape[0] != b.shape[1]:
            raise WeightSyncError(
                f"{module}: rank mismatch lora_A {tuple(a.shape)} vs lora_B {tuple(b.shape)}"
            )
        ranks.add(int(a.shape[0]))
    return {
        "num_tensors": len(state_dict),
        "num_modules": len(modules),
        "layers": sorted(layers),
        "ranks": sorted(ranks),
        "num_elements": int(sum(t.numel() for t in state_dict.values())),
    }


@dataclass
class LoRAPackage:
    """A verified LoRA export: tensors, canonical hash and where it came from."""

    state_dict: dict[str, torch.Tensor]
    sha256: str
    step: int
    provenance: dict[str, Any] = field(default_factory=dict)


def export_lora_verified(trainer: Any, *, step: int, validate: bool = True) -> LoRAPackage:
    """Call ``trainer.export_lora()`` and verify the trainer's hash against ``lora_sha256``."""

    state_dict, reported = trainer.export_lora()
    canonical = lora_sha256(state_dict)
    if reported != canonical:
        raise WeightSyncError(
            f"trainer reported lora sha256 {reported} but the canonical hash is {canonical}; "
            f"the trainer must implement {LORA_SHA256_SCHEMA}"
        )
    if validate:
        validate_lora_state_dict(state_dict)
    return LoRAPackage(dict(state_dict), canonical, step, {"source": "trainer.export_lora"})


def save_lora_package(package: LoRAPackage, path: Path) -> Path:
    """Persist a package for cross-process transfer (``torch.save``)."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": LORA_PACKAGE_SCHEMA,
        "hash_schema": LORA_SHA256_SCHEMA,
        "sha256": package.sha256,
        "step": int(package.step),
        "state_dict": {k: v.detach().cpu().contiguous() for k, v in package.state_dict.items()},
        "provenance": dict(package.provenance),
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)
    return path


def load_lora_package(path: Path) -> LoRAPackage:
    """Load a package and fail closed if the recomputed hash differs from the stored one."""

    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("schema") != LORA_PACKAGE_SCHEMA:
        raise WeightSyncError(f"{path}: not a {LORA_PACKAGE_SCHEMA} package")
    state_dict = payload["state_dict"]
    stored = str(payload["sha256"])
    actual = lora_sha256(state_dict)
    if actual != stored:
        raise WeightSyncError(f"{path}: stored sha256 {stored} != recomputed {actual}")
    provenance = dict(payload.get("provenance", {}))
    provenance["file"] = str(path)
    return LoRAPackage(dict(state_dict), actual, int(payload["step"]), provenance)


def weight_sync_identity(exported_sha256: str, rollout_sha256: Any) -> bool:
    """``weight_sync_identity`` row check: the rollout holds exactly the exported LoRA."""

    return isinstance(rollout_sha256, str) and rollout_sha256 == exported_sha256


def transfer_lora(
    package: LoRAPackage,
    rollout: Any,
    *,
    via_file: Path | None = None,
) -> dict[str, Any]:
    """Load ``package`` into ``rollout`` (in memory or through a file) and verify identity."""

    if via_file is not None:
        save_lora_package(package, via_file)
        loaded = load_lora_package(via_file)
        transport = "file"
    else:
        loaded = package
        transport = "memory"
    rollout.load_lora(loaded.state_dict, loaded.sha256)
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
    return None, "no lora_sha256 in rollout provenance"
