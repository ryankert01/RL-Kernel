# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Native-mode LoRA for the Muse Glimmer rollout: write a PEFT folder vLLM 0.31 can load.

The E2E loop syncs the trainer's LoRA (keys ``layers.{i}.{proj}.lora_{A,B}``, schema of
``e2e.sync``) into the rollout after every step in all four train/rollout modes.  In native mode
the rollout is plain vLLM, so the adapter goes through vLLM's own (punica, merged-at-runtime,
bf16) LoRA path from a PEFT folder:

* ``adapter_model.safetensors``: ``base_model.model.model.language_model.layers.{i}.self_attn.
  {q,k,v,gate,o}_proj.lora_{A,B}.weight`` and ``...mlp.{gate,up,down}_proj.lora_{A,B}.weight``
  (HF module names; vLLM maps ``self_attn.gate_proj`` -> ``output_gate_proj`` and the
  ``model.language_model.`` prefix through ``hf_to_vllm_mapper.get_rename_mapper()``);
* ``adapter_config.json``: ``r`` = rank, ``lora_alpha`` = rank (scaling 1, so the update is
  exactly ``B A`` like the strict rule), no dropout, no bias.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import torch

from rl_engine.models.muse_glimmer.e2e.sync import LORA_KEY_RE

TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "gate_proj", "o_proj", "up_proj", "down_proj")


def peft_key(key: str) -> str:
    """``layers.{i}.{proj}.lora_{A,B}`` -> the PEFT tensor name of the HF module."""

    match = LORA_KEY_RE.match(key)
    if match is None:
        raise KeyError(f"unsupported LoRA key {key!r}")
    proj = match["proj"]
    module = f"mlp.{proj[4:]}" if proj.startswith("mlp.") else f"self_attn.{proj}"
    return (
        f"base_model.model.model.language_model.layers.{match['layer']}.{module}"
        f".lora_{match['mat']}.weight"
    )


def write_peft_adapter(state_dict: Mapping[str, torch.Tensor], folder: Path) -> Path:
    """Write ``state_dict`` as a PEFT LoRA folder; returns the folder."""

    from safetensors.torch import save_file

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    tensors = {peft_key(k): v.detach().to("cpu", torch.bfloat16).contiguous() for k, v in
               state_dict.items()}
    ranks = {int(v.shape[0]) for k, v in state_dict.items() if k.endswith("lora_A")}
    if len(ranks) != 1:
        raise ValueError(f"native LoRA needs one rank, got {sorted(ranks)}")
    rank = ranks.pop()
    save_file(tensors, str(folder / "adapter_model.safetensors"))
    config = {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "r": rank,
        "lora_alpha": rank,
        "lora_dropout": 0.0,
        "bias": "none",
        "fan_in_fan_out": False,
        "use_rslora": False,
        "use_dora": False,
        "target_modules": list(TARGET_MODULES),
        "modules_to_save": None,
    }
    (folder / "adapter_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    return folder


__all__ = ["TARGET_MODULES", "peft_key", "write_peft_adapter"]
