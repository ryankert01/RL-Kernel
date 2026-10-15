# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""vLLM general-plugin shim for the Muse Glimmer strict binding.

The repo's single ``vllm.general_plugins`` entry point
(``rl_engine.integrations.engines.rollout.vllm.runtime:register_vllm_plugin``) dispatches here when
``RL_KERNEL_MUSEGLIMMER_STRICT=1``.  vLLM loads general plugins in the API process, the engine core
and every worker, so :func:`register` must be idempotent (it is: the installer caches its state).
"""

from __future__ import annotations

import os

STRICT_ENV = "RL_KERNEL_MUSEGLIMMER_STRICT"
_TRUE = {"1", "true", "True", "yes", "on"}


def strict_requested() -> bool:
    return os.getenv(STRICT_ENV, "").strip() in _TRUE


def register() -> bool:
    """Install the strict binding when requested; returns whether it was installed."""

    if not strict_requested():
        return False
    from rl_engine.models.muse_glimmer.runtime.vllm_strict import install_museglimmer_vllm_strict

    install_museglimmer_vllm_strict()
    return True


__all__ = ["STRICT_ENV", "register", "strict_requested"]
