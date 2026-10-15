# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

"""Score ``xrt_tokens.json`` with the ``full_model_chain`` strict reference (no vLLM).

The reference (``rl_engine.models.muse_glimmer.reference``, Triton backend = the same sibling
strict ops) is the in-repo ground truth the vLLM strict binding must reproduce bit for bit.  This
writes one ``muse_glimmer_xrt_logprobs_v1`` file per requested mode (``train``: one training
forward over prompt+completion; ``prefill``: one-shot paged prefill)::

    python -m rl_engine.models.muse_glimmer.runtime.vllm_xrt_reference \
        reports/muse_glimmer_poc/data/xrt_tokens.json <out>.json train,prefill

-> ``<out>_train.json``, ``<out>_prefill.json``.  Needs one GPU with ~55 GiB free.
"""

from __future__ import annotations

import json
import struct
import sys
import time
from pathlib import Path

import torch


def main(argv: list[str] | None = None) -> int:
    from rl_engine.models.muse_glimmer.reference.strict_model import MuseGlimmerStrictModel
    from rl_engine.models.muse_glimmer.reference.weights import MuseGlimmerWeights
    from rl_engine.models.muse_glimmer.runtime.run_vllm_strict import _git_head, _git_status
    from rl_engine.validation.operators.accelerator import disable_tf32

    args = sys.argv[1:] if argv is None else argv
    xrt_path, out_path = args[0], args[1]
    modes = args[2].split(",") if len(args) > 2 else ["train"]
    checkpoint = args[3] if len(args) > 3 else str(Path("~/models/Muse-Glimmer-30B").expanduser())
    device = torch.device("cuda:0")
    disable_tf32("cuda")
    t0 = time.perf_counter()
    weights = MuseGlimmerWeights.load_checkpoint(checkpoint, device=device, layout="nk")
    model = MuseGlimmerStrictModel(weights, "triton")
    print(f"loaded in {time.perf_counter() - t0:.1f}s", flush=True)
    data = json.loads(Path(xrt_path).read_text(encoding="utf-8"))
    for mode in modes:
        seqs = []
        t1 = time.perf_counter()
        for seq in data["sequences"]:
            ids = torch.tensor(seq["token_ids"])
            start = int(seq["completion_start"])
            logp = model.logprobs(ids[:start], ids[start:], mode=mode)
            seqs.append(
                {
                    "id": seq["id"],
                    "logprobs_hex": [
                        struct.pack(">f", float(v)).hex() for v in logp.float().cpu().tolist()
                    ],
                }
            )
        doc = {
            "schema": "muse_glimmer_xrt_logprobs_v1",
            "runtime": "reference_chain",
            "mode": "strict",
            "tp": 1,
            "backend": f"rl_engine.models.muse_glimmer.reference triton backend, mode={mode}",
            "score_ms": (time.perf_counter() - t1) * 1e3,
            "commit": _git_head(),
            **_git_status(),
            "command": "python -m rl_engine.models.muse_glimmer.runtime.vllm_xrt_reference "
            + " ".join(args),
            "checkpoint": checkpoint,
            "sequences": seqs,
        }
        target = Path(out_path.replace(".json", f"_{mode}.json"))
        target.write_text(json.dumps(doc, indent=1), encoding="utf-8")
        print(f"{mode}: {sum(len(s['logprobs_hex']) for s in seqs)} tokens -> {target}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
