"""Adversarial probe: a strict vLLM engine started with prefix caching ON must fail at init."""

import os
import sys
import tempfile


def main() -> int:
    os.environ["RL_KERNEL_MUSEGLIMMER_STRICT"] = "1"
    os.environ["RL_KERNEL_READBACK_DIR"] = tempfile.mkdtemp(prefix="verifyB-probe-")
    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    from vllm import LLM

    model = os.path.expanduser("~/models/Muse-Glimmer-30B")
    try:
        LLM(
            model=model,
            tensor_parallel_size=1,
            dtype="bfloat16",
            max_model_len=4096,
            enforce_eager=True,
            gpu_memory_utilization=0.9,
            enable_prefix_caching=True,
            limit_mm_per_prompt={"image": 0, "video": 0},
            attention_config={"backend": "FLASH_ATTN"},
            max_num_seqs=8,
        )
    except Exception as exc:  # noqa: BLE001
        print("PROBE_RESULT: engine init raised", type(exc).__name__, str(exc)[:500])
        return 0
    print("PROBE_RESULT: ENGINE STARTED WITH PREFIX CACHING (not fail-closed)")
    return 3


if __name__ == "__main__":
    sys.exit(main())
