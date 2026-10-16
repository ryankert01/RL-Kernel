"""Adversarial probe: strict binding under vLLM's V1 model runner (VLLM_USE_V2_MODEL_RUNNER=0).

The binding only patches the V2 Sampler / prompt-logprob path.  If the engine starts and returns
logprobs while the strict logp overwrite never ran, strict mode is silently returning native
numbers (log_softmax of the strict logits) instead of failing closed.
"""

import os
import struct
import sys


def bits(xs):
    return [struct.pack(">f", float(v)).hex() for v in xs]


def main() -> int:
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"
    from rl_engine.models.muse_glimmer.runtime.strict_rollout_engine import StrictRolloutEngine

    model = os.path.expanduser("~/models/Muse-Glimmer-30B")
    try:
        engine = StrictRolloutEngine(
            model, tensor_parallel_size=1, gpu_ids=[0], strict=True, max_model_len=1024, seed=0
        )
    except Exception as exc:  # noqa: BLE001
        print("PROBE_RESULT: engine init raised (fail closed):", type(exc).__name__, str(exc)[:600])
        return 0
    try:
        prompts = [[200000, 791, 6864, 315, 9822, 374, 279, 6864]]
        try:
            samples = engine.generate(prompts, n=1, max_tokens=16, temperature=0.0)
        except Exception as exc:  # noqa: BLE001
            print("PROBE_RESULT: generate raised (fail closed):", type(exc).__name__, str(exc)[:600])
            return 0
        seq = prompts[0] + samples[0][0].token_ids
        decoded = samples[0][0].logprobs
        try:
            rescored = engine.logprobs([seq], [len(prompts[0])])[0]
        except Exception as exc:  # noqa: BLE001
            rescored = None
            print("prefill rescore raised:", type(exc).__name__, str(exc)[:300])
        prov = engine.provenance()
        counts = prov["workers"][0]["call_counts"]
        print("model_runner field:", prov.get("model_runner"))
        print("decoded logprobs:", list(decoded)[:8])
        if rescored is not None:
            print("rescored logprobs:", list(rescored)[:8])
            mis = sum(1 for a, b in zip(bits(decoded), bits(rescored)) if a != b)
            print(f"decode vs prefill mismatches: {mis}/{len(decoded)}")
        keys = sorted(k for k in counts if "sampler" in k or "prompt_logprobs" in k or "compute_logits" in k)
        print("call counts:", {k: counts[k] for k in keys})
        if counts.get("sampler.strict_logp_calls", 0) == 0:
            print("PROBE_RESULT: SILENT NON-STRICT: generate returned logprobs but the strict logp "
                  "overwrite never ran (V1 runner)")
        else:
            print("PROBE_RESULT: strict logp overwrite ran")
    finally:
        engine.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
