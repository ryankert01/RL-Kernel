"""Strict binding under the V1 model runner: score xrt_tokens.json and compare with the committed
strict (V2) dump bit for bit.  Shows whether the V1 path returns strict O7 numbers or not."""

import json
import os
import struct
import sys


def bits(xs):
    return [struct.pack(">f", float(v)).hex() for v in xs]


def main() -> int:
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"
    os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    from rl_engine.models.muse_glimmer.runtime.strict_rollout_engine import StrictRolloutEngine

    xrt_path, dump_path = sys.argv[1], sys.argv[2]
    xrt = json.load(open(xrt_path))
    dump = {s["id"]: s["logprobs_hex"] for s in json.load(open(dump_path))["sequences"]}
    seqs = [s for s in xrt["sequences"] if s["id"] in dump]
    model = os.path.expanduser("~/models/Muse-Glimmer-30B")
    engine = StrictRolloutEngine(
        model, tensor_parallel_size=1, gpu_ids=[0], strict=True, max_model_len=4096, seed=0
    )
    try:
        got = engine.logprobs([s["token_ids"] for s in seqs], [s["completion_start"] for s in seqs])
        tot = mis = 0
        maxd = 0.0
        for s, g in zip(seqs, got):
            gb = bits(g)
            ref = dump[s["id"]]
            tot += len(ref)
            for a, b in zip(gb, ref):
                if a != b:
                    mis += 1
                    fa = struct.unpack(">f", bytes.fromhex(a))[0]
                    fb = struct.unpack(">f", bytes.fromhex(b))[0]
                    maxd = max(maxd, abs(fa - fb))
        counts = engine.provenance()["workers"][0]["call_counts"]
        print(f"V1-runner strict xrt vs committed V2 strict dump: {mis}/{tot} mismatched, max |d| {maxd}")
        print("prompt_logprobs.strict_logp_calls:", counts.get("prompt_logprobs.strict_logp_calls"))
    finally:
        engine.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
