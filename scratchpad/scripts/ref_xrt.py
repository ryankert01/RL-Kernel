"""Score xrt_tokens.json with the full_model_chain strict reference (triton backend) -> xrt format."""
import json
import struct
import sys
import time

import torch

from rl_engine.models.muse_glimmer.reference.run_chain import disable_tf32
from rl_engine.models.muse_glimmer.reference.strict_model import MuseGlimmerStrictModel
from rl_engine.models.muse_glimmer.reference.weights import MuseGlimmerWeights

xrt_path, out_path = sys.argv[1], sys.argv[2]
modes = sys.argv[3].split(",")
device = torch.device("cuda:0")
disable_tf32("cuda")
t0 = time.perf_counter()
weights = MuseGlimmerWeights.load_checkpoint(
    "/home/ubuntu/models/Muse-Glimmer-30B", device=device, layout="nk"
)
model = MuseGlimmerStrictModel(weights, "triton")
print("loaded", time.perf_counter() - t0, flush=True)
data = json.load(open(xrt_path))
for mode in modes:
    seqs = []
    for s in data["sequences"]:
        ids = torch.tensor(s["token_ids"])
        c = s["completion_start"]
        t1 = time.perf_counter()
        lp = model.logprobs(ids[:c], ids[c:], mode=mode)
        torch.cuda.synchronize()
        seqs.append(
            {
                "id": s["id"],
                "logprobs_hex": [struct.pack(">f", float(v)).hex() for v in lp.float().cpu().tolist()],
            }
        )
        print(mode, s["id"], len(seqs[-1]["logprobs_hex"]), round(time.perf_counter() - t1, 2), flush=True)
    json.dump(
        {
            "schema": "muse_glimmer_xrt_logprobs_v1",
            "runtime": "reference_chain",
            "mode": "strict",
            "tp": 1,
            "backend": f"rl_engine.models.muse_glimmer.reference triton backend, logprobs(mode={mode})",
            "sequences": seqs,
        },
        open(out_path.replace(".json", f"_{mode}.json"), "w"),
        indent=1,
    )
print("done", flush=True)
