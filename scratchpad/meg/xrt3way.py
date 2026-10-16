import json
import struct
import sys

W = "/home/ubuntu/RL-Kernel/.claude/worktrees/"
files = {
    "megatron_strict_tp1": "wf_7b064961-36f-2/reports/muse_glimmer_poc/data/xrt_megatron_strict_tp1.json",
    "megatron_strict_tp2": "wf_7b064961-36f-2/reports/muse_glimmer_poc/data/xrt_megatron_strict_tp2.json",
    "vllm_strict_tp1": "wf_7b064961-36f-1/reports/muse_glimmer_poc/data/xrt_vllm_strict_tp1.json",
    "vllm_strict_tp2": "wf_7b064961-36f-1/reports/muse_glimmer_poc/data/xrt_vllm_strict_tp2.json",
    "reference_triton": "wf_7b064961-36f-8/reports/muse_glimmer_poc/data/xrt_reference_triton.json",
    "vllm_native_tp1": "wf_7b064961-36f-1/reports/muse_glimmer_poc/data/xrt_vllm_native_tp1.json",
    "megatron_native_tp1": "wf_7b064961-36f-2/reports/muse_glimmer_poc/data/xrt_megatron_native_tp1.json",
    "megatron_native_tp2": "wf_7b064961-36f-2/reports/muse_glimmer_poc/data/xrt_megatron_native_tp2.json",
}
D = {}
meta = {}
for k, f in files.items():
    d = json.load(open(W + f))
    D[k] = {s["id"]: s["logprobs_hex"] for s in d["sequences"]}
    meta[k] = {x: d.get(x) for x in ("schema", "runtime", "mode", "tp", "backend", "commit")}
    print(k, meta[k])


def val(h):
    try:
        return struct.unpack(">f", bytes.fromhex(h))[0]
    except (ValueError, TypeError):
        return float("nan")


keys = list(files)
matrix = {}
for a in keys:
    row = []
    for b in keys:
        m = sum(x != y for i in D[a] for x, y in zip(D[a][i], D[b][i], strict=True))
        diffs = [abs(val(x) - val(y)) for i in D[a] for x, y in zip(D[a][i], D[b][i], strict=True)]
        mx = max((v for v in diffs if v == v), default=None)
        row.append(m)
        matrix[f"{a}|{b}"] = {"bit_mismatches": m, "max_abs": mx}
    print(f"{a:22s}", row)
n = sum(len(v) for v in D[keys[0]].values())
print("tokens", n)
for pair in ("megatron_strict_tp1|vllm_strict_tp1", "megatron_strict_tp1|reference_triton",
             "vllm_strict_tp1|reference_triton", "megatron_native_tp1|vllm_native_tp1",
             "megatron_strict_tp1|megatron_native_tp1"):
    print(pair, matrix[pair])
if len(sys.argv) > 1:
    json.dump({"files": files, "meta": meta, "tokens": n, "pairs": matrix}, open(sys.argv[1], "w"), indent=1)
