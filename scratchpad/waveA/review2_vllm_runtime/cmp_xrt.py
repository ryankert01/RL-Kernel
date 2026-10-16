"""Recompute the xrt comparisons from the raw dumps (read-only)."""
import json
import struct
import sys
from pathlib import Path

D = Path(sys.argv[1])  # reports/muse_glimmer_poc/data
S = Path(sys.argv[2])  # scratchpad root


def load(p):
    d = json.load(open(p))
    return {s["id"]: s["logprobs_hex"] for s in d["sequences"]}, d


def f(h):
    return struct.unpack(">f", bytes.fromhex(h))[0]


def cmp(a, b):
    tot = mism = 0
    mx = 0.0
    sm = 0.0
    pos = 0
    for k in a:
        assert len(a[k]) == len(b[k]), (k, len(a[k]), len(b[k]))
        for x, y in zip(a[k], b[k]):
            tot += 1
            if x != y:
                mism += 1
            dd = abs(f(x) - f(y))
            mx = max(mx, dd)
            sm += dd
    assert set(a) == set(b)
    return tot, mism, mx, sm / tot


files = {
    "s1": D / "xrt_vllm_strict_tp1.json",
    "s2": D / "xrt_vllm_strict_tp2.json",
    "n1": D / "xrt_vllm_native_tp1.json",
    "n2": D / "xrt_vllm_native_tp2.json",
    "sl1": D / "xrt_vllm_strict_lora_tp1.json",
    "sl2": D / "xrt_vllm_strict_lora_tp2.json",
    "nl1": D / "xrt_vllm_native_lora_tp1.json",
    "rtrain": S / "vllm_runs/xrt_reference_train.json",
    "rprefill": S / "vllm_runs/xrt_reference_prefill.json",
    "n1run1": S / "vllm_runs/xrt_native_tp1_run1.txt",
}
L = {}
for k, p in files.items():
    try:
        L[k], meta = load(p)
        print(k, p.name, {kk: meta.get(kk) for kk in ("mode", "tp", "commit", "lora_sha256", "runtime")})
    except Exception as e:  # noqa: BLE001
        print("cannot load", k, p, e)
pairs = [
    ("s2", "s1"), ("n2", "n1"), ("s1", "n1"), ("s2", "n2"), ("sl2", "sl1"), ("sl1", "nl1"),
    ("sl1", "s1"), ("s1", "rtrain"), ("s2", "rtrain"), ("s1", "rprefill"), ("rtrain", "rprefill"),
    ("n1", "n1run1"), ("nl1", "n1"),
]
for a, b in pairs:
    if a in L and b in L:
        print(a, "vs", b, cmp(L[a], L[b]))
# positive logps
for k in L:
    pos = [(sid, i, f(h)) for sid, v in L[k].items() for i, h in enumerate(v) if f(h) > 0]
    print("positive", k, len(pos), pos[:3])
tok = json.load(open(D / "xrt_tokens.json"))
for s in tok["sequences"]:
    print(s["id"], len(s["token_ids"]), s.get("completion_start"), {k: v for k, v in s.items() if k not in ("token_ids",)})
