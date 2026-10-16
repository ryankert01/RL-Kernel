import json, struct, itertools
W = "/home/ubuntu/RL-Kernel/.claude/worktrees/"
me = W + "wf_5e0d8071-7a1-3/reports/muse_glimmer_poc/data/"
files = {
    "meg_strict_tp1": me + "xrt_megatron_strict_tp1.json",
    "meg_strict_tp2": me + "xrt_megatron_strict_tp2.json",
    "meg_native_tp1": me + "xrt_megatron_native_tp1.json",
    "meg_native_tp2": me + "xrt_megatron_native_tp2.json",
    "vllm_strict_tp1": W + "wf_7b064961-36f-1/reports/muse_glimmer_poc/data/xrt_vllm_strict_tp1.json",
    "vllm_strict_tp2": W + "wf_7b064961-36f-1/reports/muse_glimmer_poc/data/xrt_vllm_strict_tp2.json",
    "vllm_native_tp1": W + "wf_7b064961-36f-1/reports/muse_glimmer_poc/data/xrt_vllm_native_tp1.json",
    "ref_triton": W + "wf_7b064961-36f-8/reports/muse_glimmer_poc/data/xrt_reference_triton.json",
    "ref_golden": W + "wf_7b064961-36f-8/reports/muse_glimmer_poc/data/xrt_reference_golden.json",
}
tok = json.load(open(me + "xrt_tokens.json"))
exp = {s["id"]: len(s["token_ids"]) - s["completion_start"] for s in tok["sequences"]}


def f32(u):
    return struct.unpack("<f", struct.pack("<I", u))[0]


def load(p):
    d = json.load(open(p))
    out = {}
    for s in d["sequences"]:
        hx = s.get("logprobs_hex")
        if hx is None:
            print(p, "no hex; keys", list(s))
            return d, None
        out[s["id"]] = [int(h, 16) for h in hx]
        if "logprobs" in s:
            for h, v in zip(hx, s["logprobs"]):
                f = f32(int(h, 16))
                if f != v:
                    print("hex/float disagree", p, s["id"], h, v)
                    break
    return d, out


data = {}
for k, p in files.items():
    d, o = load(p)
    meta = {kk: d.get(kk) for kk in ("schema", "runtime", "mode", "tp", "backend", "commit")}
    print(k, meta)
    if o is not None:
        bad = [i for i in exp if len(o.get(i, [])) != exp[i]]
        print("   lengths ok" if not bad else f"   LENGTH MISMATCH {bad}", "n=", sum(len(v) for v in o.values()))
        data[k] = o
for a, b in itertools.combinations(list(data), 2):
    mm = 0
    mx = 0.0
    n = 0
    for i in exp:
        for x, y in zip(data[a][i], data[b][i]):
            n += 1
            if x != y:
                mm += 1
                mx = max(mx, abs(f32(x) - f32(y)))
    print(f"{a:16s} vs {b:16s}: {mm}/{n} bit mismatches, max_abs {mx:.4g}")
