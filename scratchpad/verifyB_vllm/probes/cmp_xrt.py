import json
import sys

D = sys.argv[1]


def load(f):
    return json.load(open(f))


def seqs(d):
    out = {}
    for i, s in enumerate(d["sequences"]):
        key = s.get("id") or s.get("name") or i
        out[key] = s
    return out


def lp(s):
    for k in ("logprobs_hex", "logprobs"):
        if k in s:
            return k, s[k]
    raise KeyError(list(s.keys()))


def cmp(fa, fb):
    a, b = load(fa), load(fb)
    sa, sb = seqs(a), seqs(b)
    assert set(sa) == set(sb), set(sa) ^ set(sb)
    tot = mis = 0
    tokdiff = 0
    for k in sa:
        ka, la = lp(sa[k])
        kb, lb = lp(sb[k])
        assert len(la) == len(lb)
        tot += len(la)
        mis += sum(1 for x, y in zip(la, lb) if x != y)
        for tk in ("token_ids", "tokens"):
            if tk in sa[k] and sa[k][tk] != sb[k].get(tk):
                tokdiff += 1
    return mis, tot, ka, tokdiff


first = load(f"{D}/xrt_vllm_strict_tp1.json")
print("seq keys", list(first["sequences"][0].keys()))
pairs = [
    ("strict_tp2", "strict_tp1"),
    ("strict_tp4", "strict_tp1"),
    ("strict_lora_tp2", "strict_lora_tp1"),
    ("strict_lora_tp4", "strict_lora_tp1"),
    ("native_tp2", "native_tp1"),
    ("strict_lora_tp1", "strict_tp1"),
    ("native_lora_tp1", "native_tp1"),
]
for p in pairs:
    print(p, cmp(f"{D}/xrt_vllm_{p[0]}.json", f"{D}/xrt_vllm_{p[1]}.json"))
for f in [
    "strict_tp1", "strict_tp2", "strict_tp4", "strict_lora_tp1", "strict_lora_tp2",
    "strict_lora_tp4", "native_tp1", "native_tp2", "native_lora_tp1",
]:
    d = load(f"{D}/xrt_vllm_{f}.json")
    print(f, d["commit"][:7], d["tp"], d["mode"], d.get("lora_sha256"))
