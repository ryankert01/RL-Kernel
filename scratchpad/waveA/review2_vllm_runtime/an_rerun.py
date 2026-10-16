import json
import sys
from collections import Counter

r = json.load(open(sys.argv[1]))
print("status", r.get("status"), "commit", r.get("commit"), "env", r.get("env"))
g = r["generate"][0]
dv = g["decode_vs_prefill"]
print("tokens", dv["tokens"], "mismatched", dv["mismatched"], "max_abs", dv["max_abs_diff"])
ids = g["ids"]
lp = g["logprobs"]
flat = [x for row in lp for x in row]
print("seqs", len(lp), "ids", Counter(ids))
print("exact0", sum(x == 0.0 for x in flat), "|lp|<1e-4", sum(abs(x) < 1e-4 for x in flat), "lp<-0.1", sum(x < -0.1 for x in flat), "positive", sum(x > 0 for x in flat), "of", len(flat))
lw = [i for i, s in enumerate(ids) if s == "gsm8k_long_window"]
lwflat = [x for i in lw for x in lp[i]]
print("long_window seqs", len(lw), "prompt_len", g["prompt_lens"][lw[0]], "tokens", len(lwflat), "nonzero", sum(x != 0 for x in lwflat), "lp<-0.1", sum(x < -0.1 for x in lwflat))
print("long_window mismatches", sum(dv["per_sequence"][i]["mismatched"] for i in lw))
distinct = len({tuple(t) for t in g["token_ids"]})
print("distinct completions", distinct)
lens = Counter(len(t) for t in g["token_ids"])
print("completion lens", lens, "finish", Counter(g["finish_reasons"]))
meta = r.get("meta") or {}
for w in meta.get("workers", []):
    cc = w.get("call_counts", {})
    print("worker", w.get("tp_rank"), "all_ops_strict", w.get("all_ops_strict"), {k: cc.get(k) for k in ("attention.local", "attention.global", "kv_cache_write", "sampler.strict_logp_rows", "prompt_logprobs.strict_logp_rows")})
    print("engine kwargs", meta.get("engine_kwargs"))
