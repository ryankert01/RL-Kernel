"""Print the essentials of a run_vllm_strict output JSON (optionally compare to another)."""

import json
import sys

r = json.load(open(sys.argv[1]))
keys = (
    "mode",
    "strict",
    "tp",
    "generated_tokens",
    "tokens_per_s",
    "generate_seconds",
    "prefill_seconds",
    "score_seconds",
    "decode_vs_prefill_equal",
    "decode_vs_prefill_max_abs_diff",
    "lora_zero_equal_to_prefill",
    "bits_equal_to_scored_file",
    "max_abs_diff_to_scored_file",
)
print({k: r[k] for k in keys if k in r})
if "finish_reasons" in r:
    print("finish:", r["finish_reasons"])
    print("lens:", [len(t) for t in r["token_ids"]])
    print("texts:", [t[:70].replace("\n", "\\n") for t in r["texts"]])
eng = r["meta"]["engine"]
print("engine timing:", json.dumps(eng["timing"]["history"]))
print("load_seconds:", eng["timing"]["load_seconds"])
for w in r["meta"].get("workers", []):
    if "error" in w:
        print("worker error:", w["error"])
        continue
    print(
        f"worker rank {w['tp_rank']}/{w['tp']}: hooks={len(w['hooks'])} all_strict={w['all_ops_strict']} "
        f"fallbacks={w['fallbacks']} demotions={w['demotions']}"
    )
    print("  call_counts:", w["call_counts"])
    print("  checks:", w["checks"])
    print("  notes:", w["notes"])
    print("  backend_ids:", w["backend_ids"])
    print("  kv_ownership:", w["kv_ownership"])
    print("  lora:", w["lora_sha256"], w["lora_slots"])
print("lp[0][:4]:", r["logprobs"][0][:4], r["logprobs_hex"][0][:4])
if len(sys.argv) > 2:
    o = json.load(open(sys.argv[2]))
    same_tokens = o["token_ids"] == r["token_ids"]
    print("tokens equal to", sys.argv[2], ":", same_tokens)
    diffs = []
    for a, b in zip(r["logprobs"], o["logprobs"]):
        for x, y in zip(a, b):
            diffs.append(abs(x - y))
    print("max abs logprob diff vs other:", max(diffs) if diffs else None)
    print("bits equal vs other:", r["logprobs_hex"] == o["logprobs_hex"])
